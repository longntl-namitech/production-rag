"""Retrieval eval for methods that call an LLM, with a token budget.

    docker compose up -d qdrant
    uv run --extra eval --extra local-models python -m production_rag.core.eval.run_llm \
        --dataset scifact --methods hybrid_weighted_rrf multi_query --estimate-only

`core/eval/run.py` makes no LLM calls and stays that way; this runner reuses its datasets,
metrics and row format. Each row of the table is an `Arm`. Run `--estimate-only` first: it
prints an upper bound of the tokens and calls no LLM.
"""

import argparse
import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from production_rag.core.cache import DiskCache
from production_rag.core.chunking import chunk_documents
from production_rag.core.config import Settings
from production_rag.core.deps import Deps, build_deps, warm_up
from production_rag.core.eval.answer import estimate_tokens
from production_rag.core.eval.datasets import Dataset, load_dataset, subsample
from production_rag.core.eval.run import (
    K,
    chunks_for,
    collection_for,
    result_row,
    summarize,
)
from production_rag.core.indexing import index_chunks
from production_rag.core.llm import BudgetExceeded, MissingAPIKey, OpenAIChat, TokenBudget, Usage
from production_rag.core.models import SearchRequest, SearchSpec, SpecResult
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.search import run_search
from production_rag.methods import get_variant
from production_rag.methods.contextual_headers import pipeline as contextual_headers
from production_rag.methods.contextual_headers.context import MAX_TOKENS as CONTEXT_MAX_TOKENS
from production_rag.methods.contextual_headers.context import context_messages
from production_rag.methods.hyde import pipeline as hyde
from production_rag.methods.hyde.generate import MAX_TOKENS as HYDE_MAX_TOKENS
from production_rag.methods.hyde.generate import hyde_messages
from production_rag.methods.hyde.generate import queries_for as hyde_queries
from production_rag.methods.llm_rerank import pipeline as llm_rerank
from production_rag.methods.llm_rerank.rerank import UNUSABLE
from production_rag.methods.llm_rerank.rerank import estimate_tokens_per_query as rerank_tokens
from production_rag.methods.multi_query import pipeline as multi_query
from production_rag.methods.multi_query.rewrite import MAX_TOKENS as REWRITE_MAX_TOKENS
from production_rag.methods.multi_query.rewrite import rewrite, rewrite_messages
from production_rag.methods.query_decomposition.generate import MAX_TOKENS as DECOMPOSE_MAX_TOKENS
from production_rag.methods.query_decomposition.generate import decompose_messages
from production_rag.methods.query_decomposition.generate import queries_for as decompose_queries
from production_rag.methods.step_back.generate import MAX_TOKENS as STEP_BACK_MAX_TOKENS
from production_rag.methods.step_back.generate import queries_for as step_back_queries
from production_rag.methods.step_back.generate import step_back_messages

RESULTS = Path("results")
DEADLINE_MS = 30000

# question, collection, deps, llm, k
SearchFn = Callable[[str, str, Deps, Any, int], Awaitable[tuple[SpecResult, Usage]]]


# deps, llm, dataset, collection: build the collection of an arm that has its own index
IndexFn = Callable[[Deps, Any, Dataset, str], Awaitable[Usage]]


@dataclass(frozen=True)
class Arm:
    """One row of the table: a way to search the collection of `variant`."""

    name: str
    variant: str
    search: SearchFn
    estimate: Callable[[str], int]  # upper bound of LLM tokens per query, 0 without an LLM
    suffix: str = ""  # added to the collection name when the arm has its own index
    index: IndexFn | None = None
    estimate_index: Callable[[Dataset], tuple[int, int]] | None = None  # LLM calls, tokens


def prompt_tokens(messages: list[dict[str, str]], max_tokens: int) -> int:
    return sum(estimate_tokens(m["content"]) for m in messages) + max_tokens


def _plain(variant: str) -> SearchFn:
    async def search(
        question: str, collection: str, deps: Deps, llm: Any, k: int
    ) -> tuple[SpecResult, Usage]:
        req = SearchRequest(
            query=question[:2000], collection=collection, specs=[SearchSpec(name="s", k=k)]
        )
        response = await run_search(req, get_variant(variant), deps, Deadline(DEADLINE_MS))
        return response.results["s"], Usage()

    return search


async def _multi_query(
    question: str, collection: str, deps: Deps, llm: Any, k: int
) -> tuple[SpecResult, Usage]:
    rewrites, usage = await rewrite(llm, question)
    result = await multi_query.search(
        [question[:2000], *rewrites],
        collection,
        deps,
        base="hybrid_weighted_rrf",
        k=k,
        deadline_ms=DEADLINE_MS,
    )
    return result, usage


def _hyde(include_question: bool) -> SearchFn:
    async def search(
        question: str, collection: str, deps: Deps, llm: Any, k: int
    ) -> tuple[SpecResult, Usage]:
        queries, usage = await hyde_queries(llm, question[:2000], include_question=include_question)
        result = await hyde.search(queries, collection, deps, k=k, deadline_ms=DEADLINE_MS)
        return result, usage

    return search


def _hyde_estimate(question: str) -> int:
    return prompt_tokens(hyde_messages(question), HYDE_MAX_TOKENS)


async def _step_back(
    question: str, collection: str, deps: Deps, llm: Any, k: int
) -> tuple[SpecResult, Usage]:
    queries, usage = await step_back_queries(llm, question[:2000])
    result = await multi_query.search(
        queries, collection, deps, base="hybrid_weighted_rrf", k=k, deadline_ms=DEADLINE_MS
    )
    return result, usage


async def _query_decomposition(
    question: str, collection: str, deps: Deps, llm: Any, k: int
) -> tuple[SpecResult, Usage]:
    queries, usage = await decompose_queries(llm, question[:2000])
    result = await multi_query.search(
        queries, collection, deps, base="hybrid_weighted_rrf", k=k, deadline_ms=DEADLINE_MS
    )
    return result, usage


async def _llm_rerank(
    question: str, collection: str, deps: Deps, llm: Any, k: int
) -> tuple[SpecResult, Usage]:
    return await llm_rerank.search(question, collection, deps, llm, k=k, deadline_ms=DEADLINE_MS)


async def _index_headers(deps: Deps, llm: Any, ds: Dataset, collection: str) -> Usage:
    done = await contextual_headers.index(deps, llm, DiskCache(), collection, ds.corpus)
    return done.usage


def _estimate_headers(ds: Dataset) -> tuple[int, int]:
    chunks = chunk_documents(ds.corpus, 800, strategy="recursive")
    tokens = sum(
        prompt_tokens(context_messages(ds.corpus[c.doc_id][1], c.text), CONTEXT_MAX_TOKENS)
        for c in chunks
    )
    return len(chunks), tokens


def _baseline(variant: str) -> Arm:
    return Arm(variant, variant, _plain(variant), lambda question: 0)


ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in [
        _baseline("dense"),
        _baseline("hybrid_weighted_rrf"),
        _baseline("chunk_recursive_800"),
        _baseline("naive_rag"),
        Arm("llm_rerank", "naive_rag", _llm_rerank, lambda question: rerank_tokens()),
        Arm(
            "multi_query",
            "hybrid_weighted_rrf",
            _multi_query,
            lambda question: prompt_tokens(rewrite_messages(question), REWRITE_MAX_TOKENS),
        ),
        Arm("hyde_passage", "hybrid_weighted_rrf", _hyde(False), _hyde_estimate),
        Arm("hyde_passage_query", "hybrid_weighted_rrf", _hyde(True), _hyde_estimate),
        Arm(
            "step_back",
            "hybrid_weighted_rrf",
            _step_back,
            lambda question: prompt_tokens(step_back_messages(question), STEP_BACK_MAX_TOKENS),
        ),
        Arm(
            "query_decomposition",
            "hybrid_weighted_rrf",
            _query_decomposition,
            lambda question: prompt_tokens(decompose_messages(question), DECOMPOSE_MAX_TOKENS),
        ),
        Arm(
            "contextual_headers",
            "chunk_recursive_800",
            _plain("chunk_recursive_800"),
            lambda question: 0,
            suffix="-ctx",
            index=_index_headers,
            estimate_index=_estimate_headers,
        ),
    ]
}


def collection_name(
    dataset: str, variant: str, sample_docs: int | None, sample_queries: int | None
) -> str:
    """The sample size is in the name: a different sample holds different documents."""
    base = collection_for(dataset, get_variant(variant))
    return f"{base}-s{sample_docs}q{sample_queries}" if sample_docs else base


def sample_dataset(ds: Dataset, queries: int, docs: int, seed: int = 13) -> Dataset:
    """A fixed sample: some queries, all their gold documents, other documents up to `docs`."""
    qids = subsample(list(ds.queries), queries, seed)
    gold = {d for qid in qids for d in ds.qrels[qid] if d in ds.corpus}
    others = [d for d in ds.corpus if d not in gold]
    keep = gold | set(subsample(others, max(0, docs - len(gold)), seed))
    return Dataset(
        ds.name,
        {d: ds.corpus[d] for d in ds.corpus if d in keep},
        {qid: ds.queries[qid] for qid in qids},
        {qid: ds.qrels[qid] for qid in qids},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scifact", "tvpl"], required=True)
    parser.add_argument("--methods", nargs="+", required=True, help=f"rows: {', '.join(ARMS)}")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--sample-queries", type=int)
    parser.add_argument("--sample-docs", type=int)
    parser.add_argument("--max-tokens", type=int, help="token cap for the run")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--name")
    args = parser.parse_args(argv)
    unknown = [m for m in args.methods if m not in ARMS]
    if unknown:
        parser.error(f"unknown method {unknown[0]!r}, known: {', '.join(ARMS)}")
    args.methods = list(dict.fromkeys(args.methods))
    if (args.sample_queries is None) != (args.sample_docs is None):
        parser.error("--sample-queries and --sample-docs go together")
    if args.max_tokens is None and not args.estimate_only:
        parser.error("--max-tokens is required for a run that calls the LLM")
    return args


def preflight(args: argparse.Namespace) -> OpenAIChat | None:
    """The LLM client, or None for `--estimate-only`. Runs before any download or indexing."""
    if args.estimate_only:
        return None
    try:
        return OpenAIChat.from_env(budget=TokenBudget(args.max_tokens))
    except MissingAPIKey:
        raise SystemExit("set OPENAI_API_KEY to run this eval, or pass --estimate-only") from None


def estimate(arms: list[Arm], ds: Dataset) -> tuple[int, int]:
    """LLM calls and upper-bound tokens for the run."""
    calls = tokens = 0
    for arm in arms:
        if arm.estimate_index is not None:
            index_calls, index_tokens = arm.estimate_index(ds)
            calls += index_calls
            tokens += index_tokens
        for question in ds.queries.values():
            used = arm.estimate(question)
            tokens += used
            calls += used > 0
    return calls, tokens


def fail_if_all_fell_back(name: str, fallbacks: int, total: int) -> None:
    """Stop when an arm that needs an LLM never got one: its row would be the baseline."""
    if total and fallbacks == total:
        raise SystemExit(
            f"every LLM call failed for {name} (wrong key, model or an outage): "
            "its rows would be the plain search; no results written"
        )


async def run_arm(
    arm: Arm, ds: Dataset, collection: str, deps: Deps, llm: Any, concurrency: int
) -> tuple[list[dict[str, Any]], Usage, int]:
    """The rows, the LLM tokens, and how many queries of an LLM arm used no token at all.

    Those queries fell back to the plain search because the LLM call failed.
    """
    semaphore = asyncio.Semaphore(concurrency)
    usage = Usage()
    fallbacks = 0

    async def one(qid: str, question: str) -> dict[str, Any]:
        nonlocal usage, fallbacks
        async with semaphore:
            t0 = time.perf_counter()
            try:
                result, used = await arm.search(question, collection, deps, llm, K)
            except BudgetExceeded:
                raise
            except Exception as exc:  # one failing query is a failed row, not a failed run
                result = SpecResult(status="failed", elapsed_ms=0, error=type(exc).__name__)
                used = Usage()
            ms = (time.perf_counter() - t0) * 1000
        usage += used
        # a failed call used no token; a reply the method could not use still cost tokens
        fallbacks += arm.estimate(question) > 0 and (used.total == 0 or result.error == UNUSABLE)
        return result_row(arm.name, qid, result, ms, ds.qrels[qid])

    rows = await asyncio.gather(*(one(qid, q) for qid, q in ds.queries.items()))
    return list(rows), usage, fallbacks


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    llm = preflight(args)
    ds = load_dataset(args.dataset, args.query_limit)
    if args.sample_docs is not None:
        ds = sample_dataset(ds, args.sample_queries, args.sample_docs)
    arms = [ARMS[m] for m in args.methods]
    print(f"{len(ds.corpus)} documents, {len(ds.queries)} queries")

    if args.estimate_only:
        calls, tokens = estimate(arms, ds)
        print(
            f"estimate: queries={len(ds.queries)} arms={len(arms)} "
            f"llm_calls={calls} upper_bound_tokens={tokens}"
        )
        return 0

    settings = Settings.from_env()
    deps = build_deps(settings)
    collections = {
        arm.name: collection_name(ds.name, arm.variant, args.sample_docs, args.sample_queries)
        + arm.suffix
        for arm in arms
    }
    await warm_up(deps)

    rows: list[dict[str, Any]] = []
    usage = Usage()
    llm_fallbacks: dict[str, int] = {}
    done: set[str] = set()
    try:
        for arm in arms:
            coll = collections[arm.name]
            if coll in done:
                continue
            done.add(coll)
            started = time.perf_counter()
            if arm.index is not None:
                used = await arm.index(deps, llm, ds, coll)
                usage += used
                print(f"{coll}: indexed with context, {used.total} tokens")
            else:
                chunks = chunks_for(ds.corpus, get_variant(arm.variant))
                print(f"{coll}: {await index_chunks(deps, coll, chunks)} indexed")
            print(f"  ({time.perf_counter() - started:.0f}s)")
        for arm in arms:
            t0 = time.perf_counter()
            got, used, fallbacks = await run_arm(
                arm, ds, collections[arm.name], deps, llm, args.concurrency
            )
            fail_if_all_fell_back(arm.name, fallbacks, len(ds.queries))
            if fallbacks:
                llm_fallbacks[arm.name] = fallbacks
            rows += got
            usage += used
            print(f"{arm.name:24} done in {time.perf_counter() - t0:.0f}s", flush=True)
    except BudgetExceeded as exc:
        raise SystemExit(f"{exc}: no results written") from None

    summary = summarize(rows, [arm.name for arm in arms])
    meta: dict[str, Any] = {
        "kind": "retrieval_llm",
        "dataset": ds.name,
        "date": date.today().isoformat(),
        "queries": len(ds.queries),
        "documents": len(ds.corpus),
        "dense_model": settings.dense_model,
        "cross_encoder": settings.cross_encoder_model,
        "sparse_model": "Qdrant/bm25",
        "k": K,
        "model": llm.model,
        "tokens": {"total": llm.budget.used if llm.budget else usage.total, "queries": usage.total},
        "collections": collections,
    }
    if llm_fallbacks:
        meta["llm_fallbacks"] = llm_fallbacks  # queries that searched the plain question
    if args.sample_docs is not None:
        meta["sample"] = {"queries": args.sample_queries, "documents": args.sample_docs}
    for row in rows:
        row.pop("rel")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.name or ds.name + '_llm'}.json"
    out.write_text(json.dumps({"meta": meta, "summary": summary, "per_query": rows}, indent=1))
    from production_rag.core.eval.report import failures_note, table

    print(table(summary))
    print(failures_note(summary))
    print(f"tokens: {meta['tokens']['total']}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
