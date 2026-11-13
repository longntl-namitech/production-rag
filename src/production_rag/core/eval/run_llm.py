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
from production_rag.methods.hyde import pipeline as hyde
from production_rag.methods.hyde.generate import MAX_TOKENS as HYDE_MAX_TOKENS
from production_rag.methods.hyde.generate import hyde_messages
from production_rag.methods.hyde.generate import queries_for as hyde_queries
from production_rag.methods.multi_query import pipeline as multi_query
from production_rag.methods.multi_query.rewrite import MAX_TOKENS as REWRITE_MAX_TOKENS
from production_rag.methods.multi_query.rewrite import rewrite, rewrite_messages

RESULTS = Path("results")
DEADLINE_MS = 30000

# question, collection, deps, llm, k
SearchFn = Callable[[str, str, Deps, Any, int], Awaitable[tuple[SpecResult, Usage]]]


@dataclass(frozen=True)
class Arm:
    """One row of the table: a way to search the collection of `variant`."""

    name: str
    variant: str
    search: SearchFn
    estimate: Callable[[str], int]  # upper bound of LLM tokens per query, 0 without an LLM


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


def _baseline(variant: str) -> Arm:
    return Arm(variant, variant, _plain(variant), lambda question: 0)


ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in [
        _baseline("dense"),
        _baseline("hybrid_weighted_rrf"),
        _baseline("chunk_recursive_800"),
        Arm(
            "multi_query",
            "hybrid_weighted_rrf",
            _multi_query,
            lambda question: prompt_tokens(rewrite_messages(question), REWRITE_MAX_TOKENS),
        ),
        Arm("hyde_passage", "hybrid_weighted_rrf", _hyde(False), _hyde_estimate),
        Arm("hyde_passage_query", "hybrid_weighted_rrf", _hyde(True), _hyde_estimate),
    ]
}


def collection_name(dataset: str, variant: str, sample_docs: int | None) -> str:
    base = collection_for(dataset, get_variant(variant))
    return f"{base}-s{sample_docs}" if sample_docs else base


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
        for question in ds.queries.values():
            used = arm.estimate(question)
            tokens += used
            calls += used > 0
    return calls, tokens


async def run_arm(
    arm: Arm, ds: Dataset, collection: str, deps: Deps, llm: Any, concurrency: int
) -> tuple[list[dict[str, Any]], Usage]:
    semaphore = asyncio.Semaphore(concurrency)
    usage = Usage()

    async def one(qid: str, question: str) -> dict[str, Any]:
        nonlocal usage
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
        return result_row(arm.name, qid, result, ms, ds.qrels[qid])

    rows = await asyncio.gather(*(one(qid, q) for qid, q in ds.queries.items()))
    return list(rows), usage


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
        arm.name: collection_name(ds.name, arm.variant, args.sample_docs) for arm in arms
    }
    done: set[str] = set()
    for arm in arms:
        coll = collections[arm.name]
        if coll in done:
            continue
        done.add(coll)
        started = time.perf_counter()
        count = await index_chunks(deps, coll, chunks_for(ds.corpus, get_variant(arm.variant)))
        print(f"{coll}: {count} indexed ({time.perf_counter() - started:.0f}s)")
    await warm_up(deps)

    rows: list[dict[str, Any]] = []
    usage = Usage()
    try:
        for arm in arms:
            t0 = time.perf_counter()
            got, used = await run_arm(arm, ds, collections[arm.name], deps, llm, args.concurrency)
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
