"""Answer eval: retrieve, generate an answer, then judge it with an LLM.

    docker compose up -d qdrant
    uv run --extra eval --extra local-models python -m production_rag.core.eval.answer \
        --dataset mini_wiki --sample 20 --estimate-only
    uv run --extra eval --extra local-models python -m production_rag.core.eval.answer \
        --dataset mini_wiki --sample 20 --max-tokens 400000

Every front end retrieves the top k chunks from the same 800-character chunk collection, the
shared generator answers from them, and two judges grade the answer: correctness against the
gold answer and faithfulness to the passages. An abstention (`NO_ANSWER`) scores incorrect and
faithful without a judge call. `OPENAI_API_KEY` is checked before any download or indexing.
`--estimate-only` indexes and retrieves, makes no LLM call and prints an upper bound of tokens.

Token budget: `--max-tokens` caps the run. The client checks the budget before a call and adds
the usage after it, so calls already in flight can pass the cap. Each question makes its calls
one after another, so at most `--concurrency` calls are in flight and the overshoot is at most
that many calls (each at most its prompt plus its max_tokens). A question cut by the budget is
dropped, the results are written with `"complete": false` and the exit code is 1. The meta
`tokens.total` is what the client counted (billed), including calls of dropped questions;
`rag` and `judge` sum the kept items and `dropped` is the difference.

Failures: an OpenAI 401 or 403 stops the run with a message naming `OPENAI_API_KEY` and writes
no results file. A run where every generation call failed is written as incomplete and exits 1.
Any other error in one question makes that item `failed` (exception class name only).
"""

import argparse
import asyncio
import dataclasses
import json
import math
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from production_rag.core.config import Settings
from production_rag.core.deps import Deps, build_deps, warm_up
from production_rag.core.eval.datasets import ANSWER_DATASETS, DEFAULT_SAMPLE, load_answer_dataset
from production_rag.core.eval.judge import MAX_TOKENS as JUDGE_MAX_TOKENS
from production_rag.core.eval.judge import (
    SCORES,
    correctness_messages,
    faithfulness_messages,
    format_passages,
    judge_correctness,
    judge_faithfulness,
)
from production_rag.core.eval.metrics import percentile
from production_rag.core.eval.report import answer_table
from production_rag.core.eval.run import doc_ranking
from production_rag.core.generation import build_messages, generate
from production_rag.core.llm import (
    BudgetExceeded,
    LLMError,
    MissingAPIKey,
    OpenAIChat,
    TokenBudget,
    Usage,
)
from production_rag.core.models import ContextOption, Row, SpecResult
from production_rag.methods.context_expansion import pipeline as context_expansion
from production_rag.methods.hybrid_fusion import pipeline as hybrid_fusion
from production_rag.methods.multi_query import pipeline as multi_query
from production_rag.methods.multi_query.rewrite import MAX_TOKENS as REWRITE_MAX_TOKENS
from production_rag.methods.multi_query.rewrite import rewrite, rewrite_messages
from production_rag.methods.naive_rag import pipeline as naive_rag
from production_rag.methods.rerank import pipeline as rerank

RESULTS = Path("results")
ANSWER_METHODS = ("naive_rag", "hybrid_fusion", "rerank", "multi_query", "context_expansion")
DEADLINE_MS = 30000
GEN_MAX_TOKENS = 512
ANSWER_IN_JUDGE = 512  # upper bound for the answer text inside each judge prompt
RERANK_NOT_APPLIED = ("failed", "skipped_budget")  # rows came back in retrieval order
NO_RERANKER = (
    "the rerank front end needs a reranker: install the local-models extra "
    "or set RERANK_BASE_URL and RERANK_API_KEY"
)


class AuthFailed(RuntimeError):
    """OpenAI refused the key: every further call would fail the same way."""


def _is_auth_error(exc: Exception) -> bool:
    return isinstance(exc, LLMError) and str(exc).startswith(("openai 401", "openai 403"))


@dataclass
class Retrieved:
    rows: list[Row]
    usage: Usage
    ms: int
    rewrite_ms: int | None = None
    status: str = "ok"
    rerank: str = "off"


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


async def _timed(search: Awaitable[SpecResult]) -> Retrieved:
    start = time.perf_counter()
    result = await search
    return Retrieved(result.rows, Usage(), _ms(start), status=result.status, rerank=result.rerank)


async def _naive_rag(question: str, collection: str, deps: Deps, llm: Any, k: int) -> Retrieved:
    return await _timed(naive_rag.search(question, collection, deps, k=k, deadline_ms=DEADLINE_MS))


async def _hybrid_fusion(question: str, collection: str, deps: Deps, llm: Any, k: int) -> Retrieved:
    return await _timed(
        hybrid_fusion.search(question, collection, deps, k=k, deadline_ms=DEADLINE_MS)
    )


async def _rerank(question: str, collection: str, deps: Deps, llm: Any, k: int) -> Retrieved:
    if deps.reranker is None:  # the pipeline would silently skip the rerank stage
        raise RuntimeError(NO_RERANKER)
    return await _timed(rerank.search(question, collection, deps, k=k, deadline_ms=DEADLINE_MS))


async def _multi_query(question: str, collection: str, deps: Deps, llm: Any, k: int) -> Retrieved:
    """With `llm=None` (the estimate) it searches with the question alone."""
    rewrites, usage, rewrite_ms = [], Usage(), None
    if llm is not None:
        start = time.perf_counter()
        rewrites, usage = await rewrite(llm, question)
        rewrite_ms = _ms(start)
    got = await _timed(
        multi_query.search(
            [question, *rewrites],
            collection,
            deps,
            base="hybrid_weighted_rrf",
            k=k,
            deadline_ms=DEADLINE_MS,
        )
    )
    return dataclasses.replace(got, usage=usage, rewrite_ms=rewrite_ms)


async def _context_expansion(
    question: str, collection: str, deps: Deps, llm: Any, k: int
) -> Retrieved:
    return await _timed(
        context_expansion.search(
            question,
            collection,
            deps,
            context=ContextOption(prev=1, next=1),
            k=k,
            deadline_ms=DEADLINE_MS,
        )
    )


FRONT_ENDS: dict[str, Callable[[str, str, Deps, Any, int], Awaitable[Retrieved]]] = {
    "naive_rag": _naive_rag,
    "hybrid_fusion": _hybrid_fusion,
    "rerank": _rerank,
    "multi_query": _multi_query,
    "context_expansion": _context_expansion,
}


def collection_for(dataset: str) -> str:
    return f"answer-{dataset}-c{naive_rag.CHUNK_CHARS}"


def estimate_tokens(text: str) -> int:
    """Conservative for both languages: measured 4.7 chars/token English, 3.5 Vietnamese."""
    return math.ceil(len(text) / 3)


def _prompt_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(m["content"]) for m in messages)


def llm_calls(method: str) -> int:
    return 3 + (method == "multi_query")


def estimate_item(question: str, gold: str, rows: list[Row], method: str) -> int:
    """Upper bound of tokens for one question and front end: every output at its cap."""
    total = _prompt_tokens(build_messages(question, rows)) + GEN_MAX_TOKENS
    total += _prompt_tokens(correctness_messages(question, gold, "")) + ANSWER_IN_JUDGE
    total += _prompt_tokens(faithfulness_messages(question, "", rows)) + ANSWER_IN_JUDGE
    total += 2 * JUDGE_MAX_TOKENS
    if method == "multi_query":
        total += _prompt_tokens(rewrite_messages(question)) + REWRITE_MAX_TOKENS
    return total


def _tokens(rag: Usage, judge: Usage) -> dict[str, int]:
    return {
        "rag": rag.total,
        "judge": judge.total,
        "reasoning": rag.reasoning_tokens + judge.reasoning_tokens,
    }


def _new_item(method: str, qid: str) -> dict[str, Any]:
    return {
        "qid": qid,
        "front_end": method,
        "status": "ok",
        "error": None,
        "abstained": False,
        "answer": None,
        "correctness": None,
        "faithfulness": None,
        "correctness_error": None,
        "faithfulness_error": None,
        "retrieval_ms": None,
        "rewrite_ms": None,
        "gen_ms": None,
        "passage_tokens": None,
        "doc_ids": [],
        "rerank": None,
        "tokens": _tokens(Usage(), Usage()),
    }


async def run_one(
    method: str, qid: str, question: str, gold: str, deps: Deps, collection: str, llm, k: int
) -> dict[str, Any]:
    """One question through one front end. `BudgetExceeded` and `AuthFailed` propagate."""
    item = _new_item(method, qid)
    try:
        got = await FRONT_ENDS[method](question, collection, deps, llm, k)
    except BudgetExceeded:
        raise
    except Exception as exc:
        return {**item, "status": "failed", "error": f"retrieval {type(exc).__name__}"}
    rag = got.usage
    item.update(
        retrieval_ms=got.ms,
        rewrite_ms=got.rewrite_ms,
        rerank=got.rerank,
        tokens=_tokens(rag, Usage()),
    )
    if got.status != "ok":
        return {**item, "status": "failed", "error": "retrieval failed"}
    if got.rerank in RERANK_NOT_APPLIED:  # unreranked rows must not count as reranked
        return {**item, "status": "failed", "error": f"rerank {got.rerank}"}
    item.update(
        doc_ids=doc_ranking(got.rows), passage_tokens=estimate_tokens(format_passages(got.rows))
    )
    try:
        gen = await generate(llm, question, got.rows, max_tokens=GEN_MAX_TOKENS)
    except BudgetExceeded:
        raise
    except Exception as exc:
        if _is_auth_error(exc):
            raise AuthFailed(str(exc)) from None
        return {**item, "status": "failed", "error": f"generation {type(exc).__name__}"}
    rag += gen.usage
    item.update(
        answer=gen.text,
        abstained=gen.abstained,
        gen_ms=gen.ms if got.rows else None,
        tokens=_tokens(rag, Usage()),
    )
    if gen.abstained:
        return {**item, "correctness": "incorrect", "faithfulness": "faithful"}
    correctness = await judge_correctness(llm, question, gold, gen.text)
    faithfulness = await judge_faithfulness(llm, question, gen.text, got.rows)
    item.update(
        correctness=correctness.label,
        faithfulness=faithfulness.label,
        correctness_error=correctness.error,
        faithfulness_error=faithfulness.error,
        tokens=_tokens(rag, correctness.usage + faithfulness.usage),
    )
    if correctness.error or faithfulness.error:
        item["status"] = "judge_failed"
    return item


async def run_items(
    *,
    deps: Deps,
    collection: str,
    methods: list[str],
    questions: dict[str, str],
    gold: dict[str, str],
    llm,
    k: int = 5,
    concurrency: int = 4,
) -> tuple[list[dict[str, Any]], bool]:
    """Items in method then question order, and False when the token budget stopped the run.

    Raises `AuthFailed` (after in-flight questions finish) when OpenAI refused the key.
    """
    semaphore = asyncio.Semaphore(concurrency)
    stopped = False
    auth_error: AuthFailed | None = None

    async def one(method: str, qid: str) -> dict[str, Any] | None:
        nonlocal stopped, auth_error
        async with semaphore:  # latency is measured inside, so queueing is not counted
            if stopped or auth_error:
                return None
            try:
                return await run_one(
                    method, qid, questions[qid], gold[qid], deps, collection, llm, k
                )
            except BudgetExceeded:
                stopped = True
                return None
            except AuthFailed as exc:
                auth_error = exc
                return None
            except Exception as exc:  # one broken question must not lose the others
                return {**_new_item(method, qid), "status": "failed", "error": type(exc).__name__}

    items: list[dict[str, Any]] = []
    for method in methods:
        start = time.perf_counter()
        done = await asyncio.gather(*(one(method, qid) for qid in questions))
        items.extend(item for item in done if item is not None)
        if auth_error:
            raise auth_error
        print(f"{method:18} done in {time.perf_counter() - start:.0f}s", flush=True)
        if stopped:
            break
    return items, not stopped


def all_generations_failed(items: list[dict[str, Any]]) -> bool:
    """True when generation was tried and never succeeded (an outage, not a bad question)."""
    tried = [
        i for i in items if i["gen_ms"] is not None or (i["error"] or "").startswith("generation")
    ]
    return bool(tried) and all(i["status"] == "failed" for i in tried)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _int_mean(values: list[int]) -> int | None:
    return round(sum(values) / len(values)) if values else None


def _p50(values: list[int]) -> int | None:
    return round(percentile(values, 50)) if values else None


def summarize_answers(items: list[dict], methods: list[str]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for method in methods:
        mine = [i for i in items if i["front_end"] == method]
        ok = [i for i in mine if i["status"] != "failed"]
        entry = {
            "questions": len(mine),
            "failed": len(mine) - len(ok),
            "judge_failed": sum(i["status"] == "judge_failed" for i in mine),
            "correctness": _mean(
                [SCORES[i["correctness"]] for i in ok if i["correctness"] is not None]
            ),
            "faithfulness": _mean(
                [i["faithfulness"] == "faithful" for i in ok if i["faithfulness"] is not None]
            ),
            "abstained": _mean([i["abstained"] for i in ok]),
            "retrieval_p50_ms": _p50(
                [i["retrieval_ms"] for i in mine if i["retrieval_ms"] is not None]
            ),
            "generation_p50_ms": _p50([i["gen_ms"] for i in mine if i["gen_ms"] is not None]),
            "rag_tokens": _int_mean([i["tokens"]["rag"] for i in mine]),
            "passage_tokens": _int_mean(
                [i["passage_tokens"] for i in ok if i["passage_tokens"] is not None]
            ),
            "judge_tokens": _int_mean([i["tokens"]["judge"] for i in mine]),
        }
        if method == "multi_query":
            entry["rewrite_p50_ms"] = _p50(
                [i["rewrite_ms"] for i in mine if i["rewrite_ms"] is not None]
            )
        summary[method] = entry
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ANSWER_DATASETS, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ANSWER_METHODS),
        help=f"front ends (default: all of {', '.join(ANSWER_METHODS)})",
    )
    parser.add_argument("--sample", type=int, help="questions (default: per dataset)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, help="token cap for the run")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--name")
    args = parser.parse_args(argv)
    unknown = [m for m in args.methods if m not in ANSWER_METHODS]
    if unknown:
        parser.error(f"unknown method {unknown[0]!r}, known: {', '.join(ANSWER_METHODS)}")
    args.methods = list(dict.fromkeys(args.methods))
    if args.max_tokens is None and not args.estimate_only:
        parser.error("--max-tokens is required for a run that calls the LLM")
    if args.sample is None:
        args.sample = DEFAULT_SAMPLE[args.dataset]
    return args


def preflight(args: argparse.Namespace) -> OpenAIChat | None:
    """The LLM client, or None for `--estimate-only`. Runs before any download or indexing."""
    if args.estimate_only:
        return None
    try:
        return OpenAIChat.from_env(budget=TokenBudget(args.max_tokens))
    except MissingAPIKey:
        raise SystemExit(
            "set OPENAI_API_KEY to run the answer eval, or pass --estimate-only"
        ) from None


async def estimate(
    deps: Deps,
    collection: str,
    methods: list[str],
    questions: dict[str, str],
    gold: dict[str, str],
    k: int,
) -> tuple[int, int]:
    """LLM calls and upper-bound tokens, from real retrieval and no LLM call."""
    calls = tokens = 0
    for method in methods:
        for qid, question in questions.items():
            got = await FRONT_ENDS[method](question, collection, deps, None, k)
            tokens += estimate_item(question, gold[qid], got.rows, method)
            calls += llm_calls(method)
    return calls, tokens


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    llm = preflight(args)
    deps = build_deps(Settings.from_env())
    if "rerank" in args.methods and deps.reranker is None:
        raise SystemExit(NO_RERANKER)

    ds = load_answer_dataset(args.dataset, args.sample)
    collection = collection_for(args.dataset)
    started = time.perf_counter()
    chunks = await naive_rag.index(deps, collection, ds.corpus)
    print(f"{collection}: {chunks} chunks ({time.perf_counter() - started:.0f}s)")
    print(f"{len(ds.corpus)} documents, {len(ds.questions)} questions")
    await warm_up(deps)

    if args.estimate_only:
        calls, tokens = await estimate(
            deps, collection, args.methods, ds.questions, ds.gold, args.k
        )
        if "multi_query" in args.methods:
            print("multi_query: estimated from the question's own retrieval plus the rewrite call")
        print(
            f"estimate: questions={len(ds.questions)} methods={len(args.methods)} "
            f"llm_calls={calls} upper_bound_tokens={tokens}"
        )
        return 0

    try:
        items, complete = await run_items(
            deps=deps,
            collection=collection,
            methods=args.methods,
            questions=ds.questions,
            gold=ds.gold,
            llm=llm,
            k=args.k,
            concurrency=args.concurrency,
        )
    except AuthFailed as exc:  # the label is status and error code only, never the key
        raise SystemExit(f"OpenAI refused OPENAI_API_KEY ({exc}); no results written") from None
    budget_stop = not complete
    outage = all_generations_failed(items)
    complete = complete and not outage
    summary = summarize_answers(items, args.methods)
    rag = sum(i["tokens"]["rag"] for i in items)
    judge = sum(i["tokens"]["judge"] for i in items)
    budget = getattr(llm, "budget", None)
    billed = budget.used if budget is not None else rag + judge
    meta = {
        "kind": "answer",
        "dataset": args.dataset,
        "date": date.today().isoformat(),
        "questions": len(ds.questions),
        "documents": len(ds.corpus),
        "chunks": {collection: chunks},
        "model": llm.model,
        "k": args.k,
        "complete": complete,
        "tokens": {
            "total": billed,
            "rag": rag,
            "judge": judge,
            "dropped": billed - rag - judge,
            "reasoning": sum(i["tokens"]["reasoning"] for i in items),
        },
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.name or 'answer_' + args.dataset}.json"
    out.write_text(json.dumps({"meta": meta, "summary": summary, "per_item": items}, indent=1))
    print(answer_table(summary))
    print(f"tokens: rag={rag} judge={judge} dropped={billed - rag - judge} total={billed}")
    if budget_stop:
        print("stopped at the token budget: partial results, questions cut mid-way dropped")
    if outage:
        print("every generation call failed: run marked incomplete")
    print(f"saved {out}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
