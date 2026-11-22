"""Sufficiency eval: does the label agree with whether a gold document was retrieved?

    docker compose up -d qdrant
    uv run --extra eval --extra local-models python -m \
        production_rag.methods.retrieval_sufficiency.eval --dataset scifact --estimate-only
    uv run --extra eval --extra local-models python -m \
        production_rag.methods.retrieval_sufficiency.eval --dataset scifact --max-tokens 600000

Each query is retrieved and the model says `sufficient` or `insufficient`. That is compared with
a proxy: `sufficient` should mean a gold document is in the top k. The proxy is not a measure of
whether the passages answer the question. A query whose label fails (bad output, failed call or
failed retrieval) is `failed` and left out of agreement; it is never scored as `insufficient`.
The results hold no query text. `OPENAI_API_KEY` is checked before any download.
`--estimate-only` retrieves and makes no LLM call. `--max-tokens` caps the run; a query cut by
the budget is dropped and the run is marked incomplete (exit 1).
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from production_rag.core.config import Settings
from production_rag.core.deps import Deps, build_deps, warm_up
from production_rag.core.eval.answer import estimate_tokens
from production_rag.core.eval.datasets import load_dataset
from production_rag.core.eval.run import chunks_for, collection_for, doc_ranking
from production_rag.core.eval.run_llm import fail_if_all_fell_back
from production_rag.core.indexing import index_chunks
from production_rag.core.llm import BudgetExceeded, MissingAPIKey, OpenAIChat, TokenBudget
from production_rag.core.models import Row
from production_rag.methods import all_variants, get_variant
from production_rag.methods._variant import run_variant
from production_rag.methods.retrieval_sufficiency.judge import (
    LABELS,
    MAX_TOKENS,
    sufficiency_messages,
)
from production_rag.methods.retrieval_sufficiency.pipeline import BASE, search

RESULTS = Path("results")


def estimate_query(question: str, rows: list[Row]) -> int:
    """Upper bound of tokens for one query: the prompt plus the output cap."""
    prompt = sum(estimate_tokens(m["content"]) for m in sufficiency_messages(question, rows))
    return prompt + MAX_TOKENS


def _item(qid: str, label: str | None, hit: bool, error: str | None, tokens: int) -> dict[str, Any]:
    return {
        "query_id": qid,
        "predicted": label or "failed",
        "gold_hit": hit,
        "status": "ok" if label else "failed",
        "error": error,
        "tokens": tokens,
    }


async def run_queries(
    *,
    deps: Deps,
    collection: str,
    queries: dict[str, str],
    gold_docs: dict[str, list[str]],
    llm,
    base: str,
    k: int = 5,
    concurrency: int = 4,
) -> tuple[list[dict[str, Any]], bool]:
    """Items in query order, and False when the token budget stopped the run.

    A query cut by the budget is dropped.
    """
    semaphore = asyncio.Semaphore(concurrency)
    stopped = False

    async def one(qid: str) -> dict[str, Any] | None:
        nonlocal stopped
        async with semaphore:
            if stopped:
                return None
            try:
                got = await search(queries[qid], collection, deps, llm, base=base, k=k)
            except BudgetExceeded:
                stopped = True
                return None
            except Exception as exc:  # one broken query must not lose the others
                return _item(qid, None, False, type(exc).__name__, 0)
        ranked = doc_ranking(got.rows)
        hit = any(doc in ranked for doc in gold_docs[qid])
        return _item(qid, got.label, hit, got.error, got.usage.total)

    done = await asyncio.gather(*(one(qid) for qid in queries))
    return [item for item in done if item is not None], not stopped


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [i for i in items if i["status"] == "ok"]
    confusion = {
        label: {
            "hit": sum(i["predicted"] == label and i["gold_hit"] for i in ok),
            "miss": sum(i["predicted"] == label and not i["gold_hit"] for i in ok),
        }
        for label in LABELS
    }
    agreed = confusion["sufficient"]["hit"] + confusion["insufficient"]["miss"]
    return {
        "queries": len(items),
        "failed": len(items) - len(ok),
        "agreement": _ratio(agreed, len(ok)),
        "sufficient_rate": _ratio(sum(i["predicted"] == "sufficient" for i in ok), len(ok)),
        "gold_hit_rate": _ratio(sum(i["gold_hit"] for i in ok), len(ok)),
        "confusion": confusion,
    }


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def report_md(run: dict[str, Any]) -> str:
    meta, s = run["meta"], run["summary"]
    lines = [
        f"# Retrieval sufficiency on {meta['dataset']}",
        "",
        f"{meta['queries']} queries, {meta['documents']:,} documents, base `{meta['base']}`, "
        f"k={meta['k']}, model `{meta['model']}`, {meta['date']}, one run. The label is the "
        'model\'s own judgement. Agreement with "a gold document is in the top k" is a proxy '
        'for "enough to answer", not a measure of answerability. Failed items are excluded.',
        "",
    ]
    if not meta["complete"]:
        lines += [
            f"Incomplete run: the token budget stopped it, {meta['queries']} of "
            f"{meta['requested']} requested queries kept.",
            "",
        ]
    answered = s["queries"] - s["failed"]
    lines += [
        f"Agreement {_fmt(s['agreement'], '.3f')} on {answered} answered queries, "
        f"{s['failed']} failed. `sufficient` on {_fmt(s['sufficient_rate'], '.1%')} of them; "
        f"a gold document is in the top {meta['k']} for {_fmt(s['gold_hit_rate'], '.1%')}.",
        "",
        "| Label | Gold in top k | Gold not in top k |",
        "|---|---|---|",
    ]
    for label in LABELS:
        row = s["confusion"][label]
        lines.append(f"| {label} | {row['hit']} | {row['miss']} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scifact", "tvpl"], required=True)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--base", default=BASE, help="retrieval variant (default: %(default)s)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, help="token cap for the run")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--name")
    args = parser.parse_args(argv)
    if args.base not in all_variants():
        parser.error(f"unknown base {args.base!r}, known: {', '.join(all_variants())}")
    if args.max_tokens is None and not args.estimate_only:
        parser.error("--max-tokens is required for a run that calls the LLM")
    args.name = args.name or f"retrieval_sufficiency_{args.dataset}"
    return args


def preflight(args: argparse.Namespace) -> OpenAIChat | None:
    """The LLM client, or None for `--estimate-only`. Runs before any download."""
    if args.estimate_only:
        return None
    try:
        return OpenAIChat.from_env(budget=TokenBudget(args.max_tokens))
    except MissingAPIKey:
        raise SystemExit("set OPENAI_API_KEY to run this eval, or pass --estimate-only") from None


async def estimate(deps: Deps, collection: str, queries: dict[str, str], base: str, k: int):
    """LLM calls and upper-bound tokens, from real retrieval only."""
    calls = tokens = 0
    for question in queries.values():
        found = await run_variant(all_variants(), base, question[:2000], collection, deps, k, 4000)
        if found.status == "ok" and found.rows:  # no rows: no call
            calls += 1
            tokens += estimate_query(question[:2000], found.rows)
    return calls, tokens


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    llm = preflight(args)
    deps = build_deps(Settings.from_env())
    ds = load_dataset(args.dataset, args.query_limit)
    method = get_variant(args.base)
    collection = collection_for(args.dataset, method)
    started = time.perf_counter()
    count = await index_chunks(deps, collection, chunks_for(ds.corpus, method))
    print(f"{collection}: {count} indexed ({time.perf_counter() - started:.0f}s)")
    print(f"{len(ds.corpus)} documents, {len(ds.queries)} queries")
    await warm_up(deps)

    if args.estimate_only:
        calls, tokens = await estimate(deps, collection, ds.queries, args.base, args.k)
        print(f"estimate: queries={len(ds.queries)} llm_calls={calls} upper_bound_tokens={tokens}")
        return 0

    gold = {qid: [d for d, score in rel.items() if score > 0] for qid, rel in ds.qrels.items()}
    items, complete = await run_queries(
        deps=deps,
        collection=collection,
        queries=ds.queries,
        gold_docs=gold,
        llm=llm,
        base=args.base,
        k=args.k,
        concurrency=args.concurrency,
    )
    summary = summarize(items)
    fail_if_all_fell_back("retrieval_sufficiency", summary["failed"], len(items))
    meta = {
        "kind": "retrieval_sufficiency",
        "dataset": args.dataset,
        "date": date.today().isoformat(),
        "requested": len(ds.queries),
        "queries": len(items),
        "documents": len(ds.corpus),
        "base": args.base,
        "k": args.k,
        "model": llm.model,
        "complete": complete,
        "tokens": llm.budget.used,
    }
    run = {"meta": meta, "summary": summary, "per_item": items}
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.name}.json"
    out.write_text(json.dumps(run, indent=1))
    md = report_md(run)
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"tokens: {llm.budget.used}\nsaved {out}")
    if not complete:
        print("stopped at the token budget: partial results, queries cut mid-way dropped")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
