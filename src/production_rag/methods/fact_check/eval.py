"""Fact-check eval: judge SciFact dev claims against retrieved abstracts.

    docker compose up -d qdrant
    uv run --extra eval --extra local-models python -m production_rag.methods.fact_check.eval \
        --sample 300 --estimate-only
    uv run --extra eval --extra local-models python -m production_rag.methods.fact_check.eval \
        --sample 300 --max-tokens 400000
    uv run python -m production_rag.methods.fact_check.eval --compare run1.json run2.json

Each claim is retrieved against the SciFact abstracts and judged as given (no claim extraction)
as supported, refuted or unverified, then scored against the dataset label. A verdict that
fails (bad output, failed call or failed retrieval) is `failed` and left out of accuracy,
precision and recall; it is never scored as `unverified`. The results hold no claim text and no
model reason. `OPENAI_API_KEY` is checked before any download. `--estimate-only` retrieves and
makes no LLM call. `--compare` reads two results files and needs neither a key nor a cap.

Token budget and failures follow the answer eval: `--max-tokens` caps the run, a claim cut by
the budget is dropped and the run is marked incomplete (exit 1). An OpenAI 401 or 403 stops the
run with a message naming `OPENAI_API_KEY` and writes nothing.
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
from production_rag.core.eval.datasets import load_claims, load_dataset
from production_rag.core.eval.run import chunks_for, collection_for, doc_ranking
from production_rag.core.indexing import index_chunks
from production_rag.core.llm import (
    AuthFailed,
    BudgetExceeded,
    MissingAPIKey,
    OpenAIChat,
    TokenBudget,
)
from production_rag.core.models import Row
from production_rag.methods import all_variants, get_variant
from production_rag.methods.fact_check import verdict
from production_rag.methods.fact_check.pipeline import BASE, check_claim, search

RESULTS = Path("results")
GOLD_LABELS = ("supported", "refuted", "unverified")
PREDICTED = (*GOLD_LABELS, "failed")
DEFAULT_NAME = "fact_check_scifact"
RUNS_FILE = "fact_check_scifact_runs.md"


def estimate_claim(claim: str, rows: list[Row]) -> int:
    """Upper bound of tokens for one claim: the verdict prompt plus the output cap."""
    prompt = sum(estimate_tokens(m["content"]) for m in verdict.verdict_messages(claim, rows))
    return prompt + verdict.MAX_TOKENS


def _item(
    claim_id: str, gold: str, predicted: str, error: str | None, doc_ids: list[str], hit: bool
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "gold": gold,
        "predicted": predicted,
        "status": "failed" if predicted == "failed" else "ok",
        "error": error,
        "doc_ids": doc_ids,
        "evidence_hit": hit,
        "ms": None,
        "tokens": 0,
        "reasoning": 0,
    }


async def run_claims(
    *,
    deps: Deps,
    collection: str,
    claims: dict[str, str],
    labels: dict[str, str],
    gold_docs: dict[str, list[str]],
    llm,
    base: str,
    k: int = 5,
    concurrency: int = 4,
) -> tuple[list[dict[str, Any]], bool]:
    """Items in claim order, and False when the token budget stopped the run.

    A claim cut by the budget is dropped. Raises `AuthFailed` (after in-flight claims finish)
    when OpenAI refused the key.
    """
    semaphore = asyncio.Semaphore(concurrency)
    stopped = False
    auth_error: AuthFailed | None = None

    async def one(cid: str) -> dict[str, Any] | None:
        nonlocal stopped, auth_error
        async with semaphore:  # latency is measured inside, so queueing is not counted
            if stopped or auth_error:
                return None
            start = time.perf_counter()
            try:
                got = await check_claim(claims[cid], collection, deps, llm, base=base, k=k)
            except BudgetExceeded:
                stopped = True
                return None
            except AuthFailed as exc:
                auth_error = exc
                return None
            except Exception as exc:  # one broken claim must not lose the others
                return _item(cid, labels[cid], "failed", type(exc).__name__, [], False)
            ms = round((time.perf_counter() - start) * 1000)
        doc_ids = doc_ranking(got.rows)
        hit = any(doc in doc_ids for doc in gold_docs[cid])
        label, error = got.label, got.error
        if not got.rows and not error:  # judge_claim's "unverified" without a call is no verdict
            label, error = None, "no_evidence"
        item = _item(cid, labels[cid], label or "failed", error, doc_ids, hit)
        return {
            **item,
            "ms": ms,
            "tokens": got.usage.total,
            "reasoning": got.usage.reasoning_tokens,
        }

    done = await asyncio.gather(*(one(cid) for cid in claims))
    if auth_error:
        raise auth_error
    return [item for item in done if item is not None], not stopped


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize_verdicts(items: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [i for i in items if i["status"] == "ok"]
    per_label = {}
    for label in GOLD_LABELS:
        gold_ok = sum(i["gold"] == label for i in ok)
        predicted = sum(i["predicted"] == label for i in ok)
        correct = sum(i["gold"] == label == i["predicted"] for i in ok)
        per_label[label] = {
            "gold": sum(i["gold"] == label for i in items),
            "predicted": predicted,
            "correct": correct,
            "precision": _ratio(correct, predicted),
            "recall": _ratio(correct, gold_ok),
        }
    confusion = {
        gold: {
            pred: sum(i["gold"] == gold and i["predicted"] == pred for i in items)
            for pred in PREDICTED
        }
        for gold in GOLD_LABELS
    }
    checkable = [i for i in items if i["gold"] in ("supported", "refuted")]
    ok_checkable = [i for i in checkable if i["status"] == "ok"]

    def accuracy(rows: list[dict[str, Any]]) -> float | None:
        return _ratio(sum(i["gold"] == i["predicted"] for i in rows), len(rows))

    hit_ok = [i for i in ok_checkable if i["evidence_hit"]]
    miss_ok = [i for i in ok_checkable if not i["evidence_hit"]]
    return {
        "claims": len(items),
        "failed": len(items) - len(ok),
        "accuracy": accuracy(ok),
        "unverified_share": _ratio(sum(i["predicted"] == "unverified" for i in ok), len(ok)),
        "per_label": per_label,
        "confusion": confusion,
        "evidence_hit_rate": _ratio(sum(i["evidence_hit"] for i in checkable), len(checkable)),
        "accuracy_hit": accuracy([i for i in ok_checkable if i["evidence_hit"]]),
        "accuracy_miss": accuracy(miss_ok),
        "checkable": len(checkable),
        "evidence_found": sum(i["evidence_hit"] for i in checkable),
        "hit_answered": len(hit_ok),
        "hit_correct": sum(i["gold"] == i["predicted"] for i in hit_ok),
        "miss_answered": len(miss_ok),
        "miss_correct": sum(i["gold"] == i["predicted"] for i in miss_ok),
    }


def agreement(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare two runs on the claims answered (not failed) in both."""
    by_b = {i["claim_id"]: i for i in b if i["status"] == "ok"}
    pairs = [(i, by_b[i["claim_id"]]) for i in a if i["status"] == "ok" and i["claim_id"] in by_b]
    return {
        "claims": len(pairs),
        "same_label": _ratio(sum(x["predicted"] == y["predicted"] for x, y in pairs), len(pairs)),
        "accuracy_a": _ratio(sum(x["gold"] == x["predicted"] for x, _ in pairs), len(pairs)),
        "accuracy_b": _ratio(sum(y["gold"] == y["predicted"] for _, y in pairs), len(pairs)),
    }


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def report_md(run: dict[str, Any]) -> str:
    meta, s = run["meta"], run["summary"]
    lines = [
        "# Fact check on SciFact claims",
        "",
        f"{meta['claims']} claims from the dev split, {meta['documents']:,} abstracts, "
        f"base `{meta['base']}`, k={meta['k']}, model `{meta['model']}`, {meta['date']}, "
        "one run. Failed items are excluded from accuracy, precision and recall.",
        "",
    ]
    if not meta["complete"]:
        lines += [
            f"Incomplete run: the token budget stopped it, {meta['claims']} of "
            f"{meta['requested']} requested claims kept.",
            "",
        ]
    lines += [
        "| Label | Gold | Predicted | Correct | Precision | Recall |",
        "|---|---|---|---|---|---|",
    ]
    for label in GOLD_LABELS:
        r = s["per_label"][label]
        lines.append(
            f"| {label} | {r['gold']} | {r['predicted']} | {r['correct']} | "
            f"{_fmt(r['precision'], '.3f')} | {_fmt(r['recall'], '.3f')} |"
        )
    lines += [
        "",
        f"Accuracy {_fmt(s['accuracy'], '.3f')} on {s['claims'] - s['failed']} claims, "
        f"{s['failed']} failed, unverified {_fmt(s['unverified_share'], '.1%')}. "
        f"On the {s['checkable']} supported or refuted claims, the gold abstract is in the top "
        f"{meta['k']} for {s['evidence_found']} ({_fmt(s['evidence_hit_rate'], '.1%')}); "
        f"accuracy {_fmt(s['accuracy_hit'], '.3f')} when found "
        f"({s['hit_correct']}/{s['hit_answered']}), "
        f"{_fmt(s['accuracy_miss'], '.3f')} when not "
        f"({s['miss_correct']}/{s['miss_answered']}).",
        "",
        "| Gold \\ predicted | supported | refuted | unverified | failed |",
        "|---|---|---|---|---|",
    ]
    for gold in GOLD_LABELS:
        row = s["confusion"][gold]
        lines.append(f"| {gold} | " + " | ".join(str(row[p]) for p in PREDICTED) + " |")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--base", default=BASE, help="retrieval variant (default: %(default)s)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, help="token cap for the run")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two results JSON files")
    args = parser.parse_args(argv)
    if args.base not in all_variants():
        parser.error(f"unknown base {args.base!r}, known: {', '.join(all_variants())}")
    if args.max_tokens is None and not (args.estimate_only or args.compare):
        parser.error("--max-tokens is required for a run that calls the LLM")
    return args


def preflight(args: argparse.Namespace) -> OpenAIChat | None:
    """The LLM client, or None for `--estimate-only` and `--compare`. Runs before any download."""
    if args.estimate_only or args.compare:
        return None
    try:
        return OpenAIChat.from_env(budget=TokenBudget(args.max_tokens))
    except MissingAPIKey:
        raise SystemExit(
            "set OPENAI_API_KEY to run the fact-check eval, or pass --estimate-only"
        ) from None


def compare_md(a: dict[str, Any], b: dict[str, Any]) -> str:
    g = agreement(a["per_item"], b["per_item"])
    lines = [
        "# Fact check runs compared",
        "",
        f"Same label on {_fmt(g['same_label'], '.1%')} of {g['claims']} claims answered in "
        f"both runs. Accuracy on those claims: run A {_fmt(g['accuracy_a'], '.3f')}, "
        f"run B {_fmt(g['accuracy_b'], '.3f')}.",
        "",
        "| Run | Date | Model | Base | k | Claims | Accuracy |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in (("A", a), ("B", b)):
        m = run["meta"]
        lines.append(
            f"| {name} | {m['date']} | {m['model']} | {m['base']} | {m['k']} | {m['claims']} | "
            f"{_fmt(run['summary']['accuracy'], '.3f')} |"
        )
    return "\n".join(lines) + "\n"


def compare(path_a: str, path_b: str) -> int:
    runs = [json.loads(Path(p).read_text()) for p in (path_a, path_b)]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / RUNS_FILE
    out.write_text(compare_md(*runs))
    print(out.read_text())
    print(f"saved {out}")
    return 0


async def estimate(
    deps: Deps, collection: str, ds, base: str, k: int
) -> tuple[int, int, float | None, int]:
    """LLM calls, upper-bound tokens and evidence hit rate, from real retrieval only."""
    calls = tokens = failed = 0
    hits: list[bool] = []
    for cid, claim in ds.claims.items():
        found = await search(claim, collection, deps, base=base, k=k)
        rows = found.rows if found.status == "ok" else []
        failed += found.status != "ok"
        calls += bool(rows)  # no rows -> no call (the run marks it failed)
        tokens += estimate_claim(claim, rows) if rows else 0
        if ds.labels[cid] != "unverified":  # a failed retrieval is a miss, as in the run
            ranked = doc_ranking(rows)
            hits.append(any(doc in ranked for doc in ds.gold_docs[cid]))
    return calls, tokens, _ratio(sum(hits), len(hits)), failed


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    llm = preflight(args)
    deps = build_deps(Settings.from_env())

    try:
        ds = load_claims(args.sample)
        corpus = load_dataset("scifact", query_limit=1).corpus
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"missing {exc.name}: install the eval extra: uv sync --extra eval"
        ) from None
    method = get_variant(args.base)
    collection = collection_for("scifact", method)
    started = time.perf_counter()
    chunks = chunks_for(corpus, method)
    count = await index_chunks(deps, collection, chunks)
    print(f"{collection}: {count} chunks ({time.perf_counter() - started:.0f}s)")
    print(f"{len(corpus)} documents, {len(ds.claims)} claims")
    await warm_up(deps)

    if args.estimate_only:
        calls, tokens, rate, failed = await estimate(deps, collection, ds, args.base, args.k)
        print(
            f"estimate: claims={len(ds.claims)} llm_calls={calls} "
            f"upper_bound_tokens={tokens} evidence_hit_rate={_fmt(rate, '.3f')} "
            f"retrieval_failed={failed}"
        )
        return 0

    try:
        items, complete = await run_claims(
            deps=deps,
            collection=collection,
            claims=ds.claims,
            labels=ds.labels,
            gold_docs=ds.gold_docs,
            llm=llm,
            base=args.base,
            k=args.k,
            concurrency=args.concurrency,
        )
    except AuthFailed as exc:  # the label is status and error code only, never the key
        raise SystemExit(f"OpenAI refused OPENAI_API_KEY ({exc}); no results written") from None
    summary = summarize_verdicts(items)
    kept = sum(i["tokens"] for i in items)
    billed = llm.budget.used
    meta = {
        "kind": "fact_check",
        "date": date.today().isoformat(),
        "requested": len(ds.claims),
        "claims": len(items),
        "documents": len(corpus),
        "base": args.base,
        "k": args.k,
        "model": llm.model,
        "complete": complete,
        "tokens": {
            "total": billed,
            "kept": kept,
            "dropped": billed - kept,
            "reasoning": sum(i["reasoning"] for i in items),
        },
    }
    run = {"meta": meta, "summary": summary, "per_item": items}
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.name}.json"
    out.write_text(json.dumps(run, indent=1))
    md = report_md(run)
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"tokens: total={billed} kept={kept} dropped={billed - kept}")
    if not complete:
        print("stopped at the token budget: partial results, claims cut mid-way dropped")
    print(f"saved {out}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
