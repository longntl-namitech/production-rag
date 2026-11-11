"""Latency of the fan-out under shrinking deadlines, against running the specs one by one.

    docker compose up -d qdrant
    uv run --extra eval --extra local-models python -m \
        production_rag.methods.deadline_fanout.bench --dataset tvpl --query-limit 200

Every query runs as one request with three specs (dense, sparse, hybrid with
rerank) at each deadline, one query at a time. The sequential baseline runs the
three specs as separate requests with a 30 s deadline and sums the time. No LLM calls.
"""

import argparse
import asyncio
import json
import os
import platform
import time
from datetime import date
from pathlib import Path
from typing import Any

from production_rag.core.chunking import chunk_documents
from production_rag.core.config import Settings
from production_rag.core.deps import Deps, build_deps, warm_up
from production_rag.core.eval.datasets import load_dataset
from production_rag.core.eval.metrics import percentile
from production_rag.core.eval.run import collection_for
from production_rag.core.indexing import Chunk, index_chunks
from production_rag.core.models import SearchRequest, SearchSpec
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.search import DeadlineExpired, run_search
from production_rag.methods import get_variant

BASE = "hybrid_rrf"
SPECS = [
    SearchSpec(name="dense", k=10, search_type="dense"),
    SearchSpec(name="sparse", k=10, search_type="sparse"),
    SearchSpec(name="hybrid_rerank", k=10, rerank=True),
]
DEADLINES_MS = (2000, 500, 150)
SEQUENTIAL_DEADLINE_MS = 30000
RESULTS = Path("results")


def _share(rows: list[dict[str, Any]], pred) -> float:
    return sum(1 for r in rows if pred(r)) / len(rows)


def _all_ok(row: dict[str, Any]) -> bool:
    return bool(row["status"]) and all(s == "ok" for s in row["status"].values())


def _partial(row: dict[str, Any]) -> bool:
    statuses = set(row["status"].values())
    return "ok" in statuses and statuses != {"ok"}


def summarize_bench(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = list(dict.fromkeys(str(r["deadline"]) for r in rows))
    keys.sort(key=lambda k: (k == "sequential", -int(k) if k.isdigit() else 0))
    summary: dict[str, dict[str, Any]] = {}
    for key in keys:
        mine = [r for r in rows if str(r["deadline"]) == key]
        ms = [r["ms"] for r in mine]
        summary[key] = {
            "queries": len(mine),
            "p50_ms": round(percentile(ms, 50)),
            "p95_ms": round(percentile(ms, 95)),
            "all_ok": _share(mine, _all_ok),
            "partial": _share(mine, _partial),
            "expired": _share(mine, lambda r: r["expired"]),
            "rerank_ok": _share(mine, lambda r: r["rerank"] == "ok"),
            "rerank_skipped": _share(mine, lambda r: r["rerank"] == "skipped_budget"),
            "rerank_failed": _share(mine, lambda r: r["rerank"] == "failed"),
        }
    return summary


def bench_table(summary: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Deadline ms | p50 ms | p95 ms | All specs ok | Partial | Expired"
        " | Rerank ok | Rerank skipped |",
        "|---" * 8 + "|",
    ]
    for key, s in summary.items():
        label = "sequential" if key == "sequential" else key
        cells = [str(s["p50_ms"]), str(s["p95_ms"])]
        cells += [f"{s[m]:.1%}" for m in ["all_ok", "partial", "expired", "rerank_ok"]]
        cells.append(f"{s['rerank_skipped']:.1%}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def _fanout(text: str, collection: str, deps: Deps, deadline_ms: int) -> dict[str, Any]:
    req = SearchRequest(query=text[:2000], collection=collection, specs=SPECS)
    t0 = time.perf_counter()
    try:
        response = await run_search(req, get_variant(BASE), deps, Deadline(deadline_ms))
    except DeadlineExpired:
        status, rerank, expired = {}, "off", True
    else:
        status = {n: r.status for n, r in response.results.items()}
        rerank = response.results["hybrid_rerank"].rerank
        expired = False
    ms = (time.perf_counter() - t0) * 1000
    return {
        "deadline": deadline_ms,
        "ms": round(ms, 1),
        "expired": expired,
        "status": status,
        "rerank": rerank,
    }


async def _sequential(text: str, collection: str, deps: Deps) -> dict[str, Any]:
    status: dict[str, str] = {}
    rerank, expired, total = "off", False, 0.0
    for spec in SPECS:
        req = SearchRequest(query=text[:2000], collection=collection, specs=[spec])
        t0 = time.perf_counter()
        try:
            response = await run_search(
                req, get_variant(BASE), deps, Deadline(SEQUENTIAL_DEADLINE_MS)
            )
        except DeadlineExpired:
            expired = True
        else:
            result = response.results[spec.name]
            status[spec.name] = result.status
            if spec.name == "hybrid_rerank":
                rerank = result.rerank
        total += (time.perf_counter() - t0) * 1000
    return {
        "deadline": "sequential",
        "ms": round(total, 1),
        "expired": expired,
        "status": status,
        "rerank": rerank,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scifact", "tvpl"], required=True)
    parser.add_argument("--query-limit", type=int)
    return parser.parse_args(argv)


async def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    deps = build_deps(settings)
    ds = load_dataset(args.dataset, args.query_limit)
    method = get_variant(BASE)
    collection = collection_for(ds.name, method)
    if method.chunk_chars is None:
        chunks = [Chunk(doc_id, 0, text, title) for doc_id, (title, text) in ds.corpus.items()]
    else:
        chunks = chunk_documents(ds.corpus, method.chunk_chars)
    count = await index_chunks(deps, collection, chunks)
    print(f"{collection}: {count} points, {len(ds.queries)} queries")

    await warm_up(deps)
    rows: list[dict[str, Any]] = []
    for text in ds.queries.values():
        for deadline_ms in DEADLINES_MS:
            rows.append(await _fanout(text, collection, deps, deadline_ms))
        rows.append(await _sequential(text, collection, deps))

    summary = summarize_bench(rows)
    cpu = f"{platform.processor() or platform.machine()}, {os.cpu_count()} cores"
    meta = {
        "dataset": ds.name,
        "date": date.today().isoformat(),
        "queries": len(ds.queries),
        "base": BASE,
        "specs": [s.model_dump(exclude_none=True) for s in SPECS],
        "deadlines_ms": list(DEADLINES_MS),
        "cpu": cpu,
        "one_query_at_a_time": True,
    }
    RESULTS.mkdir(exist_ok=True)
    stem = RESULTS / f"deadline_fanout_{ds.name}"
    stem.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "summary": summary, "per_query": rows}, indent=1)
    )
    header = (
        f"{len(ds.queries)} queries, base `{BASE}`, specs "
        + ", ".join(f"`{s.name}`" for s in SPECS)
        + f", CPU: {cpu}, one query at a time.\n\n"
    )
    stem.with_suffix(".md").write_text(header + bench_table(summary) + "\n")
    print(bench_table(summary))
    print(f"saved {stem}.json and .md")


if __name__ == "__main__":
    asyncio.run(main())
