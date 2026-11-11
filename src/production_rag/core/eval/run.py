"""Run the methods on a public dataset and save the scores.

    docker compose up -d qdrant
    uv run --extra eval --extra local-models python -m production_rag.core.eval.run \
        --dataset scifact --query-limit 20 --name smoke

Searches run in-process through the same pipeline as the methods, one query at
a time by default so latency is not inflated by queueing. No LLM calls.
"""

import argparse
import asyncio
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from production_rag.core.chunking import chunk_documents
from production_rag.core.config import Settings
from production_rag.core.deps import build_deps, warm_up
from production_rag.core.eval.datasets import load_dataset
from production_rag.core.eval.metrics import (
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    rescue_damage,
)
from production_rag.core.indexing import Chunk, index_chunks
from production_rag.core.models import Row, SearchRequest, SearchSpec
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.search import run_search
from production_rag.core.registry import MethodConfig
from production_rag.methods import all_variants, get_variant, resolve

RESULTS = Path("results")
K = 100
RERANK_BASELINE = {"hybrid_rrf_rerank": "hybrid_rrf"}


def collection_for(dataset: str, method: MethodConfig) -> str:
    if method.chunk_chars is None:
        return f"eval-{dataset}"
    return f"eval-{dataset}-c{method.chunk_chars}"


def doc_ranking(rows: list[Row]) -> list[str]:
    """Doc ids in rank order, each once, at its best chunk."""
    return list(dict.fromkeys(r.doc_id for r in rows))


def summarize(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in methods:
        mine = [r for r in rows if r["method"] == method]
        ms = [r["ms"] for r in mine]
        entry = {
            "queries": len(mine),
            "failed": sum(r["status"] != "ok" for r in mine),
            **{
                m: round(sum(r[m] for r in mine) / len(mine), 4)
                for m in ["recall@10", "recall@100", "mrr@10", "ndcg@10"]
            },
            "p50_ms": round(percentile(ms, 50)),
            "p95_ms": round(percentile(ms, 95)),
        }
        base = RERANK_BASELINE.get(method)
        if base in methods:
            before = {r["qid"]: r for r in rows if r["method"] == base}
            # a failed query has no ranking, so it cannot be rescued or damaged
            pairs = [
                rescue_damage(before[r["qid"]]["top"], r["top"], r["rel"])
                for r in mine
                if r["status"] == "ok" and before[r["qid"]]["status"] == "ok"
            ]
            entry["rescue"] = round(sum(p[0] for p in pairs) / len(pairs), 4) if pairs else 0.0
            entry["damage"] = round(sum(p[1] for p in pairs) / len(pairs), 4) if pairs else 0.0
            entry["rerank_skipped_or_failed"] = sum(r["rerank"] != "ok" for r in mine)
        summary[method] = entry
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scifact", "tvpl"], required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(all_variants()),
        help="method folders or variant names (default: every variant)",
    )
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--name")
    args = parser.parse_args(argv)
    try:
        args.methods = resolve(args.methods)
    except ValueError as exc:
        parser.error(str(exc))
    return args


async def main() -> None:
    args = parse_args()

    settings = Settings.from_env()
    deps = build_deps(settings)
    ds = load_dataset(args.dataset, args.query_limit)
    methods = {name: get_variant(name) for name in args.methods}
    collections = {name: collection_for(ds.name, m) for name, m in methods.items()}
    chunk_counts: dict[str, int] = {}
    for coll, size in {collections[n]: m.chunk_chars for n, m in methods.items()}.items():
        started = time.perf_counter()
        if size is None:
            chunks = [Chunk(doc_id, 0, text, title) for doc_id, (title, text) in ds.corpus.items()]
        else:
            chunks = chunk_documents(ds.corpus, size)
        count = await index_chunks(deps, coll, chunks)
        took = time.perf_counter() - started
        if size is not None:
            chunk_counts[coll] = count
        print(f"{coll}: {count} {'docs' if size is None else 'chunks'} ({took:.0f}s)")
    print(f"{len(ds.corpus)} documents, {len(ds.queries)} queries")

    await warm_up(deps)
    semaphore = asyncio.Semaphore(args.concurrency)
    rows: list[dict[str, Any]] = []

    async def one(method: str, qid: str, text: str) -> None:
        req = SearchRequest(
            query=text[:2000],
            collection=collections[method],
            specs=[SearchSpec(name="s", k=K)],
        )
        async with semaphore:
            t0 = time.perf_counter()
            result = (await run_search(req, methods[method], deps, Deadline(30000))).results["s"]
            ms = (time.perf_counter() - t0) * 1000
        ranked = doc_ranking(result.rows)
        rel = ds.qrels[qid]
        rows.append(
            {
                "method": method,
                "qid": qid,
                "status": result.status,
                "rerank": result.rerank,
                "ms": round(ms, 1),
                "recall@10": recall_at_k(ranked, rel, 10),
                "recall@100": recall_at_k(ranked, rel, 100),
                "mrr@10": mrr_at_k(ranked, rel, 10),
                "ndcg@10": ndcg_at_k(ranked, rel, 10),
                "top": ranked[:1],
                "rel": rel,
            }
        )

    for method in args.methods:
        t0 = time.perf_counter()
        await asyncio.gather(*(one(method, qid, q) for qid, q in ds.queries.items()))
        print(f"{method:22} done in {time.perf_counter() - t0:.0f}s", flush=True)

    summary = summarize(rows, args.methods)
    meta = {
        "dataset": ds.name,
        "date": date.today().isoformat(),
        "queries": len(ds.queries),
        "documents": len(ds.corpus),
        "dense_model": settings.dense_model,
        "cross_encoder": settings.cross_encoder_model,
        "sparse_model": "Qdrant/bm25",
        "k": K,
    }
    if chunk_counts:
        meta["chunks"] = chunk_counts
    for row in rows:
        row.pop("rel")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.name or ds.name}.json"
    out.write_text(json.dumps({"meta": meta, "summary": summary, "per_query": rows}, indent=1))
    from production_rag.core.eval.report import failures_note, table

    print(table(summary))
    print(failures_note(summary))
    print(f"saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
