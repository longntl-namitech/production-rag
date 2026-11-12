"""Write the markdown results table: python -m production_rag.core.eval.report results/x.json"""

import json
import sys
from pathlib import Path
from typing import Any

METRICS = ["recall@10", "recall@100", "mrr@10", "ndcg@10"]


def table(summary: dict[str, Any]) -> str:
    lines = [
        "| Method | " + " | ".join(METRICS) + " | p50 ms | p95 ms | Rescue | Damage |",
        "|---" * (len(METRICS) + 5) + "|",
    ]
    for method, s in summary.items():
        cells = [f"{s[m]:.3f}" for m in METRICS] + [str(s["p50_ms"]), str(s["p95_ms"])]
        cells += [f"{s[m]:.1%}" if m in s else "" for m in ["rescue", "damage"]]
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def chunk_table(stats: dict[str, Any]) -> str:
    lines = [
        "| Collection | Chunker | Size | Overlap | Chunks | Mean chars |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in stats.items():
        lines.append(
            f"| `{name}` | {s['chunker']} | {s['size']} | {s['overlap']} "
            f"| {s['chunks']} | {s['mean_chars']} |"
        )
    return "\n".join(lines)


ANSWER_COLUMNS = [
    "Correctness",
    "Faithfulness",
    "Abstained",
    "Retrieval p50 ms",
    "Rewrite p50 ms",
    "Generation p50 ms",
    "RAG tokens / question",
    "Passage tokens (est.)",
    "Failed",
    "Judge failed",
]


def _cell(value: Any, fmt: str) -> str:
    return "" if value is None else format(value, fmt)


def answer_table(summary: dict[str, Any]) -> str:
    """Blank cells: not applicable (rewrite latency outside multi_query) or no labelled item."""
    lines = [
        "| Method | " + " | ".join(ANSWER_COLUMNS) + " |",
        "|---" * (len(ANSWER_COLUMNS) + 1) + "|",
    ]
    for method, s in summary.items():
        cells = [
            _cell(s["correctness"], ".3f"),
            _cell(s["faithfulness"], ".1%"),
            _cell(s["abstained"], ".1%"),
            _cell(s["retrieval_p50_ms"], "d"),
            _cell(s.get("rewrite_p50_ms"), "d"),
            _cell(s["generation_p50_ms"], "d"),
            _cell(s["rag_tokens"], "d"),
            _cell(s["passage_tokens"], "d"),
            str(s["failed"]),
            str(s["judge_failed"]),
        ]
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def failures_note(summary: dict[str, Any]) -> str:
    failed = [f"`{m}` {s['failed']}" for m, s in summary.items() if s["failed"]]
    return f"Failed queries: {', '.join(failed)}." if failed else "No failed queries."


def _chunk_count(chunks: dict[str, int]) -> str:
    if len(chunks) == 1:
        return str(next(iter(chunks.values())))
    return ", ".join(f"{name}: {n}" for name, n in chunks.items())


def _answer_report(run: dict[str, Any]) -> str:
    meta = run["meta"]
    header = (
        f"# {meta['dataset']}\n\n"
        f"{meta['questions']} sampled questions, {meta['documents']} documents in "
        f"{_chunk_count(meta['chunks'])} chunks of up to 800 characters, run {meta['date']}. "
        f"Generator and judge `{meta['model']}`, top {meta['k']} passages. "
        "Judged by the same model that answers.\n\n"
    )
    if not meta["complete"]:
        header += "Incomplete run: stopped at the token budget.\n\n"
    return header + answer_table(run["summary"]) + "\n"


def main(path: str) -> None:
    run = json.loads(Path(path).read_text())
    meta = run["meta"]
    out = Path(path).with_suffix(".md")
    if meta.get("kind") == "answer":
        out.write_text(_answer_report(run))
        print(f"wrote {out}")
        return
    header = (
        f"# {meta['dataset']}\n\n"
        f"{meta['queries']} queries, {meta['documents']} documents, run {meta['date']}. "
        f"Dense `{meta['dense_model']}`, sparse `{meta['sparse_model']}`, "
        f"rerank `{meta['cross_encoder']}` over the top {meta['k']}. "
        "Latency is in-process on CPU, one query at a time.\n\n"
    )
    if meta.get("chunk_stats"):
        header += (
            "Chunked variants search their own collections; a document counts once, at its "
            "best chunk, so recall@100 counts the documents in the top 100 chunks.\n\n"
            + chunk_table(meta["chunk_stats"])
            + "\n\n"
        )
    elif meta.get("chunks"):
        header += (
            f"`naive_rag` searches {_chunk_count(meta['chunks'])} chunks of up to 800 characters; "
            "a document counts once, at its best chunk, so recall@100 counts the documents "
            "in the top 100 chunks.\n\n"
        )
    out.write_text(header + table(run["summary"]) + "\n\n" + failures_note(run["summary"]) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
