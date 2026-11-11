"""Rerank a candidate pool, but never at the cost of the request.

Skipped when the budget is nearly spent; on any error or timeout the
retrieval order is kept and the status says so.
"""

import asyncio
import time

from production_rag.core.models import Row
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.rerankers.base import Reranker

SAFETY_MS = 50


def _rerank_text(row: Row) -> str:
    # same text the dense and sparse legs were indexed with
    return f"{row.title}\n{row.text}" if row.title else row.text


async def apply_rerank(
    reranker: Reranker | None,
    query: str,
    rows: list[Row],
    deadline: Deadline,
    min_budget_ms: int,
) -> tuple[list[Row], str, int | None]:
    if reranker is None:
        return rows, "off", None
    if len(rows) <= 1:
        return rows, "skipped_trivial", None
    remaining = deadline.remaining_ms()
    if remaining < max(min_budget_ms, SAFETY_MS + 1):
        return rows, "skipped_budget", None
    start = time.perf_counter()
    try:
        scores = await asyncio.wait_for(
            reranker.score(query, [_rerank_text(r) for r in rows]), (remaining - SAFETY_MS) / 1000
        )
        rescored = [r.model_copy(update={"score": s}) for r, s in zip(rows, scores, strict=True)]
    except ImportError:  # a missing extra is a setup error, not a slow reranker
        raise
    except Exception:  # fall back to retrieval order, also on a wrong score count
        return rows, "failed", round((time.perf_counter() - start) * 1000)
    rescored.sort(key=lambda r: r.score, reverse=True)
    return rescored, "ok", round((time.perf_counter() - start) * 1000)
