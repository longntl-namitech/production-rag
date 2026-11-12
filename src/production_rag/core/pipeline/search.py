"""The search pipeline shared by the methods and the eval harness.

1. Embed every subquery once, only for the vector kinds some spec needs.
2. Run each spec as its own task under the request deadline. A spec that
   raises or runs out of time becomes a failed row; the others still return.
3. Per spec: search each subquery, merge, optionally rerank a wider pool,
   optionally expand context.
"""

import asyncio
import time

from production_rag.core.deps import Deps
from production_rag.core.fusion import QueryVectors, build_query
from production_rag.core.models import SearchRequest, SearchResponse, SearchSpec, SpecResult, Timing
from production_rag.core.pipeline.context import expand_context
from production_rag.core.pipeline.deadline import Deadline
from production_rag.core.pipeline.merge import merge_max
from production_rag.core.pipeline.rerank_stage import SAFETY_MS, apply_rerank
from production_rag.core.registry import Leg, MethodConfig, legs_for
from production_rag.core.store import build_filter


class DeadlineExpired(Exception):
    """The budget was spent before any spec could run."""


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


async def _embed(subqueries: list[str], legs: set[Leg], deps: Deps) -> list[QueryVectors]:
    async def none() -> None:
        return None

    dense, sparse, sparse_vi = await asyncio.gather(
        deps.embedder.embed_queries(subqueries) if "dense" in legs else none(),
        asyncio.to_thread(deps.bm25.encode_queries, subqueries)
        if legs & {"sparse", "sparse_noidf"}
        else none(),
        asyncio.to_thread(deps.bm25_vi.encode_queries, subqueries)
        if "sparse_vi" in legs
        else none(),
    )
    return [
        QueryVectors(
            dense=dense[i] if dense else None,
            sparse=sparse[i] if sparse else None,
            sparse_vi=sparse_vi[i] if sparse_vi else None,
        )
        for i in range(len(subqueries))
    ]


async def _run_spec(
    req: SearchRequest,
    spec: SearchSpec,
    method: MethodConfig,
    vectors: list[QueryVectors],
    deps: Deps,
    deadline: Deadline,
) -> SpecResult:
    start = time.perf_counter()
    settings = deps.settings
    rerank = spec.rerank if spec.rerank is not None else method.rerank
    rerank = rerank and deps.reranker is not None
    legs = legs_for(method, spec.search_type)
    fusion = method.fusion if len(legs) > 1 else "none"
    flt = build_filter(spec.filter)
    limit = (
        min(spec.k * settings.rerank_pool_multiplier, settings.rerank_pool_max)
        if rerank
        else spec.k
    )
    # A score threshold is on the retrieval scale, which reranking replaces.
    threshold = None if rerank else spec.threshold
    prefetch_limit = limit * settings.prefetch_multiplier
    per_query = await asyncio.gather(
        *(
            deps.store.search(
                req.collection,
                build_query(legs, fusion, method.weights, v, prefetch_limit, flt),
                limit,
                threshold,
            )
            for v in vectors
        )
    )
    rows = merge_max(list(per_query), limit)
    status, rerank_ms = "off", None
    if rerank:
        rows, status, rerank_ms = await apply_rerank(
            deps.reranker,
            req.rerank_query or req.subqueries[0],
            rows,
            deadline,
            settings.rerank_min_budget_ms,
        )
    rows = rows[: spec.k]
    if spec.context:
        # Optional like rerank: if the budget runs out, return the rows unexpanded.
        try:
            rows = await asyncio.wait_for(
                expand_context(deps.store, req.collection, rows, spec.context),
                max(0, deadline.remaining_ms() - SAFETY_MS) / 1000,
            )
        except TimeoutError:
            pass
    return SpecResult(
        status="ok", rows=rows, elapsed_ms=_ms(start), rerank=status, rerank_ms=rerank_ms
    )


async def _guarded(
    req: SearchRequest,
    spec: SearchSpec,
    method: MethodConfig,
    vectors: list[QueryVectors],
    deps: Deps,
    deadline: Deadline,
) -> SpecResult:
    start = time.perf_counter()
    try:
        return await asyncio.wait_for(
            _run_spec(req, spec, method, vectors, deps, deadline), deadline.remaining_s()
        )
    except Exception as exc:  # isolate: one failing spec must not fail the request
        return SpecResult(status="failed", elapsed_ms=_ms(start), error=type(exc).__name__)


async def run_search(
    req: SearchRequest, method: MethodConfig, deps: Deps, deadline: Deadline
) -> SearchResponse:
    start = time.perf_counter()
    needed: set[Leg] = {leg for spec in req.specs for leg in legs_for(method, spec.search_type)}
    try:
        vectors = await asyncio.wait_for(
            _embed(req.subqueries, needed, deps), deadline.remaining_s()
        )
    except TimeoutError:
        raise DeadlineExpired from None
    embed_ms = _ms(start)
    if deadline.expired():
        raise DeadlineExpired
    results = await asyncio.gather(
        *(_guarded(req, spec, method, vectors, deps, deadline) for spec in req.specs)
    )
    return SearchResponse(
        method=method.name,
        results={spec.name: result for spec, result in zip(req.specs, results, strict=True)},
        timing=Timing(embed_ms=embed_ms, total_ms=_ms(start)),
    )
