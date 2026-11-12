"""Fact check: extract claims, retrieve evidence for each, judge each, aggregate."""

import asyncio
from dataclasses import dataclass

from production_rag.core.deps import Deps
from production_rag.core.llm import Usage
from production_rag.core.models import Row, SpecResult
from production_rag.core.registry import MethodConfig
from production_rag.methods import all_variants
from production_rag.methods._variant import run_variant
from production_rag.methods.fact_check.claims import MAX_CLAIMS, extract_claims
from production_rag.methods.fact_check.verdict import judge_claim

VARIANTS: dict[str, MethodConfig] = {}
DEFAULT: str | None = None
BASE = "hybrid_weighted_rrf"


async def search(
    query: str,
    collection: str,
    deps: Deps,
    *,
    base: str = BASE,
    k: int = 5,
    deadline_ms: int = 4000,
) -> SpecResult:
    return await run_variant(all_variants(), base, query, collection, deps, k, deadline_ms)


@dataclass(frozen=True)
class ClaimResult:
    claim: str
    query: str
    label: str | None
    rows: list[Row]
    usage: Usage
    error: str | None = None


@dataclass(frozen=True)
class FactCheck:
    label: str | None
    claims: list[ClaimResult]
    usage: Usage
    error: str | None = None


async def check_claim(
    claim: str,
    collection: str,
    deps: Deps,
    llm,
    *,
    query: str | None = None,
    base: str = BASE,
    k: int = 5,
) -> ClaimResult:
    query = query or claim
    found = await search(query, collection, deps, base=base, k=k)
    if found.status != "ok":
        return ClaimResult(claim, query, None, [], Usage(), "retrieval_failed")
    verdict = await judge_claim(llm, claim, found.rows)
    return ClaimResult(claim, query, verdict.label, found.rows, verdict.usage, verdict.error)


def aggregate(labels: list[str]) -> str:
    if not labels:
        return "not_relevant"
    if "refuted" in labels:
        return "refuted"
    return "unverified" if "unverified" in labels else "supported"


async def fact_check(
    statement: str,
    collection: str,
    deps: Deps,
    llm,
    *,
    base: str = BASE,
    k: int = 5,
    max_claims: int = MAX_CLAIMS,
) -> FactCheck:
    extraction = await extract_claims(llm, statement, max_claims)
    if extraction.error:
        return FactCheck(None, [], extraction.usage, "extraction_failed")
    results = await asyncio.gather(
        *(
            check_claim(c.text, collection, deps, llm, query=c.query, base=base, k=k)
            for c in extraction.claims
        )
    )
    usage = extraction.usage
    for r in results:
        usage += r.usage
    label = aggregate([r.label or "unverified" for r in results])
    return FactCheck(label, list(results), usage)
