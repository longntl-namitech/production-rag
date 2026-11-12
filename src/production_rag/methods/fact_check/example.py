"""Fact check: split a statement into claims, find evidence for each, judge each."""

import asyncio
import os

from production_rag.core.example_data import prepare_example, print_rows
from production_rag.core.llm import OpenAIChat, TokenBudget
from production_rag.methods.fact_check.pipeline import fact_check, search

STATEMENT = (
    "Sourdough rises because wild yeast ferments the dough, and tea reached Japan from Korea."
)
CLAIMS = [
    "Sourdough rises because wild yeast ferments the dough",
    "Tea reached Japan from Korea",
]


async def main() -> None:
    deps = await prepare_example()
    if not os.environ.get("OPENAI_API_KEY"):
        for claim in CLAIMS:
            print_rows(claim, (await search(claim, "example", deps, k=3)).rows)
        print("set OPENAI_API_KEY to run claim extraction and the verdicts")
        return
    llm = OpenAIChat.from_env(budget=TokenBudget(10_000))
    out = await fact_check(STATEMENT, "example", deps, llm, base="hybrid_weighted_rrf", k=3)
    for c in out.claims:
        docs = ", ".join(r.doc_id for r in c.rows)
        print(f"{c.label or c.error:<11} {c.claim}  [{docs}]")
    print(f"statement: {out.label or out.error}, tokens: {out.usage.total}")


if __name__ == "__main__":
    asyncio.run(main())
