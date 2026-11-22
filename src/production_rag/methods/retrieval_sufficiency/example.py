import asyncio
import os

from production_rag.core.example_data import prepare_example, print_rows
from production_rag.core.llm import OpenAIChat, TokenBudget
from production_rag.methods.retrieval_sufficiency.pipeline import search

QUESTIONS = ["where is japanese tea grown today", "who won the 1998 football world cup"]


async def main() -> None:
    deps = await prepare_example()
    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY to judge the passages")
        return
    llm = OpenAIChat.from_env(budget=TokenBudget(10_000))
    for question in QUESTIONS:
        got = await search(question, "example", deps, llm, k=3)
        print(f"\n{question}\nlabel: {got.label}")
        print_rows("passages", got.rows)
        print(f"tokens: {got.usage.total}")


if __name__ == "__main__":
    asyncio.run(main())
