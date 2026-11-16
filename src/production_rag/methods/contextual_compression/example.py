import asyncio
import os

from production_rag.core.example_data import prepare_example, print_rows
from production_rag.core.llm import OpenAIChat, TokenBudget
from production_rag.methods.contextual_compression.pipeline import search
from production_rag.methods.naive_rag import pipeline as naive_rag

QUESTION = "where is japanese tea grown today"


async def main() -> None:
    deps = await prepare_example()
    print_rows("before", (await naive_rag.search(QUESTION, "example", deps, k=3)).rows)
    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY to compress the passages")
        return
    llm = OpenAIChat.from_env(budget=TokenBudget(10_000))
    result, usage = await search(QUESTION, "example", deps, llm, k=3)
    print_rows("after", result.rows)
    print(f"tokens: {usage.total}")


if __name__ == "__main__":
    asyncio.run(main())
