import asyncio
import os

from production_rag.core.example_data import prepare_example, print_rows
from production_rag.core.llm import OpenAIChat, TokenBudget
from production_rag.methods.hyde.generate import queries_for
from production_rag.methods.hyde.pipeline import search

QUESTION = "how do solar cells make power"


async def main() -> None:
    deps = await prepare_example()
    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY to write the hypothetical passage; searching the question")
        print_rows("question", (await search([QUESTION], "example", deps, k=4)).rows)
        return
    llm = OpenAIChat.from_env(budget=TokenBudget(10_000))
    queries, usage = await queries_for(llm, QUESTION)
    print(f"searching with: {queries[0]}")
    print_rows("hyde", (await search(queries, "example", deps, k=4)).rows)
    print(f"tokens: {usage.total}")


if __name__ == "__main__":
    asyncio.run(main())
