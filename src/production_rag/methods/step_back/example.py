import asyncio
import os

from production_rag.core.example_data import prepare_example, print_rows
from production_rag.core.llm import OpenAIChat, TokenBudget
from production_rag.methods.step_back.generate import queries_for
from production_rag.methods.step_back.pipeline import search

QUESTION = "why does sourdough rise faster in summer"


async def main() -> None:
    deps = await prepare_example()
    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY to write the step-back question; searching the question")
        print_rows("question", (await search([QUESTION], "example", deps, k=4)).rows)
        return
    llm = OpenAIChat.from_env(budget=TokenBudget(10_000))
    queries, usage = await queries_for(llm, QUESTION)
    print(f"searching with: {queries}")
    print_rows("step_back", (await search(queries, "example", deps, k=4)).rows)
    print(f"tokens: {usage.total}")


if __name__ == "__main__":
    asyncio.run(main())
