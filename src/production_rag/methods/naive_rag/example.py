"""Naive RAG: retrieve 800-character chunks, then answer from them with an LLM."""

import asyncio

from production_rag.core.example_data import prepare_example, print_rows
from production_rag.core.llm import MissingAPIKey, OpenAIChat, TokenBudget
from production_rag.methods.naive_rag.pipeline import answer, search

QUESTION = "why does sourdough rise"
TOKEN_CAP = 5_000


async def main() -> None:
    deps = await prepare_example()
    result = await search(QUESTION, "example", deps, k=3)
    print_rows("naive_rag", result.rows)
    try:
        llm = OpenAIChat.from_env(budget=TokenBudget(TOKEN_CAP))
    except MissingAPIKey:
        print("set OPENAI_API_KEY to run the generation step")
        return
    out = await answer(QUESTION, "example", deps, llm, k=3)
    print(f"answer: {out.text}")
    print(f"tokens: {out.usage.total}")


if __name__ == "__main__":
    asyncio.run(main())
