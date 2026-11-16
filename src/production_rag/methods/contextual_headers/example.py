import asyncio
import os

from production_rag.core.cache import DiskCache
from production_rag.core.config import Settings
from production_rag.core.deps import build_deps, warm_up
from production_rag.core.example_data import print_rows
from production_rag.core.llm import OpenAIChat, TokenBudget
from production_rag.methods.contextual_headers.pipeline import index, search

DOCS = {
    "tea": (
        "History of tea",
        "Tea reached Japan from China around the ninth century. "
        "The Japanese tea ceremony is a ritual of preparing and serving matcha. "
        "Today most Japanese tea is grown in Shizuoka and Kagoshima.",
    ),
}
QUERY = "where is matcha grown now"


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY to write the context sentences")
        return
    deps = build_deps(Settings.from_env())
    await warm_up(deps)
    llm = OpenAIChat.from_env(budget=TokenBudget(10_000))
    done = await index(deps, llm, DiskCache(), "example-contextual", DOCS, size=80)
    print(f"indexed {done.count} chunks, tokens: {done.usage.total} (a rerun uses .cache/llm)")
    print_rows("contextual_headers", (await search(QUERY, "example-contextual", deps, k=3)).rows)


if __name__ == "__main__":
    asyncio.run(main())
