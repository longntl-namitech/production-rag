import asyncio

from production_rag.core.example_data import prepare_example, print_rows
from production_rag.methods.multi_query.pipeline import search

QUERIES = ["how do solar cells make power", "why does sourdough rise"]


async def main() -> None:
    deps = await prepare_example()
    result = await search(QUERIES, "example", deps, k=4)
    print_rows("multi_query", result.rows)


if __name__ == "__main__":
    asyncio.run(main())
