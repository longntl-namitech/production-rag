import asyncio

from production_rag.core.example_data import prepare_example, print_rows
from production_rag.methods.context_expansion.pipeline import search


async def main() -> None:
    deps = await prepare_example()
    result = await search("tea ceremony matcha", "example", deps, k=1)
    print_rows("context_expansion", result.rows)
    print(result.rows[0].text)


if __name__ == "__main__":
    asyncio.run(main())
