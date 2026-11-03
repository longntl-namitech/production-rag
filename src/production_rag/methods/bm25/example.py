import asyncio

from production_rag.core.example_data import prepare_example, print_rows
from production_rag.methods.bm25.pipeline import VARIANTS, search

QUERY = "ERR_CONN_RESET"


async def main() -> None:
    deps = await prepare_example()
    for variant in VARIANTS:
        result = await search(QUERY, "example", deps, variant=variant, k=3)
        print_rows(variant, result.rows)


if __name__ == "__main__":
    asyncio.run(main())
