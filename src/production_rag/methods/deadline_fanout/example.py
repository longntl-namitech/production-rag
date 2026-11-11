import asyncio

from production_rag.core.example_data import prepare_example, print_rows
from production_rag.core.models import SearchSpec
from production_rag.methods.deadline_fanout.pipeline import search


async def main() -> None:
    deps = await prepare_example()
    specs = [SearchSpec(name="top1", k=1), SearchSpec(name="top3", k=3, search_type="dense")]
    response = await search("how do tides work", "example", deps, specs, deadline_ms=4000)
    for name, result in response.results.items():
        print_rows(f"{name} ({result.status})", result.rows)


if __name__ == "__main__":
    asyncio.run(main())
