"""A tiny corpus for the runnable method examples. Text is written for this repo."""

from production_rag.core.config import Settings
from production_rag.core.deps import Deps, build_deps, warm_up
from production_rag.core.indexing import Chunk, index_chunks
from production_rag.core.models import Row

EXAMPLE_CHUNKS: list[Chunk] = [
    Chunk(
        "solar", 0, "Solar panels turn sunlight into electricity using photovoltaic cells.", "Solar"
    ),
    Chunk(
        "bread", 0, "Sourdough bread rises slowly because wild yeast ferments the dough.", "Bread"
    ),
    Chunk("tea", 0, "Tea reached Japan from China around the ninth century.", "History of tea"),
    Chunk("tea", 1, "The Japanese tea ceremony is a ritual of preparing and serving matcha.", None),
    Chunk("tea", 2, "Today most Japanese tea is grown in Shizuoka and Kagoshima.", None),
    Chunk(
        "errors",
        0,
        "Error ERR_CONN_RESET means the server closed the connection before replying.",
        "Network errors",
    ),
    Chunk("tides", 0, "Tides are caused by the gravity of the moon and the sun.", "Tides"),
    Chunk("bikes", 0, "A derailleur moves the chain between gears on a bicycle.", "Bicycles"),
]


async def prepare_example(collection: str = "example") -> Deps:
    """Build the models, warm them up and index `EXAMPLE_CHUNKS`."""
    deps = build_deps(Settings.from_env())
    await warm_up(deps)
    await index_chunks(deps, collection, EXAMPLE_CHUNKS)
    return deps


def print_rows(label: str, rows: list[Row]) -> None:
    print(f"== {label}")
    for row in rows:
        print(f"{row.doc_id:<8} {row.score:.4f}  {row.text[:60]!r}")
