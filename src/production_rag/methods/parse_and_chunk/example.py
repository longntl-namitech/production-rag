"""Parse and chunk: turn files into Markdown, compare chunkers, index and search. No LLM."""

import asyncio
import sys
from pathlib import Path

from production_rag.core.chunking import chunk_documents
from production_rag.core.config import Settings
from production_rag.core.deps import build_deps, warm_up
from production_rag.methods.parse_and_chunk.parsers import MissingExtra, parse_file
from production_rag.methods.parse_and_chunk.pipeline import index_files, search

SAMPLES = Path(__file__).parent / "samples"
COLLECTION = "example-parse"
QUESTION = "how many days of annual leave"
SIZE = 400


async def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or [SAMPLES / "handbook.md", SAMPLES / "handbook.html"]
    usable: list[Path] = []
    for path in paths:
        try:
            parsed = parse_file(path)
        except MissingExtra:
            print("install the parse extra to parse HTML, PDF and Office files")
            continue
        except Exception as exc:
            print(f"{path.name}: skipped ({exc})")
            continue
        usable.append(path)
        print(f"== {path.name} ({parsed.format})")
        print(parsed.markdown[:300])
        for strategy in ("fixed", "recursive", "markdown"):
            chunks = chunk_documents({"doc": (None, parsed.markdown)}, SIZE, strategy=strategy)
            print(f"{strategy}: {len(chunks)} chunks of up to {SIZE} characters")

    if not usable:
        print("no file could be parsed, nothing to index")
        return
    deps = build_deps(Settings.from_env())
    await warm_up(deps)
    # index_chunks adds to an existing collection, so start clean to leave no stale chunks
    await deps.store.client.delete_collection(COLLECTION)
    report = await index_files(deps, COLLECTION, usable, strategy="markdown", size=SIZE)
    print(f"indexed {report.files} files, {report.chunks} chunks")
    result = await search(QUESTION, COLLECTION, deps, k=3)
    print(f"== {QUESTION}")
    for row in result.rows:
        print(f"{Path(row.doc_id).name:<14} {row.score:.4f}  {row.text[:60]!r}")
    for label, problems in (("skipped", report.skipped), ("failed", report.failed)):
        for key, reason in problems.items():
            print(f"{label}: {Path(key).name}: {reason}")


if __name__ == "__main__":
    asyncio.run(main())
