"""Parse files into Markdown, chunk them, and index them: recursive chunking variants."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from production_rag.core.chunking import Strategy, chunk_documents
from production_rag.core.deps import Deps
from production_rag.core.indexing import index_chunks
from production_rag.core.models import SpecResult
from production_rag.core.registry import MethodConfig
from production_rag.core.semantic_chunking import semantic_chunk_documents
from production_rag.methods._variant import run_variant
from production_rag.methods.parse_and_chunk.parsers import (
    MissingExtra,
    UnsupportedFormat,
    parse_file,
)


def _variant(size: int, overlap: int, doc: str) -> MethodConfig:
    name = f"chunk_recursive_{size}" + (f"_o{overlap}" if overlap else "")
    return MethodConfig(
        name,
        ("dense",),
        "none",
        doc=doc,
        chunk_chars=size,
        chunker="recursive",
        chunk_overlap=overlap,
    )


VARIANTS: dict[str, MethodConfig] = {
    v.name: v
    for v in [
        _variant(800, 0, "recursive chunks of up to 800 characters, dense top-k"),
        _variant(800, 200, "recursive 800-character chunks with 200 characters of overlap"),
        _variant(400, 0, "recursive chunks of up to 400 characters, dense top-k"),
        MethodConfig(
            "chunk_semantic",
            ("dense",),
            "none",
            doc="semantic chunks of up to 800 characters, cut where the topic changes, dense top-k",
            chunk_chars=800,
            chunker="semantic",
        ),
    ]
}

DEFAULT = "chunk_recursive_800"

_HEADING = re.compile(r"^#{1,6} +(.*\S)", re.MULTILINE)


@dataclass(frozen=True)
class IndexReport:
    files: int
    chunks: int
    skipped: dict[str, str]
    failed: dict[str, str]


async def search(
    query: str,
    collection: str,
    deps: Deps,
    *,
    variant: str = DEFAULT,
    k: int = 5,
    deadline_ms: int = 4000,
) -> SpecResult:
    return await run_variant(VARIANTS, variant, query, collection, deps, k, deadline_ms)


async def index_files(
    deps: Deps,
    collection: str,
    paths: list[Path],
    *,
    strategy: Strategy | Literal["semantic"] = "markdown",
    size: int = 800,
    overlap: int = 0,
) -> IndexReport:
    """Parse the files and index their chunks. Unreadable files are reported, not raised.

    A missing parser library (`MissingExtra`) is a setup error and propagates. The "semantic"
    strategy embeds sentences with the index's embedder and ignores `overlap`.
    """
    docs: dict[str, tuple[str | None, str]] = {}
    skipped: dict[str, str] = {}
    failed: dict[str, str] = {}
    for path in paths:
        key = str(path)
        try:
            parsed = parse_file(path)
        except UnsupportedFormat as exc:
            skipped[key] = str(exc)
            continue
        except MissingExtra:
            raise
        except Exception as exc:
            failed[key] = type(exc).__name__
            continue
        if "no_text" in parsed.warnings:
            failed[key] = "no_text"
            continue
        heading = _HEADING.search(parsed.markdown)
        docs[key] = (heading.group(1).strip() if heading else path.name, parsed.markdown)
    if strategy == "semantic":
        chunks = await semantic_chunk_documents(docs, deps.embedder, max_chars=size)
    else:
        chunks = chunk_documents(docs, size, strategy=strategy, overlap=overlap)
    count = await index_chunks(deps, collection, chunks)
    return IndexReport(len(docs), count, skipped, failed)
