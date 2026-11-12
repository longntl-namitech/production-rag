import re
from typing import Literal

from production_rag.core.indexing import Chunk

Strategy = Literal["fixed", "recursive", "markdown"]
SEPARATORS = ("\n\n", "\n", ". ", " ")
MIN_BODY = 100
_HEADING = re.compile(r"^(#{1,6}) +(.*\S)")
_FENCE = re.compile(r"^\s*(```|~~~)")


def fixed_size_chunks(text: str, size: int) -> list[str]:
    """Pack whitespace-split words into chunks of at most `size` characters."""
    if size <= 0:
        raise ValueError("size must be positive")
    chunks: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(word[:size])
            word = word[size:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= size:
            current = f"{current} {word}"
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def _pieces(text: str, sep: str) -> list[str]:
    if sep == ". ":
        parts = text.split(sep)
        parts = [p + "." for p in parts[:-1]] + parts[-1:]
    else:
        parts = text.split(sep)
    return [p.strip() for p in parts if p.strip()]


def _pack(units: list[str], size: int, join: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + len(join) + len(unit) <= size:
            current = f"{current}{join}{unit}"
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


def _split(text: str, size: int, level: int) -> list[str]:
    text = text.strip()
    while level < len(SEPARATORS) and SEPARATORS[level] not in text:
        level += 1
    if level == len(SEPARATORS):
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep = SEPARATORS[level]
    units: list[str] = []
    for piece in _pieces(text, sep):
        if len(piece) <= size:
            units.append(piece)
        else:
            units.extend(_split(piece, size, level + 1))
    return _pack(units, size, sep if not sep.strip() else " ")


def _tail(prev: str, limit: int) -> str:
    for k in range(min(limit, len(prev)), 0, -1):
        if (len(prev) == k or prev[-k - 1].isspace()) and not prev[-k].isspace():
            return prev[-k:]
    return ""


def recursive_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split on paragraphs, lines, sentences, then words; pack pieces up to `size`.

    With `overlap`, each chunk after the first starts with a word-aligned tail of the
    previous chunk (at most `overlap` characters); the chunk stays within `size`.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    body_size = max(1, size - overlap - 1) if overlap else size
    bodies = _split(text, body_size, 0) if text.strip() else []
    if not overlap:
        return bodies
    out = bodies[:1]
    for prev, cur in zip(bodies, bodies[1:], strict=False):
        tail = _tail(prev, min(overlap, size - len(cur) - 1))
        out.append(f"{tail} {cur}" if tail else cur)
    return out


def markdown_section_chunks(markdown: str, size: int, overlap: int = 0) -> list[str]:
    """Chunk each ATX-heading section, prefixing every chunk with its heading path."""
    if size <= 0:
        raise ValueError("size must be positive")
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    stack: list[tuple[int, str]] = []
    fenced = False
    for line in markdown.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        heading = None if fenced else _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
            sections.append(([title for _, title in stack], []))
        else:
            sections[-1][1].append(line)
    out: list[str] = []
    for path, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        if not path:
            out.extend(recursive_chunks(body, size, overlap))
            continue
        prefix = " > ".join(path) + "\n"
        if size - len(prefix) < MIN_BODY:
            prefix = path[-1] + "\n"
            if size - len(prefix) < MIN_BODY:
                raise ValueError(f"size {size} leaves under {MIN_BODY} characters after heading")
        out.extend(prefix + c for c in recursive_chunks(body, size - len(prefix), overlap))
    return out


def chunk_documents(
    docs: dict[str, tuple[str | None, str]],
    size: int,
    *,
    strategy: Strategy = "fixed",
    overlap: int = 0,
) -> list[Chunk]:
    if strategy == "fixed":
        if overlap:
            raise ValueError("overlap is not supported by the fixed strategy")
        split = fixed_size_chunks
        args: tuple[int, ...] = (size,)
    elif strategy == "recursive":
        split, args = recursive_chunks, (size, overlap)
    else:
        split, args = markdown_section_chunks, (size, overlap)
    return [
        Chunk(doc_id, i, piece, title)
        for doc_id, (title, text) in docs.items()
        for i, piece in enumerate(split(text, *args))
    ]
