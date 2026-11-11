from production_rag.core.indexing import Chunk


def fixed_size_chunks(text: str, size: int) -> list[str]:
    """Pack whitespace-split words into chunks of at most `size` characters."""
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


def chunk_documents(docs: dict[str, tuple[str | None, str]], size: int) -> list[Chunk]:
    return [
        Chunk(doc_id, i, piece, title)
        for doc_id, (title, text) in docs.items()
        for i, piece in enumerate(fixed_size_chunks(text, size))
    ]
