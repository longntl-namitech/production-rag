from collections import Counter

from production_rag.core.example_data import EXAMPLE_CHUNKS


def test_example_chunks_unique_and_one_multichunk_doc():
    keys = [(c.doc_id, c.chunk_index) for c in EXAMPLE_CHUNKS]
    assert len(keys) == len(set(keys))
    by_doc = Counter(c.doc_id for c in EXAMPLE_CHUNKS)
    doc, n = by_doc.most_common(1)[0]
    assert n >= 3
    assert sorted(c.chunk_index for c in EXAMPLE_CHUNKS if c.doc_id == doc) == list(range(n))
