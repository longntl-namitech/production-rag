"""Public eval datasets, downloaded from Hugging Face at runtime (never committed).

- scifact: BEIR SciFact. `BeIR/scifact` configs `corpus` / `queries` (columns
  `_id`, `title`, `text`) and `BeIR/scifact-qrels` split `test` (`query-id`,
  `corpus-id`, `score`).
- tvpl: Vietnamese legal questions, `GreenNode/TVPL-Retrieval-VN` (CC-BY-SA-4.0),
  configs `corpus` (`id`, `title`, `text`), `queries` (`id`, `text`), `qrels`,
  all split `test`. Scores are strings ("1", "1.0").
"""

import random
from dataclasses import dataclass
from typing import Literal

DEFAULT_QUERY_LIMIT = {"scifact": None, "tvpl": 1000}


@dataclass
class Dataset:
    name: str
    corpus: dict[str, tuple[str | None, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def subsample(ids: list[str], n: int, seed: int) -> list[str]:
    ordered = sorted(ids)
    if n >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, n)


def _qrels(rows) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for row in rows:
        score = int(float(row["score"]))
        if score > 0:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = score
    return qrels


def load_dataset(
    name: Literal["scifact", "tvpl"], query_limit: int | None = None, seed: int = 13
) -> Dataset:
    from datasets import load_dataset as hf_load

    if name == "scifact":
        corpus_rows = hf_load("BeIR/scifact", "corpus", split="corpus")
        query_rows = hf_load("BeIR/scifact", "queries", split="queries")
        qrels = _qrels(hf_load("BeIR/scifact-qrels", split="test"))
        id_key = "_id"
    elif name == "tvpl":
        repo = "GreenNode/TVPL-Retrieval-VN"
        corpus_rows = hf_load(repo, "corpus", split="test")
        query_rows = hf_load(repo, "queries", split="test")
        qrels = _qrels(hf_load(repo, "qrels", split="test"))
        id_key = "id"
    else:
        raise ValueError(f"unknown dataset {name!r}")

    corpus = {
        str(r[id_key]): (r["title"] or None, r["text"].strip().strip('"').strip())
        for r in corpus_rows
    }
    queries = {str(r[id_key]): r["text"] for r in query_rows if str(r[id_key]) in qrels}
    limit = query_limit if query_limit is not None else DEFAULT_QUERY_LIMIT[name]
    if limit is not None:
        queries = {qid: queries[qid] for qid in subsample(list(queries), limit, seed)}
    return Dataset(name, corpus, queries, {qid: qrels[qid] for qid in queries})
