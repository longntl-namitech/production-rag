"""Public eval datasets, downloaded from Hugging Face at runtime (never committed).

- scifact: BEIR SciFact. `BeIR/scifact` configs `corpus` / `queries` (columns
  `_id`, `title`, `text`) and `BeIR/scifact-qrels` split `test` (`query-id`,
  `corpus-id`, `score`).
- tvpl: Vietnamese legal questions, `GreenNode/TVPL-Retrieval-VN` (CC-BY-SA-4.0),
  configs `corpus` (`id`, `title`, `text`), `queries` (`id`, `text`), `qrels`,
  all split `test`. Scores are strings ("1", "1.0").

Answer datasets (question, gold answer, corpus):

- mini_wiki: `rag-datasets/rag-mini-wikipedia`, licence `cc-by-3.0` per its card. Configs
  `text-corpus` split `passages` and `question-answer` split `test`.
- vi_legal: `DuyTa/vi_RAG` (now served from `sailor2/Vietnamese_RAG`), config `LegalRAG`
  split `train`, licence `apache-2.0` per its card. Gold answers were aligned with LLMs
  (Mixtral 8x22B, GPT-4) per the card. Each row carries `context` passages prefixed
  `ngữ cảnh <n>:`; the corpus is every unique passage across all rows.
  Checked 2026-11-11: no Vietnamese legal VLQA release exists (the only Hugging Face hit is
  the unrelated `yyyyifan/VLQA`), so vi_RAG stays the Vietnamese answer dataset.
"""

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Literal

DEFAULT_QUERY_LIMIT = {"scifact": None, "tvpl": 1000}
ANSWER_DATASETS = ("mini_wiki", "vi_legal")
DEFAULT_SAMPLE = {"mini_wiki": 150, "vi_legal": 100}
_CONTEXT_PREFIX = re.compile(r"^\s*ngữ cảnh\s*\d+\s*:\s*")


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


@dataclass
class AnswerDataset:
    name: str
    corpus: dict[str, tuple[str | None, str]]
    questions: dict[str, str]
    gold: dict[str, str]


def sample_ids(ids: list[str], n: int, seed: int = 13) -> list[str]:
    """Prefix-stable sample: the first k of a larger sample are the k-sample."""
    return random.Random(seed).sample(sorted(ids), len(ids))[:n]


def strip_context_prefix(text: str) -> str:
    return _CONTEXT_PREFIX.sub("", text).strip()


def legal_corpus(rows: list[dict]) -> dict[str, tuple[None, str]]:
    corpus: dict[str, tuple[None, str]] = {}
    for row in rows:
        for passage in row["context"]:
            text = strip_context_prefix(passage)
            if text:
                key = "vl-" + hashlib.sha1(text.encode()).hexdigest()[:12]
                corpus.setdefault(key, (None, text))
    return corpus


def _with_ids(rows, prefix: str) -> list[tuple[str, dict]]:
    return [(f"{prefix}{r['id'] if 'id' in r else i}", r) for i, r in enumerate(rows)]


def load_answer_dataset(
    name: Literal["mini_wiki", "vi_legal"], sample: int | None = None, seed: int = 13
) -> AnswerDataset:
    from datasets import load_dataset as hf_load

    if name == "mini_wiki":
        repo = "rag-datasets/rag-mini-wikipedia"
        passages = hf_load(repo, "text-corpus", split="passages")
        corpus = {
            doc_id: (None, row["passage"].strip())
            for doc_id, row in _with_ids(passages, "w")
            if row["passage"].strip()
        }
        qa = _with_ids(hf_load(repo, "question-answer", split="test"), "q")
        questions = {qid: row["question"] for qid, row in qa}
        gold = {qid: row["answer"] for qid, row in qa}
    elif name == "vi_legal":
        rows = list(hf_load("DuyTa/vi_RAG", "LegalRAG", split="train"))
        corpus = legal_corpus(rows)
        questions = {f"q{i}": row["question"] for i, row in enumerate(rows)}
        gold = {f"q{i}": row["answer"] for i, row in enumerate(rows)}
    else:
        raise ValueError(f"unknown answer dataset {name!r}")

    if sample is not None:
        keep = sample_ids(list(questions), sample, seed)
        questions = {qid: questions[qid] for qid in keep}
        gold = {qid: gold[qid] for qid in keep}
    return AnswerDataset(name, corpus, questions, gold)
