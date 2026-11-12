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

Claim labels (fact-check eval):

- SciFact claims come from the release tarball that `allenai/scifact`'s loading script
  downloads (`data/claims_dev.jsonl`). It is used directly because `datasets` 5.x does not
  run loading scripts. Licence: CC BY-NC 2.0 per the `allenai/scifact` card. Downloaded at
  runtime, never committed.
- Labels map SUPPORT -> supported, CONTRADICT -> refuted, no evidence -> unverified. The dev
  split has 124 supported, 64 refuted and 112 unverified claims (300 in all).
- The dev claim ids are the BEIR SciFact test queries.
"""

import hashlib
import json
import random
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from production_rag.core.eval.download import fetch

DEFAULT_QUERY_LIMIT = {"scifact": None, "tvpl": 1000}
ANSWER_DATASETS = ("mini_wiki", "vi_legal")
DEFAULT_SAMPLE = {"mini_wiki": 150, "vi_legal": 100}
_CONTEXT_PREFIX = re.compile(r"^\s*ngữ cảnh\s*\d+\s*:\s*")
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
LABEL_MAP = {"SUPPORT": "supported", "CONTRADICT": "refuted"}


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


@dataclass
class ClaimDataset:
    claims: dict[str, str]
    labels: dict[str, str]
    gold_docs: dict[str, list[str]]


def claim_label(evidence: dict) -> str:
    labels = {item["label"] for rationales in evidence.values() for item in rationales}
    if not labels:
        return "unverified"
    if len(labels) > 1:
        raise ValueError(f"mixed evidence labels: {sorted(labels)}")
    return LABEL_MAP[labels.pop()]


def read_claims(tar_path: Path, split: str = "dev") -> ClaimDataset:
    name = f"data/claims_{split}.jsonl"
    with tarfile.open(tar_path) as tar:
        member = tar.extractfile(name)
        if member is None:
            raise ValueError(f"{name} not found in {tar_path}")
        rows = [json.loads(line) for line in member.read().decode().splitlines() if line.strip()]
    claims: dict[str, str] = {}
    labels: dict[str, str] = {}
    gold_docs: dict[str, list[str]] = {}
    for row in rows:
        cid = str(row["id"])
        claims[cid] = row["claim"]
        labels[cid] = claim_label(row["evidence"])
        docs = list(row["evidence"]) or row["cited_doc_ids"]
        gold_docs[cid] = [str(d) for d in docs]
    return ClaimDataset(claims, labels, gold_docs)


def load_claims(sample: int | None = None, seed: int = 13) -> ClaimDataset:
    ds = read_claims(fetch(SCIFACT_URL, sha256=SCIFACT_SHA256), "dev")
    if sample is None:
        return ds
    keep = sample_ids(list(ds.claims), sample, seed)
    return ClaimDataset(
        {c: ds.claims[c] for c in keep},
        {c: ds.labels[c] for c in keep},
        {c: ds.gold_docs[c] for c in keep},
    )
