import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from production_rag.core.eval.datasets import claim_label, load_claims, read_claims


def _tar(tmp_path: Path, rows: list[dict]) -> Path:
    data = "\n".join(json.dumps(r) for r in rows).encode()
    path = tmp_path / "data.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("data/claims_dev.jsonl")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return path


def test_claim_label_mapping():
    assert claim_label({}) == "unverified"
    assert claim_label({"5": [{"sentences": [1], "label": "SUPPORT"}]}) == "supported"
    assert claim_label({"5": [{"sentences": [1], "label": "CONTRADICT"}]}) == "refuted"
    with pytest.raises(ValueError):
        claim_label({"5": [{"label": "SUPPORT"}], "6": [{"label": "CONTRADICT"}]})


def test_read_claims_gold_docs(tmp_path):
    tar = _tar(
        tmp_path,
        [
            {"id": 1, "claim": "c1", "evidence": {}, "cited_doc_ids": [7]},
            {
                "id": 3,
                "claim": "c3",
                "evidence": {"9": [{"sentences": [0], "label": "SUPPORT"}]},
                "cited_doc_ids": [9, 10],
            },
        ],
    )
    ds = read_claims(tar)
    assert ds.labels == {"1": "unverified", "3": "supported"}
    assert ds.gold_docs == {"1": ["7"], "3": ["9"]} and ds.claims["3"] == "c3"


@pytest.mark.integration
def test_load_claims_dev_counts():
    ds = load_claims()
    assert Counter(ds.labels.values()) == {"supported": 124, "refuted": 64, "unverified": 112}
    assert list(load_claims(30).claims) == list(load_claims(300).claims)[:30]
