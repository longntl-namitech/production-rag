"""Ranking metrics. `ranked` is a list of doc ids, `rel` maps doc id to graded relevance."""

import math
import statistics


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & set(rel)) / len(rel)


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    for i, doc_id in enumerate(ranked[:k], start=1):
        if rel.get(doc_id, 0) > 0:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """nDCG with linear gains, as trec_eval and BEIR compute it."""
    dcg = sum(rel.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def rescue_damage(before: list[str], after: list[str], rel: dict[str, int]) -> tuple[bool, bool]:
    """Rescue: top-1 was not relevant and now is. Damage: the reverse."""
    was = bool(before) and rel.get(before[0], 0) > 0
    now = bool(after) and rel.get(after[0], 0) > 0
    return (not was and now, was and not now)


def percentile(values: list[float], p: float) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[round(p) - 1]
