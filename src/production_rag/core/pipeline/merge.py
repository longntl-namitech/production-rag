from production_rag.core.models import Row


def merge_max(lists: list[list[Row]], k: int) -> list[Row]:
    """Union of result lists from several subqueries; each chunk keeps its best score."""
    best: dict[str, Row] = {}
    for rows in lists:
        for row in rows:
            if row.id not in best or row.score > best[row.id].score:
                best[row.id] = row
    return sorted(best.values(), key=lambda r: r.score, reverse=True)[:k]
