"""Qdrant collection layout and the few operations the pipeline needs.

One point per chunk. Named vectors: `dense`, and three sparse ones holding
BM25 weights: `sparse` (IDF on), `sparse_noidf` (same values, no IDF, kept to
measure what IDF buys) and `sparse_vi` (Vietnamese stopwords removed).
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from production_rag.core.models import Row


def point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"{doc_id}#{chunk_index}"))


@dataclass
class ChunkPoint:
    doc_id: str
    chunk_index: int
    text: str
    dense: list[float]
    sparse: models.SparseVector
    sparse_vi: models.SparseVector
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def build_filter(spec_filter: dict[str, Any]) -> models.Filter | None:
    if not spec_filter:
        return None
    conditions = [
        models.FieldCondition(
            key=key,
            match=models.MatchAny(any=value)
            if isinstance(value, list)
            else models.MatchValue(value=value),
        )
        for key, value in spec_filter.items()
    ]
    return models.Filter(must=conditions)


def _row(point: Any) -> Row:
    payload = point.payload or {}
    return Row(
        id=str(point.id),
        doc_id=payload["doc_id"],
        chunk_index=payload.get("chunk_index", 0),
        text=payload.get("text", ""),
        score=getattr(point, "score", None) or 0.0,
        title=payload.get("title"),
        tags=payload.get("tags", []),
    )


class Store:
    def __init__(self, client: AsyncQdrantClient):
        self.client = client

    async def ensure_collection(self, name: str, dense_dim: int) -> None:
        if await self.client.collection_exists(name):
            return
        idf = models.SparseVectorParams(modifier=models.Modifier.IDF)
        await self.client.create_collection(
            name,
            vectors_config={
                "dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": idf,
                "sparse_noidf": models.SparseVectorParams(),
                "sparse_vi": idf,
            },
        )
        for key, schema in [
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("tags", models.PayloadSchemaType.KEYWORD),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
        ]:
            await self.client.create_payload_index(name, key, field_schema=schema)

    async def count(self, name: str) -> int:
        return (await self.client.count(name, exact=True)).count

    async def upsert(self, name: str, points: list[ChunkPoint]) -> None:
        await self.client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=point_id(p.doc_id, p.chunk_index),
                    vector={
                        "dense": p.dense,
                        "sparse": p.sparse,
                        "sparse_noidf": p.sparse,
                        "sparse_vi": p.sparse_vi,
                    },
                    payload={
                        **p.extra,
                        "doc_id": p.doc_id,
                        "chunk_index": p.chunk_index,
                        "text": p.text,
                        "title": p.title,
                        "tags": p.tags,
                    },
                )
                for p in points
            ],
        )

    async def search(
        self, name: str, query_kwargs: dict[str, Any], limit: int, threshold: float | None
    ) -> list[Row]:
        response = await self.client.query_points(
            name, **query_kwargs, limit=limit, score_threshold=threshold, with_payload=True
        )
        return [_row(p) for p in response.points]

    async def fetch_chunks(
        self, name: str, doc_id: str, start: int | None, end: int | None
    ) -> list[Row]:
        conditions: list[Any] = [
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        ]
        if start is not None or end is not None:
            conditions.append(
                models.FieldCondition(key="chunk_index", range=models.Range(gte=start, lte=end))
            )
        points, _ = await self.client.scroll(
            name, scroll_filter=models.Filter(must=conditions), limit=1000, with_payload=True
        )
        return sorted((_row(p) for p in points), key=lambda r: r.chunk_index)
