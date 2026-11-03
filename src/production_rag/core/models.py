"""Request and response models for the search pipeline."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Query = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
FilterValue = str | int | list[str | int]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ContextOption(_Frozen):
    """Neighbour chunks to add around each hit, or the whole document."""

    prev: int = Field(0, ge=0, le=5)
    next: int = Field(0, ge=0, le=5)
    full: bool = False


class SearchSpec(_Frozen):
    """One independent search branch: own filter, k, budget and error isolation."""

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    k: int = Field(10, ge=1, le=100)
    search_type: Literal["hybrid", "dense", "sparse"] = "hybrid"
    filter: dict[str, FilterValue] = Field(default_factory=dict)
    threshold: float | None = Field(None, ge=0, le=1)
    rerank: bool | None = None
    context: ContextOption | None = None


class SearchRequest(_Frozen):
    query: Query | Annotated[list[Query], Field(min_length=1, max_length=8)]
    rerank_query: Query | None = None
    collection: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    specs: list[SearchSpec] = Field(
        default_factory=lambda: [SearchSpec(name="default")], min_length=1
    )

    @model_validator(mode="after")
    def _unique_spec_names(self) -> "SearchRequest":
        names = [s.name for s in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("spec names must be unique")
        return self

    @property
    def subqueries(self) -> list[str]:
        return [self.query] if isinstance(self.query, str) else list(self.query)


class Row(_Frozen):
    id: str
    doc_id: str
    chunk_index: int = 0
    text: str
    score: float
    title: str | None = None
    tags: list[str] = Field(default_factory=list)


class SpecResult(_Frozen):
    status: Literal["ok", "failed"]
    rows: list[Row] = Field(default_factory=list)
    elapsed_ms: int
    error: str | None = None
    rerank: Literal["off", "ok", "failed", "skipped_budget", "skipped_trivial"] = "off"
    rerank_ms: int | None = None


class Timing(_Frozen):
    embed_ms: int
    total_ms: int


class SearchResponse(_Frozen):
    method: str
    results: dict[str, SpecResult]
    timing: Timing
