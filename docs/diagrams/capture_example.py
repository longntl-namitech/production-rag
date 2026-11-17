"""Run one real SciFact query through every method and save what each step returned.

    docker compose up -d qdrant   # with the eval collections built by core/eval/run.py
    OPENAI_API_KEY=... uv run --extra local-models --extra eval \
        python docs/diagrams/capture_example.py

`make_diagrams.py` reads `example_query.json` for the example in each method diagram. The LLM
steps call the model for real, about 12 calls and under 20k tokens in all (the budget is 60k).
The collections are the ones the retrieval eval already built: `eval-scifact` holds whole
documents and `eval-scifact-c800` holds fixed 800-character chunks.
"""

import asyncio
import json
import tempfile
from pathlib import Path

from production_rag.core.cache import DiskCache
from production_rag.core.chunking import chunk_documents, fixed_size_chunks, recursive_chunks
from production_rag.core.config import Settings
from production_rag.core.deps import build_deps, warm_up
from production_rag.core.eval.datasets import load_claims, load_dataset
from production_rag.core.llm import OpenAIChat, TokenBudget
from production_rag.core.models import SearchSpec
from production_rag.core.semantic_chunking import semantic_chunk_documents
from production_rag.methods.bm25 import pipeline as bm25
from production_rag.methods.context_expansion import pipeline as context_expansion
from production_rag.methods.contextual_compression.compress import compress
from production_rag.methods.contextual_headers.context import add_context
from production_rag.methods.deadline_fanout import pipeline as deadline_fanout
from production_rag.methods.dense import pipeline as dense
from production_rag.methods.fact_check.pipeline import fact_check
from production_rag.methods.hybrid_fusion import pipeline as hybrid_fusion
from production_rag.methods.hyde import pipeline as hyde
from production_rag.methods.hyde.generate import queries_for as hyde_queries
from production_rag.methods.multi_query import pipeline as multi_query
from production_rag.methods.multi_query.rewrite import rewrite
from production_rag.methods.naive_rag import pipeline as naive_rag
from production_rag.methods.query_decomposition.generate import queries_for as decompose_queries
from production_rag.methods.rerank import pipeline as rerank
from production_rag.methods.step_back.generate import queries_for as step_back_queries

QID = "54"
FULL = "eval-scifact"
C800 = "eval-scifact-c800"
OUT = Path(__file__).parent / "example_query.json"


def hit(row) -> dict:
    return {
        "doc": row.doc_id,
        "chunk": row.chunk_index,
        "score": round(row.score, 4),
        "title": (row.title or "")[:70],
    }


def found(rows, gold: set[str]) -> dict:
    """The top 3 hits and the rank of the gold document in `rows` (None when absent)."""
    rank = next((i + 1 for i, r in enumerate(rows) if r.doc_id in gold), None)
    return {"rank": rank, "top3": [hit(r) for r in rows[:3]]}


async def main() -> None:
    ds = load_dataset("scifact")
    q = ds.queries[QID]
    gold = {d for d, score in ds.qrels[QID].items() if score > 0}
    gold_id = next(iter(gold))
    title, text = ds.corpus[gold_id]
    deps = build_deps(Settings.from_env())
    await warm_up(deps)
    llm = OpenAIChat.from_env(budget=TokenBudget(60_000))
    out: dict = {
        "qid": QID,
        "query": q,
        "model": llm.model,
        "gold": {"doc": gold_id, "title": title, "chars": len(text)},
    }

    # naive_rag: fixed 800-character chunks, dense top 5, one prompt
    top5 = await naive_rag.search(q, C800, deps, k=5)
    answer = await naive_rag.answer(q, C800, deps, llm)
    out["naive_rag"] = {
        "gold_fixed_chunks": len(fixed_size_chunks(text, 800)),
        "top5": [hit(r) for r in top5.rows],
        "answer": answer.text,
        "abstained": answer.abstained,
    }

    # parse_and_chunk: the gold abstract as Markdown, split three ways
    markdown = f"# {title}\n\n{text}"
    semantic = await semantic_chunk_documents({gold_id: (title, markdown)}, deps.embedder)
    out["parse_and_chunk"] = {
        "chars": len(markdown),
        "fixed_800": len(fixed_size_chunks(markdown, 800)),
        "recursive_800": len(recursive_chunks(markdown, 800)),
        "semantic": len(semantic),
    }

    # contextual_headers: the context sentence for the first chunk of the gold document
    doc = {gold_id: (title, text)}
    chunks = chunk_documents(doc, 800, strategy="recursive")
    with tempfile.TemporaryDirectory() as tmp:
        contextual, _ = await add_context(llm, DiskCache(Path(tmp)), doc, chunks[:1])
    out["contextual_headers"] = {
        "chunk": contextual[0].text[:160],
        "context": contextual[0].context,
        "chunks": len(chunks),
    }

    # dense, bm25, hybrid: the rank of the gold document in each leg and fused list
    legs = {
        "dense": await dense.search(q, FULL, deps, k=100),
        "bm25": await bm25.search(q, FULL, deps, variant="bm25", k=100),
        "bm25_no_idf": await bm25.search(q, FULL, deps, variant="bm25_no_idf", k=100),
        "bm25_vi_stopwords": await bm25.search(q, FULL, deps, variant="bm25_vi_stopwords", k=100),
    }
    for variant in ("hybrid_rrf", "hybrid_dbsf", "hybrid_weighted_rrf"):
        legs[variant] = await hybrid_fusion.search(q, FULL, deps, variant=variant, k=100)
    out["retrieval"] = {name: found(res.rows, gold) for name, res in legs.items()}

    # rerank: hybrid_rrf pool of min(k * 4, 100) for k = 10, then the cross-encoder
    pool = await hybrid_fusion.search(q, FULL, deps, variant="hybrid_rrf", k=40)
    reranked = await rerank.search(q, FULL, deps, k=10)
    out["rerank"] = {
        "pool": len(pool.rows),
        "before": found(pool.rows, gold),
        "after": found(reranked.rows, gold),
        "status": reranked.rerank,
        "rerank_ms": reranked.rerank_ms,
    }

    # query-side methods: one LLM call each, then the merged search
    rewrites, _ = await rewrite(llm, q)
    queries = [q, *rewrites]
    merged = await multi_query.search(queries, FULL, deps, k=100)
    out["multi_query"] = {"queries": queries, **found(merged.rows, gold)}

    passage, _ = await hyde_queries(llm, q)
    only = await hyde.search(passage, FULL, deps, k=100)
    both = await hyde.search([q, *passage], FULL, deps, k=100)
    out["hyde"] = {
        "passage": passage[0],
        "hyde_passage": found(only.rows, gold),
        "hyde_passage_query": found(both.rows, gold),
    }

    stepped, _ = await step_back_queries(llm, q)
    result = await multi_query.search(stepped, FULL, deps, k=100)
    out["step_back"] = {"queries": stepped, **found(result.rows, gold)}

    parts, _ = await decompose_queries(llm, q)
    result = await multi_query.search(parts, FULL, deps, k=100)
    out["query_decomposition"] = {"queries": parts, **found(result.rows, gold)}

    # context_expansion: the top hit with one chunk either side, on 800-character chunks
    plain = await hybrid_fusion.search(q, C800, deps, variant="hybrid_weighted_rrf", k=5)
    expanded = await context_expansion.search(q, C800, deps, k=5)
    out["context_expansion"] = {
        "hit": hit(plain.rows[0]),
        "hit_chars": len(plain.rows[0].text),
        "expanded_chars": len(expanded.rows[0].text),
        "total_chars": [
            sum(len(r.text) for r in plain.rows),
            sum(len(r.text) for r in expanded.rows),
        ],
    }

    # contextual_compression: the same top 5 as naive_rag, then the sentence picker
    kept, _ = await compress(llm, q, top5.rows)
    out["contextual_compression"] = {
        "passages": [len(top5.rows), len(kept)],
        "chars": [sum(len(r.text) for r in top5.rows), sum(len(r.text) for r in kept)],
        "first_kept": kept[0].text[:200] if kept else None,
    }

    # deadline_fanout: three specs under a 2000 ms and a 150 ms budget
    specs = [
        SearchSpec(name="dense", search_type="dense"),
        SearchSpec(name="sparse", search_type="sparse"),
        SearchSpec(name="hybrid_rerank", search_type="hybrid", rerank=True),
    ]
    await deadline_fanout.search(q, FULL, deps, specs, base="hybrid_rrf", deadline_ms=2000)
    out["deadline_fanout"] = {}
    for ms in (2000, 150):
        resp = await deadline_fanout.search(q, FULL, deps, specs, base="hybrid_rrf", deadline_ms=ms)
        out["deadline_fanout"][str(ms)] = {
            "total_ms": resp.timing.total_ms,
            "specs": {
                name: {"status": r.status, "rerank": r.rerank, "ms": r.elapsed_ms}
                for name, r in resp.results.items()
            },
        }

    # fact_check: the query is a SciFact claim, so its gold label is known
    claims = load_claims()
    checked = await fact_check(q, FULL, deps, llm)
    out["fact_check"] = {
        "gold_label": claims.labels[QID],
        "label": checked.label,
        "claims": [
            {
                "claim": c.claim,
                "query": c.query,
                "label": c.label,
                "evidence": [hit(r) for r in c.rows],
                "gold_in_evidence": any(r.doc_id in gold for r in c.rows),
            }
            for c in checked.claims
        ],
    }

    out["tokens"] = llm.budget.used
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}, {llm.budget.used} tokens")


if __name__ == "__main__":
    asyncio.run(main())
