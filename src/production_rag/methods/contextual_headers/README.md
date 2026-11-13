# Contextual headers: a context sentence for every chunk

Extra: `local-models`. Needs an LLM at index time (`OPENAI_API_KEY`).

The title prefix and the heading path already exist in the repo: `core/indexing.py` embeds every chunk as `title + "\n" + text`, and the `markdown` chunker prefixes each chunk with its heading path ([`chunking.py`](../../core/chunking.py)). This method adds only the part that is new: one LLM-written sentence per chunk. The baseline in the table is those existing title-only chunks.

## When to use
- Chunks lose their subject once they are cut out of the document: "Staff get twelve days." says nothing about leave without the page around it.

## When not to
- Documents are short or every chunk already names its subject. The sentence costs one LLM call per chunk at index time and adds nothing.

## How it works
- [`context.py`](context.py): for each chunk, the LLM reads the document and the chunk and writes one short sentence that places the chunk in the document (plain text, capped at 100 tokens, in the document's language). A document longer than 12,000 characters is cut there before it goes into the prompt.
- The sentence is embedded and BM25-encoded together with the title and the chunk. The stored and returned text stays the raw chunk, so reranking, display and generation are unchanged.
- Results are cached on disk under `.cache/llm` (git-ignored), keyed by model, prompt and input text, so a rerun costs no tokens. A failed or empty generation leaves that chunk without context and is not cached. `BudgetExceeded` propagates, and what was already generated stays cached.
- [`pipeline.py`](pipeline.py): `index(deps, llm, cache, collection, docs, size=800)` chunks the documents (recursive, 800 characters), adds the context and indexes them; `search` searches that collection with the `chunk_recursive_800` configuration (dense).

Source: Anthropic, [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (2024). Own prompt, reimplemented from the description.

## Run

```bash
OPENAI_API_KEY=... uv run --extra local-models python -m production_rag.methods.contextual_headers.example
```

Without a key the example says so and stops.

## Results
Not measured yet.
