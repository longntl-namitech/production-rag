# Contextual compression: send only the sentences that help

Extra: `local-models`. Needs an LLM at query time (`OPENAI_API_KEY`).

Differs from [`context_expansion`](../context_expansion/README.md): that one adds text around a hit, this one removes text from it.

## When to use
- Retrieved passages are long and mostly off-topic, so the prompt pays for tokens that do not help the answer.

## When not to
- Passages are already short (one or two sentences), or the answer needs the surrounding text: the extra LLM call costs more than it saves, and a trimmed passage can lose context the answer needs.

## How it works
- [`compress.py`](compress.py): after retrieval, one LLM call gets the question and every retrieved passage split into numbered sentences (`[1.1]`, `[1.2]`, `[2.1]`, ...) and replies with the ids to keep (JSON, capped at 256 tokens). The text that comes back is always the original sentences, in their original order, joined with a space; the model never writes new words. A passage with no kept sentence is dropped.
- Fallback: a reply that cannot be used, a failed call, an empty list or no valid id returns the passages whole. `BudgetExceeded` propagates.
- [`pipeline.py`](pipeline.py): retrieval is [`naive_rag`](../naive_rag/README.md) (fixed 800-character chunks, dense top 5), then `compress`.
- Retrieval does not change, so there is no retrieval row: nDCG@10 would be the same as the method it sits on. It is measured only in the answer eval, as the `contextual_compression` front end.

Sources: Xu et al., [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408) (2023), and the contextual compressor pattern of LangChain. Own prompt, reimplemented from the descriptions.

## Run

```bash
OPENAI_API_KEY=... uv run --extra local-models python -m production_rag.methods.contextual_compression.example
```

Without a key the example prints the passages before compression and says so.

## Results
Not measured yet.
