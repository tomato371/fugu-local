# Plan: an evaluated RAG rebuild

Retrieval-augmented generation is only as good as you can *measure* it. This document
plans a rebuild of Fugu-Local's RAG around explicit metrics, so changes (chunking,
embeddings, top-k, reranking) can be judged by numbers rather than vibes.

## Two layers to measure

RAG quality splits into **retrieval** (did we fetch the right context?) and
**generation** (did we answer faithfully from it?).

### 1. Retrieval metrics — implemented in `rag_eval.py`

Given a ranked list of retrieved document ids and the set of relevant ids:

| Metric | Question it answers |
| --- | --- |
| `recall@k` | Did the relevant docs make it into the top-k? |
| `precision@k` | How much of the top-k is actually relevant? |
| `MRR` / `reciprocal_rank` | How high was the first relevant doc? |
| `nDCG@k` | Are relevant docs ranked near the top? |

These are pure, offline, and unit-tested — no model or GPU needed — so they can gate CI
and be tracked across retriever changes.

### 2. Generation metrics — RAGAS-style (to add)

Judged with an LLM (locally, via the existing Ollama models):

- **Faithfulness** — is every claim in the answer supported by the retrieved context?
- **Answer relevance** — does the answer actually address the question?
- **Context precision** — is the retrieved context on-topic (not padded with noise)?
- **Context recall** — does the retrieved context contain everything needed to answer?

## Evaluation harness (proposed)

1. **Golden set.** A small JSONL of `{question, answer, relevant_doc_ids}` over a fixed
   document corpus.
2. **Retrieval pass.** Run the retriever for each question → compute the `rag_eval.py`
   metrics; aggregate mean recall@k / nDCG@k.
3. **Generation pass.** Run the full RAG answer → score faithfulness / answer-relevance /
   context precision+recall with an LLM judge.
4. **Report.** Emit a metrics table per configuration; compare against the previous run.

## Knobs to sweep

Chunk size & overlap · embedding model · `top_k` · reranking (cross-encoder) · with vs.
without query rewriting. Each becomes a row in the report, chosen by the numbers above.

## Next steps

- [ ] Build the golden set (start with ~20 Q/A over a known corpus).
- [ ] Wire `rag_eval.py` into a `rag_bench.py` retrieval-evaluation runner.
- [ ] Add the LLM-judge generation metrics.
- [ ] Add a CI job that runs the retrieval-metrics portion on a tiny fixture corpus.
