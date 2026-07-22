"""A tiny, offline retrieval-evaluation harness.

Wires the `rag_eval` metrics around a retriever so a retrieval configuration can
be scored on a labelled set. Ships with a dependency-free lexical retriever, so
the whole evaluation loop runs in CI without a model, embeddings, or a GPU.

Swap `lexical_retriever` for an embedding-based retriever to benchmark the real
RAG pipeline against the same metrics.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Mapping, Sequence, Set, Tuple

import rag_eval

Corpus = Mapping[str, str]                         # doc_id -> text
Retriever = Callable[[str, Corpus], List[str]]     # (query, corpus) -> ranked doc_ids
Golden = Sequence[Tuple[str, Set[str]]]            # (question, relevant_doc_ids)

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> Set[str]:
    return set(_WORD.findall(text.lower()))


def lexical_retriever(query: str, corpus: Corpus) -> List[str]:
    """Rank docs by token overlap with the query (score desc, id asc for stability)."""
    q = _tokens(query)
    scored = [(doc_id, len(q & _tokens(text))) for doc_id, text in corpus.items()]
    scored.sort(key=lambda it: (-it[1], it[0]))
    return [doc_id for doc_id, score in scored if score > 0]


def evaluate(
    golden: Golden,
    corpus: Corpus,
    retriever: Retriever = lexical_retriever,
    k: int = 5,
) -> Dict[str, float]:
    """Mean retrieval metrics over the golden set."""
    if not golden:
        return {"recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0, "k": float(k)}
    n = len(golden)
    rec = prec = mrr = ndcg = 0.0
    for question, relevant in golden:
        ranked = retriever(question, corpus)
        rec += rag_eval.recall_at_k(ranked, relevant, k)
        prec += rag_eval.precision_at_k(ranked, relevant, k)
        mrr += rag_eval.reciprocal_rank(ranked, relevant)
        ndcg += rag_eval.ndcg_at_k(ranked, relevant, k)
    return {
        "recall@k": rec / n,
        "precision@k": prec / n,
        "mrr": mrr / n,
        "ndcg@k": ndcg / n,
        "k": float(k),
    }


def format_report(metrics: Mapping[str, float]) -> str:
    k = int(metrics.get("k", 0))
    return (
        f"recall@{k}={metrics['recall@k']:.3f}  "
        f"precision@{k}={metrics['precision@k']:.3f}  "
        f"mrr={metrics['mrr']:.3f}  "
        f"ndcg@{k}={metrics['ndcg@k']:.3f}"
    )


if __name__ == "__main__":
    demo_corpus: Dict[str, str] = {
        "d1": "the cat sat on the mat",
        "d2": "shallow water equations model flood inundation",
        "d3": "a mixture of agents orchestrates language models",
        "d4": "physics informed neural networks predict water depth",
    }
    demo_golden: List[Tuple[str, Set[str]]] = [
        ("model flood inundation with water equations", {"d2", "d4"}),
        ("mixture of agents language model", {"d3"}),
    ]
    print(format_report(evaluate(demo_golden, demo_corpus, k=3)))
