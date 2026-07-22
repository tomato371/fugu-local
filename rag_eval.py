"""Offline retrieval-evaluation metrics for RAG.

Pure standard library. Given a ranked list of retrieved document ids and the
set of relevant ids, compute standard IR metrics. No model, embedding, or
dataset needed — this is the measurement layer for evaluating a retriever
(the groundwork for a properly evaluated RAG pipeline).
"""
from __future__ import annotations

import math
from typing import Hashable, Sequence, Set, Tuple, TypeVar

# Element (document-id) type: any hashable, but the same within a call — so a
# Set[str] argument is accepted (avoids Set's invariance biting the caller).
T = TypeVar("T", bound=Hashable)


def _hits_at_k(retrieved: Sequence[T], relevant: Set[T], k: int) -> int:
    return sum(1 for doc in retrieved[:k] if doc in relevant)


def recall_at_k(retrieved: Sequence[T], relevant: Set[T], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k. 0.0 if none are relevant."""
    if not relevant:
        return 0.0
    return _hits_at_k(retrieved, relevant, k) / len(relevant)


def precision_at_k(retrieved: Sequence[T], relevant: Set[T], k: int) -> float:
    """Fraction of the top-k that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    return _hits_at_k(retrieved, relevant, k) / k


def reciprocal_rank(retrieved: Sequence[T], relevant: Set[T]) -> float:
    """1 / rank of the first relevant doc (rank is 1-indexed). 0.0 if none found."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: Sequence[T], relevant: Set[T], k: int) -> float:
    """Discounted cumulative gain at k, binary relevance."""
    return sum(
        1.0 / math.log2(i + 1)
        for i, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant
    )


def ndcg_at_k(retrieved: Sequence[T], relevant: Set[T], k: int) -> float:
    """Normalized DCG at k, binary relevance. 0.0 if none are relevant."""
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


def mean_reciprocal_rank(results: Sequence[Tuple[Sequence[T], Set[T]]]) -> float:
    """Mean reciprocal rank across multiple (retrieved, relevant) query pairs."""
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in results) / len(results)
