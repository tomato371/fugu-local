"""Unit tests for the offline retrieval-evaluation metrics (rag_eval)."""
import math

import pytest

import rag_eval

RETRIEVED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d4"}


def test_recall_at_k():
    assert rag_eval.recall_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(0.5)  # d2 in top-3
    assert rag_eval.recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(1.0)


def test_precision_at_k():
    assert rag_eval.precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        rag_eval.precision_at_k(RETRIEVED, RELEVANT, 0)


def test_reciprocal_rank():
    assert rag_eval.reciprocal_rank(RETRIEVED, RELEVANT) == pytest.approx(0.5)  # d2 at rank 2
    assert rag_eval.reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_at_k():
    dcg = 1.0 / math.log2(3)                          # only d2 (rank 2) relevant in top-3
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)    # 2 relevant docs placed ideally
    assert rag_eval.ndcg_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(dcg / idcg)


def test_empty_relevant_is_zero():
    assert rag_eval.recall_at_k(RETRIEVED, set(), 3) == 0.0
    assert rag_eval.ndcg_at_k(RETRIEVED, set(), 3) == 0.0
    assert rag_eval.reciprocal_rank(RETRIEVED, set()) == 0.0


def test_mean_reciprocal_rank():
    q1 = (["a", "b"], {"b"})   # rr = 1/2
    q2 = (["x", "y"], {"x"})   # rr = 1/1
    assert rag_eval.mean_reciprocal_rank([q1, q2]) == pytest.approx(0.75)
    assert rag_eval.mean_reciprocal_rank([]) == 0.0
