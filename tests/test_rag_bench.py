"""Unit tests for the offline retrieval-evaluation harness (rag_bench)."""
import rag_bench

CORPUS = {
    "d1": "the cat sat on the mat",
    "d2": "shallow water equations model flood inundation",
    "d3": "a mixture of agents orchestrates language models",
    "d4": "physics informed neural networks predict water depth",
}
GOLDEN = [
    ("model flood inundation with water equations", {"d2", "d4"}),
    ("mixture of agents language model", {"d3"}),
]


def test_lexical_retriever_ranks_overlap_first():
    assert rag_bench.lexical_retriever("mixture of agents", CORPUS)[0] == "d3"


def test_lexical_retriever_drops_zero_overlap():
    ranked = rag_bench.lexical_retriever("mixture of agents", CORPUS)
    assert "d1" not in ranked  # "the cat sat on the mat" shares no query tokens


def test_evaluate_metrics_in_unit_range():
    m = rag_bench.evaluate(GOLDEN, CORPUS, k=3)
    for key in ("recall@k", "precision@k", "mrr", "ndcg@k"):
        assert 0.0 <= m[key] <= 1.0
    assert m["mrr"] > 0.0  # relevant docs are retrieved for both queries


def test_evaluate_empty_golden_is_zero():
    m = rag_bench.evaluate([], CORPUS)
    assert m["recall@k"] == 0.0 and m["mrr"] == 0.0


def test_format_report_mentions_k():
    assert "recall@3" in rag_bench.format_report(rag_bench.evaluate(GOLDEN, CORPUS, k=3))
