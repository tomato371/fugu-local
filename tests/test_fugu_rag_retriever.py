# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the optional fugu-rag bridge (offline: fugu-rag not installed here)."""
from fugu_rag_retriever import fugu_rag_retriever, get_retriever
from rag_bench import lexical_retriever


def test_get_retriever_falls_back_to_lexical_without_fugu_rag():
    # fugu-rag is not installed in fugu-local's CI, so the bridge must degrade to
    # the built-in lexical retriever rather than raising.
    assert get_retriever() is lexical_retriever


def test_fugu_rag_retriever_returns_none_when_unavailable():
    assert fugu_rag_retriever() is None


def test_fallback_retriever_is_callable_and_ranks():
    corpus = {"d1": "flood water depth model", "d2": "the cat sat on the mat"}
    ranked = get_retriever()("water depth", corpus)
    assert ranked and ranked[0] == "d1"
