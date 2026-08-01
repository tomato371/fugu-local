# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""配線1c(FUGU_RAG adaptive 検索)と配線1d(Deep Research 統合)のオフラインテスト。"""
import sys
import types

import pytest

import fugu_local


@pytest.fixture(autouse=True)
def _clear_adaptive_cache():
    fugu_local._RAG_ADAPTIVE_CACHE.clear()
    yield
    fugu_local._RAG_ADAPTIVE_CACHE.clear()


CHUNKS = [("D:/docs/a.md", "PINN は物理法則を損失に入れる"),
          ("D:/docs/b.md", "洪水は浅水方程式でモデル化する")]


# ------------------------------------------------------------------ 1c: _rag_adaptive

def test_rag_adaptive_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FUGU_RAG", raising=False)
    assert fugu_local._rag_adaptive("q", CHUNKS, 2) is None


def test_rag_adaptive_uses_bridge_retriever(monkeypatch):
    import fugu_rag_retriever
    monkeypatch.setenv("FUGU_RAG", "1")
    seen = {}

    def fake_retriever(query, corpus):
        seen["query"] = query
        seen["corpus_keys"] = sorted(corpus)
        return ["D:/docs/b.md#1", "D:/docs/a.md#0"]  # adaptive の順位

    monkeypatch.setattr(fugu_rag_retriever, "fugu_rag_retriever",
                        lambda embed_model="nomic-embed-text",
                        chat_model=None, k=5: fake_retriever)
    hits = fugu_local._rag_adaptive("洪水の質問", CHUNKS, 2)
    assert hits == [CHUNKS[1], CHUNKS[0]]           # adaptive の順で返る
    assert seen["query"] == "洪水の質問"
    assert seen["corpus_keys"] == ["D:/docs/a.md#0", "D:/docs/b.md#1"]


def test_rag_adaptive_caches_retriever(monkeypatch):
    import fugu_rag_retriever
    monkeypatch.setenv("FUGU_RAG", "1")
    builds = []

    def build(embed_model="nomic-embed-text", chat_model=None, k=5):
        builds.append(k)
        return lambda query, corpus: list(corpus)[:k]

    monkeypatch.setattr(fugu_rag_retriever, "fugu_rag_retriever", build)
    fugu_local._rag_adaptive("q1", CHUNKS, 2)
    fugu_local._rag_adaptive("q2", CHUNKS, 2)
    assert builds == [2]                             # 2回目はキャッシュ


def test_rag_adaptive_unavailable_or_failing_is_none(monkeypatch):
    import fugu_rag_retriever
    monkeypatch.setenv("FUGU_RAG", "1")
    monkeypatch.setattr(fugu_rag_retriever, "fugu_rag_retriever",
                        lambda **kw: None)           # fugu_rag 不在/embedder 不可
    assert fugu_local._rag_adaptive("q", CHUNKS, 2) is None
    fugu_local._RAG_ADAPTIVE_CACHE.clear()
    monkeypatch.setattr(fugu_rag_retriever, "fugu_rag_retriever",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert fugu_local._rag_adaptive("q", CHUNKS, 2) is None


def test_rag_search_uses_adaptive_hits(monkeypatch):
    monkeypatch.setattr(fugu_local, "_get_rag_chunks", lambda dirs: CHUNKS)
    monkeypatch.setattr(fugu_local, "_rag_adaptive",
                        lambda q, chunks, k: [CHUNKS[1]])
    out = fugu_local.rag_search("洪水", dirs=["D:/docs"], top_k=2)
    assert out.startswith("## Relevant Document Context (RAG)")
    assert "[Source: b.md]" in out and "浅水方程式" in out
    assert "a.md" not in out


def test_rag_search_falls_back_to_keyword_path(monkeypatch):
    monkeypatch.setattr(fugu_local, "_get_rag_chunks", lambda dirs: CHUNKS)
    monkeypatch.setattr(fugu_local, "_rag_adaptive", lambda q, c, k: None)
    out = fugu_local.rag_search("洪水のモデル化", dirs=["D:/docs"], top_k=2)
    assert "[Source: b.md]" in out                   # 従来キーワード検索が動く


# ------------------------------------------------------------------ 1d: adapters

def test_research_search_fn_parses_formatted_results(monkeypatch):
    monkeypatch.setattr(fugu_local, "_search_raw", lambda q: [
        "[Title A]\nsnippet body A\nSource: https://ex.com/a",
        "[Title B]\nsnippet body B\nSource: ftp://ignored",   # 非HTTPは捨てる
        "no source line here",
    ])
    hits = fugu_local._research_search_fn("q")
    assert hits == [("https://ex.com/a", "[Title A]\nsnippet body A")]


def test_make_research_retrieve_fn_none_without_dirs(monkeypatch):
    monkeypatch.setattr(fugu_local, "RAG_DIRS", [])
    assert fugu_local._make_research_retrieve_fn(None) is None


def test_make_research_retrieve_fn_returns_scored_pairs(monkeypatch):
    monkeypatch.setattr(fugu_local, "_get_rag_chunks", lambda dirs: CHUNKS)
    retrieve_fn = fugu_local._make_research_retrieve_fn(["D:/docs"])
    hits = retrieve_fn("洪水のモデル化について")
    assert ("b.md", CHUNKS[1][1]) in hits            # スコア>0 のみ・basename化
    assert all(score_pair[0].endswith(".md") for score_pair in hits)


# ------------------------------------------------------------------ 1d: run_deep_research

def _fake_fugu_rag(monkeypatch, run_research):
    pkg = types.ModuleType("fugu_rag")
    research_mod = types.ModuleType("fugu_rag.research")
    research_mod.run_research = run_research
    pkg.research = research_mod
    monkeypatch.setitem(sys.modules, "fugu_rag", pkg)
    monkeypatch.setitem(sys.modules, "fugu_rag.research", research_mod)


def test_run_deep_research_wires_all_callables(monkeypatch):
    captured = {}

    class Report:
        report = "# 引用付きレポート [1]"

    def fake_run_research(question, chat, **kwargs):
        captured["question"] = question
        captured.update(kwargs)
        return Report()

    _fake_fugu_rag(monkeypatch, fake_run_research)
    monkeypatch.setattr(fugu_local, "RAG_DIRS", ["D:/docs"])
    out = fugu_local.run_deep_research("大きな質問")
    assert out == "# 引用付きレポート [1]"
    assert captured["question"] == "大きな質問"
    assert captured["search_fn"] is fugu_local._research_search_fn
    assert captured["retrieve_fn"] is not None       # RAG_DIRS ありなので接続
    assert captured["fetch_fn"] is not None          # fugu_browser 経由
    assert captured["max_workers"] == 1              # 8GB 実測に基づく直列


def test_run_deep_research_missing_fugu_rag_is_error(monkeypatch):
    for name in ("fugu_rag", "fugu_rag.research"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    out = fugu_local.run_deep_research("q")
    assert out.startswith("__ERROR__")
    assert "pip install" in out                      # 導入手順を含む


def test_run_deep_research_engine_failure_is_error(monkeypatch):
    _fake_fugu_rag(monkeypatch, lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("engine down")))
    out = fugu_local.run_deep_research("q")
    assert out.startswith("__ERROR__") and "engine down" in out


# ------------------------------------------------------------------ 1d: ask_fugu route

def test_ask_fugu_deep_research_route(monkeypatch):
    monkeypatch.setattr(fugu_local, "setup", lambda: True)
    monkeypatch.setattr(fugu_local, "run_deep_research",
                        lambda q, rag_dirs=None: "# REPORT\n[1] src")
    out = fugu_local.ask_fugu("調べて", deep_research=True)
    assert out == "# REPORT\n[1] src"


def test_ask_fugu_deep_research_error_passthrough(monkeypatch):
    monkeypatch.setattr(fugu_local, "setup", lambda: True)
    monkeypatch.setattr(fugu_local, "run_deep_research",
                        lambda q, rag_dirs=None: "__ERROR__: no fugu_rag")
    out = fugu_local.ask_fugu("調べて", deep_research=True)
    assert out.startswith("__ERROR__")
