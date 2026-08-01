# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_local への FUGU_SEARCH / FUGU_MAV 配線のオフラインテスト。

フラグ未設定時にフックが完全な素通し(None)であること、設定時に fugu_search /
fugu_verify へ正しい引数(特に予算の同一性)が渡ることを検証する。
"""
import pytest

import fugu_local as f
import fugu_search
import fugu_verify


PROPOSALS = [("m1", "提案いち"), ("m2", "提案に"), ("m3", "提案さん")]


# ------------------------------------------------------------- フラグ未設定

def test_mav_off_is_a_pure_passthrough(monkeypatch):
    monkeypatch.delenv("FUGU_MAV", raising=False)

    def boom(*a, **k):
        raise AssertionError("must not be called when flag is off")
    monkeypatch.setattr(fugu_verify, "best_of_n", boom)
    assert f._mav_select("q", PROPOSALS, "統合") is None


def test_search_off_is_a_pure_passthrough(monkeypatch):
    monkeypatch.delenv("FUGU_SEARCH", raising=False)

    def boom(*a, **k):
        raise AssertionError("must not be called when flag is off")
    monkeypatch.setattr(fugu_search, "search", boom)
    assert f._tree_search_refine("q", "draft", "issue", ["m1"], 2) is None


# ------------------------------------------------------------------- MAV 配線

@pytest.fixture()
def mav_env(monkeypatch):
    monkeypatch.setenv("FUGU_MAV", "1")
    # 検証者の構築は AskChat の生成のみ(LLM 呼び出しなし)だが、
    # best_of_n をフェイクするので検証者自体も使われない
    monkeypatch.setattr(f, "_mav_verifiers", lambda: ["fake-verifier"])
    return monkeypatch


def test_mav_keeps_aggregate_when_it_wins(mav_env):
    def fake_bon(question, candidates, verifiers, **kw):
        reports = [fugu_verify.ScoreReport(score=s)
                   for s in (0.9, 0.4, 0.3, 0.2)]
        return 0, reports[0], reports
    mav_env.setattr(fugu_verify, "best_of_n", fake_bon)
    assert f._mav_select("q", PROPOSALS, "統合") is None   # 統合維持 → 従来と同じ


def test_mav_adopts_winning_proposal(mav_env):
    captured = {}

    def fake_bon(question, candidates, verifiers, **kw):
        captured["candidates"] = list(candidates)
        reports = [fugu_verify.ScoreReport(score=s)
                   for s in (0.3, 0.4, 0.9, 0.2)]
        return 2, reports[2], reports
    mav_env.setattr(fugu_verify, "best_of_n", fake_bon)
    picked = f._mav_select("q", PROPOSALS, "統合")
    assert picked == "提案に"
    # 候補列 = 統合(先頭) + 提案たち
    assert captured["candidates"][0] == "統合"
    assert captured["candidates"][1:] == ["提案いち", "提案に", "提案さん"]


def test_mav_respects_n_cap(mav_env):
    mav_env.setenv("FUGU_MAV_N", "2")
    captured = {}

    def fake_bon(question, candidates, verifiers, **kw):
        captured["candidates"] = list(candidates)
        reports = [fugu_verify.ScoreReport(score=0.5)] * len(candidates)
        return 0, reports[0], reports
    mav_env.setattr(fugu_verify, "best_of_n", fake_bon)
    f._mav_select("q", PROPOSALS, "統合")
    assert len(captured["candidates"]) == 3    # 統合 + 提案 2 件まで


def test_mav_failure_is_safe(mav_env):
    def boom(*a, **k):
        raise RuntimeError("verifier explosion")
    mav_env.setattr(fugu_verify, "best_of_n", boom)
    assert f._mav_select("q", PROPOSALS, "統合") is None


# ----------------------------------------------------------------- 探索の配線

@pytest.fixture()
def search_env(monkeypatch):
    monkeypatch.setenv("FUGU_SEARCH", "1")
    monkeypatch.setattr(f, "_mav_verifiers", lambda: ["fake-verifier"])
    return monkeypatch


def _fake_result(answer="探索結果", score=0.95):
    return fugu_search.SearchResult(answer=answer, score=score, calls_used=5,
                                    nodes=5, max_depth=2, root_width=3,
                                    best_depth=1)


def test_search_budget_equals_remaining_linear_rounds(search_env):
    captured = {}

    def fake_search(question, chat_factory, scorer, **kw):
        captured.update(kw)
        return _fake_result()
    search_env.setattr(fugu_search, "search", fake_search)
    out = f._tree_search_refine("q", "draft", "issue", ["m1", "m2", "m3"], 2)
    assert out == "探索結果"
    # 予算 = 残り 2 ラウンド × (提案 3 + 統合 1 + 批評 1) = 10 生成呼び出し
    assert captured["budget"] == 10
    assert captured["seed_answer"] == "draft"
    assert captured["models"] == ["m1", "m2", "m3"]
    assert captured["plan_hint"] == ["m1", "m2", "m3"]


def test_search_budget_env_override(search_env):
    search_env.setenv("FUGU_SEARCH_BUDGET", "7")
    captured = {}

    def fake_search(question, chat_factory, scorer, **kw):
        captured.update(kw)
        return _fake_result()
    search_env.setattr(fugu_search, "search", fake_search)
    f._tree_search_refine("q", "draft", "issue", ["m1"], 3)
    assert captured["budget"] == 7


def test_search_failure_falls_back_to_linear_rounds(search_env):
    def boom(*a, **k):
        raise RuntimeError("search exploded")
    search_env.setattr(fugu_search, "search", boom)
    assert f._tree_search_refine("q", "draft", "issue", ["m1"], 1) is None


def test_search_chat_factory_resolves_default_model(search_env):
    """multi 無効時(model=None)は先頭プロポーザーで生成する。"""
    captured = {}

    def fake_search(question, chat_factory, scorer, **kw):
        chat = chat_factory(None)
        captured["default_model"] = chat.model
        chat2 = chat_factory("qwen3.6:35b")
        captured["explicit_model"] = chat2.model
        return _fake_result()
    search_env.setattr(fugu_search, "search", fake_search)
    f._tree_search_refine("q", "draft", "issue", ["gpt-oss:20b", "x"], 1)
    assert captured["default_model"] == "gpt-oss:20b"
    assert captured["explicit_model"] == "qwen3.6:35b"


# ------------------------------------------------------- 検証者の割り当て

def test_mav_verifiers_use_the_small_model():
    """検証者は小型モデル(FALLBACK_MODEL)で回す — 予算を溶かさない。"""
    verifiers = f._mav_verifiers()
    aspects = [v.aspect for v in verifiers]
    assert aspects == ["logical", "computational", "factual",
                       "constraint", "codetest"]
    for v in verifiers:
        assert v.chat.model == f.FALLBACK_MODEL
