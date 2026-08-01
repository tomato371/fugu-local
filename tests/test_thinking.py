# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_thinking のオフラインテスト(FakeChat のみ、LLM/ネットワーク不要)。"""
import json

from fugu_llm import FakeChat
from fugu_thinking import (
    BUDGETS,
    DEFAULT_BUDGET,
    LEVELS,
    Budget,
    decide_budget,
    heuristic_budget,
    refine_answer,
    reflect,
)


def _boom(prompt):
    raise RuntimeError("model down")


# ------------------------------------------------------------------ BUDGETS

def test_budgets_have_six_levels_in_depth_order():
    assert LEVELS == ("minimal", "low", "medium", "high", "ultra", "max")
    assert set(BUDGETS) == set(LEVELS)
    reflections = [BUDGETS[name].reflections for name in LEVELS]
    assert reflections == sorted(reflections)          # 深いほど非減少
    assert reflections[0] == 0 and reflections[-1] == 4
    min_rounds = [BUDGETS[name].min_rounds for name in LEVELS]
    assert min_rounds == sorted(min_rounds)            # ラウンド下限も非減少
    assert min_rounds[-1] == 3
    assert DEFAULT_BUDGET in BUDGETS


def test_minimal_budget_is_speed_oriented():
    assert BUDGETS["minimal"].think is False
    assert BUDGETS["minimal"].reflections == 0
    assert BUDGETS["minimal"].num_predict is not None
    assert BUDGETS["minimal"].min_rounds == 0


def test_deep_budgets_enable_think_and_rounds():
    for name in ("high", "ultra", "max"):
        assert BUDGETS[name].think is True
    assert BUDGETS["ultra"].min_rounds == 2
    assert BUDGETS["max"].min_rounds == 3


# ------------------------------------------------------------------ heuristic

def test_heuristic_math_cue_is_high():
    assert heuristic_budget("Prove the theorem that sqrt(2) is irrational") == "high"
    assert heuristic_budget("この方程式を導出してください。境界条件は以下の通りで…") == "high"


def test_heuristic_long_hard_problem_is_ultra():
    q = ("Prove the following theorem and derive every intermediate lemma: " +
         "consider a compressible fluid in a rotating frame " * 5)
    assert len(q) > 200
    assert heuristic_budget(q) == "ultra"


def test_heuristic_greeting_is_minimal():
    assert heuristic_budget("こんにちは！") == "minimal"
    assert heuristic_budget("thanks a lot") == "minimal"


def test_heuristic_never_returns_max():
    for q in ("hi", "capital of France?", "Prove the theorem",
              "x" * 500, "Prove " + "y" * 500):
        assert heuristic_budget(q) != "max"


def test_heuristic_word_boundary_avoids_improve_false_positive():
    q = "Can you help me improve my essay about summer holidays in the countryside?"
    assert heuristic_budget(q) == "medium"


def test_heuristic_short_question_is_low():
    assert heuristic_budget("capital of France?") == "low"
    assert heuristic_budget("91は素数ですか？") == "low"


def test_heuristic_long_prose_is_medium():
    q = ("Explain the historical background of the treaty and how it influenced "
         "trade relations in the following century across the region.")
    assert heuristic_budget(q) == "medium"


# ------------------------------------------------------------------ decide_budget

def test_decide_budget_explicit_modes_bypass_chat():
    chat = FakeChat(fn=_boom)  # 呼ばれたら失敗
    for name in LEVELS:
        assert decide_budget("q", chat, name) is BUDGETS[name]
    assert not chat.calls


def test_decide_budget_auto_accepts_new_levels():
    chat = FakeChat(responses=[json.dumps({"budget": "ultra"})])
    assert decide_budget("long hard problem", chat, "auto") is BUDGETS["ultra"]


def test_decide_budget_auto_uses_chat_classification():
    chat = FakeChat(responses=[json.dumps({"budget": "high"})])
    assert decide_budget("some question", chat, "auto") is BUDGETS["high"]
    assert len(chat.calls) == 1
    assert chat.calls[0]["fmt"] is not None  # スキーマ制約付き


def test_decide_budget_auto_falls_back_on_junk():
    budget = decide_budget("capital of France?", FakeChat(responses=["not json"]), "auto")
    assert budget is BUDGETS["low"]  # ヒューリスティックへフォールバック


def test_decide_budget_auto_falls_back_on_exception():
    budget = decide_budget("capital of France?", FakeChat(fn=_boom), "auto")
    assert budget is BUDGETS["low"]


def test_decide_budget_without_chat_is_heuristic():
    assert decide_budget("Prove the theorem now", None, "auto") is BUDGETS["high"]


# ------------------------------------------------------------------ reflect

def test_reflect_ok_keeps_draft_and_stops():
    chat = FakeChat(responses=["OK"])
    out = reflect("q", "the draft", chat, BUDGETS["high"])
    assert out == "the draft"
    assert len(chat.calls) == 1  # 収束後は追加ラウンドなし


def test_reflect_accepts_ok_with_punctuation():
    assert reflect("q", "d", FakeChat(responses=["ok."]), BUDGETS["high"]) == "d"


def test_reflect_applies_correction_then_converges():
    chat = FakeChat(responses=["corrected answer", "OK"])
    out = reflect("q", "wrong draft", chat, BUDGETS["high"])
    assert out == "corrected answer"
    assert len(chat.calls) == 2
    assert "corrected answer" in chat.calls[1]["prompt"]  # 2周目は改善版を再検査


def test_reflect_bounded_by_budget_reflections():
    chat = FakeChat(fn=lambda p: "always different " + str(len(p)))
    reflect("q", "draft", chat, BUDGETS["high"])
    assert len(chat.calls) == BUDGETS["high"].reflections


def test_reflect_zero_reflections_never_calls_chat():
    chat = FakeChat(fn=_boom)
    assert reflect("q", "draft", chat, BUDGETS["minimal"]) == "draft"
    assert not chat.calls


def test_reflect_max_budget_allows_four_rounds():
    chat = FakeChat(fn=lambda p: "always different " + str(len(p)))
    reflect("q", "draft", chat, BUDGETS["max"])
    assert len(chat.calls) == 4


def test_reflect_chat_failure_returns_current_best():
    chat = FakeChat(responses=["better"])  # 2回目は応答が尽きて例外 → 打ち切り
    out = reflect("q", "draft", chat, BUDGETS["high"])
    assert out == "better"


# ------------------------------------------------------------------ refine_answer

def test_refine_answer_minimal_mode_is_passthrough():
    def factory(budget):
        raise AssertionError("minimal は chat を作らない")
    assert refine_answer("q", "ans", "minimal", factory) == "ans"


def test_refine_answer_low_now_reflects_once():
    chats = []

    def factory(budget):
        chat = FakeChat(responses=["polished", "OK"])
        chats.append(budget.name)
        return chat

    assert refine_answer("q", "ans", "low", factory) == "polished"
    assert chats == ["low"]


def test_refine_answer_high_mode_reflects():
    chats = []

    def factory(budget):
        chat = FakeChat(responses=["improved", "OK"])
        chats.append((budget, chat))
        return chat

    out = refine_answer("q", "ans", "high", factory)
    assert out == "improved"
    assert chats[0][0] is BUDGETS["high"]  # 予算が factory に渡る


def test_refine_answer_auto_classifies_with_default_budget_chat():
    budgets_seen = []

    def factory(budget):
        budgets_seen.append(budget.name)
        if len(budgets_seen) == 1:  # 1回目 = 分類用 Chat
            return FakeChat(responses=[json.dumps({"budget": "high"})])
        return FakeChat(responses=["improved", "OK"])  # 2回目 = リフレクション用

    out = refine_answer("q", "ans", "auto", factory)
    assert budgets_seen == [DEFAULT_BUDGET, "high"]
    assert out == "improved"


def test_refine_answer_passes_through_error_and_empty():
    def factory(budget):
        raise AssertionError("呼ばれない")
    assert refine_answer("q", "__ERROR__: down", "high", factory) == "__ERROR__: down"
    assert refine_answer("q", "", "high", factory) == ""


def test_refine_answer_never_raises():
    def factory(budget):
        raise RuntimeError("factory broken")
    assert refine_answer("q", "ans", "high", factory) == "ans"


def test_budget_dataclass_is_frozen():
    import pytest
    with pytest.raises(Exception):
        Budget("x", 0, None, None).reflections = 5  # type: ignore[misc]


# ------------------------------------------------------------------ rounds floor

def test_thinking_rounds_floor_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_THINKING_BUDGET", raising=False)
    assert fugu_local._thinking_rounds_floor("q") == 0


def test_thinking_rounds_floor_by_level(monkeypatch):
    import fugu_local
    for mode, expected in (("minimal", 0), ("medium", 0), ("high", 1),
                           ("ultra", 2), ("max", 3), ("off", 0)):
        monkeypatch.setenv("FUGU_THINKING_BUDGET", mode)
        assert fugu_local._thinking_rounds_floor("q") == expected, mode


def test_thinking_rounds_floor_auto_is_heuristic_only(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_THINKING_BUDGET", "auto")
    monkeypatch.setattr(fugu_local, "ask",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("ラウンド下限判定で LLM を呼ばない")))
    assert fugu_local._thinking_rounds_floor("Prove the theorem now") == 1  # high


# ------------------------------------------------------------------ fugu_local hook

def test_apply_thinking_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_THINKING_BUDGET", raising=False)

    def never_ask(*a, **k):
        raise AssertionError("既定では LLM を呼ばない")

    monkeypatch.setattr(fugu_local, "ask", never_ask)
    assert fugu_local._apply_thinking("q", "raw <think>x</think>answer") == \
        "raw <think>x</think>answer"  # 既定経路は完全に不変(strip もしない)


def test_apply_thinking_hook_reflects_final_answer(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_THINKING_BUDGET", "high")
    monkeypatch.setattr(fugu_local, "CONDUCTOR", "fake-model")
    replies = ["corrected", "OK"]
    seen = []

    def fake_ask(model, messages, temperature, **kw):
        seen.append(kw)
        return replies.pop(0)

    monkeypatch.setattr(fugu_local, "ask", fake_ask)
    assert fugu_local._apply_thinking("q", "draft") == "corrected"
    assert seen[0].get("think") is True  # high 予算の think が ask() まで届く


def test_apply_thinking_off_value(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_THINKING_BUDGET", "off")
    monkeypatch.setattr(fugu_local, "ask",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert fugu_local._apply_thinking("q", "ans") == "ans"
