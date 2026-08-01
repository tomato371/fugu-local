# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""フラグ相互作用の統合テスト(オフライン)。

opt-in env フラグが12個を超え、単体テストはあっても「同時有効」の相互作用は
未検証だった。ここでは実物の fugu_answer を、コア段(get_proposals/aggregate/
critique/verify_single/conduct)だけ決定論的に差し替えて走らせ、フック群
(dynamic subagent → debate → aggregate → critic → adversarial → thinking →
memory → compress → tasks)が同時有効でも完走し、順序が保たれることを固定する。
"""
import json

import pytest

import fugu_local

ALL_FLAGS = ("FUGU_DEBATE", "FUGU_DYNAMIC_SUBAGENTS", "FUGU_ADVERSARIAL",
             "FUGU_THINKING_BUDGET", "FUGU_MEMORY", "FUGU_COMPRESS",
             "FUGU_TASKS", "FUGU_TOOL_CALLING", "FUGU_SPECULATE")

MOA_PLAN = {"mode": "moa", "task_type": "knowledge",
            "selected_proposers": ["m1", "m2"], "rounds": 1,
            "image_only": False, "make_pptx": False,
            "use_image_generation": False, "search_required": False}

DIVERGENT = [("m1", "proposal alpha tokens entirely about topic one"),
             ("m2", "completely different beta words on another subject")]


def make_fake_ask(log):
    """フック内部の AskChat 呼び出しを label で分岐する決定論的スタブ。"""

    def fake_ask(model, messages, temperature, think=None, fmt=None,
                 label=None, num_predict=None, num_ctx=None, images=None):
        log.append((label, model))
        if label == "subagent-design":
            return json.dumps({"role": "specialist",
                               "system_prompt": "be the expert"})
        if label == "subagent":
            return "unique specialist angle zqx"
        if label == "debate":
            return f"revised by {model}"
        if label and label.startswith("skeptic"):
            return json.dumps({"refuted": False, "reason": ""})
        if label == "thinking":
            return "OK"
        if label == "compress":
            return json.dumps({"key_facts": ["fact"], "open_issues": [],
                               "constraints": [], "draft_summary": "digest"})
        if label == "tasks":
            return json.dumps({"subtasks": [
                {"subject": "sub one"},
                {"subject": "sub two", "depends_on": [1]}]})
        return "generic"

    return fake_ask


@pytest.fixture()
def wired(monkeypatch, tmp_path):
    """コア段を決定論化し、永続化先を tmp に隔離した実行環境。"""
    for flag in ALL_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("FUGU_MEMORY_PATH", str(tmp_path / "mem.jsonl"))
    monkeypatch.setenv("FUGU_SCORE_PATH", str(tmp_path / "scores.json"))
    monkeypatch.setenv("FUGU_TASKS_DIR", str(tmp_path / "boards"))
    from fugu_core import memory as memory_mod
    from fugu_core import debate as debate_mod
    memory_mod.reset_default_memory()
    debate_mod.reset_default_matrix()

    log = []
    monkeypatch.setattr(fugu_local, "ask", make_fake_ask(log))
    monkeypatch.setattr(fugu_local, "CONDUCTOR", "fake-conductor")
    monkeypatch.setattr(fugu_local, "get_proposals",
                        lambda models, q, ref, hint, history=None:
                        list(DIVERGENT))
    monkeypatch.setattr(fugu_local, "aggregate",
                        lambda q, proposals: "AGGREGATED final answer")
    monkeypatch.setattr(fugu_local, "code_check", lambda fin: None)
    monkeypatch.setattr(fugu_local, "critique", lambda q, fin: (True, ""))
    monkeypatch.setattr(fugu_local, "verify_single", lambda q, a: (True, ""))
    monkeypatch.setattr(fugu_local, "conduct",
                        lambda q, history=None, office_attached=False:
                        (dict(MOA_PLAN), ""))
    yield log
    memory_mod.reset_default_memory()
    debate_mod.reset_default_matrix()


def _labels(log):
    return [label for label, _ in log]


# ------------------------------------------------------------------ 基準線

def test_all_flags_off_runs_no_hooks(wired):
    out = fugu_local.fugu_answer("質問", plan=dict(MOA_PLAN))
    assert out == "AGGREGATED final answer"
    assert _labels(wired) == []  # フック由来の LLM 呼び出しはゼロ(全て opt-in)


# ------------------------------------------------------------------ 全部盛り(moa)

def test_full_stack_moa_completes_with_all_hooks(wired, monkeypatch):
    monkeypatch.setenv("FUGU_DEBATE", "1")
    monkeypatch.setenv("FUGU_DYNAMIC_SUBAGENTS", "1")
    monkeypatch.setenv("FUGU_ADVERSARIAL", "1")
    monkeypatch.setenv("FUGU_THINKING_BUDGET", "high")
    monkeypatch.setenv("FUGU_MEMORY", "1")
    monkeypatch.setenv("FUGU_COMPRESS", "1")

    out = fugu_local.fugu_answer("知識の質問です。詳しく説明して", plan=dict(MOA_PLAN))
    assert out == "AGGREGATED final answer"  # thinking は OK 収束で原文維持

    labels = _labels(wired)
    # 各フックが1度は発火している
    assert "subagent-design" in labels and "subagent" in labels
    assert "debate" in labels
    assert any(str(l).startswith("skeptic") for l in labels)
    assert "thinking" in labels
    assert "compress" in labels
    # 順序: 動的専門家 → 討論 → (集約後) 懐疑者 → 思考予算
    assert labels.index("subagent") < labels.index("debate")
    assert labels.index("debate") < labels.index("skeptic0")
    assert labels.index("skeptic0") < labels.index("thinking")


def test_full_stack_records_score_matrix(wired, monkeypatch, tmp_path):
    monkeypatch.setenv("FUGU_DEBATE", "1")
    fugu_local.fugu_answer("知識の質問", plan=dict(MOA_PLAN))
    scores = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))
    assert "m1" in scores and "m2" in scores  # critique 合格が行列に記録される


def test_adversarial_refutation_triggers_extra_round(wired, monkeypatch):
    monkeypatch.setenv("FUGU_ADVERSARIAL", "1")
    original = fugu_local.ask
    state = {"round": 0}

    def refuting_ask(model, messages, temperature, **kw):
        label = kw.get("label")
        if label and label.startswith("skeptic"):
            state["round"] += 1
            if state["round"] <= 3:  # 1巡目の懐疑者3体は全員反証 → 追加ラウンド
                return json.dumps({"refuted": True, "reason": "wrong"})
            return json.dumps({"refuted": False, "reason": ""})
        return original(model, messages, temperature, **kw)

    fugu_local.ask = refuting_ask
    try:
        out = fugu_local.fugu_answer("質問", plan=dict(MOA_PLAN))
    finally:
        fugu_local.ask = original
    assert out == "AGGREGATED final answer"
    assert state["round"] == 6  # 反証で2巡目が走り、2巡目は生存


# ------------------------------------------------------------------ 全部盛り(tasks 統合)

def test_full_stack_with_task_board(wired, monkeypatch):
    monkeypatch.setenv("FUGU_TASKS", "1")
    monkeypatch.setenv("FUGU_DEBATE", "1")
    monkeypatch.setenv("FUGU_DYNAMIC_SUBAGENTS", "1")
    monkeypatch.setenv("FUGU_THINKING_BUDGET", "medium")

    out = fugu_local.fugu_answer("複数手順の依頼", plan=None)
    labels = _labels(wired)
    assert labels.count("tasks") == 1            # 分解は1回
    assert "## sub one" in out and "## sub two" in out
    # サブタスクごとに full-stack フックが回る(subagent が2回以上)
    assert labels.count("subagent") >= 2
    assert "未完了サブタスク" not in out          # 全サブタスク完走


def test_task_board_reentrancy_with_flags(wired, monkeypatch):
    monkeypatch.setenv("FUGU_TASKS", "1")
    fugu_local.fugu_answer("依頼", plan=None)
    assert fugu_local._TASKS_ACTIVE is False     # ガードは必ず戻る
    assert _labels(wired).count("tasks") == 1    # サブタスク内で再分解しない


# ------------------------------------------------------------------ 単体モード

def test_single_mode_with_ranking_and_thinking(wired, monkeypatch, tmp_path):
    monkeypatch.setenv("FUGU_DEBATE", "1")
    monkeypatch.setenv("FUGU_THINKING_BUDGET", "high")
    from fugu_core import debate as debate_mod
    matrix = debate_mod.ScoreMatrix(path=str(tmp_path / "scores.json"))
    for _ in range(4):
        matrix.record("m2", "reasoning", True)
        matrix.record("m1", "reasoning", False)
    monkeypatch.setattr(debate_mod, "get_default_matrix", lambda: matrix)
    single_plan = dict(MOA_PLAN, mode="single")

    answered = {}
    original = fugu_local.ask

    def tracking_ask(model, messages, temperature, **kw):
        if kw.get("label") == "single":
            answered["model"] = model
            return f"single answer from {model}"
        return original(model, messages, temperature, **kw)

    fugu_local.ask = tracking_ask
    try:
        out = fugu_local.fugu_answer("一般的な質問をします", plan=single_plan)
    finally:
        fugu_local.ask = original
    assert answered["model"] == "m2"             # 勝率順で m2 が回答者になる
    assert out == "single answer from m2"        # thinking は OK 収束で不変
    assert "thinking" in _labels(wired)
