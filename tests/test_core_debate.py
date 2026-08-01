# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_core.debate のオフラインテスト + FUGU_DEBATE=1 フック検証。"""
import json

from fugu_llm import FakeChat
from fugu_core.debate import (
    ScoreMatrix,
    classify_domain,
    debate,
    get_default_matrix,
    reset_default_matrix,
    should_debate,
)


# ------------------------------------------------------------------ classify_domain

def test_classify_domain_by_cues():
    assert classify_domain("この Python コードのバグを直して") == "code"
    assert classify_domain("Prove the theorem about primes") == "math"
    assert classify_domain("このテーマでエッセイを書いて") == "writing"
    assert classify_domain("フランスの首都はどこ") == "factual"
    assert classify_domain("Compare these two approaches") == "reasoning"


# ------------------------------------------------------------------ ScoreMatrix

def test_score_matrix_record_and_weights():
    matrix = ScoreMatrix()
    matrix.record("m1", "code", True)
    matrix.record("m1", "code", True)
    matrix.record("m2", "code", False)
    weights = matrix.weights("code")
    assert weights["m1"] == (2 + 1) / (2 + 2)   # ラプラス平滑化
    assert weights["m2"] == (0 + 1) / (1 + 2)
    assert weights["m1"] > weights["m2"]


def test_score_matrix_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "scores.json")
    matrix = ScoreMatrix(path=path)
    matrix.record("m1", "math", True)
    reloaded = ScoreMatrix(path=path)
    assert reloaded.scores["m1"]["math"] == [1, 1]


def test_score_matrix_tolerates_corrupt_file(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text("{broken", encoding="utf-8")
    assert ScoreMatrix(path=str(path)).scores == {}


def test_score_matrix_skips_malformed_entries(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(
        {"good": {"code": [1, 2]}, "bad": "junk",
         "half": {"code": [1, "x"]}}), encoding="utf-8")
    matrix = ScoreMatrix(path=str(path))
    assert matrix.scores == {"good": {"code": [1, 2]}}


def test_select_models_ranks_by_domain_fit():
    matrix = ScoreMatrix()
    for _ in range(3):
        matrix.record("coder", "code", True)
        matrix.record("writer", "code", False)
    ranked = matrix.select_models("code", ["writer", "coder", "unknown"], k=2)
    assert ranked[0] == "coder"
    assert len(ranked) == 2


def test_select_models_unknown_models_keep_order():
    ranked = ScoreMatrix().select_models("code", ["a", "b", "c"], k=3)
    assert ranked == ["a", "b", "c"]  # 全員中立 0.5 → 元順維持


def test_get_default_matrix_honors_env(tmp_path, monkeypatch):
    path = str(tmp_path / "scores.json")
    monkeypatch.setenv("FUGU_SCORE_PATH", path)
    reset_default_matrix()
    try:
        matrix = get_default_matrix()
        assert matrix.path == path
        assert get_default_matrix() is matrix
    finally:
        reset_default_matrix()


# ------------------------------------------------------------------ should_debate

def test_should_debate_when_answers_diverge():
    proposals = [("m1", "The answer is 42 because of the calculation"),
                 ("m2", "完全に異なる内容で首都はパリだと述べる回答")]
    assert should_debate(proposals) is True


def test_should_not_debate_when_answers_agree():
    same = "The answer is 42 because 6 times 7 equals 42"
    assert should_debate([("m1", same), ("m2", same + " indeed")]) is False


def test_should_not_debate_with_fewer_than_two_valid():
    assert should_debate([("m1", "only one answer")]) is False
    assert should_debate([("m1", "ok"), ("m2", "__ERROR__: down")]) is False
    assert should_debate([]) is False


# ------------------------------------------------------------------ debate

def _factory_recorder(record):
    def factory(model):
        def reply(prompt):
            record.append((model, prompt))
            return f"revised by {model}"
        return FakeChat(fn=reply)
    return factory


def test_debate_revises_each_proposal_with_rival_context():
    record = []
    proposals = [("m1", "answer one"), ("m2", "answer two")]
    out = debate("q", proposals, _factory_recorder(record), turns=1)
    assert out == [("m1", "revised by m1"), ("m2", "revised by m2")]
    m1_prompt = record[0][1]
    assert "answer two" in m1_prompt      # ライバル回答が見える
    assert "answer one" in m1_prompt      # 自分の回答も見える


def test_debate_converges_and_stops_early():
    calls = []

    def factory(model):
        def reply(prompt):
            calls.append(model)
            return "fixed answer"  # 2ターン目は全員無変化 → 収束
        return FakeChat(fn=reply)

    debate("q", [("m1", "a"), ("m2", "b")], factory, turns=5)
    # ターン1で両者 "fixed answer"、ターン2でも同じ → 2ターンで打ち切り
    assert len(calls) == 4


def test_debate_failure_keeps_original_answer():
    def factory(model):
        if model == "m1":
            return FakeChat(fn=lambda p: (_ for _ in ()).throw(RuntimeError()))
        return FakeChat(fn=lambda p: "revised")

    out = debate("q", [("m1", "original"), ("m2", "b")], factory, turns=1)
    assert out[0] == ("m1", "original")
    assert out[1] == ("m2", "revised")


def test_debate_empty_reply_keeps_original():
    out = debate("q", [("m1", "keep me"), ("m2", "other")],
                 lambda model: FakeChat(default="   "), turns=1)
    assert out[0] == ("m1", "keep me")


# ------------------------------------------------------------------ fugu_local hooks

DIVERGENT = [("m1", "The answer is 42 from calculation"),
             ("m2", "全然違う話をしている日本語の回答です")]


def test_debate_hook_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_DEBATE", raising=False)
    assert fugu_local._debate_proposals("q", DIVERGENT) is DIVERGENT


def test_debate_hook_runs_debate_when_diverged(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_DEBATE", "1")
    monkeypatch.setattr(fugu_local, "ask",
                        lambda model, *a, **k: f"revised by {model}")
    out = fugu_local._debate_proposals("q", DIVERGENT)
    assert out == [("m1", "revised by m1"), ("m2", "revised by m2")]


def test_debate_hook_skips_agreeing_proposals(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_DEBATE", "1")
    monkeypatch.setattr(fugu_local, "ask",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    same = [("m1", "identical answer text"), ("m2", "identical answer text")]
    assert fugu_local._debate_proposals("q", same) is same


def test_debate_record_hook(monkeypatch, tmp_path):
    import fugu_local
    from fugu_core import debate as debate_mod
    monkeypatch.setenv("FUGU_DEBATE", "1")
    matrix = ScoreMatrix()
    monkeypatch.setattr(debate_mod, "get_default_matrix", lambda: matrix)
    fugu_local._debate_record("コードのバグを直して", ["m1", "m2"], True)
    assert matrix.scores["m1"]["code"] == [1, 1]
    assert matrix.scores["m2"]["code"] == [1, 1]


def test_rank_models_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_DEBATE", raising=False)
    models = ["a", "b"]
    assert fugu_local._rank_models_by_domain("q", models) is models


def test_rank_models_reorders_by_domain_fit(monkeypatch):
    import fugu_local
    from fugu_core import debate as debate_mod
    monkeypatch.setenv("FUGU_DEBATE", "1")
    matrix = ScoreMatrix()
    for _ in range(4):
        matrix.record("coder-model", "code", True)
        matrix.record("prose-model", "code", False)
    monkeypatch.setattr(debate_mod, "get_default_matrix", lambda: matrix)
    ranked = fugu_local._rank_models_by_domain(
        "この Python コードのバグを直して", ["prose-model", "coder-model"])
    assert ranked == ["coder-model", "prose-model"]  # 適性順に並べ替え(集合不変)


def test_rank_models_no_history_keeps_order(monkeypatch):
    import fugu_local
    from fugu_core import debate as debate_mod
    monkeypatch.setenv("FUGU_DEBATE", "1")
    monkeypatch.setattr(debate_mod, "get_default_matrix", lambda: ScoreMatrix())
    models = ["a", "b", "c"]
    assert fugu_local._rank_models_by_domain("コードを直して", models) == models


def test_rank_models_failure_keeps_order(monkeypatch):
    import fugu_local
    from fugu_core import debate as debate_mod
    monkeypatch.setenv("FUGU_DEBATE", "1")
    monkeypatch.setattr(debate_mod, "get_default_matrix",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    models = ["a", "b"]
    assert fugu_local._rank_models_by_domain("q", models) is models


def test_debate_record_hook_disabled_by_default(monkeypatch):
    import fugu_local
    from fugu_core import debate as debate_mod
    monkeypatch.delenv("FUGU_DEBATE", raising=False)
    monkeypatch.setattr(debate_mod, "get_default_matrix",
                        lambda: (_ for _ in ()).throw(AssertionError()))
    fugu_local._debate_record("q", ["m1"], True)  # 何も起きない
