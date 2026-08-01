# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_verify のオフラインテスト(FakeChat 注入。モデル・GPU・ネット不要)。"""
import json
import threading

import pytest

import fugu_verify as V
from fugu_llm import FakeChat


def _judge(approved, confidence=0.8, reason="r"):
    return json.dumps({"approved": approved, "confidence": confidence,
                       "reason": reason})


# ------------------------------------------------------------- 個別の検証者

def test_llm_verifier_parses_verdict():
    v = V.LogicalVerifier(FakeChat([_judge(True, 0.9, "sound")]))
    r = v.verify("q", "a")
    assert r.approved is True and r.confidence == 0.9
    assert r.aspect == "logical" and r.reason == "sound"


def test_llm_verifier_clamps_confidence_and_defaults():
    v = V.FactualVerifier(FakeChat([json.dumps({"approved": True,
                                                "confidence": 7})]))
    assert v.verify("q", "a").confidence == 1.0
    v2 = V.FactualVerifier(FakeChat([json.dumps({"approved": False})]))
    assert v2.verify("q", "a").confidence == V.DEFAULT_CONFIDENCE


def test_llm_verifier_failure_is_disapproval_with_zero_confidence():
    v = V.ConstraintVerifier(FakeChat(["not json at all"]))
    r = v.verify("q", "a")
    assert r.approved is False and r.confidence == V.FAILURE_CONFIDENCE
    assert "unavailable" in r.reason
    assert r.contribution == 0.0


def test_context_is_included_in_prompt():
    chat = FakeChat([_judge(True)])
    V.FactualVerifier(chat).verify("q", "a", context="REFERENCE-TEXT")
    assert "REFERENCE-TEXT" in chat.calls[0]["prompt"]


def test_codetest_verifier_skips_non_code_answers():
    v = V.CodeTestVerifier(FakeChat(default="unused"))
    assert v.verify("q", "ただの散文の回答") is None


def test_computational_pot_path_uses_sandbox(monkeypatch):
    """PoT: 検算コードが CHECK_OK を出せば sandbox 実測として承認する。"""
    import fugu_sandbox

    class FakeSandbox:
        def run(self, code, files=None, timeout=None):
            class R:
                stdout, stderr, returncode = "CHECK_OK\n", "", 0
            return R()

    monkeypatch.setattr(fugu_sandbox, "get_sandbox", lambda *a, **k: FakeSandbox())
    chat = FakeChat(["```python\nprint('CHECK_OK')\n```"])
    r = V.ComputationalVerifier(chat).verify("1+1?", "2")
    assert r.approved is True and r.confidence == 0.95
    assert "sandbox" in r.reason


def test_computational_falls_back_to_llm_when_pot_na(monkeypatch):
    import fugu_sandbox

    class FakeSandbox:
        def run(self, code, files=None, timeout=None):
            class R:
                stdout, stderr, returncode = "CHECK_NA\n", "", 0
            return R()

    monkeypatch.setattr(fugu_sandbox, "get_sandbox", lambda *a, **k: FakeSandbox())
    chat = FakeChat(["```python\nprint('CHECK_NA')\n```",   # PoT 起草
                     _judge(False, 0.6, "wrong units")])    # LLM 審査へフォールバック
    r = V.ComputationalVerifier(chat).verify("q", "a")
    assert r.approved is False and r.reason == "wrong units"


def test_parse_pytest_counts():
    assert V._parse_pytest_counts("3 passed, 1 failed in 0.2s") == (3, 1)
    assert V._parse_pytest_counts("5 passed in 0.1s") == (5, 0)
    assert V._parse_pytest_counts("2 error") == (0, 2)


# ------------------------------------------------------------------- 集約

class _Fixed:
    """決定論のスタブ検証者。"""

    def __init__(self, aspect, approved, confidence, applicable=True):
        self.aspect = aspect
        self._r = V.VerifierResult(aspect, approved, confidence)
        self._applicable = applicable

    def verify(self, question, answer, context=""):
        return self._r if self._applicable else None


def test_score_is_weighted_mean_of_approved_confidence():
    vs = [_Fixed("a", True, 0.8), _Fixed("b", False, 0.9), _Fixed("c", True, 0.6)]
    rep = V.score_answer("q", "ans", vs, parallel=False)
    assert rep.score == pytest.approx((0.8 + 0.0 + 0.6) / 3)
    assert len(rep.results) == 3


def test_score_weights_and_inapplicable_exclusion():
    vs = [_Fixed("a", True, 1.0), _Fixed("b", True, 0.5),
          _Fixed("skip", True, 1.0, applicable=False)]
    rep = V.score_answer("q", "ans", vs, weights={"a": 3.0}, parallel=False)
    # (3×1.0 + 1×0.5) / (3+1); 適用不能の skip は分母にも入らない
    assert rep.score == pytest.approx(3.5 / 4)
    assert [r.aspect for r in rep.results] == ["a", "b"]


def test_score_empty_when_nothing_applicable():
    rep = V.score_answer("q", "ans", [_Fixed("x", True, 1.0, applicable=False)],
                         parallel=False)
    assert rep.score == 0.0 and rep.results == []


def test_breakdown_and_failing_reasons_keep_the_audit_trail():
    vs = [_Fixed("good", True, 0.9), _Fixed("bad", False, 0.7)]
    vs[1]._r.reason = "unit mismatch"
    rep = V.score_answer("q", "ans", vs, parallel=False)
    assert "good=✓0.90" in rep.breakdown() and "bad=✗0.70" in rep.breakdown()
    assert rep.failing_reasons() == ["[bad] unit mismatch"]


def test_verifiers_run_in_parallel_not_sequentially():
    """並列要件: 5 検証者が同時に走る(barrier が 5 者揃いで開く)ことを実証。"""
    barrier = threading.Barrier(5, timeout=5)

    class Slow:
        def __init__(self, i):
            self.aspect = f"v{i}"

        def verify(self, question, answer, context=""):
            barrier.wait()      # 逐次実行ならここでタイムアウトして落ちる
            return V.VerifierResult(self.aspect, True, 1.0)

    rep = V.score_answer("q", "a", [Slow(i) for i in range(5)], parallel=True)
    assert rep.score == 1.0 and len(rep.results) == 5


def test_score_answer_preserves_verifier_order_under_parallelism():
    vs = [_Fixed(f"v{i}", True, 0.5) for i in range(5)]
    rep = V.score_answer("q", "a", vs, parallel=True)
    assert [r.aspect for r in rep.results] == [f"v{i}" for i in range(5)]


# ---------------------------------------------------------------- best-of-N

def test_best_of_n_picks_highest_and_keeps_all_reports():
    scores = {"c0": 0.2, "c1": 0.9, "c2": 0.4}

    class ByAnswer:
        aspect = "x"

        def verify(self, question, answer, context=""):
            return V.VerifierResult("x", True, scores[answer])

    idx, best, reports = V.best_of_n("q", ["c0", "c1", "c2"], [ByAnswer()])
    assert idx == 1 and best.score == pytest.approx(0.9)
    assert [r.score for r in reports] == pytest.approx([0.2, 0.9, 0.4])


def test_best_of_n_tie_is_stable_first_wins():
    class Const:
        aspect = "x"

        def verify(self, question, answer, context=""):
            return V.VerifierResult("x", True, 0.5)

    idx, _, _ = V.best_of_n("q", ["a", "b"], [Const()])
    assert idx == 0


def test_best_of_n_rejects_empty():
    with pytest.raises(ValueError):
        V.best_of_n("q", [], [])


def test_mav_n_env(monkeypatch):
    monkeypatch.delenv("FUGU_MAV_N", raising=False)
    assert V.mav_n() == 4
    monkeypatch.setenv("FUGU_MAV_N", "7")
    assert V.mav_n() == 7
    monkeypatch.setenv("FUGU_MAV_N", "junk")
    assert V.mav_n() == 4
    monkeypatch.setenv("FUGU_MAV_N", "0")
    assert V.mav_n() == 1


def test_default_verifiers_covers_five_aspects():
    made = []

    def factory(aspect):
        made.append(aspect)
        return FakeChat(default=_judge(True))

    vs = V.default_verifiers(factory)
    assert [v.aspect for v in vs] == \
        ["logical", "computational", "factual", "constraint", "codetest"]
    assert made == ["logical", "computational", "factual", "constraint", "codetest"]
