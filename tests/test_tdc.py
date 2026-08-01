# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_tdc のオフラインテスト（FakeChat + 実 SubprocessSandbox、LLM/Ollama 不要）。"""
from __future__ import annotations

import fugu_llm
import fugu_tdc

GOOD_CODE = "def add(a, b):\n    return a + b\n"
BAD_CODE = "def add(a, b):\n    return a - b\n"

VALID_TESTS = (
    "```python\n"
    "from solution import add\n\n"
    "def test_add_normal():\n"
    "    assert add(2, 3) == 5\n\n"
    "def test_add_edge_zero():\n"
    "    assert add(0, 0) == 0\n"
    "```"
)


# ---------- draft_tests ----------

def test_draft_extracts_valid_test_module():
    chat = fugu_llm.FakeChat(responses=[VALID_TESTS])
    src = fugu_tdc.draft_tests("add(a,b) returns the sum", GOOD_CODE, chat)
    assert src is not None
    assert "from solution import add" in src
    assert "requirements" in chat.calls[0]["prompt"].lower()


def test_draft_rejects_syntax_error():
    chat = fugu_llm.FakeChat(responses=["```python\ndef broken(:\n```"])
    assert fugu_tdc.draft_tests("req", GOOD_CODE, chat) is None


def test_draft_rejects_tests_not_referencing_solution():
    chat = fugu_llm.FakeChat(responses=["```python\ndef test_x():\n    assert True\n```"])
    assert fugu_tdc.draft_tests("req", GOOD_CODE, chat) is None


def test_draft_rejects_prose():
    chat = fugu_llm.FakeChat(responses=["I refuse to write tests today."])
    assert fugu_tdc.draft_tests("req", GOOD_CODE, chat) is None


def test_draft_returns_none_when_chat_raises():
    class DeadChat:
        def complete(self, prompt, *, system=None, fmt=None, temperature=0.2):
            raise RuntimeError("down")

    assert fugu_tdc.draft_tests("req", GOOD_CODE, DeadChat()) is None


# ---------- run_tdc ----------

def test_tdc_approves_good_code():
    chat = fugu_llm.FakeChat(responses=[VALID_TESTS])
    res = fugu_tdc.run_tdc(GOOD_CODE, "add(a,b) returns the sum", chat, max_fix=0)
    assert res.passed
    assert res.drafted
    assert res.attempts == 1
    assert res.code == GOOD_CODE


def test_tdc_rejects_bad_code_without_fix_budget():
    chat = fugu_llm.FakeChat(responses=[VALID_TESTS])
    res = fugu_tdc.run_tdc(BAD_CODE, "add(a,b) returns the sum", chat, max_fix=0)
    assert not res.passed
    assert res.drafted
    assert res.attempts == 1
    assert "test_add_normal" in res.report or "failed" in res.report


def test_tdc_fix_loop_repairs_solution():
    fixed_reply = f"```python\n{GOOD_CODE}```"
    chat = fugu_llm.FakeChat(responses=[VALID_TESTS, fixed_reply])
    res = fugu_tdc.run_tdc(BAD_CODE, "add(a,b) returns the sum", chat, max_fix=1)
    assert res.passed
    assert res.attempts == 2
    assert res.code.strip() == GOOD_CODE.strip()
    # 修正プロンプトに失敗ログが含まれる
    assert "solution.py" in chat.calls[1]["prompt"]


def test_tdc_draft_failure_reports_not_drafted():
    chat = fugu_llm.FakeChat(responses=["no code here"])
    res = fugu_tdc.run_tdc(GOOD_CODE, "req", chat)
    assert not res.passed
    assert not res.drafted


# ---------- fugu_local critique hook ----------

def test_critique_tdc_hook_rejects_failing_code(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_TDC", "1")
    monkeypatch.setattr(
        fugu_local, "_tdc_check", lambda q, a: (False, "TDC: drafted tests failed"))
    ok, issue = fugu_local.critique("q", "```python\nx=1\n```")
    assert not ok
    assert "TDC" in issue


def test_critique_tdc_hook_off_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_TDC", raising=False)
    called = {"n": 0}

    def spy(q, a):
        called["n"] += 1
        return (False, "should not run")

    monkeypatch.setattr(fugu_local, "_tdc_check", spy)
    monkeypatch.setattr(fugu_local, "_critic_judge",
                        lambda q, a, think=False: (True, ""))
    ok, _ = fugu_local.critique("q", "answer")
    assert ok
    assert called["n"] == 0


def test_tdc_check_returns_none_for_non_code(monkeypatch):
    import fugu_local
    assert fugu_local._tdc_check("q", "plain prose answer") is None
