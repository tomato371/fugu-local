# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_core.subagents のオフラインテスト + FUGU_DYNAMIC_SUBAGENTS=1 フック検証。"""
import json

from fugu_llm import FakeChat
from fugu_core.subagents import RoleChat, SubagentSpec, design_subagent, spawn

SPEC_JSON = json.dumps({
    "role": "hydrologist",
    "system_prompt": "You are a flood hydrology expert. Check units.",
})


def _boom(prompt):
    raise RuntimeError("model down")


# ------------------------------------------------------------------ design

def test_design_subagent_parses_spec():
    spec = design_subagent("洪水の水深予測を説明して", FakeChat(responses=[SPEC_JSON]))
    assert spec.role == "hydrologist"
    assert "flood hydrology" in spec.system_prompt


def test_design_subagent_rejects_junk_and_failures():
    assert design_subagent("q", FakeChat(default="not json")) is None
    assert design_subagent("q", FakeChat(fn=_boom)) is None
    empty = json.dumps({"role": " ", "system_prompt": "x"})
    assert design_subagent("q", FakeChat(responses=[empty])) is None


# ------------------------------------------------------------------ RoleChat / spawn

def test_rolechat_prepends_system_prompt():
    inner = FakeChat(default="expert answer")
    agent = spawn(SubagentSpec(role="r", system_prompt="BE THE EXPERT"),
                  lambda: inner)
    assert agent.complete("q") == "expert answer"
    assert inner.calls[0]["system"] == "BE THE EXPERT"


def test_rolechat_merges_caller_system():
    inner = FakeChat(default="ok")
    RoleChat(SubagentSpec(role="r", system_prompt="ROLE"),
             inner).complete("q", system="CALLER RULES")
    assert inner.calls[0]["system"] == "ROLE\n\nCALLER RULES"


def test_rolechat_passes_fmt_and_temperature():
    inner = FakeChat(default="{}")
    RoleChat(SubagentSpec(role="r", system_prompt="ROLE"), inner).complete(
        "q", fmt={"type": "object"}, temperature=0.7)
    assert inner.calls[0]["fmt"] == {"type": "object"}
    assert inner.calls[0]["temperature"] == 0.7


# ------------------------------------------------------------------ fugu_local hook

BASE = [("m1", "base answer")]


def test_dynamic_hook_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_DYNAMIC_SUBAGENTS", raising=False)
    assert fugu_local._dynamic_specialist("q", BASE) is BASE


def test_dynamic_hook_appends_specialist_proposal(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_DYNAMIC_SUBAGENTS", "1")
    monkeypatch.setattr(fugu_local, "CONDUCTOR", "fake-model")
    replies = [SPEC_JSON, "the specialist take"]

    def fake_ask(model, messages, temperature, **kw):
        return replies.pop(0)

    monkeypatch.setattr(fugu_local, "ask", fake_ask)
    out = fugu_local._dynamic_specialist("洪水の水深は?", BASE, models=["m1"])
    assert out[:-1] == BASE
    assert out[-1] == ("dynamic:hydrologist", "the specialist take")


def test_dynamic_hook_design_failure_keeps_proposals(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_DYNAMIC_SUBAGENTS", "1")
    monkeypatch.setattr(fugu_local, "CONDUCTOR", "fake-model")
    monkeypatch.setattr(fugu_local, "ask", lambda *a, **k: "not json")
    assert fugu_local._dynamic_specialist("q", BASE) is BASE


def test_dynamic_hook_error_answer_not_added(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_DYNAMIC_SUBAGENTS", "1")
    monkeypatch.setattr(fugu_local, "CONDUCTOR", "fake-model")
    replies = [SPEC_JSON, "__ERROR__: down"]
    monkeypatch.setattr(fugu_local, "ask",
                        lambda *a, **k: replies.pop(0))
    out = fugu_local._dynamic_specialist("q", BASE)
    assert out == BASE
