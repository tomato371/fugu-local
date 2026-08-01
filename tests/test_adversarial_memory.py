# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""Doc E5 のオフラインテスト: adversarial_verify と記憶統合(consolidate)。"""
import json

from fugu_llm import FakeChat
from fugu_core.debate import SKEPTIC_LENSES, adversarial_verify
from fugu_core.memory import (
    Episode,
    LexicalMemory,
    consolidate,
    maybe_consolidate,
)

CONFIRM = json.dumps({"refuted": False, "reason": ""})
REFUTE = json.dumps({"refuted": True, "reason": "2+2 is 4 not 5"})


def _factory(replies):
    """index -> FakeChat。replies は懐疑者ごとの応答(足りない分は最後を使う)。"""
    def factory(index):
        reply = replies[min(index, len(replies) - 1)]
        if reply is RuntimeError:
            return FakeChat(fn=lambda p: (_ for _ in ()).throw(RuntimeError()))
        return FakeChat(responses=[reply])
    return factory


# ------------------------------------------------------------------ adversarial

def test_verify_survives_when_all_confirm():
    verified, reasons = adversarial_verify("q", "answer", _factory([CONFIRM]), n=3)
    assert verified is True and reasons == []


def test_verify_fails_on_majority_refutation():
    verified, reasons = adversarial_verify(
        "q", "2+2=5", _factory([REFUTE, REFUTE, CONFIRM]), n=3)
    assert verified is False
    assert len(reasons) == 2 and "2+2 is 4" in reasons[0]


def test_verify_minority_refutation_survives():
    verified, _ = adversarial_verify(
        "q", "answer", _factory([REFUTE, CONFIRM, CONFIRM]), n=3)
    assert verified is True  # 反証1/3 は過半数に満たない


def test_verify_junk_and_errors_count_as_refutation():
    verified, reasons = adversarial_verify(
        "q", "answer", _factory(["not json", RuntimeError, CONFIRM]), n=3)
    assert verified is False  # 判定不能2体 → 安全側で不採用
    assert any("skeptic unavailable" in r for r in reasons)


def test_verify_uses_distinct_lenses():
    prompts = []

    def factory(index):
        def record(prompt):
            return CONFIRM
        chat = FakeChat(fn=record)
        original = chat.complete

        def wrapped(prompt, **kw):
            prompts.append(kw.get("system") or "")
            return original(prompt, **kw)
        chat.complete = wrapped
        return chat

    adversarial_verify("q", "a", factory, n=3)
    for lens in SKEPTIC_LENSES:
        assert any(lens in system for system in prompts)


def test_verify_single_skeptic():
    verified, _ = adversarial_verify("q", "a", _factory([REFUTE]), n=1)
    assert verified is False
    verified, _ = adversarial_verify("q", "a", _factory([CONFIRM]), n=1)
    assert verified is True


# ------------------------------------------------------------------ fugu_local hook

def test_adversarial_hook_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_ADVERSARIAL", raising=False)
    assert fugu_local._adversarial_check("q", "a", True, None) == (True, None)


def test_adversarial_hook_skips_when_critic_already_failed(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_ADVERSARIAL", "1")
    monkeypatch.setattr(fugu_local, "ask",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert fugu_local._adversarial_check("q", "a", False, "bad") == (False, "bad")


def test_adversarial_hook_overturns_on_refutation(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_ADVERSARIAL", "1")
    monkeypatch.setattr(fugu_local, "CONDUCTOR", "fake-model")
    monkeypatch.setattr(fugu_local, "ask", lambda *a, **k: REFUTE)
    ok, issue = fugu_local._adversarial_check("q", "2+2=5", True, None)
    assert ok is False
    assert "2+2 is 4" in issue


def test_adversarial_hook_keeps_verified_answer(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_ADVERSARIAL", "1")
    monkeypatch.setattr(fugu_local, "CONDUCTOR", "fake-model")
    monkeypatch.setattr(fugu_local, "ask", lambda *a, **k: CONFIRM)
    assert fugu_local._adversarial_check("q", "a", True, None) == (True, None)


# ------------------------------------------------------------------ consolidate

def _filled_memory(tmp_path, kind="sandbox", count=25):
    memory = LexicalMemory(path=str(tmp_path / "mem.jsonl"),
                           now_fn=lambda: "t0")
    for i in range(count):
        memory.record(Episode(kind=kind, task=f"task {i}", outcome="failure",
                              lesson=f"lesson {i}"))
    return memory


def test_consolidate_below_threshold_is_noop(tmp_path):
    memory = _filled_memory(tmp_path, count=5)
    chat = FakeChat(fn=lambda p: (_ for _ in ()).throw(AssertionError("呼ばれない")))
    assert consolidate(memory, chat, threshold=20) is False
    assert len(memory.episodes) == 5


def test_consolidate_merges_old_episodes(tmp_path):
    memory = _filled_memory(tmp_path, count=25)
    chat = FakeChat(default="merged wisdom: always pin num_ctx")
    assert consolidate(memory, chat, threshold=20, keep_recent=5) is True
    assert len(memory.episodes) == 6  # 要約1 + 直近5
    summary = memory.episodes[0]
    assert summary.outcome == "summary"
    assert "merged wisdom" in summary.lesson
    assert "consolidated 20" in summary.task
    assert memory.episodes[-1].lesson == "lesson 24"  # 直近は生のまま
    # 永続化にも反映される(reload で同じ状態)
    reloaded = LexicalMemory(path=memory.path)
    assert len(reloaded.episodes) == 6
    assert reloaded.episodes[0].outcome == "summary"


def test_consolidate_only_touches_over_threshold_kinds(tmp_path):
    memory = _filled_memory(tmp_path, kind="sandbox", count=25)
    memory.record(Episode(kind="evolve", task="t", outcome="failure", lesson="keep"))
    consolidate(memory, FakeChat(default="summary"), threshold=20)
    evolve = [ep for ep in memory.episodes if ep.kind == "evolve"]
    assert len(evolve) == 1 and evolve[0].lesson == "keep"


def test_consolidate_chat_failure_is_nondestructive(tmp_path):
    memory = _filled_memory(tmp_path, count=25)
    boom = FakeChat(fn=lambda p: (_ for _ in ()).throw(RuntimeError()))
    assert consolidate(memory, boom, threshold=20) is False
    assert len(memory.episodes) == 25  # 無傷


def test_consolidate_empty_summary_is_nondestructive(tmp_path):
    memory = _filled_memory(tmp_path, count=25)
    assert consolidate(memory, FakeChat(default="   "), threshold=20) is False
    assert len(memory.episodes) == 25


def test_maybe_consolidate_disabled_by_default(monkeypatch, tmp_path):
    from fugu_core import memory as memory_mod
    monkeypatch.delenv("FUGU_MEMORY_CONSOLIDATE", raising=False)
    called = []
    monkeypatch.setattr(memory_mod, "consolidate",
                        lambda *a, **k: called.append(1))
    maybe_consolidate(_filled_memory(tmp_path, count=25))
    assert called == []


def test_maybe_consolidate_enabled_calls_consolidate(monkeypatch, tmp_path):
    from fugu_core import memory as memory_mod
    monkeypatch.setenv("FUGU_MEMORY_CONSOLIDATE", "1")
    called = []
    monkeypatch.setattr(memory_mod, "consolidate",
                        lambda mem, chat, **k: called.append(type(chat).__name__))
    memory_mod.maybe_consolidate(_filled_memory(tmp_path, count=25))
    assert called == ["AskChat"]


def test_maybe_consolidate_ignores_non_lexical(monkeypatch):
    from fugu_core import memory as memory_mod
    monkeypatch.setenv("FUGU_MEMORY_CONSOLIDATE", "1")
    monkeypatch.setattr(memory_mod, "consolidate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    maybe_consolidate(object())  # LexicalMemory 以外は何もしない