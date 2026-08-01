# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_core.compressor のオフラインテスト + FUGU_COMPRESS=1 フック検証。"""
import json

from fugu_llm import FakeChat
from fugu_core.compressor import (
    StateDigest,
    compress_round,
    prune_context,
    render_digest,
)

DIGEST_JSON = json.dumps({
    "key_facts": ["x = 42", ""],
    "open_issues": ["edge case n=0"],
    "constraints": ["output must be JSON"],
    "draft_summary": "Draft computes x.",
})


def _boom(prompt):
    raise RuntimeError("model down")


# ------------------------------------------------------------------ prune_context

def test_prune_under_budget_is_unchanged():
    assert prune_context("short text", budget_chars=100) == "short text"


def test_prune_keeps_code_blocks_over_budget():
    code = "```python\n" + "x = 1\n" * 30 + "```"
    text = ("filler line\n" * 50) + code
    out = prune_context(text, budget_chars=100)
    assert code in out              # コードは予算超過でも完全保持
    assert out.count("filler line") < 50


def test_prune_keeps_constraint_lines():
    text = ("normal prose\n" * 60
            + "The output MUST be sorted.\n"
            + "結果は必ず昇順にすること\n"
            + "more prose\n" * 60)
    out = prune_context(text, budget_chars=200)
    assert "MUST be sorted" in out
    assert "必ず昇順" in out


def test_prune_fills_prose_greedily_in_order():
    lines = [f"line {i} with some padding text here" for i in range(20)]
    out = prune_context("\n".join(lines), budget_chars=120)
    assert "line 0" in out          # 先頭から詰める
    assert "line 19" not in out


def test_prune_never_splits_a_code_block():
    text = "a\n```py\nkeep_1\nkeep_2\n```\nb\n" + "pad\n" * 100
    out = prune_context(text, budget_chars=50)
    assert "keep_1\nkeep_2" in out


# ------------------------------------------------------------------ compress_round

def test_compress_round_parses_digest():
    chat = FakeChat(responses=[DIGEST_JSON])
    digest = compress_round("q", "long draft", chat, round_no=2)
    assert digest.round == 2
    assert digest.key_facts == ["x = 42"]      # 空要素は落とす
    assert digest.constraints == ["output must be JSON"]
    assert digest.draft_summary == "Draft computes x."
    assert chat.calls[0]["fmt"] is not None


def test_compress_round_fallback_on_junk_uses_pruned_reference():
    reference = "important draft body\nThe answer MUST be 42.\n" + "pad\n" * 500
    digest = compress_round("q", reference, FakeChat(default="not json"),
                            round_no=2, budget_chars=100)
    assert digest.key_facts == []
    assert "MUST be 42" in digest.draft_summary  # 制約行はフォールバックでも残る


def test_compress_round_fallback_on_exception():
    digest = compress_round("q", "the draft", FakeChat(fn=_boom))
    assert digest.draft_summary == "the draft"  # 予算内なら素通しトランケート


def test_compress_round_empty_digest_falls_back():
    empty = json.dumps({"key_facts": [], "open_issues": [],
                        "constraints": [], "draft_summary": ""})
    digest = compress_round("q", "the draft", FakeChat(responses=[empty]))
    assert digest.draft_summary == "the draft"


# ------------------------------------------------------------------ render_digest

def test_render_digest_full():
    digest = StateDigest(round=2, key_facts=["f1"], open_issues=["i1"],
                         constraints=["c1"], draft_summary="sum")
    text = render_digest(digest)
    assert text.startswith("## 状態ダイジェスト (round 2)")
    assert "### 確定事実\n- f1" in text
    assert "### 未解決の論点\n- i1" in text
    assert "### 守るべき制約\n- c1" in text
    assert text.rstrip().endswith("sum")


def test_render_digest_omits_empty_sections():
    text = render_digest(StateDigest(round=3, draft_summary="only summary"))
    assert "確定事実" not in text and "制約" not in text
    assert "only summary" in text


# ------------------------------------------------------------------ fugu_local hook

def test_compress_hook_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_COMPRESS", raising=False)
    assert fugu_local._compress_state("q", "ref", 2) == "ref"


def test_compress_hook_skips_round_one(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_COMPRESS", "1")
    monkeypatch.setattr(fugu_local, "ask",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert fugu_local._compress_state("q", "ref", 1) == "ref"


def test_compress_hook_renders_digest(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_COMPRESS", "1")
    monkeypatch.setattr(fugu_local, "CONDUCTOR", "fake-model")
    monkeypatch.setattr(fugu_local, "ask", lambda *a, **k: DIGEST_JSON)
    out = fugu_local._compress_state("q", "long reference", 2)
    assert "## 状態ダイジェスト (round 2)" in out
    assert "x = 42" in out


def test_compress_hook_failure_keeps_reference(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_COMPRESS", "1")
    monkeypatch.setattr(fugu_local, "CONDUCTOR", "fake-model")
    monkeypatch.setattr(fugu_local, "ask",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    # AskChat が例外 → compress_round の決定論的フォールバック(要約=素通し)に落ち、
    # reference の中身はダイジェスト内に必ず残る
    out = fugu_local._compress_state("q", "ref body", 2)
    assert "ref body" in out