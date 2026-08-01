# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_evolve.patcher のオフラインテスト(FakeChat + FakeGit Workspace)。"""
from fugu_llm import FakeChat
from fugu_evolve.cli import implement_proposal
from fugu_evolve.patcher import (
    apply_patch_to_text,
    implement_with_diff,
    parse_unified_diff,
)
from fugu_evolve.planner import Proposal
from fugu_evolve.workspace import BRANCH_PREFIX, Workspace

from tests.test_evolve_workspace import FakeGit

DIFF = """```diff
--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,3 @@
 def f():
-    return 1
+    return 2

```"""

FILE_V1 = "def f():\n    return 1\n\n"


def _ws(tmp_path):
    return Workspace(str(tmp_path), git=FakeGit(branch=BRANCH_PREFIX + "t-1"))


def _proposal(files=("mod.py",)):
    return Proposal(title="bump", category="fix", target_files=list(files),
                    rationale="return 2 instead")


# ------------------------------------------------------------------ parse

def test_parse_extracts_files_and_hunks():
    patches = parse_unified_diff(DIFF)
    assert len(patches) == 1
    assert patches[0].path == "mod.py"          # b/ 接頭辞は正規化
    hunk = patches[0].hunks[0]
    assert hunk.old_lines == ["def f():", "    return 1", ""]
    assert hunk.new_lines == ["def f():", "    return 2", ""]


def test_parse_multiple_files():
    text = ("--- a/x.py\n+++ b/x.py\n@@\n-a\n+b\n"
            "--- a/y.py\n+++ b/y.py\n@@\n-c\n+d\n")
    patches = parse_unified_diff(text)
    assert [p.path for p in patches] == ["x.py", "y.py"]


def test_parse_junk_is_empty():
    assert parse_unified_diff("no diff here") == []
    assert parse_unified_diff("") == []


def test_parse_skips_no_newline_marker():
    text = "--- a/x\n+++ b/x\n@@\n-old\n+new\n\\ No newline at end of file\n"
    hunk = parse_unified_diff(text)[0].hunks[0]
    assert hunk.old_lines == ["old"] and hunk.new_lines == ["new"]


# ------------------------------------------------------------------ apply

def test_apply_replaces_unique_block():
    hunks = parse_unified_diff(DIFF)[0].hunks
    out = apply_patch_to_text(FILE_V1, hunks)
    assert out == "def f():\n    return 2\n\n"


def test_apply_missing_context_fails():
    hunks = parse_unified_diff(DIFF)[0].hunks
    assert apply_patch_to_text("completely different\n", hunks) is None


def test_apply_ambiguous_block_fails():
    text = "x = 1\nx = 1\n"
    hunks = parse_unified_diff("--- a/f\n+++ b/f\n@@\n-x = 1\n+x = 2\n")[0].hunks
    assert apply_patch_to_text(text, hunks) is None


def test_apply_whitespace_relaxed_match():
    # 実ファイル側のインデントが半角1つ多くても、一意なら緩和マッチで当てる
    text = "def f():\n     return 1\n"
    hunks = parse_unified_diff(
        "--- a/f\n+++ b/f\n@@\n def f():\n-    return 1\n+    return 2\n")[0].hunks
    out = apply_patch_to_text(text, hunks)
    assert out is not None and "return 2" in out


def test_apply_sequential_hunks():
    text = "a\nb\nc\nd\n"
    diff = ("--- a/f\n+++ b/f\n"
            "@@\n-a\n+A\n"
            "@@\n-d\n+D\n")
    out = apply_patch_to_text(text, parse_unified_diff(diff)[0].hunks)
    assert out == "A\nb\nc\nD\n"


# ------------------------------------------------------------------ implement_with_diff

def test_implement_with_diff_applies(tmp_path):
    (tmp_path / "mod.py").write_text(FILE_V1, encoding="utf-8")
    ws = _ws(tmp_path)
    assert implement_with_diff(FakeChat(default=DIFF), ws, _proposal()) is True
    assert (tmp_path / "mod.py").read_text(encoding="utf-8") == \
        "def f():\n    return 2\n\n"


def test_implement_with_diff_rejects_unlisted_path(tmp_path):
    (tmp_path / "mod.py").write_text(FILE_V1, encoding="utf-8")
    (tmp_path / "other.py").write_text("x = 1\n", encoding="utf-8")
    diff = "--- a/other.py\n+++ b/other.py\n@@\n-x = 1\n+x = 2\n"
    ws = _ws(tmp_path)
    assert implement_with_diff(FakeChat(default=diff), ws, _proposal()) is False
    assert (tmp_path / "other.py").read_text(encoding="utf-8") == "x = 1\n"


def test_implement_with_diff_all_or_nothing(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 1\n", encoding="utf-8")
    diff = ("--- a/a.py\n+++ b/a.py\n@@\n-a = 1\n+a = 2\n"
            "--- a/b.py\n+++ b/b.py\n@@\n-MISSING\n+b = 2\n")  # b 側は不一致
    ws = _ws(tmp_path)
    result = implement_with_diff(FakeChat(default=diff), ws,
                                 _proposal(files=("a.py", "b.py")))
    assert result is False
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "a = 1\n"  # 部分適用なし


def test_implement_with_diff_compile_guard(tmp_path):
    (tmp_path / "mod.py").write_text(FILE_V1, encoding="utf-8")
    diff = "--- a/mod.py\n+++ b/mod.py\n@@\n-def f():\n+def f(:\n"
    ws = _ws(tmp_path)
    assert implement_with_diff(FakeChat(default=diff), ws, _proposal()) is False
    assert (tmp_path / "mod.py").read_text(encoding="utf-8") == FILE_V1


def test_implement_with_diff_missing_file_is_false(tmp_path):
    ws = _ws(tmp_path)
    assert implement_with_diff(FakeChat(default=DIFF), ws, _proposal()) is False


# ------------------------------------------------------------------ cli 統合

def test_implement_proposal_prefers_diff(tmp_path):
    (tmp_path / "mod.py").write_text(FILE_V1, encoding="utf-8")
    ws = _ws(tmp_path)
    chat = FakeChat(default=DIFF)  # diff だけで完結(全置換プロンプトは不要)
    assert implement_proposal(chat, ws, _proposal()) is True
    assert "return 2" in (tmp_path / "mod.py").read_text(encoding="utf-8")
    assert len(chat.calls) == 1


def test_implement_proposal_falls_back_to_whole_file(tmp_path):
    import json
    (tmp_path / "mod.py").write_text(FILE_V1, encoding="utf-8")
    ws = _ws(tmp_path)

    def chatter(prompt):
        if "unified diff" in prompt:
            return "cannot produce a diff, sorry"  # diff 経路は失敗
        return json.dumps({"edits": [{"path": "mod.py", "code": "x = 9\n"}]})

    assert implement_proposal(FakeChat(fn=chatter), ws, _proposal()) is True
    assert (tmp_path / "mod.py").read_text(encoding="utf-8") == "x = 9\n"
