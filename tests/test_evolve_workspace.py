"""fugu_evolve.workspace のテスト — FakeGit 単体 + tmp_path 実 git リポ E2E。

実 git E2E は tmp_path 上に `git init` した使い捨てリポジトリのみを対象にする
(このリポジトリ自体には一切触れない)。git コマンドが無い環境では skip。
"""
import shutil
import subprocess

import pytest

from fugu_evolve.workspace import (
    BRANCH_PREFIX,
    GitError,
    RealGit,
    Workspace,
    _slugify,
)


class FakeGit:
    """git を模倣する注入クライアント。ブランチ状態と実行履歴だけ持つ。"""

    def __init__(self, branch="main", dirty=False):
        self.branch = branch
        self.dirty = dirty
        self.calls = []

    def run(self, *args):
        self.calls.append(args)
        if args[:2] == ("status", "--porcelain"):
            return " M fugu_local.py\n" if self.dirty else ""
        if args[:3] == ("rev-parse", "--abbrev-ref", "HEAD"):
            return self.branch + "\n"
        if args[0] == "checkout" and args[1] == "-b":
            self.branch = args[2]
            return ""
        if args[0] == "checkout":
            self.branch = args[1]
            return ""
        if args[:2] == ("rev-parse", "HEAD"):
            return "abc123\n"
        if args[0] == "diff":
            return "+ diff body\n"
        return ""


# ------------------------------------------------------------------ slug

def test_slugify_normalizes():
    assert _slugify("Fix: failing tests!!") == "fix-failing-tests"
    assert _slugify("") == "change"
    assert len(_slugify("x" * 100)) <= 40


# ------------------------------------------------------------------ FakeGit 単体

def test_ensure_clean_raises_on_dirty(tmp_path):
    ws = Workspace(str(tmp_path), git=FakeGit(dirty=True))
    with pytest.raises(GitError, match="not clean"):
        ws.ensure_clean()


def test_create_branch_names_and_records_base(tmp_path):
    git = FakeGit(branch="main")
    ws = Workspace(str(tmp_path), git=git, now_fn=lambda: "20260801-120000")
    name = ws.create_branch("Fix failing tests")
    assert name == "auto-evolve/fix-failing-tests-20260801-120000"
    assert git.branch == name
    assert ws.base == "main"


def test_create_branch_refuses_dirty_tree(tmp_path):
    ws = Workspace(str(tmp_path), git=FakeGit(dirty=True))
    with pytest.raises(GitError):
        ws.create_branch("t")


def test_apply_edit_requires_evolve_branch(tmp_path):
    ws = Workspace(str(tmp_path), git=FakeGit(branch="main"))
    with pytest.raises(GitError, match="auto-evolve"):
        ws.apply_edit("a.py", "x = 1\n")


def test_apply_edit_writes_inside_repo(tmp_path):
    ws = Workspace(str(tmp_path), git=FakeGit(branch=BRANCH_PREFIX + "t-1"))
    ws.apply_edit("pkg/mod.py", "x = 1\n")
    assert (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"


def test_apply_edit_rejects_path_escape(tmp_path):
    ws = Workspace(str(tmp_path), git=FakeGit(branch=BRANCH_PREFIX + "t-1"))
    with pytest.raises(GitError, match="escapes"):
        ws.apply_edit("../outside.py", "x")


def test_commit_and_destructive_ops_guarded_off_branch(tmp_path):
    ws = Workspace(str(tmp_path), git=FakeGit(branch="main"))
    with pytest.raises(GitError):
        ws.commit("msg")
    with pytest.raises(GitError):
        ws.rollback()
    with pytest.raises(GitError):
        ws.merge_to_main()


def test_rollback_sequence_on_evolve_branch(tmp_path):
    git = FakeGit(branch=BRANCH_PREFIX + "t-1")
    ws = Workspace(str(tmp_path), git=git)
    ws.base = "main"
    ws.rollback()
    flat = [" ".join(c) for c in git.calls]
    assert "reset --hard" in flat
    assert "clean -fd" in flat
    assert "checkout main" in flat
    assert f"branch -D {BRANCH_PREFIX}t-1" in flat
    assert ws.branch is None


def test_merge_to_main_no_ff_and_cleanup(tmp_path):
    git = FakeGit(branch=BRANCH_PREFIX + "t-1")
    ws = Workspace(str(tmp_path), git=git)
    ws.base = "main"
    assert ws.merge_to_main() == "main"
    flat = [" ".join(c) for c in git.calls]
    assert any(c.startswith("merge --no-ff auto-evolve/t-1") for c in flat)
    assert f"branch -D {BRANCH_PREFIX}t-1" in flat


# ------------------------------------------------------------------ 実 git E2E

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _init_repo(path):
    def git(*args):
        subprocess.run(["git", "-C", str(path)] + list(args), check=True,
                       capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True,
                   capture_output=True)
    git("config", "user.email", "evolve@test.local")
    git("config", "user.name", "evolve-test")
    (path / "hello.py").write_text("print('v1')\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-m", "init")


@needs_git
def test_real_git_full_cycle_edit_commit_merge(tmp_path):
    _init_repo(tmp_path)
    ws = Workspace(str(tmp_path), now_fn=lambda: "20260801-000000")
    branch = ws.create_branch("improve hello")
    assert ws.current_branch() == branch

    ws.apply_edit("hello.py", "print('v2')\n")
    sha = ws.commit("auto-evolve: improve hello")
    assert len(sha) == 40

    diff = ws.diff()
    assert "-print('v1')" in diff and "+print('v2')" in diff

    assert ws.merge_to_main() == "main"
    assert ws.current_branch() == "main"
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "print('v2')\n"
    # ブランチは削除済み
    out = RealGit(str(tmp_path)).run("branch", "--list", branch)
    assert out.strip() == ""


@needs_git
def test_real_git_rollback_discards_everything(tmp_path):
    _init_repo(tmp_path)
    ws = Workspace(str(tmp_path), now_fn=lambda: "20260801-000001")
    branch = ws.create_branch("bad idea")
    ws.apply_edit("hello.py", "print('broken')\n")
    ws.apply_edit("junk.py", "garbage\n")
    ws.rollback()
    assert ws.current_branch() == "main"
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "print('v1')\n"
    assert not (tmp_path / "junk.py").exists()
    out = RealGit(str(tmp_path)).run("branch", "--list", branch)
    assert out.strip() == ""


@needs_git
def test_real_git_create_branch_refuses_dirty(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "hello.py").write_text("print('dirty')\n", encoding="utf-8")
    ws = Workspace(str(tmp_path))
    with pytest.raises(GitError, match="not clean"):
        ws.create_branch("t")
