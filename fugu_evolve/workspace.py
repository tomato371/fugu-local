# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_evolve.workspace — 隔離ブランチ上の安全な自己編集 (Doc C Phase 4=D-3)。

自己改善ループがリポジトリを書き換えるための唯一の入口。安全策は3層:

1. **注入可能な GitClient** — 実 git は :class:`RealGit`(subprocess)、テストは
   FakeGit。上位モジュールは git の存在を仮定しない。
2. **`auto-evolve/` 接頭辞ガード** — 編集・コミット・破壊的操作(reset --hard /
   clean -fd / branch -D)は、現在のブランチが :data:`BRANCH_PREFIX` で始まる
   ときのみ実行できる。main を直接壊す経路を構造的に持たない。
3. **クリーンツリー必須** — ブランチ作成は未コミット変更ゼロが前提
   (:meth:`Workspace.ensure_clean`)。人間の作業中変更を巻き込まない。
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Callable, Optional, Protocol

#: 自己改善ブランチの接頭辞。この接頭辞の外では破壊的操作を一切行わない。
BRANCH_PREFIX = "auto-evolve/"


class GitError(RuntimeError):
    """git 操作の失敗(またはガード違反)。"""


class GitClient(Protocol):
    def run(self, *args: str) -> str:
        """`git <args>` を実行して stdout を返す。失敗は :class:`GitError`。"""
        ...


class RealGit:
    """subprocess による実 git クライアント。"""

    def __init__(self, repo: str):
        self.repo = repo

    def run(self, *args: str) -> str:
        r = subprocess.run(
            ["git", "-C", self.repo] + list(args),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed (exit {r.returncode}): "
                           f"{(r.stderr or r.stdout).strip()[:500]}")
        return r.stdout


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")[:40]
    return slug or "change"


class Workspace:
    """`auto-evolve/{slug}-{ts}` ブランチ上でのみ編集を許す作業空間。

    典型フロー: ``ensure_clean`` → ``create_branch`` → ``apply_edit``* →
    ``commit`` → (evaluator 検証) → ``merge_to_main`` または ``rollback``。
    ``now_fn`` はブランチ名のタイムスタンプ生成(テストで固定値を注入)。
    """

    def __init__(self, repo: str, git: Optional[GitClient] = None,
                 now_fn: Optional[Callable[[], str]] = None):
        self.repo = os.path.abspath(repo)
        self.git: GitClient = git if git is not None else RealGit(self.repo)
        self._now = now_fn or (lambda: time.strftime("%Y%m%d-%H%M%S"))
        self.branch: Optional[str] = None  # 作成した auto-evolve ブランチ
        self.base: Optional[str] = None    # 分岐元ブランチ

    # ---------- 状態 ----------

    def ensure_clean(self) -> None:
        """未コミット変更があれば :class:`GitError`(人間の作業を巻き込まない)。"""
        if self.git.run("status", "--porcelain").strip():
            raise GitError("working tree is not clean — commit or stash first")

    def current_branch(self) -> str:
        return self.git.run("rev-parse", "--abbrev-ref", "HEAD").strip()

    def _require_evolve_branch(self) -> str:
        """接頭辞ガード: auto-evolve ブランチ上でなければ即拒否。"""
        current = self.current_branch()
        if not current.startswith(BRANCH_PREFIX):
            raise GitError(
                f"refusing: '{current}' is not an {BRANCH_PREFIX}* branch")
        return current

    # ---------- ブランチ ----------

    def create_branch(self, title: str) -> str:
        """クリーンツリーを確認して `auto-evolve/{slug}-{ts}` を作成・checkout。"""
        self.ensure_clean()
        self.base = self.current_branch()
        name = f"{BRANCH_PREFIX}{_slugify(title)}-{self._now()}"
        self.git.run("checkout", "-b", name)
        self.branch = name
        return name

    # ---------- 編集 ----------

    def apply_edit(self, rel_path: str, content: str) -> str:
        """ファイルを全置換で書き込む(auto-evolve ブランチ上のみ)。

        パスはリポジトリ内に限定(絶対パス・``..`` 脱出は拒否)。親ディレクトリ
        は必要なら作る。戻り値は書き込んだ絶対パス。
        """
        self._require_evolve_branch()
        rel = rel_path.replace("\\", "/")
        target = os.path.abspath(os.path.join(self.repo, rel))
        if not target.startswith(self.repo + os.sep):
            raise GitError(f"refusing: path escapes the repository: {rel_path}")
        os.makedirs(os.path.dirname(target) or self.repo, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        return target

    def commit(self, message: str) -> str:
        """全変更をコミットして HEAD の sha を返す(auto-evolve ブランチ上のみ)。"""
        self._require_evolve_branch()
        self.git.run("add", "-A")
        self.git.run("commit", "-m", message)
        return self.git.run("rev-parse", "HEAD").strip()

    def diff(self, base: Optional[str] = None) -> str:
        """分岐元(または指定 base)に対する差分。Critic 承認の判定材料。"""
        return self.git.run("diff", base or self.base or "HEAD")

    # ---------- 破壊的操作(ガード必須) ----------

    def rollback(self) -> None:
        """今回の試みを完全放棄: 変更破棄 → 分岐元へ戻り → ブランチ削除。

        reset --hard / clean -fd / branch -D は全て接頭辞ガードの内側でのみ
        実行される(D-3 の必須要件)。
        """
        current = self._require_evolve_branch()
        self.git.run("reset", "--hard")
        self.git.run("clean", "-fd")
        self.git.run("checkout", self.base or "main")
        if current.startswith(BRANCH_PREFIX):  # 二重ガード(削除対象名を再確認)
            self.git.run("branch", "-D", current)
        self.branch = None

    def merge_to_main(self, delete_branch: bool = True) -> str:
        """検証済みブランチを分岐元へ --no-ff マージする。戻り値はマージ先ブランチ。"""
        current = self._require_evolve_branch()
        base = self.base or "main"
        self.git.run("checkout", base)
        self.git.run("merge", "--no-ff", current, "-m", f"merge: {current}")
        if delete_branch and current.startswith(BRANCH_PREFIX):
            self.git.run("branch", "-D", current)
            self.branch = None
        return base
