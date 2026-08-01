# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_evolve.patcher — unified diff ベースのパッチ適用 (Doc E Phase 4)。

従来の implement_proposal はファイル全置換のみで、3000字に切り詰めた文脈から
全文を書き直させるため大きいファイルに安全に適用できなかった。本モジュールは
LLM に unified diff を書かせ、ハンク単位で対象箇所だけを書き換える:

- :func:`parse_unified_diff` — ``---``/``+++``/``@@`` 形式をファイル別ハンク列に
  パース(``a/``・``b/`` 接頭辞は正規化)。
- :func:`apply_patch_to_text` — ハンクの「変更前ブロック」を検索して置換する。
  位置は行番号でなく内容一致で解決(LLM の行番号は当てにならない)。一意に
  見つからなければ **None = 適用失敗**(空白無視の緩和マッチを1段だけ試す)。
- :func:`implement_with_diff` — 提案の対象ファイルに all-or-nothing で適用
  (1ファイルでも失敗したら一切書かない — 半端なパッチ状態を作らない)。
  ``.py`` は compile ガード付き。

呼び出し側 (cli.implement_proposal) は diff 適用を先に試み、失敗したら従来の
全置換にフォールバックする — 安全網は維持したままパッチの精密さだけ上がる。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: LLM 出力から diff 本体を取り出す(```diff フェンス優先、無ければ全体)。
_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)?[ \t]*\r?\n(.*?)```", re.DOTALL)

_DIFF_SYSTEM = (
    "You are a software engineer producing a minimal patch. Return ONE unified "
    "diff (---/+++/@@ hunks) that implements the proposal, wrapped in a "
    "```diff fence. Only touch the listed files. Context lines must be copied "
    "EXACTLY from the current file content. No prose outside the fence."
)


@dataclass
class Hunk:
    old_lines: List[str] = field(default_factory=list)  # ' ' + '-' 行(変更前)
    new_lines: List[str] = field(default_factory=list)  # ' ' + '+' 行(変更後)


@dataclass
class FilePatch:
    path: str
    hunks: List[Hunk] = field(default_factory=list)


def _normalize_path(raw: str) -> str:
    path = raw.strip().split("\t")[0]
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path.replace("\\", "/")


def parse_unified_diff(text: str) -> List[FilePatch]:
    """unified diff をファイル別ハンク列にパースする(壊れた部分は読み飛ばす)。"""
    m = _DIFF_FENCE_RE.search(text or "")
    body = m.group(1) if m else (text or "")
    patches: List[FilePatch] = []
    current: Optional[FilePatch] = None
    hunk: Optional[Hunk] = None
    for line in body.splitlines():
        if line.startswith("--- "):
            hunk = None
            continue
        if line.startswith("+++ "):
            current = FilePatch(path=_normalize_path(line[4:]))
            patches.append(current)
            hunk = None
            continue
        if line.startswith("@@"):
            if current is None:
                continue
            hunk = Hunk()
            current.hunks.append(hunk)
            continue
        if hunk is None:
            continue
        if line.startswith("\\"):  # "\ No newline at end of file"
            continue
        if line.startswith("-"):
            hunk.old_lines.append(line[1:])
        elif line.startswith("+"):
            hunk.new_lines.append(line[1:])
        else:
            content = line[1:] if line.startswith(" ") else line
            hunk.old_lines.append(content)
            hunk.new_lines.append(content)
    return [p for p in patches if p.hunks and any(
        h.old_lines or h.new_lines for h in p.hunks)]


def _locate(lines: List[str], block: List[str]) -> Optional[int]:
    """block の一意な出現位置を探す(まず厳密、次に空白無視で1段だけ緩和)。"""
    if not block:
        return None
    hits = [i for i in range(len(lines) - len(block) + 1)
            if lines[i:i + len(block)] == block]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return None  # 曖昧: どこを書き換えるべきか決められない
    stripped = [line.strip() for line in block]
    hits = [i for i in range(len(lines) - len(block) + 1)
            if [line.strip() for line in lines[i:i + len(block)]] == stripped]
    return hits[0] if len(hits) == 1 else None


def apply_patch_to_text(text: str, hunks: List[Hunk]) -> Optional[str]:
    """ハンク列を順に適用した新テキストを返す(どれか1つでも失敗なら None)。

    行分割は ``split("\\n")``(splitlines ではなく)— 末尾の空行・改行の有無まで
    正確にラウンドトリップさせるため(splitlines は末尾改行情報を失う)。
    """
    lines = text.split("\n")
    for hunk in hunks:
        index = _locate(lines, hunk.old_lines)
        if index is None:
            return None
        lines[index:index + len(hunk.old_lines)] = hunk.new_lines
    return "\n".join(lines)


def implement_with_diff(chat, workspace, proposal, context_chars: int = 6000,
                        max_files: int = 3) -> bool:
    """提案を unified diff として実装・適用する(失敗は False = 全置換へ委譲)。

    all-or-nothing: 全対象ファイルの新テキストを先に計算し、1つでも
    位置解決失敗・compile 失敗・対象外パスがあれば一切書かない。
    """
    allowed = {path.replace("\\", "/") for path in proposal.target_files}
    sections = []
    for rel in proposal.target_files[:max_files]:
        target = os.path.join(workspace.repo, rel)
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()[:context_chars]
        except OSError:
            return False  # diff は既存ファイル前提(新規作成は全置換の領分)
        sections.append(f"=== {rel} (current content) ===\n{content}")
    prompt = (
        f"Proposal: {proposal.title}\n"
        f"Category: {proposal.category}\n"
        f"Rationale: {proposal.rationale}\n\n"
        + "\n\n".join(sections)
        + "\n\nReturn the unified diff implementing the proposal."
    )
    try:
        raw = chat.complete(prompt, system=_DIFF_SYSTEM, temperature=0.2)
    except Exception:
        return False
    patches = parse_unified_diff(raw)
    if not patches:
        return False
    staged: List[Tuple[str, str]] = []
    seen_current: Dict[str, str] = {}
    for patch in patches[:max_files]:
        if patch.path not in allowed:
            return False  # 提示していないファイルへの diff は信用しない
        target = os.path.join(workspace.repo, patch.path)
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as fh:
                current = seen_current.setdefault(patch.path, fh.read())
        except OSError:
            return False
        new_text = apply_patch_to_text(current, patch.hunks)
        if new_text is None or new_text == current:
            return False
        if patch.path.endswith(".py"):
            try:
                compile(new_text, patch.path, "exec")
            except SyntaxError:
                return False
        staged.append((patch.path, new_text))
    if not staged:
        return False
    for rel, new_text in staged:
        workspace.apply_edit(rel, new_text)
    return True
