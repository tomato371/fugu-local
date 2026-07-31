"""fugu_core.compressor — マルチラウンド MoA の状態圧縮 (Doc D Phase 2)。

ラウンドを重ねると reference(前ラウンドの統合結果)が育ち、8GB VRAM で pin
している num_ctx(gotcha #2)を圧迫する。本モジュールは round≥2 に入る前に
reference を構造化ダイジェストへ圧縮する:

- :func:`compress_round` — LLM に JSON スキーマ制約で要点(key_facts/
  open_issues/constraints/draft_summary)を抽出させる。失敗時は LLM 不使用の
  決定論的フォールバック(:func:`prune_context`)。
- :func:`prune_context` — **LLM 不使用**の決定論的プルーニング。コードブロック
  と制約行(must/only/必ず/禁止…)は予算超過でも必ず保持する — 精度を左右する
  情報を圧縮で失わない(gotcha #7: 精度優先)。
- :func:`render_digest` — ダイジェストを次ラウンドの reference 文字列に整形。

フックは ``FUGU_COMPRESS=1`` のときだけ fugu_answer のラウンド境界で呼ばれる。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

#: フォールバック時に地の文へ許す既定予算(文字)。
DEFAULT_BUDGET_CHARS = 1200

DIGEST_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "open_issues": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "draft_summary": {"type": "string"},
    },
    "required": ["key_facts", "open_issues", "constraints", "draft_summary"],
}

_SYSTEM = (
    "You are a state compressor for a multi-round answering pipeline. Compress "
    "the current draft into: key_facts (verified facts to carry forward), "
    "open_issues (unresolved problems), constraints (requirements that must "
    "hold), draft_summary (short prose summary of the draft). Never invent "
    "content; keep numbers and identifiers exact. Reply with JSON only."
)

#: 制約行の手がかり(英/日)。これらを含む行は決定論的プルーニングで必ず残す。
_CONSTRAINT_RE = re.compile(
    r"(?i)\b(must|only|never|always|require[sd]?|constraint|do not|don't)\b"
    r"|必ず|禁止|してはならない|しないこと|すること|注意|制約|条件",
)

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class StateDigest:
    round: int
    key_facts: List[str] = field(default_factory=list)
    open_issues: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    draft_summary: str = ""


def prune_context(text: str, budget_chars: int = DEFAULT_BUDGET_CHARS) -> str:
    """LLM 不使用の決定論的プルーニング。

    予算内ならそのまま返す。超過時は (1) コードブロック全文と (2) 制約行を
    無条件で保持し、残り予算に地の文を先頭から詰める。コード+制約だけで予算を
    超えても切らない(答えの正しさを左右する部分は絶対に落とさない)。
    """
    if len(text) <= budget_chars:
        return text
    # コードブロックをプレースホルダに退避(行単位処理で分断しないため)
    blocks: List[str] = []

    def _stash(m):
        blocks.append(m.group(0))
        return f"\x00BLOCK{len(blocks) - 1}\x00"

    stashed = _FENCE_RE.sub(_stash, text)
    kept: List[str] = []
    used = 0
    for line in stashed.splitlines():
        is_block = "\x00BLOCK" in line
        is_constraint = bool(_CONSTRAINT_RE.search(line))
        if is_block or is_constraint:
            kept.append(line)  # 無条件保持(予算に数えない)
        elif used + len(line) + 1 <= budget_chars:
            kept.append(line)
            used += len(line) + 1
    out = "\n".join(kept)
    for idx, block in enumerate(blocks):
        out = out.replace(f"\x00BLOCK{idx}\x00", block)
    return out


def compress_round(question: str, reference: str, chat, round_no: int = 1,
                   budget_chars: int = DEFAULT_BUDGET_CHARS) -> StateDigest:
    """reference を構造化ダイジェストに圧縮する(失敗は決定論的フォールバック)。

    フォールバックは draft_summary= :func:`prune_context` の結果のみの
    ダイジェスト — LLM が完全に落ちていても次ラウンドへ渡す状態は必ず作れる。
    """
    prompt = (
        f"Question:\n{question}\n\n"
        f"Current draft (round {round_no}):\n{reference[:6000]}\n\n"
        'Return {"key_facts": [...], "open_issues": [...], '
        '"constraints": [...], "draft_summary": ...}.'
    )
    try:
        raw = chat.complete(prompt, system=_SYSTEM, fmt=DIGEST_SCHEMA,
                            temperature=0.0)
        obj = json.loads(raw)
        if isinstance(obj, dict):
            def _strs(key):
                value = obj.get(key)
                return [s.strip() for s in value
                        if isinstance(s, str) and s.strip()] \
                    if isinstance(value, list) else []
            summary = obj.get("draft_summary")
            digest = StateDigest(
                round=round_no,
                key_facts=_strs("key_facts"),
                open_issues=_strs("open_issues"),
                constraints=_strs("constraints"),
                draft_summary=summary.strip() if isinstance(summary, str) else "",
            )
            if digest.draft_summary or digest.key_facts:
                return digest
    except Exception:
        pass
    return StateDigest(round=round_no,
                       draft_summary=prune_context(reference, budget_chars))


def render_digest(digest: StateDigest) -> str:
    """ダイジェストを次ラウンドの reference 用テキストに整形(空セクション省略)。"""
    lines = [f"## 状態ダイジェスト (round {digest.round})"]
    for title, items in (("確定事実", digest.key_facts),
                         ("未解決の論点", digest.open_issues),
                         ("守るべき制約", digest.constraints)):
        if items:
            lines.append(f"### {title}")
            lines.extend(f"- {item}" for item in items)
    if digest.draft_summary:
        lines += ["### ドラフト要約", digest.draft_summary]
    return "\n".join(lines)
