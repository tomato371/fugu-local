# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_evolve.planner — 自律改善プランナー (Doc C Phase 2)。

profiler (C1) のヘルスレポートを LLM に渡し、JSON スキーマ制約付きで改善提案
(:class:`Proposal`)を得る。提案は :func:`validate_proposal` で厳格に検証し、
優先度(failing tests > perf > test > refactor > docs)で並べ替える。

AST-RAG などのコードコンテキストは ``context_fn`` として注入する(D-2:
fugu-rag への直接依存を持たない)。LLM が完全に落ちている場合でも、テストが
落ちていれば決定論的な「Fix failing tests」提案に必ずフォールバックする —
自己改善ループの最重要シグナル(壊れたテスト)を LLM 障害で見失わない。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

#: 提案カテゴリと優先度(小さいほど先に着手)。failing tests の修正が常に最優先。
CATEGORY_PRIORITY: Dict[str, int] = {
    "fix": 0, "perf": 1, "test": 2, "refactor": 3, "docs": 4,
}

PROPOSAL_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "category": {"type": "string",
                                 "enum": list(CATEGORY_PRIORITY)},
                    "target_files": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "category", "target_files", "rationale"],
            },
        }
    },
    "required": ["proposals"],
}

_SYSTEM = (
    "You are an autonomous software-improvement planner for a local Python "
    "repository. Read the health report (pytest results, bench, module stats) "
    "and propose a small number of concrete, independently-applicable "
    "improvements. Rules: failing tests always come first (category 'fix'); "
    "prefer minimal, verifiable changes; every proposal must name real target "
    "files from the report. Reply with JSON only."
)

#: ヘルスレポートをプロンプトへ入れる際の上限(num_ctx 予算)。
_REPORT_CHARS = 4000
_CONTEXT_CHARS = 2000


@dataclass
class Proposal:
    """検証済みの改善提案1件。priority はカテゴリ由来(小さいほど優先)。"""

    title: str
    category: str
    target_files: List[str]
    rationale: str
    steps: List[str] = field(default_factory=list)

    @property
    def priority(self) -> int:
        return CATEGORY_PRIORITY.get(self.category, 99)


def validate_proposal(obj: object,
                      repo_files: Optional[List[str]] = None) -> Optional[Proposal]:
    """LLM 出力の1提案を検証して :class:`Proposal` に変換する(不正は None)。

    title/category/rationale は非空文字列、category は既知語彙、target_files は
    非空の文字列リスト。``repo_files`` が与えられた場合は実在ファイルに絞り、
    1つも残らなければ不正(実在しないファイルへの提案は適用しようがない)。
    """
    if not isinstance(obj, dict):
        return None
    title = obj.get("title")
    category = obj.get("category")
    rationale = obj.get("rationale")
    if not (isinstance(title, str) and title.strip()):
        return None
    if not (isinstance(category, str) and category in CATEGORY_PRIORITY):
        return None
    if not (isinstance(rationale, str) and rationale.strip()):
        return None
    raw_files = obj.get("target_files")
    if not isinstance(raw_files, list):
        return None
    files = [f.strip().replace("\\", "/") for f in raw_files
             if isinstance(f, str) and f.strip()]
    if repo_files is not None:
        allowed = {f.replace("\\", "/") for f in repo_files}
        files = [f for f in files if f in allowed]
    if not files:
        return None
    raw_steps = obj.get("steps")
    steps = [s.strip() for s in raw_steps
             if isinstance(s, str) and s.strip()] if isinstance(raw_steps, list) else []
    return Proposal(title=title.strip(), category=category,
                    target_files=files, rationale=rationale.strip(), steps=steps)


def _fallback_proposals(health_report: Dict[str, object]) -> List[Proposal]:
    """LLM 不通時の決定論的フォールバック: テストが落ちていれば fix 提案1件。"""
    pytest_report = health_report.get("pytest") or {}
    if not isinstance(pytest_report, dict):
        return []
    failed = int(pytest_report.get("failed") or 0)
    errors = int(pytest_report.get("errors") or 0)
    if failed <= 0 and errors <= 0:
        return []
    modules = health_report.get("modules")
    first = (modules[0].get("path") if isinstance(modules, list) and modules
             and isinstance(modules[0], dict) else None)
    tail = str(pytest_report.get("tail") or "")[:500]
    return [Proposal(
        title=f"Fix {failed + errors} failing test(s)",
        category="fix",
        target_files=[first or "tests"],
        rationale=f"pytest reports {failed} failed / {errors} errors. Tail:\n{tail}",
        steps=["Reproduce the failures locally", "Fix the root cause",
               "Re-run the full test suite"],
    )]


def propose(health_report: Dict[str, object], chat,
            context_fn: Optional[Callable[[Dict[str, object]], str]] = None,
            max_proposals: int = 3,
            repo_files: Optional[List[str]] = None) -> List[Proposal]:
    """ヘルスレポートから優先度順の改善提案リストを得る。

    ``context_fn(health_report) -> str`` で AST-RAG 等の追加コンテキストを注入
    できる(失敗しても提案自体は続行)。LLM 呼び出し・パース・検証のどこで
    失敗しても例外は投げず、:func:`_fallback_proposals` に落ちる。返り値は
    カテゴリ優先度の安定ソート済みで、最大 ``max_proposals`` 件。
    """
    context = ""
    if context_fn is not None:
        try:
            context = str(context_fn(health_report) or "")[:_CONTEXT_CHARS]
        except Exception:
            context = ""
    prompt = (
        f"Health report (JSON):\n{json.dumps(health_report, ensure_ascii=False)[:_REPORT_CHARS]}\n\n"
        + (f"Code context:\n{context}\n\n" if context else "")
        + f"Propose at most {max_proposals} improvements. "
        f'Respond with {{"proposals": [{{"title", "category", "target_files", '
        f'"rationale", "steps"}}]}}.'
    )
    try:
        raw = chat.complete(prompt, system=_SYSTEM, fmt=PROPOSAL_SCHEMA,
                            temperature=0.2)
        obj = json.loads(raw)
        items = obj.get("proposals", []) if isinstance(obj, dict) else []
        proposals = [p for p in (validate_proposal(item, repo_files)
                                 for item in items) if p is not None]
    except Exception:
        proposals = []
    if not proposals:
        proposals = _fallback_proposals(health_report)
    proposals.sort(key=lambda p: p.priority)  # 安定ソート: 同カテゴリは提案順を保つ
    return proposals[:max_proposals]
