# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_core.subagents — 動的サブエージェント生成 (Doc E Phase 4)。

従来の council は固定4パーソナ(Proposer A〜D)からの選抜のみで、質問に固有の
専門性(例: 気象水文学者、正規表現の専門家)をその場で立てることができなかった。
本モジュールは「役割の動的設計」を提供する:

- :func:`design_subagent` — 質問から専門家ペルソナ(役割名+system prompt)を
  LLM に設計させる(スキーマ制約、失敗時 None)。
- :class:`RoleChat` — 設計された system prompt を常に前置する Chat ラッパ
  (fugu_llm.Chat と構造的互換。呼び出し側の system は後置で合成)。
- :func:`spawn` — spec + chat_factory から専門家 Chat を生成。

GPU は1基のため実行は逐次だが、「役割を動的に作る」こと自体は VRAM 非依存。
フックは ``FUGU_DYNAMIC_SUBAGENTS=1`` のときだけ fugu_answer の提案収集後に
専門家1体を追加提案として投入する(既定経路は不変)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Optional

SPEC_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "system_prompt": {"type": "string"},
    },
    "required": ["role", "system_prompt"],
}

_DESIGN_SYSTEM = (
    "You design ONE specialist AI persona for the given task. Return a short "
    "role name (a few words) and a focused system prompt that would make the "
    "model behave as a domain expert on exactly this task — include the "
    "methods, checks, and pitfalls such an expert would apply. Reply with "
    "JSON only."
)


@dataclass
class SubagentSpec:
    """動的に設計された専門家1体分の仕様。"""

    role: str
    system_prompt: str


def design_subagent(task: str, chat) -> Optional[SubagentSpec]:
    """task 専用の専門家ペルソナを設計する(失敗・不正は None)。"""
    prompt = (
        f"Task: {task}\n\n"
        'Respond with {"role": ..., "system_prompt": ...}.'
    )
    try:
        raw = chat.complete(prompt, system=_DESIGN_SYSTEM, fmt=SPEC_SCHEMA,
                            temperature=0.3)
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    role, system_prompt = obj.get("role"), obj.get("system_prompt")
    if not (isinstance(role, str) and role.strip()
            and isinstance(system_prompt, str) and system_prompt.strip()):
        return None
    return SubagentSpec(role=role.strip(), system_prompt=system_prompt.strip())


class RoleChat:
    """spec の system prompt を常に前置する Chat ラッパ(構造的に Chat 互換)。"""

    def __init__(self, spec: SubagentSpec, inner):
        self.spec = spec
        self.inner = inner

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 fmt=None, temperature: float = 0.2, images=None) -> str:
        merged = (f"{self.spec.system_prompt}\n\n{system}" if system
                  else self.spec.system_prompt)
        return self.inner.complete(prompt, system=merged, fmt=fmt,
                                   temperature=temperature, images=images)


def spawn(spec: SubagentSpec, chat_factory: Callable[[], object]) -> RoleChat:
    """spec を chat_factory 製の Chat に被せて専門家として起動する。"""
    return RoleChat(spec, chat_factory())
