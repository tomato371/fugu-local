"""fugu_thinking — 動的 Thinking Budget / テスト時計算スケーリング (Doc B Phase 2)。

質問の複雑さに応じて「どれだけ考えるか」を予算化する:

- :data:`BUDGETS` — low/medium/high の3段。リフレクション回数・ask() の think
  フラグ・num_predict 上限をまとめた :class:`Budget`。
- :func:`decide_budget` — 明示指定はそのまま、"auto" は注入 chat による分類
  (JSON スキーマ制約) + 失敗時 :func:`heuristic_budget` フォールバック。
- :func:`reflect` — 最終回答に対する自己リフレクションループ。予算回数だけ
  「誤りがあれば書き直し、正しければ OK」を繰り返し、OK で早期収束する。
- :func:`refine_answer` — fugu_local のフックから呼ぶ入口。どんな失敗でも
  必ず元の回答を返す(思考予算が答えを失わせることはない)。

すべて注入ベース: chat は fugu_llm.Chat 互換。chat_factory は Budget を受けて
その think/num_predict を反映した Chat を返す(オフラインテストでは FakeChat)。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class Budget:
    """思考予算1段分。reflections=0 は「追加の熟考なし」を意味する。"""

    name: str
    reflections: int            # 最終回答へのリフレクション回数
    think: Optional[bool]       # ask() の think(None=モデル既定)
    num_predict: Optional[int]  # 生成トークン上限(None=既定)


#: 3段の思考予算。low は速度優先(リフレクションなし・think OFF・短い生成)、
#: high は精度優先(2回リフレクション・think ON)。think=True は思考対応モデル
#: (既定の AskChat は CONDUCTOR=qwen3:4b)前提 — 非対応モデルなら ask() が
#: __ERROR__ → RuntimeError となり reflect が安全に打ち切る。
BUDGETS: Dict[str, Budget] = {
    "low": Budget("low", reflections=0, think=False, num_predict=1024),
    "medium": Budget("medium", reflections=1, think=None, num_predict=None),
    "high": Budget("high", reflections=2, think=True, num_predict=None),
}

DEFAULT_BUDGET: str = "medium"

#: 高予算を示す語彙。ASCII 語は単語境界で、日本語は部分一致で判定する
#: ("prove" が "improve" に誤反応しないように)。
_HIGH_CUES: frozenset = frozenset(
    {
        "prove", "proof", "theorem", "derive", "derivation", "integral",
        "equation", "algorithm", "implement", "optimize", "complexity",
        "simulate", "simulation",
        "証明", "導出", "方程式", "積分", "アルゴリズム", "実装", "計算量", "最適化",
    }
)

#: 語数カウント用(router 慣例と同じ)。
_WORD = re.compile(r"[a-z0-9]+")

#: 「これ以下なら短い質問」とみなす文字数(日本語は語分割できないため文字数基準)。
_SHORT_CHARS: int = 40

_CLASSIFY_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {"budget": {"type": "string", "enum": ["low", "medium", "high"]}},
    "required": ["budget"],
}

_CLASSIFY_SYSTEM: str = (
    "You are a task-complexity classifier for a local reasoning system. Classify "
    "how much thinking budget the question needs. Reply with JSON only.\n"
    "- low: simple lookups or short factual questions.\n"
    "- medium: ordinary questions needing some reasoning.\n"
    "- high: math, proofs, algorithm design, physics derivations, or multi-step "
    "engineering problems."
)

_REFLECT_SYSTEM: str = (
    "You are a careful reviewer improving your own draft answer. Check the draft "
    "for factual, logical, or computational errors against the question. If the "
    "draft is fully correct and complete, reply with exactly OK. Otherwise reply "
    "with the complete improved answer only (no meta-commentary)."
)


def heuristic_budget(question: str) -> str:
    """LLM 不要の決定的フォールバック分類。

    1. 高予算語彙(数学/証明/実装系)を含む -> "high"
    2. 短い質問(:data:`_SHORT_CHARS` 文字以下) -> "low"
    3. それ以外 -> "medium"
    """
    q = question.lower()
    for cue in _HIGH_CUES:
        if cue.isascii():
            if re.search(r"\b" + re.escape(cue) + r"\b", q):
                return "high"
        elif cue in q:
            return "high"
    if len(q.strip()) <= _SHORT_CHARS:
        return "low"
    return "medium"


def decide_budget(question: str, chat=None, mode: str = "auto") -> Budget:
    """mode に応じた :class:`Budget` を返す。

    明示指定("low"/"medium"/"high")はそのまま。"auto"(および未知値)は
    chat があれば JSON スキーマ制約付き分類、chat が無い・失敗・語彙外なら
    :func:`heuristic_budget` に落ちる — 常に有効な Budget を返す。
    """
    key = (mode or "").strip().lower()
    if key in BUDGETS:
        return BUDGETS[key]
    if chat is not None:
        try:
            raw = chat.complete(
                f'Question: {question}\n\nRespond with {{"budget": "low" | "medium" | "high"}}.',
                system=_CLASSIFY_SYSTEM,
                fmt=_CLASSIFY_SCHEMA,
                temperature=0.0,
            )
            obj = json.loads(raw)
            name = obj.get("budget") if isinstance(obj, dict) else None
            if isinstance(name, str) and name.strip().lower() in BUDGETS:
                return BUDGETS[name.strip().lower()]
        except Exception:
            pass
    return BUDGETS[heuristic_budget(question)]


def _is_converged(reply: str) -> bool:
    """リフレクション応答が「修正不要」を意味するか(OK / OK. / ok 等)。"""
    return reply.strip().strip(".。!！").upper() == "OK"


def reflect(question: str, draft: str, chat, budget: Budget) -> str:
    """最終回答 draft に budget.reflections 回まで自己リフレクションを適用する。

    各ラウンドで chat に「正しければ OK、誤りがあれば完全な改善版」を求め、
    OK・空応答・例外で打ち切る。どの経路でも現時点の最良回答を返す。
    """
    answer = draft
    for _ in range(max(0, budget.reflections)):
        prompt = (
            f"Question:\n{question}\n\n"
            f"Draft answer:\n{answer}\n\n"
            "Review the draft. Reply with exactly OK if it is fully correct, "
            "or with the complete improved answer."
        )
        try:
            reply = chat.complete(prompt, system=_REFLECT_SYSTEM, temperature=0.2).strip()
        except Exception:
            break
        if not reply or _is_converged(reply):
            break
        answer = reply
    return answer


def refine_answer(
    question: str,
    answer: str,
    mode: str,
    chat_factory: Callable[[Budget], object],
) -> str:
    """fugu_local フックの入口: mode の予算で answer をリフレクションして返す。

    chat_factory(budget) は budget の think/num_predict を反映した Chat を返す
    (分類用には :data:`DEFAULT_BUDGET` の素の Chat を使う)。エラー回答
    (__ERROR__ センチネル)・空回答は素通しし、内部で何が失敗しても必ず
    元の回答を返す。
    """
    if not answer or answer.startswith("__ERROR__"):
        return answer
    try:
        key = (mode or "").strip().lower()
        classify_chat = chat_factory(BUDGETS[DEFAULT_BUDGET]) if key not in BUDGETS else None
        budget = decide_budget(question, classify_chat, key)
        if budget.reflections <= 0:
            return answer
        return reflect(question, answer, chat_factory(budget), budget) or answer
    except Exception:
        return answer
