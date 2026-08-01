# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_verify — 多検証者スコアリング(Multi-Agent Verification / BoN-MAV 相当)。

従来の敵対的検証(fugu_core.debate.adversarial_verify)は「反証に失敗したら通す」
二値ゲートだった。本モジュールはそれを **連続値スコア [0,1] を返すスコアラー** に
拡張する。スコアは fugu_search の探索報酬にも、FUGU_MAV=1 の best-of-N 選択にも使う。

- ``AspectVerifier`` プロトコル: verify(question, answer, context) -> VerifierResult|None
  (None = この回答には適用不能。集約から除外される)
- 検証者は互いに独立なので ``concurrent.futures`` で **並列に** 投げる
  (OLLAMA_NUM_PARALLEL を活かす。逐次にしない)。
- 集約: ``score = Σ(w_i × approved_i × confidence_i) / Σw_i``(既定は均等重み)。
  **どの検証者が落としたかの内訳を必ず保持する**。

アイデアの出典: "Multi-Agent Verification: Scaling Test-Time Compute with
Multiple Verifiers" (BoN-MAV, Lifshitz et al., 2025)。コードは参照・複製していない
(アルゴリズム記述からの自前実装)。
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

from fugu_llm import Chat

# 検証者は小型モデルで回す前提(ここに大型モデルを使うと予算が溶ける)。
# 実モデルの割り当ては呼び出し側(fugu_local のフック)が chat_factory で注入する。
DEFAULT_CONFIDENCE = 0.5      # 検証者が confidence を返さなかったときの既定値
FAILURE_CONFIDENCE = 0.0      # 検証者自体が落ちた場合(安全側: 承認に寄与しない)


@dataclass
class VerifierResult:
    aspect: str
    approved: bool
    confidence: float          # [0,1]
    reason: str = ""

    @property
    def contribution(self) -> float:
        """集約式の分子側: approved × confidence。"""
        return self.confidence if self.approved else 0.0


class AspectVerifier(Protocol):
    aspect: str

    def verify(self, question: str, answer: str,
               context: str = "") -> Optional[VerifierResult]:
        """None を返した検証者は「適用不能」として集約から除外される。"""
        ...


# ---------------------------------------------------------------- LLM 検証者

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["approved"],
}

_VERIFIER_SYSTEM = (
    "You are an independent verifier auditing an answer through the {aspect} "
    "lens only. {charter} "
    "Return JSON only: {{\"approved\": true|false, "
    "\"confidence\": 0.0-1.0, \"reason\": \"...\"}}. "
    "confidence is how sure you are of your OWN verdict. Do not audit aspects "
    "outside your lens."
)


def _clamp01(x) -> float:
    try:
        return min(1.0, max(0.0, float(x)))
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE


class _LLMVerifier:
    """LLM ジャッジ型検証者の共通実装。charter が観点ごとの審査基準。"""

    aspect = "generic"
    charter = "Audit the answer."

    def __init__(self, chat: Chat):
        self.chat = chat

    def _prompt(self, question: str, answer: str, context: str) -> str:
        parts = [f"Question:\n{question}\n", f"Answer under audit:\n{answer}\n"]
        if context:
            parts.append(f"Reference context:\n{context}\n")
        parts.append('Respond with {"approved": ..., "confidence": ..., "reason": ...}.')
        return "\n".join(parts)

    def verify(self, question: str, answer: str,
               context: str = "") -> Optional[VerifierResult]:
        try:
            raw = self.chat.complete(
                self._prompt(question, answer, context),
                system=_VERIFIER_SYSTEM.format(aspect=self.aspect,
                                               charter=self.charter),
                fmt=_VERIFY_SCHEMA, temperature=0.2)
            obj = json.loads(raw)
            approved = obj.get("approved") if isinstance(obj, dict) else None
            if not isinstance(approved, bool):
                raise ValueError("approved missing")
            return VerifierResult(
                aspect=self.aspect, approved=approved,
                confidence=_clamp01(obj.get("confidence", DEFAULT_CONFIDENCE)),
                reason=str(obj.get("reason") or "")[:300])
        except Exception as exc:
            # 検証者自身の失敗は「不承認・寄与ゼロ」(安全側)。理由は内訳に残す。
            return VerifierResult(aspect=self.aspect, approved=False,
                                  confidence=FAILURE_CONFIDENCE,
                                  reason=f"verifier unavailable: {exc}"[:300])


class LogicalVerifier(_LLMVerifier):
    aspect = "logical"
    charter = ("Hunt for leaps of reasoning, self-contradictions, and "
               "conclusions that do not follow from the stated premises.")


class ComputationalVerifier(_LLMVerifier):
    """数値・単位・次元。可能なら sandbox で実際に再計算する (Program-of-Thought)。"""

    aspect = "computational"
    charter = ("Recompute every numeric claim, check units and dimensional "
               "consistency. Approve only if the arithmetic actually holds.")

    _POT_SYSTEM = (
        "Write a short standalone Python program that RECOMPUTES the key "
        "numeric/mathematical claims of the answer and prints exactly "
        "'CHECK_OK' if they all hold, or 'CHECK_FAIL: <what differs>' if not. "
        "No imports beyond the standard library (math, fractions, itertools "
        "are fine; sympy only if truly needed). If the answer contains "
        "nothing checkable by computation, print 'CHECK_NA'. "
        "Return only a fenced python code block."
    )

    def verify(self, question: str, answer: str,
               context: str = "") -> Optional[VerifierResult]:
        pot = self._pot_verify(question, answer)
        if pot is not None:
            return pot
        return super().verify(question, answer, context)

    def _pot_verify(self, question: str, answer: str) -> Optional[VerifierResult]:
        """PoT 経路: 検算プログラムを書かせて sandbox で実行。使えなければ None。"""
        try:
            import fugu_sandbox
            code_raw = self.chat.complete(
                f"Question:\n{question}\n\nAnswer under audit:\n{answer}\n",
                system=self._POT_SYSTEM, temperature=0.2)
            code = fugu_sandbox.extract_code_block(code_raw)
            if not code or "CHECK_" not in code:
                return None
            result = fugu_sandbox.get_sandbox().run(code)
            out = (result.stdout or "") + (result.stderr or "")
            if "CHECK_OK" in out:
                return VerifierResult(self.aspect, True, 0.95,
                                      "recomputed in sandbox: OK")
            if "CHECK_FAIL" in out:
                line = next((l for l in out.splitlines() if "CHECK_FAIL" in l), "")
                return VerifierResult(self.aspect, False, 0.9, line[:300])
            return None   # CHECK_NA / 実行不能 → LLM 審査へフォールバック
        except Exception:
            return None


class FactualVerifier(_LLMVerifier):
    aspect = "factual"
    charter = ("Check that every factual claim is grounded: in the provided "
               "reference context when present, otherwise in well-established "
               "knowledge. Flag fabricated entities, dates, and citations.")


class ConstraintVerifier(_LLMVerifier):
    aspect = "constraint"
    charter = ("Check ONLY the explicit constraints stated in the question "
               "itself: requested output format, length limits, language, "
               "required sections, number of items. Approve if all explicit "
               "constraints are met; if the question states none, approve "
               "with low confidence.")


class CodeTestVerifier:
    """回答がコードなら fugu_tdc のテストを走らせ、pass 率を confidence にする。
    コードが無い回答には None(適用不能)。"""

    aspect = "codetest"

    def __init__(self, chat: Chat):
        self.chat = chat

    def verify(self, question: str, answer: str,
               context: str = "") -> Optional[VerifierResult]:
        try:
            import fugu_sandbox
            import fugu_tdc
        except ImportError:
            return None
        code = fugu_sandbox.extract_code_block(answer)
        if not code:
            return None                      # コード回答ではない → 除外
        try:
            test_source = fugu_tdc.draft_tests(question, code, self.chat)
            if not test_source:
                return None                  # テスト起草不能 → 除外(減点しない)
            result = fugu_tdc.run_tests(code, test_source)
            passed, failed = _parse_pytest_counts(
                (result.stdout or "") + (result.stderr or ""))
            total = passed + failed
            if total == 0:
                return None
            ratio = passed / total
            return VerifierResult(self.aspect, failed == 0, ratio,
                                  f"pytest: {passed} passed, {failed} failed")
        except Exception as exc:
            return VerifierResult(self.aspect, False, FAILURE_CONFIDENCE,
                                  f"verifier unavailable: {exc}"[:300])


def _parse_pytest_counts(output: str) -> Tuple[int, int]:
    """pytest のサマリ行から (passed, failed+errors) を拾う。"""
    passed = sum(int(m) for m in re.findall(r"(\d+) passed", output))
    failed = sum(int(m) for m in re.findall(r"(\d+) (?:failed|error)", output))
    return passed, failed


# ------------------------------------------------------------------- 集約

@dataclass
class ScoreReport:
    score: float                              # [0,1] 重み付き集約
    results: List[VerifierResult] = field(default_factory=list)

    def failing_reasons(self) -> List[str]:
        """不承認だった検証者の内訳(探索の精緻化ヒントに使う)。"""
        return [f"[{r.aspect}] {r.reason}".strip()
                for r in self.results if not r.approved and r.reason]

    def breakdown(self) -> str:
        return ", ".join(
            f"{r.aspect}={'✓' if r.approved else '✗'}{r.confidence:.2f}"
            for r in self.results) or "(no applicable verifier)"


def default_verifiers(chat_factory: Callable[[str], Chat]) -> List[AspectVerifier]:
    """既定の 5 検証者。chat_factory(aspect) が各検証者の Chat を返す
    (小型モデル推奨 — 呼び出し側が AskChat(model=FALLBACK_MODEL) 等を注入する)。"""
    return [
        LogicalVerifier(chat_factory("logical")),
        ComputationalVerifier(chat_factory("computational")),
        FactualVerifier(chat_factory("factual")),
        ConstraintVerifier(chat_factory("constraint")),
        CodeTestVerifier(chat_factory("codetest")),
    ]


def score_answer(question: str, answer: str, verifiers: Sequence[AspectVerifier],
                 context: str = "", weights: Optional[dict] = None,
                 parallel: bool = True) -> ScoreReport:
    """全検証者を(既定で並列に)走らせ、重み付きスコアと内訳を返す。

    weights: {aspect: w} — 省略時は均等。適用不能(None)の検証者は分母からも除外。
    """
    if parallel and len(verifiers) > 1:
        with ThreadPoolExecutor(max_workers=len(verifiers)) as pool:
            futures = [pool.submit(v.verify, question, answer, context)
                       for v in verifiers]
            raw = [f.result() for f in futures]     # 提出順 = verifiers 順を維持
    else:
        raw = [v.verify(question, answer, context) for v in verifiers]

    results = [r for r in raw if r is not None]
    if not results:
        return ScoreReport(score=0.0, results=[])
    weights = weights or {}
    num = den = 0.0
    for r in results:
        w = float(weights.get(r.aspect, 1.0))
        num += w * r.contribution
        den += w
    return ScoreReport(score=(num / den if den else 0.0), results=results)


# ---------------------------------------------------------------- best-of-N

def best_of_n(question: str, candidates: Sequence[str],
              verifiers: Sequence[AspectVerifier], context: str = "",
              weights: Optional[dict] = None,
              ) -> Tuple[int, ScoreReport, List[ScoreReport]]:
    """候補列を全検証者でスコアリングし、最高スコアの添字を返す (BoN-MAV)。

    候補ごとの検証はモデル負荷を抑えるため逐次、候補内の検証者は並列。
    戻り値: (best_index, best_report, all_reports)。同点は先勝ち(安定)。
    """
    if not candidates:
        raise ValueError("no candidates")
    reports = [score_answer(question, c, verifiers, context, weights)
               for c in candidates]
    best = max(range(len(candidates)), key=lambda i: reports[i].score)
    return best, reports[best], reports


def mav_n(default: int = 4) -> int:
    """FUGU_MAV_N(best-of-N の N)。壊れた値は既定に落とす。"""
    try:
        return max(1, int(os.environ.get("FUGU_MAV_N", default)))
    except ValueError:
        return default
