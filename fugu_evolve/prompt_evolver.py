# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_evolve.prompt_evolver — メタプロンプト進化エンジン (Doc D Phase 5)。

fugu_local のプロンプト定数を対象に、変異(:func:`mutate_prompts`)→評価
(:func:`evaluate_prompts`、``eval_fn`` は注入)→勝者採用(:func:`evolve_prompt`)
のループを回す。採用は fugu_prompts の override として書き込み、Workspace (C3)
があれば ``auto-evolve/prompts-{name}`` ブランチにコミットする(mainを直接
触らない)。ベースを上回らなければ何も書かない — 進化は非退行が絶対条件。

評価関数はテストでは決定論的フェイク、実運用では :func:`make_llm_eval_fn`
(候補を system にして probe に回答→judge が採点、要 Ollama)を注入する。
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Sequence, Tuple

import fugu_prompts

VARIANTS_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {"variants": {"type": "array", "items": {"type": "string"}}},
    "required": ["variants"],
}

_MUTATE_SYSTEM = (
    "You are a prompt engineer evolving a system prompt for a local LLM "
    "pipeline. Produce improved VARIANTS of the given prompt: keep its intent "
    "and constraints, vary the strategy (structure, emphasis, step ordering). "
    "Each variant must be self-contained. Reply with JSON only."
)

#: LLM 不通時の決定論的ミューテーション(意図を保つ安全な強化のみ)。
_FALLBACK_SUFFIXES: Tuple[str, ...] = (
    "\nBe precise and verify each step before answering.",
    "\nThink step by step and state assumptions explicitly.",
    "\nAlways end with a clearly marked final answer.",
)

_JUDGE_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {"score": {"type": "number"}},
    "required": ["score"],
}

_JUDGE_SYSTEM = (
    "You are a strict answer grader. Score the answer for correctness and "
    "clarity from 0 (useless) to 10 (perfect). Reply with JSON only."
)

#: make_llm_eval_fn の既定プローブ(小さく・答えが機械的に明らかな問題)。
DEFAULT_PROBES: Tuple[str, ...] = (
    "What is 17 + 25? Answer with the number only.",
    "Name one prime number greater than 10.",
)


def mutate_prompts(base: str, chat, n: int = 3) -> List[str]:
    """base の変異体を n 個返す。LLM 失敗時は決定論的サフィックス変異に落ちる。"""
    prompt = (
        f"Base prompt:\n---\n{base}\n---\n\n"
        f'Return {{"variants": [...]}} with {n} improved variants.'
    )
    try:
        raw = chat.complete(prompt, system=_MUTATE_SYSTEM, fmt=VARIANTS_SCHEMA,
                            temperature=0.8)
        obj = json.loads(raw)
        items = obj.get("variants", []) if isinstance(obj, dict) else []
        variants = []
        for item in items:
            if isinstance(item, str) and item.strip() and item.strip() != base.strip():
                if item.strip() not in (v.strip() for v in variants):
                    variants.append(item)
        if variants:
            return variants[:n]
    except Exception:
        pass
    return [base + suffix for suffix in _FALLBACK_SUFFIXES[:n]]


def evaluate_prompts(candidates: Sequence[str],
                     eval_fn: Callable[[str], float]) -> List[Tuple[str, float]]:
    """全候補に eval_fn を適用しスコア降順で返す(失敗した候補は除外)。"""
    scored: List[Tuple[str, float]] = []
    for candidate in candidates:
        try:
            scored.append((candidate, float(eval_fn(candidate))))
        except Exception:
            continue
    scored.sort(key=lambda t: -t[1])
    return scored


def make_llm_eval_fn(chat, probes: Sequence[str] = DEFAULT_PROBES) -> Callable[[str], float]:
    """候補プロンプトを system にして probe に回答→judge 採点(平均)の eval_fn。

    実運用(Ollama live)向け。judge の応答が壊れている probe は 0 点として数える
    (候補間の相対比較が目的なので、壊れ方は全候補に等しく作用する)。
    """
    def eval_fn(candidate: str) -> float:
        total = 0.0
        for probe in probes:
            answer = chat.complete(probe, system=candidate, temperature=0.2)
            raw = chat.complete(
                f"Question: {probe}\nAnswer: {answer}\n\n"
                'Return {"score": 0-10}.',
                system=_JUDGE_SYSTEM, fmt=_JUDGE_SCHEMA, temperature=0.0)
            try:
                obj = json.loads(raw)
                total += float(obj.get("score", 0)) if isinstance(obj, dict) else 0.0
            except (ValueError, TypeError):
                pass
        return total / max(1, len(probes))

    return eval_fn


def evolve_prompt(name: str, base: str, chat,
                  eval_fn: Callable[[str], float],
                  workspace=None, n: int = 3, min_gain: float = 0.0,
                  apply: bool = True) -> Dict[str, object]:
    """1つのプロンプトを進化させる。

    ベース+変異体を評価し、勝者がベースを ``min_gain`` 超で上回るときだけ採用。
    採用時は Workspace があれば ``auto-evolve/prompts-{name}`` ブランチへ
    override ファイルをコミット(mainは不変)、無ければ fugu_prompts に直接書く。
    ``apply=False``(dry-run)は判定だけ行い一切書かない。
    """
    candidates = [base] + mutate_prompts(base, chat, n=n)
    scored = evaluate_prompts(candidates, eval_fn)
    if not scored:
        return {"name": name, "adopted": False, "branch": None,
                "reason": "no candidate could be evaluated"}
    winner, top_score = scored[0]
    base_score = next((s for c, s in scored if c == base), None)
    adopted = (winner != base
               and (base_score is None or top_score > base_score + min_gain))
    result: Dict[str, object] = {
        "name": name, "winner": winner, "score": top_score,
        "baseline_score": base_score, "adopted": adopted, "branch": None,
        "reason": "winner beats baseline" if adopted else "baseline still best",
    }
    if not (adopted and apply):
        return result
    if workspace is not None:
        branch = workspace.create_branch(f"prompts-{name}")
        workspace.apply_edit(f"fugu_prompts/overrides/{name}.txt", winner)
        workspace.commit(f"auto-evolve: prompt override {name}")
        result["branch"] = branch
    else:
        fugu_prompts.set_override(name, winner)
    return result
