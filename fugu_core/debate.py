"""fugu_core.debate — マルチスコアルーティングと構造化討論 (Doc D Phase 4)。

- :class:`ScoreMatrix` — モデル×ドメインの勝敗記録(JSON 永続化・壊れた
  ファイルは無視)。``weights`` はラプラス平滑化した勝率、``select_models`` は
  ドメイン適性順の上位選抜。Critic 判定の合否を :func:`record` し続けることで
  「どのモデルがどの領域に強いか」が経験的に蓄積される。
- :func:`classify_domain` — LLM 不使用のドメイン分類(code/math/writing/
  factual/reasoning)。
- :func:`should_debate` — 提案間の語彙類似度が低い(=意見が割れている)とき
  だけ討論する。全員ほぼ同じ回答に討論コストを払わない。
- :func:`debate` — ≤2ターンの相互批評プロトコル。各提案者が他者の回答を見て
  自案を改訂する。失敗したモデルは元の回答を保持(討論が回答を失わせない)。

フック(``FUGU_DEBATE=1`` のときのみ・既定経路不変): fugu_answer の
get_proposals→aggregate 経路で討論を挟み、critique の合否をスコアに記録する。
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

DOMAINS: Tuple[str, ...] = ("code", "math", "writing", "factual", "reasoning")

#: ドメイン手がかり(小文字部分一致 — 日本語は分かち書き不要のため部分一致で拾う)。
_CUES: Dict[str, Tuple[str, ...]] = {
    "code": ("def ", "class ", "import ", "バグ", "実装", "コード", "python",
             "関数", "refactor", "エラー", "traceback", "スクリプト"),
    "math": ("計算", "積分", "微分", "方程式", "証明", "素数", "prove", "integral",
             "equation", "theorem", "算数", "数式"),
    "writing": ("書いて", "エッセイ", "翻訳", "要約", "文章", "作文", "essay",
                "translate", "summarize", "poem", "手紙"),
    "factual": ("誰", "いつ", "どこ", "何年", "who ", "when ", "where ",
                "首都", "人口", "capital of"),
}


def classify_domain(question: str) -> str:
    """LLM 不使用のドメイン分類。手がかり無しは "reasoning"(汎用)に落ちる。"""
    q = (question or "").lower()
    for domain in ("code", "math", "writing", "factual"):  # 優先順
        if any(cue in q for cue in _CUES[domain]):
            return domain
    return "reasoning"


class ScoreMatrix:
    """モデル×ドメインの勝敗スコア(JSON 永続化)。

    構造: ``{model: {domain: [wins, total]}}``。読めないファイルは空扱い
    (スコアは損失許容データ — 壊れて本処理を止める価値はない)。
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.scores: Dict[str, Dict[str, List[int]]] = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    for model, domains in raw.items():
                        if not isinstance(domains, dict):
                            continue
                        for domain, pair in domains.items():
                            if (isinstance(pair, list) and len(pair) == 2
                                    and all(isinstance(x, int) for x in pair)):
                                self.scores.setdefault(
                                    str(model), {})[str(domain)] = list(pair)
            except Exception:
                self.scores = {}

    def record(self, model: str, domain: str, ok: bool) -> None:
        pair = self.scores.setdefault(model, {}).setdefault(domain, [0, 0])
        pair[0] += 1 if ok else 0
        pair[1] += 1
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(self.scores, fh, ensure_ascii=False, indent=1)

    def weights(self, domain: str) -> Dict[str, float]:
        """ドメイン別のモデル重み = ラプラス平滑化勝率 (wins+1)/(total+2)。

        未記録モデルは 0.5(中立)相当になり、実績が積まれるほど差が付く。
        """
        out: Dict[str, float] = {}
        for model, domains in self.scores.items():
            wins, total = domains.get(domain, [0, 0])
            out[model] = (wins + 1) / (total + 2)
        return out

    def select_models(self, domain: str, candidates: List[str],
                      k: int = 3) -> List[str]:
        """candidates をドメイン適性順に並べ上位 k を返す(同点は元順を維持)。"""
        weights = self.weights(domain)
        ranked = sorted(candidates, key=lambda m: -weights.get(m, 0.5))
        return ranked[:k]


#: プロセス共有の既定スコア行列(遅延生成)。
_DEFAULT: dict = {}


def get_default_matrix() -> ScoreMatrix:
    """既定行列。保存先は env ``FUGU_SCORE_PATH``(既定 ``~/.fugu_scores.json``)。"""
    if "matrix" not in _DEFAULT:
        path = (os.environ.get("FUGU_SCORE_PATH")
                or os.path.join(os.path.expanduser("~"), ".fugu_scores.json"))
        _DEFAULT["matrix"] = ScoreMatrix(path=path)
    return _DEFAULT["matrix"]


def reset_default_matrix() -> None:
    _DEFAULT.clear()


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    text = (text or "").lower()
    tokens = set(_WORD.findall(text))
    for run in re.findall(r"[^\x00-\x7f]+", text):
        tokens.update(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return tokens


def should_debate(proposals: List[Tuple[str, str]],
                  threshold: float = 0.4) -> bool:
    """提案間の最小 Jaccard 類似度が threshold 未満なら討論する価値がある。

    有効な提案が2件未満(単独・全滅)なら False。全員ほぼ同じ回答なら False —
    討論は「意見が割れているとき」だけコストを払う。
    """
    valid = [(m, a) for m, a in proposals
             if a and not a.startswith("__ERROR__")]
    if len(valid) < 2:
        return False
    token_sets = [_tokens(a) for _, a in valid]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            union = token_sets[i] | token_sets[j]
            if not union:
                continue
            similarity = len(token_sets[i] & token_sets[j]) / len(union)
            if similarity < threshold:
                return True
    return False


_DEBATE_SYSTEM = (
    "You are one proposer in a structured multi-agent debate. Read the rival "
    "answers, identify concrete errors or omissions in them and in your own "
    "answer, then return your REVISED complete answer only (no meta-commentary "
    "about the debate). Keep what is correct; fix what is not."
)


def debate(question: str, proposals: List[Tuple[str, str]],
           chat_factory: Callable[[str], object],
           turns: int = 2) -> List[Tuple[str, str]]:
    """≤``turns`` ターンの相互批評。各提案者が他者の回答を見て自案を改訂する。

    ``chat_factory(model)`` はそのモデルで話す Chat を返す(注入)。あるモデルの
    改訂が失敗・空なら元の回答を保持。1ターンで全員無変化なら収束として打ち切る。
    """
    current = list(proposals)
    for _ in range(max(0, turns)):
        revised: List[Tuple[str, str]] = []
        for idx, (model, answer) in enumerate(current):
            rivals = "\n\n".join(
                f"[rival {j + 1}]\n{a}" for j, (m, a) in enumerate(current)
                if j != idx and a and not a.startswith("__ERROR__"))
            prompt = (
                f"Question:\n{question}\n\n"
                f"Your current answer:\n{answer}\n\n"
                f"Rival answers:\n{rivals or '(none)'}\n\n"
                "Return your revised complete answer."
            )
            try:
                reply = chat_factory(model).complete(
                    prompt, system=_DEBATE_SYSTEM, temperature=0.3).strip()
                revised.append((model, reply or answer))
            except Exception:
                revised.append((model, answer))
        if revised == current:
            break  # 収束: 追加ターンは無駄
        current = revised
    return current
