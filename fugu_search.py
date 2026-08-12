# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_search — 適応的分岐ツリー探索 (AB-MCTS-A 相当の自前実装)。

従来の「Critic が却下したら線形に追加ラウンド」を、回答候補の木の上での
**Thompson Sampling 探索**に置き換える。各ノードで:

- **GEN (幅)**: このノードの子として新しい回答を生成する
- **CONT (深さ)**: 有望な既存の子ノードへ降りて、さらにその下を精緻化する

の選択を、GEN 用の疑似子ノード + 実在の各子ノードそれぞれの **Beta 事後分布**
(``alpha = a0 + Σscore``, ``beta = b0 + Σ(1-score)``) からのサンプリング最大値で行う。
報酬は fugu_verify の連続値スコア [0,1]。SciPy 不要 — ``random.betavariate`` で足りる。

``FUGU_SEARCH_MULTI=1`` では GEN 時に **どのモデルで生成するか** も Thompson
Sampling で選ぶ(モデルごとの Beta 事後)。Conductor の計画にあったモデルは
事前分布を楽観側に初期化する(計画をヒントとして使い、捨てない)。

アイデアの出典: SakanaAI/treequest の AB-MCTS (Apache-2.0)。**コードは参照・複製して
いない** — 論文・README のアルゴリズム記述からの自前実装(依存ゼロ維持のため
``pip install treequest`` もしない)。

並列化: 同一波で選ばれた複数の展開は独立なので ThreadPoolExecutor で並列実行する
(``FUGU_SEARCH_PARALLEL``、既定 4)。選択済みの枝には仮想損失(virtual loss)を
積んで、同じ波の中で同じ枝へ殺到しないようにする。
"""
from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from fugu_llm import Chat

DEFAULT_THRESHOLD = 0.9      # このスコアを超えたら early stop
DEFAULT_PARALLEL = 4
DEFAULT_PRIOR = (1.0, 1.0)   # GEN 疑似子ノードの事前 Beta(a0, b0) — 一様


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


def gen_prior() -> tuple:
    """FUGU_SEARCH_PRIOR="a0,b0" で GEN の事前分布を調整可能(既定 1,1 の一様)。"""
    raw = os.environ.get("FUGU_SEARCH_PRIOR", "")
    if raw:
        try:
            a0, b0 = (float(x) for x in raw.split(","))
            if a0 > 0 and b0 > 0:
                return (a0, b0)
        except ValueError:
            pass
    return DEFAULT_PRIOR


# ------------------------------------------------------------------- 探索木

@dataclass
class Node:
    """回答候補 1 つ。root だけは answer=None の仮想ノード。"""
    answer: Optional[str] = None
    score: float = 0.0
    report: object = None                  # fugu_verify.ScoreReport 等(内訳用)
    model: Optional[str] = None            # この候補を生成したモデル
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    depth: int = 0
    # 「子としての腕」の事後: backprop で自分+子孫の報酬が積まれる
    sum_score: float = 0.0
    sum_anti: float = 0.0
    # 「GEN 疑似子ノード」の事後: このノード直下で生成した子の初期報酬が積まれる
    gen_score: float = 0.0
    gen_anti: float = 0.0
    pending: int = 0                       # 仮想損失(同一波での重複選択の抑止)

    def child_sample(self, rng: random.Random) -> float:
        return rng.betavariate(1.0 + self.sum_score,
                               1.0 + self.sum_anti + self.pending)

    def gen_sample(self, rng: random.Random, prior: tuple) -> float:
        a0, b0 = prior
        return rng.betavariate(a0 + self.gen_score,
                               b0 + self.gen_anti + self.pending)


def _select(root: Node, rng: random.Random, prior: tuple) -> Node:
    """Thompson Sampling で「GEN する場所」を選ぶ。

    各ノードで GEN 疑似子ノードと実在の各子の事後からサンプルし、最大値の枝へ:
    GEN が勝てば「このノードの子として生成」(root なら幅、深部なら精緻化)、
    子が勝てばその子へ降りて同じ選択を繰り返す (= CONT)。
    """
    node = root
    while True:
        best_val = node.gen_sample(rng, prior)
        best_child = None
        for child in node.children:
            val = child.child_sample(rng)
            if val > best_val:
                best_val, best_child = val, child
        if best_child is None:
            return node                    # GEN: ここに新しい子を作る
        node = best_child                  # CONT: 有望な子へ降りる


def _backprop(node: Node, score: float) -> None:
    """新ノードの報酬を祖先の「子としての腕」へ、生成親の GEN 腕へ積む。"""
    if node.parent is not None:
        node.parent.gen_score += score
        node.parent.gen_anti += 1.0 - score
    walk = node
    while walk is not None:
        walk.sum_score += score
        walk.sum_anti += 1.0 - score
        walk = walk.parent


# ------------------------------------------------------------ モデル選択 (TS)

class ModelSelector:
    """FUGU_SEARCH_MULTI=1: GEN 時のモデルも Thompson Sampling で選ぶ。

    モデルごとに Beta 事後を持ち、実際に良いスコアを出したモデルへ予算が寄る。
    Conductor の計画に載っていたモデル(hint)は事前分布を楽観側に初期化する。
    """

    def __init__(self, models: Sequence[str], hint: Sequence[str] = (),
                 rng: Optional[random.Random] = None):
        if not models:
            raise ValueError("no models")
        self.rng = rng or random.Random()
        self.stats: Dict[str, List[float]] = {}
        for m in models:
            # ヒント(計画済み)のモデルは「1 勝ぶん」楽観 — ただし上書き可能な程度
            optimism = 1.0 if m in hint else 0.0
            self.stats[m] = [1.0 + optimism, 1.0]     # [alpha, beta]

    def pick(self) -> str:
        best_model, best_val = None, -1.0
        for m, (a, b) in self.stats.items():
            val = self.rng.betavariate(a, b)
            if val > best_val:
                best_model, best_val = m, val
        return best_model

    def record(self, model: str, score: float) -> None:
        if model in self.stats:
            self.stats[model][0] += score
            self.stats[model][1] += 1.0 - score


# ------------------------------------------------------------------- 生成

_GEN_SYSTEM = (
    "You are an expert solver. Produce a complete, self-contained answer to "
    "the question. Do not mention drafts or previous attempts."
)

_REFINE_SYSTEM = (
    "You are an expert reviser. You are given a draft answer and concrete "
    "criticisms from independent verifiers. Produce an improved COMPLETE "
    "answer that fixes every criticism while keeping what is correct. "
    "Return only the revised answer."
)


def _generate(question: str, node: Node, chat: Chat) -> str:
    """GEN の実体: root 直下なら新規回答、深部なら親候補の精緻化。"""
    if node.answer is None:                          # root → 幅を広げる
        return chat.complete(f"Question:\n{question}", system=_GEN_SYSTEM,
                             temperature=0.8)
    reasons = []
    report = node.report
    if report is not None and hasattr(report, "failing_reasons"):
        reasons = report.failing_reasons()
    critique = "\n".join(f"- {r}" for r in reasons) or "- Improve overall quality."
    return chat.complete(
        f"Question:\n{question}\n\nDraft answer:\n{node.answer}\n\n"
        f"Verifier criticisms:\n{critique}",
        system=_REFINE_SYSTEM, temperature=0.6)


# ------------------------------------------------------------------- 本体

@dataclass
class SearchResult:
    answer: str
    score: float
    calls_used: int              # 生成呼び出し数(予算の消費量)
    nodes: int                   # 生成された候補ノード数(シード含む)
    max_depth: int
    root_width: int              # root 直下の子数(幅)
    best_depth: int
    report: object = None
    tree: Optional[Node] = None

    def shape(self) -> str:
        """探索木の形の一行要約(幅/深さの比 — レポート用)。"""
        return (f"nodes={self.nodes} width={self.root_width} "
                f"depth={self.max_depth} best@{self.best_depth}")


def search(question: str,
           chat_factory: Callable[[Optional[str]], Chat],
           scorer: Callable[[str, str], object],
           *,
           budget: int = 8,
           seed_answer: Optional[str] = None,
           seed_report: object = None,
           models: Optional[Sequence[str]] = None,
           plan_hint: Sequence[str] = (),
           multi: Optional[bool] = None,
           threshold: Optional[float] = None,
           parallel: Optional[int] = None,
           prior: Optional[tuple] = None,
           rng: Optional[random.Random] = None) -> SearchResult:
    """予算 ``budget``(生成 LLM 呼び出し回数)で AB-MCTS を回し、最良候補を返す。

    - chat_factory(model_name) -> Chat。multi でなければ model_name=None で呼ぶ。
    - scorer(question, answer) -> スコア(float か、.score を持つ ScoreReport)。
    - seed_answer: 既存パイプラインの現ドラフト。root の最初の子として木に入れる
      (予算は消費しない — 既に支払われた回答)。
    - threshold 超えで early stop。予算切れなら木全体の最高スコアを返す。
    """
    rng = rng or random.Random()
    threshold = threshold if threshold is not None else \
        _env_float("FUGU_SEARCH_THRESHOLD", DEFAULT_THRESHOLD)
    parallel = parallel if parallel is not None else \
        _env_int("FUGU_SEARCH_PARALLEL", DEFAULT_PARALLEL)
    prior = prior or gen_prior()
    if multi is None:
        multi = os.environ.get("FUGU_SEARCH_MULTI") == "1"
    selector = None
    if multi and models:
        selector = ModelSelector(models, hint=plan_hint, rng=rng)

    root = Node()
    all_nodes: List[Node] = []

    def add_child(parent: Node, answer: str, score: float, report: object,
                  model: Optional[str]) -> Node:
        node = Node(answer=answer, score=score, report=report, model=model,
                    parent=parent, depth=parent.depth + 1)
        parent.children.append(node)
        all_nodes.append(node)
        _backprop(node, score)
        return node

    def _score(answer: str):
        rep = scorer(question, answer)
        return (getattr(rep, "score", rep), rep)

    if seed_answer is not None:
        seed_score = getattr(seed_report, "score", None)
        if seed_score is None:
            seed_score, seed_report = _score(seed_answer)
        add_child(root, seed_answer, float(seed_score), seed_report, None)

    calls = 0
    stop = False
    while calls < budget and not stop:
        wave = min(parallel, budget - calls)
        # --- 選択(逐次): 仮想損失を積んで同一枝への殺到を防ぐ ---
        sites: List[Node] = []
        for _ in range(wave):
            site = _select(root, rng, prior)
            site.pending += 1
            sites.append(site)

        def expand(site: Node):
            model = selector.pick() if selector else None
            chat = chat_factory(model)
            answer = _generate(question, site, chat)
            score, report = _score(answer)
            return site, answer, score, report, model

        # --- 展開(並列): 生成 + 採点は独立 ---
        if len(sites) > 1:
            with ThreadPoolExecutor(max_workers=len(sites)) as pool:
                outcomes = list(pool.map(_safe, [lambda s=s: expand(s)
                                                 for s in sites]))
        else:
            outcomes = [_safe(lambda: expand(sites[0]))]

        for site, outcome in zip(sites, outcomes):
            site.pending = max(0, site.pending - 1)
            calls += 1
            if outcome is None:
                continue                    # 生成失敗はノード化しない(予算は消費)
            _site, answer, score, report, model = outcome
            add_child(_site, answer, score, report, model)
            if selector and model:
                selector.record(model, score)
            if score >= threshold:
                stop = True                 # early stop(この波は回収済み)

    if not all_nodes:
        raise RuntimeError("search produced no candidates")
    best = max(all_nodes, key=lambda n: n.score)
    return SearchResult(
        answer=best.answer, score=best.score, calls_used=calls,
        nodes=len(all_nodes),
        max_depth=max(n.depth for n in all_nodes),
        root_width=len(root.children),
        best_depth=best.depth, report=best.report, tree=root)


def _safe(fn):
    """展開 1 件の失敗(モデル障害など)を None に落とし、探索全体は続行する。"""
    try:
        return fn()
    except Exception:
        return None
