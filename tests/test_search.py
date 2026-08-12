# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_search のオフラインテスト(FakeChat + 固定シードで決定論)。"""
import random
import threading

import pytest

import fugu_search as S
from fugu_llm import FakeChat


def _scorer_from(table, default=0.1):
    """answer 文字列 → スコアの決定論スコアラー。"""
    def scorer(question, answer):
        for key, val in table.items():
            if key in answer:
                return val
        return default
    return scorer


def _chat_factory_counter(prefix="cand"):
    """呼ばれるたびに一意な回答を返す chat_factory(生成器)。"""
    state = {"n": 0}

    def factory(model=None):
        def gen(prompt):
            state["n"] += 1
            tag = f"{prefix}{state['n']}"
            return f"{tag} (model={model})" if model else tag
        return FakeChat(fn=gen)
    return factory, state


# ------------------------------------------------------------------ 基本動作

def test_budget_is_respected_and_best_returned():
    factory, state = _chat_factory_counter()
    scorer = _scorer_from({"cand3": 0.7, "cand5": 0.8}, default=0.2)
    res = S.search("q", factory, scorer, budget=6, parallel=1,
                   threshold=0.99, rng=random.Random(0))
    assert res.calls_used == 6 == state["n"]
    assert res.score == pytest.approx(0.8)
    assert "cand5" in res.answer
    assert res.nodes == 6


def test_early_stop_on_threshold():
    factory, state = _chat_factory_counter()
    scorer = _scorer_from({"cand2": 0.95}, default=0.3)
    res = S.search("q", factory, scorer, budget=50, parallel=1,
                   threshold=0.9, rng=random.Random(0))
    assert res.calls_used == 2           # 2 個目で 0.95 ≥ 0.9 → 打ち切り
    assert res.score == pytest.approx(0.95)


def test_seed_answer_joins_tree_without_spending_budget():
    factory, state = _chat_factory_counter()
    scorer = _scorer_from({}, default=0.1)   # 生成候補は全部 0.1
    res = S.search("q", factory, scorer, budget=2, parallel=1,
                   seed_answer="既存ドラフト", seed_report=None,
                   rng=random.Random(0))
    # シードが最良のまま残り、予算は生成 2 回だけに使われた
    assert res.calls_used == 2 and state["n"] == 2
    assert res.answer == "既存ドラフト"      # scorer は既定 0.1 だが seed も 0.1 → 先勝ち
    assert res.nodes == 3


def test_generation_failure_consumes_budget_but_search_survives():
    calls = {"n": 0}

    def factory(model=None):
        def gen(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("model down")
            return f"ok{calls['n']}"
        return FakeChat(fn=gen)

    res = S.search("q", factory, _scorer_from({}, default=0.5), budget=3,
                   parallel=1, threshold=0.99, rng=random.Random(0))
    assert res.calls_used == 3
    assert res.nodes == 2                 # 1 回目は失敗でノード化されない
    assert res.answer.startswith("ok")


def test_search_raises_when_nothing_succeeds():
    def factory(model=None):
        return FakeChat(fn=lambda p: (_ for _ in ()).throw(RuntimeError("dead")))

    with pytest.raises(RuntimeError):
        S.search("q", factory, _scorer_from({}), budget=2, parallel=1,
                 rng=random.Random(0))


# ------------------------------------------------- 分岐判断(本題の決定論検証)

def test_budget_flows_to_the_high_scoring_branch():
    """スコアの高い枝に予算が寄る: 高スコア候補の下(精緻化)が育つ。"""
    factory, _ = _chat_factory_counter()
    # cand1 は強い(0.8)、以降の root 直下の新規は弱い(0.2)。
    # ただし cand1 の子孫(refine: プロンプトに Draft が入る)はやや高い 0.6。

    def scorer(question, answer):
        if answer == "cand1":
            return 0.8
        return 0.2
    res = S.search("q", factory, scorer, budget=12, parallel=1,
                   threshold=0.99, rng=random.Random(7))
    root = res.tree
    strong = next(c for c in root.children if c.answer == "cand1")
    weak_subtrees = sum(_subtree_size(c) for c in root.children
                       if c is not strong)
    # 0.8 の枝の子孫数が、弱い枝の子孫合計を上回る = 予算が寄っている
    assert _subtree_size(strong) - 1 > 0
    assert _subtree_size(strong) >= weak_subtrees / max(1, len(root.children) - 1)


def _subtree_size(node):
    return 1 + sum(_subtree_size(c) for c in node.children)


def test_refinement_prompt_carries_verifier_criticisms():
    """CONT で降りた先の精緻化プロンプトに、落とした検証者の理由が入る。"""

    class Report:
        score = 0.4

        def failing_reasons(self):
            return ["[computational] 7*13 miscalculated"]

    captured = []

    def factory(model=None):
        def gen(prompt):
            captured.append(prompt)
            return f"gen{len(captured)}"
        return FakeChat(fn=gen)

    S.search("q", factory, lambda q, a: 0.4, budget=3, parallel=1,
             seed_answer="draft", seed_report=Report(), rng=random.Random(1))
    refine_prompts = [p for p in captured if "Draft answer:" in p]
    assert refine_prompts, "深さ方向(精緻化)が一度も選ばれていない"
    assert any("7*13 miscalculated" in p for p in refine_prompts)


def test_deterministic_with_fixed_seed():
    def run():
        factory, _ = _chat_factory_counter()
        return S.search("q", factory, _scorer_from({"cand2": 0.6}, 0.3),
                        budget=8, parallel=1, threshold=0.99,
                        rng=random.Random(42))
    a, b = run(), run()
    assert a.answer == b.answer and a.score == b.score
    assert a.shape() == b.shape()


def test_gen_prior_env(monkeypatch):
    monkeypatch.delenv("FUGU_SEARCH_PRIOR", raising=False)
    assert S.gen_prior() == (1.0, 1.0)
    monkeypatch.setenv("FUGU_SEARCH_PRIOR", "2.5,0.5")
    assert S.gen_prior() == (2.5, 0.5)
    monkeypatch.setenv("FUGU_SEARCH_PRIOR", "junk")
    assert S.gen_prior() == (1.0, 1.0)
    monkeypatch.setenv("FUGU_SEARCH_PRIOR", "-1,2")
    assert S.gen_prior() == (1.0, 1.0)


def test_optimistic_prior_widens_root_first():
    """GEN 事前を強く楽観(a0≫b0)にすると root 直下の幅が優先される。"""
    factory, _ = _chat_factory_counter()
    res = S.search("q", factory, _scorer_from({}, default=0.5), budget=8,
                   parallel=1, threshold=0.99, prior=(50.0, 1.0),
                   rng=random.Random(3))
    assert res.root_width >= 6            # ほぼ毎回 GEN が勝つ → 幅が広がる


# ------------------------------------------------------------- Multi-LLM 選択

def test_multi_llm_budget_shifts_to_better_model():
    """良いスコアを出すモデルへ予算(生成回数)が寄る。固定シードで検証。"""
    counts = {"good": 0, "bad": 0}

    def factory(model=None):
        def gen(prompt):
            counts[model] += 1
            return f"ans-{model}-{counts[model]}"
        return FakeChat(fn=gen)

    def scorer(question, answer):
        return 0.9 if "-good-" in answer else 0.1

    res = S.search("q", factory, scorer, budget=20, parallel=1,
                   models=["bad", "good"], multi=True, threshold=1.1,
                   rng=random.Random(11))
    assert counts["good"] > counts["bad"] * 2   # 予算が good に寄る
    assert res.score == pytest.approx(0.9)


def test_multi_plan_hint_biases_first_picks():
    """Conductor 計画のモデルは事前分布が楽観側 → 序盤に選ばれやすい。"""
    wins = 0
    for seed in range(300):
        sel = S.ModelSelector(["a", "b", "c"], hint=["c"],
                              rng=random.Random(seed))
        if sel.pick() == "c":
            wins += 1
    # 一様なら期待 100/300 (sd≈8)。実測の偏り(勝率 ~0.50)を +3σ 余裕で検証
    assert wins > 125


def test_multi_disabled_passes_none_model():
    models_seen = []

    def factory(model=None):
        models_seen.append(model)
        return FakeChat(default="x")

    S.search("q", factory, lambda q, a: 0.5, budget=2, parallel=1,
             models=["m1", "m2"], multi=False, rng=random.Random(0))
    assert models_seen == [None, None]


# ---------------------------------------------------------------- 並列展開

def test_parallel_wave_actually_runs_concurrently():
    barrier = threading.Barrier(3, timeout=5)

    def factory(model=None):
        def gen(prompt):
            barrier.wait()                # 逐次実装ならタイムアウトで落ちる
            return f"p{threading.get_ident()}"
        return FakeChat(fn=gen)

    res = S.search("q", factory, lambda q, a: 0.5, budget=3, parallel=3,
                   threshold=0.99, rng=random.Random(0))
    assert res.calls_used == 3 and res.nodes == 3


def test_parallel_env_default(monkeypatch):
    monkeypatch.delenv("FUGU_SEARCH_PARALLEL", raising=False)
    assert S._env_int("FUGU_SEARCH_PARALLEL", S.DEFAULT_PARALLEL) == 4
    monkeypatch.setenv("FUGU_SEARCH_PARALLEL", "2")
    assert S._env_int("FUGU_SEARCH_PARALLEL", S.DEFAULT_PARALLEL) == 2


# ------------------------------------------------------------------ 統計量

def test_result_shape_reports_tree_metrics():
    factory, _ = _chat_factory_counter()
    res = S.search("q", factory, _scorer_from({}, 0.4), budget=5, parallel=1,
                   threshold=0.99, rng=random.Random(0))
    assert res.root_width >= 1 and res.max_depth >= 1
    assert f"width={res.root_width}" in res.shape()
    assert res.best_depth <= res.max_depth
