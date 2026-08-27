# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
r"""canonicalize_math_answer が「採点側」ではなく「生成側」に配線されていることを固定する
(2026-08-27)。

docs/strict-grading.md の「配線先の問題」に対応するテスト。canonicalize は当初
bench_fugu.grade_item の採点直前にしか無く、ベンチの中でしか効いていなかった。
実運用では採点器は他人が握るのだから、生成側で正規化していなければ点は資産にならない。

ここで固定するのは次の 4 点:
  1. solve_verifiable が「報告する答え」を正規表記で返すこと
  2. 「投じられた生の表記」は votes に残ること(票は値について、報告は形式について)
  3. mcq(選択肢ラベル)は正規化しないこと
  4. 本文と報告で表記が違うとき、戻り値のテキストにも正規表記が載ること
"""
import pytest

import fugu_local as f
from fugu_canonical import canonicalize_math_answer

CANONICAL_35_9 = r"-\frac{35}{9}"


def _run_sc(monkeypatch, boxed, task_type="math"):
    """ask をモックして solve_verifiable を全会一致で確定させる。"""
    def fake_ask(model, messages, temperature, think=None, fmt=None,
                 label=None, num_predict=None, num_ctx=None):
        return "reasoning...\n" + r"\boxed{" + boxed + "}"

    monkeypatch.setattr(f, "ask", fake_ask)
    monkeypatch.setattr(f, "PROPOSERS", ["m1", "m2"])
    monkeypatch.setattr(f, "REASONING_MODELS", ["m1", "m2"])
    monkeypatch.setattr(f, "SC_CHEAP_VOTES", 0)
    monkeypatch.setattr(f, "SC_POT", False)
    return f.solve_verifiable("q", task_type)


@pytest.mark.parametrize("raw,canonical", [
    ("-35/9", r"-\frac{35}{9}"),
    ("448/15625", r"\frac{448}{15625}"),
    (r"\dfrac{\pi}{6}", r"\frac{\pi}{6}"),
    ("0.09", r"\frac{9}{100}"),
    ("-sqrt(3)", r"-\sqrt{3}"),
])
def test_reported_answer_is_canonical(monkeypatch, raw, canonical):
    """1) 報告する答えは正規表記。これが無いと外部の採点器には素の表記が渡る。"""
    res = _run_sc(monkeypatch, raw)
    assert res is not None
    assert res["answer"] == canonical


def test_votes_keep_the_ballot_as_cast(monkeypatch):
    """2) votes は投じられた生の表記のまま。投票そのものは触っていない。"""
    res = _run_sc(monkeypatch, "-35/9")
    assert res is not None
    assert "-35/9" in res["votes"], "投票記録が報告用の表記に書き換わっている"
    assert res["answer"] == r"-\frac{35}{9}"


def test_canonicalization_preserves_value(monkeypatch):
    """正規化は表記だけを変え、値は変えない(採点の改竄ではないことの担保)。"""
    res = _run_sc(monkeypatch, "-35/9")
    assert f.answers_equivalent(res["answer"], "-35/9")


def test_mcq_is_not_canonicalized(monkeypatch):
    """3) mcq は選択肢ラベルなので正規化を通さない。"""
    seen = []

    def spy(s):
        seen.append(s)
        return canonicalize_math_answer(s)

    monkeypatch.setattr(f, "_canonicalize_answer", spy)
    res = _run_sc(monkeypatch, "B", task_type="mcq")
    assert res is not None
    assert res["answer"] == "B"
    assert seen == [], "mcq でも正規化が呼ばれている"


def test_answer_text_carries_the_canonical_form(monkeypatch):
    r"""4) 本文が素の表記でも、戻り値のテキストに正規表記の \boxed{} が載る。

    テキストしか読まない相手(人間・外部の採点器・answer_value=None で採点する
    bench の run_fugu 経路)にも正規表記が渡ることを保証する。
    """
    monkeypatch.setattr(f, "SC_ENABLED", True)
    monkeypatch.setattr(f, "TRINITY_ENABLED", False)
    monkeypatch.setattr(f, "solve_verifiable", lambda *a, **k: {
        "answer": r"-\frac{35}{9}",
        "text": r"計算すると答えは \boxed{-35/9} です。",
        "votes": {"-35/9": 6},
        "n_samples": 6,
    })
    out = f.fugu_answer("q", plan={"task_type": "math"}, history=[])
    assert r"\boxed{-\frac{35}{9}}" in out

    # 注意: fugu 自身の extract_final_answer は LaTeX 分数を a/b へ「戻す」正規化を
    # 持っているため、読み戻すと素の表記になる。テキストに載っているのは正規表記であり、
    # 生テキストの最後の boxed を取る外部の採点器はそちらを受け取る。値は一致しており、
    # 投票結果を握りつぶしてはいない。
    # なお bench_fugu.grade_item 側の canonicalize 呼び出しは、この「読み戻すと素の表記に
    # 戻る」性質があるので冗長ではなく、answer_value=None の run_fugu 経路には必須である。
    assert f.answers_equivalent(f.extract_final_answer(out, "math"),
                                CANONICAL_35_9)


def test_no_note_when_body_already_matches(monkeypatch):
    """表記まで一致していれば注記は付けない(出力を無駄に汚さない)。"""
    monkeypatch.setattr(f, "SC_ENABLED", True)
    monkeypatch.setattr(f, "TRINITY_ENABLED", False)
    monkeypatch.setattr(f, "solve_verifiable", lambda *a, **k: {
        "answer": "42",
        "text": r"答えは \boxed{42} です。",
        "votes": {"42": 6},
        "n_samples": 6,
    })
    out = f.fugu_answer("q", plan={"task_type": "math"}, history=[])
    assert "自己一貫性投票による最終解答" not in out


def test_canonicalize_is_idempotent():
    """bench_fugu 側にも保険として残っているため、二重適用で壊れないこと。"""
    for s in ["-35/9", r"-\frac{35}{9}", "0.09", r"\frac{9}{100}",
              "-sqrt(3)", r"-\sqrt{3}", "B", "16", "", None]:
        once = canonicalize_math_answer(s)
        assert canonicalize_math_answer(once) == once
