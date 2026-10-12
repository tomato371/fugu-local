# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""aggregate() の空提案の扱い（FAIL_SOFT=1 のときだけ変わる。無効時は旧挙動）。

背景: 推論モデルは思考だけ吐いて本文が空の提案を返すことがある。従来は __ERROR__ だけを
除いて空の提案も統合役の材料に混ぜ、統合が全滅した保険2でも先頭の空提案が
critique() を通れば「空の最終回答」を返し得た。

  FAIL_SOFT=1: 空でない提案が 1 つでもあれば、空の提案は材料・保険2の候補から外す。
               全て空なら従来どおりの流れ（外すものが無い）。
  FAIL_SOFT=0: 2 つの経路とも従来どおり（ベンチ＝fugu_local 直接実行はこちら）。

全プロポーザー __ERROR__ のとき aggregate が __ERROR__ を返す挙動は両モードで同じ
（救済は fugu-remote 側の連鎖が担う。二重救済を避けるためここでは何もしない）。
"""
import fugu_local as f


def _run(monkeypatch, proposals, *, fail_soft, ask_out):
    """aggregate を回し、(戻り値, 統合役に渡した user 文面の一覧) を返す。"""
    seen = []

    def fake_ask(model, messages, temperature, think=None, fmt=None,
                 label=None, num_predict=None):
        if label == "aggregator":
            seen.append(messages[-1]["content"])
        return ask_out

    monkeypatch.setattr(f, "FAIL_SOFT", fail_soft)
    monkeypatch.setattr(f, "CODE_EXECUTION", False)
    monkeypatch.setattr(f, "ask", fake_ask)
    # 保険2: critique は「どの提案でも ok」にして、先頭の候補がそのまま返る状態にする
    monkeypatch.setattr(f, "critique", lambda q, a: (True, ""))
    return f.aggregate("Q?", proposals), seen


# --------------------------------------------------------------------------
# 統合役に渡す材料
# --------------------------------------------------------------------------
def test_fail_soft_on_drops_empty_proposals_from_the_aggregator_input(monkeypatch):
    out, seen = _run(monkeypatch, [("m1", ""), ("m2", "real answer"), ("m3", "  \n")],
                     fail_soft=True, ask_out="merged")
    assert out == "merged"
    assert "Answer A:\nreal answer" in seen[0]
    assert "Answer B" not in seen[0]


def test_fail_soft_off_keeps_empty_proposals_in_the_aggregator_input(monkeypatch):
    out, seen = _run(monkeypatch, [("m1", ""), ("m2", "real answer")],
                     fail_soft=False, ask_out="merged")
    assert out == "merged"
    assert "Answer A:\n\n" in seen[0]          # 空の提案が A として残る（旧挙動）
    assert "Answer B:\nreal answer" in seen[0]


def test_fail_soft_on_all_empty_keeps_the_old_flow(monkeypatch):
    out, seen = _run(monkeypatch, [("m1", ""), ("m2", "")],
                     fail_soft=True, ask_out="merged")
    assert out == "merged"
    assert "Answer A" in seen[0] and "Answer B" in seen[0]


# --------------------------------------------------------------------------
# 保険2（統合が全滅したとき提案から直接返す）
# --------------------------------------------------------------------------
def test_fail_soft_on_insurance2_prefers_nonempty_candidate(monkeypatch):
    out, _ = _run(monkeypatch, [("m1", ""), ("m2", "real answer")],
                  fail_soft=True, ask_out="")
    assert out == "real answer"


def test_fail_soft_off_insurance2_can_return_the_empty_first_proposal(monkeypatch):
    out, _ = _run(monkeypatch, [("m1", ""), ("m2", "real answer")],
                  fail_soft=False, ask_out="")
    assert out == ""                           # 旧挙動（空を返し得る）を固定する


def test_fail_soft_on_insurance2_all_empty_still_returns_something_old_way(monkeypatch):
    out, _ = _run(monkeypatch, [("m1", ""), ("m2", "")],
                  fail_soft=True, ask_out="")
    assert out == ""                           # 外すものが無いので従来どおり


# --------------------------------------------------------------------------
# 全提案 __ERROR__ は両モードで同じ（local では救済しない）
# --------------------------------------------------------------------------
def test_all_error_proposals_return_error_in_both_modes(monkeypatch):
    for fs in (False, True):
        out, seen = _run(monkeypatch, [("m1", "__ERROR__: a"), ("m2", "__ERROR__: b")],
                         fail_soft=fs, ask_out="never")
        assert out.startswith("__ERROR__")
        assert seen == []                      # 統合役は呼ばれない
