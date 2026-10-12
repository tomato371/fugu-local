# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""生存プローブ結果のメモ（setup 1 回の中だけ有効）。オフライン・実 API 不要。

2026-10-13 00:20 の再起動は受付可能まで 535 秒かかった。FUGU_NIM_ALWAYS の kimi-k3 は
同じ setup の中で ALWAYS 先行(200s)→規模順(20s)→冷え再確認(75s)→固定布陣(200s)と
繰り返し叩かれていた。結果が同じになる場合だけ再利用して呼び出し回数を減らす。
"""
import contextlib
import copy
import io

import pytest

import fugu_local as f

KIMI = "moonshotai/kimi-k3"


@pytest.fixture
def http(monkeypatch):
    """_nim_probe_http を差し替え、(model, timeout) の呼び出しを記録する。
    results[model] = (status, 所要秒)。仮想時計を所要秒だけ進める。"""
    calls = []
    results = {}
    clock = {"t": 1000.0}
    monkeypatch.setattr(f.time, "time", lambda: clock["t"])

    def fake(model, timeout=None):
        calls.append((model, timeout))
        status, took = results.get(model, ("ok", 0.5))
        clock["t"] += took
        return status

    monkeypatch.setattr(f, "_nim_probe_http", fake)
    monkeypatch.setattr(f, "_NIM_PROBE_MEMO", None)
    return calls, results


def _on(monkeypatch):
    monkeypatch.setattr(f, "_NIM_PROBE_MEMO", {})


def test_memo_is_off_outside_setup(http):
    calls, _ = http
    f._nim_probe("a/m", 20)
    f._nim_probe("a/m", 20)
    assert len(calls) == 2                       # 無効時は従来どおり毎回叩く


def test_slow_at_a_longer_timeout_is_reused_for_shorter_ones(http, monkeypatch):
    calls, results = http
    _on(monkeypatch)
    results[KIMI] = ("slow", 200.0)
    assert f._nim_probe(KIMI, 200) == "slow"
    assert f._nim_probe(KIMI, 20) == "slow"     # 規模順ループ
    assert f._nim_probe(KIMI, 75) == "slow"     # 冷え再確認
    assert f._nim_probe(KIMI, 200) == "slow"    # 固定布陣への退避（同等の timeout）
    assert calls == [(KIMI, 200)]


def test_slow_at_a_shorter_timeout_is_not_reused_for_longer_ones(http, monkeypatch):
    calls, results = http
    _on(monkeypatch)
    results[KIMI] = ("slow", 20.0)
    assert f._nim_probe(KIMI, 20) == "slow"
    results[KIMI] = ("ok", 148.0)
    assert f._nim_probe(KIMI, 200) == "ok"      # 長い待ちなら生きているかもしれないので叩く
    assert calls == [(KIMI, 20), (KIMI, 200)]


def test_ok_is_reused_only_when_it_would_also_pass_the_requested_timeout(http, monkeypatch):
    calls, results = http
    _on(monkeypatch)
    results[KIMI] = ("ok", 148.0)
    assert f._nim_probe(KIMI, 200) == "ok"
    assert f._nim_probe(KIMI, 200) == "ok"      # 148s で返ったので 200s 待ちなら同じ結果
    assert len(calls) == 1
    results[KIMI] = ("slow", 20.0)
    assert f._nim_probe(KIMI, 20) == "slow"     # 20s 待ちなら 148s の ok は流用できない
    assert len(calls) == 2


def test_gone_is_reused_for_any_timeout(http, monkeypatch):
    calls, results = http
    _on(monkeypatch)
    results["x/y"] = ("gone", 0.1)
    assert f._nim_probe("x/y", 20) == "gone"
    assert f._nim_probe("x/y", 200) == "gone"
    assert len(calls) == 1


def test_default_timeout_counts_as_the_default_probe_timeout(http, monkeypatch):
    calls, results = http
    _on(monkeypatch)
    results["a/m"] = ("slow", 20.0)
    f._nim_probe("a/m")                          # timeout=None → NIM_PROBE_TIMEOUT
    f._nim_probe("a/m", f.NIM_PROBE_TIMEOUT)
    assert len(calls) == 1


def test_models_are_memoised_independently(http, monkeypatch):
    calls, results = http
    _on(monkeypatch)
    f._nim_probe("a/m", 20)
    f._nim_probe("b/m", 20)
    assert [c[0] for c in calls] == ["a/m", "b/m"]


# --------------------------------------------------------------------------
# apply_nim_profile を通して: 2026-10-13 の再起動の再現
# --------------------------------------------------------------------------
_PROFILE_GLOBALS = (
    "DESIRED_PROPOSERS", "DESIRED_AGGREGATOR", "DESIRED_CONDUCTOR", "FALLBACK_MODEL",
    "PERSONA_MODELS", "PERSONA_IDENTITY", "MODEL_TO_PERSONA", "PROPOSER_PROFILES",
    "JP_AGGREGATOR", "JP_AGGREGATOR_STRONG", "AGGREGATOR_REASONING",
    "SECOND_OPINION_MODEL", "MODEL_CONFIG", "PARALLEL_PROPOSERS", "REASONING_MODELS",
    "ARBITER_MODEL", "SC_CHEAP_VOTES", "SC_PARALLEL", "NIM_MODEL_IDS", "NIM_STRUCTURED_OK",
    "VISION_MODEL", "NIM_VISION_MODEL",
)


@pytest.fixture
def setup_env(http, monkeypatch):
    for name in _PROFILE_GLOBALS:
        monkeypatch.setattr(f, name, copy.deepcopy(getattr(f, name)))
    monkeypatch.setattr(f, "NIM_API_KEY", "nvapi-TEST-KEY")
    monkeypatch.setattr(f, "NIM_LINEUP_FILE", "")
    monkeypatch.setattr(f, "NIM_AUTO_SELECT", True)
    monkeypatch.setattr(f, "CONDUCTOR_OVERRIDE", "")
    monkeypatch.setattr(f, "FAIL_SOFT", False)
    return http


def _apply():
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        assert f.apply_nim_profile() is True
    return buf.getvalue()


def test_setup_probes_kimi_once_when_it_stays_cold(setup_env, monkeypatch):
    """ALWAYS 先行 → 規模順 → 冷え再確認 → 固定布陣、の全てで kimi-k3 を叩くのは 1 回だけ。"""
    calls, results = setup_env
    monkeypatch.setattr(f, "NIM_ALWAYS", [KIMI])
    monkeypatch.setattr(f, "NIM_ALWAYS_TIMEOUT", 200)
    # カタログには kimi と死んだ機体だけ → 4 体そろわず固定布陣へ落ちる
    monkeypatch.setattr(f, "_nim_catalog", lambda: {KIMI, "x/dead-100b"})
    results[KIMI] = ("slow", 200.0)
    results["x/dead-100b"] = ("gone", 0.1)
    _apply()
    kimi_calls = [c for c in calls if c[0] == KIMI]
    assert kimi_calls == [(KIMI, 200)]
    assert KIMI not in f.DESIRED_PROPOSERS         # 結果（kimi を採らない）は従来と同じ


def test_setup_result_is_identical_with_and_without_the_memo(setup_env, monkeypatch):
    calls, results = setup_env
    monkeypatch.setattr(f, "NIM_ALWAYS", [KIMI])
    monkeypatch.setattr(f, "_nim_catalog", lambda: {KIMI, "x/dead-100b"})
    results[KIMI] = ("slow", 200.0)
    results["x/dead-100b"] = ("gone", 0.1)
    _apply()
    with_memo = (list(f.DESIRED_PROPOSERS), f.DESIRED_CONDUCTOR, f.DESIRED_AGGREGATOR)
    n_with = len(calls)

    # メモを常に無効にして同じ setup を回す（従来の挙動）
    monkeypatch.setattr(f, "_nim_probe", lambda m, timeout=None: f._nim_probe_http(m, timeout))
    calls.clear()
    _apply()
    without = (list(f.DESIRED_PROPOSERS), f.DESIRED_CONDUCTOR, f.DESIRED_AGGREGATOR)
    assert with_memo == without
    assert n_with < len(calls)                      # 呼び出し回数だけが減る


def test_memo_is_cleared_at_the_start_and_dropped_at_the_end(setup_env, monkeypatch):
    calls, results = setup_env
    monkeypatch.setattr(f, "NIM_ALWAYS", [])
    monkeypatch.setattr(f, "_nim_catalog", lambda: set())
    monkeypatch.setattr(f, "_NIM_PROBE_MEMO", {"stale/model": [("gone", 1, 0.0)]})
    _apply()
    assert f._NIM_PROBE_MEMO is None                # setup の外では無効に戻る
    f._nim_probe("stale/model", 20)                 # 前回の結果は残らない
    assert calls[-1][0] == "stale/model"


def test_memo_is_dropped_even_if_setup_raises(setup_env, monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(f, "_nim_catalog", _boom)
    with pytest.raises(RuntimeError):
        with contextlib.redirect_stdout(io.StringIO()):
            f.apply_nim_profile()
    assert f._NIM_PROBE_MEMO is None
