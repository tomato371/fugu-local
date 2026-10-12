# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""Conductor の退避と、conduct() のエラー時既定プラン（2026-10-12 の 1709 秒事故の再発防止）。

事故: Conductor の google/gemma-4-31b-it が死んでいる（起動時プローブ「応答なし」）のに
「Conductor 候補が全滅。... をそのまま使います」で採用され、conduct が 467 秒、
critic が 1085 秒張り付いた。ここで固定するのは次の点（オフライン・実 API 不要）:

  (a) 候補が全滅 → 採用済み（生存確認済み）の機体から Conductor を選ぶ。
      候補が生きていれば従来どおり。FUGU_CONDUCTOR 指定は尊重する。
  (d) conduct が __ERROR__ を受けたら引き直さず既定プラン（_fallback=True）へ落ちる。
      JSON が壊れただけの応答は従来どおり 1 回引き直す。
"""
import copy
import io
import contextlib

import pytest

import fugu_local as f

# apply_nim_profile が書き換える global（test_nim_offline.py §11 と同じ集合）。
_PROFILE_GLOBALS = (
    "DESIRED_PROPOSERS", "DESIRED_AGGREGATOR", "DESIRED_CONDUCTOR", "FALLBACK_MODEL",
    "PERSONA_MODELS", "PERSONA_IDENTITY", "MODEL_TO_PERSONA", "PROPOSER_PROFILES",
    "JP_AGGREGATOR", "JP_AGGREGATOR_STRONG", "AGGREGATOR_REASONING",
    "SECOND_OPINION_MODEL", "MODEL_CONFIG", "PARALLEL_PROPOSERS", "REASONING_MODELS",
    "ARBITER_MODEL", "SC_CHEAP_VOTES", "SC_PARALLEL", "NIM_MODEL_IDS", "NIM_STRUCTURED_OK",
    "VISION_MODEL", "NIM_VISION_MODEL",
)

GEMMA = "google/gemma-4-31b-it"
# 2026-10-12 の実運用の布陣（nim_lineup.json）
REAL_LINEUP = {
    "moonshotai/kimi-k3",
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/muse-glimmer-30b",
    "nvidia/nemotron-3-ultra-550b-a55b",
}


@pytest.fixture
def profile(monkeypatch):
    """apply_nim_profile を実 API なしで回すハーネス。global は終了時に全部戻す。"""
    for name in _PROFILE_GLOBALS:
        monkeypatch.setattr(f, name, copy.deepcopy(getattr(f, name)))
    monkeypatch.setattr(f, "NIM_API_KEY", "nvapi-TEST-KEY")
    monkeypatch.setattr(f, "NIM_LINEUP_FILE", "")
    monkeypatch.setattr(f, "NIM_AUTO_SELECT", True)
    monkeypatch.setattr(f, "NIM_ALWAYS", [])
    monkeypatch.setattr(f, "CONDUCTOR_OVERRIDE", "")

    def run(catalog, dead=()):
        dead = set(dead)
        monkeypatch.setattr(f, "_nim_catalog", lambda: set(catalog))
        monkeypatch.setattr(f, "_nim_probe",
                            lambda m, timeout=None: "gone" if m in dead else "ok")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            applied = f.apply_nim_profile()
        assert applied is True
        return buf.getvalue()

    return run


# --------------------------------------------------------------------------
# (a) Conductor の退避
# --------------------------------------------------------------------------
def test_dead_candidates_fall_back_to_a_live_picked_model(profile):
    """全滅時は死んだ gemma をそのまま使わず、採用済み機体から選ぶ。"""
    catalog = REAL_LINEUP | set(f.NIM_CONDUCTOR_CANDIDATES)
    log = profile(catalog, dead=f.NIM_CONDUCTOR_CANDIDATES)
    assert f.DESIRED_CONDUCTOR != GEMMA
    assert f.DESIRED_CONDUCTOR in REAL_LINEUP
    # 実測 148s の kimi-k3 / 19s の ultra は充てない。思考強制の nemotron-super より
    # 思考が強制されない muse-glimmer-30b を先にする（conduct は think=False・768 トークン）。
    assert f.DESIRED_CONDUCTOR == "meta/muse-glimmer-30b"
    assert f.DESIRED_CONDUCTOR in f.NIM_MODEL_IDS          # ディスパッチ漏れなし
    assert f.FALLBACK_MODEL == f.DESIRED_CONDUCTOR
    assert "候補が全滅" in log and f.DESIRED_CONDUCTOR in log
    assert "そのまま使います" not in log                     # 旧メッセージ（死んだ機体の採用）は出ない


def test_fallback_skips_a_picked_model_that_fails_the_reprobe(profile):
    catalog = REAL_LINEUP | set(f.NIM_CONDUCTOR_CANDIDATES)
    dead = set(f.NIM_CONDUCTOR_CANDIDATES) | {"meta/muse-glimmer-30b"}
    log = profile(catalog, dead=dead)
    # muse は採用段階から外れるので、残りの採用機体（実測の速い順）から選ばれる
    assert f.DESIRED_CONDUCTOR != GEMMA
    assert f.DESIRED_CONDUCTOR not in dead
    assert "候補が全滅" in log


def test_fixed_lineup_fallback_prefers_the_fastest_measured_model(profile):
    """カタログ不通（固定布陣）では実測 0.7s の nemotron-3-super を充てる。"""
    log = profile(set(), dead=f.NIM_CONDUCTOR_CANDIDATES)
    assert f.DESIRED_CONDUCTOR == "nvidia/nemotron-3-super-120b-a12b"
    assert "実測 0.7s" in log


def test_live_candidate_is_used_as_before(profile):
    """候補が生きていれば従来どおり（先頭の gemma）。退避ロジックは発動しない。"""
    catalog = REAL_LINEUP | set(f.NIM_CONDUCTOR_CANDIDATES)
    log = profile(catalog, dead=())
    assert f.DESIRED_CONDUCTOR == GEMMA
    assert "候補が全滅" not in log


def test_second_candidate_still_wins_over_fallback(profile):
    """gemma だけ死んでいるときは従来どおり次の軽量候補（退避ロジックは発動しない）。"""
    catalog = REAL_LINEUP | set(f.NIM_CONDUCTOR_CANDIDATES)
    log = profile(catalog, dead={GEMMA})
    assert f.DESIRED_CONDUCTOR == f.NIM_CONDUCTOR_CANDIDATES[1]
    assert "候補が全滅" not in log


def test_fugu_conductor_override_is_respected(profile, monkeypatch):
    """FUGU_CONDUCTOR 指定は、候補が全滅でも死んでいても従来どおり尊重する。"""
    catalog = REAL_LINEUP | set(f.NIM_CONDUCTOR_CANDIDATES)
    monkeypatch.setattr(f, "CONDUCTOR_OVERRIDE", GEMMA)
    log = profile(catalog, dead=f.NIM_CONDUCTOR_CANDIDATES)
    assert f.DESIRED_CONDUCTOR == GEMMA
    assert "FUGU_CONDUCTOR 指定" in log
    assert "候補が全滅" not in log


def test_fallback_chooser_ranking():
    """選び方: 遅い機体は最後 → 思考強制は後ろ → 実測の短い順 → 活性パラメータの小さい順。"""
    choose = f._nim_conductor_fallback
    assert choose([]) == ("", "")
    assert choose(["moonshotai/kimi-k3"])[0] == "moonshotai/kimi-k3"    # 他になければ遅くても
    fast_forced = "nvidia/nemotron-3-super-120b-a12b"
    plain = "meta/muse-glimmer-30b"
    assert choose([fast_forced, plain])[0] == plain
    assert choose(["moonshotai/kimi-k3", fast_forced])[0] == fast_forced
    # 思考強制の機体しかなければ実測の短いものを選び、注記で打ち切りの可能性を伝える
    model, note = choose(["nvidia/nemotron-3-ultra-550b-a55b", "openai/gpt-oss-20b", fast_forced])
    assert model == fast_forced and "既定プラン" in note


# --------------------------------------------------------------------------
# (d) conduct が __ERROR__ → 既定プラン
# --------------------------------------------------------------------------
@pytest.fixture
def conduct_env(monkeypatch):
    monkeypatch.setattr(f, "PROPOSERS", ["m1", "m2", "m3", "m4"])
    monkeypatch.setattr(f, "CONDUCTOR", "cond/model")
    calls = []

    def install(replies):
        it = iter(replies)

        def fake_ask(model, messages, temperature, **kw):
            calls.append(kw.get("label"))
            return next(it)

        monkeypatch.setattr(f, "ask", fake_ask)
        return calls

    return install


def test_conduct_error_falls_back_to_default_plan_without_retry(conduct_env, capsys):
    # エラー本文に JSON が混じっていても plan として読まない（mode=single に釣られない）
    calls = conduct_env(['__ERROR__: HTTP Error 500 {"mode": "single", "task_type": "math"}',
                         '{"mode": "single"}'])
    plan, raw = f.conduct("日本の首都は？")
    assert calls == ["conductor"]                    # 死んだモデルへの引き直しはしない
    assert plan["_fallback"] is True
    assert plan["mode"] == "moa" and plan["reason"].startswith("fallback")
    assert raw.startswith("__ERROR__")
    assert "既定プラン" in capsys.readouterr().out


def test_conduct_retry_error_also_falls_back(conduct_env):
    """JSON 崩れ → 引き直し → エラー、でも既定プランに落ちる。"""
    calls = conduct_env(["not json at all", "__ERROR__: timed out"])
    plan, _raw = f.conduct("q")
    assert calls == ["conductor", "conductor"]
    assert plan["_fallback"] is True


def test_conduct_bad_json_is_still_retried_once(conduct_env):
    """従来挙動: 成功応答だが JSON が壊れているときは 1 回だけ引き直す。"""
    calls = conduct_env(["garbage", '{"mode": "moa", "task_type": "knowledge", "rounds": 1}'])
    plan, _raw = f.conduct("q")
    assert calls == ["conductor", "conductor"]
    assert plan["_fallback"] is False


def test_conduct_healthy_reply_is_a_single_call(conduct_env):
    calls = conduct_env(['{"mode": "moa", "task_type": "knowledge", "rounds": 1}'])
    plan, _raw = f.conduct("q")
    assert calls == ["conductor"]
    assert plan["_fallback"] is False
