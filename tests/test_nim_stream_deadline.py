# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""stream_max_s を「ask() 1 回全体の期限」にする変更と、Critic 失敗時の扱い
（2026-10-12 の 1709 秒事故の再発防止・その2）。オフライン・実 API 不要。

  (b) stream_max_s 指定時: 再送（通常 / 429・503 / 空応答）は期限を過ぎるなら行わず __ERROR__。
      1 ストリームの壁時計はセマフォ取得後から測る。未指定時は従来の再送回数・挙動のまま。
  (c) verify_single: Critic が __ERROR__ → think=True で再検算せず未検証のまま採用。
      Critic が正常応答したときの挙動は変えない。
  + Conductor / Critic への stream_max_s の配線（NIM のときだけ・無効化可）と、
    クールダウン / 再送の可観測性（print）。
"""
import email.message
import io
import json
import urllib.error

import pytest

import fugu_local as f

MSGS = [{"role": "user", "content": "hi"}]
MODEL = "test/model"


# ---------- モック部品（test_nim_offline.py と同型）----------
class FakeResponse:
    def __init__(self, sse_lines=None):
        self._sse = sse_lines or []
        self.status = 200

    def read(self):
        return b""

    def __iter__(self):
        return iter(self._sse)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok_body(content="hello"):
    lines = [("data: " + json.dumps({"choices": [{"delta": {"content": content}}]}) + "\n").encode(),
             ("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
              + "\n").encode(),
             b"data: [DONE]\n"]
    return FakeResponse(sse_lines=lines)


def empty_body():
    return FakeResponse(sse_lines=[b"data: [DONE]\n"])


def http_error(code, body=b"err", headers=None):
    def make():
        h = email.message.Message()
        for k, v in (headers or {}).items():
            h[k] = str(v)
        raise urllib.error.HTTPError("http://x", code, "err", h, io.BytesIO(body))
    return make


def slow_sse(advance, lines):
    """イテレートのたびに仮想時計を advance 秒進める SSE モック。"""
    class _R(FakeResponse):
        def __iter__(self):
            def gen():
                for line in lines:
                    f.time.sleep(advance)
                    yield line
            return gen()
    return _R()


@pytest.fixture
def net(monkeypatch):
    """urlopen / time.sleep / time.time を仮想時計に差し替えて fn() を回す。
    戻り値 run(responses, fn) -> (結果, 送信した Request 群, urlopen の timeout 群, sleep 秒群)"""
    monkeypatch.setattr(f, "NIM_API_KEY", "nvapi-TEST-KEY")
    monkeypatch.setattr(f, "NIM_BUDGET", 0)
    monkeypatch.setattr(f, "NIM_MODEL_IDS", {MODEL, "test/conductor"})
    monkeypatch.setattr(f, "NIM_STRUCTURED_OK", set())
    monkeypatch.setattr(f, "SHOW_TIMING", False)
    monkeypatch.setattr(f, "_nim_count_request", lambda: None)   # 課金カウンタ（ファイル書込）を切る

    def run(responses, fn):
        sent, timeouts, sleeps = [], [], []
        clock = [1_000_000.0]

        def fake_open(req, timeout=None):
            sent.append(req)
            timeouts.append(timeout)
            r = responses[min(len(sent) - 1, len(responses) - 1)]
            return r() if callable(r) else r

        def fake_sleep(s):
            sleeps.append(s)
            clock[0] += max(float(s), 0.001)

        monkeypatch.setattr(f.urllib.request, "urlopen", fake_open)
        monkeypatch.setattr(f.time, "sleep", fake_sleep)
        monkeypatch.setattr(f.time, "time", lambda: clock[0])
        f._NIM_COOLDOWN.clear()
        f._NIM_COOLDOWN_SEEN.clear()
        try:
            return fn(), sent, timeouts, sleeps
        finally:
            f._NIM_COOLDOWN.clear()
            f._NIM_COOLDOWN_SEEN.clear()

    return run


# --------------------------------------------------------------------------
# (b) stream_max_s = ask() 1 回全体の期限
# --------------------------------------------------------------------------
def test_deadline_stops_normal_retries(net):
    out, sent, _t, sleeps = net([http_error(500)],
                                lambda: f.ask(MODEL, MSGS, 0.5, stream_max_s=5))
    assert out.startswith("__ERROR__") and "deadline" in out
    assert len(sent) == 2                       # 0s 失敗→2s 待って再送→2s 失敗→次の 5s 待ちは期限超え
    assert sleeps == [2]


def test_no_deadline_keeps_the_legacy_retry_count(net):
    out, sent, _t, sleeps = net([http_error(500)], lambda: f.ask(MODEL, MSGS, 0.5))
    assert out.startswith("__ERROR__") and "deadline" not in out
    assert len(sent) == f.ASK_RETRY_ATTEMPTS
    assert sleeps == list(f.ASK_RETRY_BACKOFF)


def test_deadline_stops_rate_limit_retries(net):
    out, sent, _t, sleeps = net([http_error(429)],
                                lambda: f.ask(MODEL, MSGS, 0.5, stream_max_s=30))
    assert out.startswith("__ERROR__") and "deadline" in out and "429" in out
    assert len(sent) == 2 and sleeps == [20.0]  # 20s 待って再送、次の 40s 待ちは期限超え


def test_no_deadline_keeps_the_legacy_rate_limit_budget(net):
    out, sent, _t, sleeps = net([http_error(429), http_error(429), ok_body("calm")],
                                lambda: f.ask(MODEL, MSGS, 0.5))
    assert out == "calm" and sleeps[:2] == [20.0, 40.0] and len(sent) == 3


def test_deadline_stops_empty_response_retries(net):
    out, sent, _t, _s = net([empty_body()], lambda: f.ask(MODEL, MSGS, 0.5, stream_max_s=3))
    assert out.startswith("__ERROR__") and "deadline" in out
    assert len(sent) == 1                       # 4 秒待つと期限(3s)を超えるので再送しない


def test_no_deadline_keeps_the_legacy_empty_retries(net):
    out, sent, _t, sleeps = net([empty_body()], lambda: f.ask(MODEL, MSGS, 0.5))
    assert len(sent) == 1 + f.NIM_EMPTY_RETRIES
    assert sleeps == [f.NIM_EMPTY_RETRY_WAIT] * f.NIM_EMPTY_RETRIES
    assert out == ""                            # 従来どおり（未指定時は空のまま返る）


def test_deadline_not_hit_still_recovers(net):
    """期限内に回復できる再送は従来どおり行う。"""
    out, sent, _t, _s = net([http_error(500), ok_body("fine")],
                            lambda: f.ask(MODEL, MSGS, 0.5, stream_max_s=180))
    assert out == "fine" and len(sent) == 2


def test_cooldown_wait_respects_the_deadline(net):
    def go():
        f._NIM_COOLDOWN[MODEL] = f.time.time() + 120
        return f.ask(MODEL, MSGS, 0.5, stream_max_s=8)
    out, sent, _t, _s = net([ok_body()], go)
    assert out.startswith("__ERROR__") and "deadline" in out
    assert sent == []                           # クールダウン待ちだけで期限を使い切り、送信しない


def test_stream_clock_starts_after_the_semaphore(net, monkeypatch):
    """1 ストリームの壁時計はセマフォ取得後から。送信待ちの 100 秒を生成時間に数えない。"""
    class SlowSlot:
        def acquire(self, timeout=None):
            f.time.sleep(100)                   # 他の呼び出しの送信を待った
            return True

        def release(self):
            pass

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(f, "_NIM_SEMAPHORE", SlowSlot())
    lines = ok_body("done")._sse
    out, _s, _t, _sl = net([slow_sse(40, lines)],
                           lambda: f.ask(MODEL, MSGS, 0.5, stream_max_s=150))
    assert out == "done"                        # ストリーム自体は 3×40=120s < 150s
    out, _s, _t, _sl = net([slow_sse(40, lines)],
                           lambda: f.ask(MODEL, MSGS, 0.5, stream_max_s=100))
    assert out.startswith("__ERROR__") and "stream exceeded 100s" in out


def test_socket_timeout_is_capped_only_when_specified(net):
    _o, _s, timeouts, _sl = net([ok_body()], lambda: f.ask(MODEL, MSGS, 0.5, stream_max_s=45))
    assert timeouts == [45]
    _o, _s, timeouts, _sl = net([ok_body()], lambda: f.ask(MODEL, MSGS, 0.5))
    assert timeouts == [f.NIM_TIMEOUT]


# --- 可観測性 ---
def test_cooldown_wait_is_announced_once(net, capsys):
    def go():
        f._NIM_COOLDOWN[MODEL] = f.time.time() + 12      # 5 + 5 + 2 秒の 3 周で待つ
        return f.ask(MODEL, MSGS, 0.5)
    out, _s, _t, _sl = net([ok_body("hi")], go)
    assert out == "hi"
    said = capsys.readouterr().out
    assert said.count("クールダウン中") == 1
    assert MODEL in said and "秒待って" in said


def test_retry_prints_name_reason_wait_and_attempt(net, capsys):
    net([http_error(500), ok_body("x")], lambda: f.ask(MODEL, MSGS, 0.5))
    said = capsys.readouterr().out
    assert f"{MODEL}: HTTP 500" in said and "2秒待って再送" in said and "試行 2/4" in said
    net([http_error(429, headers={"Retry-After": "3"}), ok_body("x")],
        lambda: f.ask(MODEL, MSGS, 0.5))
    said = capsys.readouterr().out
    assert "HTTP 429" in said and "3秒待って再送" in said and "レート制限リトライ 1/" in said
    net([empty_body(), ok_body("x")], lambda: f.ask(MODEL, MSGS, 0.5))
    said = capsys.readouterr().out
    assert "空応答" in said and "空応答再送 1/" in said


# --------------------------------------------------------------------------
# Conductor / Critic への stream_max_s の配線
# --------------------------------------------------------------------------
def test_short_job_kwargs_only_for_nim_and_can_be_disabled(monkeypatch):
    monkeypatch.setattr(f, "NIM_MODEL_IDS", {"test/conductor"})
    monkeypatch.setattr(f, "CONDUCTOR", "test/conductor")
    assert f._short_job_kwargs() == {"stream_max_s": f.CONDUCTOR_STREAM_MAX_S}
    assert f._short_job_kwargs(think_recheck=True) == {
        "stream_max_s": f.CRITIC_THINK_STREAM_MAX_S}
    assert f.CONDUCTOR_STREAM_MAX_S == 180 and f.CRITIC_THINK_STREAM_MAX_S > f.CONDUCTOR_STREAM_MAX_S
    monkeypatch.setattr(f, "CONDUCTOR", "qwen3:4b")                   # Ollama は何も渡さない
    assert f._short_job_kwargs() == {} and f._short_job_kwargs(think_recheck=True) == {}
    monkeypatch.setattr(f, "CONDUCTOR", "test/conductor")
    monkeypatch.setattr(f, "CONDUCTOR_STREAM_MAX_S", 0)               # 0 で無効化
    assert f._short_job_kwargs() == {}


def test_conduct_and_critic_pass_stream_max_s_on_nim(monkeypatch):
    monkeypatch.setattr(f, "NIM_MODEL_IDS", {"test/conductor"})
    monkeypatch.setattr(f, "CONDUCTOR", "test/conductor")
    monkeypatch.setattr(f, "PROPOSERS", ["m1", "m2", "m3", "m4"])
    seen = []

    def fake_ask(model, messages, temperature, **kw):
        seen.append((kw.get("label"), kw.get("think"), kw.get("stream_max_s")))
        return '{"mode": "moa", "task_type": "knowledge", "rounds": 1}' \
            if kw.get("label") == "conductor" else '{"ok": true, "issue": ""}'

    monkeypatch.setattr(f, "ask", fake_ask)
    f.conduct("q")
    f._critic_judge("q", "a", think=False)
    f._critic_judge("q", "a", think=True)
    assert seen == [("conductor", False, f.CONDUCTOR_STREAM_MAX_S),
                    ("critic", False, f.CONDUCTOR_STREAM_MAX_S),
                    ("critic", True, f.CRITIC_THINK_STREAM_MAX_S)]


def test_non_nim_backend_passes_nothing_extra(monkeypatch):
    """Ollama 経路・偽 ask は従来どおり stream_max_s を受け取らない。"""
    monkeypatch.setattr(f, "NIM_MODEL_IDS", set())
    monkeypatch.setattr(f, "CONDUCTOR", "qwen3:4b")
    monkeypatch.setattr(f, "PROPOSERS", ["m1", "m2", "m3", "m4"])
    seen = []

    def fake_ask(model, messages, temperature, think=None, fmt=None, label=None,
                 num_predict=None, num_ctx=None, images=None):     # stream_max_s を受けない旧形
        seen.append(label)
        return '{"mode": "moa", "task_type": "knowledge", "rounds": 1}' \
            if label == "conductor" else '{"ok": true, "issue": ""}'

    monkeypatch.setattr(f, "ask", fake_ask)
    f.conduct("q")
    f._critic_judge("q", "a", think=True)
    assert seen == ["conductor", "critic"]


# --------------------------------------------------------------------------
# (c) verify_single: Critic が __ERROR__ → 再検算せず採用
# --------------------------------------------------------------------------
@pytest.fixture
def verify_env(monkeypatch):
    monkeypatch.setattr(f, "CONDUCTOR", "cond/model")
    monkeypatch.setattr(f, "SECOND_OPINION_MODEL", "second/model")
    monkeypatch.setattr(f, "_SECOND_OPINION_DISABLED", False)
    calls = []

    def install(critic_replies, second=(True, "")):
        it = iter(critic_replies)

        def fake_ask(model, messages, temperature, **kw):
            calls.append((kw.get("label"), kw.get("think")))
            return next(it)

        monkeypatch.setattr(f, "ask", fake_ask)
        monkeypatch.setattr(f, "second_opinion", lambda q, a: second)
        return calls

    return install


def test_critic_error_adopts_without_rechecking(verify_env, capsys):
    calls = verify_env(["__ERROR__: stream exceeded 180s (wall clock cap)"])
    assert f.verify_single("日本の首都は？", "東京です。") == (True, "")
    assert calls == [("critic", False)]          # think=True の再検算を投げない
    assert "未検証のまま採用" in capsys.readouterr().out


def test_critic_error_with_both_critics_failing_adopts(verify_env):
    calls = verify_env(["__ERROR__: boom"], second=(False, "critic call failed: __ERROR__: x"))
    assert f.verify_single("q", "a") == (True, "")
    assert calls == [("critic", False)]


def test_critic_error_does_not_hide_a_real_second_opinion_doubt(verify_env):
    calls = verify_env(["__ERROR__: boom"], second=(False, "answer is wrong"))
    ok, issue = f.verify_single("q", "a")
    assert ok is False and "answer is wrong" in issue
    assert calls == [("critic", False)]


def test_critic_ok_reply_behaves_as_before(verify_env):
    calls = verify_env(['{"ok": true, "issue": ""}'])
    assert f.verify_single("q", "a") == (True, "")
    assert calls == [("critic", False)]


def test_critic_real_ng_still_triggers_the_thinking_recheck(verify_env):
    calls = verify_env(['{"ok": false, "issue": "wrong"}', '{"ok": true, "issue": ""}'])
    assert f.verify_single("q", "a") == (True, "")
    assert calls == [("critic", False), ("critic", True)]
    calls.clear()
    verify_env(['{"ok": false, "issue": "wrong"}', '{"ok": false, "issue": "still wrong"}'])
    assert f.verify_single("q", "a") == (False, "still wrong")


def test_recheck_failure_after_a_real_ng_is_unchanged(verify_env):
    """最初の審査が正常に NG を出した後の再検算が失敗した場合は従来どおり NG（エスカレーション）。"""
    verify_env(['{"ok": false, "issue": "wrong"}', "__ERROR__: boom"])
    ok, issue = f.verify_single("q", "a")
    assert ok is False and issue.startswith("critic call failed")
