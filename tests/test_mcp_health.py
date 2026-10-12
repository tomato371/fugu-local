# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_health のジョブサーバー節のオフラインテスト(HTTP は全部スタブ、ネット不要)。"""
import urllib.error

import pytest

import fugu_mcp as M

HEALTH = {
    "backend": "nim", "backend_up": True, "backend_url": "https://integrate.api.nvidia.com/v1",
    "setup_done": True, "setup_ok": True, "queued": 2, "running": ["j1"],
    "current": "j1", "phase": "aggregate", "parallel": 1, "version": "1.2.3",
    "search_ready": True,
}


# fugu_remote /api/stats の実形状(_compute_stats)
REAL_STATS = {
    "generated_at": "2026-10-13T12:00:00", "subtasks": False,
    "windows": {
        "24h": {"total": 40,
                "kinds": {"ok": 33, "empty": 2, "failed": 1, "salvaged": 3,
                          "partial": 1, "cancelled": 0},
                "ok_elapsed_sec": {"n": 33, "median": 41.5, "p90": 120.0},
                "latest_bad": [{"id": "20261013-110000-ab", "kind": "empty",
                                "reason": "回答が空"}]},
        "7d": {"total": 200,
               "kinds": {"ok": 180, "empty": 4, "failed": 3, "broken": 2,
                         "salvaged": 11},
               "ok_elapsed_sec": {"n": 180, "median": 50.0, "p90": 130.0},
               "latest_bad": []},
    },
}


def _stub(monkeypatch, routes):
    """routes: {path: (status, obj, err)}。未登録の path は 404。呼ばれた path を記録。"""
    calls = []

    def fake(path, timeout=5.0):
        calls.append(path)
        return routes.get(path, (404, None, "HTTP 404"))

    monkeypatch.setattr(M, "_http_get_json", fake)
    return calls


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("FUGU_JOB_SERVER_URL", "FUGU_REMOTE_PORT", "FUGU_REMOTE_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def test_server_up_with_stats(monkeypatch):
    health = dict(HEALTH, lineup=["a", "b"], conductor={"model": "c"},
                  bad_outcomes_24h=3, api_token="SECRET-VALUE")
    stats = REAL_STATS
    calls = _stub(monkeypatch, {"/api/health": (200, health, ""),
                                "/api/stats": (200, stats, "")})
    text = M._job_server_text()
    assert calls == ["/api/health", "/api/stats"]
    assert "http://127.0.0.1:8800" in text
    assert "[OK]" in text
    assert "backend: \"nim\"" in text and "backend_up: true" in text
    assert "queued: 2" in text and "version: \"1.2.3\"" in text
    assert "lineup: " in text and "conductor: " in text and "bad_outcomes_24h: 3" in text
    assert "24h: 計40件 / 問題3件 (failed=1, empty=2)" in text
    assert "7d: 計200件 / 問題9件" in text
    assert "SECRET-VALUE" not in text            # 秘密っぽいキーは出さない


def test_server_up_stats_404(monkeypatch):
    _stub(monkeypatch, {"/api/health": (200, HEALTH, "")})   # /api/stats は 404
    text = M._job_server_text()
    assert "[OK]" in text
    assert "/api/stats は 404" in text
    assert "NG" not in text


def test_server_down_is_reported_clearly(monkeypatch):
    calls = _stub(monkeypatch, {"/api/health": (None, None, "URLError: [WinError 10061]")})
    text = M._job_server_text()
    assert "到達できません" in text and "WinError 10061" in text
    assert calls == ["/api/health"]              # 落ちているなら stats は叩かない


def test_unauthorized_and_bad_responses(monkeypatch):
    _stub(monkeypatch, {"/api/health": (401, None, "HTTP 401")})
    assert "FUGU_REMOTE_TOKEN" in M._job_server_text()
    _stub(monkeypatch, {"/api/health": (503, None, "HTTP 503")})
    assert "応答が異常" in M._job_server_text()
    _stub(monkeypatch, {"/api/health": (200, ["not", "a", "dict"], "")})
    assert "応答が異常" in M._job_server_text()


def test_backend_down_flagged(monkeypatch):
    _stub(monkeypatch, {"/api/health": (200, dict(HEALTH, backend_up=False), "")})
    assert "backend が落ちています" in M._job_server_text()


@pytest.mark.parametrize("health", [{}, {"unexpected": 1}, {"queued": None, "running": None}])
def test_missing_or_unknown_keys_do_not_crash(monkeypatch, health):
    _stub(monkeypatch, {"/api/health": (200, health, "")})
    assert "[OK]" in M._job_server_text()


@pytest.mark.parametrize("stats", [
    [], "x", 5, {}, {"foo": "bar"}, {"total": "many"},
    {"total": 3, "bad": 1}, {"last_24h": {"total": 9, "bad_by_kind": {"x": 1}}},
    {"total": 4, "bad": "weird"},
])
def test_stats_line_is_defensive(stats):
    assert isinstance(M._stats_line(stats), str)


def test_stats_line_values():
    assert M._stats_line({"last_24h": {"total": 9, "bad_by_kind": {"x": 1, "y": 2}}}) \
        == "24h: total=9 / bad=3 (x=1, y=2)"
    assert M._stats_line({"total": 3, "bad": 1}) == "24h: total=3 / bad=1"


def test_exception_inside_is_contained(monkeypatch):
    def boom(path, timeout=5.0):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(M, "_http_get_json", boom)
    assert "kaboom" in M._job_server_text()


def test_health_text_combines_ollama_and_job_server(monkeypatch):
    monkeypatch.setattr(M, "_ollama_health_text", lambda: "Ollama OK")
    _stub(monkeypatch, {"/api/health": (200, HEALTH, "")})
    text = M._health_text()
    assert text.startswith("Ollama OK")
    assert "== ジョブサーバー" in text


def test_health_tool_reports_down_server_without_error(monkeypatch):
    monkeypatch.setattr(M, "_ollama_health_text", lambda: "Ollama OK")
    _stub(monkeypatch, {"/api/health": (None, None, "URLError: refused")})
    res = M.call_tool("fugu_health", {}, M.Jobs(runner=lambda *a: ""))
    assert "isError" not in res
    assert "到達できません" in res["content"][0]["text"]


def test_base_url_resolution(monkeypatch):
    assert M._job_server_base() == "http://127.0.0.1:8800"
    monkeypatch.setenv("FUGU_REMOTE_PORT", "9000")
    assert M._job_server_base() == "http://127.0.0.1:9000"
    monkeypatch.setenv("FUGU_JOB_SERVER_URL", "http://100.1.2.3:8800/")
    assert M._job_server_base() == "http://100.1.2.3:8800"


class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_get_json_sends_token_header_and_parses(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["timeout"] = req.full_url, timeout
        seen["tok"] = req.get_header("X-fugu-token")
        return _Resp(200, '{"a": 1}'.encode())

    monkeypatch.setattr(M, "_urlopen", fake_urlopen)
    monkeypatch.setenv("FUGU_REMOTE_TOKEN", "tok123")
    assert M._http_get_json("/api/health") == (200, {"a": 1}, "")
    assert seen == {"url": "http://127.0.0.1:8800/api/health", "timeout": 10.0,
                    "tok": "tok123"}


def test_http_get_json_maps_errors(monkeypatch):
    def http_err(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)

    def conn_err(req, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(M, "_urlopen", http_err)
    assert M._http_get_json("/api/stats") == (404, None, "HTTP 404")
    monkeypatch.setattr(M, "_urlopen", conn_err)
    status, obj, err = M._http_get_json("/api/health")
    assert status is None and obj is None and "refused" in err
    monkeypatch.setattr(M, "_urlopen",
                        lambda req, timeout=None: _Resp(200, b"<html>"))
    assert M._http_get_json("/x")[0] == 200 and M._http_get_json("/x")[1] is None


def test_stats_line_real_shape():
    assert M._stats_line(REAL_STATS).splitlines() == [
        "24h: 計40件 / 問題3件 (failed=1, empty=2) / 救済4件 (salvaged=3, partial=1)"
        " / ok所要 中央値41.5秒 / p90 120.0秒 (n=33)",
        "7d: 計200件 / 問題9件 (failed=3, empty=4, broken=2) / 救済11件 (salvaged=11)",
    ]


def test_stats_line_real_shape_quiet_and_empty():
    quiet = {"windows": {
        "24h": {"total": 0, "kinds": {},
                "ok_elapsed_sec": {"n": 0, "median": None, "p90": None},
                "latest_bad": []},
        "7d": {"total": 5, "kinds": {"ok": 5}}}}
    assert M._stats_line(quiet).splitlines() == ["24h: 計0件 / 問題0件",
                                                 "7d: 計5件 / 問題0件"]


@pytest.mark.parametrize("stats", [
    {"windows": {}}, {"windows": "x"}, {"windows": {"24h": 5}},
    {"windows": {"24h": {}}}, {"windows": {"24h": {"total": "n", "kinds": [1],
                                                  "ok_elapsed_sec": "z"}}},
])
def test_stats_line_real_shape_defensive(stats):
    assert isinstance(M._stats_line(stats), str)


# ---------------------------------------------------- 査読指摘の修正分

def test_health_uses_longer_timeout_than_stats(monkeypatch):
    seen = {}

    def fake(path, timeout=None):
        seen[path] = timeout
        return {"/api/health": (200, HEALTH, "")}.get(path, (404, None, "HTTP 404"))

    monkeypatch.setattr(M, "_http_get_json", fake)
    M._job_server_text()
    assert seen == {"/api/health": 10.0, "/api/stats": 5.0}


def test_timeout_vs_refused_are_distinguished(monkeypatch):
    _stub(monkeypatch, {"/api/health": (None, None, "timeout: timed out")})
    text = M._job_server_text()
    assert "応答なし(ハングの可能性)" in text and "接続拒否" not in text
    _stub(monkeypatch, {"/api/health": (None, None, "refused: [WinError 10061]")})
    text = M._job_server_text()
    assert "接続拒否(停止中)" in text and "ハング" not in text
    _stub(monkeypatch, {"/api/health": (None, None, "gaierror: no such host")})
    text = M._job_server_text()
    assert "到達できません" in text and "ハング" not in text and "接続拒否" not in text


def test_classify_exc_real_exceptions():
    import socket
    assert M._classify_exc(socket.timeout("timed out")).startswith("timeout:")
    assert M._classify_exc(urllib.error.URLError(socket.timeout("timed out"))
                           ).startswith("timeout:")
    assert M._classify_exc(urllib.error.URLError(ConnectionRefusedError("no"))
                           ).startswith("refused:")
    assert M._classify_exc(ConnectionResetError("rst")).startswith("ConnectionResetError")


def test_http_503_body_detail_is_shown(monkeypatch):
    import io
    body = '{"detail": "fugu_remote は停止中です (exit 1)"}'.encode()

    def err503(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "x", {}, io.BytesIO(body))

    monkeypatch.setattr(M, "_urlopen", err503)
    assert M._http_get_json("/api/health") == (
        503, {"detail": "fugu_remote は停止中です (exit 1)"}, "HTTP 503")
    text = M._job_server_text()
    assert "HTTP 503" in text and "fugu_remote は停止中です (exit 1)" in text


def test_stats_error_is_shown_not_discarded(monkeypatch):
    _stub(monkeypatch, {"/api/health": (200, HEALTH, ""),
                        "/api/stats": (500, {"detail": "boom"}, "HTTP 500")})
    assert "stats: 取得できず(HTTP 500: boom)" in M._job_server_text()
    _stub(monkeypatch, {"/api/health": (200, HEALTH, ""),
                        "/api/stats": (None, None, "timeout: timed out")})
    assert "stats: 取得できず(timeout: timed out)" in M._job_server_text()
    _stub(monkeypatch, {"/api/health": (200, HEALTH, ""),
                        "/api/stats": (200, None, "JSON ではない応答")})
    assert "JSON ではない応答" in M._job_server_text()


def test_url_userinfo_and_query_are_masked(monkeypatch):
    monkeypatch.setenv("FUGU_JOB_SERVER_URL", "http://user:pw123@10.0.0.5:8800/?token=abc")
    _stub(monkeypatch, {"/api/health": (200, HEALTH, "")})
    text = M._job_server_text()
    assert "http://***@10.0.0.5:8800/" in text
    for secret in ("pw123", "user:", "token=abc"):
        assert secret not in text
    assert M._display_url("http://127.0.0.1:8800") == "http://127.0.0.1:8800"
    assert M._display_url("garbage") == "(不正な URL)"


def test_urlopen_bypasses_proxies(monkeypatch):
    import urllib.request
    seen = {}

    class FakeOpener:
        def open(self, req, timeout=None):
            seen["timeout"] = timeout
            return "resp"

    def fake_build(*handlers):
        seen["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build)
    assert M._urlopen(object(), 7) == "resp"
    (ph,) = seen["handlers"]
    assert isinstance(ph, urllib.request.ProxyHandler) and ph.proxies == {}
    assert seen["timeout"] == 7
