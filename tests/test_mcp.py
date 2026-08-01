# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_mcp のオフラインテスト(モデル・GPU・ネット不要)。

サーバーの JSON-RPC 処理 handle_message / call_tool を直接叩く。
実際の fugu_local 呼び出しはフェイクの runner に差し替える。
"""
import json
import threading

import pytest

import fugu_mcp as M


def _req(method, msg_id=1, params=None):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _tool_text(resp):
    return resp["result"]["content"][0]["text"]


@pytest.fixture()
def jobs():
    return M.Jobs(runner=lambda q, s=False, t="": f"answer to {q}")


# ------------------------------------------------------------ ハンドシェイク

def test_initialize_echoes_client_protocol_version(jobs):
    resp = M.handle_message(
        _req("initialize", params={"protocolVersion": "2024-11-05"}), jobs)
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "fugu"
    assert "tools" in resp["result"]["capabilities"]


def test_initialized_notification_gets_no_response(jobs):
    assert M.handle_message(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}, jobs) is None


def test_unknown_method_returns_method_not_found(jobs):
    resp = M.handle_message(_req("resources/list"), jobs)
    assert resp["error"]["code"] == -32601


def test_ping(jobs):
    assert M.handle_message(_req("ping"), jobs)["result"] == {}


# ------------------------------------------------------------------ tools/list

def test_tools_list_exposes_the_four_tools(jobs):
    resp = M.handle_message(_req("tools/list"), jobs)
    names = [t["name"] for t in resp["result"]["tools"]]
    assert names == ["fugu_ask", "fugu_ask_start", "fugu_ask_status", "fugu_health"]
    for tool in resp["result"]["tools"]:
        assert tool["inputSchema"]["type"] == "object"


# ------------------------------------------------------------------ tools/call

def test_fugu_ask_returns_answer_text(jobs):
    resp = M.handle_message(_req("tools/call", params={
        "name": "fugu_ask", "arguments": {"question": "91は素数?"}}), jobs)
    assert _tool_text(resp) == "answer to 91は素数?"
    assert "isError" not in resp["result"]


def test_fugu_ask_missing_question_is_tool_error_not_crash(jobs):
    resp = M.handle_message(_req("tools/call", params={
        "name": "fugu_ask", "arguments": {}}), jobs)
    assert resp["result"]["isError"] is True


def test_unknown_tool_is_tool_error(jobs):
    resp = M.handle_message(_req("tools/call", params={
        "name": "nope", "arguments": {}}), jobs)
    assert resp["result"]["isError"] is True


def test_runner_exception_becomes_tool_error(jobs):
    def boom(q, s=False, t=""):
        raise RuntimeError("Ollama down")
    failing = M.Jobs(runner=boom)
    resp = M.handle_message(_req("tools/call", params={
        "name": "fugu_ask", "arguments": {"question": "q"}}), failing)
    assert resp["result"]["isError"] is True
    assert "Ollama down" in _tool_text(resp)


# ------------------------------------------------------------------- ジョブ型

def test_job_lifecycle_start_running_done():
    gate = threading.Event()

    def slow(q, s=False, t=""):
        gate.wait(timeout=10)
        return "遅い答え"

    jobs = M.Jobs(runner=slow)
    start = M.handle_message(_req("tools/call", params={
        "name": "fugu_ask_start", "arguments": {"question": "q"}}), jobs)
    job_id = json.loads(_tool_text(start))["job_id"]

    running = json.loads(_tool_text(M.handle_message(_req("tools/call", params={
        "name": "fugu_ask_status", "arguments": {"job_id": job_id}}), jobs)))
    assert running["status"] == "running"

    gate.set()
    for _ in range(100):                      # ワーカー完了を待つ(最大 ~1 秒)
        status = json.loads(_tool_text(M.handle_message(_req("tools/call", params={
            "name": "fugu_ask_status", "arguments": {"job_id": job_id}}), jobs)))
        if status["status"] != "running":
            break
        threading.Event().wait(0.01)
    assert status["status"] == "done" and status["answer"] == "遅い答え"


def test_second_start_while_running_is_busy_error():
    gate = threading.Event()
    jobs = M.Jobs(runner=lambda q, s=False, t="": gate.wait(10) or "x")
    M.handle_message(_req("tools/call", params={
        "name": "fugu_ask_start", "arguments": {"question": "1件目"}}), jobs)
    second = M.handle_message(_req("tools/call", params={
        "name": "fugu_ask_start", "arguments": {"question": "2件目"}}), jobs)
    assert second["result"]["isError"] is True
    gate.set()


def test_job_error_is_reported_in_status():
    def boom(q, s=False, t=""):
        raise RuntimeError("model missing")
    jobs = M.Jobs(runner=boom)
    start = M.handle_message(_req("tools/call", params={
        "name": "fugu_ask_start", "arguments": {"question": "q"}}), jobs)
    job_id = json.loads(_tool_text(start))["job_id"]
    for _ in range(100):
        status = json.loads(_tool_text(M.handle_message(_req("tools/call", params={
            "name": "fugu_ask_status", "arguments": {"job_id": job_id}}), jobs)))
        if status["status"] != "running":
            break
        threading.Event().wait(0.01)
    assert status["status"] == "error" and "model missing" in status["error"]


def test_unknown_job_id_is_tool_error(jobs):
    resp = M.handle_message(_req("tools/call", params={
        "name": "fugu_ask_status", "arguments": {"job_id": "zzz"}}), jobs)
    assert resp["result"]["isError"] is True


# -------------------------------------------------------------------- 環境面

def test_redirect_stdio_reserves_rpc_channel(monkeypatch):
    """_redirect_stdio() 後: 本物の stdout は _RPC_OUT に確保され、
    print() (sys.stdout) は stderr に向く — fugu の進捗表示が通信路を汚さない。"""
    import io
    import sys
    fake_out, fake_err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_out)
    monkeypatch.setattr(sys, "stderr", fake_err)
    monkeypatch.setattr(M, "_RPC_OUT", None)
    M._redirect_stdio()
    assert M._RPC_OUT is fake_out
    assert sys.stdout is fake_err
    print("進捗メッセージ")                    # fugu_local 相当の print
    assert "進捗メッセージ" in fake_err.getvalue()
    assert fake_out.getvalue() == ""           # 通信路は無傷


def test_health_uses_launcher_preflight(monkeypatch):
    import fugu_launcher as L
    monkeypatch.setattr(L, "installed_models", lambda url, timeout=2.0: ["qwen3:4b"])
    text = M._health_text()
    assert "Ollama OK" in text
