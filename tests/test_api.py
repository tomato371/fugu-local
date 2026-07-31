"""Tests for the FastAPI wrapper (fugu_api).

These stub out the orchestrator (`ask_fugu`) and the Ollama probe (`server_up`),
so they run in CI without Ollama, a GPU, or any model.
"""
from fastapi.testclient import TestClient

import fugu_api

client = TestClient(fugu_api.app)


def test_health_ok(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "server_up", lambda: True)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_reports_unreachable(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "server_up", lambda: False)
    assert client.get("/health").json()["status"] == "ollama_unreachable"


def test_ask_returns_answer(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "ask_fugu", lambda q, **kw: f"echo: {q}")
    r = client.post("/ask", json={"question": "Is 91 prime?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo: Is 91 prime?"
    assert isinstance(body["elapsed_seconds"], (int, float))


def test_ask_empty_question_is_rejected():
    # pydantic validation (min_length=1) -> 422 before the orchestrator runs
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_setup_failure_returns_503(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "ask_fugu", lambda q, **kw: None)
    assert client.post("/ask", json={"question": "hi"}).status_code == 503


# ------------------------------------------------------------------ IDE endpoints


def test_completion_returns_inserted_code(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "setup", lambda: True)
    monkeypatch.setattr(fugu_api.fugu, "ask",
                        lambda *a, **k: "return n * 2")
    r = client.post("/completion", json={"prefix": "def double(n):\n    "})
    assert r.status_code == 200
    assert r.json()["completion"] == "return n * 2"


def test_completion_model_error_is_502(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "setup", lambda: True)
    monkeypatch.setattr(fugu_api.fugu, "ask",
                        lambda *a, **k: "__ERROR__: model exploded")
    r = client.post("/completion", json={"prefix": "x = "})
    assert r.status_code == 502


def test_completion_setup_failure_is_503(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "setup", lambda: False)
    r = client.post("/completion", json={"prefix": "x = "})
    assert r.status_code == 503


def test_refactor_returns_diff(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "setup", lambda: True)
    monkeypatch.setattr(
        fugu_api.fugu, "ask",
        lambda *a, **k: "```python\nvalue = 2\n```")
    r = client.post("/refactor",
                    json={"code": "value = 1\n", "instruction": "set to 2"})
    assert r.status_code == 200
    body = r.json()
    assert body["refactored"] == "value = 2"
    assert "-value = 1" in body["diff"]
    assert "+value = 2" in body["diff"]


def test_test_run_plain_execution():
    r = client.post("/test-run", json={"code": "print('from api')"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "from api" in body["stdout"]
    assert body["attempts"] == 1


def test_test_run_failure_reports_stderr():
    r = client.post("/test-run", json={"code": "raise ValueError('api boom')"})
    body = r.json()
    assert body["ok"] is False
    assert "api boom" in body["stderr"]


def test_test_run_tdc_mode_runs_pytest():
    r = client.post("/test-run", json={
        "code": "def add(a, b):\n    return a + b\n",
        "tests": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    })
    body = r.json()
    assert body["ok"] is True


# ------------------------------------------------------------------ streaming (B5)
# Fully offline: the fake pipeline emits events through fugu_local's real
# set_event_sink/_emit plumbing, exactly like fugu_answer does live.


def _fake_pipeline(question, **kwargs):
    fugu_api.fugu._emit("plan", mode="single", task_type="qa", rounds=1)
    fugu_api.fugu._emit("aggregate", round=1, chars=10)
    fugu_api.fugu._emit("final", chars=10)
    return f"answer to {question}"


def test_emit_is_noop_without_sink():
    fugu_api.fugu.set_event_sink(None)
    fugu_api.fugu._emit("plan", x=1)  # 例外を出さないこと


def test_emit_swallows_sink_exceptions():
    fugu_api.fugu.set_event_sink(
        lambda kind, data: (_ for _ in ()).throw(RuntimeError("sink boom")))
    try:
        fugu_api.fugu._emit("plan", x=1)  # 本処理を落とさない
    finally:
        fugu_api.fugu.set_event_sink(None)


def test_ask_stream_emits_events_answer_and_done(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "ask_fugu", _fake_pipeline)
    r = client.get("/ask/stream", params={"question": "hi"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    text = r.text
    assert "event: plan" in text
    assert "event: aggregate" in text
    assert "event: answer" in text and "answer to hi" in text
    assert text.rstrip().endswith("event: done\ndata: {}")  # 終端センチネル
    assert text.index("event: plan") < text.index("event: answer")  # イベントが先


def test_ask_stream_unregisters_sink_after_completion(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "ask_fugu", _fake_pipeline)
    client.get("/ask/stream", params={"question": "hi"})
    assert fugu_api.fugu._EVENT_SINK is None


def test_ask_stream_orchestrator_error_becomes_error_event(monkeypatch):
    def boom(question, **kwargs):
        raise RuntimeError("pipeline boom")
    monkeypatch.setattr(fugu_api.fugu, "ask_fugu", boom)
    text = client.get("/ask/stream", params={"question": "hi"}).text
    assert "event: error" in text and "pipeline boom" in text
    assert "event: done" in text  # エラーでも必ず終端する


def test_ask_stream_setup_failure_is_error_event(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "ask_fugu", lambda q, **kw: None)
    text = client.get("/ask/stream", params={"question": "hi"}).text
    assert "event: error" in text and "Setup failed" in text


def test_ask_stream_rejects_empty_question():
    assert client.get("/ask/stream", params={"question": "  "}).status_code == 422


def test_ws_ask_streams_events_then_done(monkeypatch):
    monkeypatch.setattr(fugu_api.fugu, "ask_fugu", _fake_pipeline)
    with client.websocket_connect("/ws/ask") as ws:
        ws.send_json({"question": "hi"})
        events = []
        while True:
            msg = ws.receive_json()
            events.append(msg)
            if msg["event"] == "done":
                break
    kinds = [e["event"] for e in events]
    assert "plan" in kinds and "answer" in kinds
    assert kinds[-1] == "done"
    answer = next(e for e in events if e["event"] == "answer")
    assert answer["data"]["answer"] == "answer to hi"


def test_ws_ask_empty_question_is_error_event():
    with client.websocket_connect("/ws/ask") as ws:
        ws.send_json({"question": ""})
        msg = ws.receive_json()
    assert msg["event"] == "error"
