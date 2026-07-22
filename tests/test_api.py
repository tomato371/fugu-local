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
