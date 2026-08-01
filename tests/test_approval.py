# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_approval と E3 ゲート配線のオフラインテスト。"""
import os
import sys
import threading
import time

import pytest

import fugu_approval
import fugu_sandbox
from fugu_sandbox import DockerSandbox, SubprocessSandbox, get_sandbox


def _approve_soon(expected_prefix, approve=True, poll=0.01):
    """バックグラウンドで pending を監視し、現れた要求を解決するスレッド。"""
    def worker():
        for _ in range(500):
            ids = [i for i in fugu_approval.pending()
                   if i.startswith(expected_prefix)]
            if ids:
                fugu_approval.resolve(ids[0], approve)
                return
            time.sleep(poll)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


# ------------------------------------------------------------------ core gate

def test_require_approval_passthrough_without_env(monkeypatch):
    monkeypatch.delenv("FUGU_REQUIRE_APPROVAL", raising=False)
    assert fugu_approval.require_approval("sandbox-run", "code") is True
    assert fugu_approval.pending() == []


def test_require_approval_granted(monkeypatch):
    monkeypatch.setenv("FUGU_REQUIRE_APPROVAL", "1")
    _approve_soon("sandbox-run", approve=True)
    assert fugu_approval.require_approval("sandbox-run", "code", timeout=5) is True
    assert fugu_approval.pending() == []


def test_require_approval_denied(monkeypatch):
    monkeypatch.setenv("FUGU_REQUIRE_APPROVAL", "1")
    _approve_soon("evolve-merge", approve=False)
    assert fugu_approval.require_approval("evolve-merge", "diff", timeout=5) is False


def test_require_approval_timeout_is_denial(monkeypatch):
    monkeypatch.setenv("FUGU_REQUIRE_APPROVAL", "1")
    assert fugu_approval.require_approval("sandbox-run", "code",
                                          timeout=0.05) is False
    assert fugu_approval.pending() == []  # 後始末される


def test_resolve_unknown_run_id_is_false():
    assert fugu_approval.resolve("nope-12345678", True) is False


# ------------------------------------------------------------------ sandbox gating

def test_sandbox_run_blocked_until_approved(monkeypatch):
    monkeypatch.setenv("FUGU_REQUIRE_APPROVAL", "1")
    _approve_soon("sandbox-run", approve=True)
    monkeypatch.setenv("FUGU_APPROVAL_TIMEOUT", "5")
    res = SubprocessSandbox().run("print('gated ok')")
    assert res.ok and "gated ok" in res.stdout


def test_sandbox_run_denied_returns_error_result(monkeypatch):
    monkeypatch.setenv("FUGU_REQUIRE_APPROVAL", "1")
    _approve_soon("sandbox-run", approve=False)
    monkeypatch.setenv("FUGU_APPROVAL_TIMEOUT", "5")
    res = SubprocessSandbox().run("print('never runs')")
    assert not res.ok
    assert "approval denied" in res.stderr


def test_run_argv_is_not_gated(monkeypatch):
    # pytest 等の固定コマンドは承認対象外(検証ループを承認連打にしない)
    monkeypatch.setenv("FUGU_REQUIRE_APPROVAL", "1")
    res = SubprocessSandbox().run_argv([sys.executable, "-c", "print('argv ok')"])
    assert res.ok and "argv ok" in res.stdout


# ------------------------------------------------------------------ get_sandbox

def test_get_sandbox_default_is_subprocess(monkeypatch):
    monkeypatch.delenv("FUGU_SANDBOX_BACKEND", raising=False)
    assert isinstance(get_sandbox(), SubprocessSandbox)


def test_get_sandbox_docker_optin_when_ready(monkeypatch):
    monkeypatch.setenv("FUGU_SANDBOX_BACKEND", "docker")
    monkeypatch.setenv("FUGU_SANDBOX_IMAGE", "fugu-sandbox:latest")
    monkeypatch.setattr(fugu_sandbox, "_docker_ready", lambda: True)
    box = get_sandbox()
    assert isinstance(box, DockerSandbox)
    assert box.image == "fugu-sandbox:latest"


def test_get_sandbox_docker_unready_falls_back(monkeypatch):
    monkeypatch.setenv("FUGU_SANDBOX_BACKEND", "docker")
    monkeypatch.setattr(fugu_sandbox, "_docker_ready", lambda: False)
    assert isinstance(get_sandbox(), SubprocessSandbox)


def test_get_sandbox_explicit_prefer_overrides_env(monkeypatch):
    monkeypatch.setenv("FUGU_SANDBOX_BACKEND", "docker")
    monkeypatch.setattr(fugu_sandbox, "_docker_ready", lambda: True)
    assert isinstance(get_sandbox(prefer="subprocess"), SubprocessSandbox)


# ------------------------------------------------------------------ resource limits

def test_memory_limit_env_parsing(monkeypatch):
    monkeypatch.setenv("FUGU_SANDBOX_MEMORY_MB", "256")
    assert SubprocessSandbox().memory_mb == 256
    monkeypatch.setenv("FUGU_SANDBOX_MEMORY_MB", "0")
    assert SubprocessSandbox().memory_mb is None
    monkeypatch.setenv("FUGU_SANDBOX_MEMORY_MB", "junk")
    assert SubprocessSandbox().memory_mb == fugu_sandbox.DEFAULT_MEMORY_MB
    monkeypatch.delenv("FUGU_SANDBOX_MEMORY_MB", raising=False)
    assert SubprocessSandbox(memory_mb=64).memory_mb == 64
    assert SubprocessSandbox(memory_mb=0).memory_mb is None


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object の検証")
def test_memory_limit_kills_overallocation_windows():
    # 64MB 上限で 300MB 確保 → MemoryError / 強制終了のどちらでも ok=False になる
    res = SubprocessSandbox(memory_mb=64).run(
        "data = bytearray(300 * 1024 * 1024)\nprint('allocated')")
    assert not res.ok
    assert "allocated" not in res.stdout


def test_memory_limit_allows_normal_work():
    res = SubprocessSandbox(memory_mb=256).run("print(sum(range(100)))")
    assert res.ok and "4950" in res.stdout


# ------------------------------------------------------------------ evolve merge gate

def test_merge_approval_passthrough_without_env(monkeypatch):
    from fugu_evolve.cli import merge_approval
    from fugu_evolve.planner import Proposal
    monkeypatch.delenv("FUGU_REQUIRE_APPROVAL", raising=False)
    proposal = Proposal(title="t", category="fix", target_files=["a.py"],
                        rationale="r")
    assert merge_approval(proposal, "+diff") is True


def test_pipeline_merge_denied_rolls_back(tmp_path):
    from fugu_evolve.cli import build_pipeline
    from tests.test_evolve_cli import FakeWorkspace, _deps
    ws = FakeWorkspace(str(tmp_path))
    run = build_pipeline(_deps(ws, merge_approval_fn=lambda p, d: False))
    outcome = run(str(tmp_path)).outcomes[0]
    assert outcome.merged is False
    assert "approval denied" in outcome.reason
    assert "rollback" in ws.calls and "merge" not in ws.calls


def test_pipeline_merge_approved_proceeds(tmp_path):
    from fugu_evolve.cli import build_pipeline
    from tests.test_evolve_cli import FakeWorkspace, _deps
    ws = FakeWorkspace(str(tmp_path))
    run = build_pipeline(_deps(ws, merge_approval_fn=lambda p, d: True))
    outcome = run(str(tmp_path)).outcomes[0]
    assert outcome.merged is True
    assert "merge" in ws.calls


# ------------------------------------------------------------------ API endpoints

def test_api_approvals_and_approve_flow(monkeypatch):
    from fastapi.testclient import TestClient
    import fugu_api
    client = TestClient(fugu_api.app)
    monkeypatch.setenv("FUGU_REQUIRE_APPROVAL", "1")

    outcome = {}

    def requester():
        outcome["approved"] = fugu_approval.require_approval(
            "sandbox-run", "print(1)", timeout=10)

    thread = threading.Thread(target=requester, daemon=True)
    thread.start()
    run_id = None
    for _ in range(500):
        ids = client.get("/approvals").json()["pending"]
        if ids:
            run_id = ids[0]
            break
        time.sleep(0.01)
    assert run_id is not None
    r = client.post(f"/approve/{run_id}", json={"approve": True})
    assert r.status_code == 200 and r.json()["approved"] is True
    thread.join(timeout=10)
    assert outcome["approved"] is True


def test_api_approve_unknown_is_404():
    from fastapi.testclient import TestClient
    import fugu_api
    client = TestClient(fugu_api.app)
    assert client.post("/approve/ghost-00000000",
                       json={"approve": True}).status_code == 404
