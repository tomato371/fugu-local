"""fugu_evolve.evaluator のオフラインテスト(FakeSandbox + FakeChat + FakeGit)。"""
import json

import pytest

from fugu_llm import FakeChat
from fugu_sandbox import SandboxResult
from fugu_evolve.evaluator import FAILED, VERIFIED, _request_patch, verify
from fugu_evolve.workspace import BRANCH_PREFIX, Workspace

from tests.test_evolve_workspace import FakeGit

PYTEST_OK = SandboxResult(stdout="5 passed in 1s\n", exit_code=0)
PYTEST_FAIL = SandboxResult(
    stdout="F....\nE   assert 1 == 2\n1 failed, 4 passed in 1s\n", exit_code=1)


class FakeSandbox:
    """run_argv をスクリプト順に返す(pytest / bench 共用)。"""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def run(self, code, lang="python", timeout=None, files=None):
        raise AssertionError("evaluator は run() を使わない")

    def run_argv(self, argv, cwd=None, timeout=None, env=None):
        self.calls.append(list(argv))
        return self.results.pop(0)


def _ws(tmp_path):
    """auto-evolve ブランチ上の Workspace(apply_edit が実ファイルを書ける)。"""
    return Workspace(str(tmp_path), git=FakeGit(branch=BRANCH_PREFIX + "t-1"))


def _patch_json(path="mod.py", code="x = 1\n"):
    return json.dumps({"path": path, "code": code})


BASELINE_NO_BENCH = {"pytest": {"passed": 5}, "bench": {"ran": False, "reason": "offline"}}
BASELINE_BENCH_09 = {"pytest": {"passed": 5}, "bench": {"ran": True, "score": 0.9}}


# ------------------------------------------------------------------ verify: pytest

def test_verify_green_first_try_is_verified(tmp_path):
    v = verify(_ws(tmp_path), FakeSandbox([PYTEST_OK]),
               FakeChat(fn=lambda p: pytest.fail("green なら chat 不要")),
               BASELINE_NO_BENCH)
    assert v.verdict == VERIFIED
    assert v.attempts == 1
    assert v.pytest.ok is True


def test_verify_repairs_then_green(tmp_path):
    sandbox = FakeSandbox([PYTEST_FAIL, PYTEST_OK])
    chat = FakeChat(responses=[_patch_json("mod.py", "x = 2\n")])
    v = verify(_ws(tmp_path), sandbox, chat, BASELINE_NO_BENCH)
    assert v.verdict == VERIFIED
    assert v.attempts == 2
    assert (tmp_path / "mod.py").read_text(encoding="utf-8") == "x = 2\n"
    assert any("rewrote mod.py" in n for n in v.notes)
    assert "1 failed" in chat.calls[0]["prompt"]  # 失敗ログが修正ヒントとして渡る


def test_verify_gives_up_after_max_attempts(tmp_path):
    sandbox = FakeSandbox([PYTEST_FAIL, PYTEST_FAIL])
    chat = FakeChat(responses=[_patch_json()])
    v = verify(_ws(tmp_path), sandbox, chat, BASELINE_NO_BENCH, max_attempts=2)
    assert v.verdict == FAILED
    assert v.attempts == 2
    assert len(chat.calls) == 1  # 修正は max_attempts-1 回まで
    assert any("still failing" in n for n in v.notes)


def test_verify_compile_guard_rejects_broken_patch(tmp_path):
    sandbox = FakeSandbox([PYTEST_FAIL])
    chat = FakeChat(responses=[_patch_json("mod.py", "def f(:\n")])
    v = verify(_ws(tmp_path), sandbox, chat, BASELINE_NO_BENCH)
    assert v.verdict == FAILED
    assert not (tmp_path / "mod.py").exists()  # 壊れたパッチは書かない
    assert any("no valid patch" in n for n in v.notes)


def test_verify_chat_failure_stops_repair(tmp_path):
    sandbox = FakeSandbox([PYTEST_FAIL])
    chat = FakeChat(fn=lambda p: (_ for _ in ()).throw(RuntimeError("down")))
    v = verify(_ws(tmp_path), sandbox, chat, BASELINE_NO_BENCH)
    assert v.verdict == FAILED and v.attempts == 1


def test_verify_apply_guard_failure_is_noted(tmp_path):
    sandbox = FakeSandbox([PYTEST_FAIL])
    chat = FakeChat(responses=[_patch_json("../escape.py", "x = 1\n")])
    v = verify(_ws(tmp_path), sandbox, chat, BASELINE_NO_BENCH)
    assert v.verdict == FAILED
    assert any("apply failed" in n for n in v.notes)


# ------------------------------------------------------------------ verify: bench

def test_verify_bench_regression_fails(tmp_path):
    sandbox = FakeSandbox([PYTEST_OK, SandboxResult(stdout="score 0.5\n", exit_code=0)])
    v = verify(_ws(tmp_path), sandbox, FakeChat(default="unused"),
               BASELINE_BENCH_09, bench_argv=["python", "bench.py"], offline=False)
    assert v.pytest.ok is True
    assert v.bench_regressed is True
    assert v.verdict == FAILED
    assert any("regressed" in n for n in v.notes)


def test_verify_bench_non_regression_verifies(tmp_path):
    sandbox = FakeSandbox([PYTEST_OK, SandboxResult(stdout="score 0.95\n", exit_code=0)])
    v = verify(_ws(tmp_path), sandbox, FakeChat(default="unused"),
               BASELINE_BENCH_09, bench_argv=["python", "bench.py"], offline=False)
    assert v.verdict == VERIFIED and v.bench_regressed is False


def test_verify_bench_tolerance_allows_small_drop(tmp_path):
    sandbox = FakeSandbox([PYTEST_OK, SandboxResult(stdout="score 0.88\n", exit_code=0)])
    v = verify(_ws(tmp_path), sandbox, FakeChat(default="unused"),
               BASELINE_BENCH_09, bench_argv=["python", "bench.py"],
               offline=False, tolerance=0.05)
    assert v.verdict == VERIFIED


def test_verify_bench_unavailable_is_clean_skip(tmp_path):
    v = verify(_ws(tmp_path), FakeSandbox([PYTEST_OK]), FakeChat(default="unused"),
               BASELINE_BENCH_09, offline=True)  # current が実行不能
    assert v.verdict == VERIFIED
    assert v.bench_regressed is False
    assert any("skipped" in n for n in v.notes)


def test_verify_no_baseline_bench_is_clean_skip(tmp_path):
    sandbox = FakeSandbox([PYTEST_OK])
    v = verify(_ws(tmp_path), sandbox, FakeChat(default="unused"),
               BASELINE_NO_BENCH, offline=True)
    assert v.verdict == VERIFIED
    assert any("no usable baseline" in n for n in v.notes)


# ------------------------------------------------------------------ _request_patch

def test_request_patch_unwraps_fenced_code():
    chat = FakeChat(responses=[json.dumps(
        {"path": "m.py", "code": "```python\ny = 3\n```"})])
    patch = _request_patch(chat, "tail")
    assert patch == {"path": "m.py", "code": "y = 3"}


def test_request_patch_rejects_junk_and_missing_fields():
    assert _request_patch(FakeChat(default="not json"), "t") is None
    assert _request_patch(FakeChat(responses=['{"path": "m.py"}']), "t") is None
    assert _request_patch(FakeChat(responses=[json.dumps(
        {"path": "", "code": "x"})]), "t") is None


def test_request_patch_non_python_skips_compile_guard():
    chat = FakeChat(responses=[json.dumps(
        {"path": "README.md", "code": "# not python ("})])
    patch = _request_patch(chat, "t")
    assert patch is not None and patch["path"] == "README.md"
