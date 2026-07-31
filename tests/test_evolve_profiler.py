"""fugu_evolve.profiler のオフラインテスト(FakeSandbox / tmp_path のみ)。"""
import json

from fugu_sandbox import SandboxResult
from fugu_evolve.profiler import (
    BenchReport,
    build_health_report,
    run_bench,
    run_pytest,
    scan_structure,
)


class FakeSandbox:
    """run_argv をスクリプトされた SandboxResult で返す注入用サンドボックス。"""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def run(self, code, lang="python", timeout=None, files=None):
        raise AssertionError("profiler は run() を使わない")

    def run_argv(self, argv, cwd=None, timeout=None, env=None):
        self.calls.append({"argv": list(argv), "cwd": cwd})
        if self.results:
            return self.results.pop(0)
        return SandboxResult(exit_code=0)


# ------------------------------------------------------------------ run_pytest

def test_run_pytest_parses_pass_fail_counts():
    sandbox = FakeSandbox([SandboxResult(
        stdout="....F\n1 failed, 4 passed in 0.5s\n", exit_code=1)])
    rep = run_pytest("/repo", sandbox)
    assert rep.ran and rep.passed == 4 and rep.failed == 1
    assert rep.ok is False
    assert "1 failed" in rep.tail


def test_run_pytest_all_green():
    sandbox = FakeSandbox([SandboxResult(stdout="10 passed in 1s\n", exit_code=0)])
    rep = run_pytest("/repo", sandbox)
    assert rep.ok is True and rep.passed == 10


def test_run_pytest_counts_errors():
    sandbox = FakeSandbox([SandboxResult(
        stdout="2 errors\n", stderr="", exit_code=1)])
    rep = run_pytest("/repo", sandbox)
    assert rep.errors == 2 and rep.ok is False


def test_run_pytest_timeout_means_not_ran():
    sandbox = FakeSandbox([SandboxResult(exit_code=-1, timed_out=True)])
    rep = run_pytest("/repo", sandbox)
    assert rep.ran is False and rep.ok is False


def test_run_pytest_invokes_pytest_in_repo():
    sandbox = FakeSandbox([SandboxResult(stdout="1 passed", exit_code=0)])
    run_pytest("/my/repo", sandbox)
    assert sandbox.calls[0]["cwd"] == "/my/repo"
    assert "pytest" in sandbox.calls[0]["argv"]


# ------------------------------------------------------------------ run_bench

def test_run_bench_offline_skips_with_reason():
    rep = run_bench("/repo", bench_argv=["python", "bench.py"],
                    sandbox=FakeSandbox(), offline=True)
    assert rep.ran is False and "offline" in rep.reason


def test_run_bench_env_flag_forces_offline(monkeypatch):
    monkeypatch.setenv("FUGU_EVOLVE_OFFLINE", "1")
    rep = run_bench("/repo", bench_argv=["python", "bench.py"], sandbox=FakeSandbox())
    assert rep.ran is False and "offline" in rep.reason


def test_run_bench_without_command_skips():
    rep = run_bench("/repo", sandbox=FakeSandbox(), offline=False)
    assert rep.ran is False and "no bench command" in rep.reason


def test_run_bench_parses_last_number_as_score():
    sandbox = FakeSandbox([SandboxResult(
        stdout="items: 30\naccuracy: 0.9667\n", exit_code=0)])
    rep = run_bench("/repo", bench_argv=["python", "bench.py"],
                    sandbox=sandbox, offline=False)
    assert rep == BenchReport(ran=True, score=0.9667)


def test_run_bench_failure_reports_reason():
    sandbox = FakeSandbox([SandboxResult(stderr="boom", exit_code=3)])
    rep = run_bench("/repo", bench_argv=["python", "bench.py"],
                    sandbox=sandbox, offline=False)
    assert rep.ran is False and "exit=3" in rep.reason


# ------------------------------------------------------------------ scan_structure

def _write(tmp_path, name, text):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_scan_structure_counts_and_complexity(tmp_path):
    _write(tmp_path, "mod.py",
           '"""doc."""\n\n\ndef f(x):\n    if x:\n        return 1\n'
           '    for i in range(3):\n        pass\n    return 0\n\n\nclass C:\n    pass\n')
    stats = scan_structure(str(tmp_path))
    assert len(stats) == 1
    s = stats[0]
    assert s.path == "mod.py"
    assert s.functions == 1 and s.classes == 1
    assert s.has_docstring is True
    assert s.max_complexity >= 3  # if + for で最低 1+2
    assert s.parse_error is False


def test_scan_structure_flags_broken_file(tmp_path):
    _write(tmp_path, "broken.py", "def f(:\n")
    stats = scan_structure(str(tmp_path))
    assert stats[0].parse_error is True


def test_scan_structure_skips_cache_dirs(tmp_path):
    _write(tmp_path, "__pycache__/x.py", "a = 1\n")
    _write(tmp_path, ".git/y.py", "b = 2\n")
    _write(tmp_path, "real.py", "c = 3\n")
    stats = scan_structure(str(tmp_path))
    assert [s.path for s in stats] == ["real.py"]


def test_scan_structure_respects_max_files(tmp_path):
    for i in range(5):
        _write(tmp_path, f"m{i}.py", "x = 1\n")
    assert len(scan_structure(str(tmp_path), max_files=3)) == 3


def test_scan_structure_normalizes_subdir_paths(tmp_path):
    _write(tmp_path, "pkg/sub.py", "x = 1\n")
    stats = scan_structure(str(tmp_path))
    assert stats[0].path == "pkg/sub.py"  # Windows でも posix 区切り


# ------------------------------------------------------------------ health report

def test_build_health_report_is_json_serializable(tmp_path):
    _write(tmp_path, "a.py", "def f():\n    return 1\n")
    sandbox = FakeSandbox([SandboxResult(stdout="3 passed", exit_code=0)])
    report = build_health_report(str(tmp_path), sandbox=sandbox, offline=True)
    json.dumps(report)  # 直列化可能であること(planner が LLM へ渡す)
    assert report["pytest"]["passed"] == 3
    assert report["bench"]["ran"] is False
    assert report["totals"]["modules"] == 1
    assert report["modules"][0]["path"] == "a.py"


def test_build_health_report_ranks_broken_then_complex(tmp_path):
    _write(tmp_path, "broken.py", "def f(:\n")
    _write(tmp_path, "simple.py", "x = 1\n")
    _write(tmp_path, "complex.py",
           "def f(x):\n" + "".join(f"    if x == {i}:\n        pass\n"
                                   for i in range(6)))
    sandbox = FakeSandbox([SandboxResult(stdout="1 passed", exit_code=0)])
    report = build_health_report(str(tmp_path), sandbox=sandbox, offline=True)
    paths = [m["path"] for m in report["modules"]]
    assert paths[0] == "broken.py"       # パース不能が最優先
    assert paths[1] == "complex.py"      # 次に高複雑度
    assert report["totals"]["parse_errors"] == 1


def test_build_health_report_caps_module_list(tmp_path):
    for i in range(8):
        _write(tmp_path, f"m{i}.py", "x = 1\n")
    sandbox = FakeSandbox([SandboxResult(stdout="1 passed", exit_code=0)])
    report = build_health_report(str(tmp_path), sandbox=sandbox, offline=True,
                                 top_modules=5)
    assert len(report["modules"]) == 5
    assert report["totals"]["modules"] == 8  # totals は全件を数える
