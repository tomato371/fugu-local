"""fugu_evolve.profiler — メタインスペクター: コード健全性の計測 (Doc C Phase 1)。

自己改善ループの最初の段。リポジトリの現状を3系統で観測し、planner (C2) が
LLM に渡せる JSON 直列化可能なヘルスレポートに束ねる:

- :func:`run_pytest` — Sandbox 経由で pytest を実行し ``N passed/failed`` を
  パース(テストの合否が最優先の改善シグナル)。
- :func:`run_bench` — 性能ベンチ。オフライン(Ollama/GPU 不可)なら実行せず
  理由付き skip を返す(実装中は GPU 競合のため Ollama 実行禁止の規律に従う)。
- :func:`scan_structure` — ast による静的走査(行数・関数/クラス数・最大分岐
  複雑度・docstring 有無)。リファクタ候補の発見用。

すべて注入ベース: sandbox は fugu_sandbox.Sandbox(テストは FakeSandbox)。
"""
from __future__ import annotations

import ast
import fnmatch
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from fugu_sandbox import Sandbox, SandboxResult, SubprocessSandbox

#: 走査から除外するディレクトリ名(git・キャッシュ・過去イテレーション類)。
_SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "backups", "iterations", "node_modules", ".venv", "venv",
})

_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")
_ERROR_RE = re.compile(r"(\d+) errors?")


@dataclass
class PytestReport:
    """pytest 実行の要約。ran=False は「実行自体ができなかった」(タイムアウト等)。"""

    ran: bool
    passed: int = 0
    failed: int = 0
    errors: int = 0
    exit_code: int = -1
    tail: str = ""  # 失敗詳細の末尾(planner が LLM へ渡す修正ヒント)

    @property
    def ok(self) -> bool:
        return self.ran and self.failed == 0 and self.errors == 0


@dataclass
class BenchReport:
    """ベンチ実行の要約。ran=False のとき reason に skip 理由が入る。"""

    ran: bool
    score: Optional[float] = None
    reason: str = ""


@dataclass
class ModuleStats:
    """1モジュール分の静的統計。complexity は関数単位の分岐数の最大値。"""

    path: str
    lines: int
    functions: int
    classes: int
    max_complexity: int
    has_docstring: bool
    parse_error: bool = False


def run_pytest(repo: str, sandbox: Optional[Sandbox] = None,
               timeout: float = 600.0) -> PytestReport:
    """``repo`` で ``pytest -q`` を実行し、サマリ行から件数をパースする。

    exit_code 5(テスト無し)は「実行できたが 0 件」として扱う。タイムアウト・
    ランナー自体の失敗(exit_code -1)は ran=False。
    """
    sandbox = sandbox or SubprocessSandbox(timeout=timeout)
    res: SandboxResult = sandbox.run_argv(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        cwd=repo, timeout=timeout)
    ran = (not res.timed_out) and res.exit_code >= 0
    out = (res.stdout or "") + (res.stderr or "")

    def _count(pattern: re.Pattern) -> int:
        m = pattern.search(out)
        return int(m.group(1)) if m else 0

    return PytestReport(
        ran=ran,
        passed=_count(_PASSED_RE),
        failed=_count(_FAILED_RE),
        errors=_count(_ERROR_RE),
        exit_code=res.exit_code,
        tail=out.strip()[-2000:],
    )


def run_bench(repo: str, bench_argv: Optional[List[str]] = None,
              sandbox: Optional[Sandbox] = None, offline: Optional[bool] = None,
              timeout: float = 1800.0) -> BenchReport:
    """性能ベンチを実行する。実行できない状況では理由付きで skip を返す。

    ``offline`` 未指定時は env ``FUGU_EVOLVE_OFFLINE=1`` を見る(実装セッション
    中は GPU 競合のため Ollama 実行禁止 — その規律を機械的に守るためのフラグ)。
    ``bench_argv`` 未指定も skip(ベンチはリポジトリ固有のため既定を持たない)。
    スコアは stdout の最後の数値を採用する(bench_fugu.py の accuracy 行想定)。
    """
    if offline is None:
        offline = os.environ.get("FUGU_EVOLVE_OFFLINE") == "1"
    if offline:
        return BenchReport(ran=False, reason="offline mode (FUGU_EVOLVE_OFFLINE=1)")
    if not bench_argv:
        return BenchReport(ran=False, reason="no bench command configured")
    sandbox = sandbox or SubprocessSandbox(timeout=timeout)
    res = sandbox.run_argv(list(bench_argv), cwd=repo, timeout=timeout)
    if res.timed_out or res.exit_code != 0:
        return BenchReport(ran=False,
                           reason=f"bench failed (exit={res.exit_code}, "
                                  f"timed_out={res.timed_out})")
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", res.stdout or "")
    score = float(numbers[-1]) if numbers else None
    if score is None:
        return BenchReport(ran=False, reason="no numeric score in bench output")
    return BenchReport(ran=True, score=score)


def _complexity(node: ast.AST) -> int:
    """関数1つ分の大まかな分岐複雑度(1 + 分岐系ノード数)。"""
    count = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                              ast.With, ast.BoolOp, ast.IfExp, ast.comprehension)):
            count += 1
    return count


def scan_structure(root: str, pattern: str = "*.py",
                   max_files: int = 200) -> List[ModuleStats]:
    """``root`` 配下の Python モジュールを ast で走査して統計を返す。

    パースできないファイルは parse_error=True の行として残す(壊れたファイルは
    それ自体が最優先の改善候補)。走査は ``max_files`` で打ち切る(暴走防止)。
    """
    stats: List[ModuleStats] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in sorted(filenames):
            if not fnmatch.fnmatch(name, pattern):
                continue
            if len(stats) >= max_files:
                return stats
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    source = fh.read()
                tree = ast.parse(source)
            except (OSError, SyntaxError, ValueError):
                stats.append(ModuleStats(path=rel, lines=0, functions=0, classes=0,
                                         max_complexity=0, has_docstring=False,
                                         parse_error=True))
                continue
            funcs = [n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            stats.append(ModuleStats(
                path=rel,
                lines=source.count("\n") + 1,
                functions=len(funcs),
                classes=len(classes),
                max_complexity=max((_complexity(f) for f in funcs), default=0),
                has_docstring=ast.get_docstring(tree) is not None,
            ))
    return stats


def build_health_report(repo: str, sandbox: Optional[Sandbox] = None,
                        bench_argv: Optional[List[str]] = None,
                        offline: Optional[bool] = None,
                        top_modules: int = 15) -> Dict[str, object]:
    """planner (C2) が LLM に渡す JSON 直列化可能なヘルスレポートを組み立てる。

    modules は「パース不能 → 複雑度降順」で並べた上位 ``top_modules`` 件のみ
    (num_ctx 予算: レポート全文をプロンプトに入れても溢れないサイズに保つ)。
    """
    pytest_report = run_pytest(repo, sandbox=sandbox)
    bench_report = run_bench(repo, bench_argv=bench_argv, sandbox=sandbox,
                             offline=offline)
    modules = scan_structure(repo)
    ranked = sorted(modules,
                    key=lambda m: (not m.parse_error, -m.max_complexity, -m.lines))
    return {
        "repo": os.path.abspath(repo).replace(os.sep, "/"),
        "pytest": asdict(pytest_report),
        "bench": asdict(bench_report),
        "totals": {
            "modules": len(modules),
            "lines": sum(m.lines for m in modules),
            "parse_errors": sum(1 for m in modules if m.parse_error),
        },
        "modules": [asdict(m) for m in ranked[:top_modules]],
    }
