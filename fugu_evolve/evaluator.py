"""fugu_evolve.evaluator — サンドボックス検証と自己デバッグループ (Doc C Phase 4)。

workspace (C3) 上の変更を機械的に検証する。合格条件は厳格に2つ:

1. **テスト 100% pass** — run_pytest (C1) で failed=0 かつ errors=0。失敗時は
   失敗ログを chat に渡して「1ファイル全置換パッチ」(JSON スキーマ制約+
   compile ガード)を得て再実行する自己デバッグを ``max_attempts`` 回まで回す。
2. **ベンチ非退行** — baseline と比較してスコアが下がっていないこと。ベンチが
   実行できない状況(オフライン等)は「clean skip」として理由を記録し、退行
   扱いにはしない(検証不能と退行を混同しない)。

両方を満たしたときだけ verdict は ``VERIFIED``。それ以外は ``FAILED``。
merge するかは CLI (C5) の Critic 承認に委ね、ここでは判定だけを返す。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fugu_sandbox import Sandbox, extract_code_block
from fugu_evolve.profiler import BenchReport, PytestReport, run_bench, run_pytest

VERIFIED = "VERIFIED"
FAILED = "FAILED"

PATCH_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "code": {"type": "string"},
    },
    "required": ["path", "code"],
}

_REPAIR_SYSTEM = (
    "You are a code-repair assistant inside an autonomous self-improvement "
    "loop. Given failing pytest output from a repository branch, return the "
    "COMPLETE corrected content of exactly ONE file as JSON "
    '{"path": "relative/path.py", "code": "<entire new file content>"}. '
    "Make the minimal change that fixes the failures and keep public "
    "interfaces the tests expect. Reply with JSON only."
)


@dataclass
class Verification:
    """検証結果。verdict が VERIFIED のときだけ merge 候補になる。"""

    verdict: str
    attempts: int
    pytest: PytestReport
    bench: Optional[BenchReport] = None
    bench_regressed: bool = False
    notes: List[str] = field(default_factory=list)


def _request_patch(chat, tail: str) -> Optional[Dict[str, str]]:
    """失敗ログから修正パッチ {"path", "code"} を得る。不正・失敗は None。

    code が fenced block で包まれていても救済し、.py は compile ガードを通す
    (構文の壊れたパッチを作業ツリーに書いてループを悪化させない)。
    """
    prompt = (
        f"Failing pytest output:\n```\n{tail}\n```\n\n"
        'Return {"path": ..., "code": ...} with the complete corrected file.'
    )
    try:
        raw = chat.complete(prompt, system=_REPAIR_SYSTEM, fmt=PATCH_SCHEMA,
                            temperature=0.2)
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    path, code = obj.get("path"), obj.get("code")
    if not (isinstance(path, str) and path.strip() and isinstance(code, str)):
        return None
    extracted = extract_code_block(code)
    if extracted is not None and "```" in code:
        code = extracted
    if not code.strip():
        return None
    if path.strip().endswith(".py"):
        try:
            compile(code, path, "exec")
        except SyntaxError:
            return None
    return {"path": path.strip(), "code": code}


def _compare_bench(baseline: Dict[str, object], current: BenchReport,
                   tolerance: float, notes: List[str]) -> bool:
    """ベンチ退行判定。比較不能なら clean skip(False=非退行扱い+理由記録)。"""
    base = baseline.get("bench") if isinstance(baseline, dict) else None
    base_ran = bool(isinstance(base, dict) and base.get("ran"))
    base_score = base.get("score") if isinstance(base, dict) else None
    if not base_ran or not isinstance(base_score, (int, float)):
        notes.append("bench comparison skipped: no usable baseline score")
        return False
    if not current.ran or current.score is None:
        notes.append(f"bench comparison skipped: current run unavailable "
                     f"({current.reason or 'no score'})")
        return False
    if current.score < float(base_score) - tolerance:
        notes.append(f"bench regressed: {base_score} -> {current.score}")
        return True
    notes.append(f"bench non-regression ok: {base_score} -> {current.score}")
    return False


def _record_episode(repo: str, verification: "Verification") -> None:
    """FUGU_MEMORY=1 のときだけ検証の顛末をエピソード記憶に残す (Doc D1)。"""
    if os.environ.get("FUGU_MEMORY") != "1":
        return
    try:
        from fugu_core import memory as fugu_memory
    except ImportError:
        return
    try:
        fugu_memory.get_default_memory().record(fugu_memory.Episode(
            kind="evolve", task=f"verify branch in {repo}"[:200],
            outcome="success" if verification.verdict == VERIFIED else "failure",
            lesson=("; ".join(verification.notes)[:300]
                    or f"verdict={verification.verdict}")))
    except Exception:
        pass


def verify(workspace, sandbox: Optional[Sandbox], chat,
           baseline: Dict[str, object], max_attempts: int = 3,
           bench_argv: Optional[List[str]] = None,
           offline: Optional[bool] = None,
           tolerance: float = 0.0) -> Verification:
    """workspace の現状態を検証する(C4 の入口)。

    ``max_attempts`` は pytest 実行回数の上限(修正は最大 max_attempts-1 回)。
    修正パッチは workspace.apply_edit 経由で書くため、auto-evolve ブランチ外
    では構造的に書けない(C3 のガードがそのまま効く)。baseline は
    build_health_report (C1) の dict。
    """
    notes: List[str] = []
    report = PytestReport(ran=False)
    attempts = 0
    for attempt in range(1, max(1, max_attempts) + 1):
        attempts = attempt
        report = run_pytest(workspace.repo, sandbox=sandbox)
        if report.ok:
            break
        if attempt >= max_attempts:
            notes.append(f"still failing after {attempt} run(s)")
            break
        patch = _request_patch(chat, report.tail)
        if patch is None:
            notes.append("self-repair aborted: no valid patch from model")
            break
        try:
            workspace.apply_edit(patch["path"], patch["code"])
        except Exception as exc:
            notes.append(f"self-repair aborted: apply failed ({exc})")
            break
        notes.append(f"repair attempt {attempt}: rewrote {patch['path']}")

    if not report.ok:
        verification = Verification(verdict=FAILED, attempts=attempts,
                                    pytest=report, notes=notes)
        _record_episode(workspace.repo, verification)
        return verification

    current_bench = run_bench(workspace.repo, bench_argv=bench_argv,
                              sandbox=sandbox, offline=offline)
    regressed = _compare_bench(baseline, current_bench, tolerance, notes)
    verdict = FAILED if regressed else VERIFIED
    verification = Verification(verdict=verdict, attempts=attempts, pytest=report,
                                bench=current_bench, bench_regressed=regressed,
                                notes=notes)
    _record_episode(workspace.repo, verification)
    return verification
