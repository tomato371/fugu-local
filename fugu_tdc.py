# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_tdc — Test-Driven Criticism (Doc A Phase 2)。

Critic が回答コードを評価する前に、要求仕様から pytest テストを起草し、
fugu_sandbox 内で提案コードに対して実行する。全テスト green のときだけ承認。
失敗時は失敗ログを chat に渡して解答側を修正するループ（max_fix 回）も提供する。

すべて注入ベース: chat は fugu_llm.Chat、sandbox は fugu_sandbox.Sandbox。
オフラインテストでは FakeChat + 実 SubprocessSandbox（純 subprocess・LLM不要）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import fugu_sandbox
from fugu_sandbox import Sandbox, get_sandbox

# solution.py / test_solution.py を並べた一時ディレクトリ内で pytest を回す駆動スクリプト。
# sandbox.run() は cwd を一時ディレクトリに揃えるため相対名でよい。
_PYTEST_DRIVER = (
    "import sys\n"
    "import pytest\n"
    "sys.exit(pytest.main(['-q', '--no-header', 'test_solution.py']))\n"
)

_DRAFT_SYSTEM = (
    "You are a rigorous test engineer practicing test-driven criticism. "
    "Given requirements and a candidate Python module (it will be saved as "
    "solution.py), write pytest unit tests that verify the requirements. "
    "Import the code under test with `import solution` or "
    "`from solution import ...`. Cover normal cases and at least one edge case. "
    "Return ONE fenced python code block containing ONLY the test module."
)

_FIX_SYSTEM = (
    "You are a code-repair assistant. The candidate module solution.py failed "
    "the unit tests below. Return the COMPLETE corrected solution module in ONE "
    "fenced python code block. Keep the public interface the tests expect."
)


@dataclass
class TDCResult:
    passed: bool
    test_source: str = ""
    report: str = ""
    attempts: int = 0
    code: str = ""          # 修正ループ後の最終コード（未修正なら入力そのまま）
    drafted: bool = True    # テスト起草に成功したか


def draft_tests(requirements: str, code: str, chat) -> Optional[str]:
    """要求仕様と候補コードから pytest テストモジュールを起草する。

    fenced block 抽出 + compile() 構文ガード + 「solution を参照していること」の
    正気チェックを通らなければ None（起草失敗）。"""
    prompt = (
        f"Requirements:\n{requirements}\n\n"
        f"Candidate module (will be saved as solution.py):\n"
        f"```python\n{code}\n```\n\n"
        "Write the pytest test module."
    )
    try:
        reply = chat.complete(prompt, system=_DRAFT_SYSTEM, temperature=0.2)
    except Exception:
        return None
    source = fugu_sandbox.extract_code_block(reply)
    if not source:
        return None
    try:
        compile(source, "test_solution.py", "exec")
    except SyntaxError:
        return None
    if "solution" not in source:
        # solution を一切参照しないテストは何を書いても green になるため無効。
        return None
    return source


def run_tests(code: str, test_source: str, sandbox: Optional[Sandbox] = None,
              timeout: Optional[float] = None) -> "fugu_sandbox.SandboxResult":
    """solution.py + test_solution.py を配置して pytest を 1 回実行する。"""
    sandbox = sandbox or get_sandbox()  # Doc E3: 中央解決(既定は従来どおり subprocess)
    return sandbox.run(
        _PYTEST_DRIVER,
        files={"solution.py": code, "test_solution.py": test_source},
        timeout=timeout,
    )


def run_tdc(code: str, requirements: str, chat,
            sandbox: Optional[Sandbox] = None, max_fix: int = 2,
            timeout: Optional[float] = None) -> TDCResult:
    """TDC 本体: テスト起草 → 実行 → (失敗なら) 解答修正ループ。

    承認 (passed=True) は全テスト green のときのみ。起草に失敗した場合は
    drafted=False, passed=False で返し、呼び出し側が従来の LLM 審査へ
    フォールバックできるようにする。"""
    sandbox = sandbox or get_sandbox()  # Doc E3: 中央解決(既定は従来どおり subprocess)
    test_source = draft_tests(requirements, code, chat)
    if not test_source:
        return TDCResult(passed=False, drafted=False, code=code,
                         report="test drafting failed")
    current = code
    attempts = 0
    result = None
    for _ in range(max_fix + 1):
        attempts += 1
        result = run_tests(current, test_source, sandbox=sandbox, timeout=timeout)
        if result.ok:
            return TDCResult(passed=True, test_source=test_source,
                             report=result.stdout[-2000:], attempts=attempts,
                             code=current)
        if attempts > max_fix:
            break
        failure = result.output[-2000:] or "(no output)"
        prompt = (
            f"Requirements:\n{requirements}\n\n"
            f"Current solution.py:\n```python\n{current}\n```\n\n"
            f"Test module test_solution.py:\n```python\n{test_source}\n```\n\n"
            f"Pytest failure output:\n```\n{failure}\n```\n\n"
            "Return the complete corrected solution module."
        )
        try:
            reply = chat.complete(prompt, system=_FIX_SYSTEM, temperature=0.2)
        except Exception:
            break
        fixed = fugu_sandbox.extract_code_block(reply)
        if not fixed or fixed == current:
            break
        current = fixed
    return TDCResult(passed=False, test_source=test_source,
                     report=(result.output[-2000:] if result else ""),
                     attempts=attempts, code=current)
