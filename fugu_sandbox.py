"""fugu_sandbox — 安全なローカル実行サンドボックスと自己デバッグループ (Doc A Phase 1)。

生成コード/コマンドを一時ディレクトリ内で subprocess 実行し、stdout / stderr /
exit_code / timed_out を構造化して返す。fugu_local.run_python() が PoT 検証用の
既存経路として残る一方、本モジュールは TDC (fugu_tdc)・IDE エンドポイント
(fugu_api /test-run)・自己進化 (fugu_evolve) が共有する注入可能な実行基盤となる。

オフラインテスト規約: Sandbox は Protocol であり、テストは FakeSandbox を注入する。
SubprocessSandbox 自体も LLM/ネットワーク不要（純 subprocess）なのでテスト可。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

DEFAULT_TIMEOUT = 30.0


@dataclass
class SandboxResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    cmd: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """stdout+stderr 結合（修正ヒント用途。fugu_local.run_python と同じ思想）。"""
        return ((self.stdout or "") + (self.stderr or "")).strip()


class Sandbox(Protocol):
    def run(self, code: str, lang: str = "python", timeout: Optional[float] = None,
            files: Optional[Dict[str, str]] = None) -> SandboxResult:
        ...

    def run_argv(self, argv: List[str], cwd: Optional[str] = None,
                 timeout: Optional[float] = None,
                 env: Optional[Dict[str, str]] = None) -> SandboxResult:
        ...


class SubprocessSandbox:
    """一時ディレクトリ + subprocess による標準実装。

    - stdin は DEVNULL（input() は即 EOFError で fail-fast。fugu_local.run_python の
      2026-07-22 の知見を踏襲: ハング・stdin 汚染・非決定性の除去）。
    - files でスクリプトと同じディレクトリに補助ファイルを事前配置できる
      （TDC が solution.py + test_solution.py を並べる用途）。
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def run(self, code: str, lang: str = "python", timeout: Optional[float] = None,
            files: Optional[Dict[str, str]] = None) -> SandboxResult:
        timeout = timeout or self.timeout
        with tempfile.TemporaryDirectory(prefix="fugu_sbx_") as tmp:
            for name, content in (files or {}).items():
                path = os.path.join(tmp, name)
                os.makedirs(os.path.dirname(path) or tmp, exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(content)
            # argv にはスクリプトを相対名で渡す（run_argv が cwd=tmp で実行する）。
            # 絶対 Windows パスだと WSL の bash.exe がパスを解決できない
            # （バックスラッシュが食われ "C:Users...main.sh: No such file" で exit 127）。
            # 相対名なら WSL bash は cwd を /mnt/c/... に変換して継承するため動く。
            if lang == "python":
                script = os.path.join(tmp, "main.py")
                argv = [sys.executable, "-X", "utf8", "main.py"]
            elif lang == "bash":
                bash = shutil.which("bash")
                if not bash:
                    return SandboxResult(stderr="bash not available on this system",
                                         exit_code=127)
                script = os.path.join(tmp, "main.sh")
                argv = [bash, "main.sh"]
            else:
                return SandboxResult(stderr=f"unsupported lang: {lang}", exit_code=2)
            with open(script, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(code)
            return self.run_argv(argv, cwd=tmp, timeout=timeout)

    def run_argv(self, argv: List[str], cwd: Optional[str] = None,
                 timeout: Optional[float] = None,
                 env: Optional[Dict[str, str]] = None) -> SandboxResult:
        timeout = timeout or self.timeout
        run_env = None
        if env:
            run_env = dict(os.environ)
            run_env.update(env)
        try:
            r = subprocess.run(
                argv, cwd=cwd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=timeout, stdin=subprocess.DEVNULL, env=run_env,
            )
            return SandboxResult(stdout=r.stdout or "", stderr=r.stderr or "",
                                 exit_code=r.returncode, cmd=list(argv))
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                stdout=_as_text(e.stdout), stderr=_as_text(e.stderr),
                exit_code=-1, timed_out=True, cmd=list(argv))
        except Exception as e:  # 実行環境自体の失敗（実行ファイル不在など）
            return SandboxResult(stderr=f"runner error: {e}", exit_code=-1,
                                 cmd=list(argv))


class DockerSandbox:
    """Docker コンテナ内で実行するオプショナル実装（docker が無ければ available()=False）。"""

    def __init__(self, image: str = "python:3.11-slim", timeout: float = DEFAULT_TIMEOUT):
        self.image = image
        self.timeout = timeout
        self._inner = SubprocessSandbox(timeout=timeout)

    @staticmethod
    def available() -> bool:
        return shutil.which("docker") is not None

    def run(self, code: str, lang: str = "python", timeout: Optional[float] = None,
            files: Optional[Dict[str, str]] = None) -> SandboxResult:
        if not self.available():
            return SandboxResult(stderr="docker not available", exit_code=127)
        timeout = timeout or self.timeout
        tmp = tempfile.mkdtemp(prefix="fugu_dsbx_")
        try:
            for name, content in (files or {}).items():
                with open(os.path.join(tmp, name), "w", encoding="utf-8") as fh:
                    fh.write(content)
            script = "main.py" if lang == "python" else "main.sh"
            with open(os.path.join(tmp, script), "w", encoding="utf-8",
                      newline="\n") as fh:
                fh.write(code)
            runner = ["python", f"/work/{script}"] if lang == "python" \
                else ["bash", f"/work/{script}"]
            argv = ["docker", "run", "--rm", "--network", "none",
                    "-v", f"{tmp}:/work", "-w", "/work", self.image] + runner
            return self._inner.run_argv(argv, timeout=timeout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def run_argv(self, argv: List[str], cwd: Optional[str] = None,
                 timeout: Optional[float] = None,
                 env: Optional[Dict[str, str]] = None) -> SandboxResult:
        return self._inner.run_argv(argv, cwd=cwd, timeout=timeout, env=env)


_FENCE_RE = re.compile(r"```(?:python|py|bash|sh)?\s*\n(.*?)```", re.DOTALL)

_FIX_SYSTEM = (
    "You are a code-repair assistant. The user gives you a script and the error "
    "output from running it. Return the COMPLETE corrected script in ONE fenced "
    "code block. No explanations outside the block."
)


def extract_code_block(text: str) -> Optional[str]:
    """fenced code block を抽出。無ければ、テキスト全体が Python として compile
    できる場合のみそのまま採用（LLM が裸のコードを返すケースの救済）。"""
    m = _FENCE_RE.search(text or "")
    if m:
        code = m.group(1).strip()
        return code or None
    candidate = (text or "").strip()
    if not candidate:
        return None
    try:
        compile(candidate, "<candidate>", "exec")
        return candidate
    except SyntaxError:
        return None


def _record_episode(task: str, result: SandboxResult, attempts: int) -> None:
    """FUGU_MEMORY=1 のときだけ自己デバッグの顛末をエピソード記憶に残す (Doc D1)。
    未設定なら完全 no-op。記録失敗は本処理に影響させない。"""
    if os.environ.get("FUGU_MEMORY") != "1":
        return
    try:
        from fugu_core import memory as fugu_memory
    except ImportError:
        return
    try:
        if result.ok:
            lesson = (f"self-debug fixed the script in {attempts} attempt(s)"
                      if attempts > 1 else "ran clean on first attempt")
        else:
            lesson = f"still failing after {attempts} attempt(s): " \
                     f"{result.output[:300]}"
        fugu_memory.get_default_memory().record(fugu_memory.Episode(
            kind="sandbox", task=task[:200],
            outcome="success" if result.ok else "failure", lesson=lesson))
    except Exception:
        pass


def run_with_self_debug(code: str, chat, sandbox: Optional[Sandbox] = None,
                        max_retries: int = 3, lang: str = "python",
                        files: Optional[Dict[str, str]] = None,
                        timeout: Optional[float] = None,
                        ) -> Tuple[SandboxResult, str, int]:
    """自己デバッグループ: 実行に失敗したら stderr を chat に渡して修正版を得て再実行。

    戻り値: (最終 SandboxResult, 最終コード, 実行試行回数)。
    chat は fugu_llm.Chat プロトコル（complete(prompt, *, system=...) -> str）。
    修正案からコードを抽出できない、または前回と同一コードが返った場合は
    それ以上進展しないため打ち切る。
    """
    sandbox = sandbox or SubprocessSandbox()
    attempts = 0
    current = code
    result = SandboxResult(stderr="not executed", exit_code=-1)
    for _ in range(max_retries + 1):
        attempts += 1
        result = sandbox.run(current, lang=lang, timeout=timeout, files=files)
        if result.ok:
            _record_episode(code, result, attempts)
            return result, current, attempts
        if attempts > max_retries:
            break
        error_text = result.output[-2000:] or "(no output)"
        prompt = (
            "The following script failed.\n\n"
            f"```{lang}\n{current}\n```\n\n"
            f"Error output:\n```\n{error_text}\n```\n\n"
            "Return the complete corrected script."
        )
        try:
            reply = chat.complete(prompt, system=_FIX_SYSTEM, temperature=0.2)
        except Exception:
            break  # LLM 不通なら現状の失敗結果を確定返しする
        fixed = extract_code_block(reply)
        if not fixed or fixed == current:
            break
        current = fixed
    _record_episode(code, result, attempts)
    return result, current, attempts


def _as_text(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)
