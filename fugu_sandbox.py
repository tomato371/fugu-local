# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
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

#: 生成コード実行プロセスの既定メモリ上限(MB)。None/0 で無効。
#: env FUGU_SANDBOX_MEMORY_MB で上書き可(ベストエフォート: 適用に失敗しても
#: 実行自体は従来どおり続行する)。
DEFAULT_MEMORY_MB = 1024


def _memory_limit_mb() -> Optional[int]:
    raw = os.environ.get("FUGU_SANDBOX_MEMORY_MB")
    if raw is None:
        return DEFAULT_MEMORY_MB
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        return DEFAULT_MEMORY_MB


def _assign_windows_job(process, memory_mb: int):
    """Windows Job Object でプロセスにメモリ上限を課す(ベストエフォート)。

    失敗しても None を返すだけで実行は続く。返り値のハンドルはプロセス終了まで
    呼び出し側が保持すること(GC されるとジョブが閉じる)。
    """
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        kernel32 = ctypes.windll.kernel32

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount",
                "OtherOperationCount", "ReadTransferCount",
                "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JobObjectExtendedLimitInformation = 9
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = memory_mb * 1024 * 1024
        if not kernel32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return None
        handle = kernel32.OpenProcess(
            PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, process.pid)
        if not handle:
            kernel32.CloseHandle(job)
            return None
        ok = kernel32.AssignProcessToJobObject(job, handle)
        kernel32.CloseHandle(handle)
        if not ok:
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


def _unix_preexec(memory_mb: int):
    """Unix: 子プロセス側で RLIMIT_AS を課す preexec_fn を返す。"""
    def _limit():
        import resource
        limit_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    return _limit


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


def _approved_to_run(code: str) -> bool:
    """FUGU_REQUIRE_APPROVAL=1 のときだけ人間承認を待つ(未設定は素通し)。"""
    if os.environ.get("FUGU_REQUIRE_APPROVAL") != "1":
        return True
    try:
        import fugu_approval
    except ImportError:
        return True
    return fugu_approval.require_approval("sandbox-run", code[:200])


class SubprocessSandbox:
    """一時ディレクトリ + subprocess による標準実装。

    - stdin は DEVNULL（input() は即 EOFError で fail-fast。fugu_local.run_python の
      2026-07-22 の知見を踏襲: ハング・stdin 汚染・非決定性の除去）。
    - files でスクリプトと同じディレクトリに補助ファイルを事前配置できる
      （TDC が solution.py + test_solution.py を並べる用途）。
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT,
                 memory_mb: Optional[int] = None):
        self.timeout = timeout
        # None は「env 既定に従う」。明示 0 以下で無効化。
        self.memory_mb = _memory_limit_mb() if memory_mb is None else (
            memory_mb if memory_mb > 0 else None)

    def run(self, code: str, lang: str = "python", timeout: Optional[float] = None,
            files: Optional[Dict[str, str]] = None) -> SandboxResult:
        # FUGU_REQUIRE_APPROVAL=1: 任意コード実行の入口ゲート (Doc E3)。
        # run_argv(pytest 等の固定コマンド)は対象外 — 検証ループを承認連打に
        # しないため、「LLM が書いたコードの実行」だけを守る。
        if not _approved_to_run(code):
            return SandboxResult(stderr="approval denied or timed out",
                                 exit_code=-1)
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
        # メモリ上限(ベストエフォート): Unix は setrlimit、Windows は Job Object。
        # 適用に失敗しても実行は従来どおり続行する。
        preexec = (_unix_preexec(self.memory_mb)
                   if os.name == "posix" and self.memory_mb else None)
        try:
            proc = subprocess.Popen(
                argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                stdin=subprocess.DEVNULL, env=run_env, preexec_fn=preexec,
            )
        except Exception as e:  # 実行環境自体の失敗（実行ファイル不在など）
            return SandboxResult(stderr=f"runner error: {e}", exit_code=-1,
                                 cmd=list(argv))
        job = (_assign_windows_job(proc, self.memory_mb)
               if os.name == "nt" and self.memory_mb else None)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return SandboxResult(stdout=stdout or "", stderr=stderr or "",
                                 exit_code=proc.returncode, cmd=list(argv))
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return SandboxResult(
                stdout=_as_text(stdout), stderr=_as_text(stderr),
                exit_code=-1, timed_out=True, cmd=list(argv))
        except Exception as e:
            return SandboxResult(stderr=f"runner error: {e}", exit_code=-1,
                                 cmd=list(argv))
        finally:
            del job  # ジョブハンドルはプロセス終了までこのフレームで保持


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
        if not _approved_to_run(code):  # Doc E3: 任意コード実行の入口ゲート
            return SandboxResult(stderr="approval denied or timed out",
                                 exit_code=-1)
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


# ------------------------------------------------------------------ 既定解決 (Doc E3)

_DOCKER_READY: Optional[bool] = None
_FALLBACK_WARNED = False


def _docker_ready() -> bool:
    """docker バイナリの存在に加えて daemon の応答まで確認する(結果はキャッシュ)。

    `shutil.which("docker")` だけだと Docker Desktop 停止中でも True になり、
    実行時に全コンテナ起動が失敗する — daemon probe で実際に使える時だけ昇格する。
    """
    global _DOCKER_READY
    if _DOCKER_READY is None:
        if not DockerSandbox.available():
            _DOCKER_READY = False
        else:
            try:
                probe = subprocess.run(["docker", "info"], capture_output=True,
                                       timeout=10)
                _DOCKER_READY = probe.returncode == 0
            except Exception:
                _DOCKER_READY = False
    return _DOCKER_READY


def get_sandbox(timeout: float = DEFAULT_TIMEOUT,
                prefer: Optional[str] = None) -> Sandbox:
    """コード実行用サンドボックスの中央解決点 (Doc E3)。

    既定は SubprocessSandbox(メモリ上限付き)。``prefer`` または env
    ``FUGU_SANDBOX_BACKEND`` に "docker" / "auto" を指定したときだけ、daemon が
    応答すれば DockerSandbox(イメージは env ``FUGU_SANDBOX_IMAGE`` で指定、
    `--network none`)へ昇格する。

    **無条件の自動昇格にしない理由**: 素の python:3.11-slim には pytest も
    sympy も無く、TDC のテスト実行や PoT の数式検算がコンテナ内で全滅する —
    隔離のために完走性を壊さない。docker を使う場合は依存入りイメージを
    用意して FUGU_SANDBOX_IMAGE で指すこと。docker 指名で daemon 不応答の
    場合は Subprocess にフォールバックし初回のみ警告する。
    """
    global _FALLBACK_WARNED
    prefer = prefer or os.environ.get("FUGU_SANDBOX_BACKEND") or "subprocess"
    if prefer in ("docker", "auto") and _docker_ready():
        image = os.environ.get("FUGU_SANDBOX_IMAGE")
        return (DockerSandbox(image=image, timeout=timeout) if image
                else DockerSandbox(timeout=timeout))
    if prefer == "docker" and not _FALLBACK_WARNED:
        _FALLBACK_WARNED = True
        print("   [sandbox] docker 指定ですが daemon 不応答 → "
              "SubprocessSandbox にフォールバック(コンテナ隔離なし・メモリ上限のみ)")
    return SubprocessSandbox(timeout=timeout)


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
        fugu_memory.maybe_consolidate(fugu_memory.get_default_memory())
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
    sandbox = sandbox or get_sandbox()  # Doc E3: Docker 稼働時は自動昇格
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
