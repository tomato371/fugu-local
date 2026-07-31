"""fugu_sandbox / fugu_llm のオフラインテスト（Ollama 不要・純 subprocess）。"""
from __future__ import annotations

import shutil
import sys

import pytest

import fugu_llm
import fugu_sandbox
from fugu_sandbox import SandboxResult, SubprocessSandbox, run_with_self_debug


# ---------- SubprocessSandbox ----------

def test_run_captures_stdout_and_exit0():
    res = SubprocessSandbox().run("print('hello sandbox')")
    assert res.ok
    assert res.exit_code == 0
    assert "hello sandbox" in res.stdout
    assert not res.timed_out


def test_run_captures_stderr_and_nonzero_exit():
    res = SubprocessSandbox().run("import sys; sys.stderr.write('boom'); sys.exit(3)")
    assert not res.ok
    assert res.exit_code == 3
    assert "boom" in res.stderr


def test_run_exception_traceback_in_stderr():
    res = SubprocessSandbox().run("raise ValueError('broken thing')")
    assert not res.ok
    assert "ValueError" in res.stderr
    assert "broken thing" in res.output


def test_run_timeout_kills():
    res = SubprocessSandbox().run("while True:\n    pass", timeout=1.0)
    assert res.timed_out
    assert not res.ok


def test_stdin_is_devnull_so_input_fails_fast():
    res = SubprocessSandbox().run("input()", timeout=10.0)
    assert not res.ok
    assert not res.timed_out  # ハングせず EOFError で即死する
    assert "EOFError" in res.stderr


def test_files_staging():
    code = "print(open('data.txt', encoding='utf-8').read())"
    res = SubprocessSandbox().run(code, files={"data.txt": "staged-content"})
    assert res.ok
    assert "staged-content" in res.stdout


def test_run_argv():
    res = SubprocessSandbox().run_argv([sys.executable, "-c", "print('argv path')"])
    assert res.ok
    assert "argv path" in res.stdout


def test_run_argv_env_injection():
    res = SubprocessSandbox().run_argv(
        [sys.executable, "-c", "import os; print(os.environ['FUGU_X'])"],
        env={"FUGU_X": "injected"})
    assert res.ok
    assert "injected" in res.stdout


def test_unsupported_lang():
    res = SubprocessSandbox().run("puts 'hi'", lang="ruby")
    assert not res.ok
    assert "unsupported lang" in res.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_run_bash():
    res = SubprocessSandbox().run("echo from-bash", lang="bash")
    assert res.ok
    assert "from-bash" in res.stdout


# ---------- extract_code_block ----------

def test_extract_fenced_block():
    text = "Here is the fix:\n```python\nprint('fixed')\n```\nDone."
    assert fugu_sandbox.extract_code_block(text) == "print('fixed')"


def test_extract_bare_code_that_compiles():
    assert fugu_sandbox.extract_code_block("x = 1\nprint(x)") == "x = 1\nprint(x)"


def test_extract_prose_returns_none():
    assert fugu_sandbox.extract_code_block("I cannot fix this, sorry!") is None


# ---------- run_with_self_debug ----------

def test_self_debug_fixes_on_second_attempt():
    chat = fugu_llm.FakeChat(responses=["```python\nprint('repaired')\n```"])
    result, code, attempts = run_with_self_debug(
        "raise RuntimeError('first try fails')", chat, max_retries=3)
    assert result.ok
    assert attempts == 2
    assert "repaired" in code
    assert len(chat.calls) == 1


def test_self_debug_passes_error_to_chat():
    chat = fugu_llm.FakeChat(responses=["```python\nprint('ok')\n```"])
    run_with_self_debug("raise ValueError('telltale-marker')", chat, max_retries=1)
    assert "telltale-marker" in chat.calls[0]["prompt"]


def test_self_debug_gives_up_after_max_retries():
    # 常に同じ壊れたコードを返す → 同一コード検出で早期打ち切りされないよう、
    # 毎回わずかに違う壊れコードを返して max_retries 消費経路を通す。
    counter = {"n": 0}

    def broken(prompt):
        counter["n"] += 1
        return f"```python\nraise RuntimeError('still broken {counter['n']}')\n```"

    chat = fugu_llm.FakeChat(fn=broken)
    result, _code, attempts = run_with_self_debug(
        "raise RuntimeError('v0')", chat, max_retries=2)
    assert not result.ok
    assert attempts == 3  # 初回 + 2 リトライ


def test_self_debug_stops_when_chat_returns_same_code():
    chat = fugu_llm.FakeChat(default="```python\nraise RuntimeError('v0')\n```")
    result, _code, attempts = run_with_self_debug(
        "raise RuntimeError('v0')", chat, max_retries=5)
    assert not result.ok
    assert attempts == 1  # 同一コードが返った時点で再実行せず打ち切り


def test_self_debug_stops_when_chat_raises():
    class DeadChat:
        def complete(self, prompt, *, system=None, fmt=None, temperature=0.2):
            raise RuntimeError("__ERROR__: ollama down")

    result, code, attempts = run_with_self_debug(
        "raise RuntimeError('x')", DeadChat(), max_retries=3)
    assert not result.ok
    assert attempts == 1
    assert "raise RuntimeError('x')" == code


def test_self_debug_immediate_success_never_calls_chat():
    chat = fugu_llm.FakeChat()  # 応答なし → 呼ばれたら AssertionError
    result, _code, attempts = run_with_self_debug("print('fine')", chat)
    assert result.ok
    assert attempts == 1


# ---------- fugu_llm.AskChat ----------

def test_askchat_builds_messages_and_passes_through(monkeypatch):
    import fugu_local
    recorded = {}

    def fake_ask(model, messages, temperature, think=None, fmt=None, label=None,
                 num_predict=None, num_ctx=None):
        recorded.update(model=model, messages=messages, temperature=temperature,
                        fmt=fmt, label=label, num_predict=num_predict)
        return "the answer"

    monkeypatch.setattr(fugu_local, "ask", fake_ask)
    chat = fugu_llm.AskChat(model="test-model", label="unit", num_predict=64)
    out = chat.complete("question?", system="be brief", temperature=0.5)
    assert out == "the answer"
    assert recorded["model"] == "test-model"
    assert recorded["messages"][0] == {"role": "system", "content": "be brief"}
    assert recorded["messages"][1] == {"role": "user", "content": "question?"}
    assert recorded["temperature"] == 0.5
    assert recorded["num_predict"] == 64


def test_askchat_error_sentinel_raises(monkeypatch):
    import fugu_local
    monkeypatch.setattr(fugu_local, "ask",
                        lambda *a, **k: "__ERROR__: model exploded")
    with pytest.raises(RuntimeError, match="model exploded"):
        fugu_llm.AskChat(model="m").complete("q")


def test_askchat_resolves_default_model(monkeypatch):
    import fugu_local
    seen = {}

    def fake_ask(model, messages, temperature, **kw):
        seen["model"] = model
        return "ok"

    monkeypatch.setattr(fugu_local, "ask", fake_ask)
    monkeypatch.setattr(fugu_local, "CONDUCTOR", "conductor-model")
    fugu_llm.AskChat().complete("q")
    assert seen["model"] == "conductor-model"


# ---------- fugu_local.run_python フック ----------

def test_run_python_sandbox_hook_optin(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_SANDBOX", "1")
    ok, out = fugu_local.run_python("print('via hook')")
    assert ok
    assert "via hook" in out


def test_run_python_default_path_unchanged(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_SANDBOX", raising=False)
    ok, out = fugu_local.run_python("print('default path')")
    assert ok
    assert "default path" in out


# ---------- SandboxResult ----------

def test_sandbox_result_ok_semantics():
    assert SandboxResult(exit_code=0).ok
    assert not SandboxResult(exit_code=0, timed_out=True).ok
    assert not SandboxResult(exit_code=1).ok
