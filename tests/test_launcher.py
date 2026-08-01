"""fugu_launcher のオフラインテスト(モデル・GPU・ネット不要)。

ランチャーは subprocess を起動する層なので、テストは
「どんな argv / env を組み立てるか」と「壊れた入力で落ちないか」だけを見る
(実際の起動は行わない)。
"""
import json
import sys

import pytest

import fugu_launcher as L


# ------------------------------------------------------------------ build_env

def _settings(**over):
    s = json.loads(json.dumps(L.DEFAULT_SETTINGS))
    s.update(over)
    return s


def test_build_env_omits_flags_that_are_off():
    env = L.build_env(_settings(), base={})
    for name, _d, _c in L.FLAGS:
        assert name not in env
    assert "FUGU_THINKING_BUDGET" not in env


def test_build_env_sets_enabled_flags_and_budget():
    s = _settings(thinking_budget="high", vision_model="llava")
    s["flags"]["FUGU_SANDBOX"] = True
    s["flags"]["FUGU_TDC"] = True
    env = L.build_env(s, base={})
    assert env["FUGU_SANDBOX"] == "1" and env["FUGU_TDC"] == "1"
    assert env["FUGU_THINKING_BUDGET"] == "high"
    assert env["FUGU_VISION_MODEL"] == "llava"
    assert env["OLLAMA_URL"] == "http://localhost:11434"
    assert "FUGU_BROWSER" not in env          # off のものは残さない


def test_build_env_clears_inherited_flag_when_toggled_off():
    """親シェルに残った FUGU_SANDBOX=1 が、OFF 設定を上書きしないこと。"""
    env = L.build_env(_settings(), base={"FUGU_SANDBOX": "1",
                                         "FUGU_THINKING_BUDGET": "max"})
    assert "FUGU_SANDBOX" not in env
    assert "FUGU_THINKING_BUDGET" not in env


# -------------------------------------------------------------- build_command

def test_build_command_cli_uses_only_existing_arguments():
    argv = L.build_command("cli", {"question": "91は素数?", "search": True,
                                   "rag": ["./docs"], "out": "a.md"})
    assert argv[:3] == [sys.executable, "fugu_local.py", "91は素数?"]
    assert "--search" in argv
    assert argv[argv.index("--rag") + 1] == "./docs"
    assert argv[argv.index("--out") + 1] == "a.md"


def test_build_command_cli_interactive_has_no_positional():
    assert L.build_command("cli", {"question": ""}) == \
        [sys.executable, "fugu_local.py"]


@pytest.mark.parametrize("action,expected_tail", [
    ("web", ["fugu_web.py"]),
    ("tui", ["fugu_tui.py"]),
    # 0.0.0.0 ではなく 127.0.0.1: uvicorn がログに出す URL をそのまま
    # ブラウザで開けるようにする(Windows は 0.0.0.0 に接続できない)
    ("api", ["-m", "uvicorn", "fugu_api:app", "--host", "127.0.0.1",
             "--port", "8000"]),
    ("bench-list", ["bench_fugu.py", "list"]),
    ("bench-report", ["bench_fugu.py", "report"]),
    ("evolve-dry", ["-m", "fugu_evolve", "--repo", ".", "--dry-run"]),
    ("evolve-pr", ["-m", "fugu_evolve", "--repo", ".", "--pr-mode",
                   "--max-proposals", "1"]),
])
def test_build_command_static_entries(action, expected_tail):
    assert L.build_command(action) == [sys.executable] + expected_tail


def test_build_command_bench_run_and_rag():
    assert L.build_command("bench-run", {"dataset": "aime25", "config": "fugu",
                                         "limit": 3})[2:] == \
        ["run", "--dataset", "aime25", "--config", "fugu", "--limit", "3"]
    assert L.build_command("rag-ask", {"question": "洪水は?"})[1:] == \
        ["-m", "fugu_rag", "ask", "洪水は?"]
    assert L.build_command("rag-research", {"question": "洪水は?"})[1:] == \
        ["-m", "fugu_rag", "research", "洪水は?", "--branches", "2",
         "--depth", "1"]


def test_build_command_rejects_unknown_action():
    with pytest.raises(KeyError):
        L.build_command("nope")


# ------------------------------------------------------------------ check_env

def _fake_tags(monkeypatch, names, fail=False):
    def fake(url, timeout=2.0):
        assert url  # URL は設定から来る
        return None if fail else list(names)
    monkeypatch.setattr(L, "installed_models", fake)


def test_check_env_reports_missing_models(monkeypatch):
    _fake_tags(monkeypatch, ["qwen3:4b", "gpt-oss:20b"])
    report = L.check_env(_settings())
    assert report["ollama_ok"] is True
    assert report["missing_required"] == []          # fallback はある
    assert "qwen3-coder:30b" in report["missing_council"]
    assert report["missing_embed"] == [L.EMBED_MODEL]
    assert report["ok"] is True                      # 縮退して動くので ok


def test_check_env_flags_missing_fallback_model(monkeypatch):
    _fake_tags(monkeypatch, ["gpt-oss:20b"])
    report = L.check_env(_settings())
    assert report["missing_required"] == [L.FALLBACK_MODEL]
    assert report["ok"] is False
    assert "ollama pull qwen3:4b" in L.format_check(report)


def test_check_env_survives_ollama_being_down(monkeypatch):
    _fake_tags(monkeypatch, [], fail=True)
    report = L.check_env(_settings())          # 例外を投げないこと
    assert report["ollama_ok"] is False and report["ok"] is False
    assert "ollama serve" in L.format_check(report)


def test_installed_models_returns_none_on_connection_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(L.urllib.request, "urlopen", boom)
    assert L.installed_models("http://127.0.0.1:1") is None


def test_installed_models_normalizes_latest_suffix(monkeypatch):
    class _Res:
        def read(self):
            return json.dumps({"models": [{"name": "qwen3:4b:latest"},
                                          {"name": "gpt-oss:20b"}]}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(L.urllib.request, "urlopen", lambda *a, **k: _Res())
    assert L.installed_models("http://x") == ["qwen3:4b", "gpt-oss:20b"]


# ------------------------------------------------------------------- settings

def test_settings_round_trip(tmp_path):
    path = tmp_path / "s.json"
    s = _settings(thinking_budget="ultra")
    s["flags"]["FUGU_TASKS"] = True
    L.save_settings(s, path)
    loaded = L.load_settings(path)
    assert loaded["thinking_budget"] == "ultra"
    assert loaded["flags"]["FUGU_TASKS"] is True


def test_load_settings_falls_back_on_broken_json(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert L.load_settings(path) == L.DEFAULT_SETTINGS


def test_load_settings_ignores_unknown_keys_and_bad_budget(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"thinking_budget": "turbo",
                                "flags": {"NOT_A_FLAG": True},
                                "junk": 1}), encoding="utf-8")
    loaded = L.load_settings(path)
    assert loaded["thinking_budget"] == "off"
    assert "NOT_A_FLAG" not in loaded["flags"]


def test_load_settings_missing_file_is_defaults(tmp_path):
    assert L.load_settings(tmp_path / "nope.json") == L.DEFAULT_SETTINGS


# ----------------------------------------------------------------- run_server

class _FakeProc:
    """Popen の代役: poll() が None を返す間は生存、その後 returncode で死ぬ。"""

    def __init__(self, alive_polls=0, returncode=0):
        self._alive_polls = alive_polls
        self.returncode = returncode
        self.waited = self.terminated = False

    def poll(self):
        if self._alive_polls > 0:
            self._alive_polls -= 1
            return None
        return self.returncode

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        pass


@pytest.fixture()
def server_env(monkeypatch):
    """run_server をオフラインで駆動するための共通スタブ一式。"""
    state = {"opened": [], "popen": [], "sleep": 0}
    monkeypatch.setattr(L.webbrowser, "open", lambda url: state["opened"].append(url))
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    monkeypatch.setattr(L, "missing_packages", lambda action: [])
    return state


def test_run_server_opens_browser_only_after_port_is_ready(server_env, monkeypatch):
    # 1回目の probe(起動前チェック)は閉じている → spawn → 2回目で開く
    probes = iter([False, True])
    monkeypatch.setattr(L, "_port_open", lambda *a, **k: next(probes))
    proc = _FakeProc(alive_polls=99)
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: proc)
    rc = L.run_server("api", L.DEFAULT_SETTINGS, port=8000,
                      url="http://localhost:8000/docs")
    assert server_env["opened"] == ["http://localhost:8000/docs"]
    assert proc.waited and rc == 0


def test_run_server_does_not_double_open_when_gradio_opens_itself(server_env, monkeypatch):
    probes = iter([False, True])
    monkeypatch.setattr(L, "_port_open", lambda *a, **k: next(probes))
    monkeypatch.setattr(L.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(alive_polls=99))
    L.run_server("web", L.DEFAULT_SETTINGS, port=7860,
                 url="http://localhost:7860", auto_opens_browser=True)
    assert server_env["opened"] == []          # gradio に任せる


def test_run_server_reports_early_death_without_opening_browser(server_env, monkeypatch):
    monkeypatch.setattr(L, "_port_open", lambda *a, **k: False)
    monkeypatch.setattr(L.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(alive_polls=1, returncode=3))
    rc = L.run_server("api", L.DEFAULT_SETTINGS, port=8000,
                      url="http://localhost:8000/docs")
    assert rc == 3 and server_env["opened"] == []


def test_run_server_refuses_to_spawn_on_busy_port(server_env, monkeypatch):
    monkeypatch.setattr(L, "_port_open", lambda *a, **k: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    def boom(*a, **k):
        raise AssertionError("must not spawn when the port is busy")
    monkeypatch.setattr(L.subprocess, "Popen", boom)
    assert L.run_server("api", L.DEFAULT_SETTINGS, port=8000,
                        url="http://localhost:8000/docs") is None


# ------------------------------------------------------------------ 入力処理

@pytest.mark.parametrize("typed,expected", [
    ("2", "2"),
    ("2)", "2"),        # 実際にあった: メニュー表記どおり「2)」と打った
    ("２）", "2"),      # 全角
    (" 2. ", "2"),
    ("２", "2"),
    ("0", "0"),
    ("q", "q"),
    ("ｑ", "q"),
    ("t", "t"),
])
def test_choice_normalizes_menu_input(monkeypatch, typed, expected):
    monkeypatch.setattr("builtins.input", lambda prompt="": typed)
    assert L._choice("選択> ") == expected


def test_yes_accepts_fullwidth_y(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "ｙ")
    assert L._yes("続行?") is True


def test_ask_exits_instead_of_looping_on_eof(monkeypatch):
    """パイプ実行などで stdin が閉じたとき、空文字を返して無限ループしないこと。"""
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(SystemExit):
        L._ask("選択> ")


# ---------------------------------------------------------------- ドリフト検出

def test_model_constants_match_fugu_local():
    """launcher は fugu_local を import しないので、定数のずれをここで検出する。"""
    import fugu_local as f
    assert L.FALLBACK_MODEL == f.FALLBACK_MODEL
    assert set(L.COUNCIL_MODELS) == set(f.DESIRED_PROPOSERS) | {f.DESIRED_AGGREGATOR}


#: 対応機能がまだこのブランチに無いフラグ(その実装ファイルが来たら検証対象になる)
PENDING_FLAGS = {"FUGU_PROFILE": "fugu_profile.py"}


def _repo_sources():
    import pathlib
    root = pathlib.Path(L.REPO)
    skip = {"tests", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    return [p for p in root.rglob("*.py")
            if not skip & set(p.relative_to(root).parts)]


def test_flag_names_are_actually_read_by_the_code():
    """設定画面に並ぶフラグが、実際にどこかで読まれているものであること。"""
    blob = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in _repo_sources() if p.name != "fugu_launcher.py")
    for name, _d, _c in L.FLAGS:
        owner = PENDING_FLAGS.get(name)
        if owner and not L.os.path.exists(L.os.path.join(L.REPO, owner)):
            continue
        assert f'"{name}"' in blob, f"{name} はどのモジュールからも参照されていない"
