# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""初心者向け UI 改修のオフラインテスト。

CLI(repl の help)・TUI(ヘルプ/モデル行)・Web(例文/ガイド/UI 構築)を、
モデル・GPU・ネット無しで検証する。gradio / rich が無い環境(CI)では
該当部分だけ skip する。
"""
import builtins

import pytest

import fugu_local as f


# ------------------------------------------------------------- CLI (repl)

def _run_repl(monkeypatch, capsys, inputs):
    """repl() をスクリプト入力で駆動し、標準出力を返す。"""
    seq = iter(inputs)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(seq))
    called = []
    monkeypatch.setattr(f, "ask_fugu", lambda *a, **k: called.append(a) or "x")
    f.repl()
    return capsys.readouterr().out, called


def test_repl_shows_guide_on_start(monkeypatch, capsys):
    out, called = _run_repl(monkeypatch, capsys, ["exit"])
    assert "使い方" in out and "例)" in out
    assert "数分〜十数分" in out           # 待ち時間の予告
    assert called == []


@pytest.mark.parametrize("word", ["help", "HELP", "?", "？", "ヘルプ"])
def test_repl_help_is_not_sent_to_the_model(monkeypatch, capsys, word):
    """初心者が help と打っても、質問としてモデルに送られないこと。"""
    out, called = _run_repl(monkeypatch, capsys, [word, "exit"])
    assert called == []                    # ask_fugu が呼ばれていない
    assert out.count("使い方") >= 2        # 起動時 + help 表示


def test_repl_normal_question_still_dispatches(monkeypatch, capsys):
    out, called = _run_repl(monkeypatch, capsys, ["91は素数?", "exit"])
    assert len(called) == 1 and called[0][0] == "91は素数?"


# ------------------------------------------------------------------- TUI

def test_tui_help_and_models_line():
    rich = pytest.importorskip("rich")  # noqa: F841
    import fugu_tui
    assert "例)" in fugu_tui._HELP and "数分〜十数分" in fugu_tui._HELP
    assert "/help" in fugu_tui._HELP
    line = fugu_tui._models_line()     # setup 前でも例外を出さない
    assert isinstance(line, str) and "Conductor" in line


# ------------------------------------------------------------------- Web

def test_web_examples_and_guide():
    pytest.importorskip("gradio")
    import fugu_web
    assert len(fugu_web.EXAMPLE_QUESTIONS) >= 3
    assert any("素数" in q for q in fugu_web.EXAMPLE_QUESTIONS)
    assert "数分〜十数分" in fugu_web._GUIDE_MD
    md = fugu_web._models_md()         # setup 前でも例外を出さない
    assert isinstance(md, str) and "Conductor" in md


def test_web_ui_builds_offline():
    """UI の構築(レイアウト定義)がモデル無しで通ること。"""
    pytest.importorskip("gradio")
    import fugu_web
    demo = fugu_web.build_ui()
    assert demo is not None


# --------------------------------------------------------------- ランチャー

def test_launcher_menu_recommends_web_ui_for_beginners(monkeypatch, capsys):
    import fugu_launcher as L
    inputs = iter(["0"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    L.main_menu(L.DEFAULT_SETTINGS | {"flags": dict(L.DEFAULT_SETTINGS["flags"])},
                {"ollama_ok": True, "missing_required": []})
    out = capsys.readouterr().out
    assert "はじめての方" in out and "おすすめ" in out
