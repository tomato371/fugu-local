"""fugu_prompts 設定レイヤと fugu_evolve.prompt_evolver のオフラインテスト。"""
import json

import fugu_prompts
from fugu_llm import FakeChat
from fugu_evolve import cli
from fugu_evolve.prompt_evolver import (
    evaluate_prompts,
    evolve_prompt,
    make_llm_eval_fn,
    mutate_prompts,
)
from fugu_evolve.workspace import BRANCH_PREFIX

from tests.test_evolve_cli import FakeWorkspace


def _use_tmp_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("FUGU_PROMPTS_DIR", str(tmp_path / "overrides"))


# ------------------------------------------------------------------ fugu_prompts

def test_get_prompt_defaults_without_override(monkeypatch, tmp_path):
    _use_tmp_dir(monkeypatch, tmp_path)
    assert fugu_prompts.get_prompt("MISSING", "the default") == "the default"


def test_set_get_clear_override_roundtrip(monkeypatch, tmp_path):
    _use_tmp_dir(monkeypatch, tmp_path)
    fugu_prompts.set_override("STYLE", "evolved prompt")
    assert fugu_prompts.get_prompt("STYLE", "default") == "evolved prompt"
    assert fugu_prompts.list_overrides() == {"STYLE": "evolved prompt"}
    assert fugu_prompts.clear_override("STYLE") is True
    assert fugu_prompts.get_prompt("STYLE", "default") == "default"
    assert fugu_prompts.clear_override("STYLE") is False


def test_empty_override_falls_back_to_default(monkeypatch, tmp_path):
    _use_tmp_dir(monkeypatch, tmp_path)
    fugu_prompts.set_override("STYLE", "   ")
    assert fugu_prompts.get_prompt("STYLE", "default") == "default"


def test_apply_overrides_only_touches_existing_str_keys(monkeypatch, tmp_path):
    _use_tmp_dir(monkeypatch, tmp_path)
    fugu_prompts.set_override("KNOWN", "new value")
    fugu_prompts.set_override("UNKNOWN", "should not appear")
    fugu_prompts.set_override("NOT_STR", "ignored")
    namespace = {"KNOWN": "old value", "NOT_STR": 42}
    applied = fugu_prompts.apply_overrides(namespace)
    assert applied == 1
    assert namespace["KNOWN"] == "new value"
    assert namespace["NOT_STR"] == 42
    assert "UNKNOWN" not in namespace  # タイポ override が新グローバルを作らない


# ------------------------------------------------------------------ mutate

def test_mutate_prompts_parses_variants():
    chat = FakeChat(responses=[json.dumps(
        {"variants": ["variant A", "variant A", "variant B", ""]})])
    variants = mutate_prompts("base", chat, n=3)
    assert variants == ["variant A", "variant B"]  # 重複と空は除去


def test_mutate_prompts_fallback_is_deterministic():
    variants = mutate_prompts("base prompt", FakeChat(default="junk"), n=3)
    assert len(variants) == 3
    assert all(v.startswith("base prompt") for v in variants)
    assert len(set(variants)) == 3
    boom = FakeChat(fn=lambda p: (_ for _ in ()).throw(RuntimeError()))
    assert mutate_prompts("base prompt", boom, n=2) == variants[:2]


def test_mutate_prompts_drops_base_echo():
    chat = FakeChat(responses=[json.dumps({"variants": ["base"]})])
    variants = mutate_prompts("base", chat, n=3)
    assert all(v.strip() != "base" for v in variants)  # フォールバックに落ちる


# ------------------------------------------------------------------ evaluate

def test_evaluate_prompts_sorts_desc_and_drops_failures():
    def eval_fn(candidate):
        if candidate == "bad":
            raise RuntimeError("eval down")
        return float(len(candidate))

    scored = evaluate_prompts(["aa", "bad", "aaaa"], eval_fn)
    assert scored == [("aaaa", 4.0), ("aa", 2.0)]


# ------------------------------------------------------------------ evolve

def _length_eval(candidate):
    return float(len(candidate))  # 長い方が勝つ決定論的評価


def test_evolve_adopts_winner_via_workspace(tmp_path):
    ws = FakeWorkspace(str(tmp_path))
    chat = FakeChat(responses=[json.dumps({"variants": ["base plus extra text"]})])
    result = evolve_prompt("STYLE", "base", chat, _length_eval, workspace=ws)
    assert result["adopted"] is True
    assert result["branch"] == BRANCH_PREFIX + "fake-1"
    assert "create_branch" in ws.calls
    assert any(c.startswith("commit:auto-evolve: prompt override STYLE")
               for c in ws.calls)


def test_evolve_writes_override_without_workspace(monkeypatch, tmp_path):
    _use_tmp_dir(monkeypatch, tmp_path)
    chat = FakeChat(responses=[json.dumps({"variants": ["base plus extra text"]})])
    result = evolve_prompt("STYLE", "base", chat, _length_eval)
    assert result["adopted"] is True
    assert fugu_prompts.get_prompt("STYLE", "x") == "base plus extra text"


def test_evolve_keeps_baseline_when_it_wins(monkeypatch, tmp_path):
    _use_tmp_dir(monkeypatch, tmp_path)
    chat = FakeChat(responses=[json.dumps({"variants": ["tiny"]})])
    result = evolve_prompt("STYLE", "a very long baseline prompt", chat,
                           _length_eval)
    assert result["adopted"] is False
    assert fugu_prompts.list_overrides() == {}  # 何も書かない


def test_evolve_min_gain_blocks_marginal_wins(monkeypatch, tmp_path):
    _use_tmp_dir(monkeypatch, tmp_path)
    chat = FakeChat(responses=[json.dumps({"variants": ["base!"]})])  # +1 文字
    result = evolve_prompt("STYLE", "base", chat, _length_eval, min_gain=5.0)
    assert result["adopted"] is False


def test_evolve_dry_run_never_writes(monkeypatch, tmp_path):
    _use_tmp_dir(monkeypatch, tmp_path)
    chat = FakeChat(responses=[json.dumps({"variants": ["base plus extra"]})])
    result = evolve_prompt("STYLE", "base", chat, _length_eval, apply=False)
    assert result["adopted"] is True   # 判定はする
    assert fugu_prompts.list_overrides() == {}  # 書かない


def test_evolve_no_evaluable_candidate():
    def always_fail(candidate):
        raise RuntimeError("down")
    chat = FakeChat(default="junk")
    result = evolve_prompt("STYLE", "base", chat, always_fail)
    assert result["adopted"] is False
    assert "no candidate" in result["reason"]


# ------------------------------------------------------------------ llm eval fn

def test_make_llm_eval_fn_averages_judge_scores():
    replies = ["answer 1", json.dumps({"score": 8}),
               "answer 2", json.dumps({"score": 6})]
    chat = FakeChat(responses=list(replies))
    eval_fn = make_llm_eval_fn(chat, probes=("p1", "p2"))
    assert eval_fn("candidate prompt") == 7.0
    assert chat.calls[0]["system"] == "candidate prompt"  # 候補が system に入る


def test_make_llm_eval_fn_junk_judge_counts_zero():
    chat = FakeChat(responses=["answer", "not json"])
    eval_fn = make_llm_eval_fn(chat, probes=("p1",))
    assert eval_fn("candidate") == 0.0


# ------------------------------------------------------------------ CLI --prompts

def test_run_prompt_evolution_rejects_unknown_global():
    result = cli.run_prompt_evolution("NO_SUCH_GLOBAL_XYZ", FakeChat(default="x"),
                                      ".", apply=False)
    assert result["adopted"] is False
    assert "unknown" in result["reason"]


def test_main_prompts_mode_dispatch(monkeypatch, capsys):
    seen = {}

    def fake_run(name, chat, repo, apply=True, n=3):
        seen.update({"name": name, "repo": repo, "apply": apply})
        return {"name": name, "adopted": False, "branch": None,
                "reason": "baseline still best", "winner": "W", "score": 1.0}

    monkeypatch.setattr(cli, "run_prompt_evolution", fake_run)
    rc = cli.main(["--prompts", "PRESENTATION_STYLE", "--repo", "R", "--dry-run"])
    assert rc == 0
    assert seen == {"name": "PRESENTATION_STYLE", "repo": "R", "apply": False}
    out = capsys.readouterr().out
    assert "baseline still best" in out
    assert "W" not in out  # winner 全文は出力しない(長大なため)
