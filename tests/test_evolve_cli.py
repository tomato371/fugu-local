"""fugu_evolve.cli のオフラインテスト(全依存フェイク注入で配線を検証)。"""
import json

from fugu_llm import FakeChat
from fugu_evolve import cli
from fugu_evolve.cli import (
    ProposalOutcome,
    RunResult,
    append_history,
    build_pipeline,
    critic_approve,
    format_result,
    implement_proposal,
)
from fugu_evolve.evaluator import FAILED, VERIFIED, Verification
from fugu_evolve.planner import Proposal
from fugu_evolve.profiler import PytestReport
from fugu_evolve.workspace import BRANCH_PREFIX, GitError, Workspace

from tests.test_evolve_workspace import FakeGit

HEALTH = {"pytest": {"ran": True, "passed": 5, "failed": 0, "errors": 0},
          "bench": {"ran": False, "reason": "offline"}, "modules": []}

PROPOSAL = Proposal(title="tidy up", category="refactor",
                    target_files=["mod.py"], rationale="cleaner")


def _verification(verdict=VERIFIED, attempts=1):
    return Verification(verdict=verdict, attempts=attempts,
                        pytest=PytestReport(ran=True, passed=5))


class FakeWorkspace:
    """呼び出し履歴だけを記録する Workspace フェイク。"""

    def __init__(self, repo, dirty_after_verify=False, branch_fails=False):
        self.repo = repo
        self.calls = []
        self.dirty_after_verify = dirty_after_verify
        self.branch_fails = branch_fails

    def create_branch(self, title):
        if self.branch_fails:
            raise GitError("working tree is not clean")
        self.calls.append("create_branch")
        return BRANCH_PREFIX + "fake-1"

    def apply_edit(self, rel_path, content):
        self.calls.append(f"apply_edit:{rel_path}")
        return rel_path

    def commit(self, message):
        self.calls.append(f"commit:{message}")
        return "sha"

    def ensure_clean(self):
        self.calls.append("ensure_clean")
        if self.dirty_after_verify:
            self.dirty_after_verify = False
            raise GitError("dirty")

    def diff(self, base=None):
        self.calls.append("diff")
        return "+ change"

    def rollback(self):
        self.calls.append("rollback")

    def merge_to_main(self, delete_branch=True):
        self.calls.append("merge")
        return "main"


def _deps(workspace, **over):
    deps = {
        "chat": FakeChat(default="unused"),
        "workspace_factory": lambda repo: workspace,
        "health_fn": lambda repo, **kw: HEALTH,
        "propose_fn": lambda health, chat, **kw: [PROPOSAL],
        "implement_fn": lambda chat, ws, p: True,
        "verify_fn": lambda ws, sb, chat, base, **kw: _verification(),
        "critic_fn": lambda chat, p, diff, v: (True, "looks good"),
        "history_fn": lambda repo, p, v, stamp, nightly=False: None,
        "now_fn": lambda: "2026-08-01 12:00",
    }
    deps.update(over)
    return deps


# ------------------------------------------------------------------ pipeline

def test_dry_run_reports_without_touching_workspace(tmp_path):
    ws = FakeWorkspace(str(tmp_path))
    run = build_pipeline(_deps(ws))
    result = run(str(tmp_path), dry_run=True)
    assert result.dry_run is True
    assert [o.reason for o in result.outcomes] == ["dry-run: proposal only"]
    assert ws.calls == []


def test_happy_path_merges_and_records_history(tmp_path):
    history = []
    ws = FakeWorkspace(str(tmp_path))
    run = build_pipeline(_deps(
        ws, history_fn=lambda repo, p, v, stamp, nightly=False:
        history.append((p.title, stamp, nightly))))
    result = run(str(tmp_path), nightly=True)
    o = result.outcomes[0]
    assert o.merged is True and o.approved is True
    assert "merge" in ws.calls and "rollback" not in ws.calls
    assert history == [("tidy up", "2026-08-01 12:00", True)]


def test_verification_failure_rolls_back(tmp_path):
    ws = FakeWorkspace(str(tmp_path))
    run = build_pipeline(_deps(
        ws, verify_fn=lambda *a, **kw: _verification(verdict=FAILED)))
    o = run(str(tmp_path)).outcomes[0]
    assert o.merged is False and o.reason == "verification failed"
    assert "rollback" in ws.calls and "merge" not in ws.calls


def test_critic_rejection_rolls_back(tmp_path):
    ws = FakeWorkspace(str(tmp_path))
    run = build_pipeline(_deps(
        ws, critic_fn=lambda chat, p, diff, v: (False, "diff too large")))
    o = run(str(tmp_path)).outcomes[0]
    assert o.merged is False
    assert "diff too large" in o.reason
    assert "rollback" in ws.calls


def test_pr_mode_keeps_branch_without_merge(tmp_path):
    ws = FakeWorkspace(str(tmp_path))
    o = build_pipeline(_deps(ws))(str(tmp_path), pr_mode=True).outcomes[0]
    assert o.approved is True and o.merged is False
    assert "pr-mode" in o.reason
    assert "merge" not in ws.calls and "rollback" not in ws.calls


def test_no_implementation_rolls_back(tmp_path):
    ws = FakeWorkspace(str(tmp_path))
    run = build_pipeline(_deps(ws, implement_fn=lambda chat, ws_, p: False))
    o = run(str(tmp_path)).outcomes[0]
    assert "no valid implementation" in o.reason
    assert "rollback" in ws.calls


def test_branch_failure_skips_proposal(tmp_path):
    ws = FakeWorkspace(str(tmp_path), branch_fails=True)
    o = build_pipeline(_deps(ws))(str(tmp_path)).outcomes[0]
    assert "branch failed" in o.reason
    assert ws.calls == []


def test_uncommitted_repairs_get_committed_before_diff(tmp_path):
    ws = FakeWorkspace(str(tmp_path), dirty_after_verify=True)
    build_pipeline(_deps(ws))(str(tmp_path))
    assert any(c.startswith("commit:auto-evolve: self-repair") for c in ws.calls)
    assert ws.calls.index("diff") > ws.calls.index("ensure_clean")


# ------------------------------------------------------------------ critic

def test_critic_approve_parses_json():
    chat = FakeChat(responses=[json.dumps({"approve": True, "reason": "ok"})])
    assert critic_approve(chat, PROPOSAL, "+x", _verification()) == (True, "ok")


def test_critic_rejects_on_junk_or_error():
    ok, reason = critic_approve(FakeChat(default="not json"), PROPOSAL, "+x",
                                _verification())
    assert ok is False and "safe default" in reason
    ok, _ = critic_approve(
        FakeChat(fn=lambda p: (_ for _ in ()).throw(RuntimeError())),
        PROPOSAL, "+x", _verification())
    assert ok is False


def test_critic_gets_diff_and_evidence():
    chat = FakeChat(responses=[json.dumps({"approve": False})])
    critic_approve(chat, PROPOSAL, "+ the diff", _verification())
    prompt = chat.calls[0]["prompt"]
    assert "+ the diff" in prompt and "verdict=VERIFIED" in prompt


# ------------------------------------------------------------------ implement

def test_implement_proposal_applies_edits(tmp_path):
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    ws = Workspace(str(tmp_path), git=FakeGit(branch=BRANCH_PREFIX + "t-1"))
    chat = FakeChat(responses=[json.dumps(
        {"edits": [{"path": "mod.py", "code": "x = 2\n"}]})])
    assert implement_proposal(chat, ws, PROPOSAL) is True
    assert (tmp_path / "mod.py").read_text(encoding="utf-8") == "x = 2\n"
    assert "x = 1" in chat.calls[0]["prompt"]  # 現内容がプロンプトに入る


def test_implement_proposal_compile_guard(tmp_path):
    ws = Workspace(str(tmp_path), git=FakeGit(branch=BRANCH_PREFIX + "t-1"))
    chat = FakeChat(responses=[json.dumps(
        {"edits": [{"path": "mod.py", "code": "def f(:\n"}]})])
    assert implement_proposal(chat, ws, PROPOSAL) is False
    assert not (tmp_path / "mod.py").exists()


def test_implement_proposal_junk_reply_is_false(tmp_path):
    ws = Workspace(str(tmp_path), git=FakeGit(branch=BRANCH_PREFIX + "t-1"))
    assert implement_proposal(FakeChat(default="junk"), ws, PROPOSAL) is False


# ------------------------------------------------------------------ history / format

def test_append_history_creates_then_appends(tmp_path):
    p1 = append_history(str(tmp_path), PROPOSAL, _verification(),
                        "2026-08-01 12:00")
    append_history(str(tmp_path), PROPOSAL, _verification(),
                   "2026-08-01 13:00", nightly=True)
    text = open(p1, encoding="utf-8").read()
    assert text.count("# Evolution History") == 1  # 見出しは初回のみ
    assert "2026-08-01 12:00 — tidy up" in text
    assert "mode: nightly" in text and "mode: manual" in text


def test_format_result_summarizes():
    result = RunResult(health=HEALTH, outcomes=[
        ProposalOutcome(proposal=PROPOSAL, approved=True, merged=True,
                        reason="approved")])
    text = format_result(result)
    assert "passed=5" in text
    assert "[merged] tidy up (refactor)" in text


# ------------------------------------------------------------------ main

def test_main_wires_args_into_pipeline(monkeypatch, capsys):
    captured = {}

    def fake_build(deps):
        assert "chat" in deps

        def run(repo, **kw):
            captured["repo"] = repo
            captured.update(kw)
            return RunResult(health=HEALTH, outcomes=[], dry_run=kw["dry_run"])
        return run

    monkeypatch.setattr(cli, "build_pipeline", fake_build)
    rc = cli.main(["--repo", "R", "--dry-run", "--pr-mode",
                   "--max-proposals", "2", "--offline"])
    assert rc == 0
    assert captured["repo"] == "R"
    assert captured["dry_run"] is True and captured["pr_mode"] is True
    assert captured["max_proposals"] == 2 and captured["offline"] is True
    assert "dry-run" in capsys.readouterr().out
