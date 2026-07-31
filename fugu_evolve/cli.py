"""fugu_evolve.cli — 自己改善オーケストレーター (Doc C Phase 5)。

フロー: profiler(健全性)→ planner(提案)→ workspace(隔離ブランチ)→
実装(LLM によるファイル編集)→ evaluator(検証+自己修復)→ Critic 承認
(diff+証拠を JSON 判定)→ merge → `docs/evolution_history.md` 追記。

:func:`build_pipeline` は全依存(chat/sandbox/workspace_factory/各段の関数)を
dict 注入で受けて実行 callable を返す — テストは全段フェイクで配線だけを検証
できる。安全既定: 検証失敗・Critic 否認・実装不能は必ず rollback(C3 の
接頭辞ガード内)。`--pr-mode` は merge せずブランチを人間レビューに残す。

計画では「トップレベル fugu_evolve.py」だったが fugu_evolve/ パッケージと
モジュール名が衝突するため、エントリポイントは ``python -m fugu_evolve``。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from fugu_sandbox import extract_code_block
from fugu_evolve.evaluator import VERIFIED, Verification, verify
from fugu_evolve.planner import Proposal, propose
from fugu_evolve.profiler import build_health_report
from fugu_evolve.workspace import GitError, Workspace

HISTORY_REL_PATH = os.path.join("docs", "evolution_history.md")

IMPLEMENT_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "code": {"type": "string"},
                },
                "required": ["path", "code"],
            },
        }
    },
    "required": ["edits"],
}

_IMPLEMENT_SYSTEM = (
    "You are an autonomous software engineer implementing an approved "
    "improvement proposal on an isolated branch. Return the COMPLETE new "
    "content of each file you change as JSON "
    '{"edits": [{"path": "relative/path.py", "code": "..."}]}. '
    "Change as few files as possible and keep all public interfaces working. "
    "Reply with JSON only."
)

CRITIC_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["approve"],
}

_CRITIC_SYSTEM = (
    "You are a strict code reviewer gating an autonomous self-improvement "
    "loop. Approve ONLY if the diff plausibly implements the proposal, is "
    "minimal, and the verification evidence shows all tests passing without "
    "bench regression. When in doubt, reject. Reply with JSON only."
)


@dataclass
class ProposalOutcome:
    """提案1件の顛末。merged=True は main へ取り込み済みを意味する。"""

    proposal: Proposal
    branch: Optional[str] = None
    verification: Optional[Verification] = None
    approved: bool = False
    merged: bool = False
    reason: str = ""


@dataclass
class RunResult:
    health: Dict[str, object]
    outcomes: List[ProposalOutcome] = field(default_factory=list)
    dry_run: bool = False


def _sanitize_edit(path: object, code: object) -> Optional[Tuple[str, str]]:
    """LLM の編集1件を検証: 非空 path/code、fenced 救済、.py は compile ガード。"""
    if not (isinstance(path, str) and path.strip() and isinstance(code, str)):
        return None
    if "```" in code:
        extracted = extract_code_block(code)
        if extracted is not None:
            code = extracted
    if not code.strip():
        return None
    if path.strip().endswith(".py"):
        try:
            compile(code, path, "exec")
        except SyntaxError:
            return None
    return path.strip(), code


def implement_proposal(chat, workspace, proposal: Proposal,
                       max_edits: int = 3, context_chars: int = 3000) -> bool:
    """提案を LLM にファイル編集として実装させ、workspace に適用する。

    対象ファイルの現内容(先頭 ``context_chars`` 文字)をプロンプトに入れる。
    1件も有効な編集を適用できなければ False(呼び出し側が rollback する)。
    """
    sections = []
    for rel in proposal.target_files[:max_edits]:
        target = os.path.join(workspace.repo, rel)
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()[:context_chars]
        except OSError:
            text = "(file does not exist yet)"
        sections.append(f"--- {rel} ---\n{text}")
    prompt = (
        f"Proposal: {proposal.title}\n"
        f"Category: {proposal.category}\n"
        f"Rationale: {proposal.rationale}\n"
        f"Steps: {'; '.join(proposal.steps) or '(none)'}\n\n"
        f"Current file contents:\n" + "\n\n".join(sections) + "\n\n"
        'Return {"edits": [{"path", "code"}]} implementing the proposal.'
    )
    try:
        raw = chat.complete(prompt, system=_IMPLEMENT_SYSTEM,
                            fmt=IMPLEMENT_SCHEMA, temperature=0.2)
        obj = json.loads(raw)
        items = obj.get("edits", []) if isinstance(obj, dict) else []
    except Exception:
        return False
    applied = 0
    for item in items[:max_edits]:
        if not isinstance(item, dict):
            continue
        edit = _sanitize_edit(item.get("path"), item.get("code"))
        if edit is None:
            continue
        try:
            workspace.apply_edit(edit[0], edit[1])
        except Exception:
            continue
        applied += 1
    return applied > 0


def critic_approve(chat, proposal: Proposal, diff: str,
                   verification: Verification) -> Tuple[bool, str]:
    """diff と検証証拠を Critic に JSON 判定させる。失敗は安全側=否認。"""
    prompt = (
        f"Proposal: {proposal.title} ({proposal.category})\n"
        f"Rationale: {proposal.rationale}\n\n"
        f"Verification: verdict={verification.verdict}, "
        f"attempts={verification.attempts}, "
        f"notes={'; '.join(verification.notes) or '(none)'}\n\n"
        f"Diff:\n```\n{diff[:4000]}\n```\n\n"
        'Return {"approve": true|false, "reason": ...}.'
    )
    try:
        raw = chat.complete(prompt, system=_CRITIC_SYSTEM, fmt=CRITIC_SCHEMA,
                            temperature=0.0)
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("approve"), bool):
            return obj["approve"], str(obj.get("reason") or "")
    except Exception:
        pass
    return False, "critic unavailable — rejecting by safe default"


def append_history(repo: str, proposal: Proposal, verification: Verification,
                   stamp: str, nightly: bool = False) -> str:
    """`docs/evolution_history.md` に採用実績を追記する(無ければ見出し付きで作成)。"""
    path = os.path.join(repo, HISTORY_REL_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    entry = (
        f"## {stamp} — {proposal.title}\n\n"
        f"- category: {proposal.category}\n"
        f"- files: {', '.join(proposal.target_files)}\n"
        f"- verdict: {verification.verdict} (attempts={verification.attempts})\n"
        f"- mode: {'nightly' if nightly else 'manual'}\n"
        f"- notes: {'; '.join(verification.notes) or '(none)'}\n\n"
        f"{proposal.rationale}\n\n"
    )
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        if is_new:
            fh.write("# Evolution History\n\n自己改善ループが main へ取り込んだ変更の記録。\n\n")
        fh.write(entry)
    return path


def build_pipeline(deps: Dict[str, object]) -> Callable[..., RunResult]:
    """全依存注入でオーケストレーターを組み立てる(C5 の合成点)。

    deps: ``chat``(必須)/ ``sandbox`` / ``workspace_factory`` / ``health_fn`` /
    ``propose_fn`` / ``implement_fn`` / ``verify_fn`` / ``critic_fn`` /
    ``history_fn`` / ``now_fn`` / ``context_fn``。省略時は本物が使われる。
    """
    chat = deps["chat"]
    sandbox = deps.get("sandbox")
    workspace_factory = deps.get("workspace_factory", Workspace)
    health_fn = deps.get("health_fn", build_health_report)
    propose_fn = deps.get("propose_fn", propose)
    implement_fn = deps.get("implement_fn", implement_proposal)
    verify_fn = deps.get("verify_fn", verify)
    critic_fn = deps.get("critic_fn", critic_approve)
    history_fn = deps.get("history_fn", append_history)
    now_fn = deps.get("now_fn", lambda: time.strftime("%Y-%m-%d %H:%M"))
    context_fn = deps.get("context_fn")

    def run(repo: str, dry_run: bool = False, pr_mode: bool = False,
            nightly: bool = False, max_proposals: int = 3,
            bench_argv: Optional[List[str]] = None,
            offline: Optional[bool] = None) -> RunResult:
        health = health_fn(repo, sandbox=sandbox, bench_argv=bench_argv,
                           offline=offline)
        proposals = propose_fn(health, chat, context_fn=context_fn,
                               max_proposals=max_proposals)
        outcomes: List[ProposalOutcome] = []
        for proposal in proposals:
            if dry_run:
                outcomes.append(ProposalOutcome(
                    proposal=proposal, reason="dry-run: proposal only"))
                continue
            workspace = workspace_factory(repo)
            try:
                branch = workspace.create_branch(proposal.title)
            except Exception as exc:
                outcomes.append(ProposalOutcome(
                    proposal=proposal, reason=f"branch failed: {exc}"))
                continue
            if not implement_fn(chat, workspace, proposal):
                workspace.rollback()
                outcomes.append(ProposalOutcome(
                    proposal=proposal, branch=branch,
                    reason="no valid implementation from model"))
                continue
            workspace.commit(f"auto-evolve: {proposal.title}")
            verification = verify_fn(workspace, sandbox, chat, health,
                                     bench_argv=bench_argv, offline=offline)
            if verification.verdict != VERIFIED:
                workspace.rollback()
                outcomes.append(ProposalOutcome(
                    proposal=proposal, branch=branch, verification=verification,
                    reason="verification failed"))
                continue
            try:
                workspace.ensure_clean()
            except GitError:
                # evaluator の自己修復が未コミット編集を残している場合
                workspace.commit("auto-evolve: self-repair during verification")
            diff = workspace.diff()
            approved, reason = critic_fn(chat, proposal, diff, verification)
            if not approved:
                workspace.rollback()
                outcomes.append(ProposalOutcome(
                    proposal=proposal, branch=branch, verification=verification,
                    reason=f"critic rejected: {reason}"))
                continue
            if pr_mode:
                outcomes.append(ProposalOutcome(
                    proposal=proposal, branch=branch, verification=verification,
                    approved=True, reason="pr-mode: branch left for human review"))
                continue
            workspace.merge_to_main()
            history_fn(repo, proposal, verification, now_fn(), nightly=nightly)
            outcomes.append(ProposalOutcome(
                proposal=proposal, branch=branch, verification=verification,
                approved=True, merged=True, reason=reason or "approved"))
        return RunResult(health=health, outcomes=outcomes, dry_run=dry_run)

    return run


def format_result(result: RunResult) -> str:
    """実行結果の人間可読サマリ。"""
    pytest_report = result.health.get("pytest", {}) if result.health else {}
    lines = [
        f"health   : pytest passed={pytest_report.get('passed', '?')} "
        f"failed={pytest_report.get('failed', '?')}",
        f"mode     : {'dry-run' if result.dry_run else 'apply'}",
        f"proposals: {len(result.outcomes)}",
    ]
    for outcome in result.outcomes:
        status = ("merged" if outcome.merged
                  else "approved" if outcome.approved else "skipped")
        lines.append(f"  - [{status}] {outcome.proposal.title} "
                     f"({outcome.proposal.category}) — {outcome.reason}")
    return "\n".join(lines)


def run_prompt_evolution(name: str, chat, repo: str,
                         apply: bool = True, n: int = 3) -> Dict[str, object]:
    """--prompts モード本体: fugu_local のプロンプト定数 ``name`` を進化させる。

    採用時は Workspace 経由で ``auto-evolve/prompts-{name}`` ブランチにコミット
    される(mainは不変)。``apply=False`` は判定のみで一切書かない。
    """
    import fugu_local
    from fugu_evolve import prompt_evolver

    base = getattr(fugu_local, name, None)
    if not (isinstance(base, str) and base.strip()):
        return {"name": name, "adopted": False, "branch": None,
                "reason": f"unknown or non-string prompt global: {name}"}
    workspace = Workspace(repo) if apply else None
    eval_fn = prompt_evolver.make_llm_eval_fn(chat)
    return prompt_evolver.evolve_prompt(name, base, chat, eval_fn,
                                        workspace=workspace, n=n, apply=apply)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fugu_evolve",
        description="自律自己改善オーケストレーター (profiler→planner→workspace→"
                    "evaluator→critic→merge)")
    parser.add_argument("--repo", default=".", help="対象リポジトリ(既定: カレント)")
    parser.add_argument("--dry-run", action="store_true",
                        help="提案の生成まで(ブランチ作成・編集を一切しない)")
    parser.add_argument("--pr-mode", action="store_true",
                        help="merge せず auto-evolve ブランチを人間レビューに残す")
    parser.add_argument("--nightly", action="store_true",
                        help="無人定期実行マーク(evolution_history に記録される)")
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--bench", nargs="+", default=None, metavar="CMD",
                        help="非退行比較に使うベンチコマンド(省略時は比較 skip)")
    parser.add_argument("--offline", action="store_true",
                        help="ベンチ実行を強制 skip(GPU 競合中の安全弁)")
    parser.add_argument("--prompts", metavar="NAME", default=None,
                        help="プロンプト進化モード: fugu_local の対象グローバル名"
                             "(例 PRESENTATION_STYLE)。要 Ollama")
    args = parser.parse_args(argv)

    import fugu_llm
    if args.prompts:
        result = run_prompt_evolution(
            args.prompts, fugu_llm.AskChat(label="evolve-prompts"),
            args.repo, apply=not args.dry_run)
        print(json.dumps(
            {k: v for k, v in result.items() if k != "winner"},
            ensure_ascii=False, indent=1))
        return 0
    pipeline = build_pipeline({"chat": fugu_llm.AskChat(label="evolve")})
    result = pipeline(
        args.repo, dry_run=args.dry_run, pr_mode=args.pr_mode,
        nightly=args.nightly, max_proposals=args.max_proposals,
        bench_argv=args.bench, offline=True if args.offline else None,
    )
    print(format_result(result))
    return 0
