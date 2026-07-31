"""fugu_evolve.planner のオフラインテスト(FakeChat のみ)。"""
import json

from fugu_llm import FakeChat
from fugu_evolve.planner import (
    PROPOSAL_SCHEMA,
    Proposal,
    propose,
    validate_proposal,
)

HEALTHY = {"pytest": {"ran": True, "passed": 10, "failed": 0, "errors": 0},
           "bench": {"ran": False, "reason": "offline"},
           "modules": [{"path": "a.py"}]}

BROKEN = {"pytest": {"ran": True, "passed": 8, "failed": 2, "errors": 0,
                     "tail": "FAILED tests/test_a.py::test_x"},
          "bench": {"ran": False, "reason": "offline"},
          "modules": [{"path": "a.py"}]}


def _plan_json(*items):
    return json.dumps({"proposals": list(items)})


def _item(title="t", category="refactor", files=("a.py",), rationale="r", **kw):
    d = {"title": title, "category": category,
         "target_files": list(files), "rationale": rationale}
    d.update(kw)
    return d


def _boom(prompt):
    raise RuntimeError("model down")


# ------------------------------------------------------------------ schema

def test_schema_requires_core_fields():
    item = PROPOSAL_SCHEMA["properties"]["proposals"]["items"]
    assert set(item["required"]) == {"title", "category", "target_files", "rationale"}
    assert "fix" in item["properties"]["category"]["enum"]


# ------------------------------------------------------------------ validate

def test_validate_accepts_well_formed():
    p = validate_proposal(_item(steps=["s1", "", 42]))
    assert isinstance(p, Proposal)
    assert p.steps == ["s1"]  # 空文字と非文字列は落とす


def test_validate_rejects_bad_shapes():
    assert validate_proposal("not a dict") is None
    assert validate_proposal(_item(title="  ")) is None
    assert validate_proposal(_item(category="world-domination")) is None
    assert validate_proposal(_item(rationale="")) is None
    assert validate_proposal(_item(files=())) is None
    assert validate_proposal({"title": "t", "category": "fix",
                              "target_files": "a.py", "rationale": "r"}) is None


def test_validate_filters_to_repo_files():
    p = validate_proposal(_item(files=("a.py", "ghost.py")), repo_files=["a.py"])
    assert p.target_files == ["a.py"]
    assert validate_proposal(_item(files=("ghost.py",)), repo_files=["a.py"]) is None


def test_validate_normalizes_windows_separators():
    p = validate_proposal(_item(files=("pkg\\mod.py",)), repo_files=["pkg/mod.py"])
    assert p.target_files == ["pkg/mod.py"]


# ------------------------------------------------------------------ propose

def test_propose_parses_and_sorts_by_priority():
    chat = FakeChat(responses=[_plan_json(
        _item(title="docs", category="docs"),
        _item(title="speed", category="perf"),
        _item(title="bugfix", category="fix"),
    )])
    got = propose(HEALTHY, chat)
    assert [p.title for p in got] == ["bugfix", "speed", "docs"]


def test_propose_stable_within_category():
    chat = FakeChat(responses=[_plan_json(
        _item(title="r1"), _item(title="r2"), _item(title="r3"))])
    assert [p.title for p in propose(HEALTHY, chat)] == ["r1", "r2", "r3"]


def test_propose_caps_count():
    chat = FakeChat(responses=[_plan_json(*[_item(title=f"p{i}") for i in range(6)])])
    assert len(propose(HEALTHY, chat, max_proposals=2)) == 2


def test_propose_drops_invalid_items():
    chat = FakeChat(responses=[_plan_json(
        _item(title="good"), {"title": "bad"}, "junk")])
    got = propose(HEALTHY, chat)
    assert [p.title for p in got] == ["good"]


def test_propose_includes_report_and_schema_in_call():
    chat = FakeChat(responses=[_plan_json(_item())])
    propose(HEALTHY, chat)
    call = chat.calls[0]
    assert '"passed": 10' in call["prompt"]
    assert call["fmt"] is PROPOSAL_SCHEMA


def test_propose_injects_context_fn():
    seen = []

    def context_fn(report):
        seen.append(report)
        return "call graph: a.py -> b.py"

    chat = FakeChat(responses=[_plan_json(_item())])
    propose(HEALTHY, chat, context_fn=context_fn)
    assert seen == [HEALTHY]
    assert "call graph: a.py -> b.py" in chat.calls[0]["prompt"]


def test_propose_survives_broken_context_fn():
    chat = FakeChat(responses=[_plan_json(_item(title="ok"))])
    got = propose(HEALTHY, chat, context_fn=lambda r: (_ for _ in ()).throw(RuntimeError()))
    assert [p.title for p in got] == ["ok"]


# ------------------------------------------------------------------ fallback

def test_propose_falls_back_to_fix_when_tests_fail():
    got = propose(BROKEN, FakeChat(fn=_boom))
    assert len(got) == 1
    assert got[0].category == "fix"
    assert "2" in got[0].title
    assert "FAILED tests/test_a.py" in got[0].rationale


def test_propose_junk_reply_with_failures_still_yields_fix():
    got = propose(BROKEN, FakeChat(default="not json"))
    assert got and got[0].category == "fix"


def test_propose_healthy_repo_with_broken_model_returns_empty():
    assert propose(HEALTHY, FakeChat(fn=_boom)) == []


def test_propose_repo_files_filter_applies():
    chat = FakeChat(responses=[_plan_json(
        _item(title="real", files=("a.py",)),
        _item(title="ghost", files=("nope.py",)))])
    got = propose(HEALTHY, chat, repo_files=["a.py"])
    assert [p.title for p in got] == ["real"]
