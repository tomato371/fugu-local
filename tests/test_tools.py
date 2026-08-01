"""fugu_tools のオフラインテスト(FakeChat + フェイクツールのみ)。"""
import json

from fugu_llm import FakeChat
from fugu_tools import (
    ToolRegistry,
    ToolSpec,
    build_default_registry,
    build_toolcalls_schema,
    decide_tool_calls,
    execute_tool_calls,
    gather_tool_context,
    render_results,
)


def _echo_spec(name="echo", required=("text",)):
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        schema={"type": "object",
                "properties": {key: {"type": "string"} for key in required},
                "required": list(required)},
        handler=lambda args: f"{name}: {args}")


def _registry(*specs):
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


def _calls_json(*items):
    return json.dumps({"tool_calls": list(items)})


# ------------------------------------------------------------------ registry

def test_register_get_list():
    registry = _registry(_echo_spec("a"), _echo_spec("b"))
    assert registry.get("a").name == "a"
    assert registry.get("missing") is None
    assert [s.name for s in registry.list_specs()] == ["a", "b"]


def test_duplicate_registration_raises():
    registry = _registry(_echo_spec("a"))
    import pytest
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_echo_spec("a"))


def test_render_catalog_lists_names_and_args():
    catalog = _registry(_echo_spec("search", required=("query",))).render_catalog()
    assert "- search(query): search tool" in catalog


def test_toolcalls_schema_constrains_names():
    schema = build_toolcalls_schema(_registry(_echo_spec("a"), _echo_spec("b")))
    names = schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"]
    assert names == ["a", "b"]


# ------------------------------------------------------------------ decide

def test_decide_parses_valid_calls():
    registry = _registry(_echo_spec("search", required=("query",)))
    chat = FakeChat(responses=[_calls_json({"name": "search",
                                            "args": {"query": "flood"}})])
    calls = decide_tool_calls("q", registry, chat)
    assert len(calls) == 1
    assert calls[0][0].name == "search" and calls[0][1] == {"query": "flood"}
    assert chat.calls[0]["fmt"] is not None  # スキーマ制約付き


def test_decide_drops_unknown_and_invalid():
    registry = _registry(_echo_spec("search", required=("query",)))
    chat = FakeChat(responses=[_calls_json(
        {"name": "rm_rf", "args": {"query": "x"}},      # 未知ツール
        {"name": "search", "args": "not a dict"},        # 引数が非dict
        {"name": "search", "args": {}},                  # required 欠落
        "junk",                                          # 非dict要素
        {"name": "search", "args": {"query": "ok"}},     # 唯一の正当な呼び出し
    )])
    calls = decide_tool_calls("q", registry, chat)
    assert [(s.name, a) for s, a in calls] == [("search", {"query": "ok"})]


def test_decide_caps_max_calls():
    registry = _registry(_echo_spec("t", required=()))
    chat = FakeChat(responses=[_calls_json(
        *[{"name": "t", "args": {}} for _ in range(6)])])
    assert len(decide_tool_calls("q", registry, chat, max_calls=2)) == 2


def test_decide_empty_and_failures_return_no_calls():
    registry = _registry(_echo_spec("t"))
    assert decide_tool_calls("q", registry, FakeChat(responses=[_calls_json()])) == []
    assert decide_tool_calls("q", registry, FakeChat(default="not json")) == []
    boom = FakeChat(fn=lambda p: (_ for _ in ()).throw(RuntimeError()))
    assert decide_tool_calls("q", registry, boom) == []


def test_decide_empty_registry_skips_llm():
    chat = FakeChat(fn=lambda p: (_ for _ in ()).throw(AssertionError("呼ばれない")))
    assert decide_tool_calls("q", ToolRegistry(), chat) == []


# ------------------------------------------------------------------ execute / render

def test_execute_runs_in_order_and_isolates_errors():
    ok_spec = _echo_spec("ok", required=())
    bad = ToolSpec(name="bad", description="d", schema={"required": []},
                   handler=lambda args: (_ for _ in ()).throw(RuntimeError("down")))
    results = execute_tool_calls([(ok_spec, {}), (bad, {}), (ok_spec, {})])
    assert [name for name, _ in results] == ["ok", "bad", "ok"]
    assert "tool error: down" in results[1][1]
    assert results[2][1].startswith("ok:")  # 後続は実行され続ける


def test_execute_truncates_output():
    big = ToolSpec(name="big", description="d", schema={"required": []},
                   handler=lambda args: "x" * 99999)
    results = execute_tool_calls([(big, {})], max_output_chars=100)
    assert len(results[0][1]) == 100


def test_render_results_format_and_empty():
    text = render_results([("web_search", "result body")])
    assert text.startswith("## ツール実行結果")
    assert "### web_search\nresult body" in text
    assert render_results([]) == ""


# ------------------------------------------------------------------ gather

def test_gather_tool_context_end_to_end():
    registry = _registry(_echo_spec("search", required=("query",)))
    chat = FakeChat(responses=[_calls_json({"name": "search",
                                            "args": {"query": "flood"}})])
    ctx = gather_tool_context("q", chat, registry=registry)
    assert "### search" in ctx and "flood" in ctx


def test_gather_returns_empty_when_no_tools_needed():
    registry = _registry(_echo_spec("search", required=("query",)))
    assert gather_tool_context("q", FakeChat(responses=[_calls_json()]),
                               registry=registry) == ""


# ------------------------------------------------------------------ default registry

def test_default_registry_has_expected_tools():
    names = {s.name for s in build_default_registry().list_specs()}
    assert names == {"web_search", "rag_search", "run_python", "fetch_page"}


def test_default_web_search_delegates_to_fugu_local(monkeypatch):
    import fugu_local
    monkeypatch.setattr(fugu_local, "web_search",
                        lambda query, max_results=None: f"results for {query}")
    spec = build_default_registry().get("web_search")
    assert spec.handler({"query": "rain"}) == "results for rain"


def test_default_run_python_formats_ok_and_output(monkeypatch):
    import fugu_local
    monkeypatch.setattr(fugu_local, "run_python",
                        lambda code, timeout=None, stdout_only=False: (True, "42"))
    spec = build_default_registry().get("run_python")
    assert spec.handler({"code": "print(6*7)"}) == "ok=True\n42"


# ------------------------------------------------------------------ fugu_local hook

def test_tool_context_hook_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_TOOL_CALLING", raising=False)
    assert fugu_local._tool_context("q") == ""


def test_tool_context_hook_enabled(monkeypatch):
    import fugu_local
    import fugu_tools
    monkeypatch.setenv("FUGU_TOOL_CALLING", "1")
    monkeypatch.setattr(fugu_tools, "gather_tool_context",
                        lambda question, chat, registry=None, max_calls=3:
                        f"## ツール実行結果\n### fake\nctx for {question}")
    out = fugu_local._tool_context("my question")
    assert "ctx for my question" in out


def test_tool_context_hook_never_raises(monkeypatch):
    import fugu_local
    import fugu_tools
    monkeypatch.setenv("FUGU_TOOL_CALLING", "1")
    monkeypatch.setattr(fugu_tools, "gather_tool_context",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert fugu_local._tool_context("q") == ""
