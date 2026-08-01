"""fugu_tools — 汎用ツール呼び出し層 (Doc E Phase 1)。

従来のツール起動は Conductor 固定スキーマの bool フラグ+fugu_local.py 内の
ハードコード分岐であり、新ツールの追加に本体編集が必須だった。本モジュールは
MCP 的な「実行時にモデルがツール一覧を見て選ぶ」層を提供する:

- :class:`ToolSpec` / :class:`ToolRegistry` — 名前+説明+引数スキーマ+handler。
  新ツールは register() だけで追加でき、fugu_local.py の編集は不要になる。
- :func:`decide_tool_calls` — カタログを注入した LLM にスキーマ制約
  ``{"tool_calls": [{"name", "args"}]}`` で選択させる。未知ツール・不正引数は
  黙って落とし、失敗時は空リスト(ツール無しで従来どおり回答できる)。
- :func:`execute_tool_calls` — 逐次実行。1ツールの失敗は他を殺さない。
- :func:`build_default_registry` — 既存機能(web_search / rag_search /
  run_python / fetch_page)を**ラップするだけ**のアダプタ登録(既存関数は不変、
  handler 内 lazy import で循環参照を回避)。

フックは ``FUGU_TOOL_CALLING=1`` のときだけ ask_fugu がコンテキスト構築の
直後に :func:`gather_tool_context` を呼ぶ(既定経路は完全に不変)。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

#: 1回の質問で許すツール呼び出し数と、1ツール出力の注入上限(num_ctx 予算)。
DEFAULT_MAX_CALLS = 3
DEFAULT_OUTPUT_CHARS = 2000
#: 選択→実行の反復上限(ReAct 型)。env FUGU_TOOL_ROUNDS で上書き可。
DEFAULT_MAX_ROUNDS = 2


@dataclass
class ToolSpec:
    """ツール1個の定義。schema は引数の JSON スキーマ(required を検証に使う)。"""

    name: str
    description: str
    schema: Dict[str, object]
    handler: Callable[[Dict[str, object]], str]


class ToolRegistry:
    """名前→ToolSpec の登録簿。重複登録は設計ミスとして即例外。"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def render_catalog(self) -> str:
        """プロンプト注入用のツール一覧(名前・説明・引数スキーマ)。"""
        lines = []
        for spec in self._tools.values():
            props = spec.schema.get("properties", {})
            args = ", ".join(sorted(props)) if isinstance(props, dict) else ""
            lines.append(f"- {spec.name}({args}): {spec.description}")
        return "\n".join(lines)


def build_toolcalls_schema(registry: ToolRegistry) -> Dict[str, object]:
    """モデル出力を「登録済みツール名のみ」に制約する JSON スキーマ。"""
    names = [spec.name for spec in registry.list_specs()]
    return {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": names or ["none"]},
                        "args": {"type": "object"},
                    },
                    "required": ["name", "args"],
                },
            }
        },
        "required": ["tool_calls"],
    }


_SYSTEM = (
    "You are a tool router for a local answering pipeline. Read the user's "
    "question and the tool catalog, and select the tools (with concrete "
    "arguments) whose outputs would materially improve the answer. Select "
    "NOTHING when the question needs no tools — an empty list is a good "
    "answer. Reply with JSON only."
)


def decide_tool_calls(question: str, registry: ToolRegistry, chat,
                      max_calls: int = DEFAULT_MAX_CALLS,
                      prior_results: str = "",
                      ) -> List[Tuple[ToolSpec, Dict[str, object]]]:
    """質問に対して実行すべきツール呼び出し列を LLM に選ばせる。

    ``prior_results`` が与えられた場合(ReAct の2巡目以降)は前巡の実行結果を
    提示し、「まだ足りない情報がある場合のみ追加ツールを選べ」と求める。
    未知ツール名・非 dict 引数・required 欠落は黙って落とす。モデル障害・
    パース失敗は空リスト(ツール無しで従来どおり回答が続く — 選択層の失敗が
    回答自体を止めることはない)。
    """
    if not registry.list_specs():
        return []
    prior_block = (
        f"Results from tools already executed:\n{prior_results}\n\n"
        "Select ADDITIONAL tools ONLY if the results above are insufficient "
        "to answer; otherwise return an empty list.\n\n"
        if prior_results else "")
    prompt = (
        f"Question: {question}\n\n"
        f"Tool catalog:\n{registry.render_catalog()}\n\n"
        + prior_block +
        'Respond with {"tool_calls": [{"name": ..., "args": {...}}]} '
        "(empty list if no tool is needed)."
    )
    try:
        raw = chat.complete(prompt, system=_SYSTEM,
                            fmt=build_toolcalls_schema(registry),
                            temperature=0.0)
        obj = json.loads(raw)
        items = obj.get("tool_calls", []) if isinstance(obj, dict) else []
    except Exception:
        return []
    calls: List[Tuple[ToolSpec, Dict[str, object]]] = []
    for item in items:
        if len(calls) >= max(0, max_calls):
            break
        if not isinstance(item, dict):
            continue
        spec = registry.get(item.get("name")) if isinstance(item.get("name"), str) \
            else None
        args = item.get("args")
        if spec is None or not isinstance(args, dict):
            continue
        required = spec.schema.get("required", [])
        if isinstance(required, list) and not all(key in args for key in required):
            continue
        calls.append((spec, args))
    return calls


def execute_tool_calls(calls: List[Tuple[ToolSpec, Dict[str, object]]],
                       max_output_chars: int = DEFAULT_OUTPUT_CHARS,
                       ) -> List[Tuple[str, str]]:
    """ツール呼び出しを逐次実行し (name, output) を返す。

    1ツールの例外はエラーメッセージとして結果に残し、他の実行は続ける
    (モデルが「このツールは失敗した」と分かることにも価値がある)。
    """
    results: List[Tuple[str, str]] = []
    for spec, args in calls:
        try:
            output = str(spec.handler(args))
        except Exception as exc:
            output = f"(tool error: {exc})"
        results.append((spec.name, output.strip()[:max_output_chars]))
    return results


def render_results(results: List[Tuple[str, str]]) -> str:
    """実行結果をコンテキスト注入用テキストに整形("" if empty)。"""
    if not results:
        return ""
    lines = ["## ツール実行結果 (tool calling)"]
    for name, output in results:
        lines.append(f"### {name}")
        lines.append(output or "(no output)")
    return "\n".join(lines)


def _max_rounds() -> int:
    try:
        value = int(os.environ.get("FUGU_TOOL_ROUNDS") or DEFAULT_MAX_ROUNDS)
        return max(1, value)
    except ValueError:
        return DEFAULT_MAX_ROUNDS


def gather_tool_context(question: str, chat,
                        registry: Optional[ToolRegistry] = None,
                        max_calls: int = DEFAULT_MAX_CALLS,
                        max_rounds: Optional[int] = None) -> str:
    """フックの入口: 選択→実行を最大 ``max_rounds`` 回反復する(ReAct 型)。

    2巡目以降はそれまでの実行結果をルーターに提示し、不足があるときだけ追加
    ツールを選ばせる。同一(ツール名, 引数)の再実行は抑止。追加選択が空に
    なった時点で終了(単純な質問は従来どおり1巡で済む)。ツール不要なら ""。
    """
    registry = registry if registry is not None else build_default_registry()
    if max_rounds is None:
        max_rounds = _max_rounds()
    all_results: List[Tuple[str, str]] = []
    executed = set()
    for _ in range(max_rounds):
        prior = render_results(all_results) if all_results else ""
        calls = decide_tool_calls(question, registry, chat,
                                  max_calls=max_calls, prior_results=prior)
        fresh = []
        for spec, args in calls:
            key = (spec.name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if key not in executed:
                executed.add(key)
                fresh.append((spec, args))
        if not fresh:
            break
        all_results.extend(execute_tool_calls(fresh))
    return render_results(all_results)


# ------------------------------------------------------------------ 既定レジストリ

def build_default_registry() -> ToolRegistry:
    """既存機能をアダプタ登録した既定レジストリを返す。

    handler は呼び出し時に fugu_local / fugu_browser を lazy import する
    (本モジュールの import 自体は軽量に保ち、循環参照も避ける)。既存関数は
    一切変更しない — ここは純粋なラッパ層。
    """
    registry = ToolRegistry()

    def _web_search(args: Dict[str, object]) -> str:
        import fugu_local
        return fugu_local.web_search(str(args["query"]))

    def _rag_search(args: Dict[str, object]) -> str:
        import fugu_local
        return fugu_local.rag_search(str(args["question"]))

    def _run_python(args: Dict[str, object]) -> str:
        import fugu_local
        ok, output = fugu_local.run_python(str(args["code"]))
        return f"ok={ok}\n{output}"

    def _fetch_page(args: Dict[str, object]) -> str:
        import fugu_browser
        return fugu_browser.as_fetcher(max_chars=DEFAULT_OUTPUT_CHARS)(
            str(args["url"]))

    registry.register(ToolSpec(
        name="web_search",
        description="DuckDuckGo で Web 検索し、最新情報のスニペットを返す",
        schema={"type": "object", "properties": {"query": {"type": "string"}},
                "required": ["query"]},
        handler=_web_search))
    registry.register(ToolSpec(
        name="rag_search",
        description="設定済みローカル文書ディレクトリ(RAG)を検索して関連チャンクを返す",
        schema={"type": "object", "properties": {"question": {"type": "string"}},
                "required": ["question"]},
        handler=_rag_search))
    registry.register(ToolSpec(
        name="run_python",
        description="Python コードを実行して stdout/stderr を返す(検証・計算用)",
        schema={"type": "object", "properties": {"code": {"type": "string"}},
                "required": ["code"]},
        handler=_run_python))
    registry.register(ToolSpec(
        name="fetch_page",
        description="URL のページ本文テキストを取得する",
        schema={"type": "object", "properties": {"url": {"type": "string"}},
                "required": ["url"]},
        handler=_fetch_page))
    return registry
