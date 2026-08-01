"""Fugu MCP サーバー — Claude Code から fugu を道具として呼べるようにする。

MCP (Model Context Protocol) の stdio トランスポート(1 行 1 JSON-RPC メッセージ)を
標準ライブラリだけで実装した最小サーバー。Claude Code に登録すると
`mcp__fugu__fugu_ask` などのツールとして見える:

    claude mcp add --scope user fugu -- python D:/repos/fugu-local-integ/fugu_mcp.py

設計上の要点:

* **stdout は JSON-RPC 専用**。fugu_local は進捗を print で大量に出すため、
  起動直後に本物の stdout を確保し、以降の ``sys.stdout`` を stderr へ差し替える
  (MCP クライアントは stderr をログとして扱うので進捗はそちらに残る)。
* **回答は数分〜数十分かかる**。同期版 ``fugu_ask`` に加え、すぐ返る
  ``fugu_ask_start`` + ``fugu_ask_status`` のジョブ型を用意する(クライアント側の
  ツールタイムアウトに縛られない)。8GB VRAM では並列に回せないので同時実行は 1 件。
* fugu_local の import は初回ツール呼び出しまで遅延する(起動ハンドシェイクを
  すぐ返すため)。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid

# JSON-RPC 専用に確保した本物の stdout。_redirect_stdio()(main 起動時)が設定する。
_RPC_OUT = sys.stdout


def _redirect_stdio():
    """stdout を JSON-RPC 専用に確保し、以降の print() を stderr へ逃がす。

    fugu_local は進捗を print で大量に出すため、これを怠ると通信路が壊れる。
    import 時ではなく main() で行う(テストや対話利用で stdout を奪わないため)。
    """
    global _RPC_OUT
    _RPC_OUT = sys.stdout
    if hasattr(_RPC_OUT, "reconfigure"):
        _RPC_OUT.reconfigure(encoding="utf-8", errors="replace", newline="\n")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
    sys.stdout = sys.stderr          # fugu_local の print() をすべてログ側へ

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)                   # 履歴ファイル等の相対パスをリポジトリ基準にする

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "fugu", "version": "1.0.0"}

THINKING_CHOICES = ["off", "minimal", "low", "medium", "high", "ultra", "max", "auto"]

TOOLS = [
    {
        "name": "fugu_ask",
        "description": (
            "ローカル LLM 群(Ollama)の動的 Mixture-of-Agents で質問に答える。"
            "完了までブロックするので短時間で済む質問向け。数分かかりそうな質問は "
            "fugu_ask_start / fugu_ask_status を使うこと。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "質問文"},
                "search": {"type": "boolean",
                           "description": "Web 検索結果をコンテキストに注入する"},
                "thinking_budget": {"type": "string", "enum": THINKING_CHOICES,
                                    "description": "思考の深さ(既定 off)"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "fugu_ask_start",
        "description": (
            "fugu への質問をバックグラウンドジョブとして開始し、job_id を即返す。"
            "結果は fugu_ask_status で取得する。長い質問はこちらを推奨。"
            "同時実行は 1 件(8GB VRAM のため)。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "質問文"},
                "search": {"type": "boolean",
                           "description": "Web 検索結果をコンテキストに注入する"},
                "thinking_budget": {"type": "string", "enum": THINKING_CHOICES,
                                    "description": "思考の深さ(既定 off)"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "fugu_ask_status",
        "description": ("fugu_ask_start で開始したジョブの状態を返す"
                        "(running / done+回答 / error)。"),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "fugu_ask_start が返した ID"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "fugu_health",
        "description": "Ollama の生死・導入済みモデル・任意依存の充足を即時チェックする。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------- fugu 呼び出し

def _run_ask(question: str, search: bool = False,
             thinking_budget: str = "") -> str:
    """fugu_local を(初回のみ)読み込み、質問 1 件に答える。ワーカー内で実行される。"""
    if thinking_budget and thinking_budget != "off":
        os.environ["FUGU_THINKING_BUDGET"] = thinking_budget
    else:
        os.environ.pop("FUGU_THINKING_BUDGET", None)
    import fugu_local as f
    answer = f.ask_fugu(question, use_search=bool(search))
    if answer is None:
        raise RuntimeError("fugu の初期化に失敗しました(Ollama とモデルを確認)")
    if answer.startswith("__ERROR__"):
        raise RuntimeError(answer[len("__ERROR__"):].lstrip(":").strip()
                           or "パイプラインがエラーを返しました")
    return answer


class Jobs:
    """1 件ずつ実行するジョブ台帳。8GB VRAM なので並列実行は許可しない。"""

    def __init__(self, runner=_run_ask):
        self._runner = runner
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def _busy(self):
        return any(j["status"] == "running" for j in self._jobs.values())

    def start(self, question, search=False, thinking_budget=""):
        with self._lock:
            if self._busy():
                return None                       # 呼び出し側で busy エラーにする
            job_id = uuid.uuid4().hex[:12]
            job = {"status": "running", "question": question,
                   "started": time.time(), "answer": None, "error": None}
            self._jobs[job_id] = job

        def work():
            try:
                job["answer"] = self._runner(question, search, thinking_budget)
                job["status"] = "done"
            except Exception as exc:              # ジョブの失敗はサーバーを殺さない
                job["error"] = str(exc)
                job["status"] = "error"

        threading.Thread(target=work, daemon=True).start()
        return job_id

    def status(self, job_id):
        job = self._jobs.get(job_id)
        if job is None:
            return None
        out = {"status": job["status"],
               "elapsed_sec": round(time.time() - job["started"], 1),
               "question": job["question"]}
        if job["status"] == "done":
            out["answer"] = job["answer"]
        elif job["status"] == "error":
            out["error"] = job["error"]
        return out


def _health_text() -> str:
    """ランチャーの preflight を流用(fugu_local を import しない軽量チェック)。"""
    from fugu_launcher import check_env, format_check, load_settings
    return format_check(check_env(load_settings()))


# ---------------------------------------------------------------- JSON-RPC 処理

def _text_result(text, is_error=False):
    result = {"content": [{"type": "text", "text": text}]}
    if is_error:
        result["isError"] = True
    return result


def call_tool(name, args, jobs):
    """ツール 1 呼び出し → MCP result(例外は isError 付き結果に変換)。"""
    args = args or {}
    try:
        if name == "fugu_ask":
            return _text_result(jobs._runner(args["question"],
                                             bool(args.get("search")),
                                             args.get("thinking_budget", "")))
        if name == "fugu_ask_start":
            job_id = jobs.start(args["question"], bool(args.get("search")),
                                args.get("thinking_budget", ""))
            if job_id is None:
                return _text_result(
                    "別のジョブが実行中です(同時実行は 1 件)。"
                    "fugu_ask_status で完了を待ってください。", is_error=True)
            return _text_result(json.dumps(
                {"job_id": job_id,
                 "note": "fugu_ask_status をこの job_id で数十秒おきに呼ぶこと"},
                ensure_ascii=False))
        if name == "fugu_ask_status":
            status = jobs.status(args["job_id"])
            if status is None:
                return _text_result(f"job_id が見つかりません: {args['job_id']}",
                                    is_error=True)
            return _text_result(json.dumps(status, ensure_ascii=False))
        if name == "fugu_health":
            return _text_result(_health_text())
        return _text_result(f"unknown tool: {name}", is_error=True)
    except KeyError as exc:
        return _text_result(f"必須引数がありません: {exc}", is_error=True)
    except Exception as exc:
        return _text_result(f"エラー: {exc}", is_error=True)


def handle_message(msg, jobs):
    """JSON-RPC メッセージ 1 件 → 応答 dict(通知には None)。"""
    method = msg.get("method")
    msg_id = msg.get("id")
    if method is None or msg_id is None:          # 応答 or 通知(initialized 等)
        return None
    if method == "initialize":
        client_ver = (msg.get("params") or {}).get("protocolVersion")
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": client_ver or PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        result = call_tool(params.get("name", ""), params.get("arguments"), jobs)
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main():
    _redirect_stdio()
    jobs = Jobs()
    print(f"[fugu-mcp] ready (repo={REPO})", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue                              # 壊れた行は無視(通信路は守る)
        response = handle_message(msg, jobs)
        if response is not None:
            _RPC_OUT.write(json.dumps(response, ensure_ascii=False) + "\n")
            _RPC_OUT.flush()
    print("[fugu-mcp] stdin closed; exiting", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
