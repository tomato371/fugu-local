# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
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
        "description": ("Ollama の生死・導入済みモデル・任意依存の充足と、"
                        "ジョブサーバー(実運用の経路)の稼働状況を即時チェックする。"),
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


def _ollama_health_text() -> str:
    """ランチャーの preflight を流用(fugu_local を import しない軽量チェック)。"""
    from fugu_launcher import check_env, format_check, load_settings
    return format_check(check_env(load_settings()))


# ---------------------------------------------------------------- ジョブサーバー
# 実運用の経路は fugu_ctl.py(:8800 プロキシ)→ fugu_remote.py(NVIDIA NIM)。
# Ollama だけ見て "OK" と言っても本番の健全性は分からないので、ここでも見る。

# /api/health はサーバー側で NIM の生死確認(urlopen timeout=5、60 秒キャッシュ)を
# 行うことがあり、それだけで ~5 秒かかりうる。5 秒だと「NIM が遅い」を「サーバーが
# 落ちている」と誤診するので 10 秒取る。stats は jobs/ 走査(30 秒キャッシュ)なので 5 秒。
# 最悪ケースの合計は 10 + 5 = 15 秒(health が応答なし/拒否ならそこで打ち切り=最大 10 秒)。
_HEALTH_TIMEOUT = 10.0
_STATS_TIMEOUT = 5.0
_ERR_TIMEOUT = "timeout"      # _http_get_json の err 文字列の先頭に付く分類タグ
_ERR_REFUSED = "refused"
_SECRETISH = ("token", "secret", "password", "api_key", "apikey")
# /api/health の既知キー(表示順とラベル)。未知・欠落は黙って飛ばす。
_HEALTH_FIELDS = [
    ("backend", "backend"), ("backend_up", "backend_up"),
    ("backend_url", "backend_url"), ("setup_done", "setup_done"),
    ("setup_ok", "setup_ok"), ("queued", "queued"), ("running", "running"),
    ("current", "current"), ("phase", "phase"), ("parallel", "parallel"),
    ("version", "version"),
]
_HEALTH_EXTRA_HINTS = ("lineup", "conductor", "bad")   # 名前に含むキーも拾う


def _job_server_base() -> str:
    """ジョブサーバーの base URL。FUGU_JOB_SERVER_URL > FUGU_REMOTE_PORT > 8800。"""
    url = os.environ.get("FUGU_JOB_SERVER_URL", "").strip()
    if url:
        return url.rstrip("/")
    port = os.environ.get("FUGU_REMOTE_PORT", "").strip() or "8800"
    return f"http://127.0.0.1:{port}"


def _display_url(url: str) -> str:
    """表示用 URL。userinfo(user:pass@)と query/fragment(?token=...)は出さない。"""
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:                             # IPv6 リテラル
            host = f"[{host}]"
        netloc = host + (f":{parts.port}" if parts.port else "")
        if parts.username is not None or parts.password is not None:
            netloc = "***@" + netloc
        return f"{parts.scheme}://{netloc}{parts.path}" if parts.scheme else "(不正な URL)"
    except ValueError:
        return "(不正な URL)"


def _urlopen(req, timeout):
    """プロキシを経由せずに開く(127.0.0.1 宛の診断が HTTP(S)_PROXY で曲がらないように)。"""
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=timeout)


def _classify_exc(exc) -> str:
    """接続系の例外を (timeout / refused / その他) に分けて err 文字列にする。"""
    import socket
    reason = getattr(exc, "reason", None)
    inner = reason if isinstance(reason, BaseException) else exc
    if isinstance(inner, (socket.timeout, TimeoutError)) or "timed out" in str(inner):
        return f"{_ERR_TIMEOUT}: {inner}"
    if isinstance(inner, ConnectionRefusedError):
        return f"{_ERR_REFUSED}: {inner}"
    return f"{type(inner).__name__}: {inner}"


def _http_get_json(path: str, timeout: float = _HEALTH_TIMEOUT):
    """ジョブサーバーへ GET → (HTTP ステータス or None, JSON or None, エラー文)。

    認証は fugu_remote と同じ X-Fugu-Token ヘッダ(環境変数 FUGU_REMOTE_TOKEN)。
    例外は投げない(到達不能も戻り値で表す)。接続失敗の err は先頭が
    "timeout:" / "refused:" のどちらかなら原因が分かっている。HTTP エラー
    (fugu_ctl の 503 {"detail": ...} 等)でも本文が JSON ならそれを返す。
    """
    import urllib.error
    import urllib.request
    req = urllib.request.Request(_job_server_base() + path)
    token = os.environ.get("FUGU_REMOTE_TOKEN", "").strip()
    if token:
        req.add_header("X-Fugu-Token", token)
    try:
        with _urlopen(req, timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        body = None
        try:
            body = json.loads(exc.read(65536).decode("utf-8", "replace"))
        except Exception:                          # 本文なし/非 JSON は無視
            pass
        return exc.code, body, f"HTTP {exc.code}"
    except Exception as exc:                       # URLError / timeout / reset 等
        return None, None, _classify_exc(exc)
    try:
        return status, json.loads(raw.decode("utf-8", "replace")), ""
    except ValueError:
        return status, None, "JSON ではない応答"


def _detail_of(obj, err="") -> str:
    """エラー応答の detail(無ければ err か本文の要約)。"""
    if isinstance(obj, dict) and "detail" in obj:
        d = obj["detail"]
        return d if isinstance(d, str) else _compact(d, 160)
    return err or (_compact(obj, 120) if obj is not None else "")


def _compact(value, limit=200) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      default=str)
    return text if len(text) <= limit else text[:limit - 1] + "…"


# fugu_remote の OUTCOME_BAD(+ 壊れた job.json = broken)。回答を受け取れなかった結果。
_STATS_BAD_KINDS = ("failed", "interrupted", "empty", "error_answer", "broken")
# 回答は出たが品質が落ちた結果(問題件数には数えず、救済として別掲)。
_STATS_SALVAGE_KINDS = ("salvaged", "partial")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _kind_counts(kinds, order):
    """kinds から order の種類だけ(0 件は除く)取り出し (合計, 'k=n, ...') を返す。"""
    nums = {k: kinds[k] for k in order
            if _is_num(kinds.get(k)) and kinds[k] > 0}
    return sum(nums.values()), ", ".join(f"{k}={v}" for k, v in nums.items())


def _window_line(name, w, with_timing=False) -> str:
    kinds = w.get("kinds") if isinstance(w.get("kinds"), dict) else {}
    total = w.get("total")
    parts = [f"{name}: 計{total}件" if _is_num(total) else f"{name}: 計?件"]
    nbad, bad_txt = _kind_counts(kinds, _STATS_BAD_KINDS)
    parts.append(f"問題{nbad}件" + (f" ({bad_txt})" if bad_txt else ""))
    nsal, sal_txt = _kind_counts(kinds, _STATS_SALVAGE_KINDS)
    if nsal:
        parts.append(f"救済{nsal}件 ({sal_txt})")
    ok = w.get("ok_elapsed_sec")
    if with_timing and isinstance(ok, dict) and _is_num(ok.get("median")):
        txt = f"ok所要 中央値{ok['median']}秒"
        if _is_num(ok.get("p90")):
            txt += f" / p90 {ok['p90']}秒"
        if _is_num(ok.get("n")):
            txt += f" (n={ok['n']})"
        parts.append(txt)
    return " / ".join(parts)


def _stats_line(stats) -> str:
    """/api/stats の要約。fugu_remote の実形状 {"windows": {"24h": {...}, "7d": {...}}}
    を優先して読み、知らない形なら汎用の総当たり(_stats_line_generic)に落とす。
    複数行になりうる(24h の次の行に 7d)。"""
    wins = stats.get("windows") if isinstance(stats, dict) else None
    if isinstance(wins, dict):
        picked = [(n, wins[n]) for n in ("24h", "7d")
                  if isinstance(wins.get(n), dict)]
        if picked:
            return "\n".join(_window_line(n, w, with_timing=(n == "24h"))
                             for n, w in picked)
    return _stats_line_generic(stats)


def _stats_line_generic(stats) -> str:
    """未知の形状向けの 1 行要約(キー名を総当たりで防御的に)。"""
    if not isinstance(stats, dict):
        return _compact(stats)
    for key in ("last_24h", "24h", "window_24h"):   # 入れ子ならそちらを見る
        if isinstance(stats.get(key), dict):
            stats = stats[key]
            break
    total = next((stats[k] for k in ("total", "jobs", "count", "n")
                  if isinstance(stats.get(k), int)
                  and not isinstance(stats.get(k), bool)), None)
    bad = next((stats[k] for k in ("bad_by_kind", "bad", "bad_outcomes",
                                   "bad_kinds") if k in stats), None)
    if total is None and bad is None:
        return _compact(stats)                      # 形が分からなければそのまま
    parts = [f"total={total}" if total is not None else "total=?"]
    if isinstance(bad, dict):
        nums = {k: v for k, v in bad.items()
                if isinstance(v, int) and not isinstance(v, bool)}
        kinds = ", ".join(f"{k}={v}" for k, v in nums.items())
        parts.append(f"bad={sum(nums.values())}" + (f" ({kinds})" if kinds else ""))
    elif isinstance(bad, int) and not isinstance(bad, bool):
        parts.append(f"bad={bad}")
    elif bad is not None:
        parts.append(f"bad={_compact(bad, 80)}")
    return "24h: " + " / ".join(parts)


def _job_server_text() -> str:
    """ジョブサーバー(本番経路)の健全性。到達不能でも例外を出さず明示する。"""
    base = _job_server_base()
    lines = ["== ジョブサーバー(実運用の経路) ==", f"URL: {_display_url(base)}"]
    try:
        status, health, err = _http_get_json("/api/health", _HEALTH_TIMEOUT)
        if status is None:
            if err.startswith(_ERR_TIMEOUT):
                lines.append(f"[NG] 応答なし(ハングの可能性): {_HEALTH_TIMEOUT:g} 秒待っても"
                             f"返事がありません ({err})")
                lines.append("     ポートは開いているのに返事が無い状態です。"
                             "fugu_remote の詰まり、または NIM 確認の遅延が疑われます。")
            elif err.startswith(_ERR_REFUSED):
                lines.append(f"[NG] 接続拒否(停止中): {err}")
                lines.append("     fugu_ctl.py が起動していません(START_FUGU_REMOTE.bat 等で起動)。")
            else:
                lines.append(f"[NG] ジョブサーバーに到達できません: {err}")
                lines.append("     fugu_ctl.py / fugu_remote.py が落ちている可能性があります。")
            return "\n".join(lines)
        if status in (401, 403):
            lines.append(f"[NG] 認証エラー(HTTP {status}): 環境変数 FUGU_REMOTE_TOKEN を確認")
            return "\n".join(lines)
        if status != 200 or not isinstance(health, dict):
            lines.append(f"[NG] 応答が異常です(HTTP {status}): "
                         f"{_detail_of(health, err)}")
            return "\n".join(lines)
        lines.append("[OK] 応答あり")
        shown = set()
        for key, label in _HEALTH_FIELDS:
            if key in health:
                shown.add(key)
                lines.append(f"  {label}: {_compact(health[key], 120)}")
        for key in health:
            low = str(key).lower()
            if (key not in shown and any(h in low for h in _HEALTH_EXTRA_HINTS)
                    and not any(s in low for s in _SECRETISH)):
                lines.append(f"  {key}: {_compact(health[key])}")
        if health.get("backend_up") is False:
            lines.append("  [NG] backend が落ちています(NIM / Ollama への接続を確認)")
        s_status, stats, s_err = _http_get_json("/api/stats", _STATS_TIMEOUT)
        if s_status == 200 and stats is not None:
            lines.extend("  " + ln for ln in _stats_line(stats).splitlines())
        elif s_status == 404:
            lines.append("  stats: 未対応(/api/stats は 404)")
        else:
            why = _detail_of(stats, s_err) or "不明"
            lines.append(f"  stats: 取得できず({why})" if s_status is None
                         else f"  stats: 取得できず(HTTP {s_status}: {why})")
    except Exception as exc:                        # 診断が診断対象を壊さない
        lines.append(f"[NG] ジョブサーバーの確認中にエラー: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def _health_text() -> str:
    """Ollama(ローカル)+ ジョブサーバー(実運用)の二段チェック。"""
    return _ollama_health_text() + "\n\n" + _job_server_text()


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
