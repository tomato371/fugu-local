"""NIM クラウドバックエンドのオフライン回帰テスト。実 API・Ollama 不要・数秒で完走。
実行: python test_nim_offline.py

urllib.request.urlopen をモックして _ask_nim の送信 payload・リトライ・予算・
ディスパッチ・apply_nim_profile の不変条件を検証する。test_fugu_offline.py と
同じスクリプト形式（check() + 終了コード）。
"""
import contextlib
import copy
import io
import json
import tempfile
import urllib.error
import email.message
from pathlib import Path

import fugu_local as f

_FAILS = []


def check(name, cond):
    print(f"[{'OK' if cond else 'NG'}] {name}")
    if not cond:
        _FAILS.append(name)


# ---------- モック部品 ----------

class FakeResponse:
    """非ストリーム応答 (read) と SSE ストリーム応答 (行イテレート) の両対応モック。
    _ask_nim は常に stream=True で送り応答を行単位でイテレートする。/models 等は read()。"""
    def __init__(self, body=None, sse_lines=None):
        self._body = json.dumps(body).encode("utf-8") if body is not None else b""
        self._sse = sse_lines or []
        self.status = 200

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._sse)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok_body(content="hello", finish="stop"):
    """content を2チャンクに割った SSE ストリーム応答を作る（実 NIM の形を模す）。"""
    lines = []
    half = max(1, len(content) // 2)
    for piece in (content[:half], content[half:]):
        if piece:
            lines.append(("data: " + json.dumps(
                {"choices": [{"delta": {"content": piece}}]}) + "\n").encode())
    # reasoning_content チャンクは無視されるべきノイズとして混ぜる
    lines.insert(0, ("data: " + json.dumps(
        {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}) + "\n").encode())
    lines.append(("data: " + json.dumps(
        {"choices": [{"delta": {}, "finish_reason": finish}]}) + "\n").encode())
    lines.append(b"data: [DONE]\n")
    return FakeResponse(sse_lines=lines)


def http_error(code, body=b"err", headers=None):
    """毎回新しい HTTPError を作るファクトリ（read() は一度しか呼べないため）。"""
    def make():
        h = email.message.Message()
        for k, v in (headers or {}).items():
            h[k] = str(v)
        raise urllib.error.HTTPError("http://x", code, "err", h, io.BytesIO(body))
    return make


def run_mocked(responses, fn):
    """urlopen / time.sleep / time.time をモックして fn() を実行。
    sleep は仮想時計を進める（進めないと 429 グローバルクールダウンの送信前待機が
    「時刻が進まないのに sleep だけ空回りする」無限ループになる）。
    responses: 各送信に対する応答のリスト。FakeResponse か callable(raise 用)。
    リストが尽きたら最後の要素を繰り返す。
    戻り値: (fn の結果 or 送出された SystemExit, 送信された Request のリスト, sleep 秒のリスト)"""
    sent, sleeps = [], []
    clock = [1_000_000.0]
    orig_open, orig_sleep, orig_time = f.urllib.request.urlopen, f.time.sleep, f.time.time

    def fake_open(req, timeout=None):
        sent.append(req)
        r = responses[min(len(sent) - 1, len(responses) - 1)]
        if callable(r):
            return r()
        return r

    def fake_sleep(s):
        sleeps.append(s)
        clock[0] += max(float(s), 0.001)

    f.urllib.request.urlopen = fake_open
    f.time.sleep = fake_sleep
    f.time.time = lambda: clock[0]
    f._NIM_COOLDOWN.clear()
    try:
        out = fn()
    except SystemExit as e:
        out = e
    finally:
        f.urllib.request.urlopen = orig_open
        f.time.sleep = orig_sleep
        f.time.time = orig_time
        f._NIM_COOLDOWN.clear()
    return out, sent, sleeps


def payload_of(req):
    return json.loads(req.data.decode("utf-8"))


_TMP = Path(tempfile.mkdtemp(prefix="nim_test_"))

# NIM グローバルの退避（このテストが差し替えるもの全部）
_SAVED = dict(
    NIM_MODEL_IDS=set(f.NIM_MODEL_IDS), NIM_STRUCTURED_OK=set(f.NIM_STRUCTURED_OK),
    NIM_API_KEY=f.NIM_API_KEY, NIM_BUDGET=f.NIM_BUDGET, NIM_BUDGET_FILE=f.NIM_BUDGET_FILE,
    NIM_REQUEST_COUNT=f.NIM_REQUEST_COUNT, FUGU_BACKEND=f.FUGU_BACKEND,
)
f.NIM_API_KEY = "nvapi-TEST-KEY"
f.NIM_BUDGET = 0
f.NIM_BUDGET_FILE = _TMP / "nim_usage.json"
f.NIM_MODEL_IDS = {"test/model", "test/structured"}
f.NIM_STRUCTURED_OK = {"test/structured"}

MSGS = [{"role": "user", "content": "hi"}]

# ---------- 1. 送信 payload の形 ----------
out, sent, _ = run_mocked([(ok_body("world"))],
                          lambda: f.ask("test/model", MSGS, 0.5,
                                        num_predict=123, num_ctx=8192))
check("payload: 応答 content がそのまま返る", out == "world")
check("payload: URL は {NIM_URL}/chat/completions",
      sent and sent[0].full_url == f"{f.NIM_URL}/chat/completions")
check("payload: Authorization は Bearer キー",
      sent and sent[0].get_header("Authorization") == "Bearer nvapi-TEST-KEY")
p = payload_of(sent[0])
check("payload: num_predict → max_tokens", p.get("max_tokens") == 123)
check("payload: temperature 透過", p.get("temperature") == 0.5)
check("payload: num_ctx / options / keep_alive / think は送らない",
      all(k not in p for k in ("num_ctx", "options", "keep_alive", "think")))
check("payload: 未指定 think は reasoning_effort 不送信", "reasoning_effort" not in p)

# ---------- 2. ディスパッチ（レジストリ方式）----------
# 未登録モデル（`/` を含むローカルモデル）は Ollama 経路へ行き num_ctx ピン留めが維持される
out, sent, _ = run_mocked([FakeResponse({"message": {"content": "local"}})],
                          lambda: f.ask("NitrAI/VibeThinker-3B", MSGS, 0.3))
check("dispatch: `/` 入りでも未登録なら Ollama 経路 (/api/chat)",
      sent and sent[0].full_url.endswith("/api/chat"))
check("dispatch: Ollama 経路は options.num_ctx ピン留め維持 (gotcha #1/#2)",
      "num_ctx" in payload_of(sent[0]).get("options", {}))
check("dispatch: Ollama 経路の応答契約も不変", out == "local")

# ---------- 3. __ERROR__ 契約 ----------
out, sent, sleeps = run_mocked([http_error(500)],
                               lambda: f.ask("test/model", MSGS, 0.5))
check("error: HTTP 500 連発は __ERROR__: 文字列（例外を上げない）",
      isinstance(out, str) and out.startswith("__ERROR__"))
check("error: 通常リトライ予算は ASK_RETRY_ATTEMPTS 回",
      len(sent) == f.ASK_RETRY_ATTEMPTS)
out, sent, _ = run_mocked([FakeResponse(sse_lines=[b"data: [DONE]\n"])],
                          lambda: f.ask("test/model", MSGS, 0.5))
check("error: choices 空でもクラッシュせず空文字/エラー", isinstance(out, str))

# ---------- 4. 429 / Retry-After（別予算・cap・attempt 不消費）----------
out, sent, sleeps = run_mocked(
    [http_error(429, headers={"Retry-After": "3"}),
     http_error(429, headers={"Retry-After": "999"}),
     (ok_body("recovered"))],
    lambda: f.ask("test/model", MSGS, 0.5))
check("429: 待って成功すれば通常の応答", out == "recovered")
check("429: 送信は 3 回（初回+リトライ2）", len(sent) == 3)
check("429: Retry-After を尊重", sleeps and sleeps[0] == 3.0)
check("429: Retry-After は cap で抑える", len(sleeps) >= 2 and sleeps[1] == f.NIM_RETRY_AFTER_CAP)
check("429: 通常予算の指数バックオフは混ざらない",
      all(s not in f.ASK_RETRY_BACKOFF for s in sleeps))

# Retry-After 無し(NIM の実挙動): 指数バックオフ 20→40→… + グローバルクールダウン
out, sent, sleeps = run_mocked(
    [http_error(429), http_error(429), (ok_body("calm"))],
    lambda: f.ask("test/model", MSGS, 0.5))
check("429: Retry-After無しは指数バックオフ(20,40)", out == "calm" and sleeps[:2] == [20.0, 40.0])
_cool_seen = []
def _cool_probe():
    r = f.ask("test/model", MSGS, 0.5)
    _cool_seen.append(dict(f._NIM_COOLDOWN))
    return r
out, _, _ = run_mocked([http_error(429), (ok_body())], _cool_probe)
check("429: モデル別クールダウンが将来時刻に設定される(同モデルの全ワーカー抑制)",
      _cool_seen and _cool_seen[0].get("test/model", 0) > 1_000_000.0)
check("429: 他モデルのクールダウンは汚さない(per-model分離)",
      _cool_seen and list(_cool_seen[0].keys()) == ["test/model"])

# ---------- 5. length 打ち切り + max_tokens 自動増額 ----------
out, sent, _ = run_mocked([(ok_body("", finish="length")),
                           (ok_body("rescued"))],
                          lambda: f.ask("test/model", MSGS, 0.5, num_predict=16384))
check("length: 打ち切り時は max_tokens 倍増で1回だけ再送し票を救済",
      out == "rescued" and len(sent) == 2
      and payload_of(sent[1])["max_tokens"] == 32768)
out, sent, _ = run_mocked([(ok_body("", finish="length"))],
                          lambda: f.ask("test/model", MSGS, 0.5, num_predict=16384))
check("length: 増額後も打ち切りなら __ERROR__: truncated（SC 無効票化・再送は1回のみ）",
      isinstance(out, str) and out.startswith("__ERROR__") and "truncated" in out
      and len(sent) == 2)
out, sent, _ = run_mocked([(ok_body("", finish="length"))],
                          lambda: f.ask("test/model", MSGS, 0.5, num_predict=32768))
check("length: 既に上限なら増額しない", len(sent) == 1 and out.startswith("__ERROR__"))
out, sent, _ = run_mocked([(ok_body("", finish="length"))],
                          lambda: f.ask("test/model", MSGS, 0.5))
check("length: max_tokens未指定は増額対象外", len(sent) == 1 and out.startswith("__ERROR__"))
out, _, _ = run_mocked([(ok_body("partial answer", finish="length"))],
                       lambda: f.ask("test/model", MSGS, 0.5))
check("length: 本文が一部でもあればそのまま使う", out == "partial answer")

# ---------- 6. リクエスト予算（SystemExit 42・送信前ブロック）----------
f.NIM_BUDGET_FILE.write_text(json.dumps({"total_requests": 2}), encoding="utf-8")
f.NIM_BUDGET = 2
out, sent, _ = run_mocked([(ok_body())],
                          lambda: f.ask("test/model", MSGS, 0.5))
check("budget: 上限到達で SystemExit(42)",
      isinstance(out, SystemExit) and out.code == 42)
check("budget: 送信自体が行われない", len(sent) == 0)
f.NIM_BUDGET = 0
before = f.NIM_REQUEST_COUNT
out, sent, _ = run_mocked([(ok_body())],
                          lambda: f.ask("test/model", MSGS, 0.5))
check("budget: カウンタは送信ごとに増える", f.NIM_REQUEST_COUNT == before + 1)
check("budget: nim_usage.json に累計が永続化される",
      json.loads(f.NIM_BUDGET_FILE.read_text(encoding="utf-8"))["total_requests"] == 3)

# ---------- 7. reasoning_effort / response_format の 400 落とし再送 ----------
out, sent, _ = run_mocked([http_error(400, body=b"param not supported"),
                           (ok_body("ok2"))],
                          lambda: f.ask("test/model", MSGS, 0.5, think="high"))
check("400drop: 1回目に reasoning_effort を送る",
      "reasoning_effort" in payload_of(sent[0]))
check("400drop: 400 なら落として即再送・成功", out == "ok2" and len(sent) == 2)
check("400drop: 再送 payload に拡張パラメータが無い",
      "reasoning_effort" not in payload_of(sent[1]))
check("400drop: think=True は high に写像",
      payload_of(run_mocked([(ok_body())],
                            lambda: f.ask("test/model", MSGS, 0.5, think=True))[1][0]
                 ).get("reasoning_effort") == "high")

# ---------- 7b. think_style=template（gemma-4 系: effort ではなく chat_template_kwargs）----------
# gemma-4 は reasoning_effort を 400 にせず黙って無視するため、think を効かせるには
# chat_template_kwargs.enable_thinking を送るしかない（2026-08-19 実測）。
_saved_ts = copy.deepcopy(f.MODEL_CONFIG)
try:
    f.MODEL_CONFIG["test/model"] = {"think_style": "template"}
    p = payload_of(run_mocked([(ok_body())],
                              lambda: f.ask("test/model", MSGS, 0.5, think=True))[1][0])
    check("think_style: think=True は enable_thinking で開く（effort は送らない）",
          p.get("chat_template_kwargs") == {"enable_thinking": True}
          and "reasoning_effort" not in p)
    p = payload_of(run_mocked([(ok_body())],
                              lambda: f.ask("test/model", MSGS, 0.5, think=False))[1][0])
    check("think_style: think=False は enable_thinking=False で明示的に閉じる",
          p.get("chat_template_kwargs") == {"enable_thinking": False})
    f.MODEL_CONFIG["test/model"] = {"think_style": "template", "reasoning_budget": 4096}
    p = payload_of(run_mocked([(ok_body())],
                              lambda: f.ask("test/model", MSGS, 0.5, think=True))[1][0])
    check("think_style: reasoning_budget があれば併送", p.get("reasoning_budget") == 4096)
    _o, _s, _ = run_mocked([http_error(400, body=b"param not supported"), (ok_body("ok3"))],
                           lambda: f.ask("test/model", MSGS, 0.5, think=True))
    check("think_style: 400 なら chat_template_kwargs も落として再送",
          _o == "ok3" and len(_s) == 2
          and "chat_template_kwargs" not in payload_of(_s[1])
          and "reasoning_budget" not in payload_of(_s[1]))
finally:
    f.MODEL_CONFIG.clear()
    f.MODEL_CONFIG.update(_saved_ts)

# ---------- 8. スキーマ（fmt）の扱い ----------
schema = {"type": "object", "properties": {"mode": {"type": "string"}}}
out, sent, _ = run_mocked([(ok_body())],
                          lambda: f.ask("test/model", MSGS, 0.5, fmt=schema))
p = payload_of(sent[0])
check("fmt: 非対応モデルは response_format を送らない", "response_format" not in p)
check("fmt: schema は system へ文字列注入される",
      any(m["role"] == "system" and "mode" in m["content"] for m in p["messages"]))
check("fmt: 元の messages リストは破壊しない（コピーに注入）",
      all(m["role"] != "system" for m in MSGS))
out, sent, _ = run_mocked([(ok_body())],
                          lambda: f.ask("test/structured", MSGS, 0.5, fmt=schema))
p = payload_of(sent[0])
check("fmt: NIM_STRUCTURED_OK は response_format=json_object 併用",
      p.get("response_format") == {"type": "json_object"})

# ---------- 8b. nim_extra（モデル別追加ペイロード, nemotron 系の思考有効化等）----------
_saved_mc = copy.deepcopy(f.MODEL_CONFIG)
try:
    f.MODEL_CONFIG["test/model"] = {
        "nim_extra": {"chat_template_kwargs": {"enable_thinking": True},
                      "reasoning_budget": 999}}
    out, sent, _ = run_mocked([(ok_body())],
                              lambda: f.ask("test/model", MSGS, 0.5))
    p = payload_of(sent[0])
    check("nim_extra: トップレベルにマージされる",
          p.get("chat_template_kwargs") == {"enable_thinking": True}
          and p.get("reasoning_budget") == 999)
    out, sent, _ = run_mocked([http_error(400), (ok_body("ok3"))],
                              lambda: f.ask("test/model", MSGS, 0.5))
    check("nim_extra: 400 なら extra キーも落として再送・成功",
          out == "ok3" and len(sent) == 2
          and all(k not in payload_of(sent[1])
                  for k in ("chat_template_kwargs", "reasoning_budget")))
finally:
    f.MODEL_CONFIG = _saved_mc

# ---------- 8c. ストリーミング暴走ガード（壁時計上限・ping飢餓） ----------
def _slow_sse(advance, lines):
    """イテレートのたびに仮想時計を advance 秒進める SSE モック。"""
    def gen():
        for l in lines:
            f.time.sleep(advance)   # run_mocked の fake_sleep が仮想時計を進める
            yield l
    r = FakeResponse()
    r.__iter__ = None
    class _R(FakeResponse):
        def __iter__(self):
            return gen()
    return _R()

out, sent, _ = run_mocked(
    [_slow_sse(f.NIM_STREAM_MAX_S / 2 + 1,
               [b": ping\n", b": ping\n", b": ping\n", b": ping\n"])],
    lambda: f.ask("test/model", MSGS, 0.5))
check("stream: 壁時計上限超過は __ERROR__ に落ちる（無期限張り付き防止）",
      isinstance(out, str) and out.startswith("__ERROR__"))
out, sent, _ = run_mocked(
    [_slow_sse(f.NIM_STREAM_IDLE_S + 1, [b": ping\n", b": ping\n"])],
    lambda: f.ask("test/model", MSGS, 0.5))
check("stream: data 行が来ない ping 飢餓も __ERROR__ に落ちる",
      isinstance(out, str) and out.startswith("__ERROR__"))

# ---------- 9. <think> 混入は呼び出し側 strip_think の責務（Ollama 経路と同一分担）----------
out, _, _ = run_mocked([(ok_body("<think>reasoning...</think>42"))],
                       lambda: f.ask("test/model", MSGS, 0.5))
check("think: _ask_nim は content を素通し", out == "<think>reasoning...</think>42")
check("think: strip_think が除去（SC 抽出を汚染しない）", f.strip_think(out) == "42")

# ---------- 10. サーバー/導入チェックのクラウドバイパス ----------
f.FUGU_BACKEND = "nim"
out, sent, _ = run_mocked([FakeResponse({"data": []})], f.server_up)
check("bypass: server_up は /models を叩く",
      out is True and sent and sent[0].full_url == f"{f.NIM_URL}/models")
check("bypass: installed_models はレジストリ（API を叩かない）",
      f.installed_models() == sorted(f.NIM_MODEL_IDS))
check("bypass: is_installed は NIM ID の厳密一致で成立（無改修）",
      f.is_installed("test/model", f.installed_models()))
f.FUGU_BACKEND = "ollama"

# ---------- 11. apply_nim_profile の不変条件（try/finally で完全復元）----------
_g = ("DESIRED_PROPOSERS", "DESIRED_AGGREGATOR", "DESIRED_CONDUCTOR", "FALLBACK_MODEL",
      "PERSONA_MODELS", "PERSONA_IDENTITY", "MODEL_TO_PERSONA", "PROPOSER_PROFILES",
      "JP_AGGREGATOR", "JP_AGGREGATOR_STRONG", "AGGREGATOR_REASONING",
      "SECOND_OPINION_MODEL", "MODEL_CONFIG", "PARALLEL_PROPOSERS", "REASONING_MODELS",
      "ARBITER_MODEL", "SC_CHEAP_VOTES", "SC_PARALLEL", "NIM_MODEL_IDS", "NIM_STRUCTURED_OK")
_saved_profile = {k: copy.deepcopy(getattr(f, k)) for k in _g}
try:
    # 選抜はカタログ取得と生存プローブを並列で叩くため、HTTP モックだと送信順が不定になる。
    # ここは選抜の判断そのものを見たいので、その 2 関数を差し替えて決定的に検査する。
    _CAT = {
        "meta/llama-3.1-8b-instruct",            # 8B  (Conductor 候補)
        "meta/llama-3.3-70b-instruct",           # 70B
        "openai/gpt-oss-120b",                   # 120B
        "mistralai/mistral-medium-3.5-128b",     # 128B
        "z-ai/glm-5.2",                          # 公称 355B
        "minimaxai/minimax-m3",                  # 公称 456B
        "nvidia/nemotron-3-ultra-550b-a55b",     # 550B/a55B
        "moonshotai/kimi-k2.6",                  # 公称 1000B
        "baai/bge-m3",                           # 埋め込み → 除外されること
        "meta/codellama-70b",                    # コード専用 → 除外されること
    }
    _orig_cat, _orig_avail = f._nim_catalog, f._nim_probe

    def _profile_with(dead=(), catalog=None):
        """dead に挙げた ID だけ落ちている状態で apply_nim_profile を回す。"""
        f._nim_catalog = lambda: (set(_CAT) if catalog is None else set(catalog))
        f._nim_probe = lambda m, timeout=None: ("gone" if m in dead else "ok")
        _b = io.StringIO()
        with contextlib.redirect_stdout(_b):
            return f.apply_nim_profile(), _b.getvalue()

    applied, _log = _profile_with()
    buf = io.StringIO()
    buf.write(_log)
    check("profile: 適用成功 (True)", applied is True)
    check("profile: タイポ警告なし（採用 ID は全て実在扱い）", "⚠" not in buf.getvalue())
    check("profile: MODEL_TO_PERSONA が再導出される",
          set(f.MODEL_TO_PERSONA) == set(f.PERSONA_MODELS.values()))
    check("profile: PERSONA_IDENTITY/PROPOSER_PROFILES も新 ID キー",
          set(f.PERSONA_IDENTITY) == set(f.PERSONA_MODELS.values())
          and set(f.PROPOSER_PROFILES) == set(f.PERSONA_MODELS.values()))
    roles = (list(f.DESIRED_PROPOSERS)
             + [f.DESIRED_AGGREGATOR, f.DESIRED_CONDUCTOR, f.JP_AGGREGATOR,
                f.SECOND_OPINION_MODEL, f.ARBITER_MODEL]
             + list(f.REASONING_MODELS))
    check("profile: 全ロールが NIM_MODEL_IDS に被覆される（ディスパッチ漏れゼロ）",
          all(m in f.NIM_MODEL_IDS for m in roles))
    check("profile: REASONING_MODELS ⊆ DESIRED 側レジストリ + SC 3系統",
          len(f.REASONING_MODELS) == 3)
    check("profile: 全 NIM モデルに num_predict がある（打ち切り保険）",
          all(f.model_cfg(m, "num_predict") for m in f.NIM_MODEL_IDS))
    check("profile: NIM エントリに num_ctx キーが無い",
          all("num_ctx" not in f.MODEL_CONFIG[m] for m in f.NIM_MODEL_IDS))
    check("profile: SC_CHEAP_VOTES=0 / 並列 ON",
          f.SC_CHEAP_VOTES == 0 and f.PARALLEL_PROPOSERS and f.SC_PARALLEL)
    # --- 起動時選抜 (2026-08-08) ---
    check("選抜: 規模の大きい順に 5 体を採用する",
          list(f.DESIRED_PROPOSERS) == ["moonshotai/kimi-k2.6",
                                        "nvidia/nemotron-3-ultra-550b-a55b",
                                        "minimaxai/minimax-m3",
                                        "z-ai/glm-5.2"]
          and f.DESIRED_AGGREGATOR == "mistralai/mistral-medium-3.5-128b")
    check("選抜: 裁定は最大規模の生存モデル",
          f.ARBITER_MODEL == "moonshotai/kimi-k2.6")
    check("選抜: 埋め込み/コード専用はそもそも候補に入らない",
          "baai/bge-m3" not in f.NIM_MODEL_IDS
          and "meta/codellama-70b" not in f.NIM_MODEL_IDS)
    check("選抜: Conductor は候補先頭の Gemma 4（フロンティア機）を採る",
          f.DESIRED_CONDUCTOR == "google/gemma-4-31b-it")
    check("選抜: Conductor に think_style=template が付く（effort が効かない系統）",
          f.MODEL_CONFIG["google/gemma-4-31b-it"].get("think_style") == "template")
    check("選抜: SC 3 系統はベンダーが重複しない",
          len({m.split("/")[0] for m in f.REASONING_MODELS}) == 3)
    check("選抜: 思考の効かせ方は系統から引く（gpt-oss は effort / nemotron は template）",
          f.MODEL_CONFIG["nvidia/nemotron-3-ultra-550b-a55b"]["nim_extra"]
          ["chat_template_kwargs"]["enable_thinking"] is True)

    # 死んでいる ID は飛ばして次点を採る（固定 ID の腐りへの対処そのもの）
    applied2, _log2 = _profile_with(dead={"moonshotai/kimi-k2.6", "minimaxai/minimax-m3"})
    check("選抜: 死んでいるモデルは採らず次点へ送る",
          applied2 is True
          and "moonshotai/kimi-k2.6" not in f.NIM_MODEL_IDS
          and "minimaxai/minimax-m3" not in f.NIM_MODEL_IDS
          and f.ARBITER_MODEL == "nvidia/nemotron-3-ultra-550b-a55b")
    check("選抜: 落としたモデルはログに残す", "NG（未付与/廃止）" in _log2)

    # カタログ不通なら旧固定布陣へ退避する（選抜が使えないだけで停止はしない）
    applied3, _log3 = _profile_with(catalog=set())
    check("選抜: カタログ不通なら固定布陣へ退避",
          applied3 is True
          and "nvidia/nemotron-3-ultra-550b-a55b" in f.DESIRED_PROPOSERS
          and "⚠" in _log3)

    # Conductor が落ちていれば従来の軽量機へ退避する（可用性は落とさない）
    applied4, _log4 = _profile_with(dead={"google/gemma-4-31b-it"})
    check("選抜: Conductor が死んでいれば軽量機へ退避",
          applied4 is True and f.DESIRED_CONDUCTOR == "meta/llama-3.1-8b-instruct")

    # FUGU_CONDUCTOR は選抜より優先（プローブ結果に関わらず固定）
    _saved_ovr = f.CONDUCTOR_OVERRIDE
    try:
        f.CONDUCTOR_OVERRIDE = "meta/llama-3.3-70b-instruct"
        applied5, _log5 = _profile_with()
        check("選抜: FUGU_CONDUCTOR は選抜より優先される",
              applied5 is True
              and f.DESIRED_CONDUCTOR == "meta/llama-3.3-70b-instruct"
              and "FUGU_CONDUCTOR 指定" in _log5)
    finally:
        f.CONDUCTOR_OVERRIDE = _saved_ovr

    f._nim_catalog, f._nim_probe = _orig_cat, _orig_avail
    # キー未設定なら False
    f.NIM_API_KEY = ""
    with contextlib.redirect_stdout(io.StringIO()):
        _nokey = f.apply_nim_profile()
    check("profile: キー未設定は False（setup 中断）", _nokey is False)
    f.NIM_API_KEY = "nvapi-TEST-KEY"
finally:
    for k, v in _saved_profile.items():
        setattr(f, k, v)

# ---------- 12. SC_PARALLEL の投入順決定性 ----------
_saved_sc = {k: getattr(f, k) for k in
             ("SC_PARALLEL", "SC_WORKERS", "REASONING_MODELS", "PROPOSERS",
              "SC_CHEAP_VOTES", "SC_POT", "SC_INITIAL", "SC_MAX")}
_orig_sc_sample = f._sc_sample
try:
    def fake_sc(model, q, tt, pot=False, history=None):
        # 後に投入したジョブほど早く終わるよう逆順の遅延（完了順≠投入順を強制）
        delay = {"m/a": 0.05, "m/b": 0.03, "m/c": 0.01}[model]
        f_real_sleep(delay)
        return ("X", "text")
    import time as _t
    f_real_sleep = _t.sleep
    f._sc_sample = fake_sc
    f.SC_PARALLEL, f.SC_WORKERS = True, 4
    f.REASONING_MODELS = ["m/a", "m/b", "m/c"]
    f.PROPOSERS = ["m/a", "m/b", "m/c"]
    f.SC_CHEAP_VOTES, f.SC_POT = 0, True
    f.SC_INITIAL, f.SC_MAX = 6, 6
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    lines = [l for l in buf.getvalue().splitlines() if l.strip().startswith("[SC ")]
    order = [l.split("]")[1].split("(")[0].strip() for l in lines]
    check("sc-par: 完了順に依らず投入順で記録される",
          order == ["m/a", "m/a", "m/b", "m/b", "m/c", "m/c", "m/a"])
    check("sc-par: PoT は末尾で先頭モデル", "(PoT)" in lines[-1] and lines[-1].split("]")[1].split("(")[0].strip() == "m/a")
    check("sc-par: 投票は通常どおり成立", res is not None and res["answer"] == "X")
    # 2026-08-03: REASONING_MODELS のフィルタは PROPOSERS 外でも NIM レジストリ登録済み
    # なら通す（minimax-m3 第3系統が黙って脱落し SC が2系統で回っていた実測バグの回帰）
    used = set()
    def fake_sc2(model, q, tt, pot=False, history=None):
        used.add(model)
        return ("Y", "t")
    f._sc_sample = fake_sc2
    f.REASONING_MODELS = ["m/a", "nim/only"]
    f.PROPOSERS = ["m/a"]
    _saved_ids = set(f.NIM_MODEL_IDS)
    f.NIM_MODEL_IDS = {"nim/only"}
    with contextlib.redirect_stdout(io.StringIO()):
        f.solve_verifiable("dummy", "math")
    f.NIM_MODEL_IDS = _saved_ids
    check("sc-filter: NIMレジスタ登録モデルはPROPOSERS外でもSC系統に参加", "nim/only" in used)
finally:
    f._sc_sample = _orig_sc_sample
    for k, v in _saved_sc.items():
        setattr(f, k, v)

# ---------- 12b. DeepConf: 自信度計算・logprobs収集・加重投票 (2026-08-04) ----------
# _deepconf_confidence: 最悪区間を見る(全体平均では隠れる崩れ区間を検出)
_good = [-0.05] * 3000
_bad_middle = [-0.05] * 1400 + [-5.0] * 200 + [-0.05] * 1400
check("deepconf: 均一な高自信トレースは0に近い",
      f._deepconf_confidence(_good) > -0.1)
check("deepconf: 途中で崩れたトレースは最悪窓が捉えて大きく下がる",
      f._deepconf_confidence(_bad_middle) < f._deepconf_confidence(_good) - 0.1)
check("deepconf: 空はNone", f._deepconf_confidence([]) is None)
check("deepconf: 窓幅未満の短いトレースも動く",
      isinstance(f._deepconf_confidence([-0.5] * 10), float))

# SSE に logprobs を混ぜた収集: NIM_CAPTURE_LOGPROBS=True で payload に logprobs:true、
# TLS にトレース自信度が入る
def sse_with_logprobs(content, lps):
    lines = []
    lines.append(("data: " + json.dumps(
        {"choices": [{"delta": {"content": content},
                      "logprobs": {"content": [{"token": "x", "logprob": v} for v in lps]}}]}
    ) + "\n").encode())
    lines.append(("data: " + json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}) + "\n").encode())
    lines.append(b"data: [DONE]\n")
    return FakeResponse(sse_lines=lines)

f.NIM_CAPTURE_LOGPROBS = True
out, sent, _ = run_mocked([sse_with_logprobs("42", [-0.1, -0.2, -0.3])],
                          lambda: f.ask("test/model", MSGS, 0.5))
check("deepconf: capture ON で payload に logprobs:true",
      payload_of(sent[0]).get("logprobs") is True)
check("deepconf: 応答は従来どおり content", out == "42")
check("deepconf: TLS にトレース自信度が入る",
      isinstance(getattr(f._NIM_TLS, "last_conf", None), float))
f.NIM_CAPTURE_LOGPROBS = False
out, sent, _ = run_mocked([(ok_body("plain"))], lambda: f.ask("test/model", MSGS, 0.5))
check("deepconf: capture OFF では logprobs を送らない(既定不変)",
      "logprobs" not in payload_of(sent[0]))

# _conf_weighted_vote: 多数派だが低自信 vs 少数派だが高自信 → 加重が勝者を反転
def _mk(ans, conf, pot=False):
    return {"answer": ans, "text": "t", "model": "m", "pot": pot, "conf": conf}
_samples = ([_mk("1", -2.0)] * 4 + [_mk("2", -0.05)] * 3)
wtop, wcl = f._conf_weighted_vote(_samples)
check("deepconf: 低自信の多数派より高自信の少数派が勝つ", wtop == "2")
check("deepconf: 下位η除外で低自信票が集約から消える",
      all(c[0] != "1" or c[2] < 4 for c in wcl))
wtop2, _ = f._conf_weighted_vote([_mk("1", -0.5)] * 3)   # 有効票3 < 発動下限4
check("deepconf: 有効conf票4未満は発動しない(素の多数決へ委譲)", wtop2 is None)
wtop3, _ = f._conf_weighted_vote([_mk("1", None)] * 8 + [_mk("2", -0.1, pot=True)] * 2)
check("deepconf: conf無し/PoT票のみでは発動しない", wtop3 is None)

# solve_verifiable 統合: fake _sc_sample が TLS 経由で conf を渡し、加重が勝者を変える
_saved_sc4 = {k: getattr(f, k) for k in
              ("SC_PARALLEL", "SC_WORKERS", "REASONING_MODELS", "PROPOSERS",
               "SC_CHEAP_VOTES", "SC_POT", "SC_INITIAL", "SC_STEP", "SC_MAX",
               "SC_CONF_VOTE")}
_orig_sc_sample2 = f._sc_sample
try:
    _call_no = [0]
    def fake_sc_conf(model, q, tt, pot=False, history=None):
        _call_no[0] += 1
        # 9票: 先の5票は低自信で「7」、後の4票は高自信で「11」(正答想定)
        if _call_no[0] <= 5:
            f._NIM_TLS.last_conf = -3.0
            return ("7", "low-conf")
        f._NIM_TLS.last_conf = -0.02
        return ("11", "high-conf")
    f._sc_sample = fake_sc_conf
    f.SC_PARALLEL = False
    f.REASONING_MODELS = ["m/a"]; f.PROPOSERS = ["m/a"]
    f.SC_CHEAP_VOTES, f.SC_POT = 0, False
    f.SC_INITIAL, f.SC_STEP, f.SC_MAX = 9, 4, 9
    f.SC_CONF_VOTE = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    check("deepconf: solve_verifiable で加重が勝者を変更(7多数→11高自信)",
          res is not None and res["answer"] == "11")
    check("deepconf: 勝者変更がログに出る", "自信度加重が勝者を変更" in buf.getvalue())
    # 既定OFF: 同条件でも素の多数決のまま
    _call_no[0] = 0
    f.SC_CONF_VOTE = False
    with contextlib.redirect_stdout(io.StringIO()):
        res2 = f.solve_verifiable("dummy", "math")
    check("deepconf: SC_CONF_VOTE=False なら従来どおり多数決(7)",
          res2 is not None and res2["answer"] == "7")
finally:
    f._sc_sample = _orig_sc_sample2
    for k, v in _saved_sc4.items():
        setattr(f, k, v)

# ---------- 12c. SC-Court 対審集約 + 抽出失敗票救済 (2026-08-05) ----------
_saved_court = {k: getattr(f, k) for k in
                ("SC_COURT", "SC_RESCUE_VOTES", "SC_COURT_JUDGES", "SC_COURT_TOPK",
                 "SC_COURT_DEVILS", "SC_COURT_MARGIN", "ARBITER_MODEL",
                 "REASONING_MODELS", "PROPOSERS", "SC_PARALLEL", "SC_CHEAP_VOTES",
                 "SC_POT", "SC_INITIAL", "SC_STEP", "SC_MAX", "CONDUCTOR",
                 "SC_RESCUE_MODEL")}
_orig_sc_sample3 = f._sc_sample
_orig_ask = f.ask
_orig_installed = f.installed_models
try:
    f.installed_models = lambda: []
    f.ARBITER_MODEL = None
    f.REASONING_MODELS = ["m/a", "m/b", "m/c"]
    f.PROPOSERS = ["m/a", "m/b", "m/c"]
    f.SC_PARALLEL = False
    f.SC_CHEAP_VOTES, f.SC_POT = 0, False
    f.SC_COURT_JUDGES, f.SC_COURT_TOPK, f.SC_COURT_DEVILS = 3, 3, 3
    f.SC_COURT_MARGIN = 2.0
    f.CONDUCTOR, f.SC_RESCUE_MODEL = "m/cond", None

    # --- _rescue_vote: 幻覚ガード ---
    _long_hit = ("thinking " * 60) + " therefore the count is 248 and then it was cut"
    _long_miss = ("thinking " * 60) + " nothing conclusive here at all, just musing on"
    f.ask = lambda *a, **kw: "\\boxed{248}"
    check("rescue: トレースに実在する答えは回収される",
          f._rescue_vote(_long_hit, "math") == "248")
    check("rescue: トレースに無い数値は捏造とみなし棄却",
          f._rescue_vote(_long_miss, "math") is None)
    f.ask = lambda *a, **kw: "\\boxed{NONE}"
    check("rescue: NONE は棄却", f._rescue_vote(_long_hit, "math") is None)
    check("rescue: 短すぎるトレースは呼ばず棄却", f._rescue_vote("short 248", "math") is None)

    # --- 共通の fake _sc_sample ビルダー: 事前に並べた票列を順に返す(枯渇後は無効票) ---
    def _mk_sampler(seq, devil_ans=None):
        state = {"i": 0, "devil": 0, "court_asked": []}
        def fake(model, q, tt, pot=False, history=None):
            if "[Verification note" in q:                      # 悪魔の代弁人ラウンド
                a = devil_ans[state["devil"] % len(devil_ans)] if devil_ans else None
                state["devil"] += 1
                return (a, f"devil-text-{a}")
            a = seq[state["i"]] if state["i"] < len(seq) else None
            state["i"] += 1
            return (a, f"trace-{a}-" + "x" * 40)
        return fake, state

    def _mk_judge_ask(verdicts):
        calls = {"court": 0, "labels": []}
        def fake_ask(model, messages, temp=0.5, **kw):
            lbl = kw.get("label")
            calls["labels"].append(lbl)
            if lbl == "court":
                v = verdicts[calls["court"] % len(verdicts)]
                calls["court"] += 1
                return f"analysis...\nVERDICT: {v}"
            return "\\boxed{NONE}"                              # rescue等は不発
        return fake_ask, calls

    # --- court: 散逸した少数票の正解候補を判決で確定 (aime25-27 シナリオ) ---
    f.SC_COURT, f.SC_RESCUE_VOTES = True, False
    f.SC_INITIAL, f.SC_STEP, f.SC_MAX = 8, 4, 8
    fake, st = _mk_sampler(["248", "248", "337", "611", None, None, None, None])
    f._sc_sample = fake
    f.ask, jcalls = _mk_judge_ask(["A", "A", "NONE"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    check("court: 床未満(2票)の散逸でも判決過半数で確定",
          res is not None and res["answer"] == "248")
    check("court: 判決確定がログに出る", "[SC/court] 判決確定: 248" in buf.getvalue())

    # --- court: 誤った多数派を少数票の正解へ覆す ---
    fake, st = _mk_sampler(["7", "7", "7", "385", "385", None, None, None])
    f._sc_sample = fake
    f.ask, jcalls = _mk_judge_ask(["B", "B", "A"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("court: 判決過半数が誤った多数派(7x3)を覆し385を採用",
          res is not None and res["answer"] == "385")

    # --- court: 多数決が明確に強い時は審理しない(margin skip) ---
    fake, st = _mk_sampler(["7", "7", "7", "7", "385", None, None, None])
    f._sc_sample = fake
    f.ask, jcalls = _mk_judge_ask(["B"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("court: 4票vs1票(margin 2.0)は審理省略で多数決のまま",
          res is not None and res["answer"] == "7" and jcalls["court"] == 0)

    # --- court: 全候補 FLAWED → 悪魔の代弁人が新答で一致 → 採用 (合意型誤答の攻略) ---
    fake, st = _mk_sampler(["16", "16", "1740", "1740", None, None, None, None],
                           devil_ans=["83", "83", "51"])
    f._sc_sample = fake
    f.ask, jcalls = _mk_judge_ask(["NONE", "NONE", "NONE"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    check("court: 全候補FLAWED→悪魔の代弁人の一致新答83を採用",
          res is not None and res["answer"] == "83")
    check("court: 悪魔の代弁人ラウンドがログに出る",
          "悪魔の代弁人ラウンド" in buf.getvalue())

    # --- court v2: 評決不成立(hung jury)でも悪魔の代弁人が発動、新答不一致なら従来どおり None ---
    fake, st = _mk_sampler(["248", "248", "337", "611", None, None, None, None],
                           devil_ans=["901", "777", "555"])   # 一致なし → 採用されない
    f._sc_sample = fake
    f.ask, jcalls = _mk_judge_ask(["A", "B", "NONE"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    check("court-v2: 評決不成立で悪魔の代弁人が発動する", "評決不成立" in buf.getvalue())
    check("court-v2: 悪魔の新答が不一致なら従来どおり床未満でNone", res is None)

    # --- court v2: 評決不成立 + 悪魔の一致新答 → 採用 ---
    fake, st = _mk_sampler(["248", "248", "337", "611", None, None, None, None],
                           devil_ans=["901", "901", "555"])
    f._sc_sample = fake
    f.ask, jcalls = _mk_judge_ask(["A", "B", "NONE"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("court-v2: 評決不成立でも悪魔の一致新答901を採用",
          res is not None and res["answer"] == "901")

    # --- court v2: 棄権は枠を消費せず補充裁判官が審理する ---
    f.REASONING_MODELS = ["m/a", "m/b", "m/c", "m/d"]
    f.PROPOSERS = ["m/a", "m/b", "m/c", "m/d"]
    fake, st = _mk_sampler(["248", "248", "337", "611", None, None, None, None])
    f._sc_sample = fake
    f.ask, jcalls = _mk_judge_ask(["GARBAGE", "A", "A", "NONE"])  # 1人目は書式不正=棄権
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    check("court-v2: 棄権1名を補充し有効判決3件で248確定",
          res is not None and res["answer"] == "248" and jcalls["court"] == 4)
    check("court-v2: 補充のログが出る", "補充裁判官へ" in buf.getvalue())
    f.REASONING_MODELS = ["m/a", "m/b", "m/c"]
    f.PROPOSERS = ["m/a", "m/b", "m/c"]

    # --- court v2: 相対的に強いだけの少数派多数(真過半数未満)は審理にかける ---
    fake, st = _mk_sampler(["7", "7", "7", "8", "9", "10", "11", None])
    f._sc_sample = fake
    f.ask, jcalls = _mk_judge_ask(["A", "A", "A"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("court-v2: 3/7票(真過半数未満)は margin を満たしても審理される",
          res is not None and res["answer"] == "7" and jcalls["court"] >= 1)

    # --- S3C: 戦略列挙のパース (JSON / 番号付き行 / 失敗) ---
    f.ARBITER_MODEL = "m/judge"
    f.installed_models = lambda: ["m/judge"]
    _orig_is_installed = f.is_installed
    f.is_installed = lambda m, inst=None: True
    f.ask = lambda *a, **kw: 'plan... ["casework on loop sizes", "generating functions", "transfer matrix"]'
    _strats = f._enumerate_strategies("q")
    check("s3c: JSON配列の戦略列挙をパース", _strats == ["casework on loop sizes",
          "generating functions", "transfer matrix"])
    f.ask = lambda *a, **kw: "1. direct casework\n2) complementary counting\nnot a strategy line"
    _strats2 = f._enumerate_strategies("q")
    check("s3c: 番号付き行フォールバック",
          _strats2 == ["direct casework", "complementary counting"])
    f.ask = lambda *a, **kw: "__ERROR__: down"
    check("s3c: 列挙失敗は空リスト(従来サンプリングへ劣化)", f._enumerate_strategies("q") == [])
    f.is_installed = _orig_is_installed
    f.ARBITER_MODEL = None
    f.installed_models = lambda: []

    # --- S3C: 各CoTサンプルに戦略指令がラウンドロビンで注入される ---
    f.SC_STRATIFY, f.SC_COURT, f.SC_RESCUE_VOTES = True, False, False
    _seen_q = []
    def fake_strat_sampler(model, q, tt, pot=False, history=None):
        _seen_q.append((q, pot))
        return ("5", "trace")
    f._sc_sample = fake_strat_sampler
    def fake_strat_ask(model, messages, temp=0.5, **kw):
        return '["strategy ONE", "strategy TWO"]'
    f.ask = fake_strat_ask
    f.SC_INITIAL, f.SC_STEP, f.SC_MAX = 4, 4, 4
    f.SC_POT = True
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("base-question", "math")
    _cot_qs = [q for q, pot in _seen_q if not pot]
    _pot_qs = [q for q, pot in _seen_q if pot]
    check("s3c: 全CoTサンプルに戦略指令が入る",
          _cot_qs and all("[Strategy directive:" in q for q in _cot_qs))
    check("s3c: 2戦略がラウンドロビンで両方使われる",
          any("strategy ONE" in q for q in _cot_qs) and any("strategy TWO" in q for q in _cot_qs))
    check("s3c: PoTサンプルは層化しない",
          _pot_qs and all("[Strategy directive:" not in q for q in _pot_qs))
    check("s3c: 投票は通常どおり成立", res is not None and res["answer"] == "5")
    f.SC_STRATIFY = False
    f.SC_POT = False

    # --- 既定OFF: 同条件でも従来挙動(床未満→None、courtは呼ばれない) ---
    f.SC_COURT = False
    fake, st = _mk_sampler(["248", "248", "337", "611", None, None, None, None])
    f._sc_sample = fake
    f.ask, jcalls = _mk_judge_ask(["A", "A", "A"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("court: SC_COURT=False は完全に従来挙動(None, 審理ゼロ)",
          res is None and jcalls["court"] == 0)

    # --- 境界監査 (sc8): 全員一致の修正のみ採用 ---
    f.SC_COURT, f.SC_RESCUE_VOTES, f.SC_AUDIT = True, False, True
    def _mk_audit_ask(verdicts, audits):
        calls = {"court": 0, "audit": 0}
        def fake_ask(model, messages, temp=0.5, **kw):
            lbl = kw.get("label")
            if lbl == "court":
                v = verdicts[calls["court"] % len(verdicts)]
                calls["court"] += 1
                return f"...\nVERDICT: {v}"
            if lbl == "audit":
                a = audits[calls["audit"] % len(audits)]
                calls["audit"] += 1
                return f"...\nAUDIT: {a}"
            return "\\boxed{NONE}"
        return fake_ask, calls
    # 悪魔の一致新答384を監査全員一致で385へ修正 (aime24-I-12 シナリオ)
    fake, st = _mk_sampler(["8", "8", "16", "256", None, None, None, None],
                           devil_ans=["384", "384", "999"])
    f._sc_sample = fake
    f.ask, acalls = _mk_audit_ask(["NONE", "NONE", "NONE"],
                                  ["CORRECTED 385", "CORRECTED 385"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    check("audit: 悪魔の一致新答384を全員一致監査で385へ修正採用",
          res is not None and res["answer"] == "385")
    check("audit: 修正採用がログに出る", "監査全員一致の修正を採用: 384 -> 385" in buf.getvalue())
    # 監査が割れたら原案維持
    fake, st = _mk_sampler(["8", "8", "16", "256", None, None, None, None],
                           devil_ans=["384", "384", "999"])
    f._sc_sample = fake
    f.ask, acalls = _mk_audit_ask(["NONE", "NONE", "NONE"],
                                  ["CORRECTED 385", "CONFIRMED 384"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("audit: 監査が割れたら原案維持(384)", res is not None and res["answer"] == "384")
    # 低票の判決勝者も監査対象 (aime26-83 型が誤修正されない: 全員CONFIRMED)
    fake, st = _mk_sampler(["83", "83", "1740", "16", None, None, None, None])
    f._sc_sample = fake
    f.ask, acalls = _mk_audit_ask(["A", "A", "NONE"], ["CONFIRMED 83", "CONFIRMED 83"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("audit: 判決勝者(低票)は監査を通りCONFIRMEDで維持",
          res is not None and res["answer"] == "83" and acalls["audit"] == 2)
    # SC_AUDIT=False では監査ゼロ (sc7互換)
    f.SC_AUDIT = False
    fake, st = _mk_sampler(["83", "83", "1740", "16", None, None, None, None])
    f._sc_sample = fake
    f.ask, acalls = _mk_audit_ask(["A", "A", "NONE"], ["CORRECTED 999"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("audit: SC_AUDIT=Falseは監査ゼロで従来どおり",
          res is not None and res["answer"] == "83" and acalls["audit"] == 0)

    # --- 挑戦者制度 (sc9): 現職追認でも悪魔が挑戦し決選審理で決める ---
    f.SC_COURT, f.SC_RESCUE_VOTES, f.SC_AUDIT, f.SC_CHALLENGER = True, False, True, True
    f.SC_INITIAL, f.SC_STEP, f.SC_MAX = 8, 4, 8
    def _mk_ch_ask(verdicts, runoffs, audits):
        calls = {"court": 0, "runoff": 0, "audit": 0}
        def fake_ask(model, messages, temp=0.5, **kw):
            lbl = kw.get("label")
            if lbl == "court":
                v = verdicts[calls["court"] % len(verdicts)]; calls["court"] += 1
                return f"...\nVERDICT: {v}"
            if lbl == "runoff":
                v = runoffs[calls["runoff"] % len(runoffs)]; calls["runoff"] += 1
                return f"...\nVERDICT: {v}"
            if lbl == "audit":
                a = audits[calls["audit"] % len(audits)]; calls["audit"] += 1
                return f"...\nAUDIT: {a}"
            return "\\boxed{NONE}"
        return fake_ask, calls
    # 現職24(判決2/3)に悪魔合意384が挑戦→決選でB勝利→監査で385へ修正 (I-12完全シナリオ)
    fake, st = _mk_sampler(["24", "24", "24", "16", "16", "8", None, None],
                           devil_ans=["384", "384", "999"])
    f._sc_sample = fake
    f.ask, ccalls = _mk_ch_ask(["A", "A", "NONE"], ["B", "B", "A"],
                               ["CORRECTED 385", "CORRECTED 385"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    check("challenger: 現職24を悪魔384が決選で破り監査が385へ修正",
          res is not None and res["answer"] == "385")
    check("challenger: 挑戦・決選のログが出る",
          "挑戦者制度: 現職 24 への挑戦" in buf.getvalue()
          and "挑戦者の勝利" in buf.getvalue())
    # 決選で現職勝利なら現職維持(監査は生票多数なのでスキップ)
    fake, st = _mk_sampler(["24", "24", "24", "16", "16", "8", None, None],
                           devil_ans=["384", "384", "999"])
    f._sc_sample = fake
    f.ask, ccalls = _mk_ch_ask(["A", "A", "NONE"], ["A", "B", "A"], ["CORRECTED 1"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("challenger: 決選で現職勝利なら24を維持(監査スキップ)",
          res is not None and res["answer"] == "24" and ccalls["audit"] == 0)
    # 悪魔が合意しなければ現職を従来どおり確定
    fake, st = _mk_sampler(["24", "24", "24", "16", "16", "8", None, None],
                           devil_ans=["384", "999", "555"])
    f._sc_sample = fake
    f.ask, ccalls = _mk_ch_ask(["A", "A", "NONE"], ["B"], ["CORRECTED 1"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("challenger: 悪魔不一致なら現職24を確定(決選なし)",
          res is not None and res["answer"] == "24" and ccalls["runoff"] == 0)
    f.SC_CHALLENGER = False

    # --- 機械検証官 (sc10): 独立プログラム2本の実行一致が全てに優先 ---
    f.SC_COURT, f.SC_RESCUE_VOTES, f.SC_AUDIT, f.SC_CHALLENGER = True, False, False, False
    f.SC_MACHINE = True
    def _mk_machine_ask(verdicts, codes):
        calls = {"court": 0, "machine": 0}
        def fake_ask(model, messages, temp=0.5, **kw):
            lbl = kw.get("label")
            if lbl == "court":
                v = verdicts[calls["court"] % len(verdicts)]; calls["court"] += 1
                return f"...\nVERDICT: {v}"
            if lbl == "machine":
                c = codes[calls["machine"] % len(codes)]; calls["machine"] += 1
                return f"reasoning...\n```python\n{c}\n```"
            return "\\boxed{NONE}"
        return fake_ask, calls
    # 現候補384(悪魔不要: 審理が候補Aを支持)を機械合意385が上書き (I-12 の狙い撃ちシナリオ)
    fake, st = _mk_sampler(["384", "384", "384", "16", "16", "8", None, None])
    f._sc_sample = fake
    f.ask, mcalls = _mk_machine_ask(["A", "A", "A"],
                                    ["print(385)", "print(385)", "print(999)"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    check("machine: プログラム2本の実行一致385が言語側の384を上書き",
          res is not None and res["answer"] == "385")
    check("machine: 機械合意ログが出る", "実行可能な合意: 385" in buf.getvalue())
    # プログラム間の合意なし → 言語側の結論を維持
    fake, st = _mk_sampler(["384", "384", "384", "16", "16", "8", None, None])
    f._sc_sample = fake
    f.ask, mcalls = _mk_machine_ask(["A", "A", "A"],
                                    ["print(1)", "print(2)", "print(3)"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("machine: 合意なしは言語側の384を維持", res is not None and res["answer"] == "384")
    # 機械合意が現候補と同値なら確定を後押し(床バイパス)
    fake, st = _mk_sampler(["248", "248", "337", "611", None, None, None, None])
    f._sc_sample = fake
    f.ask, mcalls = _mk_machine_ask(["NONE", "NONE", "NONE"],
                                    ["print(248)", "print(248)", "print(248)"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("machine: 機械合意248が床未満(2票)でも確定させる",
          res is not None and res["answer"] == "248")
    # SC_MACHINE=False は不変
    f.SC_MACHINE = False
    fake, st = _mk_sampler(["384", "384", "384", "16", "16", "8", None, None])
    f._sc_sample = fake
    f.ask, mcalls = _mk_machine_ask(["A", "A", "A"], ["print(385)"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("machine: SC_MACHINE=Falseは機械呼び出しゼロで従来どおり",
          res is not None and res["answer"] == "384" and mcalls["machine"] == 0)

    # --- 機械裁定 v2 (sc11): 生成失敗の補充 + コード審査で決着 (2026-08-06) ---
    f.SC_COURT, f.SC_RESCUE_VOTES, f.SC_AUDIT, f.SC_CHALLENGER = True, False, False, False
    f.SC_MACHINE, f.SC_MACHINE_ARBITRATE = True, True
    _saved_mn = (f.SC_MACHINE_N, f.SC_MACHINE_MAX_TRIES)
    f.SC_MACHINE_N, f.SC_MACHINE_MAX_TRIES = 3, 6
    def _mk_m2_ask(verdicts, gens, reviews):
        calls = {"court": 0, "machine": 0, "review": 0}
        def fake_ask(model, messages, temp=0.5, **kw):
            lbl = kw.get("label")
            if lbl == "court":
                v = verdicts[calls["court"] % len(verdicts)]; calls["court"] += 1
                return f"...\nVERDICT: {v}"
            if lbl == "machine":
                g = gens[calls["machine"]] if calls["machine"] < len(gens) else "__ERROR__: x"
                calls["machine"] += 1
                return g if g.startswith("__ERROR__") else f"```python\n{g}\n```"
            if lbl == "machine-review":
                r = reviews[calls["review"] % len(reviews)]; calls["review"] += 1
                return f"...\nPROGRAM: {r}"
            return "\\boxed{NONE}"
        return fake_ask, calls
    # I-12 完全再現: 言語側24が確定 → プログラムは 1 / 385 で割れる → コード審査で385
    fake, st = _mk_sampler(["24", "24", "24", "16", "16", "8", None, None])
    f._sc_sample = fake
    # 実行に成功するのは 2 本(出力 1 と 385)なので、審査のラベルは A=1 / B=385
    f.ask, m2 = _mk_m2_ask(["A", "A", "A"],
                           ["print(1)", "__ERROR__: empty", "print(385)", "__ERROR__: empty"],
                           ["B", "B", "A"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    _log = buf.getvalue()
    check("machine-v2: 生成失敗を補充して実行成功本を集める", "補充へ" in _log)
    check("machine-v2: 割れた出力をコード審査で決着させ385を採用",
          res is not None and res["answer"] == "385")
    check("machine-v2: コード審査の決着がログに出る", "コード審査で決着" in _log)
    # 2本一致するなら従来どおりコード審査は不要
    fake, st = _mk_sampler(["24", "24", "24", "16", "16", "8", None, None])
    f._sc_sample = fake
    f.ask, m2 = _mk_m2_ask(["A", "A", "A"], ["print(385)", "print(385)"], ["A"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("machine-v2: 2本一致なら従来経路で採用(コード審査は呼ばない)",
          res is not None and res["answer"] == "385" and m2["review"] == 0)
    # 審査が過半数に達しなければ棄権して言語側を維持
    fake, st = _mk_sampler(["24", "24", "24", "16", "16", "8", None, None])
    f._sc_sample = fake
    f.ask, m2 = _mk_m2_ask(["A", "A", "A"], ["print(1)", "print(385)", "print(7)"],
                           ["A", "B", "C"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("machine-v2: 審査が割れたら棄権して言語側24を維持",
          res is not None and res["answer"] == "24")
    # ARBITRATE=False は sc10 の挙動のまま(棄権)
    f.SC_MACHINE_ARBITRATE = False
    fake, st = _mk_sampler(["24", "24", "24", "16", "16", "8", None, None])
    f._sc_sample = fake
    f.ask, m2 = _mk_m2_ask(["A", "A", "A"], ["print(1)", "print(385)"], ["B"])
    with contextlib.redirect_stdout(io.StringIO()):
        res = f.solve_verifiable("dummy", "math")
    check("machine-v2: ARBITRATE=False は sc10 と同じく棄権(24のまま)",
          res is not None and res["answer"] == "24" and m2["review"] == 0)
    f.SC_MACHINE, f.SC_MACHINE_ARBITRATE = False, False
    f.SC_MACHINE_N, f.SC_MACHINE_MAX_TRIES = _saved_mn

    # --- rescue 統合: 抽出失敗2票を回収して全会一致で確定 ---
    f.SC_COURT, f.SC_RESCUE_VOTES = False, True
    _t248 = ("reasoning " * 40) + " so the answer is 248 clearly but"
    def fake_rescue_seq(model, q, tt, pot=False, history=None):
        seq = [("248", "trace-248"), ("248", "trace-248"),
               (None, _t248), (None, _t248),
               (None, "__ERROR__: dead"), (None, "short"),
               (None, _t248.replace("248", "999")), (None, "short2")]
        i = fake_rescue_seq.i; fake_rescue_seq.i += 1
        return seq[i] if i < len(seq) else (None, "exhausted")
    fake_rescue_seq.i = 0
    f._sc_sample = fake_rescue_seq
    f.ask = lambda *a, **kw: "\\boxed{248}"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f.solve_verifiable("dummy", "math")
    check("rescue: 回収2票で 248 が 2→4票になり確定",
          res is not None and res["answer"] == "248"
          and res["votes"].get("248") == 4)
    check("rescue: 回収ログが出る", "[SC/rescue] 抽出失敗トレースから 2 票を回収" in buf.getvalue())
finally:
    f._sc_sample = _orig_sc_sample3
    f.ask = _orig_ask
    f.installed_models = _orig_installed
    for k, v in _saved_court.items():
        setattr(f, k, v)

# ---------- 12d. 統合 MoA (UNIFIED_MOA_SPEC §7, 2026-08-05) ----------
_saved_u = {k: getattr(f, k) for k in
            ("FUGU_UNIFIED", "FUGU_DECOMPOSE", "FUGU_HIERARCHICAL", "FUGU_RESIDUAL",
             "FUGU_EXEC_KEY", "FUGU_EARLY_UNANIMOUS", "POOL_MIN_KEYS",
             "DECOMP_VOTES", "DECOMP_MAX_DEPTH", "ARBITER_MODEL",
             "REASONING_MODELS", "PROPOSERS", "SC_PARALLEL", "SC_WORKERS",
             "SC_STRATIFY", "SC_MIN_VOTES")}
_orig_ask_u = f.ask
_orig_agg_u = f.aggregate
_orig_installed_u = f.installed_models
try:
    # (13) 最重要: 既定でマスタスイッチ OFF(既存経路ビット同一の前提)
    check("unified: FUGU_UNIFIED は既定 False", _saved_u["FUGU_UNIFIED"] is False)

    # (1)(2) extract_key の優先順位と None
    check("unified: math キー発火", f.extract_key("thus \\boxed{42}") == ("math", "42"))
    _mcq = f.extract_key("After comparing all options, the correct choice is \\boxed{C}")
    check("unified: mcq キー発火", _mcq is not None and _mcq[0] == "mcq" and _mcq[1].upper() == "C")
    check("unified: キー無しは None", f.extract_key("no conclusion drawn here at all") is None)

    # (12) exec キー: 実装の異なる2コードが同出力なら同一キー
    _q_exec = "Implement add(a, b).\n>>> add(2, 3)\n>>> add(10, -4)"
    _c1 = "answer:\n```python\ndef add(a, b):\n    return a + b\n```"
    _c2 = "another:\n```python\ndef add(x, y):\n    s = x\n    s += y\n    return s\n```"
    _c3 = "wrong:\n```python\ndef add(a, b):\n    return a - b\n```"
    f.FUGU_EXEC_KEY = True
    _k1 = f.extract_key(_c1, _q_exec)
    _k2 = f.extract_key(_c2, _q_exec)
    _k3 = f.extract_key(_c3, _q_exec)
    check("unified: exec キー発火(kind=exec)", _k1 is not None and _k1[0] == "exec")
    check("unified: 実装違い同出力→同一キー", _k1 == _k2)
    check("unified: 挙動が違えば別キー", _k3 is not None and _k3[1] != _k1[1])

    # (3)(4) pool: ハード/ソフトの落ち方と確定条件(既存SCと同一)
    f.aggregate = lambda q, props: "SOFT-MERGED"
    f.POOL_MIN_KEYS, f.SC_MIN_VOTES = 3, 3
    _r = f.pool("q", ["\\boxed{7}", "\\boxed{7}", "\\boxed{7}"])
    check("unified: 有効キー3でハード・全会一致n>=3で確定",
          _r.kind == "hard" and _r.confirmed and _r.key == "7")
    _r = f.pool("q", ["\\boxed{7}", "\\boxed{7}", "no key here at all"])
    check("unified: 有効キー2はソフトへ(aggregate統合・confirmed)",
          _r.kind == "soft" and _r.confirmed and _r.answer == "SOFT-MERGED")
    _r = f.pool("q", ["\\boxed{7}", "\\boxed{7}", "\\boxed{7}", "\\boxed{9}"])
    check("unified: n>=4かつ過半数(3/4)で確定", _r.kind == "hard" and _r.confirmed)
    _r = f.pool("q", ["\\boxed{7}", "\\boxed{7}", "\\boxed{9}", "\\boxed{9}"])
    check("unified: 2-2同数は未確定", _r.kind == "hard" and not _r.confirmed)

    # (5) kind 混在: 多数派 kind のみ有効票
    _mix = ["\\boxed{7}", "\\boxed{7}", "\\boxed{7}", _c1, _c2]
    _r = f.pool(_q_exec, _mix)
    check("unified: kind混在は多数派(math)のみ有効票",
          _r.kind == "hard" and _r.n_valid == 3 and _r.key == "7")

    # (6)(7)(8) decompose: 縮退・深さ上限・分解投票
    f.installed_models = lambda: []
    f.ARBITER_MODEL = None
    f.REASONING_MODELS = ["m/a", "m/b", "m/c"]
    f.PROPOSERS = ["m/a", "m/b", "m/c"]
    f.DECOMP_VOTES, f.DECOMP_MAX_DEPTH = 3, 2
    _dec_calls = [0]
    def fake_decomp_ask(model, messages, temp=0.5, **kw):
        if kw.get("label") == "decomp":
            _dec_calls[0] += 1
            plans = ['{"steps": [{"goal": "A"}, {"goal": "B"}, {"goal": "C"}]}',
                     '{"steps": [{"goal": "X"}, {"goal": "Y"}]}',
                     '{"steps": [{"goal": "P"}, {"goal": "Q"}, {"goal": "R"}]}']
            return plans[(_dec_calls[0] - 1) % 3]
        return "\\boxed{5}"
    f.ask = fake_decomp_ask
    _steps = f.decompose("problem")
    check("unified: 分解投票で段数最頻(3段)を採用",
          len(_steps) == 3 and _steps[0]["goal"] == "A")
    check("unified: 深さ上限で分解しない",
          f.decompose("problem", depth=2) == [{"goal": "problem"}])
    f.ask = lambda *a, **kw: '{"steps": [{"goal": "only-one"}]}'
    check("unified: 2段未満は縮退", f.decompose("problem") == [{"goal": "problem"}])
    f.ask = lambda *a, **kw: "__ERROR__: down"
    check("unified: 分解失敗も縮退", f.decompose("problem") == [{"goal": "problem"}])

    # (9)(10) sample_step: 残差結合と因果性
    _seen_u = []
    def fake_step_ask(model, messages, temp=0.5, **kw):
        _seen_u.append(messages[-1]["content"])
        return "work...\nCONCLUSION: done \\boxed{1}"
    f.ask = fake_step_ask
    f.SC_PARALLEL, f.SC_STRATIFY = False, False
    f.FUGU_RESIDUAL = True
    f.sample_step({"goal": "goal-K"}, "ORIGINAL-Q", ["c1-done", "c2-done"], 1)
    check("unified: 残差結合で原問題がプロンプトに入る", "ORIGINAL-Q" in _seen_u[-1])
    check("unified: 因果文脈(段1..k-1)が入り、段kのゴール以外の未来情報が無い",
          "c1-done" in _seen_u[-1] and "c2-done" in _seen_u[-1]
          and "goal-K" in _seen_u[-1])
    f.FUGU_RESIDUAL = False
    f.sample_step({"goal": "goal-K"}, "ORIGINAL-Q", [], 1)
    check("unified: FUGU_RESIDUAL=False では原問題を含めない",
          "ORIGINAL-Q" not in _seen_u[-1])
    f.FUGU_RESIDUAL = True

    # (11) 並列時の回収が投入順で決定的
    import time as _t_u
    _delays = {0: 0.05, 1: 0.03, 2: 0.01}
    _ncall = [0]
    def fake_par_ask(model, messages, temp=0.5, **kw):
        i = _ncall[0]; _ncall[0] += 1
        _t_u.sleep(_delays.get(i % 3, 0))
        return f"CONCLUSION: \\boxed{{{i}}}"
    f.ask = fake_par_ask
    f.SC_PARALLEL, f.SC_WORKERS = True, 4
    _outs = f.sample_step({"goal": "g"}, "q", [], 3)
    check("unified: 並列でも回収は投入順",
          [f.extract_key(o)[1] for o in _outs] == ["0", "1", "2"])
finally:
    f.ask = _orig_ask_u
    f.aggregate = _orig_agg_u
    f.installed_models = _orig_installed_u
    for k, v in _saved_u.items():
        setattr(f, k, v)

# ---------- 12e. 思考テキストからの答え回収 (2026-08-06, rank1) ----------
# 抽出ラダー単体
check("salvage: \\boxed{} を回収",
      f._salvage_from_thinking("考えている" * 20 + " so \\boxed{385} done")[0] == "385")
check("salvage: マーカー付きの値を回収",
      f._salvage_from_thinking("計算" * 30 + " Final Answer: 248 で確定")[0] == "248")
check("salvage: 日本語マーカーも拾う",
      f._salvage_from_thinking("検討" * 30 + " よって答えは 60 である")[0] == "60")
check("salvage: 裸の最終数値は拾わない(中間値の誤採用を防ぐ)",
      f._salvage_from_thinking("途中式 12 + 34 = 46 なので次に 99 を計算" * 5)[0] is None)
check("salvage: 短すぎる思考は対象外",
      f._salvage_from_thinking("短い")[0] is None)
check("salvage: 複数マーカーは後ろを採用(結論はより後ろ)",
      f._salvage_from_thinking("x" * 60 + " answer: 11 ... 訂正して final answer: 22")[0] == "22")
check("salvage: mcq で数値が来たら棄却",
      f._salvage_from_thinking("y" * 60 + " the answer is 42", "mcq")[0] is None)

# _send 統合: reasoning_content のみで content 空 → 回収して \boxed{} で返す
def sse_reasoning_only(think, finish="length"):
    lines = [("data: " + json.dumps(
        {"choices": [{"delta": {"reasoning_content": think}}]}) + "\n").encode(),
        ("data: " + json.dumps(
            {"choices": [{"delta": {}, "finish_reason": finish}]}) + "\n").encode(),
        b"data: [DONE]\n"]
    return FakeResponse(sse_lines=lines)

_saved_sv = (f.NIM_SALVAGE_THINKING, f.NIM_SALVAGE_LOG, dict(f.NIM_FAIL_COUNTS))
try:
    f.NIM_SALVAGE_LOG = False
    _think = "長い思考" * 40 + " Final Answer: 385"
    f.NIM_SALVAGE_THINKING = False
    out, _s, _ = run_mocked([sse_reasoning_only(_think)],
                            lambda: f.ask("test/model", MSGS, 0.5, num_predict=100))
    check("salvage: 既定(False)では従来どおり __ERROR__: truncated",
          out.startswith("__ERROR__: truncated"))
    f.NIM_SALVAGE_THINKING = True
    for k in f.NIM_FAIL_COUNTS:
        f.NIM_FAIL_COUNTS[k] = 0
    out, _s, _ = run_mocked([sse_reasoning_only(_think)],
                            lambda: f.ask("test/model", MSGS, 0.5, num_predict=100))
    check("salvage: 有効時は回収して boxed で返す", out == "\\boxed{385}")
    check("salvage: 呼び出し側の抽出器が拾える",
          f.extract_final_answer(out, "math") == "385")
    check("salvage: 型Aとして計上される",
          f.NIM_FAIL_COUNTS["A_think_only"] >= 1 and f.NIM_FAIL_COUNTS["A_salvaged"] >= 1)
    for k in f.NIM_FAIL_COUNTS:
        f.NIM_FAIL_COUNTS[k] = 0
    out, _s, _ = run_mocked([FakeResponse(sse_lines=[
        ("data: " + json.dumps({"choices": [{"delta": {},
         "finish_reason": "length"}]}) + "\n").encode(), b"data: [DONE]\n"])],
        lambda: f.ask("test/model", MSGS, 0.5, num_predict=100))
    # truncated は max_tokens 倍増リトライを起こすので送信は2回。カウントも2になる
    check("salvage: 思考も本文も空は型Bとして計上",
          f.NIM_FAIL_COUNTS["B_both_empty"] >= 1
          and f.NIM_FAIL_COUNTS["A_think_only"] == 0
          and out.startswith("__ERROR__"))
    for k in f.NIM_FAIL_COUNTS:
        f.NIM_FAIL_COUNTS[k] = 0
    out, _s, _ = run_mocked([sse_reasoning_only("中間値 7 と 13 を計算" * 20)],
                            lambda: f.ask("test/model", MSGS, 0.5, num_predict=100))
    check("salvage: マーカー無しは回収せず従来どおり無効票",
          out.startswith("__ERROR__") and f.NIM_FAIL_COUNTS["A_salvage_failed"] >= 1
          and f.NIM_FAIL_COUNTS["A_salvaged"] == 0)
    # 本文がある通常応答には一切影響しない
    out, _s, _ = run_mocked([ok_body("normal")],
                            lambda: f.ask("test/model", MSGS, 0.5))
    check("salvage: 本文がある応答は不変", out == "normal")
finally:
    f.NIM_SALVAGE_THINKING, f.NIM_SALVAGE_LOG, _c = _saved_sv
    f.NIM_FAIL_COUNTS.update(_c)

# ---------- 13. 採点正規化の表記ゆれ吸収 (2026-08-04, math500実測NGの回帰) ----------
check("norm: 度数 30^\\circ == 30", f.answers_equivalent("30", "30^\\circ"))
check("norm: 度数 90^{\\circ} == 90", f.answers_equivalent("90", "90^{\\circ}"))
check("norm: 度数 30° == 30", f.answers_equivalent("30", "30°"))
check("norm: x \\in [-2,7] == [-2,7]", f.answers_equivalent("[-2,7]", "x \\in [-2,7]"))
check("norm: \\text{(C)} == C", f.answers_equivalent("C", "\\text{(C)}"))
check("norm: 288 \\pi == 288\\pi", f.answers_equivalent("288\\pi", "288 \\pi"))
check("norm: % は従来どおり保持(50% != 50)", not f.answers_equivalent("50%", "50"))
check("norm: 別の度数値は不一致のまま", not f.answers_equivalent("30", "60^\\circ"))
check("norm: 式中間の^\\circは触らない",
      f.normalize_answer("30^\\circ + 5") == "30^\\circ + 5")
check("norm: (C)単独はC", f.normalize_answer("(C)") == "C")
check("norm: 散文の括弧は触らない", f.normalize_answer("(see C)") == "(see C)")

# ---------- 後始末・結果 ----------
for k, v in _SAVED.items():
    setattr(f, k, v)

print()
if _FAILS:
    print(f"FAILED: {len(_FAILS)} 件")
    for n in _FAILS:
        print(" -", n)
    raise SystemExit(1)
print("ALL PASSED")
