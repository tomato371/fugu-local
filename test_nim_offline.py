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
    _catalog = FakeResponse({"data": [{"id": m} for m in
                                      ("meta/llama-3.1-8b-instruct", "openai/gpt-oss-120b",
                                       "moonshotai/kimi-k2.6",
                                       "nvidia/nemotron-3-ultra-550b-a55b",
                                       "deepseek-ai/deepseek-v4-pro", "z-ai/glm-5.2",
                                       "mistralai/mistral-medium-3.5-128b",
                                       "minimaxai/minimax-m3")]})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # 送信順: kimi プローブ → deepseek プローブ → /v1/models 照合（全部可用の想定）
        def _profile():
            return f.apply_nim_profile()
        applied, _, _ = run_mocked(
            [FakeResponse({"ok": True}), FakeResponse({"ok": True}), _catalog],
            _profile)
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
    check("profile: 可用プローブOKなら kimi が Proposer B・deepseek がSC第1系統/裁定",
          f.PERSONA_MODELS["Proposer B"] == "moonshotai/kimi-k2.6"
          and f.REASONING_MODELS[0] == "deepseek-ai/deepseek-v4-pro"
          and f.ARBITER_MODEL == "deepseek-ai/deepseek-v4-pro")
    # 混雑/404 時の縮退: kimi 404 → mistral、deepseek 429 → nemotron が代替
    with contextlib.redirect_stdout(io.StringIO()):
        applied2, _, _ = run_mocked(
            [http_error(404), http_error(429), _catalog], _profile)
    check("profile: プローブNGなら mistral / nemotron へ自動縮退",
          applied2 is True
          and f.PERSONA_MODELS["Proposer B"] == "mistralai/mistral-medium-3.5-128b"
          and f.REASONING_MODELS[0] == "nvidia/nemotron-3-ultra-550b-a55b"
          and f.ARBITER_MODEL == "nvidia/nemotron-3-ultra-550b-a55b")
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
