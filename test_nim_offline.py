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
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")
        self.status = 200

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok_body(content="hello", finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def http_error(code, body=b"err", headers=None):
    """毎回新しい HTTPError を作るファクトリ（read() は一度しか呼べないため）。"""
    def make():
        h = email.message.Message()
        for k, v in (headers or {}).items():
            h[k] = str(v)
        raise urllib.error.HTTPError("http://x", code, "err", h, io.BytesIO(body))
    return make


def run_mocked(responses, fn):
    """urlopen と time.sleep をモックして fn() を実行。
    responses: 各送信に対する応答のリスト。FakeResponse か callable(raise 用)。
    リストが尽きたら最後の要素を繰り返す。
    戻り値: (fn の結果 or 送出された SystemExit, 送信された Request のリスト, sleep 秒のリスト)"""
    sent, sleeps = [], []
    orig_open, orig_sleep = f.urllib.request.urlopen, f.time.sleep

    def fake_open(req, timeout=None):
        sent.append(req)
        r = responses[min(len(sent) - 1, len(responses) - 1)]
        if callable(r):
            return r()
        return r

    f.urllib.request.urlopen = fake_open
    f.time.sleep = lambda s: sleeps.append(s)
    try:
        out = fn()
    except SystemExit as e:
        out = e
    finally:
        f.urllib.request.urlopen = orig_open
        f.time.sleep = orig_sleep
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
out, sent, _ = run_mocked([FakeResponse(ok_body("world"))],
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
out, sent, _ = run_mocked([FakeResponse({"choices": []})],
                          lambda: f.ask("test/model", MSGS, 0.5))
check("error: choices 空でもクラッシュせず空文字/エラー", isinstance(out, str))

# ---------- 4. 429 / Retry-After（別予算・cap・attempt 不消費）----------
out, sent, sleeps = run_mocked(
    [http_error(429, headers={"Retry-After": "3"}),
     http_error(429, headers={"Retry-After": "999"}),
     FakeResponse(ok_body("recovered"))],
    lambda: f.ask("test/model", MSGS, 0.5))
check("429: 待って成功すれば通常の応答", out == "recovered")
check("429: 送信は 3 回（初回+リトライ2）", len(sent) == 3)
check("429: Retry-After を尊重", sleeps and sleeps[0] == 3.0)
check("429: Retry-After は cap で抑える", len(sleeps) >= 2 and sleeps[1] == f.NIM_RETRY_AFTER_CAP)
check("429: 通常予算の指数バックオフは混ざらない",
      all(s not in f.ASK_RETRY_BACKOFF for s in sleeps))

# ---------- 5. length 打ち切り ----------
out, _, _ = run_mocked([FakeResponse(ok_body("", finish="length"))],
                       lambda: f.ask("test/model", MSGS, 0.5))
check("length: 本文空 + finish_reason=length は __ERROR__: truncated（SC 無効票化）",
      isinstance(out, str) and out.startswith("__ERROR__") and "truncated" in out)
out, _, _ = run_mocked([FakeResponse(ok_body("partial answer", finish="length"))],
                       lambda: f.ask("test/model", MSGS, 0.5))
check("length: 本文が一部でもあればそのまま使う", out == "partial answer")

# ---------- 6. リクエスト予算（SystemExit 42・送信前ブロック）----------
f.NIM_BUDGET_FILE.write_text(json.dumps({"total_requests": 2}), encoding="utf-8")
f.NIM_BUDGET = 2
out, sent, _ = run_mocked([FakeResponse(ok_body())],
                          lambda: f.ask("test/model", MSGS, 0.5))
check("budget: 上限到達で SystemExit(42)",
      isinstance(out, SystemExit) and out.code == 42)
check("budget: 送信自体が行われない", len(sent) == 0)
f.NIM_BUDGET = 0
before = f.NIM_REQUEST_COUNT
out, sent, _ = run_mocked([FakeResponse(ok_body())],
                          lambda: f.ask("test/model", MSGS, 0.5))
check("budget: カウンタは送信ごとに増える", f.NIM_REQUEST_COUNT == before + 1)
check("budget: nim_usage.json に累計が永続化される",
      json.loads(f.NIM_BUDGET_FILE.read_text(encoding="utf-8"))["total_requests"] == 3)

# ---------- 7. reasoning_effort / response_format の 400 落とし再送 ----------
out, sent, _ = run_mocked([http_error(400, body=b"param not supported"),
                           FakeResponse(ok_body("ok2"))],
                          lambda: f.ask("test/model", MSGS, 0.5, think="high"))
check("400drop: 1回目に reasoning_effort を送る",
      "reasoning_effort" in payload_of(sent[0]))
check("400drop: 400 なら落として即再送・成功", out == "ok2" and len(sent) == 2)
check("400drop: 再送 payload に拡張パラメータが無い",
      "reasoning_effort" not in payload_of(sent[1]))
check("400drop: think=True は high に写像",
      payload_of(run_mocked([FakeResponse(ok_body())],
                            lambda: f.ask("test/model", MSGS, 0.5, think=True))[1][0]
                 ).get("reasoning_effort") == "high")

# ---------- 8. スキーマ（fmt）の扱い ----------
schema = {"type": "object", "properties": {"mode": {"type": "string"}}}
out, sent, _ = run_mocked([FakeResponse(ok_body())],
                          lambda: f.ask("test/model", MSGS, 0.5, fmt=schema))
p = payload_of(sent[0])
check("fmt: 非対応モデルは response_format を送らない", "response_format" not in p)
check("fmt: schema は system へ文字列注入される",
      any(m["role"] == "system" and "mode" in m["content"] for m in p["messages"]))
check("fmt: 元の messages リストは破壊しない（コピーに注入）",
      all(m["role"] != "system" for m in MSGS))
out, sent, _ = run_mocked([FakeResponse(ok_body())],
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
    out, sent, _ = run_mocked([FakeResponse(ok_body())],
                              lambda: f.ask("test/model", MSGS, 0.5))
    p = payload_of(sent[0])
    check("nim_extra: トップレベルにマージされる",
          p.get("chat_template_kwargs") == {"enable_thinking": True}
          and p.get("reasoning_budget") == 999)
    out, sent, _ = run_mocked([http_error(400), FakeResponse(ok_body("ok3"))],
                              lambda: f.ask("test/model", MSGS, 0.5))
    check("nim_extra: 400 なら extra キーも落として再送・成功",
          out == "ok3" and len(sent) == 2
          and all(k not in payload_of(sent[1])
                  for k in ("chat_template_kwargs", "reasoning_budget")))
finally:
    f.MODEL_CONFIG = _saved_mc

# ---------- 9. <think> 混入は呼び出し側 strip_think の責務（Ollama 経路と同一分担）----------
out, _, _ = run_mocked([FakeResponse(ok_body("<think>reasoning...</think>42"))],
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
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # タイポ検出の /v1/models 照合もモック経由（全 ID 実在扱い）
        def _profile():
            return f.apply_nim_profile()
        applied, _, _ = run_mocked(
            [lambda: FakeResponse({"data": [{"id": m} for m in
                                            ("meta/llama-3.1-8b-instruct", "openai/gpt-oss-120b",
                                             "moonshotai/kimi-k2.6",
                                             "nvidia/nemotron-3-ultra-550b-a55b",
                                             "deepseek-ai/deepseek-v4-pro", "z-ai/glm-5.2",
                                             "mistralai/mistral-medium-3.5-128b",
                                             "minimaxai/minimax-m3")]})],
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
