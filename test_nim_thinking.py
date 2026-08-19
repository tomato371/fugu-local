"""ネイティブ思考予算（reasoning_budget / reasoning_effort）のオフライン回帰テスト。
実 API 不要・数秒で完走。実行: python test_nim_thinking.py

FUGU_THINKING_BUDGET の 6 段階が、モデル自身の思考量にどう写像されるかを検証する。
従来この環境変数が動かしていたのは fugu 側の外側ループ（fugu_thinking の
reflections / min_rounds）だけで、モデルの思考量は 16384 固定だった。
test_nim_offline.py と同じスクリプト形式（check() + 終了コード）。
"""
import email.message
import io
import json
import os
import tempfile
import urllib.error
from pathlib import Path

import fugu_local as f

_FAILS = []


def check(name, cond):
    print(f"[{'OK' if cond else 'NG'}] {name}")
    if not cond:
        _FAILS.append(name)


# ---------- モック部品（test_nim_offline.py と同型）----------

class FakeResponse:
    def __init__(self, sse_lines=None):
        self._sse = sse_lines or []
        self.status = 200

    def read(self):
        return b""

    def __iter__(self):
        return iter(self._sse)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok_body(content="hello", think=None, finish="stop"):
    """content（と任意の reasoning_content）を持つ SSE ストリーム応答。"""
    lines = []
    if think:
        lines.append(("data: " + json.dumps(
            {"choices": [{"delta": {"reasoning_content": think}}]}) + "\n").encode())
    lines.append(("data: " + json.dumps(
        {"choices": [{"delta": {"content": content}}]}) + "\n").encode())
    lines.append(("data: " + json.dumps(
        {"choices": [{"delta": {}, "finish_reason": finish}]}) + "\n").encode())
    lines.append(b"data: [DONE]\n")
    return FakeResponse(sse_lines=lines)


def http_error(code, body=b"err"):
    def make():
        raise urllib.error.HTTPError("http://x", code, "err",
                                     email.message.Message(), io.BytesIO(body))
    return make


def run_mocked(responses, fn):
    sent, clock = [], [1_000_000.0]
    o_open, o_sleep, o_time = f.urllib.request.urlopen, f.time.sleep, f.time.time

    def fake_open(req, timeout=None):
        sent.append(req)
        r = responses[min(len(sent) - 1, len(responses) - 1)]
        return r() if callable(r) else r

    f.urllib.request.urlopen = fake_open
    f.time.sleep = lambda s: clock.__setitem__(0, clock[0] + max(float(s), 0.001))
    f.time.time = lambda: clock[0]
    f._NIM_COOLDOWN.clear()
    try:
        out = fn()
    except SystemExit as e:
        out = e
    finally:
        f.urllib.request.urlopen = o_open
        f.time.sleep = o_sleep
        f.time.time = o_time
        f._NIM_COOLDOWN.clear()
    return out, sent


def payload_of(req):
    return json.loads(req.data.decode("utf-8"))


def send(model, level=None, keep=None, **kw):
    """1 回送って payload を返す（level は FUGU_THINKING_BUDGET）。"""
    _set("FUGU_THINKING_BUDGET", level)
    _set("FUGU_KEEP_THINKING", keep)
    out, sent = run_mocked([ok_body(**{k: v for k, v in kw.items()
                                       if k in ("content", "think", "finish")})],
                           lambda: f.ask(model, MSGS, 0.5,
                                         **{k: v for k, v in kw.items()
                                            if k in ("fmt", "num_predict", "think")}))
    return payload_of(sent[0]), out, sent


def _set(key, value):
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


_TMP = Path(tempfile.mkdtemp(prefix="nim_think_"))
_SAVED = dict(
    NIM_MODEL_IDS=set(f.NIM_MODEL_IDS), NIM_API_KEY=f.NIM_API_KEY,
    NIM_BUDGET=f.NIM_BUDGET, NIM_BUDGET_FILE=f.NIM_BUDGET_FILE,
    MODEL_CONFIG=dict(getattr(f, "MODEL_CONFIG", {}) or {}),
)
_SAVED_ENV = {k: os.environ.get(k) for k in
              ("FUGU_THINKING_BUDGET", "FUGU_KEEP_THINKING")}
f.NIM_API_KEY = "nvapi-TEST-KEY"
f.NIM_BUDGET = 0
f.NIM_BUDGET_FILE = _TMP / "nim_usage.json"

NEMO = "nvidia/nemotron-3-super-120b-a12b"
OSS = "openai/gpt-oss-120b"
OTHER = "deepseek-ai/deepseek-r1"
f.NIM_MODEL_IDS = {NEMO, OSS, OTHER}
MSGS = [{"role": "user", "content": "hi"}]

# ---------- 1. 既定では何も変わらない ----------
p, _o, _s = send(NEMO)
check("既定: 思考レベル未設定なら reasoning_budget を足さない",
      "reasoning_budget" not in p)
p, _o, _s = send(NEMO, level="off")
check("既定: off も同じく素通り", "reasoning_budget" not in p)
p, _o, _s = send(NEMO, level="ぼんやり")
check("既定: 未知の値は無視される（暴走させない）", "reasoning_budget" not in p)

# ---------- 2. reasoning_budget 系統（nemotron-3/4）----------
EXPECT = {"minimal": 256, "low": 512, "medium": 2048,
          "high": 8192, "ultra": 16384, "max": 16384}
for level, tokens in EXPECT.items():
    p, _o, _s = send(NEMO, level=level)
    check(f"nemotron/{level}: reasoning_budget={tokens}",
          p.get("reasoning_budget") == tokens)
    check(f"nemotron/{level}: 思考は常に有効（切らない）",
          (p.get("chat_template_kwargs") or {}).get("enable_thinking") is True)

# 最小段だけ low_effort を立てる。enable_thinking=False にはしない —
# 推論前提のモデルに「考えるな」と言うと空応答が返る（2026-08-19 実測）。
p, _o, _s = send(NEMO, level="minimal")
check("nemotron/minimal: low_effort=True",
      (p.get("chat_template_kwargs") or {}).get("low_effort") is True)
check("nemotron/minimal: 思考を切らない（空応答の原因になる）",
      (p.get("chat_template_kwargs") or {}).get("enable_thinking") is True)
for level in ("low", "medium", "high", "ultra", "max"):
    p, _o, _s = send(NEMO, level=level)
    check(f"nemotron/{level}: low_effort は立てない",
          "low_effort" not in (p.get("chat_template_kwargs") or {}))

check("nemotron: 段階が深いほど予算が増える（単調）",
      [send(NEMO, level=lv)[0].get("reasoning_budget", 0)
       for lv in f.NIM_THINK_TOKENS] ==
      sorted(send(NEMO, level=lv)[0].get("reasoning_budget", 0)
             for lv in f.NIM_THINK_TOKENS))

# ---------- 3. reasoning_effort 系統（gpt-oss / mistral）----------
for level, effort in (("low", "low"), ("medium", "medium"),
                      ("high", "high"), ("ultra", "high"), ("max", "high")):
    p, _o, _s = send(OSS, level=level)
    check(f"gpt-oss/{level}: reasoning_effort={effort}",
          p.get("reasoning_effort") == effort)
p, _o, _s = send(OSS, level="minimal")
check("gpt-oss/minimal: reasoning_effort=low（送らないのではなく最小段）",
      p.get("reasoning_effort") == "low")
check("gpt-oss: reasoning_budget は送らない（受けない系統）",
      "reasoning_budget" not in p)

# ---------- 4. 系統不明のモデルには何も足さない ----------
p, _o, _s = send(OTHER, level="max")
check("系統不明: reasoning_budget を送らない（400→drop の往復を作らない）",
      "reasoning_budget" not in p)
check("系統不明: chat_template_kwargs も送らない", "chat_template_kwargs" not in p)

# ---------- 5. JSON スキーマ制約のある呼び出しは対象外 ----------
schema = {"type": "object", "properties": {"mode": {"type": "string"}}}
p, _o, _s = send(NEMO, level="max", fmt=schema)
check("fmt あり: 思考を伸ばさない（計画・分類の JSON を壊さない）",
      "reasoning_budget" not in p)
p, _o, _s = send(OSS, level="max", fmt=schema)
check("fmt あり: reasoning_effort も送らない", "reasoning_effort" not in p)

# ---------- 6. 出力上限の確保 ----------
p, _o, _s = send(NEMO, level="ultra", num_predict=4096)
check("max_tokens: 思考予算より小さい上限は引き上げる（本文ゼロを防ぐ）",
      p.get("max_tokens") == 16384 + 4096)
p, _o, _s = send(NEMO, level="low", num_predict=30000)
check("max_tokens: 十分大きい指定はそのまま", p.get("max_tokens") == 30000)
p, _o, _s = send(NEMO, level="minimal", num_predict=1024)
check("max_tokens: 最小段でも思考ぶんは確保する", p.get("max_tokens") == 256 + 4096)

# ---------- 7. 非対応モデルに当たっても 1 回で復帰する ----------
_set("FUGU_THINKING_BUDGET", "high")
out, sent = run_mocked([http_error(400, b"reasoning_budget not supported"),
                        ok_body("ok2")],
                       lambda: f.ask(NEMO, MSGS, 0.5))
check("400drop: 1回目に reasoning_budget を送る",
      "reasoning_budget" in payload_of(sent[0]))
check("400drop: 400 なら落として即再送・成功", out == "ok2" and len(sent) == 2)
check("400drop: 再送 payload から思考キーが消えている",
      all(k not in payload_of(sent[1])
          for k in ("reasoning_budget", "chat_template_kwargs")))

# ---------- 7.5 思考を絞って空応答なら通常量へ戻して再送 ----------
_set("FUGU_THINKING_BUDGET", "minimal")
empty = FakeResponse(sse_lines=[b"data: [DONE]\n"])
out, sent = run_mocked([empty, ok_body("戻したら出た")],
                       lambda: f.ask(NEMO, MSGS, 0.5))
check("空応答: 1回目は絞った予算で送る",
      payload_of(sent[0]).get("reasoning_budget") == 256)
check("空応答: 2回目は medium の予算へ戻す",
      len(sent) >= 2 and payload_of(sent[1]).get("reasoning_budget")
      == f.NIM_THINK_TOKENS["medium"])
check("空応答: 2回目は low_effort を外す",
      len(sent) >= 2 and "low_effort" not in
      (payload_of(sent[1]).get("chat_template_kwargs") or {}))
check("空応答: 戻したら本文が返る", out == "戻したら出た")

_set("FUGU_THINKING_BUDGET", "minimal")
out, sent = run_mocked([empty], lambda: f.ask(NEMO, MSGS, 0.5))
check("空応答: 戻しても空なら諦める（無限ループしない）", len(sent) == 2 and out == "")

_set("FUGU_THINKING_BUDGET", "ultra")
out, sent = run_mocked([empty, ok_body("x")], lambda: f.ask(NEMO, MSGS, 0.5))
check("空応答: 元から大きい予算では戻し再送をしない", len(sent) == 1 and out == "")

# ---------- 8. 思考本文の回収 ----------
f.thinking_reset()
p, out, _s = send(NEMO, level="high", keep=None, content="答え", think="考え中…")
check("回収: 既定 OFF なら控えない", f.thinking_blocks() == [])
check("回収: OFF でも本文は従来どおり返る", out == "答え")

f.thinking_reset()
p, out, _s = send(NEMO, level="high", keep="1", content="答え", think="考え中…")
blocks = f.thinking_blocks()
check("回収: ON なら 1 本控える", len(blocks) == 1)
check("回収: モデル名が入る", blocks and blocks[0]["model"] == NEMO)
check("回収: 本文が入る", blocks and blocks[0]["text"] == "考え中…")
check("回収: 文字数が入る", blocks and blocks[0]["chars"] == len("考え中…"))
check("回収: 回答本文は思考に汚染されない", out == "答え")

f.thinking_reset()
check("回収: reset で空になる", f.thinking_blocks() == [])

f.thinking_reset()
send(NEMO, level="high", keep="1", content="答え", think=None)
check("回収: 思考が無い応答では控えない", f.thinking_blocks() == [])

f.thinking_reset()
for _ in range(f.NIM_THINK_KEEP_MAX + 5):
    send(NEMO, level="high", keep="1", content="答え", think="x" * 100)
check(f"回収: 上限 {f.NIM_THINK_KEEP_MAX} 本で打ち止め",
      len(f.thinking_blocks()) == f.NIM_THINK_KEEP_MAX)

f.thinking_reset()
send(NEMO, level="high", keep="1", content="答え",
     think="A" + "y" * (f.NIM_THINK_KEEP_CHARS + 500))
b = f.thinking_blocks()[0]
check("回収: 長い思考は末尾を残して切る",
      len(b["text"]) == f.NIM_THINK_KEEP_CHARS and b["text"].endswith("y"))
check("回収: 切っても元の長さは記録される",
      b["chars"] == f.NIM_THINK_KEEP_CHARS + 501)

# ---------- 9. auto の解決 ----------
_set("FUGU_THINKING_BUDGET", "auto")
check("auto: 挨拶は minimal", f.resolve_thinking_level("こんにちは") == "minimal")
check("auto: 証明問題は深い予算",
      f.resolve_thinking_level("この定理を証明してください") in ("high", "ultra"))
check("auto: 短い質問は low", f.resolve_thinking_level("Python とは") == "low")
_set("FUGU_THINKING_BUDGET", "ultra")
check("auto: 明示指定はそのまま返る", f.resolve_thinking_level("なんでも") == "ultra")
_set("FUGU_THINKING_BUDGET", None)
check("auto: 未設定は空", f.resolve_thinking_level("なんでも") == "")
_set("FUGU_THINKING_BUDGET", "off")
check("auto: off は空", f.resolve_thinking_level("なんでも") == "")

_set("FUGU_THINKING_BUDGET", "high")
check("thinking_level: 有効なレベルを返す", f.thinking_level() == "high")
_set("FUGU_THINKING_BUDGET", "auto")
check("thinking_level: auto は未解決なので空", f.thinking_level() == "")

# ---------- 後始末・結果 ----------
for k, v in _SAVED.items():
    setattr(f, k, v)
for k, v in _SAVED_ENV.items():
    _set(k, v)
f.thinking_reset()

print()
if _FAILS:
    print(f"FAILED: {len(_FAILS)} 件")
    for n in _FAILS:
        print(" -", n)
    raise SystemExit(1)
print("ALL PASSED")
