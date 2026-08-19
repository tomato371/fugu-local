"""速度モード（プラン外部指定）と布陣一覧のオフライン回帰テスト。

実 API 不要・数秒で完走。実行: python test_nim_routing.py

Conductor を通さずに mode / 顔ぶれ / ラウンド数を外から渡せることと、
その際も出力形態のガードレール（PowerPoint・画像）は効くことを確かめる。
test_nim_offline.py と同じスクリプト形式（check() + 終了コード）。
"""
import fugu_local as f

_FAILS = []


def check(name, cond):
    print(f"[{'OK' if cond else 'NG'}] {name}")
    if not cond:
        _FAILS.append(name)


# ---------- 布陣を差し替えて検証する ----------
_SAVED = {k: getattr(f, k) for k in (
    "PERSONA_MODELS", "MODEL_TO_PERSONA", "PROPOSERS", "AGGREGATOR", "CONDUCTOR",
    "ARBITER_MODEL", "SECOND_OPINION_MODEL", "REASONING_MODELS", "NIM_MODEL_IDS",
    "NIM_VISION_MODEL", "VISION_MODEL", "FUGU_BACKEND", "PROPOSER_PROFILES",
    "DESIRED_PROPOSERS", "DESIRED_AGGREGATOR", "DESIRED_CONDUCTOR", "MODEL_CONFIG")}

A = "nvidia/nemotron-3-ultra-550b-a55b"
B = "mistralai/mistral-nemotron"
C = "openai/gpt-oss-120b"
D = "nvidia/nemotron-3-super-120b-a12b"
AGG = "stepfun-ai/step-3.7-flash"
COND = "meta/llama-3.1-8b-instruct"
VIS = "nvidia/nemotron-nano-12b-v2-vl"

f.FUGU_BACKEND = "nim"
f.PERSONA_MODELS = {"Proposer A": A, "Proposer B": B, "Proposer C": C, "Proposer D": D}
f.MODEL_TO_PERSONA = {v: k for k, v in f.PERSONA_MODELS.items()}
f.PROPOSERS = [A, B, C, D]
f.AGGREGATOR = AGG
f.CONDUCTOR = COND
f.ARBITER_MODEL = A
f.SECOND_OPINION_MODEL = B
f.REASONING_MODELS = [A, B, C]
f.NIM_VISION_MODEL = VIS
f.NIM_MODEL_IDS = {A, B, C, D, AGG, COND, VIS}
f.PROPOSER_PROFILES = {}

# ---------- 1. 布陣一覧 ----------
r = f.roster()
ids = [m["id"] for m in r["models"]]
check("roster: 7 体すべてが並ぶ", sorted(ids) == sorted([A, B, C, D, AGG, COND, VIS]))
check("roster: 提案役の順序は A→D", ids[:4] == [A, B, C, D])
check("roster: 役割が付く", r["models"][0]["roles"][:1] == ["proposer_a"])
check("roster: 兼務は 1 行にまとまる",
      set(next(m for m in r["models"] if m["id"] == A)["roles"])
      == {"proposer_a", "arbiter", "sc"})
check("roster: ペルソナ名が入る",
      next(m for m in r["models"] if m["id"] == B)["persona"] == "Proposer B")
check("roster: 統合役/Conductor/vision を指す", (r["aggregator"], r["conductor"],
      r["vision"]) == (AGG, COND, VIS))
check("roster: 同じ ID が重複しない", len(ids) == len(set(ids)))

f.ARBITER_MODEL = "私は生きていない/model"
check("roster: 布陣外の役割も欠かさず出す",
      "私は生きていない/model" in [m["id"] for m in f.roster()["models"]])
f.ARBITER_MODEL = A

# ---------- 2. 実モデル名の直接指名 ----------
check("resolve: ペルソナ名は従来どおり解決する", f._resolve_proposer("Proposer B") == B)
check("resolve: 略記も従来どおり", f._resolve_proposer("a") == A)
check("resolve: 提案役の実 ID はそのまま", f._resolve_proposer(C) == C)
check("resolve: 提案役でない統合役も指名できる（UI から選ぶため）",
      f._resolve_proposer(AGG) == AGG)
check("resolve: Conductor も指名できる", f._resolve_proposer(COND) == COND)
check("resolve: 布陣に無い ID は拒否する", f._resolve_proposer("evil/model") is None)
check("resolve: 非文字列は拒否する（従来の不変条件）",
      f._resolve_proposer(["a"]) is None and f._resolve_proposer(None) is None)

# ---------- 3. プランの検証 ----------
p = f.validate_plan({"mode": "single", "selected_proposers": [AGG]})
check("plan: single は 1 体に絞られる", p["selected_proposers"] == [AGG])
check("plan: single のまま", p["mode"] == "single")

p = f.validate_plan({"mode": "moa", "selected_proposers": [A, C], "rounds": 2})
check("plan: moa は指定した顔ぶれをそのまま使う", p["selected_proposers"] == [A, C])
check("plan: rounds が通る", p["rounds"] == 2)

p = f.validate_plan({"mode": "moa", "selected_proposers": [A, A, C]})
check("plan: 重複は落とす", p["selected_proposers"] == [A, C])

p = f.validate_plan({"mode": "moa", "selected_proposers": [A, B, C, D, AGG]})
check("plan: 顔ぶれは 4 体まで", len(p["selected_proposers"]) == 4)

p = f.validate_plan({"mode": "moa", "selected_proposers": ["evil/model"]})
check("plan: 未知モデルだけなら既定の布陣へ落ちる",
      p["selected_proposers"] == f.PROPOSERS[:3])

p = f.validate_plan({"mode": "moa", "selected_proposers": [A], "rounds": 999})
check("plan: rounds は MAX_ROUNDS で頭打ち", p["rounds"] == f.MAX_ROUNDS)

# ---------- 4. 出力形態のガードレールは効き続ける ----------
p = f._apply_routing_guardrails(
    "この内容でパワポを作って", f.validate_plan({"mode": "single",
                                                "selected_proposers": [AGG]}))
check("guardrail: パワポ指示は single 指定でも合議へ回す", p["mode"] == "moa")
check("guardrail: make_pptx が立つ", p["make_pptx"] is True)

p = f._apply_routing_guardrails(
    "猫のイラストを描いて", f.validate_plan({"mode": "single",
                                             "selected_proposers": [AGG]}))
check("guardrail: 画像指示を拾う", p["use_image_generation"] is True)

p = f._apply_routing_guardrails(
    "型ヒントの書き方を教えて", f.validate_plan({"mode": "single",
                                                 "selected_proposers": [AGG]}))
check("guardrail: ふつうの質問は single のまま（速さの指定を覆さない）",
      p["mode"] == "single")

# 精度ガードレールは「明示指定」経路では通さない（覆さないことの確認）
p = f.validate_plan({"mode": "single", "selected_proposers": [AGG]})
p2 = f._apply_accuracy_guardrails("この関数を実装して", dict(p))
check("参考: 精度ガードレール単体ならコード質問を moa へ上げる", p2["mode"] == "moa")
check("明示指定: routing だけを通すので single のまま",
      f._apply_routing_guardrails("この関数を実装して", dict(p))["mode"] == "single")

# ---------- 5. ask_fugu が Conductor をスキップする ----------
seen = {}
_orig = (f.setup, f.conduct, f.fugu_answer, f.build_context, f.notify_slack,
         f.save_history_file, f.strip_think)


def _boom(*a, **kw):
    raise AssertionError("conduct が呼ばれた（スキップされていない）")


def _capture(q, plan=None, history=None):
    seen["plan"] = plan
    seen["question"] = q
    return "こたえ"


f.setup = lambda: True
f.conduct = _boom
f.fugu_answer = _capture
f.build_context = lambda q, use_search=False, rag_dirs=None: ""
f.notify_slack = lambda *a, **kw: None
f.save_history_file = lambda *a, **kw: False
try:
    f._HISTORY.clear()
    out = f.ask_fugu("なにか教えて", plan={"mode": "single",
                                           "selected_proposers": [AGG]})
    check("ask_fugu: plan 指定なら conduct を呼ばない", True)
    check("ask_fugu: 回答が返る", out == "こたえ")
    check("ask_fugu: 指定した mode で実行される", seen["plan"]["mode"] == "single")
    check("ask_fugu: 指定した顔ぶれで実行される",
          seen["plan"]["selected_proposers"] == [AGG])

    f._HISTORY.clear()
    f.ask_fugu("この件を深く検討して", plan={"mode": "moa",
                                            "selected_proposers": [A, C],
                                            "rounds": 3})
    check("ask_fugu: moa の顔ぶれも通る", seen["plan"]["selected_proposers"] == [A, C])
    check("ask_fugu: rounds も通る", seen["plan"]["rounds"] == 3)

    f._HISTORY.clear()
    f.ask_fugu("パワポにして", plan={"mode": "single", "selected_proposers": [AGG]})
    check("ask_fugu: plan 指定でも出力形態のガードレールは効く",
          seen["plan"]["mode"] == "moa" and seen["plan"]["make_pptx"] is True)

    # plan を渡さない従来経路は conduct を呼ぶ（呼ばれることを _boom で確認）
    f._HISTORY.clear()
    raised = False
    try:
        f.ask_fugu("なにか教えて")
    except AssertionError:
        raised = True
    check("ask_fugu: plan 未指定なら従来どおり conduct を呼ぶ", raised)
finally:
    (f.setup, f.conduct, f.fugu_answer, f.build_context, f.notify_slack,
     f.save_history_file, f.strip_think) = _orig
    for k, v in _SAVED.items():
        setattr(f, k, v)
    f._HISTORY.clear()

# ---------- 6. 布陣キャッシュ（子プロセスの起動を速くする仕掛け）----------
import json as _json
import tempfile as _tf

import pathlib as _pl
_LIN = _pl.Path(_tf.mkdtemp(prefix="lineup_")) / "nim_lineup.json"
_sf, _st = f.NIM_LINEUP_FILE, f.NIM_LINEUP_TTL
f.NIM_LINEUP_FILE, f.NIM_LINEUP_TTL = str(_LIN), 3600

f._nim_save_lineup([A, B, C, D, AGG], COND, VIS)
got = f._nim_load_lineup()
check("lineup: 書いて読める", got and got["picked"] == [A, B, C, D, AGG])
check("lineup: conductor も残る", got["conductor"] == COND)
check("lineup: vision も残る", got["vision"] == VIS)

_raw = _json.loads(_LIN.read_text(encoding="utf-8"))
check("lineup: どのバックエンドのものか記録する", _raw["url"] == f.NIM_URL)
_raw["url"] = "https://other.invalid/v1"
_LIN.write_text(_json.dumps(_raw), encoding="utf-8")
check("lineup: バックエンドが違えば使わない", f._nim_load_lineup() is None)

_raw["url"] = f.NIM_URL
_raw["at"] = 0
_LIN.write_text(_json.dumps(_raw), encoding="utf-8")
check("lineup: 古すぎれば使わない（TTL）", f._nim_load_lineup() is None)
f.NIM_LINEUP_TTL = 0
check("lineup: TTL=0 なら無期限", f._nim_load_lineup() is not None)
f.NIM_LINEUP_TTL = 3600

_LIN.write_text(_json.dumps({"url": f.NIM_URL, "at": f.time.time(),
                             "picked": [A, B]}), encoding="utf-8")
check("lineup: 4体そろわない布陣は使わない", f._nim_load_lineup() is None)

_LIN.write_text("これは JSON ではない", encoding="utf-8")
check("lineup: 壊れていても例外にせず None", f._nim_load_lineup() is None)

_LIN.unlink()
check("lineup: 無ければ None", f._nim_load_lineup() is None)
f.NIM_LINEUP_FILE = ""
check("lineup: 未設定なら常に None（従来どおり毎回プローブ）",
      f._nim_load_lineup() is None)
f._nim_save_lineup([A, B, C, D], COND, VIS)
check("lineup: 未設定なら書きもしない", not _LIN.exists())
f.NIM_LINEUP_FILE = str(_LIN)
f._nim_save_lineup([A, B], COND, VIS)
check("lineup: 4体未満は書かない", not _LIN.exists())

# 生存プローブを本当に省くか（apply_nim_profile を通して確認）
f._nim_save_lineup([A, B, C, D, AGG], COND, VIS)
probes = []
_o2 = (f._nim_catalog, f._nim_pick_live, f._nim_model_available,
       f._nim_pick_vision, f._nim_scale_of, f.NIM_API_KEY)
f._nim_catalog = lambda: probes.append("catalog") or {A, B, C, D, AGG, COND, VIS}
f._nim_pick_live = lambda ranked, n: (probes.append("probe"), ([], 0))[1]
f._nim_model_available = lambda m: probes.append("avail") or True
f._nim_pick_vision = lambda cat, n: probes.append("vision") or [VIS]
f._nim_scale_of = lambda m: (100.0, 100.0)
f.NIM_API_KEY = "nvapi-TEST"
try:
    ok = f.apply_nim_profile()
    check("lineup: apply_nim_profile が成功する", ok is True)
    check("lineup: 生存プローブを1回も走らせない", probes == [])
    # apply_nim_profile が決めるのは DESIRED_*（PROPOSERS/CONDUCTOR は setup() が確定させる）
    check("lineup: キャッシュの布陣が提案役に入る", f.DESIRED_PROPOSERS == [A, B, C, D])
    check("lineup: キャッシュの conductor を使う", f.DESIRED_CONDUCTOR == COND)
    check("lineup: 統合役はキャッシュの5体目", f.DESIRED_AGGREGATOR == AGG)
    check("lineup: キャッシュの vision を使う", f.NIM_VISION_MODEL == VIS)
finally:
    (f._nim_catalog, f._nim_pick_live, f._nim_model_available,
     f._nim_pick_vision, f._nim_scale_of, f.NIM_API_KEY) = _o2
    f.NIM_LINEUP_FILE, f.NIM_LINEUP_TTL = _sf, _st
    for k, v in _SAVED.items():
        setattr(f, k, v)

print()
if _FAILS:
    print(f"FAILED: {len(_FAILS)} 件")
    for n in _FAILS:
        print(" -", n)
    raise SystemExit(1)
print("ALL PASSED")
