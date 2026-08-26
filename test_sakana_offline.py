"""fugu_core.sakana(TRINITY / Conductor)のオフライン回帰テスト。実 API 不要・即時完走。
実行: python test_sakana_offline.py

runner を注入して LLM を完全に置き換え、役割割当・停止条件・トポロジ検証・
退避経路を検証する。test_nim_offline.py と同じスクリプト形式(check() + 終了コード)。
"""
import re

from fugu_core import sakana as sk

_FAILS = []


def check(name, cond):
    print(f"[{'OK' if cond else 'NG'}] {name}")
    if not cond:
        _FAILS.append(name)


class Runner:
    """呼ばれた (model, prompt, system) を記録し、台本どおりに返すモック。

    script: model -> 応答 or 応答リスト(呼ばれるたびに次へ)。既定は "__ERROR__: no script"。
    """
    def __init__(self, script=None, default="ok"):
        self.script = script or {}
        self.default = default
        self.calls = []

    def __call__(self, model, prompt, system=None, json_mode=False):
        self.calls.append({"model": model, "prompt": prompt, "system": system,
                           "json_mode": json_mode})
        v = self.script.get(model, self.default)
        if isinstance(v, list):
            return v.pop(0) if v else "__ERROR__: script exhausted"
        return v

    def roles(self):
        """system プロンプトから役を復元する(役割割当の検証用)。"""
        out = []
        for c in self.calls:
            s = c["system"] or ""
            for r in sk.ROLES:
                if s.startswith(f"You are the {r.upper()}"):
                    out.append((c["model"], r))
                    break
        return out


POOL = ["mA", "mB", "mC"]
BOXED = "steps... \\boxed{42}"


def extract(text):
    m = re.search(r"\\boxed\{([^}]*)\}", text or "")
    return m.group(1) if m else None


# ---------- 1. TRINITY: 指揮者の割当が使われる ----------
r = Runner({"coord": '{"model": "mB", "role": "worker"}', "mB": BOXED})
res = sk.solve("Q", POOL, runner=r, coordinator="coord", max_turns=1, extract=extract)
check("trinity: 指揮者が選んだ (モデル, 役) がそのまま実行される",
      res and r.roles() == [("mB", "worker")])
check("trinity: Worker の \\boxed から答えを取る", res and res["answer"] == "42")
check("trinity: 指揮者への問い合わせは JSON モード(温度低・思考なし)",
      [c["json_mode"] for c in r.calls if c["model"] == "coord"] == [True])

# ---------- 2. 壊れた JSON / プール外モデルは決定的な既定へ落ちる ----------
r = Runner({"coord": "not json at all", "mA": BOXED, "mB": BOXED, "mC": BOXED})
res = sk.solve("Q", POOL, runner=r, coordinator="coord", max_turns=2, extract=extract)
check("trinity: 壊れた JSON は既定スケジュール(thinker→worker)へ",
      [x[1] for x in r.roles()] == ["thinker", "worker"])

r = Runner({"coord": '{"model": "not-in-pool", "role": "worker"}',
            "mA": BOXED, "mB": BOXED, "mC": BOXED})
sk.solve("Q", POOL, runner=r, coordinator="coord", max_turns=1, extract=extract)
check("trinity: プール外モデルの指名は採用しない",
      all(m in POOL for m, _ in r.roles()))

# ---------- 3. 停止条件 τ = min{k : verifier かつ ACCEPT} ----------
picks = iter([("mA", "worker"), ("mB", "verifier"), ("mC", "worker")])


def pick_seq(pool, transcript, turn, max_turns):
    try:
        return next(picks)
    except StopIteration:
        return ("mA", "worker")


r = Runner({"mA": BOXED, "mB": "ACCEPT - the derivation checks out", "mC": BOXED})
res = sk.solve("Q", POOL, runner=r, max_turns=5, extract=extract, pick_fn=pick_seq)
check("trinity: Verifier の ACCEPT で予算を残して停止する",
      res and res["accepted"] and res["turns"] == 2)

picks = iter([("mA", "worker"), ("mB", "verifier"), ("mA", "worker"), ("mB", "verifier")])
r = Runner({"mA": BOXED, "mB": "REVISE - step 3 is unjustified", "mC": BOXED})
res = sk.solve("Q", POOL, runner=r, max_turns=4, extract=extract, pick_fn=pick_seq)
check("trinity: REVISE では停止せず予算まで続ける",
      res and not res["accepted"] and res["turns"] == 4)

# ---------- 4. 仕事が無いのに verifier を選んだら worker へ矯正 ----------
r = Runner({"coord": '{"model": "mA", "role": "verifier"}', "mA": BOXED})
sk.solve("Q", POOL, runner=r, coordinator="coord", max_turns=1, extract=extract)
check("trinity: 検証対象が無い verifier は worker に矯正される",
      r.roles() == [("mA", "worker")])

# ---------- 5. 失敗しても既存経路の能力を奪わない ----------
r = Runner({}, default="__ERROR__: dead")
res = sk.solve("Q", POOL, runner=r, max_turns=3, extract=extract)
check("trinity: 全滅時は None(呼び出し側は従来経路へ)", res is None)

r = Runner({"mA": "__ERROR__: dead", "mB": BOXED, "mC": BOXED})
res = sk.solve("Q", POOL, runner=r, max_turns=3, extract=extract)
check("trinity: 1 ターン死んでも走り切る", res is not None and res["answer"] == "42")


def _raise(model, prompt, system=None, **kw):
    raise RuntimeError("boom")


check("trinity: runner の例外を外へ漏らさない",
      sk.solve("Q", POOL, runner=_raise, max_turns=2, extract=extract) is None)

# ---------- 6. 答えの回収は遡る ----------
picks2 = iter([("mA", "worker"), ("mB", "worker")])
r = Runner({"mA": BOXED, "mB": "I could not finish."})
res = sk.solve("Q", POOL, runner=r, max_turns=2, extract=extract,
               pick_fn=lambda *a: next(picks2))
check("trinity: 最後の Worker に答えが無ければ前の Worker から回収",
      res and res["answer"] == "42")

# ---------- 7. Conductor: トポロジの設計と実行 ----------
TOPO = ('{"workers":[{"id":"w1","model":"mA","instruction":"enumerate cases","inputs":[]},'
        '{"id":"w2","model":"mB","instruction":"check w1 and conclude","inputs":["w1"]}],'
        '"final":"w2"}')
r = Runner({"coord": TOPO, "mA": "case 1, case 2", "mB": "final: 42"})
topo = sk.plan_topology("Q", POOL, coordinator="coord", runner=r)
check("conductor: 妥当な JSON トポロジをそのまま採用する",
      [w["id"] for w in topo["workers"]] == ["w1", "w2"] and topo["final"] == "w2")

out = sk.run_topology("Q", topo, runner=r)
check("conductor: final ワーカーの出力を返す", out and out["text"] == "final: 42")
w2_prompt = [c["prompt"] for c in r.calls if c["model"] == "mB"][-1]
check("conductor: 下流ワーカーは上流の出力を見る", "case 1, case 2" in w2_prompt)
check("conductor: 各ワーカーに個別の指示が渡る",
      "check w1 and conclude" in w2_prompt
      and "enumerate cases" in [c["prompt"] for c in r.calls if c["model"] == "mA"][-1])

# ---------- 8. トポロジ検証(巡回・前方参照・プール外・上限) ----------
def _plan(js, **kw):
    return sk.plan_topology("Q", POOL, coordinator="coord",
                            runner=Runner({"coord": js, "mA": "x", "mB": "x", "mC": "x"}), **kw)


cyc = ('{"workers":[{"id":"w1","model":"mA","instruction":"a","inputs":["w2"]},'
       '{"id":"w2","model":"mB","instruction":"b","inputs":["w1"]}],"final":"w2"}')
check("conductor: 巡回/前方参照は棄却して退避トポロジへ",
      all(not w["inputs"] for w in _plan(cyc)["workers"]))
check("conductor: プール外モデルのトポロジは棄却",
      all(w["model"] in POOL for w in _plan(
          '{"workers":[{"id":"w1","model":"evil","instruction":"a","inputs":[]}],'
          '"final":"w1"}')["workers"]))
check("conductor: 指示が空のワーカーは棄却",
      _plan('{"workers":[{"id":"w1","model":"mA","instruction":"","inputs":[]}],'
            '"final":"w1"}')["workers"][0]["instruction"] != "")
many = ('{"workers":[' + ",".join(
    '{"id":"w%d","model":"mA","instruction":"i%d","inputs":[]}' % (i, i)
    for i in range(1, 9)) + '],"final":"w1"}')
check("conductor: ワーカー数の上限を超えない", len(_plan(many, max_workers=2)["workers"]) == 2)
check("conductor: 指揮者が壊れていても退避トポロジで走る",
      len(_plan("garbage")["workers"]) >= 1)

# ---------- 9. 自己再帰(論文の recursive topology) ----------
selfy = ('{"workers":[{"id":"w1","model":"coord","instruction":"do it myself","inputs":[]}],'
         '"final":"w1"}')
check("conductor: allow_self なら指揮者自身もワーカーになれる",
      _plan(selfy)["workers"][0]["model"] == "coord")
check("conductor: allow_self=False なら自己指名は棄却",
      _plan(selfy, allow_self=False)["workers"][0]["model"] in POOL)

# ---------- 10. 実行時の耐障害 ----------
r = Runner({}, default="__ERROR__: dead")
check("conductor: 全ワーカー失敗なら None(従来経路へ)",
      sk.run_topology("Q", sk.flat_topology(POOL, "Q"), runner=r) is None)
r = Runner({"mA": "__ERROR__: dead", "mB": "b out", "mC": "c out"})
out = sk.run_topology("Q", sk.flat_topology(POOL, "Q"), runner=r)
check("conductor: 一部失敗でも生き残りの出力を返す", out and out["text"] == "c out")
topo = {"workers": [{"id": "w1", "model": "mB", "instruction": "i", "inputs": []}],
        "final": "missing"}
out = sk.run_topology("Q", topo, runner=Runner({"mB": "b out"}))
check("conductor: final が存在しなければ最後に成功したワーカーへ", out["text"] == "b out")

print()
if _FAILS:
    print(f"FAILED: {len(_FAILS)} 件")
    for n in _FAILS:
        print(" -", n)
    raise SystemExit(1)
print("ALL PASSED")
