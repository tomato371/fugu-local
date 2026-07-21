"""フェーズ3 評価ハーネス: 単体ベースライン / 静的MoA(思考ON/OFF) / 動的Fugu を
同じ質問セットで比較し、正誤・所要時間・段階別タイミングを出す。

注意: 8GB 逐次 + 思考モデルのため各合議は数分かかる。結果は 1 問ごとに逐次出力する。
採点: 決定的な問は正規化＋数値/トークンの厳密マッチ、証明系は Critic(スキーマ強制JSON)を
LLM ジャッジとして流用（ブレ対策に 3 回多数決）。
部分実行: `python eval_fugu.py 17x23 豪州` のようにラベルの部分一致で問題を、
`c=a,d` のように構成を絞れる（引数なしで全問・全構成）。例: `python eval_fugu.py c=b,c 6の倍数`
"""
import re
import sys
import time
import fugu_local as f

f.SHOW_TIMING = True
f.SHOW_PLAN = False
f.SHOW_PROPOSALS = False

# (ラベル, 質問, 採点関数) — 採点は content 正規化後に判定
def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower())


def has_num(a, val):
    """数値 val が独立した数として現れるか。素の in 判定だと '7' が '17' に、
    '391' が '3910' に誤マッチするため、隣接する数字を弾く。ピリオドは直後に数字が
    続く場合のみ小数点とみなす（文末の '391.' は独立した数として許容する）。"""
    pat = r"(?<!\d)(?<!\d\.)" + re.escape(val) + r"(?!\d)(?!\.\d)"
    return re.search(pat, a or "") is not None


# LLM ジャッジ採点: Critic(qwen3:4b + think=False + スキーマ強制JSON)を流用。
# トークンマッチで判定できない問（証明系）と、決定的採点が曖昧になるケースのフォールバックに
# 使う。単発判定はぶれうるので 3 回の多数決にする（temperature=0.1 でもゼロではないため。
# 判定時間は採点側なので dt には含まれない）。
def judge(q, a):
    votes = [f.critique(q, a)[0] for _ in range(3)]
    return votes.count(True) >= 2


PROOF_Q = "3つの連続する整数の積は必ず6で割り切れる。なぜか簡潔に証明して。"
BATBALL_Q = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
             "How much does the ball cost? Give just the amount.")


def judge_proof(a):
    return judge(PROOF_Q, a)


# コード問題の採点: 回答のコードブロックに検収テスト(assert)を継ぎ足して実際に実行する。
# LLM ジャッジより強い決定的採点で、コード生成(主用途)の品質を直接測る。
FIB_Q = ("PythonでN番目のフィボナッチ数を返す関数 fib(n) を書いて。fib(1)=1, fib(2)=1 とする。"
         "完成したコードを1つの```pythonブロックで示して。")


def grade_code_fib(a):
    code = f.extract_code(a)
    if not code:
        return False
    test = (code + "\n\n"
            "assert fib(1) == 1 and fib(2) == 1 and fib(10) == 55 and fib(20) == 6765\n"
            "print('GRADE_PASS')\n")
    ok, out = f.run_python(test)
    return ok and "GRADE_PASS" in out


def grade_batball(a):
    """bat&ball の 2 段階採点。0.05 が無ければ NG。0.05 のみなら OK。
    0.05 と 0.10 が併記される場合（検算・誤答訂正の文脈で正解でも 0.10 に触れうる。
    2026-07-03 のフル評価で静的MoA の正答が旧ルールの一律除外で偽陰性になった）は
    トークンでは判定できないため LLM ジャッジに委ねる。"""
    if not (has_num(a, "0.05") or "5 cent" in norm(a) or "5セント" in a):
        return False
    if not has_num(a.replace("1.10", ""), "0.10"):
        return True
    return judge(BATBALL_Q, a)


TESTS = [
    ("bat&ball(トラップ)",
     BATBALL_Q,
     grade_batball),
    ("17x23(計算)",
     "Compute 17 * 23. Answer with the number only.",
     lambda a: has_num(a, "391")),
    ("91は素数?(数論)",
     "Is 91 a prime number? Answer yes or no, then give its prime factorization.",
     lambda a: ("no" in norm(a) or "いいえ" in a or "not prime" in norm(a))
               and has_num(a, "7") and has_num(a, "13")),
    ("豪州首都(トラップ)",
     "What is the capital city of Australia? Answer with the city name only.",
     lambda a: "canberra" in norm(a) or "キャンベラ" in a),
    ("28日ある月(トラップ)",
     "How many months of the year have at least 28 days? Answer with the number only.",
     lambda a: has_num(a, "12")),
    ("3割引(JP計算)",
     "1000円の商品を3割引で買うと支払いはいくら？金額の数字だけで答えて。",
     lambda a: has_num(a, "700")),
    ("6の倍数証明(LLM判定)",
     PROOF_Q,
     judge_proof),
    ("fib実装(コード実行採点)",
     FIB_Q,
     grade_code_fib),
]

ALL = None  # setup 後に確定


def stage_summary():
    """直近の _TIMINGS を段階別に集計して文字列化し、バッファをクリア。"""
    agg = {}
    for label, model, sec in f._TIMINGS:
        agg.setdefault(label, [0.0, 0])
        agg[label][0] += sec
        agg[label][1] += 1
    f._TIMINGS.clear()
    parts = [f"{k}:{v[0]:.0f}s x{v[1]}" for k, v in agg.items()]
    return " | ".join(parts)


def run_baseline(q):
    # 単体ベースライン: qwen3:4b 直答（動的版が simple で選ぶ既定モデル）
    return f.strip_think(f.ask(
        "qwen3:4b",
        [{"role": "system", "content": f.PROPOSER_SYS},
         {"role": "user", "content": q}],
        f.PROPOSER_TEMP, label="single"))


def run_static(q, think):
    # 静的MoA: 全プロポーザー固定 + アグリゲータ, 1ラウンド
    old = f.PROPOSER_THINK
    f.PROPOSER_THINK = think
    try:
        props = f.get_proposals(ALL, q)
        final = f.aggregate(q, props)
    finally:
        f.PROPOSER_THINK = old
    return f.strip_think(final)


def run_dynamic(q):
    return f.strip_think(f.fugu_answer(q) or "")


CONFIGS = [
    ("a:単体qwen3",      run_baseline),
    ("b:静的MoA think=ON",  lambda q: run_static(q, None)),
    ("c:静的MoA think=OFF", lambda q: run_static(q, False)),
    ("d:動的Fugu",        run_dynamic),
]


def main():
    global ALL
    # 引数: `c=a,d` は構成フィルタ(プレフィクス一致)、それ以外はラベル部分一致の問題フィルタ。
    # 例: `python eval_fugu.py c=b,c 6の倍数` — 長時間のフル評価を分割・再開するために使う。
    cfg_sel = None
    tsel = []
    for s in sys.argv[1:]:
        if s.startswith("c="):
            cfg_sel = [x.strip() for x in s[2:].split(",") if x.strip()]
        else:
            tsel.append(s)
    configs = [c for c in CONFIGS
               if not cfg_sel or c[0].split(":")[0] in cfg_sel]
    tests = [t for t in TESTS if not tsel or any(s in t[0] for s in tsel)]
    if not tests:
        print("引数に一致する問題がありません:", tsel); return
    if not configs:
        print("c= に一致する構成がありません:", cfg_sel); return
    if not f.setup():
        print("setup 失敗"); return
    ALL = list(f.PROPOSERS)
    print(f"proposers(static)= {ALL}  baseline= qwen3:4b  aggregator= {f.AGGREGATOR}")
    print(f"tests= {[t[0] for t in tests]}  configs= {[c[0] for c in configs]}\n")
    results = {name: {"ok": 0, "sec": 0.0} for name, _ in configs}
    for label, q, grade in tests:
        print(f"\n########## {label} ##########")
        for cname, fn in configs:
            f._TIMINGS.clear()
            t0 = time.time()
            ans = fn(q)
            dt = time.time() - t0
            ok = False
            try:
                ok = bool(grade(ans))
            except Exception:
                ok = False
            results[cname]["ok"] += int(ok)
            results[cname]["sec"] += dt
            head = norm(ans)[:70]
            print(f"  [{cname:20}] {'OK ' if ok else 'NG '} {dt:6.1f}s  "
                  f"stages({stage_summary()})  ans='{head}'")
    n = len(tests)
    print("\n==================== SUMMARY ====================")
    print(f"{'config':22} {'acc':>8} {'total_sec':>10} {'avg_sec':>9}")
    for cname, _ in configs:
        r = results[cname]
        print(f"{cname:22} {r['ok']}/{n:<6} {r['sec']:>10.1f} {r['sec']/n:>9.1f}")


if __name__ == "__main__":
    main()
