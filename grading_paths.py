"""採点がどの段で通ったかを分解し、「+1 は緩い等価判定のおかげでは?」に数字で答える。

`regrade_nim.py` が「救済で何問増えたか」を数えるのに対し、こちらは
**同一の採点器を fugu と Fable の両方に当てて**、各問がどの段で正解と判定されたかを
段別に数える。段は厳しい順に:

    raw          生の文字列が完全一致
    ws           空白を潰すと一致
    norm         normalize_answer(決定的な表記正規化)で一致
    frac         Fraction による数値一致
    math_verify  sympy ベースの数式等価(唯一の非決定的・環境依存な段)
    code         HumanEval のみ。文字列ではなく公式テストの実行結果

`math_verify`/`frac` が 0 であることが重要で、それが 0 である限り
「スコアが sympy の気分に依存している」という批判は当たらない。
2026-08-20 実測では両側とも 0 だった(fugu 200 / Fable 199、下の想定出力を参照)。

想定出力(2026-08-20):
    TOTAL fugu  200  raw=157 ws=1 norm=13 code:OK=29  frac=0 math_verify=0
    TOTAL fable 199  raw=163 ws=1 norm=7  code:OK=28  frac=0 math_verify=0
    生の文字列一致のみ: fugu 186 / Fable 191

実行: python grading_paths.py
"""
import io
import json
import os
import re
import sys
from collections import Counter

import fugu_local as f

RB = os.path.expanduser("~/fugu_bench")

# score_now.py と同じ単一構成集合。和集合(オラクル)は絶対に取らない。
BASE = {
    "aime24":    ("sc@nim",    ["sc-fix@nim", "adapt@nim"]),
    "aime25":    ("sc@nim",    ["sc-fix@nim", "adapt@nim"]),
    "aime26":    ("sc@nim",    ["sc-fix@nim", "adapt@nim"]),
    "math500":   ("sc@nim",    ["sc-fix@nim", "adapt@nim"]),
    "humaneval": ("coder@nim", ["coder5@nim"]),
    "jmmlu":     ("fugu@nim",  ["scmcq@nim"]),
}
ORDER = ["raw", "ws", "norm", "frac", "math_verify", "code:OK", "code:NG", "NG", "NOANS"]
CREDIT = {"raw", "ws", "norm", "frac", "math_verify", "code:OK"}


def load(ds, cfg):
    p = os.path.join(RB, "results", f"{ds}__{cfg}.jsonl")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in io.open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["id"]] = r        # 再実行の重複は最後を採用(既存 report と同じ規約)
    return out


def path_of(got, exp):
    """answers_equivalent と同じ規則群を、通った段が分かる形に開いたもの。
    判定結果そのものは answers_equivalent と一致していなければならない
    (test_nim_offline.py の該当セクションで固定してある)。"""
    g, e = str(got), str(exp)
    if g == e:
        return "raw"
    if re.sub(r"\s+", "", g) == re.sub(r"\s+", "", e):
        return "ws"
    ng, ne = f.normalize_answer(g), f.normalize_answer(e)
    if not ng or not ne:
        return "NG"
    if ng.lower() == ne.lower():
        return "norm"
    if re.fullmatch(r"[A-Ea-e]", ng) and re.fullmatch(r"[A-Ea-e]", ne):
        return "NG"                 # 選択肢文字は決定的ガードで打ち切られる段
    try:
        from fractions import Fraction
        if Fraction(ng.replace(" ", "")) == Fraction(ne.replace(" ", "")):
            return "frac"
    except Exception:
        pass
    if f.answers_equivalent(g, e):
        return "math_verify"
    return "NG"


def score(items):
    c = Counter()
    for r in items.values():
        if r.get("task_type") == "code" or str(r.get("got")) in ("PASS", "FAIL"):
            c["code:OK" if r.get("correct") in (True, "True") else "code:NG"] += 1
            continue
        got, exp = r.get("got"), r.get("expected")
        if not got or exp is None or str(got) == "None":
            c["NOANS"] += 1
            continue
        c[path_of(got, exp)] += 1
    return c


def line(tag, c):
    ok = sum(v for k, v in c.items() if k in CREDIT)
    body = "  ".join(f"{k}={c[k]}" for k in ORDER if c[k])
    print(f"  {tag:6s} {ok:3d}  {body}")
    return ok


def main():
    fugu_all, fable_all = Counter(), Counter()
    for ds, (base, overs) in BASE.items():
        st = load(ds, base)
        if not st:
            print(f"■ {ds}: 基準構成 {base} の結果がない — スキップ")
            continue
        for cfg in overs:
            # 上書きは基準構成が測った問題集合の中だけ(データセットのずれ対策)
            st.update({k: v for k, v in load(ds, cfg).items() if k in st})
        fable = {k: v for k, v in load(ds, "fable").items() if k in st}
        print(f"■ {ds}")
        cf = score(st)
        fugu_all.update(cf)
        line("fugu", cf)
        if fable:
            cb = score(fable)
            fable_all.update(cb)
            line("fable", cb)
        else:
            print("   fable: 結果ファイルなし")

    print()
    total = {}
    for tag, c in (("fugu", fugu_all), ("fable", fable_all)):
        total[tag] = line("TOTAL " + tag, c)
        if c["math_verify"] or c["frac"]:
            print(f"    ! {tag}: スコアが数式等価判定に依存している "
                  f"(frac={c['frac']}, math_verify={c['math_verify']})")
    print()
    raw_only = {t: c["raw"] + c["code:OK"] for t, c in
                (("fugu", fugu_all), ("fable", fable_all))}
    print(f"  参考: 生の文字列一致のみで採点 "
          f"fugu {raw_only['fugu']} / Fable {raw_only['fable']} "
          f"({raw_only['fugu'] - raw_only['fable']:+d})")
    print("  ※ MATH-500 の正解表記は LaTeX、解答側は ASCII 混在なので、生の一致は")
    print("     どちらの系にとっても意味のある指標ではない(両側とも点を落とす)。")
    return total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
