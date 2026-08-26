"""厳密採点 — regrade_nim.py の対になるスクリプト（2026-08-26 追加）。

`docs/nim-benchmark-results.md` の caveat 2 が言う「厳密な文字列完全一致で採点し直すと
fugu 189 / Fable 199 に反転する」を再現し、**どの問題がどの段階で救済されているか**を出す。

採点を4段階に分解する（上ほど厳しい）:

  exact       got と expected が文字列として完全一致
  normalized  normalize_answer() を通すと一致（表記ゆれの救済）
  equivalent  さらに Fraction / math_verify で数学的に同値（answers_equivalent）
  wrong       どれでも一致しない

「寛容採点」= equivalent まで許す（＝現行）。「厳密採点」= exact のみ。
fugu と Fable の両方を同じ手続きで採点し、**IDの積集合**だけで比較する。

結果ファイルは書き換えない（生データ保存）。出力は --out で指定した JSON。

実行例:
    python regrade_strict.py --results ~/fugu_bench/results --out ./strict_regrade.json
"""
import argparse
import glob
import json
import os

import fugu_local as f

# データセット -> (fugu の結果ファイル glob, fable の結果ファイル)
PAIRS = [
    ("aime24",    "aime24__r0821@nim.jsonl",     "aime24__fable.jsonl"),
    ("aime25",    "aime25__r0821@nim.jsonl",     "aime25__fable.jsonl"),
    ("aime26",    "aime26__r0821@nim.jsonl",     "aime26__fable.jsonl"),
    ("math500",   "math500__r0821@nim.jsonl",    "math500__fable.jsonl"),
    ("humaneval", "humaneval__rc0821@nim.jsonl", "humaneval__fable.jsonl"),
    ("jmmlu",     "jmmlu__rq0821@nim.jsonl",     "jmmlu__fable.jsonl"),
]

# コード課題はテスト実行で採点されるため、文字列比較の対象外
CODE_DATASETS = {"humaneval"}


def load(path):
    """id -> record。再実行の重複は最後を採用（regrade_nim.py と同じ規約）。"""
    by_id = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_id[r["id"]] = r
    return by_id


def tier(got, expected):
    """救済がどの段階で効いたかを返す。"""
    g, e = ("" if got is None else str(got)), ("" if expected is None else str(expected))
    if g.strip() == e.strip():
        return "exact"
    ng, ne = f.normalize_answer(g), f.normalize_answer(e)
    if ng and ne and ng == ne:
        return "normalized"
    if f.answers_equivalent(g, e):
        return "equivalent"
    return "wrong"


def score(by_id, ids, dataset):
    """積集合 ids に対して段階別の内訳を返す。"""
    out = {"exact": 0, "normalized": 0, "equivalent": 0, "wrong": 0,
           "recorded_correct": 0, "n": len(ids), "detail": []}
    for i in sorted(ids):
        r = by_id[i]
        rec = bool(r.get("correct"))
        out["recorded_correct"] += int(rec)
        if dataset in CODE_DATASETS:
            # テスト実行で採点済み。文字列比較は行わず記録値をそのまま exact 扱いにする
            t = "exact" if rec else "wrong"
        else:
            t = tier(r.get("got"), r.get("expected"))
        out[t] += 1
        if t != "exact":
            out["detail"].append({
                "id": i, "tier": t,
                "expected": r.get("expected"), "got": r.get("got"),
                "recorded_correct": rec,
                "note": (r.get("note") or "")[:200],
            })
    out["strict"] = out["exact"]
    out["lenient"] = out["exact"] + out["normalized"] + out["equivalent"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.expanduser("~/fugu_bench/results"))
    ap.add_argument("--out", default="./strict_regrade.json")
    a = ap.parse_args()

    report = {"datasets": {}, "totals": {}}
    tot = {k: {"strict": 0, "lenient": 0, "recorded": 0, "n": 0}
           for k in ("fugu", "fable")}

    for ds, fugu_name, fable_name in PAIRS:
        pf = os.path.join(a.results, fugu_name)
        pb = os.path.join(a.results, fable_name)
        if not (os.path.exists(pf) and os.path.exists(pb)):
            print(f"[skip] {ds}: ファイルなし")
            continue
        A, B = load(pf), load(pb)
        ids = set(A) & set(B)                      # ★積集合だけで比較する
        sa, sb = score(A, ids, ds), score(B, ids, ds)
        report["datasets"][ds] = {
            "n_intersection": len(ids),
            "n_fugu_file": len(A), "n_fable_file": len(B),
            "fugu": sa, "fable": sb,
        }
        for key, s in (("fugu", sa), ("fable", sb)):
            tot[key]["strict"] += s["strict"]
            tot[key]["lenient"] += s["lenient"]
            tot[key]["recorded"] += s["recorded_correct"]
            tot[key]["n"] += s["n"]
        print(f"{ds:10} n={len(ids):3}  "
              f"fugu 厳密{sa['strict']:3} 寛容{sa['lenient']:3} 記録{sa['recorded_correct']:3}  |  "
              f"fable 厳密{sb['strict']:3} 寛容{sb['lenient']:3} 記録{sb['recorded_correct']:3}")

    report["totals"] = tot
    print("\n" + "=" * 72)
    for k in ("fugu", "fable"):
        t = tot[k]
        print(f"  {k:6} 厳密 {t['strict']:3}/{t['n']}   寛容 {t['lenient']:3}/{t['n']}   "
              f"記録 {t['recorded']:3}/{t['n']}")
    d_strict = tot["fugu"]["strict"] - tot["fable"]["strict"]
    d_len = tot["fugu"]["lenient"] - tot["fable"]["lenient"]
    print(f"\n  差分  厳密 {d_strict:+d}   寛容 {d_len:+d}")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
