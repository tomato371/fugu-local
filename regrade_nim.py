"""NIM ベンチ結果を改善版 answers_equivalent で再採点し、表記ゆれ起因の NG を救済集計する。
結果ファイル自体は書き換えない(生データ保存)。集計は ~/fugu_bench/regrade_summary.json へ。
実行: python regrade_nim.py
"""
import glob
import json
import os

import fugu_local as f

summary = {}
for path in sorted(glob.glob(os.path.expanduser("~/fugu_bench/results/*__*@nim.jsonl"))):
    name = os.path.basename(path).replace(".jsonl", "")
    by_id = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        by_id[r["id"]] = r          # 再実行の重複は最後を採用(既存reportと同じ規約)
    flips = []
    ok = 0
    for r in by_id.values():
        c = r.get("correct", False)
        if not c and r.get("got") not in (None, ""):
            if f.answers_equivalent(str(r["got"]), str(r["expected"])):
                flips.append(f"{r['id']}: {r['got']} == {r['expected']}")
                c = True
        ok += int(c)
    summary[name] = {"ok": ok, "n": len(by_id), "flips": flips}
    fl = ("  FLIPS: " + "; ".join(flips)) if flips else ""
    print(f"{name}: {ok}/{len(by_id)}{fl}")

out = os.path.expanduser("~/fugu_bench/regrade_summary.json")
json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved:", out)
