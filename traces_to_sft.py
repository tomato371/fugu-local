"""FUGU_TRACE_DIR に溜めた SC トレースを rejection sampling の学習データに変換する。

`solve_verifiable` が残す 1 問 1 行の JSONL には、その問題に対する全サンプル
(推論本文つき)と、各サンプルが最終答えと同値かどうかの `agrees` が入っている。
ここではそれを「多数決で検証された答えに到達したトレースだけ」に絞り、
chat 形式の SFT データセットへ落とす。

    python traces_to_sft.py --out sft.jsonl                 # ~/fugu_traces を変換
    python traces_to_sft.py IN_DIR --out sft.jsonl --min-votes 4 --per-question 2

採否の既定:
  * agrees=True のサンプルのみ(最終答えと同値に到達したもの)
  * PoT(コード実行)サンプルは除外 — 学習させたいのは自然言語の推論であって、
    stdout 最終行を答えとする経路ではない(--keep-pot で残せる)
  * 死票・__ERROR__・空本文は除外
  * 同一問題から採るのは既定 1 本(--per-question)。同じ問題の似た推論を何本も
    入れると、その問題だけが過剰に重みを持つ
  * 票が割れた問題は既定で除外しない(--min-votes で下限を上げられる)。
    多数決が薄い問題ほど答え自体が疑わしいので、厳しく学習させたいときに使う

**ライセンス**: 出力は NIM 上のモデル群が書いた本文である。学習に使う前に
docs/distillation-licensing.md の各モデルの条項を確認すること。
"""
import argparse
import glob
import json
import os
import re
import sys

SYS_PROMPTS = {}
try:
    import fugu_local as f
    SYS_PROMPTS = {"math": f.SC_PROMPT_MATH, "mcq": f.SC_PROMPT_MCQ, "pot": f.SC_PROMPT_POT}
except Exception:                                   # 依存なしでも動かす(集計だけなら不要)
    pass

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def usable(s, keep_pot):
    if not s.get("agrees"):
        return False
    if s.get("pot") and not keep_pot:
        return False
    if s.get("relayed"):                            # 継投票は本文を持たない
        return False
    text = (s.get("text") or "").strip()
    return bool(text) and not text.startswith("__ERROR__")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", nargs="?", default=os.path.expanduser("~/fugu_traces"),
                    help="トレース JSONL のあるディレクトリ (既定: ~/fugu_traces)")
    ap.add_argument("--out", required=True, help="出力する SFT データセット (JSONL)")
    ap.add_argument("--min-votes", type=int, default=0,
                    help="最終答えの得票数の下限。薄い多数決の問題を捨てる")
    ap.add_argument("--per-question", type=int, default=1,
                    help="1 問から採るトレース数の上限 (既定 1)")
    ap.add_argument("--keep-pot", action="store_true", help="PoT サンプルも採用する")
    ap.add_argument("--exclude-model", default="",
                    help="この正規表現に一致するモデルの出力を除外する。"
                         "ライセンス条項が伝播する系統を落とすのに使う "
                         "(docs/distillation-licensing.md)。例: 'gemma|minimax'")
    ap.add_argument("--strip-think", action="store_true",
                    help="<think>...</think> を落として最終本文だけを学習対象にする")
    ap.add_argument("--dry-run", action="store_true", help="集計だけして書き出さない")
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.indir, "*.jsonl")))
    if not files:
        print(f"トレースが無い: {a.indir}", file=sys.stderr)
        return 1

    excl = re.compile(a.exclude_model, re.I) if a.exclude_model else None
    stat = {"問題": 0, "採用": 0, "無回答で除外": 0, "薄票で除外": 0,
            "重複で除外": 0, "不採用サンプル": 0, "ライセンス除外": 0}
    seen = set()
    rows = []
    for path in files:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:                      # 書き込み途中の行は飛ばす
                continue
            stat["問題"] += 1
            final = rec.get("final")
            if final is None:
                stat["無回答で除外"] += 1
                continue
            votes = rec.get("votes") or {}
            if a.min_votes and max(list(votes.values()) or [0]) < a.min_votes:
                stat["薄票で除外"] += 1
                continue
            q = rec.get("question") or ""
            if q in seen:
                stat["重複で除外"] += 1
                continue
            seen.add(q)
            cands = [s for s in rec.get("samples", []) if usable(s, a.keep_pot)]
            if excl:
                kept = [s for s in cands if not excl.search(s.get("model") or "")]
                stat["ライセンス除外"] += len(cands) - len(kept)
                cands = kept
            stat["不採用サンプル"] += len(rec.get("samples", [])) - len(cands)
            # 自信度の高い順(conf は平均 logprob なので 0 に近いほど高い)
            cands.sort(key=lambda s: (s.get("conf") is None, -(s.get("conf") or 0)))
            for s in cands[:max(1, a.per_question)]:
                text = s["text"]
                if a.strip_think:
                    text = THINK_RE.sub("", text).strip()
                sysp = SYS_PROMPTS.get(s.get("sys") or "math", "")
                msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [
                    {"role": "user", "content": s.get("prompt") or q},
                    {"role": "assistant", "content": text}]
                rows.append({"messages": msgs, "answer": final,
                             "source_model": s.get("model"), "task_type": rec.get("task_type"),
                             "votes": max(list(votes.values()) or [0])})
                stat["採用"] += 1

    for k, v in stat.items():
        print(f"  {k}: {v}")
    if a.dry_run:
        print("(--dry-run のため書き出さない)")
        return 0
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False), file=fh)
    print(f"書き出し: {a.out} ({len(rows)} 件)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
