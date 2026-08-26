# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""精度の再計測ランナー（2026-08-20）。

2026-08-03 に測った精度（AIME26 29/30 / AIME24 27/30 / AIME25 23/30 /
MATH-500 37/42）以降、fugu_local には大きな変更が入っている：

  * 思考予算のネイティブ制御（FUGU_THINKING_BUDGET → reasoning_budget /
    reasoning_effort。未設定なら従来どおり 16384 固定で挙動不変）
  * 速度モード（プランを外から渡して Conductor を省く経路）
  * コードモード
  * ジョブの並列化

これらの後で精度を測り直していないので、同じ物差しで測り直すためのランナー。

データセットの取得・読み込み・採点・実行構成は bench_fugu.py の実装をそのまま
import して使う（再実装しない）。このスクリプトが足すのは次の4点だけ：

  1. 中断・再開の粒度を上げる（1問ごとに追記。採点できた id だけを「済」とみなす）
  2. 1問ごとに「実際に使われたモデル・思考レベル・リトライ回数」を残す
  3. 429 / タイムアウト / 空応答は指数バックオフで再試行し、駄目なら【未採点】。
     誤答としては数えない（次回の再開で拾い直す）
  4. 2026-08-03 の基準値との比較表を出す（差の原因は断定しない）

使い方:
  python bench_recheck.py                                   # AIME26 のみ・sc@nim
  python bench_recheck.py --datasets aime26,aime24,aime25,math500 --config sc@nim
  python bench_recheck.py --dry-run                         # 走らせずに見積もりだけ
  python bench_recheck.py --report-only                     # 既存の結果から表だけ作る

API キーはコードにもファイルにも書かない。Windows のユーザー環境変数
NVIDIA_API_KEY（build.nvidia.com で生成した nvapi-...）から読む。
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import bench_fugu as bench
import fugu_local as f

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

# ==================================================
# 2026-08-03 の基準値
# ==================================================
# 【推定であることの明記】当時の結果ファイルは手元に残っていないため、構成名は
# コードとコメントからの推定である。根拠は RUN_BENCH.bat と最終報告にも記す：
#   * bench_queue.py NIM_QUEUE (2026-08-02) が aime26/25/24 を sc@nim 全問、
#     math500 を sc@nim limit=50 で回す並びになっている
#   * bench_fugu.py run_sc_budget (2026-08-07) の
#     「sc@nim の数学の失敗16件のうち13件は無回答」が
#     (30-29)+(30-27)+(30-23)+(42-37) = 1+3+7+5 = 16 と一致する
#   * MATH-500 の母数 42 は limit=50 のジョブが 42 問目まで進んだ状態と整合する
#   * bench_fugu.py run_sc3 が「通常 sc@nim(3系統・6/20票・16k開始)」と書いており、
#     当時の主線が sc@nim だったことを示す
# 2026-08-03 当日に 429 ストーム修正後の再測定用として sc2@nim が追加されている
# ので、無投票敗退した一部の問題は sc2@nim で取り直されている可能性がある。
BASELINE_DATE = "2026-08-03"
BASELINE_CONFIG = "sc@nim"          # 推定（上記コメントの根拠による）
#: dataset -> (正解数, 採点された問題数)
BASELINE = {
    "aime26": (29, 30),
    "aime24": (27, 30),
    "aime25": (23, 30),
    "math500": (37, 42),
}
#: 当時のジョブが使っていた --limit（bench_queue.NIM_QUEUE より）。None は全問。
BASELINE_LIMIT = {
    "aime26": None,
    "aime24": None,
    "aime25": None,
    "math500": 50,      # 50 問スライスのうち 42 問まで採点されている状態が 37/42
}
#: データ未取得でも --dry-run で試算できるようにするための既知の問題数。
#: 実データがあれば必ずそちらを使う（この値は表示上の目安）。
EXPECTED_SIZE = {"aime26": 30, "aime24": 30, "aime25": 30, "math500": 500}
#: 当時の SC 3系統（bench_fugu.run_sc_glm の「既存3系統」と run_sc3 の記述より）。
#: これも推定。当時の roster() を記録したファイルは残っていない。
BASELINE_SC_MODELS = [
    "deepseek-ai/deepseek-v4-pro",
    "openai/gpt-oss-120b",
    "minimaxai/minimax-m3",
]

# ==================================================
# 保存先
# ==================================================
# bench_fugu.RESULTS_DIR の直下ではなく recheck/ サブディレクトリに置く。理由は2つ：
#   * bench_fugu.report() が RESULTS_DIR.glob("*.jsonl") を舐めるので、先頭の
#     メタ行（"id" を持たない）を置くと report() が KeyError で落ちる
#   * 2026-08-03 の実測ファイルと同じ名前に追記すると、当時の記録が混ざって
#     「基準値」が基準でなくなる
RESULT_SUBDIR = "recheck"
REPORT_NAME = "recheck_report.md"


def results_root():
    """結果を書くディレクトリ。FUGU_RECHECK_DIR でテスト用に差し替えられる。"""
    override = (os.environ.get("FUGU_RECHECK_DIR") or "").strip()
    return Path(override) if override else (bench.RESULTS_DIR / RESULT_SUBDIR)


def results_path(dataset, config):
    return results_root() / f"{dataset}__{config}.jsonl"


def report_path():
    override = (os.environ.get("FUGU_RECHECK_DIR") or "").strip()
    return (Path(override) if override else bench.RESULTS_DIR) / REPORT_NAME


# ==================================================
# リトライ方針
# ==================================================
RETRY_MAX = 3                        # 1問あたりの再試行回数（初回を除く）
RETRY_BACKOFF = (20, 60, 180, 300)   # 指数バックオフ（秒）。超えた分は末尾を使う
SHUFFLE_SEED = 42                    # bench_fugu.run_bench と同じ並び順にする

_RE_RATE = re.compile(r"\b429\b|rate.?limit|too many requests", re.I)
_RE_TIMEOUT = re.compile(
    r"timed? ?out|timeout|stream idle|stream exceeded|\b(502|503|504)\b"
    r"|connection reset|connection aborted|remote end closed", re.I)


def transient_reason(text, answer_value, exc=None):
    """一過性の失敗なら理由文字列、そうでなければ None を返す（純粋関数）。

    「一過性」は 429 / タイムアウト / 空応答の3つ。fugu_local は失敗を
    "__ERROR__: ..." という文字列で返す契約なので、例外だけでなく本文も見る。
    空応答（solve_verifiable が有効票ゼロで None を返した場合を含む）を
    一過性に含めるのは、2026-08-03 の 429 ストームで無投票敗退が多発した
    実績があるため。誤答として数えず、次回の再開で取り直す。
    """
    blob = f"{type(exc).__name__}: {exc}" if exc is not None else (text or "")
    if _RE_RATE.search(blob):
        return "rate_limit"
    if _RE_TIMEOUT.search(blob):
        return "timeout"
    if exc is not None:
        return "error"
    if (text or "").startswith("__ERROR__"):
        return "error"
    if not (text or "").strip() and answer_value in (None, ""):
        return "empty"
    return None


def backoff_seconds(attempt):
    """attempt 回目の失敗直後に待つ秒数（1 起点）。"""
    idx = max(1, int(attempt)) - 1
    return RETRY_BACKOFF[min(idx, len(RETRY_BACKOFF) - 1)]


# ==================================================
# 1問ごとの計測（使われたモデル・リトライ回数）
# ==================================================
def _timing_mark():
    rows = getattr(f, "_TIMINGS", None)
    return len(rows) if isinstance(rows, list) else 0


def _models_since(mark):
    """mark 以降に ask() が呼んだモデル名（重複なし・呼ばれた順）と呼び出し回数。"""
    rows = getattr(f, "_TIMINGS", None)
    if not isinstance(rows, list):
        return [], 0
    used, calls = [], 0
    for row in rows[mark:]:
        calls += 1
        model = row[1] if isinstance(row, (list, tuple)) and len(row) > 1 else None
        if model and model not in used:
            used.append(model)
    return used, calls


def current_thinking_level():
    """いま有効な思考レベル。未設定なら ""（＝従来どおりの固定予算）。"""
    try:
        return f.thinking_level()
    except Exception:
        return (os.environ.get("FUGU_THINKING_BUDGET") or "").strip().lower()


def _unpack(out):
    """CONFIGS の戻り値 (text, answer_value, n_samples) を安全にほどく。"""
    if isinstance(out, tuple):
        vals = list(out) + [None, None, 0][len(out):]
        return vals[0], vals[1], vals[2] or 0
    return (out or ""), None, 0


def solve_item(item, runner, retries=RETRY_MAX, sleep_fn=time.sleep, log=None):
    """1問を解いて採点用の素材を返す。一過性の失敗だけ指数バックオフで再試行する。

    戻り値: dict(text, answer_value, n_samples, retries, reason, error, models,
                 model_calls, nim_requests, seconds)
    reason が None でなければ「未採点」。KeyboardInterrupt と SystemExit は
    握りつぶさずそのまま上へ通す（Ctrl-C と NIM 予算到達 rc=42 を殺さない）。
    """
    log = log or (lambda msg: None)
    attempt = 0
    t0 = time.time()
    mark = _timing_mark()
    req0 = getattr(f, "NIM_REQUEST_COUNT", 0)
    text, value, n_samples, reason, err = "", None, 0, None, None
    while True:
        attempt += 1
        try:
            text, value, n_samples = _unpack(runner(item))
            exc = None
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:                      # noqa: BLE001 - 記録して判定する
            text, value, n_samples, exc = "", None, 0, e
            err = f"{type(e).__name__}: {e}"
        reason = transient_reason(text, value, exc)
        if reason is None:
            err = None
            break
        if reason == "error" and exc is None and (text or "").startswith("__ERROR__"):
            err = text[:400]
        if attempt > retries:
            break
        wait = backoff_seconds(attempt)
        log(f"    再試行 {attempt}/{retries} ({reason}) → {wait}s 待機")
        sleep_fn(wait)
    models, calls = _models_since(mark)
    return {"text": text, "answer_value": value, "n_samples": n_samples,
            "retries": attempt - 1, "reason": reason, "error": err,
            "models": models, "model_calls": calls,
            "nim_requests": getattr(f, "NIM_REQUEST_COUNT", 0) - req0,
            "seconds": round(time.time() - t0, 1)}


def build_record(item, dataset, config, solved):
    """solve_item の結果を採点して JSONL 1行分の dict にする。"""
    rec = {"id": item["id"], "dataset": dataset, "config": config,
           "expected": item.get("answer"),
           "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "seconds": solved["seconds"],
           "models": solved["models"],
           "model_calls": solved["model_calls"],
           "nim_requests": solved["nim_requests"],
           "thinking_level": current_thinking_level(),
           "retries": solved["retries"],
           "n_samples": solved["n_samples"]}
    if solved["reason"] is not None:
        # 【未採点】誤答ではない。次回の再開で拾い直す。
        rec.update({"graded": False, "correct": None, "got": None,
                    "status": "ungraded", "reason": solved["reason"],
                    "error": solved["error"],
                    "answer_text": (solved["text"] or "")[-1000:]})
        return rec
    correct, got, note = bench.grade_item(
        item, answer_text=solved["text"], answer_value=solved["answer_value"])
    rec.update({"graded": True, "correct": bool(correct), "got": got, "note": note,
                "status": "ok", "answer_text": (solved["text"] or "")[-4000:]})
    return rec


# ==================================================
# JSONL の読み書き（メタ行つき・再開対応）
# ==================================================
def read_records(path):
    """(meta 行のリスト, 結果行のリスト) を返す。壊れた行は黙って捨てる。"""
    path = Path(path)
    metas, recs = [], []
    if not path.exists():
        return metas, recs
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        (metas if obj.get("_meta") else recs).append(obj)
    return metas, recs


def graded_ids(path):
    """採点まで終わった id の集合。未採点（status=ungraded）は含めない。"""
    _, recs = read_records(path)
    done = set()
    for r in recs:
        if r.get("graded") and r.get("id"):
            done.add(r["id"])
    return done


def append_line(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()


def write_meta(path, config, dataset, roster, extra=None):
    """布陣メタ行を書く。新規ファイルなら先頭行になる。"""
    meta = {"_meta": True, "kind": "roster", "dataset": dataset, "config": config,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_date": BASELINE_DATE, "baseline_config": BASELINE_CONFIG,
            "thinking_level": current_thinking_level(),
            "thinking_budget_env": os.environ.get("FUGU_THINKING_BUDGET", ""),
            "roster": roster}
    if extra:
        meta.update(extra)
    append_line(path, meta)
    return meta


# ==================================================
# 集計と比較表
# ==================================================
def summarize(recs):
    """同一 id は最後の記録を採用して集計する（bench_fugu.report と同じ方針）。"""
    by_id = {}
    for r in recs:
        if r.get("id"):
            by_id[r["id"]] = r
    rows = list(by_id.values())
    graded = [r for r in rows if r.get("graded")]
    ungraded = [r for r in rows if not r.get("graded")]
    reasons = {}
    for r in ungraded:
        key = r.get("reason") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    secs = [r.get("seconds") or 0 for r in rows]
    models = []
    for r in rows:
        for m in r.get("models") or []:
            if m not in models:
                models.append(m)
    return {"graded": len(graded),
            "correct": sum(1 for r in graded if r.get("correct")),
            "ungraded": len(ungraded), "ungraded_reasons": reasons,
            "attempted": len(rows), "seconds": round(sum(secs), 1),
            "retries": sum(int(r.get("retries") or 0) for r in rows),
            "models": models}


def _pct(ok, n):
    return (ok / n * 100.0) if n else 0.0


def compare_rows(summaries):
    """{dataset: summarize()} を 2026-08-03 の基準値と突き合わせて行を作る。"""
    rows = []
    for ds, s in summaries.items():
        base = BASELINE.get(ds)
        now_ok, now_n = s["correct"], s["graded"]
        # 未採点を誤答として数えた場合（当時の sc@nim は無回答を失敗に数えていた）
        worst_n = now_n + s["ungraded"]
        row = {"dataset": ds,
               "base_ok": base[0] if base else None,
               "base_n": base[1] if base else None,
               "base_pct": _pct(*base) if base else None,
               "now_ok": now_ok, "now_n": now_n, "now_pct": _pct(now_ok, now_n),
               "worst_ok": now_ok, "worst_n": worst_n,
               "worst_pct": _pct(now_ok, worst_n),
               "ungraded": s["ungraded"], "ungraded_reasons": s["ungraded_reasons"],
               "seconds": s["seconds"], "retries": s["retries"],
               "models": s["models"]}
        if base:
            row["d_ok"] = now_ok - base[0]
            row["d_pct"] = row["now_pct"] - row["base_pct"]
            row["n_mismatch"] = (now_n != base[1])
        else:
            row["d_ok"] = row["d_pct"] = None
            row["n_mismatch"] = False
        rows.append(row)
    return rows


def _fmt_hms(sec):
    sec = int(round(sec or 0))
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def _width(text):
    """端末での表示幅（全角=2）。日本語見出しの桁ずれを防ぐ。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1
               for ch in str(text))


def _pad(text, width, right=True):
    text = str(text)
    fill = " " * max(0, width - _width(text))
    return (fill + text) if right else (text + fill)


#: 比較表の (見出し, 桁数, 右寄せか)
_TABLE_COLS = (("データセット", 12, False),
               (f"基準({BASELINE_DATE})", 18, True),
               ("今回(採点分)", 18, True),
               ("差分", 14, True),
               ("参考:未採点=誤答", 20, True),
               ("未採点", 8, True),
               ("所要", 10, True))


def render_table(rows):
    """比較表を文字列で返す（標準出力とレポートで同じものを使う）。"""
    head = " ".join(_pad(t, w, r) for t, w, r in _TABLE_COLS)
    out = [head, "-" * _width(head)]
    for r in rows:
        base = (f"{r['base_ok']}/{r['base_n']} ({r['base_pct']:.1f}%)"
                if r["base_ok"] is not None else "-")
        now = f"{r['now_ok']}/{r['now_n']} ({r['now_pct']:.1f}%)" if r["now_n"] else "-"
        if r["d_ok"] is None or not r["now_n"]:
            diff = "-"
        else:
            # 母数が基準と違うときは * を付ける。正解数の差をそのまま比べられない。
            diff = (f"{r['d_ok']:+d} ({r['d_pct']:+.1f}pt)"
                    + ("*" if r["n_mismatch"] else ""))
        worst = (f"{r['worst_ok']}/{r['worst_n']} ({r['worst_pct']:.1f}%)"
                 if r["worst_n"] else "-")
        cells = (r["dataset"], base, now, diff, worst, str(r["ungraded"]),
                 _fmt_hms(r["seconds"]))
        out.append(" ".join(_pad(c, w, rt)
                            for c, (_t, w, rt) in zip(cells, _TABLE_COLS)))
    if any(r["n_mismatch"] and r["now_n"] for r in rows):
        out.append("* 母数が基準と違う行。正解数の差ではなくパーセントで見ること。")
    return "\n".join(out)


def roster_diff(now_models, base_models=None):
    """SC 系統の布陣差分 (消えた, 増えた, 共通) を返す。"""
    base = list(base_models if base_models is not None else BASELINE_SC_MODELS)
    now = list(now_models or [])
    gone = [m for m in base if m not in now]
    added = [m for m in now if m not in base]
    same = [m for m in base if m in now]
    return {"gone": gone, "added": added, "same": same}


def render_report_md(rows, config, roster, started, finished):
    """判断材料を並べた Markdown。差の原因は断定しない。"""
    sc_now = list((roster or {}).get("sc_models") or [])
    diff = roster_diff(sc_now)
    total_ok = sum(r["now_ok"] for r in rows)
    total_n = sum(r["now_n"] for r in rows)
    total_un = sum(r["ungraded"] for r in rows)
    total_sec = sum(r["seconds"] for r in rows)
    lines = [
        "# 精度の再計測レポート",
        "",
        f"- 生成: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 構成: `{config}`",
        f"- 実行: {started} 〜 {finished}",
        f"- 思考レベル: `{current_thinking_level() or '(未設定＝従来どおり固定予算)'}`"
        f"  (FUGU_THINKING_BUDGET=`{os.environ.get('FUGU_THINKING_BUDGET', '')}`)",
        f"- 基準: {BASELINE_DATE} の実測（構成は `{BASELINE_CONFIG}` と**推定**。"
        "当時の結果ファイルは残っていない）",
        "",
        "## 比較",
        "",
        "```",
        render_table(rows),
        "```",
        "",
        f"合計: 採点 {total_ok}/{total_n} ({_pct(total_ok, total_n):.1f}%) / "
        f"未採点 {total_un} 問 / 実測所要 {_fmt_hms(total_sec)}",
        "",
        "「参考:未採点=誤答」列は、未採点を全部誤答として数えた場合の値。"
        f"{BASELINE_DATE} の実測は無回答を失敗として数えていた"
        "（bench_fugu.py run_sc_budget の「失敗16件のうち13件は無回答」）ので、"
        "基準値と同じ数え方に揃えたいときはこちらを見る。",
        "",
        "## 判断材料",
        "",
        "差が出ていた場合、その原因が「布陣の違い」「変更の影響」「単なる分散」の"
        "どれなのかは、この表だけでは決められない。判断に使える材料を並べる。",
        "",
        "### 1. 布陣（SC 投票系統）",
        "",
        f"- 今回: {', '.join(sc_now) if sc_now else '(未記録)'}",
        f"- 基準（推定）: {', '.join(BASELINE_SC_MODELS)}",
        f"- 消えた: {', '.join(diff['gone']) if diff['gone'] else 'なし'}",
        f"- 増えた: {', '.join(diff['added']) if diff['added'] else 'なし'}",
        f"- 共通: {', '.join(diff['same']) if diff['same'] else 'なし'}",
        "",
        "NIM の布陣は起動ごとの生存プローブで決まる（fugu_local.apply_nim_profile）ため、"
        f"{BASELINE_DATE} と顔ぶれが違っていても異常ではない。"
        "基準側の顔ぶれはコード中のコメントからの**推定**であり、実測記録ではない。",
        "",
        "### 2. 未採点の数",
        "",
    ]
    for r in rows:
        reasons = ", ".join(f"{k}={v}" for k, v in
                            sorted(r["ungraded_reasons"].items())) or "なし"
        lines.append(f"- {r['dataset']}: 未採点 {r['ungraded']} 問 ({reasons}) / "
                     f"リトライ計 {r['retries']} 回")
    lines += [
        "",
        "未採点が多いほど、今回の値は「実力」よりも「通信の当たり外れ」を"
        "反映している可能性が上がる。再実行すると未採点分だけ拾い直せる。",
        "",
        "### 3. 所要時間",
        "",
    ]
    for r in rows:
        n_try = r["now_n"] + r["ungraded"]
        per = (r["seconds"] / n_try) if n_try else 0
        lines.append(f"- {r['dataset']}: 合計 {_fmt_hms(r['seconds'])} / "
                     f"1問あたり {per:.0f} 秒")
    lines += [
        "",
        "1問あたりの時間が基準当時より大きく短ければ、思考予算やサンプル数が"
        "実効的に減っている可能性がある（逆も同様）。ただしクラウド側の混雑でも"
        "変わるので、単独では根拠にならない。",
        "",
        "### 4. 基準以降に入った主な変更",
        "",
        "- 思考予算のネイティブ制御（FUGU_THINKING_BUDGET → reasoning_budget /"
        " reasoning_effort）。未設定なら従来の 16384 固定のままで、この経路には入らない。",
        "- 速度モード（プランを外から渡して Conductor を省く経路）",
        "- コードモード",
        "- ジョブの並列化",
        "",
        "上の思考レベル欄が空なら、少なくとも 1 番目の変更は今回の測定では"
        "有効になっていない。",
        "",
        "### 5. 分散について",
        "",
        "SC は temperature 0.7 のサンプリングを多数決するので、同じ構成・同じ布陣でも"
        "走るたびに結果は動く。30 問での ±1〜2 問の差は、この表だけでは"
        "分散と区別できない。区別したければ同一条件でもう一度走らせて比較する。",
        "",
        "## 生データ",
        "",
    ]
    for r in rows:
        lines.append(f"- `{results_path(r['dataset'], config)}`")
    return "\n".join(lines) + "\n"


# ==================================================
# 所要時間の見積もり
# ==================================================
# 1 波 = SC_WORKERS 本の並列サンプリングが 1 回終わるまで。
# 秒数の根拠: NIM の思考モデルは reasoning_budget=16384 を上限に生成するので、
# 実効 40〜80 tok/s なら 1 本 60〜300 秒。fugu_local の NIM_STREAM_MAX_S の
# コメントは「minimax の正当な長考は ~30分」と実測を記しており、外れ値は
# これよりさらに長い。幅を持たせて出す。
WAVE_SEC_LOW = float(os.environ.get("FUGU_RECHECK_WAVE_LOW") or 60)
WAVE_SEC_MID = float(os.environ.get("FUGU_RECHECK_WAVE_SEC") or 120)
WAVE_SEC_HIGH = float(os.environ.get("FUGU_RECHECK_WAVE_HIGH") or 300)
#: 第2バッチ以降に入る問題の割合。bench_fugu.run_adapt の
#: 「影響範囲は第2バッチ以降に入った問題(AIME 90問中31問)に限定される」より。
HARD_RATIO = 31 / 90
HARD_EXTRA_BATCHES = 2              # 割れた問題が追加バッチを回す典型回数
#: 1回しかモデルを呼ばない構成（SC ではないので波は 1）。
SINGLE_SHOT = ("think", "think@nim", "vibe", "coder1")


def estimate_waves(config):
    """構成と fugu_local の SC 設定から (易問の波数, 難問の波数, 平均波数) を出す。"""
    if config in SINGLE_SHOT:
        return 1.0, 1.0, 1.0
    initial = max(1, int(getattr(f, "SC_INITIAL", 6) or 6))
    step = max(1, int(getattr(f, "SC_STEP", 4) or 4))
    workers = max(1, int(getattr(f, "SC_WORKERS", 4) or 4))
    pot = 1 if getattr(f, "SC_POT", True) else 0
    easy = math.ceil((initial + pot) / workers)
    extra = math.ceil((step + pot) / workers) * HARD_EXTRA_BATCHES
    hard = easy + extra
    avg = easy * (1 - HARD_RATIO) + hard * HARD_RATIO
    return float(easy), float(hard), float(avg)


def estimate_seconds(n_items, config):
    """(低め, 中央, 高め) の推定所要秒。"""
    _easy, _hard, avg = estimate_waves(config)
    return (n_items * avg * WAVE_SEC_LOW,
            n_items * avg * WAVE_SEC_MID,
            n_items * avg * WAVE_SEC_HIGH)


# ==================================================
# 実行
# ==================================================
def effective_limit(dataset, limit):
    """--limit 未指定なら 2026-08-03 のジョブと同じスライスに合わせる。"""
    if limit is None:
        return BASELINE_LIMIT.get(dataset)
    return None if limit <= 0 else limit


def select_items(dataset, limit=None, items=None):
    """bench_fugu.run_bench と同じ決定的シャッフルで対象問題を選ぶ。"""
    src = list(items) if items is not None else list(bench.load_items(dataset))
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(src)
    return src[:limit] if limit else src


def collect_roster():
    """いまの布陣を記録用の dict にする。副作用なし。"""
    try:
        r = f.roster()
    except Exception as e:                          # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    r = dict(r)
    r["sc_models"] = list(getattr(f, "REASONING_MODELS", []) or [])
    r["sc_initial"] = getattr(f, "SC_INITIAL", None)
    r["sc_step"] = getattr(f, "SC_STEP", None)
    r["sc_max"] = getattr(f, "SC_MAX", None)
    r["sc_workers"] = getattr(f, "SC_WORKERS", None)
    r["nim_url"] = getattr(f, "NIM_URL", "")
    return r


def require_api_key(config):
    """@nim 構成に必要な API キーの確認。無ければ分かりやすく落とす。"""
    if not str(config).endswith("@nim"):
        return
    if getattr(f, "NIM_API_KEY", "") or os.environ.get("NVIDIA_API_KEY"):
        return
    raise SystemExit(
        "\n環境変数 NVIDIA_API_KEY が設定されていません。\n"
        f"  構成 {config} は NVIDIA NIM クラウドを使うので API キーが要ります。\n\n"
        "  1) https://build.nvidia.com/ でキー (nvapi-... ) を発行する\n"
        "  2) PowerShell で次を1回実行（ユーザー環境変数に保存されます）:\n"
        "       setx NVIDIA_API_KEY \"nvapi-ここに貼る\"\n"
        "  3) PowerShell を開き直してから、もう一度このコマンドを実行する\n\n"
        "  ※ キーはコードにもファイルにも書かないこと。\n")


def prepare_backend(config):
    """@nim 構成なら setup() より前にバックエンドを切り替える（run_bench と同じ手順）。"""
    if str(config).endswith("@nim"):
        f.FUGU_BACKEND = "nim"
    elif getattr(f, "FUGU_BACKEND", "") == "nim":
        raise SystemExit(
            f"FUGU_BACKEND=nim ですが構成 {config} は @nim ではありません。"
            "（ローカル結果の汚染防止のため中止）")


def run_dataset(dataset, config, limit=None, items=None, runner=None,
                retries=RETRY_MAX, sleep_fn=time.sleep, out_path=None,
                roster=None, quiet=False, on_record=None):
    """1データセットを走らせる。1問ごとに追記し、済みの id は飛ばす。

    テストからは items / runner / roster / sleep_fn を差し替えて、
    ネットワークにもモデルにも触らずに全経路を通せる。
    """
    runner = runner or bench.CONFIGS[config]
    path = Path(out_path) if out_path else results_path(dataset, config)
    targets = select_items(dataset, limit=limit, items=items)
    done = graded_ids(path)
    todo = [it for it in targets if it["id"] not in done]

    def say(msg):
        if not quiet:
            print(msg, flush=True)

    metas, _ = read_records(path)
    write_meta(path, config, dataset, roster if roster is not None else collect_roster(),
               extra={"session": len(metas) + 1, "todo": len(todo),
                      "targets": len(targets)})
    say(f"\n=== {dataset} x {config}: 残り {len(todo)} 問"
        f"（対象 {len(targets)} / 採点済 {len(targets) - len(todo)}）===")
    t_batch = time.time()
    ok = graded = 0
    for k, item in enumerate(todo, 1):
        solved = solve_item(item, runner, retries=retries, sleep_fn=sleep_fn,
                            log=say)
        rec = build_record(item, dataset, config, solved)
        append_line(path, rec)
        if on_record:
            on_record(rec)
        if rec["graded"]:
            graded += 1
            ok += int(bool(rec["correct"]))
            mark = "OK " if rec["correct"] else "NG "
        else:
            mark = "未採点"
        elapsed = time.time() - t_batch
        per = elapsed / k
        eta = per * (len(todo) - k)
        say(f"[{k}/{len(todo)}] {item['id']} {mark} "
            f"got={rec.get('got')} exp={item.get('answer')} "
            f"正解 {ok}/{graded} ({_pct(ok, graded):.1f}%) "
            f"経過 {_fmt_hms(elapsed)} 残り≈{_fmt_hms(eta)} "
            f"({rec['seconds']:.0f}s/問 retry={rec['retries']})")
    _, recs = read_records(path)
    return summarize(recs)


def dry_run(datasets, config, limit):
    """何も書き込まずに「何問走るか・どこに書くか・推定所要時間」だけ出す。"""
    print("[dry-run] 何も書き込みません", flush=True)
    print(f"  構成      : {config}"
          f"{'  ← ' + BASELINE_DATE + ' の推定構成' if config == BASELINE_CONFIG else ''}")
    print(f"  思考レベル: {current_thinking_level() or '(未設定＝従来どおり固定予算)'}")
    print(f"  レポート  : {report_path()}")
    if not (getattr(f, "NIM_API_KEY", "") or os.environ.get("NVIDIA_API_KEY")) \
            and config.endswith("@nim"):
        print("  ※ NVIDIA_API_KEY が未設定です。実行するには設定が必要です。")
    total, lo, mid, hi = 0, 0.0, 0.0, 0.0
    for ds in datasets:
        eff = effective_limit(ds, limit)
        path = results_path(ds, config)
        done = graded_ids(path)
        try:
            items = select_items(ds, limit=eff)
            todo = [it for it in items if it["id"] not in done]
            note = ""
        except SystemExit:
            n = EXPECTED_SIZE.get(ds, 0)
            items = [None] * (min(n, eff) if eff else n)
            todo = list(items)
            note = "  ※データ未取得。既知の問題数で試算しています"
            print(f"  {ds:<9}: データ未取得 → "
                  f"python bench_fugu.py download {ds}")
        a, b, c = estimate_seconds(len(todo), config)
        total, lo, mid, hi = total + len(todo), lo + a, mid + b, hi + c
        base = BASELINE.get(ds)
        base_s = f"  基準 {base[0]}/{base[1]}" if base else ""
        print(f"  {ds:<9}: 走らせるのは {len(todo)} 問 "
              f"（対象 {len(items)} / 採点済 {len(items) - len(todo)}"
              f"{' / limit=' + str(eff) if eff else ''}）{base_s}{note}")
        print(f"             書き込み先: {path}")
        print(f"             推定所要  : {_fmt_hms(b)}（幅 {_fmt_hms(a)}〜{_fmt_hms(c)}）")
    easy, hard, avg = estimate_waves(config)
    print(f"  合計      : {total} 問 / 推定 {_fmt_hms(mid)}"
          f"（幅 {_fmt_hms(lo)}〜{_fmt_hms(hi)}）")
    print(f"  見積の根拠: 1問 {avg:.1f} 波（易 {easy:.0f} / 難 {hard:.0f}・"
          f"難問率 {HARD_RATIO * 100:.0f}%）× 1波 {WAVE_SEC_MID:.0f}秒。"
          f"SC_INITIAL={getattr(f, 'SC_INITIAL', '?')} "
          f"SC_STEP={getattr(f, 'SC_STEP', '?')} "
          f"SC_WORKERS={getattr(f, 'SC_WORKERS', '?')} より算出", flush=True)
    return total


def load_summaries(datasets, config):
    """既存の JSONL から集計だけ作る（--report-only 用）。"""
    out = {}
    for ds in datasets:
        _, recs = read_records(results_path(ds, config))
        if recs:
            out[ds] = summarize(recs)
    return out


def emit_report(summaries, config, roster, started, finished, path=None):
    rows = compare_rows(summaries)
    table = render_table(rows)
    print("\n===== 2026-08-03 との比較 =====", flush=True)
    print(table, flush=True)
    md = render_report_md(rows, config, roster, started, finished)
    path = Path(path) if path else report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    print(f"\nレポート: {path}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=f"fugu 精度の再計測（{BASELINE_DATE} の実測と比較する）")
    ap.add_argument("--datasets", default="aime26",
                    help="カンマ区切り。既定は aime26（30問・一番重要）")
    ap.add_argument("--config", default=BASELINE_CONFIG,
                    help=f"bench_fugu.CONFIGS の構成名。既定 {BASELINE_CONFIG}"
                         f"（{BASELINE_DATE} の推定構成）")
    ap.add_argument("--limit", type=int, default=None,
                    help="1データセットあたりの問題数（0 で全問）。"
                         "未指定なら基準当時と同じスライス")
    ap.add_argument("--retries", type=int, default=RETRY_MAX,
                    help=f"一過性失敗の再試行回数（既定 {RETRY_MAX}）")
    ap.add_argument("--dry-run", action="store_true",
                    help="実際には叩かず、問題数・書き込み先・推定所要時間だけ出す")
    ap.add_argument("--report-only", action="store_true",
                    help="既存の結果ファイルから比較表とレポートだけ作り直す")
    args = ap.parse_args(argv)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    unknown = [d for d in datasets if d not in bench.DATASETS]
    if unknown:
        raise SystemExit(f"未知のデータセット: {unknown} "
                         f"(選べるもの: {list(bench.DATASETS)})")
    if args.config not in bench.CONFIGS:
        raise SystemExit(f"未知の構成: {args.config} "
                         f"(選べるもの: {sorted(bench.CONFIGS)})")

    if args.dry_run:
        dry_run(datasets, args.config, args.limit)
        return 0

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.report_only:
        summaries = load_summaries(datasets, args.config)
        if not summaries:
            print("結果ファイルがまだありません。", flush=True)
            return 1
        emit_report(summaries, args.config, collect_roster(), started,
                    time.strftime("%Y-%m-%d %H:%M:%S"))
        return 0

    require_api_key(args.config)
    prepare_backend(args.config)
    if not f.setup():
        raise SystemExit("fugu setup 失敗（NIM 構成なら API キーと疎通を確認）")
    f.SHOW_PLAN = False
    f.SHOW_PROPOSALS = False
    f.SHOW_TIMING = True        # 1問ごとに「使われたモデル」を拾うために必要
    roster = collect_roster()
    print(f"[recheck] 布陣: SC={', '.join(roster.get('sc_models') or []) or '(不明)'}"
          f" / aggregator={roster.get('aggregator')}", flush=True)
    diff = roster_diff(roster.get("sc_models"))
    if diff["gone"] or diff["added"]:
        print(f"[recheck] {BASELINE_DATE}（推定）からの布陣差分: "
              f"消えた={diff['gone'] or 'なし'} 増えた={diff['added'] or 'なし'}",
              flush=True)

    summaries = {}
    try:
        for ds in datasets:
            summaries[ds] = run_dataset(ds, args.config,
                                        limit=effective_limit(ds, args.limit),
                                        retries=args.retries, roster=roster)
    except KeyboardInterrupt:
        print("\n[recheck] 中断しました。結果は保存済みです。"
              "同じコマンドで再実行すると続きから走ります。", flush=True)
        summaries.update(load_summaries([d for d in datasets if d not in summaries],
                                        args.config))
    emit_report(summaries, args.config, roster, started,
                time.strftime("%Y-%m-%d %H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
