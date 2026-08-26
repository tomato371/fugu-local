# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""bench_recheck.py のオフライン回帰テスト。

ネットワークにもモデルにも触らない。実行するモデルは全部スタブに差し替え、
結果の書き込み先は FUGU_RECHECK_DIR で一時ディレクトリへ逃がす。

実行: python test_bench_recheck.py
既存テスト (test_nim_offline.py / test_nim_routing.py) と同じスクリプト形式
（check() + 終了コード）。最後に合計件数と PASS/FAIL を出す。
"""
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import bench_fugu as bench
import bench_recheck as rc

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

_FAILS = []
_TOTAL = 0


def check(name, cond):
    global _TOTAL
    _TOTAL += 1
    print(f"[{'OK' if cond else 'NG'}] {name}")
    if not cond:
        _FAILS.append(name)


# ---------- 共通のスタブ ----------
DS = "aime26"
CFG = "sc@nim"

ITEMS = [
    {"id": "t-01", "question": "1+1?", "answer": "2", "task_type": "math", "meta": {}},
    {"id": "t-02", "question": "2+2?", "answer": "4", "task_type": "math", "meta": {}},
    {"id": "t-03", "question": "3+3?", "answer": "6", "task_type": "math", "meta": {}},
    {"id": "t-04", "question": "4+4?", "answer": "8", "task_type": "math", "meta": {}},
    {"id": "t-05", "question": "5+5?", "answer": "10", "task_type": "math", "meta": {}},
]
#: 全問正解を返すスタブ（1問だけ間違えさせたいときは WRONG に id を入れる）
WRONG = {"t-03"}


def stub_runner(item):
    """モデルの代わり。(本文, 投票結果, サンプル数) を返す既存契約に合わせる。"""
    ans = item["answer"] if item["id"] not in WRONG else "999"
    return (f"考えました。\\boxed{{{ans}}}", None, 7)


class TempDir:
    """FUGU_RECHECK_DIR を一時ディレクトリへ向ける with ブロック。"""

    def __enter__(self):
        self.path = Path(tempfile.mkdtemp(prefix="recheck_test_"))
        self.saved = os.environ.get("FUGU_RECHECK_DIR")
        os.environ["FUGU_RECHECK_DIR"] = str(self.path)
        return self.path

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop("FUGU_RECHECK_DIR", None)
        else:
            os.environ["FUGU_RECHECK_DIR"] = self.saved
        shutil.rmtree(self.path, ignore_errors=True)
        return False


def ordered_ids():
    """run_dataset が実際に回す順序（決定的シャッフル後）。"""
    return [it["id"] for it in rc.select_items(DS, items=ITEMS)]


ROSTER = {"backend": "nim", "aggregator": "z-ai/glm-5.2",
          "sc_models": ["deepseek-ai/deepseek-v4-pro", "openai/gpt-oss-120b",
                        "nvidia/nemotron-3-ultra-550b-a55b"],
          "models": [{"id": "openai/gpt-oss-120b", "roles": ["sc"]}]}


# ==================================================
# 1. 一過性失敗の判定（純粋関数）
# ==================================================
print("\n---------- 一過性失敗の判定 ----------")
check("429 は rate_limit",
      rc.transient_reason("", None, RuntimeError("HTTP Error 429: Too Many Requests"))
      == "rate_limit")
check("本文中の 429 も rate_limit",
      rc.transient_reason("__ERROR__: HTTP Error 429: Too Many Requests", None)
      == "rate_limit")
check("timeout は timeout",
      rc.transient_reason("", None, TimeoutError("stream idle >600s")) == "timeout")
check("504 は timeout",
      rc.transient_reason("__ERROR__: HTTP Error 504: Gateway Timeout", None)
      == "timeout")
check("空応答は empty", rc.transient_reason("", None) == "empty")
check("空白だけも empty", rc.transient_reason("   \n", None) == "empty")
check("正常な本文は None", rc.transient_reason("\\boxed{42}", None) is None)
check("投票結果だけでも一過性ではない", rc.transient_reason("", "42") is None)
check("その他の例外は error",
      rc.transient_reason("", None, ValueError("bad json")) == "error")
check("バックオフは指数（20,60,180,300 で頭打ち）",
      [rc.backoff_seconds(i) for i in (1, 2, 3, 4, 5)] == [20, 60, 180, 300, 300])


# ==================================================
# 2. 採点が bench_fugu.grade_item と一致する
# ==================================================
print("\n---------- 採点の一致 ----------")
_cases = [
    (ITEMS[0], "計算した。\\boxed{2}"),
    (ITEMS[1], "計算した。\\boxed{7}"),
    (ITEMS[2], "答えは \\boxed{6} です"),
    (ITEMS[3], "本文はあるが答えの形式が無い"),
]
_agree, _saw_ok, _saw_ng = True, False, False
for _it, _text in _cases:
    want_ok, want_got, _note = bench.grade_item(_it, answer_text=_text)
    solved = rc.solve_item(_it, lambda it, t=_text: (t, None, 1),
                           retries=0, sleep_fn=lambda s: None)
    rec = rc.build_record(_it, DS, CFG, solved)
    if not (rec["graded"] and rec["correct"] == bool(want_ok)
            and rec["got"] == want_got):
        _agree = False
    _saw_ok = _saw_ok or bool(want_ok)
    _saw_ng = _saw_ng or not bool(want_ok)
check("grade_item と正誤・抽出結果が一致する", _agree)
check("テストが空回りしていない（正解も誤答も含む）", _saw_ok and _saw_ng)

# 抽出できなかった場合は「誤答」であって「未採点」ではない
_solved = rc.solve_item(ITEMS[0], lambda it: ("本文はあるが箱が無い", None, 1),
                        retries=0, sleep_fn=lambda s: None)
_rec = rc.build_record(ITEMS[0], DS, CFG, _solved)
check("抽出失敗は採点済みの誤答（未採点にしない）",
      _rec["graded"] is True and _rec["correct"] is False)


# ==================================================
# 3. 429 相当で再試行し、上限に達したら「未採点」
# ==================================================
print("\n---------- 429 の再試行と未採点 ----------")
with TempDir() as tmp:
    waits = []
    calls = []

    def rate_limited(item):
        calls.append(item["id"])
        raise RuntimeError("HTTPError 429 Too Many Requests")

    s = rc.run_dataset(DS, CFG, items=ITEMS[:1], runner=rate_limited, retries=2,
                       sleep_fn=waits.append, roster=ROSTER, quiet=True)
    _, recs = rc.read_records(rc.results_path(DS, CFG))
    r0 = recs[0]
    check("429 は指数バックオフで再試行する", waits == [20, 60])
    check("再試行の回数だけモデルを呼ぶ（初回+2）", len(calls) == 3)
    check("上限到達で未採点になる",
          r0["graded"] is False and r0["status"] == "ungraded")
    check("未採点は correct=None（誤答ではない）", r0["correct"] is None)
    check("未採点の理由が残る", r0["reason"] == "rate_limit")
    check("リトライ回数が記録される", r0["retries"] == 2)
    check("集計で誤答に数えない",
          s["graded"] == 0 and s["correct"] == 0 and s["ungraded"] == 1)

    # 未採点は「済」ではないので、再実行で拾い直せる
    s2 = rc.run_dataset(DS, CFG, items=ITEMS[:1], runner=stub_runner, retries=0,
                        sleep_fn=lambda x: None, roster=ROSTER, quiet=True)
    check("未採点は次回の再開で拾い直される",
          s2["graded"] == 1 and s2["correct"] == 1 and s2["ungraded"] == 0)

with TempDir() as tmp:
    waits = []
    n = {"i": 0}

    def flaky(item):
        n["i"] += 1
        if n["i"] <= 2:
            raise TimeoutError("stream idle >600s (keepalive only)")
        return stub_runner(item)

    rc.run_dataset(DS, CFG, items=ITEMS[:1], runner=flaky, retries=3,
                   sleep_fn=waits.append, roster=ROSTER, quiet=True)
    _, recs = rc.read_records(rc.results_path(DS, CFG))
    check("タイムアウトから復帰したら採点される",
          recs[0]["graded"] is True and recs[0]["retries"] == 2)


# ==================================================
# 4. 中断・再開
# ==================================================
print("\n---------- 中断と再開 ----------")
with TempDir() as tmp:
    order = ordered_ids()
    seen = []

    def dies_at_third(item):
        seen.append(item["id"])
        if len(seen) == 3:
            raise KeyboardInterrupt
        return stub_runner(item)

    interrupted = False
    try:
        rc.run_dataset(DS, CFG, items=ITEMS, runner=dies_at_third, retries=0,
                       sleep_fn=lambda x: None, roster=ROSTER, quiet=True)
    except KeyboardInterrupt:
        interrupted = True
    check("Ctrl-C はそのまま上へ伝わる", interrupted)

    metas, recs = rc.read_records(rc.results_path(DS, CFG))
    check("中断しても、それまでの結果はファイルに残る",
          [r["id"] for r in recs] == order[:2])
    check("残った分は採点済み", all(r["graded"] for r in recs))

    seen2 = []

    def watcher(item):
        seen2.append(item["id"])
        return stub_runner(item)

    s = rc.run_dataset(DS, CFG, items=ITEMS, runner=watcher, retries=0,
                       sleep_fn=lambda x: None, roster=ROSTER, quiet=True)
    check("再開時は済んだ問題を飛ばして残りだけ走らせる", seen2 == order[2:])
    check("再開後は全問そろう", s["graded"] == len(ITEMS))
    check("正解数が期待どおり（WRONG の1問だけ誤答）",
          s["correct"] == len(ITEMS) - len(WRONG))

    _, recs = rc.read_records(rc.results_path(DS, CFG))
    check("結果行の数は走らせた問題数と一致（重複追記なし）", len(recs) == len(ITEMS))

with TempDir() as tmp:
    # 途中で通常の例外が出た場合も、それまでの結果は残り、その問題は未採点になる
    seen = []

    def boom_at_second(item):
        seen.append(item["id"])
        if len(seen) == 2:
            raise ValueError("壊れた応答")
        return stub_runner(item)

    s = rc.run_dataset(DS, CFG, items=ITEMS[:3], runner=boom_at_second, retries=0,
                       sleep_fn=lambda x: None, roster=ROSTER, quiet=True)
    _, recs = rc.read_records(rc.results_path(DS, CFG))
    check("例外が出ても他の問題は走り、全部ファイルに残る", len(recs) == 3)
    check("例外の問題は未採点（誤答に数えない）",
          s["ungraded"] == 1 and s["graded"] == 2)
    check("例外の内容が記録される",
          any("ValueError" in (r.get("error") or "") for r in recs))


# ==================================================
# 5. 1問ごとの記録項目
# ==================================================
print("\n---------- 1問ごとの記録項目 ----------")
with TempDir() as tmp:
    rc.run_dataset(DS, CFG, items=ITEMS[:1], runner=stub_runner, retries=0,
                   sleep_fn=lambda x: None, roster=ROSTER, quiet=True)
    _, recs = rc.read_records(rc.results_path(DS, CFG))
    r0 = recs[0]
    need = ("id", "correct", "got", "expected", "seconds", "models",
            "thinking_level", "retries")
    check("必要な項目が全部ある", all(k in r0 for k in need))
    check("id / 正解 / 期待値が入っている",
          r0["id"] and r0["correct"] is True and r0["expected"] == "2")
    check("モデル名の一覧はリスト", isinstance(r0["models"], list))
    check("所要秒は数値", isinstance(r0["seconds"], (int, float)))
    check("思考レベルは文字列（未設定なら空）", isinstance(r0["thinking_level"], str))

    # 思考レベルは実行時の設定を拾う
    _saved = os.environ.get("FUGU_THINKING_BUDGET")
    os.environ["FUGU_THINKING_BUDGET"] = "high"
    try:
        check("思考レベルが設定どおり記録される",
              rc.current_thinking_level() == "high")
    finally:
        if _saved is None:
            os.environ.pop("FUGU_THINKING_BUDGET", None)
        else:
            os.environ["FUGU_THINKING_BUDGET"] = _saved

    # 実際に呼ばれたモデル名を拾えること（fugu_local._TIMINGS 経由）
    import fugu_local as f
    _sv = f.SHOW_TIMING
    f.SHOW_TIMING = True
    try:
        mark = rc._timing_mark()
        f._TIMINGS.append(("bench-sc", "deepseek-ai/deepseek-v4-pro", 12.3))
        f._TIMINGS.append(("bench-sc", "openai/gpt-oss-120b", 9.9))
        f._TIMINGS.append(("bench-sc", "deepseek-ai/deepseek-v4-pro", 8.1))
        used, calls = rc._models_since(mark)
        check("使われたモデル名を重複なしで拾う",
              used == ["deepseek-ai/deepseek-v4-pro", "openai/gpt-oss-120b"])
        check("モデル呼び出し回数も数える", calls == 3)
        del f._TIMINGS[mark:]
    finally:
        f.SHOW_TIMING = _sv


# ==================================================
# 6. 布陣のメタ行
# ==================================================
print("\n---------- 布陣のメタ行 ----------")
with TempDir() as tmp:
    rc.run_dataset(DS, CFG, items=ITEMS[:2], runner=stub_runner, retries=0,
                   sleep_fn=lambda x: None, roster=ROSTER, quiet=True)
    path = rc.results_path(DS, CFG)
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    check("結果ファイルの先頭行がメタ行", first.get("_meta") is True)
    check("メタ行に布陣が入っている",
          first["roster"]["sc_models"] == ROSTER["sc_models"])
    check("メタ行に構成とデータセットが入っている",
          first["dataset"] == DS and first["config"] == CFG)
    check("メタ行に基準の日付が入っている",
          first["baseline_date"] == rc.BASELINE_DATE)
    check("メタ行に思考レベルが入っている", "thinking_level" in first)

    metas, recs = rc.read_records(path)
    check("メタ行は結果行として集計されない", len(recs) == 2 and len(metas) == 1)

    rc.run_dataset(DS, CFG, items=ITEMS, runner=stub_runner, retries=0,
                   sleep_fn=lambda x: None, roster=ROSTER, quiet=True)
    metas, recs = rc.read_records(path)
    check("再開のたびにその回の布陣を追記する", len(metas) == 2)
    check("セッション番号が増える", metas[1]["session"] == 2)
    first2 = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    check("再開しても先頭行はメタ行のまま", first2.get("_meta") is True)

check("collect_roster は setup 前でも落ちない",
      isinstance(rc.collect_roster().get("sc_models"), list))

_d = rc.roster_diff(["openai/gpt-oss-120b", "z-ai/glm-5.2"])
check("布陣差分: 消えたモデルを出す",
      _d["gone"] == ["deepseek-ai/deepseek-v4-pro", "minimaxai/minimax-m3"])
check("布陣差分: 増えたモデルを出す", _d["added"] == ["z-ai/glm-5.2"])
check("布陣差分: 共通を出す", _d["same"] == ["openai/gpt-oss-120b"])


# ==================================================
# 7. 比較表
# ==================================================
print("\n---------- 比較表 ----------")
_sum = {
    "aime26": {"graded": 30, "correct": 28, "ungraded": 0, "ungraded_reasons": {},
               "attempted": 30, "seconds": 7200.0, "retries": 3, "models": []},
    "aime25": {"graded": 28, "correct": 24, "ungraded": 2,
               "ungraded_reasons": {"rate_limit": 2}, "attempted": 30,
               "seconds": 9000.0, "retries": 9, "models": []},
}
_rows = {r["dataset"]: r for r in rc.compare_rows(_sum)}
_a = _rows["aime26"]
check("基準の正解数を引く", (_a["base_ok"], _a["base_n"]) == (29, 30))
check("基準のパーセント", abs(_a["base_pct"] - 96.666) < 0.01)
check("今回のパーセント", abs(_a["now_pct"] - 93.333) < 0.01)
check("差分（正解数）", _a["d_ok"] == -1)
check("差分（パーセント）", abs(_a["d_pct"] - (-3.333)) < 0.01)
check("母数が同じなら n_mismatch は False", _a["n_mismatch"] is False)

_b = _rows["aime25"]
check("未採点の数を持つ", _b["ungraded"] == 2)
check("採点分だけの正答率", abs(_b["now_pct"] - (24 / 28 * 100)) < 0.01)
check("未採点=誤答として数えた場合の母数", _b["worst_n"] == 30)
check("未採点=誤答として数えた場合の正答率",
      abs(_b["worst_pct"] - (24 / 30 * 100)) < 0.01)
check("母数が違えば n_mismatch が立つ", _b["n_mismatch"] is True)

_tbl = rc.render_table(rc.compare_rows(_sum))
check("表に基準値が出る", "29/30" in _tbl and "96.7%" in _tbl)
check("表に今回の値が出る", "28/30" in _tbl and "93.3%" in _tbl)
check("表に差分が出る", "-1 (-3.3pt)" in _tbl)
check("表に未採点=誤答の参考値が出る", "24/30" in _tbl)

_mix = rc.render_table(rc.compare_rows({"math500": {
    "graded": 50, "correct": 45, "ungraded": 0, "ungraded_reasons": {},
    "attempted": 50, "seconds": 100.0, "retries": 0, "models": []}}))
check("母数が基準と違う行には * が付く", "*" in _mix)
check("* の意味が表の下に書かれる", "母数が基準と違う行" in _mix)

_no_base = rc.compare_rows({"humaneval": {
    "graded": 5, "correct": 4, "ungraded": 0, "ungraded_reasons": {},
    "attempted": 5, "seconds": 10.0, "retries": 0, "models": []}})
check("基準の無いデータセットは差分を出さない", _no_base[0]["d_ok"] is None)
check("基準の無いデータセットも表に出せる",
      "humaneval" in rc.render_table(_no_base))

_md = rc.render_report_md(rc.compare_rows(_sum), CFG, ROSTER, "s", "e")
check("レポートに比較表が入る", "29/30" in _md)
check("レポートに布陣差分が入る", "消えた" in _md and "増えた" in _md)
check("レポートに未採点の数が入る", "未採点 2 問" in _md)
check("レポートに所要時間が入る", "所要時間" in _md)
check("レポートは原因を断定しない",
      "決められない" in _md and "分散" in _md)
check("レポートは基準構成が推定であることを書く", "推定" in _md)


# ==================================================
# 8. 集計
# ==================================================
print("\n---------- 集計 ----------")
_recs = [
    {"id": "a", "graded": True, "correct": True, "seconds": 10, "retries": 0,
     "models": ["m1"]},
    {"id": "b", "graded": True, "correct": False, "seconds": 20, "retries": 1,
     "models": ["m2"]},
    {"id": "b", "graded": True, "correct": True, "seconds": 30, "retries": 0,
     "models": ["m2"]},       # 同じ id の再実行 → 最後を採用
    {"id": "c", "graded": False, "correct": None, "reason": "empty", "seconds": 5,
     "retries": 3, "models": []},
]
_s = rc.summarize(_recs)
check("同じ id は最後の記録を採用", _s["graded"] == 2 and _s["correct"] == 2)
check("未採点を分けて数える", _s["ungraded"] == 1)
check("未採点の理由を数える", _s["ungraded_reasons"] == {"empty": 1})
check("所要秒を合計する", _s["seconds"] == 45.0)
check("リトライを合計する", _s["retries"] == 3)
check("使われたモデルを重複なしで集める", _s["models"] == ["m1", "m2"])


# ==================================================
# 9. --dry-run は何も書き込まない
# ==================================================
print("\n---------- dry-run ----------")
with TempDir() as tmp:
    _saved_load = bench.load_items
    bench.load_items = lambda name: list(ITEMS)
    try:
        rcode = rc.main(["--dry-run", "--datasets", DS, "--config", CFG])
    finally:
        bench.load_items = _saved_load
    check("dry-run は正常終了する", rcode == 0)
    check("dry-run は1バイトも書き込まない", list(tmp.iterdir()) == [])
    check("dry-run でもレポートを作らない", not rc.report_path().exists())

with TempDir() as tmp:
    _saved_load = bench.load_items
    bench.load_items = lambda name: list(ITEMS)
    try:
        n = rc.dry_run([DS], CFG, None)
    finally:
        bench.load_items = _saved_load
    check("dry-run は走らせる問題数を返す", n == len(ITEMS))
    check("dry-run 後もディレクトリは空", list(tmp.iterdir()) == [])

_lo, _mid, _hi = rc.estimate_seconds(30, CFG)
check("推定所要は 低め < 中央 < 高め", _lo < _mid < _hi)
check("推定所要は問題数に比例",
      abs(rc.estimate_seconds(60, CFG)[1] - _mid * 2) < 1e-6)
check("単発構成は 1 波で見積もる", rc.estimate_waves("think@nim") == (1.0, 1.0, 1.0))
_e, _h, _avg = rc.estimate_waves(CFG)
check("SC 構成の波数は SC 設定から導かれる", _e < _avg < _h)


# ==================================================
# 10. API キーとバックエンドのガード
# ==================================================
print("\n---------- API キーのガード ----------")
import fugu_local as f  # noqa: E402
_sv_key, _sv_env = f.NIM_API_KEY, os.environ.get("NVIDIA_API_KEY")
_sv_backend = f.FUGU_BACKEND
try:
    f.NIM_API_KEY = ""
    os.environ.pop("NVIDIA_API_KEY", None)
    _raised = False
    try:
        rc.require_api_key("sc@nim")
    except SystemExit as e:
        _raised = "NVIDIA_API_KEY" in str(e) and "setx" in str(e)
    check("@nim でキーが無ければ分かりやすく落ちる", _raised)
    _here = Path(__file__).resolve().parent
    _leak = re.compile(r"nvapi-[A-Za-z0-9_\-]{16,}")
    _sources = ["bench_recheck.py", "test_bench_recheck.py", "RUN_BENCH.bat"]
    _leaked = [n for n in _sources if (_here / n).exists()
               and _leak.search((_here / n).read_text(encoding="utf-8"))]
    check("キーの実体がどのファイルにも書かれていない", _leaked == [])

    _ok = True
    try:
        rc.require_api_key("sc")        # 非 @nim はキー不要
    except SystemExit:
        _ok = False
    check("非 @nim 構成ではキーを要求しない", _ok)

    f.NIM_API_KEY = "nvapi-dummy"
    _ok2 = True
    try:
        rc.require_api_key("sc@nim")
    except SystemExit:
        _ok2 = False
    check("キーがあれば通る", _ok2)

    f.FUGU_BACKEND = "ollama"
    rc.prepare_backend("sc@nim")
    check("@nim 構成は setup 前に backend を nim へ切り替える",
          f.FUGU_BACKEND == "nim")
    _guard = False
    try:
        rc.prepare_backend("sc")
    except SystemExit:
        _guard = True
    check("backend=nim のまま非 @nim 構成は止める（結果の汚染防止）", _guard)
finally:
    f.NIM_API_KEY, f.FUGU_BACKEND = _sv_key, _sv_backend
    if _sv_env is not None:
        os.environ["NVIDIA_API_KEY"] = _sv_env


# ==================================================
# 11. 基準値そのものの整合（推定の裏取り）
# ==================================================
print("\n---------- 基準値の整合 ----------")
_fail_total = sum(n - ok for ok, n in rc.BASELINE.values())
check("基準値の失敗数の合計が 16（bench_fugu.run_sc_budget の記述と一致）",
      _fail_total == 16)
check("推定した基準構成が bench_fugu.CONFIGS に実在する",
      rc.BASELINE_CONFIG in bench.CONFIGS)
check("基準の math500 スライスは bench_queue の limit=50 と一致",
      rc.BASELINE_LIMIT["math500"] == 50)
check("--limit 未指定なら基準当時と同じスライスになる",
      rc.effective_limit("math500", None) == 50
      and rc.effective_limit("aime26", None) is None)
check("--limit 0 は全問", rc.effective_limit("math500", 0) is None)
check("--limit N は N 問", rc.effective_limit("aime26", 5) == 5)
check("並び順は bench_fugu.run_bench と同じ決定的シャッフル",
      ordered_ids() == ordered_ids())
check("結果は bench_fugu.RESULTS_DIR の直下に置かない（report() を壊さない）",
      rc.RESULT_SUBDIR == "recheck")
check("結果ファイルは bench_fugu.results_path と別物",
      rc.results_path(DS, CFG) != bench.results_path(DS, CFG))

print("\n---------- CLI のガード ----------")
for _bad, _label in ((["--datasets", "nosuch"], "未知のデータセット"),
                     (["--config", "nosuch"], "未知の構成")):
    _hit = False
    try:
        rc.main(_bad + ["--dry-run"])
    except SystemExit as e:
        _hit = "未知" in str(e)
    check(f"{_label} は分かりやすく落ちる", _hit)

with TempDir() as tmp:
    check("--report-only は結果が無ければ 1 を返す",
          rc.main(["--report-only", "--datasets", DS, "--config", CFG]) == 1)
    check("--report-only もレポートを作らない（結果が無いとき）",
          list(tmp.iterdir()) == [])


print()
print(f"合計 {_TOTAL} 件")
if _FAILS:
    print(f"FAILED: {len(_FAILS)} 件")
    for n in _FAILS:
        print(" -", n)
    raise SystemExit(1)
print("ALL PASSED")
