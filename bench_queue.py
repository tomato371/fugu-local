"""fugu ベンチ直列キューランナー（2026-07-12）。

nk108 方針（時間無制限・精度最優先）のフルキューを無人で回す。
前回はセッション連動のバックグラウンド起動でセッション終了と共に死に、ログも残らなかった。
本スクリプトは detached 起動（PowerShell Start-Process）前提で、

  - 各ジョブを **サブプロセス** で実行（1ジョブのクラッシュ・メモリリークがキューを殺さない）
  - stdout/stderr を ~/fugu_bench/logs/<dataset>__<config>.log へ追記（死因が必ず残る）
  - 進捗を ~/fugu_bench/queue_status.json に書く（外から進捗確認できる）
  - bench_fugu.py の JSONL resume に全面依存 → どこで落ちても再実行で続きから

使い方:
  python bench_queue.py            # キューを先頭から（完了済み問題は自動スキップ）
  python bench_queue.py --dry-run  # ジョブ一覧の確認のみ
  進捗確認: python bench_fugu.py report / type %USERPROFILE%\\fugu_bench\\queue_status.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path.home() / "fugu_bench"
LOG_DIR = BENCH_DIR / "logs"
STATUS_FILE = BENCH_DIR / "queue_status.json"

# (dataset, config, limit) — 安い検証 → 本命 → 補完 の順。
# limit=None は全問。--limit は決定的シャッフル後の先頭 N 問（Fable 採点済みスライスと同一順序）。
QUEUE = [
    # 1. VibeThinker-3B 汚染検証（3B・高速）: 26(カットオフ後) vs 24/25(前) の落差で判定
    ("aime26", "vibe", None),
    ("aime25", "vibe", None),
    ("aime24", "vibe", None),
    # 2. SC エンジン広域検証 + 改修前ベースライン比較（Fable は同順序の先頭100問を採点済み）
    ("math500", "sc+pot", 50),
    ("math500", "moa-old", 50),
    # 3. 本命: AIME を sc+pot で。26 は汚染ゼロの Fable 比較（29/30）
    ("aime26", "sc+pot", None),
    ("aime25", "sc+pot", None),
    ("aime24", "sc+pot", None),
    # 4. 補完: コードと日本語知識（Fable スライスに合わせる）
    ("humaneval", "coder", 30),
    ("jmmlu", "fugu", 40),
]


def write_status(status):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def classify_exit_code(rc):
    """ジョブの returncode を 'ok' / 'error' / 'crash' に分類する（純粋関数・I/O無し）。

    2026-07-21 追記（gotcha 8 対応）: 2026-07-14 に job 4 (math500/sc+pot) が
    Windows の NTSTATUS 系異常終了コード rc=1073807364 (0x40010004) で落ちた際、
    旧実装は rc をそのまま job['rc'] に記録するだけで成功/通常失敗/GPU・ドライバ
    クラッシュを一切区別しておらず、main() のループは無条件に次のジョブへ進み、
    以降の全ジョブが rc=3221226091 (0xC0000373) で連鎖的に即死してもキューは
    気づかず、最後には top-level status['finished'] タイムスタンプを書いて
    「正常終了」したかのように見えていた（誰も気づけないまま止まっていた）。
    ここで rc を分類し、main() 側でクラッシュを大きく警告・記録できるようにする。
    自動リトライは行わない（人間が確認するまでキューを止めるのが安全）。
    """
    if rc == 0:
        return "ok"
    if rc < 0:
        # POSIX: 負値はシグナル番号によるプロセス強制終了 (-9 = SIGKILL 等)。
        return "crash"
    if rc >= 0x40000000:
        # Windows の NTSTATUS 系致命的終了コード（例: 1073807364, 3221226091 は
        # いずれもこの閾値を超える）。上位ビットが立つ巨大な値は GPU/ドライバ
        # クラッシュ等の異常終了であり、通常の Python 例外(1, 2 等)とは区別する。
        return "crash"
    return "error"


def run_job(dataset, config, limit):
    """1ジョブをサブプロセスで実行し、ログへ追記。returncode を返す。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{dataset}__{config}.log"
    cmd = [sys.executable, str(Path(__file__).parent / "bench_fugu.py"),
           "run", "--dataset", dataset, "--config", config]
    if limit:
        cmd += ["--limit", str(limit)]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"   # cp932 リダイレクト先でも UTF-8 でログを残す
    env["PYTHONUNBUFFERED"] = "1"       # 途中クラッシュ時もログが最後まで書かれるように
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== queue start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"cmd={' '.join(cmd)} =====\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"===== queue end {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"rc={proc.returncode} =====\n")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="fugu ベンチ直列キュー")
    ap.add_argument("--dry-run", action="store_true", help="ジョブ一覧の表示のみ")
    args = ap.parse_args()

    if args.dry_run:
        for i, (ds, cfg, lim) in enumerate(QUEUE, 1):
            print(f"{i:2}. {ds:10} {cfg:8} limit={lim or 'all'}")
        return

    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(),
              "jobs": [], "current": None}
    t_all = time.time()
    halted_on_crash = False
    for i, (ds, cfg, lim) in enumerate(QUEUE, 1):
        job = {"n": i, "dataset": ds, "config": cfg, "limit": lim,
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        status["current"] = job
        write_status(status)
        t0 = time.time()
        try:
            rc = run_job(ds, cfg, lim)
        except Exception as e:            # サブプロセス起動自体の失敗でも記録して次へ
            rc = -1
            job["error"] = f"{type(e).__name__}: {e}"
        # gotcha 8 (2026-07-21): rc を分類して job['status'] に記録する。
        # rc をそのまま握りつぶさず、'error'/'crash' は次で明示的に警告する。
        category = classify_exit_code(rc)
        job.update({"rc": rc, "status": category, "seconds": round(time.time() - t0, 1),
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        status["jobs"].append(job)
        status["current"] = None
        if category != "ok":
            # ASCII のみ（bench_queue には cp932 reconfigure guard が無いため
            # 絵文字/記号は使わない）。目立つように大文字・感嘆符で警告する。
            print(f"!!! ALERT: job {i} ({ds}/{cfg}) exited with status={category.upper()} "
                  f"rc={rc} - see log at {LOG_DIR / f'{ds}__{cfg}.log'} !!!")
        write_status(status)
        if category == "crash":
            # 自動リトライはしない・そのまま次のジョブには進まない。GPU/ドライバ
            # クラッシュ後に残りジョブを流すと gotcha 8 の連鎖即死を再現するだけなので、
            # 人間が確認するまでキューを止める（黙って完走したように見せない）。
            halted_on_crash = True
            print(f"!!! HALTING QUEUE: job {i} ({ds}/{cfg}) crashed (rc={rc}). "
                  f"Remaining jobs were NOT run. Investigate before restarting. !!!")
            break
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["total_seconds"] = round(time.time() - t_all, 1)
    crashed_jobs = [j for j in status["jobs"] if j.get("status") == "crash"]
    failed_jobs = [j for j in status["jobs"] if j.get("status") == "error"]
    # 機械可読な総合結果。'finished' タイムスタンプだけでは正常終了と区別できない
    # （gotcha 8 の再発防止: 外部から「本当に綺麗に終わったか」を判定できるようにする）。
    status["ok"] = not crashed_jobs and not failed_jobs
    status["crashed_jobs"] = len(crashed_jobs)
    status["failed_jobs"] = len(failed_jobs)
    status["halted_on_crash"] = halted_on_crash
    write_status(status)
    if not status["ok"]:
        print(f"!!! QUEUE FINISHED WITH PROBLEMS: {len(crashed_jobs)} crashed, "
              f"{len(failed_jobs)} failed. See {STATUS_FILE} !!!")


if __name__ == "__main__":
    main()
