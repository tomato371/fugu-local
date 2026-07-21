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
    for i, (ds, cfg, lim) in enumerate(QUEUE, 1):
        job = {"n": i, "dataset": ds, "config": cfg, "limit": lim,
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        status["current"] = job
        write_status(status)
        t0 = time.time()
        try:
            rc = run_job(ds, cfg, lim)
        except Exception as e:            # サブプロセス起動自体の失敗でも次のジョブへ進む
            rc = -1
            job["error"] = f"{type(e).__name__}: {e}"
        job.update({"rc": rc, "seconds": round(time.time() - t0, 1),
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        status["jobs"].append(job)
        status["current"] = None
        write_status(status)
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["total_seconds"] = round(time.time() - t_all, 1)
    write_status(status)


if __name__ == "__main__":
    main()
