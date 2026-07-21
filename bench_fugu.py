"""fugu_local ベンチマークハーネス（Phase 2, 2026-07-11）。
旧版（代表16問のミニベンチ）は bench_quick.py に温存してある。

AIME / MATH-500 / HumanEval / JMMLU / GPQA-Diamond を統一スキーマで取得し、
複数の実行構成（旧MoA / 単発think / 自己一貫性投票 / PoT / コード修正ループ）で解かせ、
決定的に採点して JSONL に逐次追記する。1 問ごとに追記するため中断・再開が安全
（同じ dataset×config の既存 id はスキップ）。Fable の回答シートも同じ採点系で採点できる。

使い方:
  python bench_fugu.py download [aime24 aime25 ...]     # データ取得（省略時: 主要セット）
  python bench_fugu.py list                             # データセット件数一覧
  python bench_fugu.py run --dataset aime25 --config sc [--limit N] [--ids id1,id2] [--notify]
  python bench_fugu.py export-questions --dataset aime25 --out fable_questions/aime25.jsonl
  python bench_fugu.py grade-answers --dataset aime25 --answers fable_answers/aime25.jsonl
  python bench_fugu.py report

構成(--config):
  moa-old : 改善前相当の静的MoA（MODEL_CONFIG無効化＝ctx8192・think既定、統合1回）
  fugu    : 現行の動的Fugu（Conductor→task_type→SC/MoA 自動）
  think   : 最強1体の単発（think高・低temp）
  sc      : 自己一貫性投票（PoTなし）
  sc+pot  : 自己一貫性投票 + Python実行票
  sc+cheap: sc+pot + VibeThinker 量産票8
  coder   : コード: MoA + 実行検証の自律修正ループ
  coder1  : コード: qwen3-coder 単発（修正ループなし）
  vibe    : VibeThinker-3B 単発（汚染検証用。AIME24/25=カットオフ前 vs 26=後で実力を判定）
"""
import argparse
import csv
import io
import json
import random
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import fugu_local as f

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

BENCH_DIR = Path.home() / "fugu_bench"
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
NOTIFY_EVERY = 10          # Slack 進捗通知の間隔（--notify 時）
CODE_TEST_TIMEOUT = 30     # HumanEval テスト実行のタイムアウト秒

# ==================================================
# データセット取得（統一スキーマ: id / question / answer / task_type / meta）
# ==================================================

# HF リポジトリ候補（先頭から順に試す）。(repo, config, split)
_AIME_CANDIDATES = {
    "aime24": [("Maxwell-Jia/AIME_2024", None, "train"),
               ("HuggingFaceH4/aime_2024", None, "train"),
               ("math-ai/aime24", None, "test")],
    "aime25": [("yentinglin/aime_2025", None, "train"),
               ("math-ai/aime25", None, "test"),
               ("opencompass/AIME2025", "AIME2025-I", "test"),
               ("MathArena/aime_2025", None, "train")],
    "aime26": [("MathArena/aime_2026", None, "train"),
               ("math-ai/aime26", None, "test"),
               ("yentinglin/aime_2026", None, "train"),
               ("opencompass/AIME2026", "AIME2026-I", "test")],
}

_Q_KEYS = ["problem", "question", "Problem", "Question", "prompt"]
_A_KEYS = ["answer", "Answer", "expected_answer", "final_answer", "solution_answer"]


def _pick(row, keys):
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip():
            return str(row[k]).strip()
    return None


def _load_hf_any(candidates):
    """候補 (repo, config, split) を順に試して (rows, used_repo) を返す。"""
    from datasets import load_dataset
    last_err = None
    for repo, config, split in candidates:
        for sp in ([split] if split else []) + ["train", "test"]:
            try:
                ds = load_dataset(repo, config, split=sp) if config else \
                     load_dataset(repo, split=sp)
                rows = [dict(r) for r in ds]
                if rows:
                    return rows, f"{repo}" + (f"/{config}" if config else "") + f":{sp}"
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"全候補の取得に失敗: {candidates} (最終エラー: {last_err})")


def _dl_aime(name):
    rows, src = _load_hf_any(_AIME_CANDIDATES[name])
    items = []
    for i, r in enumerate(rows, 1):
        q, a = _pick(r, _Q_KEYS), _pick(r, _A_KEYS)
        if not q or a is None:
            continue
        # AIME の答えは 0-999 の整数。"012" 表記は "12" に正規化
        a = str(int(a)) if re.fullmatch(r"\d+", str(a).strip()) else str(a).strip()
        rid = str(r.get("id") or r.get("ID") or f"{name}-{i:02d}")
        items.append({"id": rid if rid.startswith(name) else f"{name}-{rid}",
                      "question": q, "answer": a, "task_type": "math",
                      "meta": {"source": src}})
    return items


def _dl_math500(name="math500"):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for i, r in enumerate(ds, 1):
        items.append({"id": str(r.get("unique_id") or f"math500-{i:03d}").replace("/", "_"),
                      "question": str(r["problem"]).strip(),
                      "answer": str(r["answer"]).strip(),
                      "task_type": "math",
                      "meta": {"subject": r.get("subject"), "level": r.get("level")}})
    return items


def _dl_humaneval(name="humaneval"):
    from datasets import load_dataset
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        ds = load_dataset("openai_humaneval", split="test")
    items = []
    for r in ds:
        prompt = r["prompt"]
        q = ("Complete the following Python function.\n"
             "Output the COMPLETE function implementation (keep the signature exactly as "
             "given, include any needed imports) in ONE ```python block. "
             "Do not include example usage, prints, or tests outside the function.\n\n"
             f"```python\n{prompt}\n```")
        items.append({"id": r["task_id"].replace("/", "_"),
                      "question": q, "answer": "", "task_type": "code",
                      "meta": {"prompt": prompt, "test": r["test"],
                               "entry_point": r["entry_point"]}})
    return items


def _dl_jmmlu(name="jmmlu", per_subject=4):
    """nlp-waseda/JMMLU の CSV 群を GitHub zip から取得し、科目ごとに per_subject 問を層化抽出。"""
    url = "https://codeload.github.com/nlp-waseda/JMMLU/zip/refs/heads/main"
    req = urllib.request.Request(url, headers={"User-Agent": "fugu-bench/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    items = []
    rng = random.Random(42)
    csv_names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
    for csv_name in csv_names:
        subject = Path(csv_name).stem
        try:
            text = zf.read(csv_name).decode("utf-8-sig", errors="replace")
        except Exception:
            continue
        rows = [row for row in csv.reader(io.StringIO(text))
                if len(row) >= 6 and str(row[5]).strip().upper() in "ABCD"]
        if not rows:
            continue
        rng.shuffle(rows)
        for j, row in enumerate(rows[:per_subject]):
            q, ca, cb, cc, cd, ans = (row[0], row[1], row[2], row[3], row[4],
                                      str(row[5]).strip().upper())
            question = (f"{q.strip()}\n\n選択肢:\nA) {ca.strip()}\nB) {cb.strip()}\n"
                        f"C) {cc.strip()}\nD) {cd.strip()}")
            items.append({"id": f"jmmlu-{subject}-{j:02d}",
                          "question": question, "answer": ans, "task_type": "mcq",
                          "meta": {"subject": subject}})
    if not items:
        raise RuntimeError("JMMLU CSV の解析に失敗（フォーマット変更の可能性）")
    return items


def _dl_gpqa(name="gpqa"):
    """GPQA Diamond（HF gated: 事前に huggingface-cli login と利用同意が必要）。"""
    from datasets import load_dataset
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        raise RuntimeError(
            "GPQA の取得に失敗。gated データセットのため "
            "1) hf.co/datasets/Idavidrein/gpqa で利用同意 2) huggingface-cli login "
            f"が必要です。({e})")
    items = []
    for i, r in enumerate(ds, 1):
        correct = str(r["Correct Answer"]).strip()
        wrongs = [str(r[k]).strip() for k in
                  ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")]
        choices = [correct] + wrongs
        rng = random.Random(1000 + i)          # 問題ごとに決定的なシャッフル
        rng.shuffle(choices)
        letter = "ABCD"[choices.index(correct)]
        body = "\n".join(f"{c}) {t}" for c, t in zip("ABCD", choices))
        question = f"{str(r['Question']).strip()}\n\nChoices:\n{body}"
        items.append({"id": f"gpqa-{i:03d}", "question": question,
                      "answer": letter, "task_type": "mcq",
                      "meta": {"domain": r.get("High-level domain")}})
    return items


DATASETS = {
    "aime24": lambda: _dl_aime("aime24"),
    "aime25": lambda: _dl_aime("aime25"),
    "aime26": lambda: _dl_aime("aime26"),
    "math500": _dl_math500,
    "humaneval": _dl_humaneval,
    "jmmlu": _dl_jmmlu,
    "gpqa": _dl_gpqa,
}


def data_path(name):
    return DATA_DIR / f"{name}.jsonl"


def download(names):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in DATASETS:
            print(f"[download] 未知のデータセット: {name}")
            continue
        try:
            items = DATASETS[name]()
        except Exception as e:
            print(f"[download] {name}: 失敗 -> {e}")
            continue
        with data_path(name).open("w", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[download] {name}: {len(items)} 問 -> {data_path(name)}")


def load_items(name):
    p = data_path(name)
    if not p.exists():
        raise SystemExit(f"データ未取得: {name}。先に `python bench_fugu.py download {name}`")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]

# ==================================================
# 採点
# ==================================================


def grade_code(answer_text, item, timeout=CODE_TEST_TIMEOUT):
    """HumanEval 形式: 回答のコード + 公式テストを実行して合否。(ok, note)"""
    code = f.extract_code(answer_text or "")
    if not code:
        return False, "no code block"
    prompt = item["meta"]["prompt"]
    header = "\n".join(l for l in prompt.splitlines()
                       if l.startswith(("import ", "from ")))
    prog = (header + "\n\n" + code + "\n\n" + item["meta"]["test"]
            + f"\n\ncheck({item['meta']['entry_point']})\nprint('BENCH_PASS')\n")
    ok, out = f.run_python(prog, timeout=timeout)
    return (ok and "BENCH_PASS" in out), (out or "")[-300:]


def grade_item(item, answer_text=None, answer_value=None):
    """統一採点。(correct, got, note) を返す。
    answer_value があればそれを優先（SC の投票結果など）。無ければ answer_text から抽出。"""
    t = item["task_type"]
    if t == "code":
        ok, note = grade_code(answer_text, item)
        return ok, ("PASS" if ok else "FAIL"), note
    got = answer_value if answer_value is not None else \
        f.extract_final_answer(answer_text or "", t)
    if got is None:
        return False, None, "answer extraction failed"
    return bool(f.answers_equivalent(got, item["answer"])), got, ""

# ==================================================
# 実行構成
# ==================================================


def _best_model():
    for m in f.REASONING_MODELS:
        if m in f.PROPOSERS:
            return m
    return f.PROPOSERS[0] if f.PROPOSERS else f.AGGREGATOR


def run_moa_old(item):
    """改善前相当: MODEL_CONFIG を無効化（ctx8192・think既定）した静的 MoA 1 ラウンド。"""
    saved_cfg, saved_sc = f.MODEL_CONFIG, f.SC_ENABLED
    f.MODEL_CONFIG, f.SC_ENABLED = {}, False
    try:
        plan = {"mode": "moa", "task_type": "chat",
                "selected_proposers": f.PROPOSERS[:3], "rounds": 1,
                "use_image_generation": False, "image_only": False,
                "make_pptx": False, "search_required": False,
                "reason": "bench moa-old", "_fallback": False}
        text = f.fugu_answer(item["question"], plan=plan) or ""
        return text, None, 0
    finally:
        f.MODEL_CONFIG, f.SC_ENABLED = saved_cfg, saved_sc


def run_fugu(item):
    """現行の動的 Fugu（Conductor → task_type → SC/MoA 自動ルーティング）。"""
    text = f.fugu_answer(item["question"]) or ""
    return text, None, 0


def run_think(item):
    """最強 1 体の単発（think は MODEL_CONFIG 適用、低温度）。"""
    model = _best_model()
    sysp = f.SC_PROMPT_MCQ if item["task_type"] == "mcq" else f.SC_PROMPT_MATH
    raw = f.ask(model, [{"role": "system", "content": sysp},
                        {"role": "user", "content": item["question"]}],
                0.3, num_predict=f.proposer_predict_for(model), label="bench-think")
    return f.strip_think(raw), None, 1


def run_sc(item, pot, cheap=0):
    saved_pot, saved_cheap = f.SC_POT, f.SC_CHEAP_VOTES
    f.SC_POT, f.SC_CHEAP_VOTES = pot, cheap
    try:
        res = f.solve_verifiable(item["question"], item["task_type"])
        if not res:
            return "", None, 0
        return res.get("text") or "", res.get("answer"), res.get("n_samples", 0)
    finally:
        f.SC_POT, f.SC_CHEAP_VOTES = saved_pot, saved_cheap


def run_vibe(item):
    """VibeThinker-3B 単発（汚染検証）。AIME26(カットオフ後)も高得点なら実力、
    24/25 だけ高得点なら学習データ汚染 → SC_CHEAP_VOTES は封印のまま。"""
    model = f.SC_CHEAP_MODEL
    sysp = f.SC_PROMPT_MCQ if item["task_type"] == "mcq" else f.SC_PROMPT_MATH
    raw = f.ask(model, [{"role": "system", "content": sysp},
                        {"role": "user", "content": item["question"]}],
                0.3, num_predict=f.proposer_predict_for(model), label="bench-vibe")
    return f.strip_think(raw), None, 1


def run_coder(item, single=False):
    coder = f.AGGREGATOR if f.AGGREGATOR else _best_model()
    if single:
        raw = f.ask(coder, [{"role": "system", "content": f.PROPOSER_SYS},
                            {"role": "user", "content": item["question"]}],
                    0.3, num_predict=f.proposer_predict_for(coder), label="bench-coder1")
        return f.strip_think(raw), None, 1
    models = [m for m in [coder, "gpt-oss:20b", "qwen3.6:35b"]
              if m in f.PROPOSERS or m == f.AGGREGATOR][:3]
    plan = {"mode": "moa", "task_type": "code",
            "selected_proposers": models or f.PROPOSERS[:3], "rounds": 1,
            "use_image_generation": False, "image_only": False,
            "make_pptx": False, "search_required": False,
            "reason": "bench coder", "_fallback": False}
    text = f.fugu_answer(item["question"], plan=plan) or ""
    return text, None, 0


CONFIGS = {
    "moa-old": run_moa_old,
    "fugu": run_fugu,
    "think": run_think,
    "sc": lambda it: run_sc(it, pot=False),
    "sc+pot": lambda it: run_sc(it, pot=True),
    "sc+cheap": lambda it: run_sc(it, pot=True, cheap=8),
    "coder": lambda it: run_coder(it, single=False),
    "coder1": lambda it: run_coder(it, single=True),
    "vibe": run_vibe,
}

# ==================================================
# ランナー（1問ずつ JSONL 追記・再開可能）
# ==================================================


def results_path(dataset, config):
    return RESULTS_DIR / f"{dataset}__{config}.jsonl"


def load_done_ids(dataset, config):
    p = results_path(dataset, config)
    done = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def append_result(dataset, config, rec):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with results_path(dataset, config).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_bench(dataset, config, limit=None, ids=None, offset=0, notify=False,
              shuffle_seed=42):
    if config not in CONFIGS:
        raise SystemExit(f"未知の構成: {config} (choices: {list(CONFIGS)})")
    items = load_items(dataset)
    if ids:
        want = set(ids)
        items = [it for it in items if it["id"] in want]
    else:
        rng = random.Random(shuffle_seed)
        rng.shuffle(items)          # 決定的シャッフル（--limit の偏り防止・全構成で同順）
        items = items[offset:offset + limit] if limit else items[offset:]
    done = load_done_ids(dataset, config)
    todo = [it for it in items if it["id"] not in done]
    print(f"[bench] {dataset} x {config}: {len(todo)} 問（済 {len(items) - len(todo)}）")
    if not todo:
        return
    if not f.setup():
        raise SystemExit("fugu setup 失敗（Ollama を確認）")
    f.SHOW_PLAN = False
    f.SHOW_PROPOSALS = False
    ok_count = err_count = 0
    t_batch = time.time()
    for k, it in enumerate(todo, 1):
        print(f"\n=== [{k}/{len(todo)}] {it['id']} ({dataset}/{config}) ===")
        t0 = time.time()
        rec = {"id": it["id"], "dataset": dataset, "config": config,
               "expected": it["answer"], "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            text, ans_value, n_samples = CONFIGS[config](it)
            correct, got, note = grade_item(it, answer_text=text, answer_value=ans_value)
            rec.update({"correct": bool(correct), "got": got, "note": note,
                        "n_samples": n_samples,
                        "answer_text": (text or "")[-4000:]})
        except KeyboardInterrupt:
            print("\n[bench] 中断されました（結果は保存済み。再実行で再開します）")
            raise
        except Exception as e:
            rec.update({"correct": False, "got": None,
                        "error": f"{type(e).__name__}: {e}"})
            err_count += 1
        rec["seconds"] = round(time.time() - t0, 1)
        append_result(dataset, config, rec)
        ok_count += int(rec.get("correct", False))
        print(f"    -> {'OK' if rec.get('correct') else 'NG'} "
              f"got={rec.get('got')} expected={it['answer']} ({rec['seconds']}s)")
        if notify and (k % NOTIFY_EVERY == 0 or k == len(todo)):
            f.notify_slack(f"bench {dataset}/{config}",
                           f"{k}/{len(todo)} 完了  acc={ok_count}/{k}  err={err_count}",
                           round(time.time() - t_batch, 1))
    print(f"\n[bench] 完了: {dataset}/{config}  acc={ok_count}/{len(todo)}")

# ==================================================
# Fable 回答シート（外部回答の採点）と問題エクスポート
# ==================================================


def export_questions(dataset, out, limit=None, offset=0, shuffle_seed=42):
    items = load_items(dataset)
    if limit or offset:
        # run_bench と同一のシャッフル・切り出しで、ローカル実行とFable回答のサブセットを一致させる
        rng = random.Random(shuffle_seed)
        rng.shuffle(items)
        items = items[offset:offset + limit] if limit else items[offset:]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({"id": it["id"], "task_type": it["task_type"],
                                 "question": it["question"]}, ensure_ascii=False) + "\n")
    print(f"[export] {len(items)} 問 -> {out}")


def grade_answers(dataset, answers_file, config_name="fable"):
    items = {it["id"]: it for it in load_items(dataset)}
    done = load_done_ids(dataset, config_name)
    n = ok = 0
    for line in Path(answers_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        it = items.get(rec["id"])
        if not it or rec["id"] in done:
            continue
        text = rec.get("answer_text") or rec.get("answer") or ""
        correct, got, note = grade_item(it, answer_text=text)
        append_result(dataset, config_name, {
            "id": it["id"], "dataset": dataset, "config": config_name,
            "expected": it["answer"], "correct": bool(correct), "got": got,
            "note": note, "seconds": rec.get("seconds", 0),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "answer_text": text[-4000:]})
        n += 1
        ok += int(correct)
    print(f"[grade] {dataset}/{config_name}: {ok}/{n} 正解")

# ==================================================
# 集計
# ==================================================


def report():
    rows = {}
    if not RESULTS_DIR.exists():
        print("結果がまだありません")
        return
    for p in sorted(RESULTS_DIR.glob("*.jsonl")):
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        if not recs:
            continue
        by_id = {}
        for r in recs:               # 同一 id の重複（再実行）は最後を採用
            by_id[r["id"]] = r
        recs = list(by_id.values())
        key = (recs[0]["dataset"], recs[0]["config"])
        n = len(recs)
        ok = sum(1 for r in recs if r.get("correct"))
        secs = [r.get("seconds", 0) for r in recs if r.get("seconds")]
        rows[key] = (n, ok, (sum(secs) / len(secs)) if secs else 0)
    print(f"{'dataset':12} {'config':10} {'acc':>14} {'avg_sec':>9}")
    print("-" * 50)
    for (ds, cfg), (n, ok, avg) in sorted(rows.items()):
        print(f"{ds:12} {cfg:10} {ok:>4}/{n:<4} ({ok / n * 100:5.1f}%) {avg:>8.1f}")


# ==================================================
# CLI
# ==================================================

def main():
    ap = argparse.ArgumentParser(description="fugu ベンチマークハーネス")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download")
    p_dl.add_argument("names", nargs="*",
                      default=["aime24", "aime25", "aime26", "math500",
                               "humaneval", "jmmlu"])

    sub.add_parser("list")

    p_run = sub.add_parser("run")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--limit", type=int)
    p_run.add_argument("--offset", type=int, default=0)
    p_run.add_argument("--ids", help="カンマ区切りの問題 id で絞る")
    p_run.add_argument("--notify", action="store_true", help="Slack 進捗通知")

    p_exp = sub.add_parser("export-questions")
    p_exp.add_argument("--dataset", required=True)
    p_exp.add_argument("--out", required=True)
    p_exp.add_argument("--limit", type=int)
    p_exp.add_argument("--offset", type=int, default=0)

    p_gr = sub.add_parser("grade-answers")
    p_gr.add_argument("--dataset", required=True)
    p_gr.add_argument("--answers", required=True)
    p_gr.add_argument("--name", default="fable", help="results 上の構成名")

    sub.add_parser("report")

    args = ap.parse_args()
    if args.cmd == "download":
        download(args.names or ["aime24", "aime25", "aime26", "math500",
                                "humaneval", "jmmlu"])
    elif args.cmd == "list":
        for name in DATASETS:
            p = data_path(name)
            n = sum(1 for l in p.read_text(encoding="utf-8").splitlines()
                    if l.strip()) if p.exists() else 0
            print(f"{name:10} {'取得済 ' + str(n) + ' 問' if n else '未取得'}")
    elif args.cmd == "run":
        ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
        run_bench(args.dataset, args.config, limit=args.limit, ids=ids,
                  offset=args.offset, notify=args.notify)
    elif args.cmd == "export-questions":
        export_questions(args.dataset, args.out, limit=args.limit, offset=args.offset)
    elif args.cmd == "grade-answers":
        grade_answers(args.dataset, args.answers, config_name=args.name)
    elif args.cmd == "report":
        report()


if __name__ == "__main__":
    main()
