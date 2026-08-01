"""Fugu 統一ランチャー — 「どのファイルを実行すれば何が起動するか」を 1 画面に集約する。

エクスプローラーから ``START_FUGU.bat`` をダブルクリック(あるいは
``python fugu_launcher.py``)すると番号選択メニューが出て、CLI / Web UI / TUI /
REST API / ベンチ / 自己進化 / 姉妹プロジェクト fugu-rag のすべてに到達できる。

設計方針:

* **標準ライブラリのみ**。core と同じく追加依存ゼロで起動できる(足りない任意依存は
  メニューを選んだ時点で ``pip install ...`` を表示して戻るだけ)。
* **既存の挙動は一切変えない**。各機能は今までどおりのコマンドを ``subprocess`` で
  起動するだけで、フラグは子プロセスの環境変数として渡す(親環境は汚さない)。
* **fugu_local を import しない**。4,681 行の起動コストと副作用を避けるため、
  モデル名などの定数はここに再掲し、ドリフトは ``tests/test_launcher.py`` で検出する。
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

# Windows の cp932 パイプで ✓ ⚠ ❌ などを出しても落ちないようにする(fugu_local と同じ作法)
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

REPO = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(REPO, "fugu_launcher.json")

# ---------------------------------------------------------------- 定数(再掲)
# fugu_local.py の DESIRED_PROPOSERS / DESIRED_AGGREGATOR / FALLBACK_MODEL と
# 一致していること。ずれたら tests/test_launcher.py が落ちる。
FALLBACK_MODEL = "qwen3:4b"                     # これが無いと何も動かない
COUNCIL_MODELS = [                              # 欠けても縮退するだけ
    "gpt-oss:20b", "qwen3-coder:30b", "gemma4:26b", "qwen3.6:35b",
]
EMBED_MODEL = "nomic-embed-text"                # fugu-rag 用

DEFAULT_RAG_REPO = os.path.join(os.path.dirname(REPO), "fugu-rag")

# 機能フラグ: (環境変数, 既定, 説明) — README のフラグ表と同じ一行説明
FLAGS = [
    ("FUGU_SANDBOX", False, "生成コードをサブプロセスのサンドボックスで実行する"),
    ("FUGU_TDC", False, "Critic が pytest を先に書き、緑になるまでコードを承認しない"),
    ("FUGU_BROWSER", False, "Web 検索スニペットを実ページ本文で補強する"),
    ("FUGU_MEMORY", False, "過去の実行から学んだ教訓を新しい質問に注入する"),
    ("FUGU_MEMORY_CONSOLIDATE", False, "古いエピソード記憶を要約に統合する"),
    ("FUGU_COMPRESS", False, "2 ラウンド目以降、下書きを構造化ダイジェストに圧縮する"),
    ("FUGU_SPECULATE", False, "Conductor の立案中に Web/RAG 取得を先回りする"),
    ("FUGU_DEBATE", False, "提案が割れたとき相互批評ディベートを行う"),
    ("FUGU_TOOL_CALLING", False, "ツールレジストリから実行時にツールを選ばせる"),
    ("FUGU_TASKS", False, "多段の依頼をチェックポイント付きタスクボードに分解する"),
    ("FUGU_ADVERSARIAL", False, "懐疑役パネルが論破に失敗して初めて回答を確定する"),
    ("FUGU_DYNAMIC_SUBAGENTS", False, "課題専用の専門家ペルソナを その場で作って合議に加える"),
    ("FUGU_REQUIRE_APPROVAL", False, "コード実行と自己進化 merge を人間承認までブロックする"),
    ("FUGU_HIGH_VRAM", False, "8GB 前提の context 制限を緩める(大きい GPU 向け)"),
    ("FUGU_PROFILE", False, "LLM 呼び出しの時間内訳を JSONL に記録する"),
]

THINKING_CHOICES = ["off", "minimal", "low", "medium", "high", "ultra", "max", "auto"]

DEFAULT_SETTINGS = {
    "flags": {name: default for name, default, _ in FLAGS},
    "thinking_budget": "off",
    "vision_model": "",
    "ollama_url": "http://localhost:11434",
    "rag_repo": DEFAULT_RAG_REPO,
}


# ---------------------------------------------------------------- 設定の保存

def load_settings(path=SETTINGS_PATH):
    """設定 JSON を読む。無い/壊れている場合は既定値にフォールバックする。"""
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))  # deep copy
    try:
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        return settings
    if not isinstance(saved, dict):
        return settings
    for key, value in saved.items():
        if key == "flags" and isinstance(value, dict):
            for flag, on in value.items():
                if flag in settings["flags"]:
                    settings["flags"][flag] = bool(on)
        elif key in settings and isinstance(value, type(settings[key])):
            settings[key] = value
    if settings["thinking_budget"] not in THINKING_CHOICES:
        settings["thinking_budget"] = "off"
    return settings


def save_settings(settings, path=SETTINGS_PATH):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, ensure_ascii=False, indent=2, sort_keys=True)


def build_env(settings, base=None):
    """設定 → 子プロセスに渡す環境変数。OFF のフラグは *設定しない*
    (未設定と "0" は fugu_local 側で同じ扱いだが、素の環境と等価に保つため)。"""
    env = dict(os.environ if base is None else base)
    for name, _default, _desc in FLAGS:
        if settings["flags"].get(name):
            env[name] = "1"
        else:
            env.pop(name, None)
    budget = settings.get("thinking_budget", "off")
    if budget and budget != "off":
        env["FUGU_THINKING_BUDGET"] = budget
    else:
        env.pop("FUGU_THINKING_BUDGET", None)
    if settings.get("vision_model"):
        env["FUGU_VISION_MODEL"] = settings["vision_model"]
    if settings.get("ollama_url"):
        env["OLLAMA_URL"] = settings["ollama_url"]
    return env


# ---------------------------------------------------------------- コマンド組立

def build_command(action, params=None):
    """メニュー項目 → 実行する argv。副作用なし(テストはこれだけを検証する)。"""
    p = params or {}
    py = sys.executable or "python"

    if action == "cli":
        argv = [py, "fugu_local.py"]
        if p.get("question"):
            argv.append(p["question"])
        if p.get("search"):
            argv.append("--search")
        if p.get("rag"):
            argv += ["--rag"] + list(p["rag"])
        if p.get("file"):
            argv += ["--file", p["file"]]
        if p.get("out"):
            argv += ["--out", p["out"]]
        if p.get("image"):
            argv += ["--image"] + list(p["image"])
        if p.get("resume"):
            argv += ["--resume", p["resume"]]
        return argv
    if action == "web":
        return [py, "fugu_web.py"]
    if action == "tui":
        return [py, "fugu_tui.py"]
    if action == "api":
        # 127.0.0.1 に束縛する: uvicorn がログに出す URL がそのままブラウザで
        # 開ける(0.0.0.0 だと Windows のブラウザは「到達できません」になる)。
        # LAN に公開したい場合は手動で `uvicorn fugu_api:app --host 0.0.0.0`。
        return [py, "-m", "uvicorn", "fugu_api:app",
                "--host", "127.0.0.1", "--port", str(p.get("port", 8000))]

    if action == "bench-list":
        return [py, "bench_fugu.py", "list"]
    if action == "bench-download":
        return [py, "bench_fugu.py", "download"] + list(p.get("names") or [])
    if action == "bench-run":
        argv = [py, "bench_fugu.py", "run",
                "--dataset", p["dataset"], "--config", p["config"]]
        if p.get("limit"):
            argv += ["--limit", str(p["limit"])]
        return argv
    if action == "bench-report":
        return [py, "bench_fugu.py", "report"]
    if action == "bench-queue":
        argv = [py, "bench_queue.py"]
        if p.get("dry_run"):
            argv.append("--dry-run")
        return argv
    if action == "eval":
        return [py, "eval_fugu.py"] + list(p.get("args") or [])
    if action == "profile-report":
        return [py, "bench_profile_report.py"] + list(p.get("paths") or [])

    if action == "evolve-dry":
        return [py, "-m", "fugu_evolve", "--repo", ".", "--dry-run"]
    if action == "evolve-pr":
        return [py, "-m", "fugu_evolve", "--repo", ".",
                "--pr-mode", "--max-proposals", str(p.get("max_proposals", 1))]
    if action == "evolve-auto":
        return [py, "-m", "fugu_evolve", "--repo", ".",
                "--max-proposals", str(p.get("max_proposals", 1))]
    if action == "evolve-prompts":
        return [py, "-m", "fugu_evolve", "--prompts", p["name"]]

    if action == "rag-ask":
        return [py, "-m", "fugu_rag", "ask", p["question"]]
    if action == "rag-bench":
        argv = [py, "-m", "fugu_rag", "bench",
                "--dataset", p.get("dataset", "eval/golden")]
        if p.get("rerank"):
            argv.append("--rerank")
        return argv
    if action == "rag-research":
        return [py, "-m", "fugu_rag", "research", p["question"],
                "--branches", str(p.get("branches", 2)),
                "--depth", str(p.get("depth", 1))]
    if action == "rag-eval-gen":
        return [py, "-m", "fugu_rag", "eval-gen",
                "--dataset", p.get("dataset", "eval/golden"),
                "--limit", str(p.get("limit", 5))]
    if action == "rag-eval-crag":
        return [py, "-m", "fugu_rag", "eval-crag",
                "--dataset", p.get("dataset", "eval/golden"),
                "--limit", str(p.get("limit", 6))]

    raise KeyError(f"unknown action: {action}")


#: 各アクションが必要とする任意 pip パッケージ(未導入ならメニューで案内する)
REQUIRED_PACKAGES = {
    "web": ["gradio"],
    "tui": ["rich"],
    "api": ["fastapi", "uvicorn"],
}


def missing_packages(action):
    return [pkg for pkg in REQUIRED_PACKAGES.get(action, [])
            if importlib.util.find_spec(pkg) is None]


# ---------------------------------------------------------------- 環境チェック

def _norm(model):
    """`qwen3:4b` と `qwen3:4b:latest` 的な表記ゆれを吸収する。"""
    return model[: -len(":latest")] if model.endswith(":latest") else model


def installed_models(url, timeout=2.0):
    """Ollama の /api/tags からモデル名一覧を得る。落ちていれば None。"""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as res:
            data = json.loads(res.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []
    return [_norm(m.get("name", "")) for m in models if isinstance(m, dict)]


def check_env(settings):
    """起動前チェック。例外は投げず、結果を dict で返す。"""
    url = settings.get("ollama_url") or DEFAULT_SETTINGS["ollama_url"]
    have = installed_models(url)
    report = {
        "ollama_url": url,
        "ollama_ok": have is not None,
        "installed": have or [],
        "missing_required": [],
        "missing_council": [],
        "missing_embed": [],
        "missing_packages": {},
        "rag_repo_ok": os.path.isdir(settings.get("rag_repo") or ""),
    }
    if have is not None:
        present = set(have)
        if _norm(FALLBACK_MODEL) not in present:
            report["missing_required"].append(FALLBACK_MODEL)
        report["missing_council"] = [m for m in COUNCIL_MODELS
                                     if _norm(m) not in present]
        if _norm(EMBED_MODEL) not in present:
            report["missing_embed"].append(EMBED_MODEL)
    for action in REQUIRED_PACKAGES:
        missing = missing_packages(action)
        if missing:
            report["missing_packages"][action] = missing
    report["ok"] = report["ollama_ok"] and not report["missing_required"]
    return report


def format_check(report):
    """check_env の結果を人間向けの複数行テキストにする。"""
    lines = [f"  python: {sys.executable} "
             f"({sys.version_info.major}.{sys.version_info.minor})"]
    if not report["ollama_ok"]:
        lines.append(f"❌ Ollama に接続できません ({report['ollama_url']})")
        lines.append("   → 別のターミナルで `ollama serve` を実行してください")
        lines.append("   → 別ホストの Ollama を使う場合は 8) 設定で OLLAMA_URL を変更")
    else:
        lines.append(f"✓ Ollama OK ({report['ollama_url']}) "
                     f"— モデル {len(report['installed'])} 個")
    for model in report["missing_required"]:
        lines.append(f"❌ 必須モデルがありません → ollama pull {model}")
    if report["missing_council"]:
        lines.append("⚠ 合議メンバーの一部が未導入です(その分だけ縮退して動きます):")
        for model in report["missing_council"]:
            lines.append(f"   ollama pull {model}")
    if report["missing_embed"]:
        lines.append(f"⚠ fugu-rag の埋め込みモデル未導入(BM25 に縮退) "
                     f"→ ollama pull {report['missing_embed'][0]}")
    for action, pkgs in sorted(report["missing_packages"].items()):
        lines.append(f"⚠ {action} には未導入の依存があります "
                     f"→ pip install {' '.join(pkgs)}")
    if not report["rag_repo_ok"]:
        lines.append("⚠ fugu-rag リポジトリが見つかりません(7 は使えません) "
                     "→ 8) 設定でパスを指定")
    return "\n".join(lines)


# ---------------------------------------------------------------- 実行

def run(action, settings, params=None):
    """アクションを子プロセスとして実行し、終わったら呼び出し元に戻る。"""
    missing = missing_packages(action)
    if missing:
        print(f"\n⚠ この機能には {', '.join(missing)} が必要です。")
        print(f"  pip install {' '.join(missing)}")
        return None
    cwd = REPO
    if action.startswith("rag-"):
        cwd = settings.get("rag_repo") or DEFAULT_RAG_REPO
        if not os.path.isdir(cwd):
            print(f"\n⚠ fugu-rag が {cwd} にありません。")
            print("  git clone https://github.com/tomato371/fugu-rag.git")
            print("  もしくは 8) 設定でパスを指定してください。")
            return None
    argv = build_command(action, params)
    script = argv[1] if len(argv) > 1 else ""
    if script.endswith(".py") and not os.path.exists(os.path.join(cwd, script)):
        print(f"\n⚠ {script} がこのブランチにはありません(未マージの機能かもしれません)。")
        return None
    print(f"\n$ {' '.join(argv[1:])}   (cwd={cwd})\n", flush=True)
    try:
        return subprocess.run(argv, cwd=cwd, env=build_env(settings)).returncode
    except KeyboardInterrupt:
        print("\n(中断しました)")
        return 130
    except OSError as exc:
        print(f"\n❌ 起動に失敗しました: {exc}")
        return 1


def _port_open(port, host="127.0.0.1", timeout=1.0):
    """ポートが LISTEN しているか(=サーバーが応答するか)。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def run_server(action, settings, port, url, auto_opens_browser=False,
               startup_timeout=120.0):
    """Web UI / REST API 用: ポートが開くまで待ってからブラウザを開く。

    「起動前に URL を案内する」と、初回起動の数十秒の間にユーザーがページを
    開いて『このページに到達できません』になるため、こちらで待つ。
    """
    missing = missing_packages(action)
    if missing:
        print(f"\n⚠ この機能には {', '.join(missing)} が必要です。")
        print(f"  pip install {' '.join(missing)}")
        return None
    if _port_open(port):
        print(f"\n⚠ ポート {port} は既に使われています — 前回の分が動いたままの"
              f"可能性があります。")
        if _yes(f"{url} をブラウザで開いてみる?", True):
            webbrowser.open(url)
        return None
    argv = build_command(action, {"port": port})
    print(f"\n$ {' '.join(argv[1:])}   (cwd={REPO})", flush=True)
    print("起動中です… 初回は数十秒かかることがあります。", flush=True)
    try:
        proc = subprocess.Popen(argv, cwd=REPO, env=build_env(settings))
    except OSError as exc:
        print(f"❌ 起動に失敗しました: {exc}")
        return 1
    try:
        deadline = time.monotonic() + startup_timeout
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:          # 起動前に死んだ
                print(f"\n❌ サーバーが終了しました(終了コード {proc.returncode})。"
                      "上のエラーメッセージを確認してください。")
                return proc.returncode
            if _port_open(port):
                ready = True
                break
            time.sleep(0.5)
        if ready:
            print(f"\n✓ 起動しました: {url}")
            if not auto_opens_browser:
                webbrowser.open(url)
            print("止めるにはこのウィンドウで Ctrl+C を押してください。", flush=True)
        else:
            print(f"\n⚠ {int(startup_timeout)} 秒待ちましたがまだ応答がありません。"
                  "そのまま待ちます(Ctrl+C で中止)。", flush=True)
        proc.wait()
        return proc.returncode
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\n(サーバーを停止しました)")
        return 130


# ---------------------------------------------------------------- 画面

def _ask(prompt, default=""):
    """入力を 1 行読む。標準入力が閉じている(パイプ実行など)なら即終了する
    — 空文字を返し続けるとメニューが無限ループするため。"""
    try:
        answer = input(prompt).strip().lstrip("﻿")  # BOM 付きパイプ対策
    except EOFError:
        print("\n(標準入力が終了したので終了します)")
        raise SystemExit(0)
    return answer or default


#: 全角の数字・記号・よく使う英字を半角に寄せる(メニュー入力の表記ゆれ対策)
_ZENKAKU = str.maketrans("０１２３４５６７８９ｑｔｖｕｒｙｎ）．", "0123456789qtvuryn).")


def _choice(prompt):
    """メニュー選択を 1 つ読む。「2)」「２」「 2. 」のような入力も「2」に正規化する
    — 実際に『2)』と打たれて何も起動しない事故があった。"""
    raw = _ask(prompt).translate(_ZENKAKU).lower()
    return raw.strip().rstrip(").。.").strip()


def _yes(prompt, default=False):
    suffix = "[Y/n]" if default else "[y/N]"
    answer = _ask(f"{prompt} {suffix}: ").translate(_ZENKAKU).lower()
    if not answer:
        return default
    return answer.startswith("y")


def _pause():
    _ask("\n[Enter] でメニューに戻る ")


def _flag_summary(settings):
    on = [name.replace("FUGU_", "").lower()
          for name, _d, _c in FLAGS if settings["flags"].get(name)]
    budget = settings.get("thinking_budget", "off")
    if budget != "off":
        on.append(f"thinking={budget}")
    return ", ".join(on) if on else "なし(既定の動作)"


def cli_menu(settings):
    print("\n--- 対話CLI (fugu_local.py) ---")
    question = _ask("質問(空 Enter で対話モードに入る): ")
    params = {"question": question}
    params["search"] = _yes("Web 検索を使う?")
    rag = _ask("RAG に使うフォルダ(空でなし、複数はスペース区切り): ")
    if rag:
        params["rag"] = rag.split()
    path = _ask("入力ファイル --file (空でなし): ")
    if path:
        params["file"] = path
    out = _ask("出力ファイル --out (空でなし): ")
    if out:
        params["out"] = out
    image = _ask("画像 --image (空でなし): ")
    if image:
        params["image"] = image.split()
    run("cli", settings, params)
    _pause()


def bench_menu(settings):
    while True:
        print("""
--- ベンチマーク・評価 ---
 1) データセット一覧            bench_fugu.py list
 2) データセット取得            bench_fugu.py download <name>
 3) ベンチ実行                  bench_fugu.py run --dataset .. --config ..
 4) 集計レポート                bench_fugu.py report
 5) キュー実行(まず --dry-run)  bench_queue.py
 6) 手元の評価セット            eval_fugu.py
 7) 実行時間の内訳レポート      bench_profile_report.py
 0) 戻る""")
        choice = _choice("選択> ")
        if choice in ("0", ""):
            return
        if choice == "1":
            run("bench-list", settings)
        elif choice == "2":
            names = _ask("データセット名(例 aime25 humaneval): ").split()
            run("bench-download", settings, {"names": names})
        elif choice == "3":
            dataset = _ask("--dataset (例 aime25): ")
            print("  --config: moa-old / fugu / think / sc / sc+pot / sc+cheap "
                  "/ coder / coder1 / vibe")
            config = _ask("--config (既定 fugu): ", "fugu")
            limit = _ask("--limit 問題数(空で全部): ")
            if not dataset:
                print("⚠ dataset は必須です")
                continue
            run("bench-run", settings,
                {"dataset": dataset, "config": config, "limit": limit or None})
        elif choice == "4":
            run("bench-report", settings)
        elif choice == "5":
            run("bench-queue", settings, {"dry_run": _yes("まず一覧だけ表示する?", True)})
        elif choice == "6":
            run("eval", settings, {"args": _ask("引数(例 c=b,c): ").split()})
        elif choice == "7":
            run("profile-report", settings, {"paths": _ask("JSONL パス(空で既定): ").split()})
        else:
            continue
        _pause()


def evolve_menu(settings):
    while True:
        print("""
--- 自己進化 (fugu_evolve) ---
 1) 提案だけ見る(変更なし)      --dry-run
 2) ブランチに実装して人間レビュー待ち(推奨)  --pr-mode --max-proposals 1
 3) 検証が通れば merge まで自動  --max-proposals 1
 4) プロンプト定数を進化させる   --prompts NAME
 0) 戻る
 ※ 2/3 は auto-evolve/* ブランチ上でのみ編集・commit されます。""")
        choice = _choice("選択> ")
        if choice in ("0", ""):
            return
        if choice == "1":
            run("evolve-dry", settings)
        elif choice == "2":
            run("evolve-pr", settings)
        elif choice == "3":
            if _yes("pytest 100% + bench 非退行 + Critic 承認を満たせば自動 merge します。続行?"):
                run("evolve-auto", settings)
            else:
                continue
        elif choice == "4":
            name = _ask("プロンプト定数名(例 PRESENTATION_STYLE): ")
            if not name:
                continue
            run("evolve-prompts", settings, {"name": name})
        else:
            continue
        _pause()


def rag_menu(settings):
    while True:
        print(f"""
--- fugu-rag ({settings.get('rag_repo')}) ---
 1) 質問に答える              ask
 2) 検索性能ベンチ            bench --dataset eval/golden
 3) 深掘りリサーチ(引用付き)  research
 4) 生成品質の評価            eval-gen
 5) 棄却性能の評価(CRAG)      eval-crag
 0) 戻る""")
        choice = _choice("選択> ")
        if choice in ("0", ""):
            return
        if choice == "1":
            question = _ask("質問: ")
            if not question:
                continue
            run("rag-ask", settings, {"question": question})
        elif choice == "2":
            run("rag-bench", settings, {"rerank": _yes("LLM リランクも比較する?")})
        elif choice == "3":
            question = _ask("リサーチしたいこと: ")
            if not question:
                continue
            run("rag-research", settings, {"question": question})
        elif choice == "4":
            run("rag-eval-gen", settings)
        elif choice == "5":
            run("rag-eval-crag", settings)
        else:
            continue
        _pause()


def settings_menu(settings):
    while True:
        print("\n--- 設定(次に起動するプロセスへ環境変数として渡されます) ---")
        for i, (name, _d, desc) in enumerate(FLAGS, start=1):
            mark = "ON " if settings["flags"].get(name) else "off"
            print(f" {i:2}) [{mark}] {name:<26} {desc}")
        print(f" t) 思考の深さ FUGU_THINKING_BUDGET = {settings['thinking_budget']}")
        print(f" v) 画像モデル FUGU_VISION_MODEL    = {settings['vision_model'] or '(既定)'}")
        print(f" u) Ollama     OLLAMA_URL           = {settings['ollama_url']}")
        print(f" r) fugu-rag のパス                 = {settings['rag_repo']}")
        print(" 0) 戻る(保存されます)")
        choice = _choice("選択(番号でトグル)> ")
        if choice in ("0", ""):
            save_settings(settings)
            return
        if choice == "t":
            print("  " + " / ".join(THINKING_CHOICES))
            value = _ask("値: ")
            if value in THINKING_CHOICES:
                settings["thinking_budget"] = value
        elif choice == "v":
            settings["vision_model"] = _ask("モデル名(空で既定): ")
        elif choice == "u":
            settings["ollama_url"] = _ask("URL: ", settings["ollama_url"])
        elif choice == "r":
            settings["rag_repo"] = _ask("パス: ", settings["rag_repo"])
        elif choice.isdigit() and 1 <= int(choice) <= len(FLAGS):
            name = FLAGS[int(choice) - 1][0]
            settings["flags"][name] = not settings["flags"].get(name)
        save_settings(settings)


def main_menu(settings, report):
    while True:
        status = ("Ollama OK" if report["ollama_ok"] else "Ollama 未接続")
        if report["missing_required"]:
            status += " / 必須モデル不足"
        print(f"""
============================================================
 Fugu Local — 起動メニュー   [{status}]
 有効な機能: {_flag_summary(settings)}
============================================================
 1) 対話CLI          fugu_local.py
 2) Web UI           fugu_web.py            http://localhost:7860
 3) TUI              fugu_tui.py
 4) REST API         fugu_api:app           http://localhost:8000/docs
 5) ベンチマーク・評価 >
 6) 自己進化 >        fugu_evolve
 7) fugu-rag >        検索・リサーチ
 8) 設定(機能フラグ / 思考の深さ / OLLAMA_URL)
 9) 環境チェックをやり直す
 0) 終了""")
        choice = _choice("選択> ")
        if choice in ("0", "q"):
            return 0
        if choice == "1":
            cli_menu(settings)
        elif choice == "2":
            # gradio 側(inbrowser=True)がブラウザを開くので、こちらでは開かない
            run_server("web", settings, port=7860,
                       url="http://localhost:7860", auto_opens_browser=True)
            _pause()
        elif choice == "3":
            run("tui", settings)
            _pause()
        elif choice == "4":
            run_server("api", settings, port=8000,
                       url="http://localhost:8000/docs")
            _pause()
        elif choice == "5":
            bench_menu(settings)
        elif choice == "6":
            evolve_menu(settings)
        elif choice == "7":
            rag_menu(settings)
        elif choice == "8":
            settings_menu(settings)
        elif choice == "9":
            report = check_env(settings)
            print("\n" + format_check(report))
            _pause()
        elif choice:
            print(f"⚠ 「{choice}」は選択肢にありません。0〜9 の番号だけを入力して"
                  "ください(例: 2)。")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    settings = load_settings()
    report = check_env(settings)
    if "--check" in argv:
        print(format_check(report))
        return 0 if report["ok"] else 1
    print("Fugu ランチャー — 環境を確認しています...\n")
    print(format_check(report))
    if not report["ok"]:
        print("\n(このままメニューには入れますが、実行は失敗する可能性があります)")
    try:
        return main_menu(settings, report)
    except KeyboardInterrupt:
        print("\n終了します。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
