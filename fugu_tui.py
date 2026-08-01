# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Fugu Local - Rich TUI (Claude Code 風ターミナル)
起動: python fugu_tui.py
"""
import sys
import queue
import threading
import builtins

for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
except ImportError:
    sys.exit("pip install rich  が必要です")

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.styles import Style
    _HAS_PT = True
except ImportError:
    _HAS_PT = False

import fugu_local as fugu

console = Console()
_lock = threading.Lock()


# ──────────────────────────────────────────────────
# バックエンド
# ──────────────────────────────────────────────────

def _run_fugu(question, use_search, rag_dirs, out_file, log_q):
    orig = builtins.print

    def _hook(*args, sep=" ", end="\n", file=None, flush=False):
        msg = sep.join(str(a) for a in args)
        if msg.strip():
            log_q.put(msg)

    builtins.print = _hook
    answer = ""
    try:
        answer = fugu.ask_fugu(
            question,
            use_search=use_search,
            rag_dirs=rag_dirs if rag_dirs else None,
            out_file=out_file if out_file else None,
        ) or ""
    except Exception as e:
        answer = f"エラー: {e}"
    finally:
        builtins.print = orig
        log_q.put(None)
    return answer


def ask_rich(question, use_search=False, rag_dirs=None, out_file=None):
    log_q = queue.Queue()
    answer_ref = [None]
    logs = []

    def _worker():
        with _lock:
            answer_ref[0] = _run_fugu(question, use_search, rag_dirs, out_file, log_q)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while True:
            try:
                item = log_q.get(timeout=0.2)
            except queue.Empty:
                if not t.is_alive():
                    break
                if logs:
                    recent = "\n".join(f"  [dim]{l}[/dim]" for l in logs[-7:])
                    live.update(f"[bold cyan]Fugu 処理中...[/bold cyan]\n{recent}")
                else:
                    live.update("[bold cyan]Fugu Conductor が計画中...[/bold cyan]")
                continue
            if item is None:
                break
            logs.append(item)
            recent = "\n".join(f"  [dim]{l}[/dim]" for l in logs[-7:])
            live.update(f"[bold cyan]Fugu 処理中...[/bold cyan]\n{recent}")

    t.join()
    answer = answer_ref[0] or ""

    if answer and not answer.startswith("エラー"):
        console.print(Panel(
            Markdown(answer),
            title="[bold green]Fugu[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
    elif answer:
        console.print(Panel(
            answer,
            title="[bold red]エラー[/bold red]",
            border_style="red",
        ))

    return answer


# ──────────────────────────────────────────────────
# REPL
# ──────────────────────────────────────────────────

_HELP = """使い方: 質問をそのまま入力して Enter を押すだけです。
  例) 91は素数ですか？
  例) このコードのバグを直して: print(1/0)

⏳ 回答には数分〜十数分かかります(複数のAIが議論して答えを作るため)。
   途中経過が流れている間はお待ちください。Ctrl+C で中断できます。

コマンド一覧(質問の代わりに入力):
  /search on|off    Web 検索を使う / やめる
  /rag <dir>        参考にするフォルダを設定 (カンマ区切りで複数可)
  /rag off          フォルダ参照をやめる
  /out <file>       次の回答をファイルに保存 (answer.md, result.py など)
  /reset            会話履歴を消してやり直す
  /history          いままでの会話を表示
  /help             このヘルプを表示
  /exit | /quit     終了"""


def _models_line():
    """導入済みモデル構成(setup 後に呼ぶ)。未解決でも落とさない。"""
    try:
        proposers = ", ".join(fugu.PROPOSERS or []) or "(未解決)"
        return f"Conductor: {fugu.CONDUCTOR or '(未解決)'}  |  Proposers: {proposers}"
    except Exception:
        return "モデル構成は起動ログを参照"


def repl():
    use_search = False
    rag_dirs = []
    out_file = None

    console.rule("[bold blue]🐡 Fugu Local — ターミナル版[/bold blue]")
    console.print(f"[dim]{_models_line()}[/dim]")
    console.print(Panel(_HELP, title="はじめての方へ", border_style="dim"))

    if _HAS_PT:
        session = PromptSession(
            history=InMemoryHistory(),
            style=Style.from_dict({"": "bold", "prompt": "bold cyan"}),
        )
        def _input():
            return session.prompt("You> ")
    else:
        def _input():
            return input("You> ")

    while True:
        try:
            raw = _input().strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]終了します[/dim]")
            break

        if not raw:
            continue

        # スラッシュ無しの help も拾う(初心者が最初に打ちがち。
        # 質問としてモデルに送って数分待たせない)
        if raw.lower() in ("help", "?", "？", "ヘルプ"):
            console.print(Panel(_HELP, title="Help", border_style="dim"))
            continue

        # ── コマンド処理 ──
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                console.print("[dim]終了します[/dim]")
                break

            elif cmd == "/help":
                console.print(Panel(_HELP, title="Help", border_style="dim"))

            elif cmd == "/reset":
                fugu._HISTORY.clear()
                fugu.save_history_file([])
                console.print("[yellow]会話履歴をクリアしました[/yellow]")

            elif cmd == "/history":
                if not fugu._HISTORY:
                    console.print("[dim](履歴なし)[/dim]")
                else:
                    for m in fugu._HISTORY:
                        role = m.get("role", "")
                        content = (m.get("content") or "")[:100].replace("\n", " ")
                        color = "cyan" if role == "user" else "green"
                        label = "You " if role == "user" else "Fugu"
                        console.print(f"[{color}]{label}[/{color}]: {content}")

            elif cmd == "/search":
                if arg.lower() == "on":
                    use_search = True
                    console.print("[green]Web 検索: ON[/green]")
                elif arg.lower() == "off":
                    use_search = False
                    console.print("[yellow]Web 検索: OFF[/yellow]")
                else:
                    console.print(f"Web 検索: {'ON' if use_search else 'OFF'}")

            elif cmd == "/rag":
                if arg.lower() == "off":
                    rag_dirs = []
                    console.print("[yellow]RAG: 無効[/yellow]")
                elif arg:
                    rag_dirs = [d.strip() for d in arg.split(",") if d.strip()]
                    console.print(f"[green]RAG ディレクトリ: {rag_dirs}[/green]")
                else:
                    console.print(f"RAG: {rag_dirs if rag_dirs else '無効'}")

            elif cmd == "/out":
                if arg:
                    out_file = arg
                    console.print(f"[green]次の回答を保存: {out_file}[/green]")
                else:
                    out_file = None
                    console.print("[yellow]出力ファイル: 無効[/yellow]")

            else:
                console.print(f"[red]不明なコマンド: {cmd}  (/help で一覧)[/red]")
            continue

        # ── Fugu に質問 ──
        console.print(Panel(
            raw,
            title="[bold cyan]You[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        ))
        console.print("[dim]⏳ 複数のAIが議論して答えを作ります"
                      "(数分〜十数分)。Ctrl+C で中断できます。[/dim]")

        ask_rich(raw, use_search=use_search,
                 rag_dirs=rag_dirs or None, out_file=out_file)

        # /out は 1 回限り
        if out_file:
            out_file = None


if __name__ == "__main__":
    console.print("[dim]Ollama 接続確認中...[/dim]")
    if not fugu.setup():
        console.print("[red]セットアップ失敗。Ollama が起動しているか確認してください。[/red]")
        sys.exit(1)
    repl()
