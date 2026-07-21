#!/usr/bin/env python3
"""
Fugu Local - Rich TUI (Claude Code 風ターミナル)
起動: python fugu_tui.py
"""
import sys
import queue
import threading
import builtins
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    from rich.rule import Rule
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

_HELP = """コマンド一覧:
  /search on|off    Web 検索を有効/無効化
  /rag <dir>        RAG ディレクトリを設定 (カンマ区切りで複数可)
  /rag off          RAG を無効化
  /out <file>       次の回答をファイルに保存 (answer.md, result.py など)
  /reset            会話履歴をクリア
  /history          現在の会話履歴を表示
  /help             このヘルプを表示
  /exit | /quit     終了"""


def repl():
    use_search = False
    rag_dirs = []
    out_file = None

    console.rule("[bold blue]Fugu Local MoA[/bold blue]")
    console.print(
        "[dim]Conductor: qwen3:4b  |  "
        "Proposers: qwen3-coder:30b, phi4, gpt-oss:20b[/dim]"
    )
    console.print("[dim]/help でコマンド一覧  |  /exit で終了[/dim]\n")

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
