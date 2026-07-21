#!/usr/bin/env python3
"""
Fugu Local - Gradio Web Chat UI
起動: python fugu_web.py
ブラウザが http://localhost:7860 を自動で開きます
"""
import sys
import queue
import threading
import builtins
from datetime import datetime
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, "reconfigure"):
        _s.reconfigure(errors="replace")

try:
    import gradio as gr
except ImportError:
    sys.exit("pip install gradio  が必要です")

import fugu_local as fugu

# GPU 1 基のため同時実行は 1 件に制限
_lock = threading.Lock()

# ──────────────────────────────────────────────────
# セッション管理
# "default" は従来の ~/.fugu_history.json、それ以外は ~/.fugu_sessions/<name>.json
# ──────────────────────────────────────────────────

DEFAULT_HISTORY = Path.home() / ".fugu_history.json"
SESS_DIR = Path.home() / ".fugu_sessions"

THINK_CHOICES = ["モデル既定", "OFF（高速）"]


def _session_path(name: str) -> Path:
    if not name or name == "default":
        return DEFAULT_HISTORY
    return SESS_DIR / f"{name}.json"


def _list_sessions() -> list:
    names = ["default"]
    if SESS_DIR.is_dir():
        names += sorted(p.stem for p in SESS_DIR.glob("*.json"))
    return names


def _load_chat(name: str) -> list:
    """セッション JSON を Gradio messages 形式で返す。"""
    msgs = fugu.load_history_file(_session_path(name))
    return [
        {"role": m["role"], "content": m["content"]}
        for m in msgs
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]


# ──────────────────────────────────────────────────
# バックエンド: builtins.print をキャプチャして Fugu を実行
# ──────────────────────────────────────────────────

def _run_fugu(question, use_search, rag_dirs, out_file, log_q):
    """Fugu パイプラインを実行し、print 出力を log_q へ送る。完了後 None を送信。"""
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


def _stream(message, history, use_search, think_mode, rag_dirs_str, out_file,
            session_name):
    """(チャット表示テキスト, 全処理ログ) を逐次 yield するジェネレーター。"""
    # Gradio の chatbot 履歴を fugu._HISTORY 形式に変換
    fugu._HISTORY.clear()
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role", "")
            content = str(entry.get("content") or "")
            if role == "user" and content:
                fugu._HISTORY.append({"role": "user", "content": content})
            elif role == "assistant" and content and not content.startswith("*Fugu "):
                fugu._HISTORY.append({"role": "assistant", "content": content})

    fugu.HISTORY_FILE = _session_path(session_name)
    # think:true は qwen3-coder/phi4 が 400 で拒否するため 既定/OFF のみ
    fugu.PROPOSER_THINK = False if think_mode == THINK_CHOICES[1] else None

    rag_dirs = (
        [d.strip() for d in rag_dirs_str.split(",") if d.strip()]
        if rag_dirs_str.strip() else []
    )

    log_q = queue.Queue()
    answer_ref = [None]

    def _worker():
        with _lock:
            answer_ref[0] = _run_fugu(message, use_search, rag_dirs, out_file, log_q)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    logs = []
    while True:
        try:
            item = log_q.get(timeout=0.2)
        except queue.Empty:
            if not t.is_alive():
                break
            recent = "\n".join(logs[-8:]) if logs else "Conductor が計画中..."
            yield f"*Fugu 処理中...*\n```\n{recent}\n```", "\n".join(logs)
            continue
        if item is None:
            break
        logs.append(item)
        recent = "\n".join(logs[-8:])
        yield f"*Fugu 処理中...*\n```\n{recent}\n```", "\n".join(logs)

    t.join()
    yield answer_ref[0] or "エラーが発生しました", "\n".join(logs)


# ──────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────

_MODELS_MD = (
    "**Conductor**: qwen3:4b  \n"
    "**Proposer A**: qwen3-coder:30b  \n"
    "**Proposer B**: phi4  \n"
    "**Proposer C**: gpt-oss:20b  \n"
    "**Aggregator**: qwen3-coder:30b  \n"
    "\n*質問は数分〜十数分かかります*"
)


def build_ui():
    with gr.Blocks(title="Fugu Local MoA") as demo:
        gr.Markdown(
            "# Fugu Local MoA\n"
            "完全ローカル Mixture-of-Agents (qwen3-coder:30b + phi4 + gpt-oss:20b)"
        )

        with gr.Row(equal_height=False):
            # メインチャット
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=_load_chat("default"),
                    height=540,
                    show_label=False,
                    render_markdown=True,
                    buttons=["copy"],
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="質問を入力 (Shift+Enter で改行、Enter で送信)",
                        show_label=False,
                        lines=3,
                        scale=5,
                    )
                    send = gr.Button("送信", variant="primary", scale=1, min_width=80)
                with gr.Row():
                    gr.ClearButton([msg, chatbot], value="履歴クリア")

                with gr.Accordion("処理ログ（途中経過）", open=True):
                    process_log = gr.Textbox(
                        show_label=False,
                        lines=14,
                        max_lines=30,
                        interactive=False,
                        autoscroll=True,
                        placeholder="質問を送信すると Conductor/Proposer の途中経過がここに流れます",
                    )

            # 設定サイドバー
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### セッション")
                session_dd = gr.Dropdown(
                    choices=_list_sessions(),
                    value="default",
                    show_label=False,
                    interactive=True,
                )
                new_chat = gr.Button("＋ 新しいチャット", variant="secondary")

                gr.Markdown("### 設定")
                think_mode = gr.Radio(
                    choices=THINK_CHOICES,
                    value=THINK_CHOICES[0],
                    label="思考 (thinking)",
                    info="OFF は思考トークンを省いて高速化（品質とトレードオフ）",
                )
                use_search = gr.Checkbox(label="Web 検索", value=False)
                rag_dirs = gr.Textbox(
                    label="RAG ディレクトリ (カンマ区切り)",
                    placeholder="/path/to/docs",
                    lines=2,
                )
                out_file = gr.Textbox(
                    label="出力ファイル (answer.md など)",
                    placeholder="answer.md",
                )
                gr.Markdown(_MODELS_MD)

        # ストリーミングレスポンス (Gradio 6 は messages 形式のみ)
        def _respond(message, chat_history, us, think, rd, of, sess):
            if not message.strip():
                yield message, chat_history, ""
                return
            for partial, log_text in _stream(
                message, chat_history, us, think, rd, of, sess
            ):
                new_history = chat_history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": partial},
                ]
                yield "", new_history, log_text

        inputs = [msg, chatbot, use_search, think_mode, rag_dirs, out_file, session_dd]
        outputs = [msg, chatbot, process_log]
        send.click(_respond, inputs=inputs, outputs=outputs)
        msg.submit(_respond, inputs=inputs, outputs=outputs)

        # ── セッション操作 ──
        def _new_chat():
            SESS_DIR.mkdir(exist_ok=True)
            name = "chat-" + datetime.now().strftime("%Y%m%d-%H%M%S")
            _session_path(name).write_text("[]", encoding="utf-8")
            fugu._HISTORY.clear()
            return (
                gr.Dropdown(choices=_list_sessions(), value=name),
                [],
                "",
            )

        new_chat.click(_new_chat, inputs=None,
                       outputs=[session_dd, chatbot, process_log])

        def _switch_session(name):
            chat = _load_chat(name)
            fugu._HISTORY.clear()
            fugu._HISTORY.extend(
                {"role": m["role"], "content": m["content"]} for m in chat
            )
            return chat, ""

        session_dd.change(_switch_session, inputs=[session_dd],
                          outputs=[chatbot, process_log])

    return demo


if __name__ == "__main__":
    print("Ollama 接続確認中...")
    if not fugu.setup():
        sys.exit("セットアップ失敗。Ollama が起動しているか確認してください。")
    print("起動します -> http://localhost:7860")
    try:
        theme = gr.themes.Soft(primary_hue="blue")
    except Exception:
        theme = "soft"
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        quiet=True,
        theme=theme,
    )
