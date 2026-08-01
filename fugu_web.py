# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Fugu Local - Gradio Web Chat UI
起動: python fugu_web.py
ブラウザが http://localhost:7860 を自動で開きます
"""
import os
import sys
import queue
import tempfile
import threading
import builtins
from datetime import datetime
from pathlib import Path

import fugu_artifacts

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

# 拡張思考(fugu_thinking): OFF なら従来動作。6段階+auto。値は
# FUGU_THINKING_BUDGET env 経由で fugu_answer のフックに伝わる
# （CLI --thinking-budget と同じ経路）。
BUDGET_CHOICES = ["OFF", "minimal", "low", "medium", "high", "ultra", "max", "auto"]

# Canvas エクスポートの書き出し先（gr.File はパスを受けるためファイル化が必要）
ARTIFACT_DIR = Path(tempfile.gettempdir()) / "fugu_artifacts"


def _update_canvas(chat_history, prev_code):
    """最後のアシスタント回答から Canvas(Preview/Code/Diff/Export)を更新する。
    ロジックは fugu_artifacts に隔離済み — ここは配線のみ(Gradio ドリフト対策)。"""
    last = ""
    for m in reversed(chat_history or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            last = str(m.get("content") or "")
            break
    view = fugu_artifacts.build_canvas(last, prev_code or "")
    file_path = None
    if view["has_artifact"]:
        try:
            ARTIFACT_DIR.mkdir(exist_ok=True)
            file_path = str(ARTIFACT_DIR / view["filename"])
            Path(file_path).write_text(str(view["code"]), encoding="utf-8")
        except OSError:
            file_path = None  # エクスポート不可でもプレビューは出す
    new_prev = view["code"] if view["has_artifact"] else (prev_code or "")
    return view["preview_html"], view["code"], view["diff"], file_path, new_prev


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


def _stream(message, history, use_search, think_mode, budget_mode, rag_dirs_str,
            out_file, session_name):
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

    # 思考予算: OFF は env を消して従来動作に戻す（他セッションへ漏らさない）
    if budget_mode and budget_mode != "OFF":
        os.environ["FUGU_THINKING_BUDGET"] = budget_mode
    else:
        os.environ.pop("FUGU_THINKING_BUDGET", None)

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

#: ワンクリックで試せる例文(初心者向け)。クリックすると入力欄に入る。
EXAMPLE_QUESTIONS = [
    "91は素数ですか？",
    "このPythonコードのバグを直して: print(1/0)",
    "次の文章を丁寧な英語に翻訳して: 明日の会議は10時からです",
    "簡単なToDoリストのHTMLページを作って",
]

_GUIDE_MD = """\
**このアプリは、あなたのPCの中だけで動くAIチャットです**(インターネット上のAIには送信されません)。

1. 下の入力欄に質問を書いて **送信** を押す(例文をクリックしても入ります)
2. ⏳ **回答には数分〜十数分かかります**。複数のAIが議論して答えを作るためです。
   「処理ログ」に途中経過が流れていれば正常に動いています
3. 回答が表示されたら、続けて追加の質問ができます(会話は覚えています)

**よくある質問**
- *止めたいとき*: このウィンドウを閉じるか、起動した黒い画面で Ctrl+C
- *新しい話題にしたいとき*: 右の「＋ 新しいチャット」
- *最新情報を調べてほしいとき*: 右の「Web 検索」にチェック
- *コードやHTMLを作らせたとき*: 右端の Canvas に自動でプレビューが出ます
"""


def _models_md():
    """導入済みモデル構成を動的に表示(setup 後に呼ばれる前提。未解決なら省略表示)。"""
    try:
        lines = [f"**Conductor**: {fugu.CONDUCTOR or '(未解決)'}  "]
        for label, model in fugu.PERSONA_MODELS.items():
            mark = "" if model in (fugu.PROPOSERS or []) else " (未導入)"
            lines.append(f"**{label}**: {model}{mark}  ")
        lines.append(f"**Aggregator**: {fugu.AGGREGATOR or '(未解決)'}  ")
        return "\n".join(lines) + "\n\n*質問は数分〜十数分かかります*"
    except Exception:
        return "*モデル構成は起動ログを参照*"


def build_ui():
    with gr.Blocks(title="Fugu Local MoA") as demo:
        gr.Markdown(
            "# 🐡 Fugu Local\n"
            "**完全ローカルのAIチャット** — 質問を入力して送信するだけ。"
            "複数のAIが議論してから答えます(そのぶん時間はかかります)。"
        )
        with gr.Accordion("❓ はじめての方へ(使い方)", open=False):
            gr.Markdown(_GUIDE_MD)

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
                        placeholder="ここに質問を入力 (Enter で送信 / Shift+Enter で改行)",
                        show_label=False,
                        lines=3,
                        scale=5,
                    )
                    send = gr.Button("送信 ▶", variant="primary", scale=1,
                                     min_width=80)
                gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUESTIONS],
                    inputs=[msg],
                    label="例文(クリックすると入力欄に入ります)",
                )
                with gr.Row():
                    gr.ClearButton([msg, chatbot], value="🗑 履歴クリア")

                with gr.Accordion("処理ログ（いま何をしているか）", open=True):
                    process_log = gr.Textbox(
                        show_label=False,
                        lines=14,
                        max_lines=30,
                        interactive=False,
                        autoscroll=True,
                        placeholder="質問を送信すると途中経過がここに流れます。"
                                    "流れている間は AI が考え中です — "
                                    "数分〜十数分かかるのが正常です",
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
                use_search = gr.Checkbox(
                    label="Web 検索を使う", value=False,
                    info="最新の情報が必要な質問(ニュース・相場など)のときにオン",
                )
                with gr.Accordion("詳細設定(通常は変更不要)", open=False):
                    think_mode = gr.Radio(
                        choices=THINK_CHOICES,
                        value=THINK_CHOICES[0],
                        label="思考モード",
                        info="OFF にすると速くなりますが、答えの質は下がることがあります",
                    )
                    budget_mode = gr.Radio(
                        choices=BUDGET_CHOICES,
                        value=BUDGET_CHOICES[0],
                        label="考える深さ",
                        info="深いほど丁寧に考えますが時間がかかります。"
                             "auto は質問の難しさで自動調整",
                    )
                    rag_dirs = gr.Textbox(
                        label="参考にするフォルダ (RAG)",
                        placeholder="例: D:\\docs (複数はカンマ区切り)",
                        info="手元の文書フォルダを指定すると、その内容を参照して答えます",
                        lines=2,
                    )
                    out_file = gr.Textbox(
                        label="回答の保存先ファイル",
                        placeholder="例: answer.md",
                        info="指定すると回答をファイルにも保存します"
                             "(.md .py .pdf .docx .xlsx など)",
                    )
                with gr.Accordion("使用モデル", open=False):
                    gr.Markdown(_models_md())

            # Canvas / Artifacts ワークスペース（右ペイン）
            with gr.Column(scale=3, min_width=320):
                gr.Markdown("### Canvas（作ったものが映る画面）\n"
                            "<small>HTMLやコードを作らせると、ここに自動で"
                            "プレビューが出ます</small>")
                canvas_prev = gr.State("")
                with gr.Tabs():
                    with gr.Tab("Preview"):
                        preview_html = gr.HTML(fugu_artifacts.EMPTY_PREVIEW)
                    with gr.Tab("Code"):
                        code_view = gr.Code(value="", language=None,
                                            interactive=False, lines=22)
                    with gr.Tab("Diff"):
                        diff_view = gr.Code(value="", language=None,
                                            interactive=False, lines=22)
                    with gr.Tab("Export"):
                        export_file = gr.File(label="ダウンロード", value=None)

        # ストリーミングレスポンス (Gradio 6 は messages 形式のみ)
        def _respond(message, chat_history, us, think, budget, rd, of, sess):
            if not message.strip():
                yield message, chat_history, ""
                return
            for partial, log_text in _stream(
                message, chat_history, us, think, budget, rd, of, sess
            ):
                new_history = chat_history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": partial},
                ]
                yield "", new_history, log_text

        inputs = [msg, chatbot, use_search, think_mode, budget_mode, rag_dirs, out_file,
                  session_dd]
        outputs = [msg, chatbot, process_log]
        canvas_in = [chatbot, canvas_prev]
        canvas_out = [preview_html, code_view, diff_view, export_file, canvas_prev]
        send.click(_respond, inputs=inputs, outputs=outputs).then(
            _update_canvas, inputs=canvas_in, outputs=canvas_out)
        msg.submit(_respond, inputs=inputs, outputs=outputs).then(
            _update_canvas, inputs=canvas_in, outputs=canvas_out)

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
