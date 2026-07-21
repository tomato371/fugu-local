"""
Local Fugu-style Orchestrator (RTX 4060 Laptop 8GB VRAM / RAM 48GB / i7-13700H 向け)

本家 Sakana Fugu との違いを埋めるための版。
古典的な静的 MoA（＝Sakana でいう "Fusion"）は「固定のプロポーザー全員が毎回走り、
固定のアグリゲーターが統合する」だけだが、Fugu の肝は指揮者(Conductor)LLM 自身が
「単体で十分か / 誰を何体使うか / 何回反復するか / 追加ラウンドが要るか」を
"動的に" 決めるところにある。

そこでこの版では:
  1) Conductor が質問を見て実行プランを JSON で出す（単体 or 合議、使うモデル、ラウンド数）
  2) 簡単な質問は 1 モデルで即答（MoA のオーバーヘッドを払わない）
  3) 単体回答が弱ければ Critic が検知して合議へ "エスカレーション"
  4) 合議後も不十分なら上限付きで "再帰的に" 追加ラウンド
を行う。全てローカルモデルだけで完結。
"""

import os
import re
import sys
import json
import time
import shutil
import tempfile
import argparse
import subprocess
import urllib.request
import urllib.parse
import concurrent.futures  # 並列処理用（8GB では既定で逐次）
from pathlib import Path

# Windows の cp932 コンソール/パイプでは ⚠ ✓ ❌ ⤴ ↻ 等の表示記号が encode できず、
# print 自体が UnicodeEncodeError で落ちる（実測 2026-07-04: 保険1の通知 print がクラッシュし、
# 空返答の救済パスそのものが死んだ）。記号が化けるのは許容し、encode 不能文字は置換して続行する。
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

# 推論は Ollama native /api/chat を urllib で直接叩く（依存ゼロ）。
# 旧版は openai クライアント + /v1 互換エンドポイントを使っていたが、/v1 は num_ctx を
# 無視してモデル最大 context(例: qwen3 は 262144)を確保しようとし、8GB VRAM では
# KV キャッシュ確保に失敗して llama-server がクラッシュする（500）。native /api/chat なら
# options.num_ctx がリクエスト単位で効くため、これで context を安全域に固定する。

# ==================================================
# 設定
# ==================================================

OLLAMA_URL = "http://localhost:11434"

# --- モデルの役割 ---
# Conductor + Critic: qwen3:4b（軽量・高速・JSON安定。ルーティング専用、VRAM常駐最小化）
# Proposers: qwen3-coder:30b(コード特化MoE) / phi4(数学・PINN・物理) / gpt-oss:20b(汎用推論MoE)
#   8GB VRAM 環境では大型モデルは RAM オフロード実行（Ollama 自動制御、48GB RAM で吸収）。
#   RTX 4090 24GB 移行後はすべて VRAM 内で動作。
# Aggregator: qwen3-coder:30b（コード統合に最適）
# JP Aggregator: qwen3:4b（日本語は実績ある qwen3 で確実に処理）
# ドキュメント "3大AIオールスター" 構成: A=GPT / B=Claude / C=Gemini + D=理数専門家。
# 順序はペルソナ A,B,C,D に対応させる（gemma-4-26b-a4b の実タグは gemma4:26b）。
DESIRED_PROPOSERS = ["gpt-oss:20b", "qwen3-coder:30b", "gemma4:26b", "qwen3.6:35b"]
DESIRED_AGGREGATOR = "qwen3-coder:30b"
DESIRED_CONDUCTOR = "qwen3:4b"
FALLBACK_MODEL = "qwen3:4b"

# --- ペルソナ層（3大AIオールスター）---
# Conductor は selected_proposers に「Proposer A」〜「Proposer D」のペルソナ名を出す。
# ここで実モデルへ解決する（_resolve_proposer 参照）。モデル未導入なら解決結果から除外される。
PERSONA_MODELS = {
    "Proposer A": "gpt-oss:20b",      # ChatGPT(GPT)の存在
    "Proposer B": "qwen3-coder:30b",  # Claudeの存在
    "Proposer C": "gemma4:26b",       # Geminiの存在
    "Proposer D": "qwen3.6:35b",      # 理数・物理・PINN 専門家（2026-07-11: phi4 → qwen3.6:35b
                                      # へ更新。思考型 MoE(A3B) で数学・推論が大幅に強い）
}
# 各 proposer に注入する人格プロンプト（PROPOSER_SYS の前に前置し「個性」を再現する）
PERSONA_IDENTITY = {
    "gpt-oss:20b":     "あなたは『ChatGPT(GPT)の存在』。バランス感覚に優れ、一般的な対話と文章の骨組み作りを担当する。",
    "qwen3-coder:30b": "あなたは『Claudeの存在』。高度なプログラミング、厳密な論理チェック、コードの自己修復を担当する。",
    "gemma4:26b":      "あなたは『Geminiの存在』。RAG(Office文書)のコンテキスト分析、大量ドキュメントとWeb検索結果の集約を担当する。",
    "qwen3.6:35b":     "あなたは理数・物理・PINN(物理情報ニューラルネット)・偏微分方程式の専門家。厳密に段階を追って考える。",
}
MODEL_TO_PERSONA = {v: k for k, v in PERSONA_MODELS.items()}

# 日本語質問では qwen3-coder の日本語品質が未検証のため、実績ある qwen3:4b に切替える。
JP_AGGREGATOR = "qwen3:4b"

# 統合役の強化（2026-07-11）: 統合は「最終の誤り検出」なので思考モデルにやらせるのが筋。
# - コードを含む統合 → qwen3-coder:30b（従来どおり。実行検証タグの扱いに実績）
# - 日本語の記述式 → qwen3.6:35b（思考ON・日本語堅牢。JPサニティ不合格なら "" にして qwen3:4b 維持）
# - 英語の記述式 → gpt-oss:20b（think:"high" が MODEL_CONFIG で自動適用される）
JP_AGGREGATOR_STRONG = "qwen3.6:35b"
AGGREGATOR_REASONING = "gpt-oss:20b"

# 自己評価バイアス対策: Conductor/Critic が qwen3:4b のため、別系統モデルで独立チェックする。
# gpt-oss:20b は OpenAI 系で qwen3 と出自が異なるため second opinion に適切。
SECOND_OPINION_MODEL = "gpt-oss:20b"

# Conductor がモデル選抜に使う「各プロポーザーの得意分野」ヒント（ペルソナ付き）
PROPOSER_PROFILES = {
    "gpt-oss:20b":     "ChatGPT(GPT)の存在。バランス・一般的な対話・文章の骨組み (OpenAI OSS MoE・3.6B active・思考high対応)",
    "qwen3-coder:30b": "Claudeの存在。高度なプログラミング・厳密な論理チェック・自己修復 (SWE-bench最強クラスMoE)",
    "gemma4:26b":      "Geminiの存在。RAG(Office文書)分析・大量ドキュメント・Web検索結果の集約 (26B)",
    "qwen3.6:35b":     "理数・物理・PINN・偏微分方程式・アルゴリズム証明に強い思考型 (35B MoE A3B, 2026-02世代)",
}

# --- 生成パラメータ ---
# 大型モデルの RAM オフロード時に1コール最大2時間を確保（8192tok × 1tok/s の保険）。
REQUEST_TIMEOUT = 7200
PROPOSER_TEMP = 0.7       # 多様性のため高め
AGGREGATOR_TEMP = 0.5
CONDUCTOR_TEMP = 0.1      # 判定はブレを抑える

# proposer の思考(thinking)制御。None=モデル既定(qwen3/gemma は思考ON=高品質だが遅い)、
# False=思考を無効化して高速化(思考トークンは最終回答に使わず破棄されるため、8GB逐次では
# 大きな時間短縮になる)。品質とのトレードオフなので既定は None。非思考モデル(phi4-mini)に
# False を渡しても無害(実測でエラーなし)。
PROPOSER_THINK = None       # None=モデル既定。think:true を送ると qwen3-coder/phi4 等が
                            # 400 "does not support thinking" で即失敗するため送らない。
                            # gpt-oss:20b は think:true 対応済み（実測 2026-07-05）だが、
                            # 個別制御より既定委任の方が安全。

# --- モデル別推論設定（精度最優先の中核。2026-07-11 追加）---
# think:  None=モデル既定 / False=無効 / True=有効 / "low"|"medium"|"high"=gpt-oss 系の
#         思考量段階指定（Ollama 0.31.2 で "high" が gpt-oss:20b に効くことを実測済み。
#         high は数学・推論の精度が跳ね上がる最大のレバー）。
#         PROPOSER_THINK が None 以外ならそちらが最優先（eval の一括 OFF 等を維持）。
# num_ctx: モデル別コンテキスト長。従来は全モデル一律 MODEL_NUM_CTX=8192 だったが、
#         それは「VRAM 常駐する小型モデルの KV 上限」の話。大型モデルは重みも KV も
#         RAM オフロードで動くため 32768 まで拡大できる（思考が 8k を超えて打ち切られる
#         事故＝done_reason=length・本文空、を根本から防ぐ）。qwen3:4b 等の VRAM 常駐組は
#         8192 を維持して高速なまま使う。
# num_predict: 生成上限（思考トークン込み）。num_ctx 拡大に合わせて引き上げる。
# 【Phase 0 実測 2026-07-12, RTX 4060 8GB / RAM48GB / Ollama 0.31.2】
# - gpt-oss:20b think:"high" は動作OK。ただし num_ctx で速度が激変:
#   8192→14.9 tok/s / 32768→3.4 tok/s（KV が RAM オフロードされ帯域律速）。
# - qwen3.6:35b think:true は num_predict=8192 だと長い思考で使い切り本文ゼロ(done=length)に
#   なる実測。num_predict は厚めに、num_ctx は 16384 で throughput を確保するのが妥当。
# - VibeThinker-3B は 11.2 tok/s・VRAM 常駐、<think> タグは content 内(strip_think が除去)。
# 方針: 思考モデルは num_ctx=16384（8192 だと AIME の思考が入り切らず、32768 は遅すぎる中間点）、
# num_predict は打ち切り回避のため厚め。非思考の大型は 16384 で十分。
MODEL_CONFIG = {
    "gpt-oss:20b":     {"think": "high", "num_ctx": 16384, "num_predict": 14336},
    "gpt-oss:120b":    {"think": "high", "num_ctx": 12288, "num_predict": 8192},
    "qwen3.6:35b":     {"think": True,  "num_ctx": 16384, "num_predict": 14336},
    "NitrAI/VibeThinker-3B": {"num_ctx": 16384, "num_predict": 14336},
    "qwen3-coder:30b": {"num_ctx": 16384, "num_predict": 12288},
    "gemma4:26b":      {"num_ctx": 16384, "num_predict": 12288},
}


def model_cfg(model, key, default=None):
    """MODEL_CONFIG からモデル別設定を引く（無ければ default）。"""
    return MODEL_CONFIG.get(model, {}).get(key, default)


# ==================================================
# 大 VRAM プロファイル（将来の 96GB 等の環境向け・一発切り替え）
# ==================================================
# 8GB ラップトップは「大型モデルを RAM/NVMe にオフロードして逐次で回す」制約下にあるが、
# nk108 は本手法が有効なら VRAM 96GB 級の環境で実験予定。そこでは全モデルが VRAM 常駐でき、
# 制約が一変する（並列プロポーザー可・context 大幅拡大・SC のサンプル数を大量に増やせる・
# 120b arbiter も高速）。環境変数 FUGU_HIGH_VRAM=1 で下記を一括適用する（コード改変不要）。
#   PowerShell: $env:FUGU_HIGH_VRAM=1 ; python fugu_local.py ...
# 値は 96GB を想定した保守的な既定。より大きい環境ならさらに引き上げてよい。
def apply_high_vram_profile():
    """VRAM 潤沢環境向けに設定を一括で引き上げる。setup() 冒頭で env 判定して呼ぶ。"""
    global MODEL_CONFIG, PARALLEL_PROPOSERS, MODEL_NUM_CTX
    global SC_INITIAL, SC_STEP, SC_MAX, SC_CHEAP_VOTES, MODEL_KEEP_ALIVE, ARBITER_MODEL
    print("[setup] FUGU_HIGH_VRAM=1 → 大VRAMプロファイルを適用します")
    # 全モデル常駐前提: context を広げ生成上限も引き上げる（KV が VRAM に載るため安全）
    for m, cfg in MODEL_CONFIG.items():
        cfg["num_ctx"] = 65536
        cfg["num_predict"] = 32768
    MODEL_NUM_CTX = 32768
    # 96GB なら複数モデルを同時常駐でき、プロポーザー並列が効く（8GB では逆効果だった）
    PARALLEL_PROPOSERS = True
    MODEL_KEEP_ALIVE = "30m"          # 常駐維持でロード/アンロードの往復を消す
    # サンプルを大量に回せる＝自己一貫性の精度が上がる主レバー
    SC_INITIAL, SC_STEP, SC_MAX = 12, 8, 48
    SC_CHEAP_VOTES = 16               # VibeThinker を大量票に（多様性の底上げ）
    # 96GB では 65GB の 120b が VRAM 常駐できるため、拮抗時の裁定をローカル最上位知能に任せる
    # （8GB 既定は NVMe ページング回避で qwen3.6:35b）
    ARBITER_MODEL = "gpt-oss:120b"
    print(f"[setup] high-vram: num_ctx=65536 parallel=ON SC(init={SC_INITIAL},max={SC_MAX}) "
          f"cheap_votes={SC_CHEAP_VOTES} arbiter={ARBITER_MODEL}")


def proposer_think_for(model):
    """proposer の think 解決: グローバル PROPOSER_THINK(≠None) > MODEL_CONFIG > モデル既定。"""
    if PROPOSER_THINK is not None:
        return PROPOSER_THINK
    return model_cfg(model, "think")


def proposer_predict_for(model):
    """proposer/aggregator の生成上限: MODEL_CONFIG > 役割既定。"""
    return model_cfg(model, "num_predict", NUM_PREDICT_PROPOSER)

# --- 生成長の上限（暴走保険）---
# 未指定だと思考モデルの生成が無制限で、実測では deepseek-r1 が統合 1 回に ~4100 トークン
# (411秒) 生成した例がある。精度優先のため「打ち切り」が起きない余裕を持たせた上限とし、
# タイトな時間予算にはしない。目的は無限の暴走を有界にすることだけ。
# 【実測の教訓 2026-07-04】上限は思考トークンも消費する。長いコード/証明の統合では
# 思考だけで 5120 を食い尽くし「done_reason=length・本文空」で終わる事象が 3 回発生
# （保険2が救済）。上限は num_ctx=8192 から入力(~2k)を引いた範囲で最大限に取り、
# ask() 側で「思考中に打ち切られて本文空」を __ERROR__ として可視化する。
NUM_PREDICT_PROPOSER = 8192      # 時間無制限・打ち切りゼロのため上限最大化
NUM_PREDICT_AGGREGATOR = 8192    # 統合も上限最大化
NUM_PREDICT_JUDGE = 768          # Conductor/Critic の高速JSON(think=False)
NUM_PREDICT_JUDGE_THINK = 6144   # Critic 再検算（思考トークンに十分な余裕）

# --- Fugu 風オーケストレーションの挙動 ---
MAX_ROUNDS = 4            # 時間無制限・精度優先のため反復を増やす
ADAPTIVE_ESCALATION = True  # 単体回答が弱いと合議へ格上げ
ALLOW_RECURSION = True      # 合議後、批評 → 必要なら追加ラウンド

# --- コード実行検証（主用途: コード生成の自律修正ループ）---
# 回答中の ```python ブロックを実際に subprocess で実行し、失敗したら traceback を
# 次ラウンドの修正ヒントとして渡す。LLM の自己審査と違い実行結果は決定的なので、
# コードに関しては最強の Critic になる。エラーが残る限り MAX_ROUNDS_CODE まで
# 修正ラウンドを繰り返す（nk108 の方針: 時間をかけてでも精度優先）。
# 注意: 生成コードをこのマシンで直接実行する。信頼できる自分の質問にだけ使うこと。
CODE_EXECUTION = True
CODE_EXEC_TIMEOUT = 15      # 秒。input() 待ちや無限ループはタイムアウトで失敗扱い
MAX_ROUNDS_CODE = 8         # コード修正は時間をかけて完全に直す

# --- 表示 ---
SHOW_PLAN = True          # Conductor の判断を表示（Fugu 風の動作を可視化）
SHOW_PROPOSALS = True     # 各提案を表示（think は除去して表示）
SHOW_BASELINE = False     # 比較用の単体直答。学習用に True でも可

# --- 計測（フェーズ3用）---
# True にすると ask() が各呼び出しの (label, model, 秒) を _TIMINGS に記録する。
# 段階別（conductor/proposer/aggregator/critic）の所要時間を可視化するための軽量フック。
SHOW_TIMING = False
_TIMINGS = []

# --- 実行時フラグ ---
_SECOND_OPINION_DISABLED = False  # second opinion が未インストール時に True にセット

# --- セッション永続化 ---
# 会話履歴を JSON ファイルに保存し、次回起動時に復元する。
HISTORY_FILE: Path = Path.home() / ".fugu_history.json"
SESSION_SAVE = True          # False にすると永続化を無効化（--no-history フラグでも制御）
MAX_HISTORY_TURNS_SAVED = 50 # ファイルに保存する最大往復数（古い順に削除）

# --- Web 検索 ---
# duckduckgo_search パッケージ（pip install duckduckgo_search）が入っていれば
# フル検索結果を取得。未インストール時は DuckDuckGo Instant Answer API（urllib 内蔵）
# にフォールバック（インスタント回答のみ・件数少ない）。
WEB_SEARCH_MAX_RESULTS = 5       # 1 クエリあたりの取得件数
WEB_SEARCH_SNIPPET_CHARS = 400   # 各スニペットの文字数上限
WEB_SEARCH_TIMEOUT = 15          # 秒

# --- 反復リサーチ ---
# 1 回の検索では具体的事実（型番・アーキテクチャ名等）が欠けたまま、モデルが古い学習知識で
# 穴埋めする事故が起きる（実測 2026-07-06: RTX 5090 のアーキテクチャを Hopper と誤答）。
# そこで Conductor(qwen3:4b) に「十分な事実が集まったか」を判定させ、不足なら不足点を狙った
# 追加クエリを生成して検索を繰り返す。
SEARCH_MAX_ROUNDS = 3            # リサーチ反復の上限（十分と判定されたら早期終了）
SEARCH_CONTEXT_CHARS = 4000      # 質問に注入する検索コンテキスト上限（num_ctx=8192 の安全域）

# --- RAG（ローカル文書検索）---
# RAG_DIRS に 1 つ以上のディレクトリを指定すると、質問と関連するチャンクを
# 自動抽出してプロポーザーへのコンテキストとして注入する。
# CLI: --rag /path/to/docs  または  --rag dir1 dir2 ...
RAG_DIRS: list = []              # 空 = 無効
RAG_CHUNK_CHARS = 600            # チャンクサイズ（文字数）
RAG_CHUNK_OVERLAP = 100          # チャンク間のオーバーラップ文字数
RAG_TOP_K = 3                    # 注入する上位チャンク数
RAG_EXTENSIONS = {
    # テキスト・コード
    ".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".env",
    ".py", ".ipynb", ".r", ".m", ".jl",
    ".js", ".ts", ".jsx", ".tsx", ".mjs",
    ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".java", ".kt", ".swift",
    ".rb", ".php", ".sh", ".bat", ".ps1", ".sql", ".graphql",
    ".html", ".htm", ".xml", ".svg", ".css", ".scss",
    ".tex", ".bib",
    # ドキュメント（要ライブラリ）
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".odt", ".ods", ".odp",
}

# --- 画像生成（ローカル Stable Diffusion / ComfyUI 連携）---
# Ollama はテキスト専用のため、画像生成は別バックエンドへ委譲する。Conductor が
# use_image_generation=true を出すと MoA を経由せず画像エージェントへバイパスする
# （ドキュメントの特殊ルーティング #1）。AUTOMATIC1111 stable-diffusion-webui
# (/sdapi/v1/txt2img・簡潔) を主とし、ComfyUI (/prompt+/history+/view) をフォールバック。
IMAGE_BACKEND = "auto"                       # "auto" | "a1111" | "comfyui" | "off"
A1111_URL = "http://127.0.0.1:7860"          # stable-diffusion-webui 既定ポート
COMFYUI_URL = "http://127.0.0.1:8188"        # ComfyUI 既定ポート
IMAGE_OUT_DIR = Path.home() / "fugu_images"  # 生成画像の保存先
IMAGE_STEPS = 30
IMAGE_WIDTH = 1024   # SDXL ネイティブ解像度（RTX 4060 8GB で実測OK。SD1.5 利用時は 512〜768 推奨）
IMAGE_HEIGHT = 1024
IMAGE_TIMEOUT = 600                          # 秒（ローカルGPUでの生成待ち上限）
IMAGE_TRANSLATE_PROMPT = True                # 日本語要求を英語SDプロンプトへ変換（qwen3:4b・MoA無効時のフォールバック）
COMFYUI_CKPT = ""                            # ComfyUI 用チェックポイント名（空ならサーバ既定を自動取得）
IMAGE_PROMPT_MOA = True                       # True: 画像プロンプトを LLM 群(proposers)で起草→統合。False: qwen3単独翻訳
IMAGE_PROMPT_PANEL = 2                        # プロンプト起草に使う proposer 数の上限（精度と速度の折衷）

# --- PowerPoint 生成（画像入りスライド）---
# make_pptx または --out X.pptx で、MoA が作った本文をスライド化し、内容連動で画像を埋め込む。
PPTX_OUT_DIR = Path.home() / "fugu_pptx"     # make_pptx 時の既定保存先
PPTX_MAX_SLIDES = 12                          # 生成する最大スライド数（タイトル除く）
PPTX_MAX_IMAGES = 4                           # 生成する最大画像枚数（タイトル画像含む・内容連動）
PPTX_MAX_BULLETS = 7                          # 1 スライドの最大箇条書き数

# --- 会話履歴 ---
# (user, assistant) ペアを保持し、古い交換を削除して num_ctx に収める。
# 14B モデル + num_ctx=8192 の場合、入力の余裕は ~2000 トークン（~6000文字）程度。
# 保守的に 4000 文字を上限とし、超えた分を先頭ペアから削除する。
MAX_HISTORY_CHARS = 4000
_HISTORY: list = []   # グローバル会話履歴

# --- VRAM 対策 ---
# 逐次実行でも Ollama は keep_alive でモデルを常駐させ続けるため、複数モデルだと
# 呼び出しごとにロード/アンロード（数GBのディスク読み込み）が多発して遅くなる。
# 最も効くのは「同時ロードは 1 体」を強制する環境変数（サーバ起動前に設定）:
#   Windows(PowerShell): $env:OLLAMA_MAX_LOADED_MODELS=1 ; ollama serve
#   Linux / mac:         OLLAMA_MAX_LOADED_MODELS=1 ollama serve
# 下の keep_alive をコード側から渡したい場合のみ文字列を設定（例 "0"=即アンロード, "5m"）。
# 既定 None は「渡さない」＝互換性リスクなし。
MODEL_KEEP_ALIVE = None

# 【重要】コンテキスト長。未指定だと Ollama がモデル最大(qwen3=262144 等)を確保しようとし、
# 8GB VRAM では KV キャッシュが破綻して runner がクラッシュする。実測で 8192 なら
# deepseek-r1:7b / gemma4:e2b-it-qat / qwen3:4b / phi4-mini いずれも VRAM に収まり安定。
# MoA のアグリゲータは「質問＋複数提案＋推論」で入力が伸びるので、これ以上は下げない方がよい。
MODEL_NUM_CTX = 8192

# 2026-07-02 実測: e2b-it-qat 置換後は 3 プロポーザー同時常駐が可能（qwen3 3.9 + phi4 3.7 +
# gemma-qat 1.9 = 9.5GB 表示でもクラッシュなし）だが、並列はむしろ遅い
# （warm・think=False で逐次 177.8s vs 並列 208.2s = x0.85）。GPU 演算が 1 基で奪い合いに
# なるため。よって False を維持する（安全性の問題ではなく速度メリットが無い）。
PARALLEL_PROPOSERS = False

PROPOSERS = []
AGGREGATOR = None
CONDUCTOR = None

# ==================================================
# Ollama ブートストラップ
# ==================================================


# ==================================================
# セッション永続化
# ==================================================

def load_history_file(path: Path = None) -> list:
    """JSON ファイルから会話履歴を読み込む。ファイルが無い/壊れている場合は空リストを返す。"""
    path = path or HISTORY_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # 最新 MAX_HISTORY_TURNS_SAVED 往復分のみ保持
            msgs = [m for m in data if isinstance(m, dict) and "role" in m and "content" in m]
            return msgs[-(MAX_HISTORY_TURNS_SAVED * 2):]
    except Exception:
        pass
    return []


def save_history_file(history: list, path: Path = None):
    """会話履歴を JSON ファイルに保存する。SESSION_SAVE=False 時は何もしない。"""
    if not SESSION_SAVE:
        return
    path = path or HISTORY_FILE
    try:
        path.write_text(
            json.dumps(history[-(MAX_HISTORY_TURNS_SAVED * 2):],
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"   [履歴保存エラー: {e}]")


# ==================================================
# Slack 完了通知
# ==================================================

# Slack Incoming Webhook URL。環境変数 FUGU_SLACK_WEBHOOK に設定すると
# ask_fugu() 完了時（成功・失敗とも）に通知を送る。未設定なら何もしない。
# 1問数分〜十数分かかるため、離席していても完了が分かるようにする。
SLACK_WEBHOOK_URL = os.environ.get("FUGU_SLACK_WEBHOOK", "")
SLACK_NOTIFY_TIMEOUT = 10    # 秒。通知は本処理を止めない
SLACK_Q_PREVIEW = 200        # 通知に載せる質問の文字数上限
SLACK_A_PREVIEW = 500        # 通知に載せる回答の文字数上限


def _slack_truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def notify_slack(question: str, answer: str, elapsed: float):
    """完了通知を Slack Incoming Webhook へ送る。失敗しても本処理には影響させない。"""
    if not SLACK_WEBHOOK_URL:
        return
    ok = not (answer or "").startswith("__ERROR__")
    icon = ":white_check_mark:" if ok else ":x:"
    head = f"Fugu {'完了' if ok else '失敗'} ({elapsed} 秒)"
    text = (
        f"{icon} *{head}*\n"
        f"*Q:* {_slack_truncate(question, SLACK_Q_PREVIEW)}\n"
        f"*A:* {_slack_truncate(answer, SLACK_A_PREVIEW)}"
    )
    req = urllib.request.Request(
        SLACK_WEBHOOK_URL,
        data=json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=SLACK_NOTIFY_TIMEOUT)
        print("   [Slack 通知を送信しました]")
    except Exception as e:
        print(f"   [Slack 通知エラー: {e}]")


# ==================================================
# Web 検索
# ==================================================

def _ddg_full(query: str, max_results: int) -> list:
    """ddgs パッケージ（旧 duckduckgo_search）を使ってフル検索結果を返す。"""
    try:
        from ddgs import DDGS  # 後継パッケージ（pip install ddgs）
    except ImportError:
        from duckduckgo_search import DDGS  # 旧名（非推奨）
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            snippet = (r.get("body") or "")[:WEB_SEARCH_SNIPPET_CHARS]
            results.append(f"[{r.get('title', '')}]\n{snippet}\nSource: {r.get('href', '')}")
    return results


def _ddg_instant(query: str, max_results: int) -> list:
    """DuckDuckGo Instant Answer API (urllib のみ、フォールバック)。"""
    url = ("https://api.duckduckgo.com/?" +
           urllib.parse.urlencode({"q": query, "format": "json",
                                   "no_redirect": "1", "no_html": "1"}))
    req = urllib.request.Request(url, headers={"User-Agent": "fugu-local/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=WEB_SEARCH_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return []
    results = []
    if data.get("Abstract"):
        results.append(f"[{data.get('AbstractTitle', '')}]\n{data['Abstract']}\n"
                       f"Source: {data.get('AbstractURL', '')}")
    for t in data.get("RelatedTopics", []):
        if isinstance(t, dict) and t.get("Text"):
            results.append(t["Text"])
        if len(results) >= max_results:
            break
    return results[:max_results]


def _search_raw(query: str, max_results: int = None) -> list:
    """1 クエリ分の検索結果をリストで返す。失敗時は空リスト（呼び出し側を止めない）。"""
    max_results = max_results or WEB_SEARCH_MAX_RESULTS
    try:
        return _ddg_full(query, max_results)
    except ImportError:
        # Instant Answer API は事実系クエリでほぼ空を返す。無警告だと「検索したのに
        # 古い知識で回答」する事故になる（実測 2026-07-06: 最新GPUで1世代前を回答）。
        print("   [警告: ddgs 未インストールのため Instant Answer フォールバック中。"
              "pip install ddgs でフル検索が有効になります]")
        return _ddg_instant(query, max_results)
    except Exception as e:
        print(f"   [Web検索エラー: {e}]")
        return []


def web_search(query: str, max_results: int = None) -> str:
    """Web 検索 1 回分をフォーマット済み文字列で返す（後方互換用の単発検索）。"""
    results = _search_raw(query, max_results)
    if not results:
        return ""
    return "## Web Search Results (DuckDuckGo)\n" + "\n\n".join(results)


# 十分性判定（Conductor と同じ think=False + スキーマ拘束パターン）
RESEARCH_SYS = (
    "You are a research assistant judging web search results. "
    "Given a user question and accumulated search results, decide whether the results "
    "contain enough SPECIFIC and up-to-date facts (exact product names, architecture "
    "names, versions, dates, numbers) to answer the question accurately without "
    "guessing. Snippets that merely mention the topic are NOT sufficient. "
    "If not sufficient, state what is missing and give up to 3 NEW search queries "
    "targeting the missing facts. Use different keywords than previous queries; "
    "include English queries for technical topics. Return ONLY JSON."
)

RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "missing": {"type": "string"},
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sufficient", "queries"],
}


def research_search(question: str) -> str:
    """十分な事実が集まるまで検索を反復するリサーチループ。
    件数固定の単発検索と違い、Conductor が不足を判定して追加クエリで掘り下げる。"""
    results = []      # フォーマット済み結果（重複排除済み）
    seen = set()      # Source URL による重複排除
    tried = set()     # 実行済みクエリ（同一クエリの再実行を防ぐ）
    queries = [question]

    for rnd in range(1, SEARCH_MAX_ROUNDS + 1):
        for q in queries:
            q = str(q).strip()
            if not q or q.lower() in tried:
                continue
            tried.add(q.lower())
            print(f"   [Web検索 R{rnd}: {q[:60]}]")
            for item in _search_raw(q):
                m = re.search(r"Source: (\S+)", item)
                key = m.group(1) if m else item[:80]
                if key in seen:
                    continue
                seen.add(key)
                results.append(item)

        if rnd == SEARCH_MAX_ROUNDS:
            break

        # 十分性判定（qwen3:4b、~15s）。判定不能時は安全側＝そこで打ち切り（結果は使う）
        joined = "\n\n".join(results)[:SEARCH_CONTEXT_CHARS]
        raw = ask(
            CONDUCTOR,
            [{"role": "system", "content": RESEARCH_SYS},
             {"role": "user", "content": (
                 f"Question:\n{question}\n\n"
                 f"Previous queries: {sorted(tried)}\n\n"
                 f"Search results so far:\n{joined or '(no results)'}\n\n"
                 "Return ONLY the JSON judgement.")}],
            CONDUCTOR_TEMP,
            think=False, fmt=RESEARCH_SCHEMA,
            num_predict=NUM_PREDICT_JUDGE, label="research",
        )
        j = extract_json(raw)
        if not isinstance(j, dict) or j.get("sufficient"):
            break
        missing = str(j.get("missing", ""))[:120]
        queries = [str(x) for x in (j.get("queries") or []) if str(x).strip()][:3]
        if not queries:
            break
        print(f"   [リサーチ継続 R{rnd + 1}: 不足={missing or '(詳細なし)'}]")

    if not results:
        return ""
    # 注入上限で切る（結果の区切り単位）
    body = ""
    for item in results:
        if len(body) + len(item) > SEARCH_CONTEXT_CHARS:
            break
        body += item + "\n\n"
    header = (
        f"## Web Search Results (取得日: {time.strftime('%Y-%m-%d')})\n"
        "重要: 以下はあなたの学習データより新しい一次情報である。学習知識と矛盾する場合は"
        "必ず検索結果を優先すること。検索結果に書かれていない具体的事実"
        "（型番・アーキテクチャ名・日付・数値など）は推測で断定しないこと。\n\n"
    )
    return header + body.rstrip()


# ==================================================
# RAG（ローカル文書検索）
# ==================================================

# ==================================================
# ユニバーサルファイル読み込み
# ==================================================

def _read_pdf(path: Path) -> str:
    """PDF からテキストを抽出。pdfplumber → pypdf → PyPDF2 の順で試行。"""
    # pdfplumber (最高品質)
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n\n".join(pages)
    except ImportError:
        pass
    # pypdf (軽量・新しい)
    try:
        import pypdf
        with open(path, "rb") as f:
            r = pypdf.PdfReader(f)
            return "\n\n".join(p.extract_text() or "" for p in r.pages)
    except ImportError:
        pass
    # PyPDF2 (旧名称)
    try:
        import PyPDF2
        with open(path, "rb") as f:
            r = PyPDF2.PdfReader(f)
            return "\n\n".join(p.extract_text() or "" for p in r.pages)
    except ImportError:
        pass
    return f"[PDF: {path.name} — テキスト抽出には pdfplumber or pypdf が必要: pip install pdfplumber]"


def _read_docx(path: Path) -> str:
    """Word (.docx) からテキストを抽出。"""
    try:
        import docx
        doc = docx.Document(str(path))
        parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)
        for tbl in doc.tables:
            for row in tbl.rows:
                parts.append("\t".join(c.text for c in row.cells))
        return "\n".join(parts)
    except ImportError:
        pass
    return f"[DOCX: {path.name} — python-docx が必要: pip install python-docx]"


def _read_excel(path: Path) -> str:
    """Excel (.xlsx/.xls) を CSV ライクなテキストに変換。"""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        parts = []
        for ws in wb.worksheets:
            parts.append(f"[Sheet: {ws.title}]")
            for row in ws.iter_rows(values_only=True):
                row_str = "\t".join("" if v is None else str(v) for v in row)
                if row_str.strip():
                    parts.append(row_str)
        return "\n".join(parts)
    except ImportError:
        pass
    try:
        import pandas as pd
        xl = pd.ExcelFile(str(path))
        parts = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            parts.append(f"[Sheet: {sheet}]\n{df.to_csv(index=False)}")
        return "\n\n".join(parts)
    except ImportError:
        pass
    return f"[Excel: {path.name} — openpyxl or pandas が必要: pip install openpyxl]"


def _read_pptx(path: Path) -> str:
    """PowerPoint (.pptx) からテキストを抽出。"""
    try:
        from pptx import Presentation
        prs = Presentation(str(path))
        parts = []
        for i, slide in enumerate(prs.slides, 1):
            texts = [sh.text for sh in slide.shapes if hasattr(sh, "text") and sh.text.strip()]
            if texts:
                parts.append(f"[Slide {i}]\n" + "\n".join(texts))
        return "\n\n".join(parts)
    except ImportError:
        pass
    return f"[PPTX: {path.name} — python-pptx が必要: pip install python-pptx]"


def _read_html(path: Path) -> str:
    """HTML からタグを除去してテキストを返す（stdlib html.parser 使用）。"""
    from html.parser import HTMLParser

    class _Strip(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
            self._skip = False
        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip = True
        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self._skip = False
        def handle_data(self, data):
            if not self._skip and data.strip():
                self.parts.append(data.strip())

    raw = path.read_text(encoding="utf-8", errors="replace")
    p = _Strip()
    p.feed(raw)
    return "\n".join(p.parts)


def _read_ipynb(path: Path) -> str:
    """Jupyter Notebook からコードセルとマークダウンセルを抽出。"""
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        parts = []
        for cell in nb.get("cells", []):
            ct = cell.get("cell_type", "")
            src = "".join(cell.get("source", []))
            if not src.strip():
                continue
            if ct == "code":
                parts.append(f"```python\n{src}\n```")
            elif ct == "markdown":
                parts.append(src)
        return "\n\n".join(parts)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")


# テキストとして直接読めない拡張子
_BINARY_SKIP = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib",
    ".bin", ".dat", ".pkl", ".pt", ".pth", ".onnx",
}


def read_file_text(path: Path) -> str:
    """あらゆるファイルからテキストを抽出する。
    対応形式: テキスト・コード類 / PDF / Word / Excel / PowerPoint / HTML / Notebook。
    バイナリ（画像・動画・アーカイブ等）はスキップして空文字を返す。"""
    suffix = path.suffix.lower()
    if suffix in _BINARY_SKIP:
        return ""
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".docx", ".doc"}:
        return _read_docx(path)
    if suffix in {".xlsx", ".xls"}:
        return _read_excel(path)
    if suffix in {".pptx", ".ppt"}:
        return _read_pptx(path)
    if suffix in {".html", ".htm"}:
        return _read_html(path)
    if suffix == ".ipynb":
        return _read_ipynb(path)
    # その他: テキストとして読む（コード・設定ファイル・Markdown など）
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _load_rag_chunks(dirs: list) -> list:
    """指定ディレクトリ群からファイルを読み込み、
    (filepath, chunk_text) のリストを返す。"""
    chunks = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"   [RAG] ディレクトリが見つかりません: {d}")
            continue
        for fp in sorted(p.rglob("*")):
            if not fp.is_file():
                continue
            if fp.suffix.lower() in _BINARY_SKIP:
                continue
            text = read_file_text(fp)
            if not text or text.startswith("["):  # ライブラリ未インストール通知はスキップ
                continue
            # チャンク分割（オーバーラップ付き）
            start = 0
            while start < len(text):
                end = start + RAG_CHUNK_CHARS
                chunks.append((str(fp), text[start:end]))
                start += RAG_CHUNK_CHARS - RAG_CHUNK_OVERLAP
    return chunks


_RAG_CHUNKS: list = []   # キャッシュ（初回のみ読み込み）
_RAG_DIRS_LOADED: list = []


def _get_rag_chunks(dirs: list) -> list:
    global _RAG_CHUNKS, _RAG_DIRS_LOADED
    if dirs != _RAG_DIRS_LOADED:
        _RAG_CHUNKS = _load_rag_chunks(dirs)
        _RAG_DIRS_LOADED = list(dirs)
        print(f"   [RAG] {len(_RAG_CHUNKS)} チャンク読み込み完了（{len(dirs)} ディレクトリ）")
    return _RAG_CHUNKS


def _tokenize(text: str) -> set:
    """英字・数字・日本語を混在テキストから別々に抽出してトークンセットを返す。
    例: 'PINNについて' → {'pinn', 'について'} (単純 \\w+ では 'pinnについて' 1トークンになる)。"""
    lower = text.lower()
    # ASCII: 英字・数字・アンダースコア
    tokens = set(re.findall(r'[a-z0-9_]+', lower))
    # 非ASCII連続列（日本語・CJK など）
    tokens |= set(re.findall(r'[^\x00-\x7f\s]+', text))
    return tokens - {''}


def _score_chunk(query_tokens: set, chunk: str) -> float:
    """クエリトークンとチャンクのキーワード重複スコア（TF-IDF 簡易版）。"""
    chunk_tokens = _tokenize(chunk)
    if not chunk_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens)
    return overlap / (len(chunk_tokens) ** 0.5 + 1) * 100


def rag_search(question: str, dirs: list = None, top_k: int = None) -> str:
    """ローカル文書をキーワード検索して上位チャンクをフォーマット済み文字列で返す。
    dirs が空（RAG_DIRS も空）なら空文字を返す。"""
    dirs = dirs or RAG_DIRS
    if not dirs:
        return ""
    top_k = top_k or RAG_TOP_K
    chunks = _get_rag_chunks(dirs)
    if not chunks:
        return ""
    query_tokens = _tokenize(question)
    scored = [(path, chunk, _score_chunk(query_tokens, chunk))
              for path, chunk in chunks]
    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:top_k]
    if not top or top[0][2] == 0:
        return ""
    parts = []
    for path, chunk, score in top:
        parts.append(f"[Source: {Path(path).name}]\n{chunk.strip()}")
    return "## Relevant Document Context (RAG)\n\n" + "\n\n---\n\n".join(parts)


def build_context(question: str, use_search: bool = False,
                  rag_dirs: list = None) -> str:
    """Web検索 + RAG の結果を組み合わせてコンテキスト文字列を返す。
    空の場合は空文字を返す（質問がそのまま使われる）。"""
    parts = []
    if use_search:
        print("   [Web検索中...]")
        s = research_search(question)
        if s:
            n = s.count("Source:")
            print(f"   [Web検索: 計 {n} 件収集 ({len(s)} 文字)]")
            parts.append(s)
        else:
            print("   [警告: Web検索の結果が 0 件でした。回答はモデルの学習知識のみに"
                  "基づきます（最新情報は反映されません）]")
    rag_result = rag_search(question, dirs=rag_dirs or RAG_DIRS)
    if rag_result:
        parts.append(rag_result)
    return "\n\n".join(parts)


def _with_context(question: str, context: str) -> str:
    """コンテキストがあれば質問に前置する。"""
    if not context:
        return question
    return f"{context}\n\n---\n\n{question}"


def _trim_history(history):
    """_HISTORY が MAX_HISTORY_CHARS を超えたら古い (user, assistant) ペアを先頭から削除する。"""
    while (sum(len(m["content"]) for m in history) > MAX_HISTORY_CHARS
           and len(history) >= 2):
        history.pop(0)
        if history and history[0]["role"] == "assistant":
            history.pop(0)


def server_up():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_server():
    if server_up():
        return True
    if shutil.which("ollama") is None:
        print("⚠ ollama が見つかりません。https://ollama.com からインストールしてください。")
        return False
    print("[setup] Ollama サーバーが見つからないので起動を試みます…")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print("[setup] 自動起動に失敗:", e)
    for _ in range(30):
        if server_up():
            print("[setup] 起動を確認しました。")
            return True
        time.sleep(0.5)
    print("⚠ Ollama に接続できません。別ターミナルで `ollama serve` を起動してください。")
    return False


def installed_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def is_installed(model, inst):
    # 呼び出しは厳密タグで行うので、原則は厳密一致で判定（旧 startswith の誤検知を回避）。
    # タグ無し指定のときだけ :latest を許容する。
    cands = {model}
    if ":" not in model:
        cands.add(model + ":latest")
    return any(n in cands for n in inst)


def pull(model):
    print(f"\n[setup] 取得開始: {model}")
    print("※初回のみダウンロードが発生します。")
    try:
        rc = subprocess.run(["ollama", "pull", model]).returncode
    except FileNotFoundError:
        print("❌ ollama コマンドが見つかりません。")
        return False
    if rc == 0:
        print("[setup] 完了:", model)
        return True
    print(f"❌ 取得失敗: {model} (code {rc}) — スキップします。")
    return False


def resolve_models():
    inst = installed_models()
    pool = []
    for m in DESIRED_PROPOSERS:
        if is_installed(m, inst):
            print("[setup] OK (proposer)", m)
            pool.append(m)
        elif pull(m):
            pool.append(m)

    # アグリゲーター
    agg = DESIRED_AGGREGATOR
    if not (is_installed(agg, installed_models()) or agg in pool):
        if not pull(agg):
            print(f"⚠ アグリゲーター {agg} の取得に失敗 → プール先頭で代用します。")
            agg = pool[0] if pool else None

    # コンダクター（プロポーザー兼務なら追加ロード不要）
    cond = DESIRED_CONDUCTOR
    if not (is_installed(cond, installed_models()) or cond in pool or cond == agg):
        if not pull(cond):
            print(f"⚠ コンダクター {cond} の取得に失敗 → プール先頭で代用します。")
            cond = pool[0] if pool else agg
    if cond is None:
        cond = pool[0] if pool else agg

    # 全滅時の保険
    if not pool:
        print("[setup] 利用可能なモデルが無いため保険を取得します。")
        if pull(FALLBACK_MODEL):
            pool = [FALLBACK_MODEL]
            agg = agg or FALLBACK_MODEL
            cond = cond or FALLBACK_MODEL

    return pool, agg, cond

# ==================================================
# 推論ヘルパ
# ==================================================


def ask(model, messages, temperature, think=None, fmt=None, label=None, num_predict=None,
        num_ctx=None):
    """Ollama native /api/chat を叩く。num_ctx を必ず options で渡して context を安全域に固定する
    （/v1 互換エンドポイントは num_ctx を無視するため使わない）。失敗時は __ERROR__: を返す。
    一過性の失敗(HTTP 500 等)には 1 回だけ再試行する。
    num_predict: 生成トークン上限（None=無制限）。思考モデルの暴走保険として役割別に渡す。
    num_ctx: コンテキスト長の明示指定。None なら MODEL_CONFIG > MODEL_NUM_CTX の順で解決。
      「必ず明示 pin する」不変条件は維持（未指定だと Ollama がモデル最大を確保して 8GB VRAM で
      クラッシュするため）。

    think: None=モデル既定(MODEL_CONFIG があればそれを適用) / False=無効 / True=有効 /
      "low"|"medium"|"high"=gpt-oss 系の思考量段階指定（0.31.2 実測で有効）。
      注意: native /api/chat の "think" パラメータは効くが、プロンプトに書く "/no_think" は
      このルートでは無視される（実測）。gemma4(e2b/e2b-it-qat) は think=true/false とも
      正常に受け付ける（2026-07-02 実測。思考は message.thinking に分離される）。
      非thinkingモデル phi4-mini も False を渡して無害（実測）。
    fmt: Ollama の "format"。"json" か JSON スキーマ(dict)を渡すと構造化出力を強制できる。
      Conductor/Critic では think=False + スキーマの併用が要点:
      think=False 単独だと qwen3 は思考を content に地の文で垂れ流して JSON が壊れるが、
      スキーマを与えると enum 値まで含めて妥当な JSON に拘束され、かつ高速（実測 ~14s）。"""
    if think is None:
        think = model_cfg(model, "think")   # 呼び出し側が未指定ならモデル別設定を適用
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature,
                    "num_ctx": num_ctx or model_cfg(model, "num_ctx", MODEL_NUM_CTX)},
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict
    if think is not None:
        payload["think"] = think
    if fmt is not None:
        payload["format"] = fmt
    if MODEL_KEEP_ALIVE is not None:
        payload["keep_alive"] = MODEL_KEEP_ALIVE  # 既定 None では渡さない
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    out = "__ERROR__: unreachable"

    def _do_call(request):
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as r:
            body = json.loads(r.read().decode("utf-8"))
        msg = body.get("message") or {}
        # think 分離型モデルは thinking が別フィールドに来る場合があるので content のみ採用。
        result = (msg.get("content") or "").strip()
        # 思考が num_predict を食い尽くして本文ゼロで打ち切られた場合(実測: 空返答の
        # 根本原因)は、沈黙の空文字ではなく明示的なエラーにして上位のフォールバックを
        # 確実に発動させる。本文が一部でも出ていればそのまま使う。
        if not result and body.get("done_reason") == "length":
            result = ("__ERROR__: truncated by num_predict during thinking "
                      "(no content was generated)")
        return result

    for attempt in (1, 2):
        try:
            out = _do_call(req)
            break
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            # think:true を送ったが非対応モデル（例: qwen3-coder, phi4）→ think なしで即リトライ。
            # これは設定ミスの安全網。通常は PROPOSER_THINK=None で発生しないはず。
            if e.code == 400 and "does not support thinking" in err_body and "think" in payload:
                payload.pop("think")
                req = urllib.request.Request(
                    f"{OLLAMA_URL}/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                # 重要(バグ修正): このブランチはかつて continue で外側の for に戻り、
                # 「次のイテレーション」が実際に送信されることに依存していた。しかし
                # attempt==1 が一過性の 500（ロード直後）で、attempt==2 で初めて
                # 「thinking非対応」400 が出た場合、continue しても for (1, 2) は
                # 既に尽きており、組み直したリクエストは一度も送信されずに
                # ループを抜けて __ERROR__: think_stripped_retry がそのまま最終値に
                # なってしまっていた（SC投票/提案が黙って1票失われる）。
                # 修正: think を pop 済みなのでこの分岐は高々1回しか到達し得ない
                # （無限ループの可能性なし）。よってここでその場で確定的に1回だけ
                # 追加送信し、一過性リトライの残り予算（sleep(2)して attempt を
                # 消費する経路）は一切消費しない。
                try:
                    out = _do_call(req)
                except urllib.error.HTTPError as e2:
                    err_body2 = e2.read().decode("utf-8", errors="replace")
                    out = f"__ERROR__: {e2} {err_body2}"
                except Exception as e2:
                    out = f"__ERROR__: {e2}"
                break  # think は pop済みでこの分岐は再発し得ないため、ここで確定終了
            out = f"__ERROR__: {e} {err_body}"
            if attempt == 1:
                time.sleep(2)
        except Exception as e:
            out = f"__ERROR__: {e}"
            if attempt == 1:
                time.sleep(2)  # 一過性の失敗(ロード直後の500等)向け。2回目も失敗なら諦める
    if SHOW_TIMING:
        _TIMINGS.append((label or "?", model, round(time.time() - t0, 1)))
    return out


# deepseek-r1 / qwen3 などの思考ログを除去。
# 注: Gemma 4 の思考は 2026-07-02 実測で /api/chat が message.thinking に分離して返す
#     （content には混入しない）ため、Gemma 用の除去パターン追加は不要。
#     gemma4:e2b / e2b-it-qat とも think=true/false パラメータが正常に効くことも確認済み。
_THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
]
# 2026-07-22: num_predict 打ち切り（gotcha #2）で思考モデルの出力が
# '<think>途中の推論...' のまま閉じタグが来ずに切れることがある。上の
# _THINK_PATTERNS は非貪欲マッチのため閉じタグが無いと一切マッチせず、
# 破棄されるべき生の思考過程がそのまま「回答」として extract_final_answer
# まで漏れ、抽出失敗 → nums[-1] フォールバックで思考中の中間値が SC 投票の
# 1票として数えられてしまう(extract_boxed の #2214 と同根の失敗モード)。
# 対策として、バランス除去の後に「閉じタグの無い開始タグ」を検出したら、
# その開始タグ以降を末尾まで丸ごと切り捨てる（開始タグより前のテキストは
# 保持）。無投票の方が誤投票より安全という方針（精度優先）に従う。
_UNTERMINATED_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)


def strip_think(text):
    if not text:
        return text
    for pat in _THINK_PATTERNS:
        text = pat.sub("", text)
    m = _UNTERMINATED_THINK_OPEN.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def extract_json(text):
    """モデル出力から最初の JSON オブジェクトを頑健に抽出（失敗時 None）。"""
    if not text:
        return None
    text = strip_think(text)
    # 1) そのまま
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) ```json ... ``` フェンス内
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3) 最初のバランスの取れた { ... } を波括弧の深さで走査して探す
    # 2026-07-22: 旧実装は re.search(r"\{.*\}", ..., re.DOTALL) という貪欲マッチで、
    # 最初の '{' から最後の '}' までを一括で span にしていた。本文中に地の文の
    # 集合記法 "{1,2,3}" や末尾の "{x}"、あるいは2つ目のJSONオブジェクトなど
    # 「余分な波括弧」が存在すると、その span 全体は JSON として不正な形になり
    # json.loads が例外を投げて None を返していた（本来 docstring が約束する
    # 「最初の JSON オブジェクト」を回収可能なのに握りつぶす）。
    # これは conduct() のルーティングプランと research_search() の
    # RESEARCH_SCHEMA 充足判定の両方を壊し、None が来た側で default_plan() への
    # 劣化や、リサーチの誤った早期終了を引き起こしていた。
    # 修正: 文字列中の状態（ダブルクォート内かどうか・直前のバックスラッシュに
    # よるエスケープ）を追跡しながら、'{' を見つけるたびに深さカウントで対応する
    # '}' を探し、最初に json.loads が成功した部分文字列を返す。
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "{":
            depth = 0
            in_string = False
            escape = False
            j = i
            while j < n:
                ch = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = text[i:j + 1]
                            try:
                                return json.loads(candidate)
                            except Exception:
                                break  # この候補は失敗、次の '{' から再探索
                j += 1
            # depth が閉じ切らなかった（切り詰められた/不正）場合もここに来る
        i += 1
    return None

# ==================================================
# コード実行検証（決定的 Critic）
# ==================================================


def extract_code(text):
    """回答から最初の ```python コードブロックを抽出（無ければ None）。
    言語タグ無しフェンスも Python とみなして拾う（proposer には python タグを指示済み）。
    2026-07-22: 旧実装は re.search(r"```(?:python|py)?[ \t]*\n(.*?)```") を使っており、
    先行する非python フェンス（```json/```bash/```text/```output 等）の開始タグに
    マッチできず、re.search が前方走査してそのブロックの「閉じフェンス」を開始フェンス
    と誤認し、2つのブロックの間にあるプロース（地の文）や本文を「コード」として誤抽出
    していた。これにより code_check が非コードを実行して見せかけの実行失敗を報告し、
    無駄な修復ラウンド(MAX_ROUNDS_CODE)を消費したり、PoT の投票が壊れたりしていた。
    修正: 全てのフェンスブロックを走査し、言語タグが python/py/python3 または
    タグ無し(bare)の「最初のブロック」の本文をそのまま(strip無し)返す。それ以外の
    タグ(json/bash/sh/text/output/js 等)のブロックは読み飛ばす。該当ブロックが
    無ければ None。"""
    if not text:
        return None
    for m in re.finditer(r"```([^\n`]*)\n(.*?)```", text, re.DOTALL):
        lang = m.group(1).strip().lower()
        if lang in ("", "python", "py", "python3"):
            return m.group(2)
    return None


def run_python(code, timeout=None, stdout_only=False):
    """コードを一時ファイル経由で subprocess 実行する。(ok: bool, output: str) を返す。
    ok は exit code 0。既定(stdout_only=False)では stdout+stderr 結合の末尾を返す
    （traceback を修正ヒントに使うため）。stdout_only=True かつ成功時(returncode==0)
    は stdout のみを返す（sympy/numpy の DeprecationWarning 等が stderr に出て
    末尾行が汚染され、PoT の投票が壊れるのを防ぐため）。ただし失敗時(returncode!=0)
    は stdout_only の値によらず常に stdout+stderr の結合を返す — code-repair loop が
    traceback を見えるようにするため。"""
    timeout = timeout or CODE_EXEC_TIMEOUT
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(code)
        r = subprocess.run(
            [sys.executable, "-X", "utf8", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        if stdout_only and r.returncode == 0:
            out = (r.stdout or "").strip()
        else:
            out = ((r.stdout or "") + (r.stderr or "")).strip()
        return r.returncode == 0, out[-2000:]
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: code did not finish within {timeout}s (infinite loop or input() wait?)"
    except Exception as e:
        return False, f"runner error: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def code_check(answer):
    """回答中の Python コードを実行して検証する。問題があればエラー要約(str)、
    問題なし・コードなし・機能OFF なら None を返す。"""
    if not CODE_EXECUTION:
        return None
    code = extract_code(answer)
    if not code:
        return None
    ok, out = run_python(code)
    if ok:
        return None
    return f"code execution FAILED:\n{out[-800:]}"

# ==================================================
# 画像生成（ローカル SD / ComfyUI へバイパス）
# ==================================================


def _sd_prompt_from_request(user_request):
    """ユーザーの画像要求を SD 用プロンプトへ変換する。IMAGE_TRANSLATE_PROMPT なら
    qwen3:4b で英語の高品質プロンプト（+ネガティブ）へ翻訳する（SD系は英語で精度が出る）。
    戻り値: (prompt, negative)。"""
    if not IMAGE_TRANSLATE_PROMPT:
        return user_request, ""
    sys = (
        "You convert a user's image request into a Stable Diffusion prompt. "
        'Output ONLY JSON: {"prompt": "...", "negative": "..."}. '
        "prompt: a concise comma-separated ENGLISH prompt with quality tags "
        "(e.g. 'masterpiece, best quality, highly detailed'). "
        "negative: common negatives (e.g. 'lowres, bad anatomy, blurry, watermark'). "
        "No prose, no thinking."
    )
    raw = ask(
        CONDUCTOR,
        [{"role": "system", "content": sys},
         {"role": "user", "content": user_request}],
        CONDUCTOR_TEMP, think=False,
        fmt={"type": "object",
             "properties": {"prompt": {"type": "string"},
                            "negative": {"type": "string"}},
             "required": ["prompt"]},
        num_predict=512, label="img-prompt",
    )
    j = extract_json(raw) or {}
    prompt = str(j.get("prompt") or user_request).strip()
    negative = str(j.get("negative") or "").strip()
    return prompt, negative


def _http_post_json(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _backend_up(url):
    """GET で 200 が返れば True（バックエンドの疎通確認）。"""
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def generate_image_a1111(prompt, negative=""):
    """AUTOMATIC1111 stable-diffusion-webui の txt2img API で生成し保存パスを返す。"""
    import base64
    payload = {"prompt": prompt, "negative_prompt": negative,
               "steps": IMAGE_STEPS, "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT}
    data = _http_post_json(f"{A1111_URL}/sdapi/v1/txt2img", payload, IMAGE_TIMEOUT)
    images = data.get("images") or []
    if not images:
        return None
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = IMAGE_OUT_DIR / f"fugu_{time.strftime('%Y%m%d_%H%M%S')}.png"
    # data URI プレフィックスが付く場合があるのでカンマ以降を取る
    out.write_bytes(base64.b64decode(images[0].split(",", 1)[-1]))
    return out


def generate_image_comfyui(prompt, negative=""):
    """ComfyUI に最小 txt2img ワークフローを投げて生成し保存パスを返す。
    COMFYUI_CKPT が空ならサーバの利用可能チェックポイント先頭を自動採用する。"""
    import uuid
    ckpt = COMFYUI_CKPT
    if not ckpt:
        try:
            with urllib.request.urlopen(
                    f"{COMFYUI_URL}/object_info/CheckpointLoaderSimple", timeout=5) as r:
                info = json.loads(r.read().decode("utf-8"))
            ckpt = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0][0]
        except Exception as e:
            print(f"   [ComfyUI: チェックポイント取得に失敗: {e}]")
            return None
    client_id = uuid.uuid4().hex
    wf = {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": int(time.time()) % (2 ** 32), "steps": IMAGE_STEPS, "cfg": 7,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {
            "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "fugu", "images": ["8", 0]}},
    }
    try:
        resp = _http_post_json(f"{COMFYUI_URL}/prompt",
                               {"prompt": wf, "client_id": client_id}, 30)
        pid = resp["prompt_id"]
    except Exception as e:
        print(f"   [ComfyUI: プロンプト投入に失敗: {e}]")
        return None
    # 完了までポーリング
    deadline = time.time() + IMAGE_TIMEOUT
    hist = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{COMFYUI_URL}/history/{pid}", timeout=10) as r:
                h = json.loads(r.read().decode("utf-8"))
            if pid in h:
                hist = h[pid]
                break
        except Exception:
            pass
        time.sleep(2)
    if not hist:
        print("   [ComfyUI: 生成がタイムアウトしました]")
        return None
    for node in hist.get("outputs", {}).values():
        for img in node.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"],
                                        "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
            try:
                with urllib.request.urlopen(f"{COMFYUI_URL}/view?{q}", timeout=30) as r:
                    blob = r.read()
            except Exception:
                continue
            IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
            out = IMAGE_OUT_DIR / f"fugu_{time.strftime('%Y%m%d_%H%M%S')}_{img['filename']}"
            out.write_bytes(blob)
            return out
    return None


# --- LLM 群による画像プロンプト起草 ---
IMAGE_PROMPT_SYS = (
    "あなたは Stable Diffusion(SDXL) のプロンプトエンジニアです。"
    "与えられた要望(または回答内容)を、高品質な画像を生成するための英語プロンプトへ変換します。"
    'JSON のみ出力: {"prompt": "...", "negative": "..."}. '
    "prompt: カンマ区切りの英語。主題・画風・構図・光・品質タグ"
    "(masterpiece, best quality, highly detailed 等)を含める。"
    "negative: 典型的なネガティブ(lowres, bad anatomy, blurry, watermark, text 等)。"
    "散文や思考は出力しない。"
)
_IMG_PROMPT_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}, "negative": {"type": "string"}},
    "required": ["prompt"],
}


def moa_image_prompt(base_text, panel=None):
    """LLM 群(proposers)が SDXL プロンプト候補を起草し、qwen3:4b が最良の1つへ統合する。
    候補が得られなければ None（呼び出し側が単独翻訳へフォールバック）。"""
    models = [m for m in (panel or []) if m in PROPOSERS] or PROPOSERS[:IMAGE_PROMPT_PANEL]
    models = models[:IMAGE_PROMPT_PANEL]
    cands = []
    for m in models:
        raw = ask(m, [{"role": "system", "content": IMAGE_PROMPT_SYS},
                      {"role": "user", "content": base_text}],
                  PROPOSER_TEMP, think=False, fmt=_IMG_PROMPT_SCHEMA,
                  num_predict=512, label="img-moa")
        j = extract_json(raw)
        if isinstance(j, dict) and j.get("prompt"):
            cands.append((m, j))
    if not cands:
        return None
    if len(cands) == 1:
        j = cands[0][1]
        return str(j.get("prompt")).strip(), str(j.get("negative") or "").strip()
    listing = "\n".join(
        f"[{MODEL_TO_PERSONA.get(m, m)}] prompt={j.get('prompt')} | negative={j.get('negative', '')}"
        for m, j in cands)
    merge_sys = (IMAGE_PROMPT_SYS + " 以下は複数の専門家が起草した候補です。"
                 "最も的確で高品質な1つの SDXL プロンプトへ統合しなさい。")
    raw = ask(CONDUCTOR, [{"role": "system", "content": merge_sys},
                          {"role": "user", "content": base_text + "\n\n候補:\n" + listing}],
              CONDUCTOR_TEMP, think=False, fmt=_IMG_PROMPT_SCHEMA,
              num_predict=512, label="img-merge")
    j = extract_json(raw)
    if isinstance(j, dict) and j.get("prompt"):
        return str(j.get("prompt")).strip(), str(j.get("negative") or "").strip()
    j = cands[0][1]
    return str(j.get("prompt")).strip(), str(j.get("negative") or "").strip()


def author_image_prompt(base_text, panel=None):
    """画像プロンプトを決める。IMAGE_PROMPT_MOA なら LLM 群で起草→統合、
    失敗時や無効時は qwen3:4b の単独翻訳へフォールバック。戻り値 (prompt, negative)。"""
    if IMAGE_PROMPT_MOA:
        p = moa_image_prompt(base_text, panel)
        if p:
            return p
    return _sd_prompt_from_request(base_text)


# --- バックエンド検出と生成コア ---
def _detect_backend():
    """使用するバックエンドを返す（"a1111" / "comfyui" / None）。"""
    if IMAGE_BACKEND in ("a1111", "comfyui"):
        return IMAGE_BACKEND
    if IMAGE_BACKEND == "auto":
        if _backend_up(f"{A1111_URL}/sdapi/v1/sd-models"):
            return "a1111"
        if _backend_up(f"{COMFYUI_URL}/system_stats"):
            return "comfyui"
    return None


def generate_image(prompt, negative=""):
    """プロンプトから実際に画像を生成し保存 Path を返す（失敗は None）。
    PowerPoint ビルダーからも直接使う低レベル API。"""
    backend = _detect_backend()
    if backend is None:
        return None
    try:
        if backend == "a1111":
            return generate_image_a1111(prompt, negative)
        return generate_image_comfyui(prompt, negative)
    except Exception as e:
        print(f"   [画像生成エラー ({backend}): {e}]")
        return None


def handle_image_generation(user_request, *, panel=None, prompt=None, negative=None):
    """画像を生成し、人間可読な結果テキスト（保存パス）または __ERROR__ を返す。
    prompt/negative 未指定なら LLM 群で起草する。panel はプロンプト起草に使う proposer 群。"""
    if IMAGE_BACKEND == "off":
        return "__ERROR__: 画像生成は無効化されています (IMAGE_BACKEND='off')。"
    if _detect_backend() is None:
        return ("__ERROR__: 画像生成バックエンドが見つかりません。\n"
                f"  AUTOMATIC1111 を {A1111_URL} で、または ComfyUI を {COMFYUI_URL} で"
                "起動してください（IMAGE_BACKEND で明示指定も可）。")
    if prompt is None:
        prompt, negative = author_image_prompt(user_request, panel=panel)
    print(f"   [画像生成プロンプト] {prompt[:120]}")
    out = generate_image(prompt, negative or "")
    if not out:
        return "__ERROR__: 画像生成に失敗しました（バックエンドが画像を返しませんでした）。"
    msg = (f"画像を生成しました。\n"
           f"- 保存先: {out}\n"
           f"- prompt: {prompt}\n")
    if negative:
        msg += f"- negative: {negative}\n"
    return msg

# ==================================================
# プロンプト
# ==================================================

# ChatGPT / Gemini のような読みやすい提示スタイル（最終回答に付与する）
PRESENTATION_STYLE = (
    "\n\n【回答の体裁（ChatGPT / Gemini 風）】\n"
    "- 最初に結論・要点を1〜2文で述べ、その後に詳細を続ける。\n"
    "- Markdown を適切に使う: 見出し(##)、箇条書き(-)、番号付き手順、重要語の**強調**、"
    "コードは``` で囲む、比較や一覧は表を使う。\n"
    "- 丁寧で分かりやすい対話的なトーン。冗長な前置きや『AIとして』等の自己言及は避ける。\n"
    "- 長い回答は見出しで構造化し、必要なら最後に一言まとめを付ける。質問と同じ言語で書く。"
)

PROPOSER_SYS = (
    "You are one expert in a panel. Answer the user's question as accurately and "
    "concretely as you can. Think step-by-step internally to avoid math or logical errors. "
    "Briefly show key reasoning if it helps. Respond in the same language as the question.\n"
    "If the question asks for code: put ONE complete, runnable program in a single "
    "```python code block, and end that same block with a few assert-based self-tests "
    "that verify it (the code will be executed to check it works). No input() calls."
)

AGGREGATOR_SYS = (
    "You are the aggregator of a Mixture-of-Agents system. Several independent "
    "models each answered the SAME question; their answers are given below, "
    "anonymized as Answer A, B, C, ...\n\n"
    "Do NOT merely summarize or average them. Produce ONE answer that is more "
    "accurate and complete than any single one, by reasoning critically:\n"
    "1. Where they AGREE: a signal of likely correctness — but verify it is not a shared mistake.\n"
    "2. Where they DISAGREE or CONTRADICT: treat it as a red flag. Decide which side is actually correct using your own reasoning, not majority vote.\n"
    "3. Detect and DISCARD errors, hallucinations, unsupported claims, even if several answers share them.\n"
    "4. PRESERVE the strongest reasoning and the most useful specifics; combine them.\n"
    "5. If all answers are weak, override them with a better one.\n"
    "6. If an answer carries an [Execution check: ...] tag, that is GROUND TRUTH from "
    "actually running its code — prefer code that PASSED; never base the final answer on "
    "code that FAILED without fixing the reported error.\n\n"
    "Output only the final polished answer for the user. Do not mention the other answers or this process.\n"
    "CRITICAL LANGUAGE RULE: Write the ENTIRE final answer in exactly the same language as the "
    "question. Do NOT mix in words or characters from any other language. For a Japanese question, "
    "use natural Japanese only — never Chinese-only words/characters (e.g. 个, 至少, 确, 说) and no "
    "stray English words; use the Japanese equivalents (個, 少なくとも, など)."
    + PRESENTATION_STYLE
)

CONDUCTOR_SYS = (
    "あなたは Fugu オーケストレーションシステムの最高司令塔(Conductor)です。"
    "ユーザーの入力(および RAG で読み込まれた Office ファイルや Web 検索結果)を分析し、"
    "最適な実行プランを厳密な JSON で出力します。あなた自身は質問に答えません。\n\n"
    "【専門家チームの布陣】\n"
    "- Proposer A (ChatGPT/GPTの存在): バランス・一般的な対話・文章の骨組み担当。\n"
    "- Proposer B (Claudeの存在): 高度なプログラミング・厳密な論理チェック・コード自己修復担当。\n"
    "- Proposer C (Geminiの存在): RAG(Officeファイル)のコンテキスト分析・大量ドキュメント・"
    "Web 検索結果の集約担当。\n"
    "- Proposer D (理数専門家): 数学・物理・PINN・偏微分方程式・アルゴリズム証明担当。\n\n"
    "【特殊ルーティング指示】\n"
    "1. 画像生成の要素がある要求では use_image_generation=true にすること。さらに:\n"
    "   - 『絵を描いて』『イラストを作って』のように画像だけが目的なら image_only=true"
    "(テキスト回答は不要)。\n"
    "   - 『〜を説明して図も作って』のようにテキスト回答も画像も必要なら image_only=false とし、"
    "mode と selected_proposers は通常どおり選ぶ(本文は MoA が作り、その内容から画像を生成する)。\n"
    "   画像が不要な通常の質問は use_image_generation=false, image_only=false。\n"
    "2. 『パワポ』『スライド』『プレゼン』『PowerPoint』『資料を作って』等のスライド作成要求では "
    "make_pptx=true とし、mode='moa' で本文(見出し・箇条書き構成)を作ること。画像はスライド内容に応じて"
    "自動生成されるので use_image_generation は false で良い。\n"
    "3. Office ファイル(.docx/.xlsx/.pdf 等)が添付・指定され、その解説・分析を求めている場合は、"
    "必ず mode='moa' とし、selected_proposers に必ず 'Proposer C' を含めて主軸に据えること。\n"
    "4. 質問が『最新』『今』『現在』『2025年』『2026年』『価格』『相場』『型番』『バージョン』など、"
    "時間とともに変わる事実・時事情報を含む場合は search_required=true にすること。"
    "自分の知識だけで答えられると過信しないこと(学習データは古くなる)。"
    "純粋に不変な知識(数学・定義・確立した歴史的事実)のみ false。\n"
    "5. task_type を分類すること: 最終答が数値・式で一意に定まる問題は 'math'、"
    "選択肢(A/B/C/D等)から選ぶ問題は 'mcq'、コード作成・デバッグは 'code'、"
    "事実知識・解説は 'knowledge'、文章作成は 'writing'、雑談・その他は 'chat'。"
    "証明や『なぜ』の説明は答えが一意の数値でないため 'math' ではなく 'knowledge'。\n\n"
    "【精度優先の原則(最重要・厳守)】\n"
    "- mode='single' を許可するのは、挨拶・単純な事実確認・短い定義・簡単な計算のみ。\n"
    "- コード生成/実装/デバッグ・数学/証明・論理パズル・多段推論・比較/要約/翻訳/説明は、"
    "たとえ簡単そうに見えても必ず mode='moa' とすること。単一モデルが見落とす誤りを"
    "複数 Proposer の相互チェックで潰すのが目的。特にコードと証明は例外なく moa とし 3〜4 体選ぶ。\n"
    "- 迷ったら moa を選ぶ(精度優先。計算コストは二の次)。\n"
    f"- rounds は通常 1。明確に洗練が必要な難問だけ 2 以上(最大 {MAX_ROUNDS})。\n\n"
    "selected_proposers には上記のペルソナ名(\"Proposer A\"〜\"Proposer D\")だけを使うこと。"
    "質問の内容と各 Proposer の得意分野を照合して選ぶこと。\n"
    "思考や散文を一切出さず、以下の JSON オブジェクトのみを出力すること。"
)

CRITIC_SYS = (
    "You are a strict reviewer. Given a question and a candidate answer, judge whether "
    "the answer is correct, complete and well-reasoned. "
    'Output ONLY JSON: {"ok": true or false, "issue": "short reason if not ok"}. '
    "Be conservative: only set ok=false when there is a real, identifiable problem. "
    "No prose, no thinking."
)

# Ollama の format に渡す JSON スキーマ。enum で値域まで拘束し、think=False と併用して
# Conductor/Critic を高速・確実にする（freeform "json" だと mode に "mathematical_proof" 等の
# enum 外の値を捏造するため、スキーマで縛るのが要点）。
CONDUCTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["single", "moa"]},
        "task_type": {"type": "string",
                      "enum": ["math", "code", "mcq", "knowledge", "writing", "chat"]},
        "selected_proposers": {"type": "array", "items": {"type": "string"}},
        "rounds": {"type": "integer", "minimum": 1, "maximum": MAX_ROUNDS},
        "use_image_generation": {"type": "boolean"},
        "image_only": {"type": "boolean"},
        "make_pptx": {"type": "boolean"},
        "search_required": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["mode", "task_type", "selected_proposers", "rounds",
                 "use_image_generation", "search_required", "reason"],
}

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "issue": {"type": "string"},
    },
    "required": ["ok"],
}

# ==================================================
# Conductor（動的プランニング）
# ==================================================


def build_proposer_desc():
    """ペルソナ名付きで、導入済みモデルのみ列挙する。"""
    lines = []
    for label, model in PERSONA_MODELS.items():
        if model in PROPOSERS:
            prof = PROPOSER_PROFILES.get(model, "汎用")
            lines.append(f"- {label} ({model}): {prof}")
    return "\n".join(lines)


def _resolve_proposer(name):
    """ペルソナ名 or モデル名を実モデル名へ解決する（未導入・未知なら None）。
    'Proposer A' / 'proposer a' / 'A' / 実モデル名 の緩い表記を許容。"""
    if name in PERSONA_MODELS:
        m = PERSONA_MODELS[name]
        return m if m in PROPOSERS else None
    if name in PROPOSERS:
        return name
    key = str(name).strip().lower()
    for label, model in PERSONA_MODELS.items():
        if key in (label.lower(), label.lower().replace("proposer ", "")):
            return model if model in PROPOSERS else None
    return None


def _persona_str(model):
    """表示用に 'Proposer X (model)' 形式へ整形する。"""
    p = MODEL_TO_PERSONA.get(model)
    return f"{p} ({model})" if p else str(model)


# コード/証明/論理は単一モデルが誤りやすいので、Conductor が single を選んでも moa へ
# 格上げする決定的ガードレール。小型 Conductor(qwen3:4b) が精度優先ルールを取りこぼす件の
# 安全網（旧スキーマの complexity フィールドが担っていた難易度分類の代替）。
# 精度優先・計算コスト非優先の方針に沿う。画像生成バイパスは対象外。
_HARD_SIGNALS = re.compile(
    r"実装|コード|プログラム|関数|クラス|アルゴリズム|デバッグ|バグ|ソート|計算量|"
    r"証明|背理法|論理的|パズル|嘘つき|"
    r"\bcode\b|\bimplement|\bfunction\b|\balgorithm|\bdebug|\bprove\b|\bproof\b|```",
    re.IGNORECASE)


def _apply_accuracy_guardrails(question, plan):
    """精度優先: コード/証明/論理の質問で single が選ばれたら moa へ格上げする。"""
    if plan.get("image_only"):
        return plan  # 画像のみはテキスト側の格上げ不要
    if plan["mode"] == "single" and _HARD_SIGNALS.search(question or ""):
        plan["mode"] = "moa"
        if len(plan.get("selected_proposers", [])) < 2:
            plan["selected_proposers"] = PROPOSERS[:3]
        plan["reason"] = "[guardrail: code/proof→moa] " + plan.get("reason", "")
    return plan


# 出力形態（画像/PowerPoint）の決定的ガードレール。小型 Conductor(qwen3:4b) は
# image_only / make_pptx の取りこぼしが多いため、明確なキーワードで補正する。
_PPTX_SIGNALS = re.compile(
    r"パワポ|パワーポイント|スライド|プレゼン(テーション)?|power\s*point|\bpptx\b|スライド資料|プレゼン資料",
    re.IGNORECASE)
_IMAGE_SIGNALS = re.compile(
    r"絵を描|イラスト|描いて|画像を?(生成|作|つく)|図[をもの]?(作|生成|描|示)|"
    r"イメージ図|図解|概念図|挿絵|ダイアグラム|"
    r"\bdraw\b|illustration|\bpicture\b|diagram|generate.{0,10}image",
    re.IGNORECASE)
# 画像に加えテキスト解説も要る＝イラスト付き回答（image_only=False）の手掛かり
_TEXT_TASK_SIGNALS = re.compile(
    r"説明|解説|教えて|まとめ|要約|について|比較|分析|解析|理由|方法|とは|手順|"
    r"explain|describe|summar|analy|\bwhy\b|\bhow\b",
    re.IGNORECASE)


def _apply_routing_guardrails(question, plan):
    """出力形態を補正: PowerPoint / 画像のみ / イラスト付き回答 を明確なキーワードで確定する。"""
    q = question or ""
    if _PPTX_SIGNALS.search(q):
        plan["make_pptx"] = True
        plan["image_only"] = False
        plan["mode"] = "moa"
        if len(plan.get("selected_proposers", [])) < 2:
            plan["selected_proposers"] = PROPOSERS[:3]
        return plan
    if _IMAGE_SIGNALS.search(q):
        plan["use_image_generation"] = True
        # テキスト解説も求めている＝イラスト付き / 画像だけ＝image_only
        plan["image_only"] = not bool(_TEXT_TASK_SIGNALS.search(q))
    return plan


# task_type の決定的ガードレール（小型 Conductor の分類ミス対策。2026-07-11 追加）。
# math/mcq は solve_verifiable(自己一貫性投票) へルートされるため、誤って自由記述問題を
# math にすると答えの抽出ができない。確実なキーワードでのみ確定させ、証明・説明系は外す。
_MCQ_SIGNALS = re.compile(
    r"which of the following|次のうち|選びなさい|選べ|正しいものを|適切なものを|"
    r"(?:^|\n)\s*\(?A[).:：]\s*\S.*\n\s*\(?B[).:：]\s*\S",
    re.IGNORECASE)
_MATH_TASK_SIGNALS = re.compile(
    r"\\boxed|求めよ|求めなさい|を計算し|を求め|何通り|余りを|剰余|確率を求?|"
    r"\bhow many\b|\bcompute\b|\bcalculate\b|"
    r"\bfind the (?:number|value|sum|remainder|area|probability|smallest|largest)\b|"
    r"answer with (?:the )?number",
    re.IGNORECASE)
_CODE_TASK_SIGNALS = re.compile(
    r"実装|コード|プログラム|関数を書|クラスを書|デバッグ|"
    r"\bimplement|\bwrite (?:a |the )?(?:function|program|code|class|script)\b|```",
    re.IGNORECASE)
_FREEFORM_SIGNALS = re.compile(r"証明|\bprove\b|\bproof\b|説明して|解説して|なぜ",
                               re.IGNORECASE)


def _apply_tasktype_guardrails(question, plan):
    """task_type を決定的キーワードで補正する。確実なシグナルが無ければ Conductor の分類を尊重。"""
    q = question or ""
    t = str(plan.get("task_type") or "").lower()
    if t not in ("math", "code", "mcq", "knowledge", "writing", "chat"):
        t = ""
    if _MCQ_SIGNALS.search(q):
        t = "mcq"
    elif _CODE_TASK_SIGNALS.search(q):
        t = "code"
    elif _MATH_TASK_SIGNALS.search(q):
        t = "math"
    # 証明・説明は最終答が一意の数値/文字にならないため投票では解けない → MoA 経路へ
    if t == "math" and _FREEFORM_SIGNALS.search(q):
        t = "knowledge"
    plan["task_type"] = t or "chat"
    return plan


def default_plan():
    return {
        "mode": "moa",
        "task_type": "",
        "selected_proposers": PROPOSERS[:3],
        "rounds": 1,
        "use_image_generation": False,
        "image_only": False,
        "make_pptx": False,
        "search_required": False,
        "reason": "fallback (planner unavailable)",
        "_fallback": True,
    }


def validate_plan(p):
    base = default_plan()
    if not isinstance(p, dict):
        return base
    out = dict(base)
    out["_fallback"] = False

    out["use_image_generation"] = bool(p.get("use_image_generation", False))
    out["image_only"] = bool(p.get("image_only", False))
    out["make_pptx"] = bool(p.get("make_pptx", False))
    out["search_required"] = bool(p.get("search_required", False))

    # make_pptx / illustrated(text+image) は本文が要るので image_only を無効化
    if out["make_pptx"]:
        out["image_only"] = False

    mode = str(p.get("mode", "moa")).lower()
    out["mode"] = "single" if mode == "single" else "moa"

    t = str(p.get("task_type", "")).lower()
    out["task_type"] = t if t in ("math", "code", "mcq",
                                  "knowledge", "writing", "chat") else ""

    # selected_proposers（ペルソナ名 or モデル名）を実モデル名へ解決・重複排除
    models = []
    raw_props = p.get("selected_proposers")
    if isinstance(raw_props, list):
        for name in raw_props:
            m = _resolve_proposer(name)
            if m and m not in models:
                models.append(m)

    # use_image_generation は非排他フラグ。テキスト側の mode/proposers は通常どおり解決する。
    if out["image_only"]:
        # 画像のみ: テキスト提案は不要（プロンプト起草用に models は保持してよい）
        out["selected_proposers"] = models[:IMAGE_PROMPT_PANEL]
    elif out["mode"] == "single":
        out["selected_proposers"] = (models[:1] or PROPOSERS[:1])
    else:
        out["selected_proposers"] = (models[:4] or PROPOSERS[:3])

    try:
        r = int(p.get("rounds", 1))
    except Exception:
        r = 1
    out["rounds"] = max(1, min(MAX_ROUNDS, r))

    out["reason"] = str(p.get("reason", ""))[:200]
    return out


def conduct(question, history=None, office_attached=False):
    desc = build_proposer_desc()
    hist_note = ""
    if history:
        recent = history[-4:]  # 直近 2 往復分をテキストとして埋め込む
        lines = [
            ("[User]" if m["role"] == "user" else "[Assistant]")
            + ": " + m["content"][:200] + ("..." if len(m["content"]) > 200 else "")
            for m in recent
        ]
        hist_note = "\n\n直近の会話(参考):\n" + "\n".join(lines) + "\n"
    hint = ""
    if office_attached:
        hint = ("\n[注記] Office 文書(.docx/.xlsx/.pdf 等)が添付されています。"
                "特殊ルーティング指示 #2 を適用し、mode='moa' かつ selected_proposers に "
                "'Proposer C' を含めてください。\n")
    user = (
        f"利用可能な Proposer とその強み:\n{desc}\n"
        f"{hist_note}"
        f"{hint}"
        f"\nユーザーの質問:\n{question}\n\n"
        "JSON プランのみを返すこと。"
    )
    msgs = [{"role": "system", "content": CONDUCTOR_SYS},
            {"role": "user", "content": user}]
    raw = ask(
        CONDUCTOR, msgs, CONDUCTOR_TEMP,
        think=False,   # 思考は無効化（+スキーマ拘束）でプランJSONを高速・確実に得る
        fmt=CONDUCTOR_SCHEMA,
        num_predict=NUM_PREDICT_JUDGE,
        label="conductor",
    )
    plan = extract_json(raw)
    if plan is None:
        # スキーマ強制でも稀に JSON が崩れる(実測 ~1/10 程度の一過性)。固定フォールバック
        # プランより正しいプランの方が良いので、1 回だけ引き直す。
        raw = ask(
            CONDUCTOR, msgs, CONDUCTOR_TEMP,
            think=False, fmt=CONDUCTOR_SCHEMA,
            num_predict=NUM_PREDICT_JUDGE, label="conductor",
        )
        plan = extract_json(raw)
    plan = _apply_routing_guardrails(question, validate_plan(plan))
    plan = _apply_accuracy_guardrails(question, plan)
    return _apply_tasktype_guardrails(question, plan), raw


def _critic_judge(question, answer, think):
    """Critic 1 回分の呼び出し。(ok, issue)。think=True は再検算（思考込みで遅いが正確）。"""
    raw = ask(
        CONDUCTOR,
        [{"role": "system", "content": CRITIC_SYS},
         {"role": "user", "content": (
             f"Question:\n{question}\n\nCandidate answer:\n{answer}\n\n"
             "Return ONLY JSON."
         )}],
        CONDUCTOR_TEMP,
        think=think,
        fmt=CRITIC_SCHEMA,
        num_predict=(NUM_PREDICT_JUDGE_THINK if think else NUM_PREDICT_JUDGE),
        label="critic",
    )
    # 2026-07-22: __ERROR__ センチネル（ask() の通信/モデル失敗）と、空文字や
    # パース不能だが正常な出力とを区別する。後者は gpt-oss:20b の思考予算切れで
    # 本文が空になる既知ケースのため ok=True 既定を維持するが、前者は critic 呼び出し
    # そのものが失敗しているだけで「回答に問題なし」を意味しない。ここで黙って
    # ok=True にすると verify_single() の最終審判（think=True critic）が事実上
    # 無審査で通ってしまい、精度優先の方針に反するため ok=False にしてエスカレーション
    # させる（呼び出し元は MoA パネルへフォールバックできる）。
    if strip_think(raw).startswith("__ERROR__"):
        return False, f"critic call failed: {strip_think(raw)}"[:200]
    p = extract_json(raw) or {}
    return bool(p.get("ok", True)), str(p.get("issue", ""))[:200]


def critique(question, answer):
    """回答の十分性を 2 段階で判定。(ok: bool, issue: str) を返す。
    1段目: think=False + スキーマで高速判定。ok ならそこで確定（高速パス維持）。
    2段目: 1段目が NG のときだけ think=True で再検算して最終判定。
    think=False の Critic は頭の中で再計算ができず、正答 '700' を誤って NG にする
    偽エスカレーション(310秒浪費)が 2026-07-03 のフル評価で実測されたための対策。"""
    ok, _issue = _critic_judge(question, answer, think=False)
    if ok:
        return True, ""
    return _critic_judge(question, answer, think=True)


def second_opinion(question, answer):
    """自己評価バイアス対策の独立チェック。Conductor(qwen3) とは別系統の
    SECOND_OPINION_MODEL に同じ審査をさせる。(ok, issue) を返す。
    phi4-mini は非thinkingモデルなので think パラメータは送らない。"""
    global _SECOND_OPINION_DISABLED
    if SECOND_OPINION_MODEL not in PROPOSERS:
        if not _SECOND_OPINION_DISABLED:
            print(f"   ? second_opinion モデル {SECOND_OPINION_MODEL} が見つかりません "
                  f"→ 自己評価バイアス対策が無効化されます。verify_single() は思考ON再検算を必須にします。")
            _SECOND_OPINION_DISABLED = True
        return True, ""
    raw = ask(
        SECOND_OPINION_MODEL,
        [{"role": "system", "content": CRITIC_SYS},
         {"role": "user", "content": (
             f"Question:\n{question}\n\nCandidate answer:\n{answer}\n\n"
             "Return ONLY JSON."
         )}],
        CONDUCTOR_TEMP,
        fmt=CRITIC_SCHEMA,
        # gpt-oss:20b は MODEL_CONFIG で think:"high" が自動適用されるため、思考が予算を
        # 食い尽くして本文空(=ok既定)になるのを防ぐべく思考込みの上限を使う。
        num_predict=NUM_PREDICT_JUDGE_THINK,
        label="critic2",
    )
    # 2026-07-22: _critic_judge と同様、__ERROR__ センチネル（second opinion モデルの
    # 通信/モデル失敗）は「空文字/非JSONだが正常出力」（gpt-oss:20b の思考予算切れ既定）
    # とは別物として扱う。second_opinion は自己評価バイアス対策の独立チェックであり、
    # ここが黙って ok=True になると verify_single() でバイアス対策が機能しないまま
    # 高速パスが通ってしまう。エラー時は ok=False にして think=True 再検算に回す。
    if strip_think(raw).startswith("__ERROR__"):
        return False, f"critic call failed: {strip_think(raw)}"[:200]
    p = extract_json(raw) or {}
    return bool(p.get("ok", True)), str(p.get("issue", ""))[:200]


def verify_single(question, answer):
    """単体回答の採用可否。(ok, issue) を返す。
    高速チェック 2 系統（qwen3 think=False と、別系統 phi4-mini による独立チェック＝
    自己評価バイアス対策）を先に走らせ、どちらかが疑義を出したときだけ
    qwen3 think=True の再検算を最終審判にする。

    second_opinion が無効化されている場合は、高速チェック 1 系統だけになり、疑義が
    ある場合は即座に think=True 再検算を通す（手厚い保護）。

    実測(2026-07-03): 高速チェックはどちらも正答 '700' を誤って NG にすることがあるが、
    思考ONの再検算は正しく ok と判定した。逆に誤答は再検算が明確な理由付きで NG にする。
    コード回答はまず実行(決定的)で検証し、失敗なら LLM 審査を待たず即 NG。"""
    code_issue = code_check(answer)
    if code_issue:
        return False, code_issue
    ok1, issue1 = _critic_judge(question, answer, think=False)
    ok2, issue2 = second_opinion(question, answer)

    if _SECOND_OPINION_DISABLED:
        doubt = issue1
        if ok1:
            return True, ""
    else:
        if ok1 and ok2:
            return True, ""
        doubt = issue1 if not ok1 else f"second opinion ({SECOND_OPINION_MODEL}): {issue2}"

    ok3, issue3 = _critic_judge(question, answer, think=True)
    if ok3:
        return True, ""
    return False, (issue3 or doubt)

# ==================================================
# 提案・統合
# ==================================================


def proposer_sys_for(model):
    """モデルのペルソナ人格を PROPOSER_SYS の前に前置したシステムプロンプトを返す。"""
    identity = PERSONA_IDENTITY.get(model)
    return f"{identity}\n{PROPOSER_SYS}" if identity else PROPOSER_SYS


def get_single_proposal(model, question, reference, issue=None, history=None):
    """issue: Critic の指摘。history: 過去の会話履歴（コンテキスト継続用）。"""
    history = history or []
    sys_prompt = proposer_sys_for(model)
    if reference is None:
        content = question
        if issue:
            content = (f"{question}\n\n(Note: a previous attempt was flagged by a reviewer: "
                       f"{issue} — avoid that pitfall.)")
        msgs = (
            [{"role": "system", "content": sys_prompt}]
            + history
            + [{"role": "user", "content": content}]
        )
    else:
        note = (f"A reviewer flagged this issue with the draft: {issue}\n\n"
                if issue else "")
        msgs = (
            [{"role": "system", "content": sys_prompt}]
            + history
            + [{"role": "user", "content": (
                f"Question:\n{question}\n\n"
                f"A draft answer from the panel:\n{reference}\n\n"
                f"{note}"
                "Improve it: fix errors, add missing points, make it more "
                "accurate. Output your improved answer only."
            )}]
        )
    return model, ask(model, msgs, PROPOSER_TEMP, think=PROPOSER_THINK,
                      num_predict=proposer_predict_for(model), label="proposer")


def get_proposals(models, question, reference=None, issue=None, history=None):
    """Conductor が選んだ models のみで提案を集める。history: 会話コンテキスト。
    多様性維持: reference がある回でも先頭 1 体は元の質問から新規に回答する。"""
    jobs = [(m, (None if (reference is not None and i == 0) else reference))
            for i, m in enumerate(models)]
    if PARALLEL_PROPOSERS:
        out = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
            futs = [ex.submit(get_single_proposal, m, question, ref, issue, history)
                    for m, ref in jobs]
            for f in concurrent.futures.as_completed(futs):
                out.append(f.result())
        return out
    return [get_single_proposal(m, question, ref, issue, history) for m, ref in jobs]


def use_jp_aggregator(text):
    """qwen3 を統合役にすべき質問かどうか。ひらがな/カタカナに加え、漢字のみの短文
    （「東京都の人口は?」等）も対象にする。漢字だけでは日本語と中国語を確実に区別
    できないが、qwen3 は中国語も堅牢なので qwen3 側に倒すのが安全
    （deepseek-r1 の言語混入の既知問題を踏まない）。"""
    t = text or ""
    return bool(re.search(r"[぀-ヿ]", t)) or bool(re.search(r"[㐀-鿿]", t))


def pick_aggregator(question, has_code=False):
    """統合役の選定。コード付き→AGGREGATOR(qwen3-coder)、日本語→強い思考型JPモデル
    （未導入なら従来の qwen3:4b）、それ以外→思考型 AGGREGATOR_REASONING（未導入なら既定）。"""
    if has_code:
        return AGGREGATOR
    if use_jp_aggregator(question):
        if JP_AGGREGATOR_STRONG and JP_AGGREGATOR_STRONG in PROPOSERS:
            return JP_AGGREGATOR_STRONG
        if JP_AGGREGATOR in PROPOSERS or JP_AGGREGATOR == CONDUCTOR:
            return JP_AGGREGATOR
        return AGGREGATOR
    if AGGREGATOR_REASONING in PROPOSERS:
        return AGGREGATOR_REASONING
    return AGGREGATOR


def aggregate(question, proposals):
    # 【修正】アグリゲーターに渡す前に各提案の think を除去（文脈圧迫と審判の混乱を防ぐ）
    good = [(m, strip_think(a)) for (m, a) in proposals if not a.startswith("__ERROR__")]
    if not good:
        return "__ERROR__: 全プロポーザーが失敗しました（モデル/Ollama/VRAM を確認）。"
    # コード付き提案には実行結果の証拠を添える。どの案が実際に動くかは決定的情報なので、
    # アグリゲータの取捨選択の判断材料として最強（AGGREGATOR_SYS のルール6が対応）。
    # 【2026-07-22 修正】以前はここで `good` 自体をタグ付き版で上書きしていたため、
    # 保険2（統合失敗時に good から直接返す）が [Execution check: ...] タグや生の
    # トレースバックをユーザー向け回答に漏らしていた。`good` はクリーンな
    # (model, strip_think(answer)) のまま保持し、タグ付きビューは別変数
    # （annotated）に持たせて、アグリゲータへの block 文字列構築にのみ使う。
    annotated = good
    if CODE_EXECUTION:
        annotated = []
        for m, ans in good:
            if extract_code(ans):
                issue = code_check(ans)
                tag = ("[Execution check: PASSED]" if issue is None
                       else f"[Execution check: FAILED]\n{issue}")
                ans = f"{ans}\n\n{tag}"
            annotated.append((m, ans))

    labels = [chr(ord("A") + i) for i in range(len(annotated))]
    block = "\n\n".join(
        f"Answer {lab}:\n{ans}" for lab, (_m, ans) in zip(labels, annotated)
    )
    user = f"Question:\n{question}\n\n{block}"

    def _run(model, think=None):
        return ask(
            model,
            [{"role": "system", "content": AGGREGATOR_SYS},
             {"role": "user", "content": user}],
            AGGREGATOR_TEMP,
            think=think,
            num_predict=model_cfg(model, "num_predict", NUM_PREDICT_AGGREGATOR),
            label="aggregator",
        )

    def _bad(out):
        return out.startswith("__ERROR__") or not strip_think(out).strip()

    primary = pick_aggregator(question,
                              has_code=any(extract_code(a) for _m, a in good))
    out = _run(primary)

    # 【保険1】空/エラーなら qwen3 の think=False で再統合。空返答の根本原因は
    # 「思考が num_predict を食い尽くして本文ゼロ」(2026-07-04 実測)なので、
    # 思考を切って全予算を本文に回すのが確実。primary が qwen3 だった場合にも有効。
    if _bad(out):
        print(f"   ⚠ アグリゲータ({primary})が空返答/失敗 → {JP_AGGREGATOR}(think=False) で再統合")
        out = _run(JP_AGGREGATOR, think=False)

    # 【保険2】それでもダメなら、Critic が ok と判定した提案をそのまま返す。
    # 3体分の正しい提案が手元にあるのに空回答で失点するのが最悪ケースなので、それを塞ぐ。
    if _bad(out):
        print("   ⚠ 統合に失敗 → 提案から直接選択します")
        for _m, a in good:
            ok, _issue = critique(question, a)
            if ok:
                return a
        return max(good, key=lambda x: len(x[1]))[1]

    return out

# ==================================================
# 自己一貫性投票（Self-Consistency + PoT）
# ==================================================
# 答えが機械照合できるタスク(math / mcq)では、1 回の MoA 統合より
# 「k 回独立に解かせて最終答を抽出し多数決」の方が確実に強い（Self-Consistency）。
# さらに math では「Python を書かせて実行し、その出力を 1 票にする」PoT 票を混ぜ、
# 計算ミス系の誤答を機械的に排除する。時間無制限・精度最優先の方針の中核機能（2026-07-11）。

# nk108 方針: 時間は無制限・精度最優先。SC は「サンプルを増やすほど当たる」ので上限を高めに取る。
SC_ENABLED = True
SC_INITIAL = 6          # 第1バッチの CoT サンプル数（精度優先で厚め）
SC_STEP = 4             # 過半数が取れないときの追加サンプル数
SC_MAX = 20             # 主力 CoT サンプルの上限（PoT・安価票は別枠）。時間無制限方針で高め
# 全会一致判定 (cnt == n) は抽出成功サンプルのみで n を数えるため、thinking の
# num_predict 打ち切りで __ERROR__ になる／PoT コードが実行失敗する／\boxed{} が
# 出ない、といった抽出失敗が第1バッチで多発すると n=1 (残り全滅) でも「全会一致」
# 扱いになり、事実上 k=1 で確定してしまう（精度優先方針に反する縮退）。過半数側は
# 既に n>=4 の下限があるため、全会一致側にも同じ考え方の下限を設ける（2026-07-21）。
SC_MIN_VOTES = 3        # 全会一致で確定してよい最小サンプル数（これ未満は追加サンプリングへ）
# 2026-07-22: _arbitrate に同時提示する同数タイ候補の上限。3-way以上の拮抗も正しく
# 全候補を裁定役に見せるための変更（下記参照）だが、病的に多い同数タイで num_ctx
# (gotcha #2: 8192/16384 に固定)を溢れさせないよう上限で保護する。超過分は
# 黙って捨てず _arbitrate 内でログに出す。
ARBITRATE_MAX_CANDIDATES = 4
SC_TEMP = 0.7           # 多様性確保（投票の独立性）
SC_POT = True           # math で PoT(Python 実行)票を混ぜる
SC_POT_TIMEOUT = 90     # PoT コードの実行タイムアウト秒（総当たり解法に余裕を持たせる）
REASONING_MODELS = ["gpt-oss:20b", "qwen3.6:35b"]  # SC の主力（導入済みのものだけ使われる）
SC_CHEAP_MODEL = "NitrAI/VibeThinker-3B"  # VRAM 常駐の量産サンプラー（3B・高速）
SC_CHEAP_VOTES = 0      # 安価票の数。VibeThinker の AIME ミニ実測で合格したら 6〜12 へ引き上げる
# 票が拮抗したときの最終審判。8GB 環境では gpt-oss:120b(65GB) が RAM48+VRAM8=56GB を超え
# NVMe ページングで1裁定に数十分〜数時間かかり得るため、qwen3.6:35b（思考型・理数最強格）で裁く
# （2026-07-12 nk108 決定）。120b は FUGU_HIGH_VRAM=1（96GB 環境）で解禁される。
# 失敗/空/未導入なら _arbitrate が REASONING_MODELS へ自動フォールバックする。
ARBITER_MODEL = "qwen3.6:35b"

SC_PROMPT_MATH = (
    "Solve the problem step by step, rigorously. Verify your result before answering. "
    "At the very end, put ONLY the final answer in \\boxed{}."
)
SC_PROMPT_MCQ = (
    "Solve the problem step by step. Compare all choices before deciding. "
    "At the very end, output ONLY the letter of the correct choice in \\boxed{} "
    "(for example \\boxed{B})."
)
SC_PROMPT_POT = (
    "Solve the problem by writing ONE complete Python program. "
    "Prefer exact arithmetic (integers, fractions, sympy) over floats. Brute force is fine. "
    "The program must print ONLY the final answer on its last line. "
    "Wrap it in a single ```python block. No input() calls."
)

# 2026-07-22: 全角数字/A-E に加えて、CJK 寄りのプロポーザ (qwen/gemma 系) が
# しばしば出力する Unicode 記号の等価表記もここで正規化する:
#   U+2212 (MINUS SIGN) / U+FF0D (fullwidth hyphen-minus) -> '-'
#   U+FF0E (fullwidth full stop)                          -> '.'
#   U+FF0F (fullwidth solidus)                            -> '/'
#   U+FF0C (fullwidth comma)                               -> ','
# これらを潰さないと "−5"(U+2212) と "-5" は na.lower()==nb.lower() でも
# Fraction() でも一致せず（Fraction は U+2212 を拒否する）、answers_equivalent
# は math_verify 頼みになる。math_verify が失敗すると本来同じ答えが
# vote_answers で票が2系統に割れ、誤答がプルラリティを取ったり無駄なサンプル
# 消費・仲裁が発生する（精度優先の自己整合性投票が崩れる）。曖昧な en-dash
# (U+2013) / em-dash (U+2014) は区間表記等と衝突しうるため、意図的にここでは
# マッピングしない。
_FW_TRANS = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥａｂｃｄｅ−－．／，",
    "0123456789ABCDEabcde--./,",
)


def extract_boxed(text):
    """最後の \\boxed{...} の中身を波括弧の対応を数えて取り出す（無ければ None）。"""
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    # 2026-07-22: while ループが depth>0 のまま text 末尾に達した場合、
    # \boxed{ が閉じられていない（thinking モデルの num_predict 打ち切り等で
    # 出力が途中で切れた場合の既知の失敗モード。gotcha #2 参照）。
    # このとき out には「答え」ではなく切れた出力の残骸が入っているだけなので、
    # それを answer として返すと solve_verifiable の多数決 (cnt*2 > n) で
    # 分母 n を水増しし、誤答が票として数えられてしまう。
    # 「無投票」の方が「誤った票」より安全という方針（精度優先）に従い、
    # 閉じ括弧に到達できなかった場合は None を返す。
    if depth > 0:
        return None
    ans = "".join(out).strip()
    return ans or None


def normalize_answer(ans):
    """投票用の答え正規化: 全角→半角、$ と桁区切り除去、宣言前置きの除去、外殻 \\text{} 剥がし。"""
    if ans is None:
        return ""
    s = str(ans).strip().translate(_FW_TRANS)
    s = s.replace("$", "").replace("\\!", "").replace("\\,", "").strip()
    s = re.sub(r"^(?:the\s+)?(?:final\s+)?(?:answer|答え|正解)\s*(?:is|[:：は])?\s*",
               "", s, flags=re.IGNORECASE)
    # 桁区切りの除去。"11,\! 111,\! 100" のように区切り後に空白が入る表記（MATH-500 の
    # 正解表記で実在、2026-07-12 実測）も潰すため \s* を許容する
    s = re.sub(r"(?<=\d),\s*(?=\d{3}\b)", "", s)   # 12,345 / 12, 345 → 12345
    s = s.rstrip("。．.").strip()
    s = re.sub(r"\s+", " ", s)
    m = re.fullmatch(r"\\(?:text|mathrm)\{(.*)\}", s)
    if m:
        s = m.group(1).strip()
    return s


def extract_final_answer(text, task_type="math"):
    """回答テキストから最終答を抽出する（見つからなければ None）。
    優先順: \\boxed{} > 「答え/Answer」宣言 > 最後の数値。mcq は選択肢文字 A-E を返す。"""
    if not text:
        return None
    text = strip_think(text)
    boxed = extract_boxed(text)
    if task_type == "mcq":
        if boxed:
            # 2026-07-21: 「A-E のどれかを本文中どこでも最後の1文字」ではなく、boxed 内容の
            # 先頭にある選択肢文字だけを拾う。答えの文字は慣例的に boxed の先頭に来るため、
            # \boxed{C, because it is the largest} のような散文混じりでも "Because" の B を
            # 誤って拾わない。先頭にマッチしなければ（\boxed{None of the above} 等）誤った
            # 文字を返さず、下の宣言パターン探索 → 最終的に None（無投票、誤投票より安全）に
            # フォールスルーさせる。
            m = re.match(r"\(?\s*([A-E])\b", normalize_answer(boxed).upper())
            if m:
                return m.group(1)
        for pat in (r"(?:answer|答え|正解)\s*(?:is|[:：は])?\s*\(?([A-EＡ-Ｅ])\)?(?![A-Za-z])",
                    r"^\s*\(?([A-E])\)?\s*(?:が正解|です)?\s*$"):
            ms = re.findall(pat, text, re.IGNORECASE | re.MULTILINE)
            if ms:
                return ms[-1].translate(_FW_TRANS).upper()
        return None
    if boxed:
        return normalize_answer(boxed) or None
    ms = re.findall(r"(?:final answer|answer|答え|正解)\s*(?:is|[:：は])\s*([^\n]{1,60})",
                    text, re.IGNORECASE)
    if ms:
        cand = normalize_answer(ms[-1])
        if cand:
            # 「700 円です」のような後置き単位・助詞を落とす: 数値で始まるなら数値部のみ
            m = re.match(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?", cand)
            return m.group(0).replace(" ", "") if m else cand
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?", text)
    return normalize_answer(nums[-1]) if nums else None


def answers_equivalent(a, b):
    """2つの答えが数学的に同値か。正規化一致 → 分数/小数の数値一致 → math_verify の順で判定。"""
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return False
    if na.lower() == nb.lower():
        return True
    try:
        from fractions import Fraction
        if Fraction(na.replace(" ", "")) == Fraction(nb.replace(" ", "")):
            return True
    except Exception:
        pass
    try:
        # タイムアウトは無効化して呼ぶ: math_verify の既定タイムアウトは multiprocessing を
        # 使い、Windows では __main__ 再import でハンドルエラーを撒き散らす（実測 2026-07-11）。
        # 答えは短い文字列（<=80字）なので sympy が固まるリスクは実用上無視できる。
        import logging as _logging
        _logging.getLogger("math_verify").setLevel(_logging.ERROR)
        from math_verify import parse as _mv_parse, verify as _mv_verify
        return bool(_mv_verify(_mv_parse(f"${na}$", parsing_timeout=None),
                               _mv_parse(f"${nb}$", parsing_timeout=None),
                               timeout_seconds=None))
    except Exception:
        return False


def vote_answers(answers):
    """答えリストを同値クラスへ集約し (最多答, その票数, クラス一覧) を返す。
    クラス一覧は [[代表答え, 票数], ...] 票数降順。答えが無ければ (None, 0, [])。"""
    classes = []
    for a in answers:
        if not a:
            continue
        for c in classes:
            if answers_equivalent(a, c[0]):
                c[1] += 1
                break
        else:
            classes.append([a, 1])
    if not classes:
        return None, 0, []
    classes.sort(key=lambda c: -c[1])
    return classes[0][0], classes[0][1], classes


def _sc_sample(model, question, task_type, pot=False, history=None):
    """SC の 1 サンプル。(answer, full_text) を返す（抽出/実行失敗は answer=None）。"""
    if pot:
        sysp = SC_PROMPT_POT
    else:
        sysp = SC_PROMPT_MCQ if task_type == "mcq" else SC_PROMPT_MATH
    raw = ask(model,
              [{"role": "system", "content": sysp}] + list(history or [])
              + [{"role": "user", "content": question}],
              SC_TEMP, think=proposer_think_for(model),
              num_predict=proposer_predict_for(model), label="sc")
    if raw.startswith("__ERROR__"):
        return None, raw
    text = strip_think(raw)
    if pot:
        code = extract_code(text)
        if not code:
            return None, text
        ok, out = run_python(code, timeout=SC_POT_TIMEOUT, stdout_only=True)
        out = (out or "").strip()
        if not ok or not out:
            return None, text
        ans = out.splitlines()[-1].strip()
        if not ans or len(ans) > 80:
            return None, text
        return (normalize_answer(ans) or None), text + f"\n\n[PoT execution output]\n{out[-500:]}"
    return extract_final_answer(text, task_type), text


def _representative_text(samples, answer):
    """勝ったクラスの代表解答テキスト（CoT 優先、無ければ PoT、最後は最長）を返す。"""
    fallback = None
    for s in samples:
        if s["answer"] and answers_equivalent(s["answer"], answer):
            if not s["pot"]:
                return s["text"]
            fallback = fallback or s["text"]
    if fallback:
        return fallback
    if samples:
        return max(samples, key=lambda s: len(s.get("text") or "")).get("text") or ""
    return ""


def _arbitrate(question, task_type, samples, classes):
    """票が拮抗した上位 2 クラスの代表解答を突き合わせて裁定する。
    戻り値: (裁定answer, 裁定役自身の解答テキスト) のタプル。全裁定役が失敗/空なら None。
    裁定役の優先順: ARBITER_MODEL（導入済みなら。既定 gpt-oss:120b の最上位知能）→
    それが失敗/空/未導入なら REASONING_MODELS の先頭へ堅牢にフォールバック。
    120b は 65GB で RAM(48GB)+VRAM(8GB) を超え NVMe ページングで非常に遅い可能性があるが、
    精度優先方針で「拮抗時だけ最上位が裁く」価値を取る。ダメでも degrade して止まらない。"""
    inst = installed_models()
    chain = []
    if ARBITER_MODEL and is_installed(ARBITER_MODEL, inst):
        chain.append(ARBITER_MODEL)
    for m in REASONING_MODELS:                 # 120b が失敗しても軽い思考モデルで必ず裁く
        if m in PROPOSERS and m not in chain:
            chain.append(m)
    if not chain:
        return None
    # 2026-07-22: 呼び出し側の拮抗判定(classes[0][1]==classes[1][1])は「トップ2クラスが
    # 同数」であることしか見ておらず、3クラス以上が同数タイになる N 択拮抗（例: 票数
    # [2,2,2]）もここに到達しうる。従来は classes[:2] で常に先頭2クラスしか裁定役に
    # 見せておらず、3番目以降の同数クラス（それが正解かもしれない）が黙って握りつぶされ、
    # プロンプトも "Two candidate solutions disagree" と決め打ちだった。ここではトップ
    # 票数と同数のクラスを全て（ただし num_ctx 保護のため上限 ARBITRATE_MAX_CANDIDATES
    # 件までに制限し、超過分は省略せずログに出す）候補として提示する。
    # 2件のみの通常拮抗では従来と完全に同じ挙動(先頭2件)になる。
    top_count = classes[0][1]
    tied = [c for c in classes if c[1] == top_count]
    if len(tied) > ARBITRATE_MAX_CANDIDATES:
        omitted = tied[ARBITRATE_MAX_CANDIDATES:]
        tied = tied[:ARBITRATE_MAX_CANDIDATES]
        omitted_desc = ", ".join(str(c[0]) for c in omitted)
        print(f"   [SC] {len(omitted)}件の同数タイ候補は上限のため裁定役に提示されません: {omitted_desc}")
    reps = []
    for canon, _cnt in tied:
        for s in samples:
            if s["answer"] and answers_equivalent(s["answer"], canon):
                reps.append((canon, strip_think(s["text"] or "")[:3000]))
                break
    listing = "\n\n".join(
        f"### Candidate {chr(ord('A') + i)} (final answer: {c})\n{t}"
        for i, (c, t) in enumerate(reps))
    prompt = (f"Problem:\n{question}\n\n"
              f"{len(reps)} candidate solutions disagree:\n\n{listing}\n\n"
              "Carefully check both, find the flaw in the wrong one, and solve the problem "
              "yourself if needed. At the very end, put ONLY the correct final answer in \\boxed{}.")
    for arb in chain:
        print(f"   [SC] 票が拮抗 → {arb} が裁定します")
        raw = ask(arb, [{"role": "user", "content": prompt}], 0.1,
                  num_predict=model_cfg(arb, "num_predict", 8192), label="arbiter")
        text = strip_think(raw)
        ans = extract_final_answer(text, task_type)
        if ans:
            return ans, text
        print(f"   [SC] {arb} の裁定が空/抽出不能 → 次の裁定役へ")
    return None


def solve_verifiable(question, task_type="math", history=None):
    """Self-Consistency + PoT で math/mcq を解く。
    戻り値: {"answer", "text", "votes", "n_samples"}。票が全く得られなければ None
    （呼び出し側が通常の MoA へフォールバックする）。"""
    models = [m for m in REASONING_MODELS if m in PROPOSERS]
    if not models:
        models = list(PROPOSERS[:2])
    if not models:
        return None
    cheap_ok = (SC_CHEAP_VOTES > 0 and SC_CHEAP_MODEL
                and is_installed(SC_CHEAP_MODEL, installed_models()))
    samples = []

    def add(model, pot=False):
        ans, text = _sc_sample(model, question, task_type, pot=pot, history=history)
        samples.append({"answer": ans, "text": text, "model": model, "pot": pot})
        kind = "PoT" if pot else "CoT"
        print(f"   [SC {len(samples)}] {model} ({kind}) -> {ans if ans else '(抽出失敗)'}")

    def main_cot_count():
        return sum(1 for s in samples if not s["pot"] and s["model"] != SC_CHEAP_MODEL)

    # 【重要】OLLAMA_MAX_LOADED_MODELS=1 では毎サンプルでモデルを切り替えると 13〜23GB の
    # 再ロードが多発して致命的に遅い。そこで「モデルごとにまとめて」サンプリングし再ロードを
    # 最小化する（多様性は temp=0.7 の複数サンプルで確保）。各モデルから同数ずつ引く。
    def add_batch(n):
        per = max(1, n // len(models))
        for m in models:
            for _ in range(per):
                add(m)
        # PoT は先頭モデルがロード済みのうちに末尾で実行（追加ロードなし）
        if SC_POT and task_type == "math":
            add(models[0], pot=True)

    add_batch(SC_INITIAL)
    if cheap_ok:                       # 安価票は最後にまとめて（VibeThinker を1回ロード）
        for _ in range(SC_CHEAP_VOTES):
            add(SC_CHEAP_MODEL)

    while True:
        answers = [s["answer"] for s in samples if s["answer"]]
        top, cnt, classes = vote_answers(answers)
        n = len(answers)
        # 確定条件: 全会一致（ただし n < SC_MIN_VOTES の疑似全会一致は不可）、
        # または 4 票以上で過半数
        if top is not None and n > 0 and (
            (cnt == n and n >= SC_MIN_VOTES) or (n >= 4 and cnt * 2 > n)
        ):
            break
        if main_cot_count() >= SC_MAX:
            break
        head = [(c[0], c[1]) for c in classes[:3]]
        print(f"   [SC] 票が割れています {head} → 追加サンプリング")
        add_batch(SC_STEP)

    answers = [s["answer"] for s in samples if s["answer"]]
    top, cnt, classes = vote_answers(answers)
    if top is None:
        return None
    rep = None
    if len(classes) >= 2 and classes[0][1] == classes[1][1]:
        arb_result = _arbitrate(question, task_type, samples, classes)
        if arb_result:
            top, rep = arb_result
            # 2026-07-22: 裁定役（_arbitrate）は既存の票クラスと無関係な「第三の答え」を
            # 返すことがある（例: 拮抗した {'1','2'} に対し裁定役が '3' を新規提示）。
            # 従来はここで cnt/classes を再計算せず、拮抗していた旧トップの票数
            # （classes[0][1]）をそのまま流用していたため、
            #   - res['votes'] に裁定結果の答えが載らず、実際は0票の新答えなのに
            #     まるで敗者候補が「無投票」であるかのような矛盾した内訳になる
            #   - 直後の「[SC] 確定: {top} (票 {cnt}/...)」ログが、敗者候補（旧トップ）の
            #     票数を裁定結果の答えの票数であるかのように誤表示する
            # という報告面のバグがあった。ここでは裁定後の top に対応する真の票数を
            # classes から同値判定で引き直し（一致するクラスが無ければ 0 票）、
            # votes 辞書にも裁定結果の答えを必ずキーとして載せる。
            # なお SC_MIN_VOTES の床判定（下）は rep is not None の間は素通りする既存の
            # 挙動のままで変更していない。
            match = next((c for c in classes if answers_equivalent(top, c[0])), None)
            cnt = match[1] if match else 0
            if match is None:
                classes = classes + [[top, cnt]]
            elif match[0] != top:
                # 2026-07-22 (iteration 10 の続き): 上の一致判定は answers_equivalent な
                # クラスを見つけられるが、そのクラスの代表表記（match[0]）が裁定役の
                # 返した文字列（top）と食い違うことがある（例: 拮抗クラスの代表が '1/2'
                # で裁定役は '0.5' と書く、'1000' vs '1,000'、'012' vs '12' など、
                # 分数⇄小数や桁区切りの書き直しは裁定役がよくやる）。
                # 旧コードは「match is None or match[0] != top」を一括りにして
                # 新規クラス [top, cnt] を無条件追加していたため、同値のはずの票が
                # 旧代表表記のキーと裁定後表記のキーの二つに分裂して計上され
                # （例: res['votes'] == {'1/2': 3, '0.5': 3}）、合計票数が実際の
                # 有効票数の2倍になる「truthful でない votes」を生んでいた。これは
                # iteration 10 が退治したはずの矛盾内訳バグの兄弟ケースにあたる。
                # ここでは新規クラスを追加せず、一致したクラス自身のキーを裁定後の
                # 表記に書き換えるだけにして、同じ票が二重に数えられないようにする。
                classes = [([top, cnt] if c is match else c) for c in classes]
    # 2026-07-21: ループ内の早期確定条件（cnt==n and n>=SC_MIN_VOTES / n>=4 and cnt*2>n）は
    # SC_MIN_VOTES 未満の疑似全会一致を弾くが、それは while ループの break 条件だけの話。
    # SC_MAX 消化で抜けた場合（多くのサンプルが __ERROR__/抽出失敗/\boxed{}欠落 等で無効票になった
    # ケース）はここを素通りしてしまい、1〜2票しか残っていない「勝者」をそのまま確定扱いで返して
    # いた。理由は違えど中身は同じ疑似全会一致問題なので、最終returnにも同じ床（floor）をかける。
    # ただし裁定（_arbitrate）が成功して rep が既に埋まっている場合は、裁定役が新たに出した
    # answer/text をそのまま尊重し、票数に関わらずここでは弾かない。
    if rep is None and cnt < SC_MIN_VOTES:
        print(f"   [SC] 確定票が {cnt} 票のみ (< SC_MIN_VOTES={SC_MIN_VOTES}) → MoA フォールバックへ")
        return None
    if rep is None:
        rep = _representative_text(samples, top)
    print(f"   [SC] 確定: {top}  (票 {cnt}/{len(answers)}, サンプル計 {len(samples)})")
    return {"answer": top, "text": rep,
            "votes": {c[0]: c[1] for c in classes}, "n_samples": len(samples)}


# ==================================================
# Fugu 風オーケストレーション本体
# ==================================================


def _print_plan(plan):
    tag = " (フォールバック)" if plan.get("_fallback") else ""
    print(f"\n🎼 Conductor の判断{tag}:")
    if plan.get("make_pptx"):
        print("   output      = PowerPoint (画像は内容連動で自動生成)")
    if plan.get("use_image_generation"):
        kind = "画像のみ" if plan.get("image_only") else "テキスト+イラスト"
        print(f"   image_gen   = True ({kind})")
    if plan.get("image_only"):
        # テキスト提案は走らないので mode 表示は省略
        pass
    elif plan["mode"] == "single":
        sel = plan.get("selected_proposers") or PROPOSERS[:1]
        model = sel[0] if sel else AGGREGATOR
        print(f"   mode        = single ({_persona_str(model)})")
    else:
        labels = [_persona_str(m) for m in plan.get("selected_proposers", [])]
        print(f"   mode        = moa {labels}  rounds={plan['rounds']}")
    print(f"   search_req  = {plan.get('search_required', False)}")
    if plan.get("task_type"):
        print(f"   task_type   = {plan['task_type']}")
    if plan.get("reason"):
        print(f"   reason      = {plan['reason']}")


def fugu_answer(question, plan=None, history=None):
    """事前に conduct() で得た plan に従って回答を生成する。
    plan は validate_plan 済み（selected_proposers は実モデル名で解決済み）。
    plan=None のときは内部で conduct() を実行する（eval など単体呼び出し向けの後方互換）。"""
    history = history or []
    if plan is None:
        plan, _raw = conduct(question, history=history)
        if SHOW_PLAN:
            _print_plan(plan)

    # ---------- 検証可能タスク（math/mcq）: 自己一貫性投票で解く ----------
    if (SC_ENABLED and plan.get("task_type") in ("math", "mcq")
            and not plan.get("image_only") and not plan.get("make_pptx")
            and not plan.get("use_image_generation")):
        print(f"   [SC] 検証可能タスク({plan['task_type']}) → 自己一貫性投票で解く")
        res = solve_verifiable(question, plan["task_type"], history=history)
        if res and res.get("answer"):
            txt = res.get("text") or ""
            # 裁定で答えが差し替わった場合など、本文の結論と投票結果がずれたら明示する
            body_ans = extract_final_answer(txt, plan["task_type"])
            if not (body_ans and answers_equivalent(body_ans, res["answer"])):
                txt += f"\n\n(自己一貫性投票による最終解答: {res['answer']})"
            return txt
        print("   [SC] 投票不成立 → 通常の合議へフォールバック")

    seed_answer = None  # エスカレーション時、単体回答を捨てずに合議の初期ドラフトにする
    seed_issue = None   # Critic の指摘も合議側へ伝える

    # ---------- 単体モード ----------
    if plan["mode"] == "single":
        sel = plan.get("selected_proposers") or PROPOSERS[:1]
        model = sel[0] if sel else (PROPOSERS[0] if PROPOSERS else AGGREGATOR)
        ans = strip_think(ask(
            model,
            ([{"role": "system", "content": proposer_sys_for(model) + PRESENTATION_STYLE}]
             + history
             + [{"role": "user", "content": question}]),
            PROPOSER_TEMP,
            think=PROPOSER_THINK,
            num_predict=proposer_predict_for(model),
            label="single",
        ))
        if ans.startswith("__ERROR__"):
            print("   (単体モデル失敗 → 合議へ切替)")
            plan["mode"] = "moa"
            plan["selected_proposers"] = PROPOSERS[:3]
        elif ADAPTIVE_ESCALATION:
            # 高速チェック2系統 + 疑義があれば思考ON再検算（verify_single 参照）
            ok, issue = verify_single(question, ans)
            if ok:
                return ans
            print(f"   ⤴ 単体回答に難あり（{issue}）→ 合議へエスカレーション")
            seed_answer, seed_issue = ans, issue
            plan["mode"] = "moa"
            plan["selected_proposers"] = PROPOSERS[:3]
            plan["rounds"] = max(1, plan["rounds"])
        else:
            return ans

    # ---------- 合議(MoA)モード：選抜した分だけ、必要なら再帰的に反復 ----------
    models = plan["selected_proposers"] or PROPOSERS[:3]
    planned = min(MAX_ROUNDS, max(1, plan["rounds"]))
    reference = seed_answer  # エスカレーションなら単体回答を初期ドラフトとして再利用
    issue_hint = seed_issue
    final = None
    r = 0
    while True:
        proposals = get_proposals(models, question, reference, issue_hint, history=history)
        if SHOW_PROPOSALS:
            mode = "並列" if PARALLEL_PROPOSERS else "逐次"
            print(f"\n--- ラウンド {r + 1}: 各提案（{mode}・{len(models)}体） ---")
            for m, a in proposals:
                print(f"[{m}]\n{strip_think(a)}\n")
        final = aggregate(question, proposals)
        reference = final  # 次ラウンドは今回の統合結果を土台に改善
        issue_hint = None  # 指摘は消費済み。以降のチェックが新しい指摘を設定する
        r += 1

        # コード回答は実行検証で誤りが機械的に見つかるため、修正ラウンドの上限を広げる
        fin = strip_think(final)
        limit = (MAX_ROUNDS_CODE if (CODE_EXECUTION and extract_code(fin))
                 else MAX_ROUNDS)
        if r >= limit:
            break

        # 全プロポーザーが失敗した場合は即打ち切り。
        # これ以上ラウンドを重ねても同じ失敗が繰り返されるだけで、
        # Critic が「エラーメッセージは不十分」と正しく判定するため
        # MAX_ROUNDS 分だけ無駄なループが発生する（実測: x8 proposer + x12 critic 呼び出し）。
        if fin.startswith("__ERROR__"):
            break

        # 続行判断1（決定的・最優先）: コードを実行し、失敗なら traceback を修正ヒントに
        # して追加ラウンド。これが「自律的にコードを直し続ける」ループの本体。
        code_issue = code_check(fin)
        if code_issue:
            issue_hint = code_issue
            tail = code_issue.strip().splitlines()[-1][:80]
            print(f"   ↻ コード実行に失敗 → 修正ラウンド {r + 1}（{tail}）")
            continue

        # 続行判断2: 計画分がまだ残っていれば続行。消化済みなら Critic に委ねる（再帰）。
        if r < planned:
            need_more = True
        elif ALLOW_RECURSION:
            ok, issue = critique(question, fin)
            need_more = not ok
            if need_more:
                issue_hint = issue  # 何が不十分かを次ラウンドの提案へ伝える
                print(f"   ↻ 品質不足のため追加ラウンド（{issue}）")
            else:
                print(f"   ✓ 十分な品質と判断 → 反復を打ち切り")
        else:
            need_more = False
        if not need_more:
            break

    return final

# ==================================================
# 実行制御
# ==================================================

_READY = False


def setup():
    global PROPOSERS, AGGREGATOR, CONDUCTOR, _READY
    if _READY:
        return True
    if not ensure_server():
        return False
    if os.environ.get("FUGU_HIGH_VRAM") in ("1", "true", "True"):
        apply_high_vram_profile()
    print("[setup] ローカルモデル構成を確認します…")
    PROPOSERS, AGGREGATOR, CONDUCTOR = resolve_models()
    if not PROPOSERS or AGGREGATOR is None or CONDUCTOR is None:
        print("利用可能なモデルを用意できませんでした。")
        return False
    persona_lines = "\n".join(
        f"    {label} = {model}"
        + ("" if model in PROPOSERS else "  [未導入]")
        for label, model in PERSONA_MODELS.items()
    )
    print(f"""
===================================================
 🐡 Local Fugu-style MoA Orchestrator (3大AIオールスター)
  conductor : {CONDUCTOR}   (動的に委譲を決定)
  proposers :
{persona_lines}
  aggregator: {AGGREGATOR}
  image_gen : backend={IMAGE_BACKEND}  (a1111={A1111_URL} / comfyui={COMFYUI_URL})
  max_rounds: {MAX_ROUNDS}  escalation: {ADAPTIVE_ESCALATION}  recursion: {ALLOW_RECURSION}
  mode      : {"並列" if PARALLEL_PROPOSERS else "逐次"}
===================================================
 OLLAMA_MAX_LOADED_MODELS=1 は恒久設定済み（ユーザー環境変数）。
""")
    _READY = True
    return True


def ask_fugu(question, baseline=SHOW_BASELINE, *,
             use_search=False, rag_dirs=None, out_file=None,
             history_file=None, office_attached=False):
    """質問を Fugu パイプラインで処理する。
    use_search: True なら Web 検索を行いコンテキストに注入する（Conductor が
      search_required=true を出した場合も自動で有効化される）。
    rag_dirs: ローカル文書ディレクトリ（省略時は RAG_DIRS グローバル設定を使用）。
    out_file: 回答を保存するファイルパス（.md 推奨）。
    history_file: 永続化に使う JSON ファイルパス（省略時は HISTORY_FILE）。
    office_attached: Office 文書が添付されている旨を Conductor へ伝えるヒント。
    """
    global _HISTORY
    if not setup():
        return None
    t0 = time.time()

    # --- Conductor プランを先に取得（検索要否・画像生成・Office ルーティングを決める）---
    print("\n[Fugu] Conductor がオーケストレーションを開始します...")
    plan, _raw = conduct(question, history=list(_HISTORY),
                         office_attached=office_attached)
    if SHOW_PLAN:
        _print_plan(plan)

    panel = plan.get("selected_proposers") or PROPOSERS[:IMAGE_PROMPT_PANEL]

    # --- 経路1: 画像のみ（テキスト回答不要・LLM群がプロンプト起草）---
    if plan.get("use_image_generation") and plan.get("image_only"):
        print("\n[Fugu] 画像のみ生成（LLM群がプロンプトを起草）...")
        result = handle_image_generation(question, panel=panel)
        elapsed = round(time.time() - t0, 1)
        print("\n===== 画像生成結果 =====")
        print(result)
        print(f"\n(所要 {elapsed} 秒)")
        notify_slack(question, result, elapsed)
        if out_file and not result.startswith("__ERROR__"):
            _save_answer_to_file(question, result, elapsed, out_file, context="")
        return result

    # --- コンテキスト構築（Web検索 + RAG）。検索は CLI フラグ or Conductor 判断で有効化 ---
    do_search = use_search or plan.get("search_required", False)
    context = build_context(question, use_search=do_search,
                            rag_dirs=rag_dirs or RAG_DIRS)
    question_with_ctx = _with_context(question, context)

    if baseline:
        print("\n===== 単体ベースライン（aggregator モデル直答） =====")
        base = ask(
            AGGREGATOR,
            [{"role": "system", "content": PROPOSER_SYS},
             {"role": "user", "content": question_with_ctx}],
            AGGREGATOR_TEMP,
            label="baseline",
        )
        print(strip_think(base))

    # --- 本文を MoA で生成 ---
    final = strip_think(
        fugu_answer(question_with_ctx, plan, history=list(_HISTORY)) or ""
    )
    text_answer = final  # 履歴にはテキスト本文のみ保存する

    # --- 経路2: PowerPoint（本文をスライド化し内容連動で画像を埋め込む）---
    if plan.get("make_pptx") and not final.startswith("__ERROR__"):
        print("\n[Fugu] 本文を PowerPoint 化します（画像は内容連動で自動生成）...")
        pptx_out = out_file if (out_file and str(out_file).lower().endswith(
            (".pptx", ".ppt"))) else None
        deck = build_pptx(question, final, pptx_out)
        final = text_answer + f"\n\n---\n## 生成した PowerPoint\n- 保存先: {deck}"
        out_file = None  # 保存はここで完結（下の汎用保存は行わない）

    # --- 経路3: イラスト付き回答（本文＋回答内容から画像生成）---
    elif (plan.get("use_image_generation") and not plan.get("image_only")
          and not final.startswith("__ERROR__")):
        print("\n[Fugu] 回答内容からイラストを生成します...")
        base = f"{question}\n\n[回答の要点]\n{text_answer[:800]}"
        img = handle_image_generation(base, panel=panel)
        final = text_answer + "\n\n---\n## 生成画像\n" + img

    elapsed = round(time.time() - t0, 1)
    print("\n===== 最終回答 =====")
    if final.startswith("__ERROR__"):
        print("生成に失敗しました:", final)
    else:
        print(final)
    print(f"\n(所要 {elapsed} 秒)")
    notify_slack(question, final, elapsed)

    # --- 会話履歴を更新（エラーでなければ記録・永続化）---
    if not final.startswith("__ERROR__"):
        _HISTORY.append({"role": "user", "content": question})   # 元の質問を保存
        _HISTORY.append({"role": "assistant", "content": final})
        _trim_history(_HISTORY)
        save_history_file(_HISTORY, path=history_file)
        print(f"   [会話履歴: {len(_HISTORY) // 2} 往復保持中]")

    # --- ファイル出力 ---
    if out_file and not final.startswith("__ERROR__"):
        _save_answer_to_file(question, final, elapsed, out_file,
                             context=context)

    return final


# コード系拡張子（コードブロックを抽出してそのまま書き出す）
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".java", ".kt", ".swift",
    ".rb", ".php", ".sh", ".bat", ".ps1", ".sql", ".r", ".m", ".jl",
}


def _extract_code_for_output(answer: str, suffix: str) -> str:
    """回答から対象言語のコードブロックを抽出して返す。
    見つからない場合は回答全体からマークダウン装飾を除いたテキストを返す。

    2026-07-22: iteration-7 の extract_code と同じ誤抽出クラスの修正。
    旧実装は言語指定ありフェンスを re.search(rf"```{lang}[ \t]*\n(.*?)```") で
    検索し（```python3 の '3' が [ \t]*\n にマッチせず python3 ブロックを取り
    こぼす）、言語指定なしフォールバックも re.search(r"```(?:\w+)?[ \t]*\n(.*?)```")
    による前方走査だったため、```json/```text/```output 等の非コードブロックが
    先行すると、その閉じフェンスを誤って開始フェンスとみなし、2ブロック間の
    プロースや先行ブロックの中身をコードとして誤抽出していた。
    修正: extract_code と同一のフェンス正規表現 re.finditer(r"```([^\n`]*)\n(.*?)```")
    で全ブロックを一度に収集し、(1) suffix の対象言語タグ一致 → (2) タグ無し
    (bare) → (3) 既知の非コードタグ以外 の優先順で最初に見つかったブロックの
    本文を返す。該当ブロックが無ければ従来通りフェンス無しフォールバックを返す。"""
    lang_map = {
        ".py": ["python", "py", "python3"],
        ".js": ["javascript", "js"],
        ".ts": ["typescript", "ts"],
        ".go": ["go"],
        ".rs": ["rust"],
        ".c": ["c"],
        ".cpp": ["cpp", "c++"],
        ".cs": ["csharp", "cs"],
        ".java": ["java"],
        ".rb": ["ruby", "rb"],
        ".sh": ["bash", "sh", "shell"],
        ".sql": ["sql"],
        ".r": ["r"],
    }
    # 非コードとみなす既知のドキュメント系タグ（保守的なスキップリスト）
    _NON_CODE_TAGS = {
        "json", "text", "txt", "output", "console", "log",
        "yaml", "yml", "xml", "csv", "markdown", "md", "diff", "ini", "toml",
    }
    langs = {l.lower() for l in lang_map.get(suffix, [])}

    blocks = [(m.group(1).strip().lower(), m.group(2))
              for m in re.finditer(r"```([^\n`]*)\n(.*?)```", answer, re.DOTALL)]

    # (1) 対象言語タグと一致する最初のブロック
    for lang, body in blocks:
        if lang in langs:
            return body
    # (2) タグ無し(bare)の最初のブロック
    for lang, body in blocks:
        if lang == "":
            return body
    # (3) 既知の非コードタグ以外の最初のブロック
    for lang, body in blocks:
        if lang not in _NON_CODE_TAGS:
            return body
    # フェンスなし: マークダウン見出し行を除いた本文を返す
    lines = [l for l in answer.splitlines() if not l.startswith("#")]
    return "\n".join(lines).strip()


def _save_as_markdown(out: Path, question: str, answer: str,
                      elapsed: float, context: str):
    """Markdown 形式で追記保存。"""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = f"## Q ({ts})\n\n{question}\n\n"
    if context:
        block += (f"<details><summary>Context (search/RAG)</summary>\n\n"
                  f"{context}\n\n</details>\n\n")
    block += f"## A\n\n{answer}\n\n*所要: {elapsed}s*\n\n---\n\n"
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    out.write_text(existing + block, encoding="utf-8")


def _save_as_text(out: Path, question: str, answer: str, elapsed: float):
    """プレーンテキスト形式で追記保存。"""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = f"[{ts}]\nQ: {question}\n\nA:\n{answer}\n\n(所要 {elapsed}s)\n{'='*60}\n\n"
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    out.write_text(existing + block, encoding="utf-8")


def _save_as_code(out: Path, answer: str):
    """コード拡張子のファイルとして保存。コードブロックを抽出して書き出す。"""
    code = _extract_code_for_output(answer, out.suffix.lower())
    out.write_text(code + "\n", encoding="utf-8")


def _save_as_html(out: Path, question: str, answer: str, elapsed: float):
    """HTML 形式で保存。"""
    from datetime import datetime
    import html as _html
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    q_esc = _html.escape(question)
    a_lines = []
    for line in answer.splitlines():
        if line.startswith("```"):
            tag = "<pre><code>" if not line[3:].strip() == "" else "<pre><code>"
            a_lines.append("<pre><code>")
        else:
            a_lines.append(_html.escape(line) + "<br>")
    body = (f"<h2>Q <small>({ts})</small></h2>\n<p>{q_esc}</p>\n"
            f"<h2>A</h2>\n<p>{''.join(a_lines)}</p>\n"
            f"<hr><p><small>所要: {elapsed}s</small></p>\n")
    existing_body = ""
    if out.exists():
        content = out.read_text(encoding="utf-8")
        m = re.search(r"(<body>)(.*?)(</body>)", content, re.DOTALL)
        if m:
            existing_body = m.group(2)
    html_content = (f"<!DOCTYPE html>\n<html lang='ja'><head>"
                    f"<meta charset='UTF-8'><title>Fugu Output</title></head>\n"
                    f"<body>\n{existing_body}{body}</body></html>")
    out.write_text(html_content, encoding="utf-8")


def _save_as_pdf(out: Path, question: str, answer: str, elapsed: float):
    """PDF 形式で保存。fpdf2 が必要。未インストール時は .md にフォールバック。"""
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        # Unicode フォント: fpdf2 は内蔵 DejaVu を使用
        try:
            pdf.set_font("DejaVu", size=12)
        except Exception:
            pdf.set_font("Helvetica", size=12)
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for heading, text in [("Q", question), ("A", answer)]:
            pdf.set_font_size(14)
            pdf.cell(0, 10, f"{heading} ({ts})" if heading == "Q" else heading, ln=True)
            pdf.set_font_size(11)
            for line in text.splitlines():
                pdf.multi_cell(0, 7, line or " ")
            pdf.ln(5)
        pdf.output(str(out))
        return
    except ImportError:
        pass
    # フォールバック: .md として保存
    md_path = out.with_suffix(".md")
    _save_as_markdown(md_path, question, answer, elapsed, "")
    print(f"   [PDF生成には fpdf2 が必要 (pip install fpdf2)。代わりに保存: {md_path}]")
    return md_path


def _save_as_docx(out: Path, question: str, answer: str, elapsed: float):
    """Word (.docx) 形式で保存。python-docx が必要。未インストール時は .md にフォールバック。"""
    try:
        import docx as _docx
        from datetime import datetime
        doc = _docx.Document()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc.add_heading(f"Q ({ts})", level=1)
        doc.add_paragraph(question)
        doc.add_heading("A", level=1)
        in_code = False
        code_lines = []
        for line in answer.splitlines():
            if line.startswith("```"):
                if in_code:
                    doc.add_paragraph("\n".join(code_lines), style="No Spacing")
                    code_lines = []
                    in_code = False
                else:
                    in_code = True
            elif in_code:
                code_lines.append(line)
            else:
                doc.add_paragraph(line) if line.strip() else doc.add_paragraph("")
        doc.add_paragraph(f"所要: {elapsed}s")
        doc.save(str(out))
        return
    except ImportError:
        pass
    md_path = out.with_suffix(".md")
    _save_as_markdown(md_path, question, answer, elapsed, "")
    print(f"   [DOCX保存には python-docx が必要 (pip install python-docx)。代わりに保存: {md_path}]")
    return md_path


def _save_as_excel(out: Path, answer: str):
    """回答中のCSVライクな表を Excel (.xlsx) として保存。openpyxl が必要。"""
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Fugu Output"
        for line in answer.splitlines():
            if line.strip():
                cols = [c.strip() for c in re.split(r"[,\t|]", line)]
                ws.append(cols)
        wb.save(str(out))
        return
    except ImportError:
        pass
    txt_path = out.with_suffix(".csv")
    txt_path.write_text(answer, encoding="utf-8")
    print(f"   [Excel保存には openpyxl が必要 (pip install openpyxl)。代わりに保存: {txt_path}]")
    return txt_path


# ==================================================
# PowerPoint 生成（画像入りスライド）
# ==================================================

def _parse_slides(answer):
    """Markdown 回答をスライド構造 [{'title':..,'bullets':[..]}] へ分解する。
    見出し(#〜####)でスライドを区切り、箇条書き/段落を bullets にする。"""
    slides = []
    cur = None
    in_code = False
    for ln in answer.splitlines():
        s = ln.rstrip()
        if s.strip().startswith("```"):
            in_code = not in_code
            continue
        m = re.match(r"^\s*#{1,4}\s+(.*)$", s)
        if m and not in_code:
            if cur is not None:
                slides.append(cur)
            cur = {"title": re.sub(r"[*_`#]+", "", m.group(1)).strip()[:80], "bullets": []}
            continue
        t = s.strip()
        if not t:
            continue
        if not in_code:
            t = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", t)  # 箇条書き記号除去
            t = re.sub(r"[*_`#]+", "", t).strip()           # 強調記号除去
        if not t:
            continue
        if cur is None:
            cur = {"title": "概要", "bullets": []}
        cur["bullets"].append(t[:200])
    if cur is not None:
        slides.append(cur)
    return slides


def _deck_title(question, slides):
    q = (question or "").strip().splitlines()[0] if question else ""
    if 0 < len(q) <= 40:
        return q
    if slides and slides[0].get("title"):
        return slides[0]["title"]
    return "プレゼンテーション"


_PPTX_IMG_SCHEMA = {
    "type": "object",
    "properties": {"images": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "prompt": {"type": "string"}},
        "required": ["index", "prompt"]}}},
    "required": ["images"],
}


def plan_pptx_images(title, slides):
    """タイトル+各スライドの見出しから、画像が効果的なスライドを選び英語SDプロンプトを割り当てる。
    戻り値 {index: prompt}（index 0=タイトルのヒーロー画像 / 1..=各スライド）。最大 PPTX_MAX_IMAGES。"""
    outline = f"Title (index 0): {title}\n" + "\n".join(
        f"Slide {i + 1}: {s['title']} — {'; '.join(s['bullets'][:3])}"
        for i, s in enumerate(slides))
    sys = (
        "You plan illustrative images for a slide deck. Include index 0 (a title hero image) AND "
        f"as many conceptual slides as add value, aiming for {PPTX_MAX_IMAGES} images total when the "
        "content allows. Only SKIP a slide if it is purely a numeric table, code, or a bare list of "
        "figures. For each chosen slide give a vivid English Stable Diffusion prompt with quality tags "
        "that visually represents that slide's topic. "
        'Output ONLY JSON: {"images":[{"index":int,"prompt":str}]}. No prose, no thinking.'
    )
    raw = ask(CONDUCTOR, [{"role": "system", "content": sys},
                          {"role": "user", "content": outline}],
              CONDUCTOR_TEMP, think=False, fmt=_PPTX_IMG_SCHEMA,
              num_predict=768, label="pptx-img-plan")
    j = extract_json(raw) or {}
    out = {}
    for it in (j.get("images") or []):
        try:
            idx = int(it.get("index"))
        except Exception:
            continue
        p = str(it.get("prompt") or "").strip()
        if p and idx not in out and 0 <= idx <= len(slides):
            out[idx] = p
        if len(out) >= PPTX_MAX_IMAGES:
            break
    return out


def build_pptx(question, answer, out_path=None):
    """MoA 回答をスライド化し、内容連動で画像を埋め込んだ .pptx を生成して Path を返す。
    python-pptx 不在時は .md にフォールバック。画像バックエンド不在時はテキストのみで生成。"""
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
    except ImportError:
        md = (Path(out_path).with_suffix(".md") if out_path
              else PPTX_OUT_DIR / f"fugu_{time.strftime('%Y%m%d_%H%M%S')}.md")
        md.parent.mkdir(parents=True, exist_ok=True)
        _save_as_markdown(md, question, answer, 0.0, "")
        print(f"   [PowerPoint には python-pptx が必要。代わりに保存: {md}]")
        return md

    if out_path is None:
        PPTX_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PPTX_OUT_DIR / f"fugu_{time.strftime('%Y%m%d_%H%M%S')}.pptx"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_slides = _parse_slides(answer) or [{"title": "概要", "bullets": [answer[:400]]}]
    # 箇条書きを PPTX_MAX_BULLETS 単位に分割
    slides = []
    for s in raw_slides:
        bullets = s["bullets"] or [""]
        for i in range(0, len(bullets), PPTX_MAX_BULLETS):
            slides.append({"title": s["title"] + ("（続き）" if i else ""),
                           "bullets": bullets[i:i + PPTX_MAX_BULLETS]})
    slides = slides[:PPTX_MAX_SLIDES]
    title = _deck_title(question, raw_slides)

    # 画像計画 → 生成
    imgs = {}
    if _detect_backend() is not None and IMAGE_BACKEND != "off":
        plan = plan_pptx_images(title, slides)
        plan.setdefault(0, None)  # タイトルには必ずヒーロー画像
        print(f"   [PPTX画像: {min(len(plan), PPTX_MAX_IMAGES)} 枚を生成します...]")
        for idx, pr in list(plan.items())[:PPTX_MAX_IMAGES]:
            if pr:
                path = generate_image(pr, "")
            else:
                base = title if idx == 0 else slides[idx - 1]["title"]
                p2, n2 = author_image_prompt(base)
                path = generate_image(p2, n2)
            if path:
                imgs[idx] = str(path)

    prs = Presentation()
    prs.slide_width = Inches(13.333)   # 16:9
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def add_textbox(slide, text, left, top, width, height, size, bold=False):
        tb = slide.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame
        tf.word_wrap = True
        first = True
        for line in (text if isinstance(text, list) else [text]):
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            run = p.add_run()
            run.text = ("• " + line) if isinstance(text, list) else line
            run.font.size = Pt(size)
            run.font.bold = bold
        return tb

    def add_image(slide, path, left, top, width):
        try:
            slide.shapes.add_picture(path, left, top, width=width)
        except Exception as e:
            print(f"   [PPTX画像埋込エラー: {e}]")

    # タイトルスライド
    s0 = prs.slides.add_slide(blank)
    if 0 in imgs:
        add_image(s0, imgs[0], Inches(4.17), Inches(2.5), Inches(5.0))
        add_textbox(s0, title, Inches(0.7), Inches(0.6), Inches(12.0), Inches(1.3), 40, True)
        add_textbox(s0, "Fugu MoA 生成", Inches(0.7), Inches(1.9), Inches(12.0), Inches(0.6), 18)
    else:
        add_textbox(s0, title, Inches(0.9), Inches(2.6), Inches(11.5), Inches(1.6), 44, True)
        add_textbox(s0, "Fugu MoA 生成", Inches(0.9), Inches(4.2), Inches(11.5), Inches(0.7), 20)

    # コンテンツスライド
    for i, s in enumerate(slides, start=1):
        sl = prs.slides.add_slide(blank)
        add_textbox(sl, s["title"], Inches(0.6), Inches(0.4), Inches(12.1), Inches(1.0), 30, True)
        has_img = i in imgs
        body_w = Inches(7.0) if has_img else Inches(12.1)
        add_textbox(sl, s["bullets"], Inches(0.6), Inches(1.6), body_w, Inches(5.4), 18)
        if has_img:
            add_image(sl, imgs[i], Inches(7.9), Inches(1.7), Inches(4.9))

    prs.save(str(out_path))
    return out_path


def _save_answer_to_file(question: str, answer: str, elapsed: float,
                         path: str, context: str = ""):
    """回答を --out で指定した拡張子に合わせた形式で保存する。
    .py/.js 等のコード拡張子 → コード抽出して書き出し
    .md/.txt → テキスト形式追記
    .pdf     → fpdf2 で生成（未インストール時 .md にフォールバック）
    .docx    → python-docx で生成（未インストール時 .md にフォールバック）
    .xlsx    → openpyxl で生成（未インストール時 .csv にフォールバック）
    .pptx    → python-pptx で画像入りスライド生成（未インストール時 .md にフォールバック）
    .html    → HTML で生成
    その他   → Markdown で保存
    """
    out = Path(path)
    suffix = out.suffix.lower()
    actual = out  # 実際に書かれたファイル（フォールバック時に変わる可能性あり）

    if suffix in _CODE_EXTENSIONS:
        _save_as_code(out, answer)
    elif suffix == ".txt":
        _save_as_text(out, question, answer, elapsed)
    elif suffix in {".pdf"}:
        result = _save_as_pdf(out, question, answer, elapsed)
        if result:
            actual = result
    elif suffix in {".docx", ".doc"}:
        result = _save_as_docx(out, question, answer, elapsed)
        if result:
            actual = result
    elif suffix in {".xlsx", ".xls"}:
        result = _save_as_excel(out, answer)
        if result:
            actual = result
    elif suffix in {".pptx", ".ppt"}:
        result = build_pptx(question, answer, out)
        if result:
            actual = result
    elif suffix in {".html", ".htm"}:
        _save_as_html(out, question, answer, elapsed)
    else:
        # .md またはその他 → Markdown
        _save_as_markdown(out, question, answer, elapsed, context)

    print(f"   [回答を保存しました: {actual}]")


def repl(use_search=False, rag_dirs=None, history_file=None):
    global _HISTORY
    hfile = history_file or HISTORY_FILE
    flags = []
    if use_search:
        flags.append("Web検索ON")
    if rag_dirs or RAG_DIRS:
        dirs = rag_dirs or RAG_DIRS
        flags.append(f"RAG:{','.join(str(d) for d in dirs)}")
    if flags:
        print(f"   [{', '.join(flags)}]")
    print("コマンド: 'exit'/'quit' で終了  'reset' で会話履歴クリア  "
          "'search on/off' で Web検索切替  'save <path>' で履歴エクスポート")
    while True:
        try:
            q = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break
        if not q:
            continue
        low = q.lower()
        if low in ("exit", "quit"):
            break
        if low == "reset":
            _HISTORY.clear()
            save_history_file(_HISTORY, path=hfile)
            print("   [会話履歴をクリアしました]")
            continue
        if low == "search on":
            use_search = True
            print("   [Web検索: ON]")
            continue
        if low == "search off":
            use_search = False
            print("   [Web検索: OFF]")
            continue
        if low.startswith("save "):
            save_path = q[5:].strip()
            save_history_file(_HISTORY, path=Path(save_path))
            print(f"   [履歴を保存しました: {save_path}]")
            continue
        ask_fugu(q, use_search=use_search, rag_dirs=rag_dirs,
                 history_file=hfile)


def main():
    parser = argparse.ArgumentParser(
        description="Local Fugu-style MoA オーケストレーター",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用例:
  python fugu_local.py                               # 対話モード
  python fugu_local.py "91は素数ですか？"             # 一問一答
  python fugu_local.py --file task.txt               # .txt から質問を読む
  python fugu_local.py --file spec.pdf               # PDF から読む（pdfplumber 要）
  python fugu_local.py --file task.py   --out fix.py  # Python コードを修正して .py で保存
  python fugu_local.py --file report.md --out out.md  # Markdown → Markdown
  python fugu_local.py --file spec.docx --out result.pdf  # Word → PDF（fpdf2 要）
  python fugu_local.py --search "最新のS&P500は？"    # Web検索あり
  python fugu_local.py --rag ./docs "PINNを実装して"  # RAGあり
  python fugu_local.py --no-history "一時的な質問"    # 履歴を使わない
  python fugu_local.py --session ./project.json       # プロジェクト専用履歴

対応ライブラリ（pip install で追加）:
  PDF読込: pdfplumber  PDF書出: fpdf2
  Word:    python-docx  Excel: openpyxl  PowerPoint: python-pptx""",
    )
    parser.add_argument("question", nargs="?",
                        help="質問文（省略時は対話モード）")
    parser.add_argument("--file", "-f", metavar="PATH",
                        help="質問をテキストファイルから読む")
    parser.add_argument("--out", "-o", metavar="PATH",
                        help="回答を保存するファイル（拡張子で形式自動選択: "
                             ".md/.txt/.py/.js/.pdf/.docx/.xlsx/.html 等）")
    parser.add_argument("--search", "-s", action="store_true",
                        help="Web 検索を有効化してコンテキストに注入する")
    parser.add_argument("--rag", "-r", nargs="+", metavar="DIR",
                        help="RAG 用ドキュメントディレクトリ（複数指定可）")
    parser.add_argument("--no-history", action="store_true",
                        help="セッション永続化を無効化（履歴を読まず保存もしない）")
    parser.add_argument("--session", metavar="PATH",
                        help=f"会話履歴ファイルパス（既定: {HISTORY_FILE}）")
    args = parser.parse_args()

    # --- セッション設定 ---
    global SESSION_SAVE, _HISTORY
    hfile = Path(args.session) if args.session else HISTORY_FILE
    if args.no_history:
        SESSION_SAVE = False
    else:
        _HISTORY = load_history_file(hfile)
        if _HISTORY:
            print(f"[session] 会話履歴を読み込みました: {len(_HISTORY) // 2} 往復 ({hfile})")

    # --- RAG ディレクトリ設定 ---
    rag_dirs = args.rag or (RAG_DIRS if RAG_DIRS else None)

    if not setup():
        return

    # --- 質問の取得 ---
    _OFFICE_SUFFIXES = {".docx", ".doc", ".xlsx", ".xls", ".pdf", ".pptx", ".ppt"}
    question = None
    office_attached = False
    if args.file:
        fp = Path(args.file)
        if not fp.exists():
            print(f"ファイルが見つかりません: {args.file}")
            return
        question = read_file_text(fp).strip()
        if not question:
            print(f"ファイルからテキストを抽出できませんでした: {args.file}")
            return
        office_attached = fp.suffix.lower() in _OFFICE_SUFFIXES
        print(f"[file] {fp.name} ({fp.suffix}) から {len(question)} 文字を読み込みました"
              + ("  [Office→Proposer C 主軸]" if office_attached else ""))
    elif args.question:
        question = args.question

    # --- 実行 ---
    if question:
        ask_fugu(question, use_search=args.search,
                 rag_dirs=rag_dirs, out_file=args.out, history_file=hfile,
                 office_attached=office_attached)
    elif sys.stdin.isatty():
        repl(use_search=args.search, rag_dirs=rag_dirs, history_file=hfile)
    else:
        # パイプ入力: stdin を質問として読む
        q = sys.stdin.read().strip()
        if q:
            ask_fugu(q, use_search=args.search,
                     rag_dirs=rag_dirs, out_file=args.out, history_file=hfile)
        else:
            print("質問が入力されませんでした。")
            parser.print_help()


if __name__ == "__main__":
    main()
