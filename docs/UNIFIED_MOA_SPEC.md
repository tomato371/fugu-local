# Quorum（旧 fugu-nim）統合 MoA — 実装プロンプト（単一実装・段階分割なし）

**このファイルが唯一の作業指示書である。** 段階（Phase）には分けない。
以下を **1 回の作業で全部実装し**、機能ごとのフラグで寄与を切り分けて計測する。

---

## 0. ゴール

2 つの改修を同時に入れる。

1. **経路分岐の廃止** — `task_type` による「math/mcq は投票、それ以外は合議」の分岐をなくし、
   単一の MoA ループへ統合する
2. **合成性の導入** — 問題を推論の段へ分解し、段ごとにプーリングする（因果畳み込み構造）

---

## 1. 厳守事項

1. **既存関数を書き換えない。** すべて追記で実装する（§5）
2. **既存フラグの既定値を変えない。** 新フラグはすべて既定 `False`
3. **既存 config を削除・リネームしない。** `sc@nim` 〜 `sc9@nim` / `fugu@nim` / `coder@nim` は残す
4. **結果 JSONL・ログを削除／上書きしない**
5. **git commit / push をしない。** 本人の `!` 実行を待つ
6. **悪い結果を報告しないことは禁止。** スコアが下がったらそのまま報告する

---

## 2. 設計の原理

### 2.1 なぜ分岐が要らないか

現行の 2 経路は**集約方法が違うだけ**である。

| 経路 | 集約 |
|---|---|
| `solve_verifiable`（math/mcq） | 同値クラスの argmax（ハードプーリング） |
| `fugu_answer` の MoA | Aggregator が統合（ソフトプーリング） |

そして**どちらを使うかは事前に予測する必要がない**。提案を生成してから、
比較可能な結論が抽出できたかを見ればよい。

> **原理: 入力の分類（early binding）ではなく、出力の性質（late binding）で集約方式を決める。**

現行の分岐には実害がある。`task_type` を誤ると投票エンジンを丸ごと失う一方、
その判定器は 6 ベンチのうち `fugu@nim` でしか動いていない
（他は `solve_verifiable(question, item["task_type"])` のようにデータセットのラベルを直接渡している）。
**最も高リスクな判定が、最も測られていない場所にある。**

### 2.2 なぜ合成性か

現行はすべてのサンプルが問題を丸ごと解くため、n 段の推論で誤りが乗算で効く。
各段 0.8 でも 5 段なら 0.8⁵ ≈ 0.33。全系統が同じ段で転べば票は揃って間違い、
票を増やしても直らない。実測でも `sc3`（幅 32）は 1 問救うのに 1,149 req を要した。

`run_sc7` の docstring より:

> 全 38 サンプル中に正解が一度も生成されない問題は集約では救えない

構造は **TCN と同型**にする — 因果性（段 k は段 1..k−1 のみ参照）、
重み共有（同一ソルバプールを全段に）、階層分解（dilation 相当）、残差結合。
軸が「時刻」ではなく「推論の段」に変わっただけである。

---

## 3. 実装するもの

すべて `fugu_local.py` に**追記**する。

### 3.0 全体フロー

```
fugu_solve(question, history)
  ├─ steps = decompose(question)            # 1 段しか取れなければ従来動作へ縮退
  ├─ for step in steps:                     # 因果ループ
  │     cands  = sample_step(step, question, ctx, n=STEP_VOTES_INIT)
  │     result = pool(question, cands)      # ★遅延束縛
  │     while not result.confirmed and 予算内:
  │         cands += sample_step(step, question, ctx, n=STEP_VOTES_STEP)
  │         result = pool(question, cands)  # 割れた段にだけ追加
  │     if not result.confirmed and FUGU_HIERARCHICAL:
  │         result = 再分解して再帰           # dilation・深さ上限 DECOMP_MAX_DEPTH
  │     ctx.append(result)
  ├─ final = compose(question, steps, ctx)
  └─ verify → NG かつ round < MAX_ROUNDS なら reference=final でもう一周（既存 MoA の再帰を維持）
```

**`if task_type == ...` を一切書かないこと。** 分解が 1 段しか返さない場合は
「丸ごと解く」に自然に縮退する。これは分岐ではなく**縮退**として実装する。

### 3.1 `extract_key(text, question=None)` — 統合の要

候補から「比較可能な結論」を取り出す。**これが `task_type` を置き換える。**

```python
def extract_key(text, question=None):
    """(kind, key) を返す。kind は "math" | "mcq" | "exec"。取れなければ None。"""
```

試行順序（**先に成功したものを採用**）:

1. **math** — 既存 `extract_final_answer(text, "math")`（`\boxed{}` / 最終解答宣言）
2. **mcq** — 既存 `extract_final_answer(text, "mcq")`（選択肢ラベル）
3. **exec** — `FUGU_EXEC_KEY=True` のとき、`extract_code(text)` でコードが取れたら
   **実行して振る舞いをキーにする**
4. どれも取れなければ `None`

**3 が新規かつ重要。** 実装の異なる 2 つのコードが同じ入力に同じ出力を返すなら等価である——
これは投票に必要な同値関係そのもの。**コード生成タスクにも投票が使えるようになる。**

exec キーの作り方:
- 問題文の docstring 例、または `question` から抽出できる入出力例に対して実行する
- 例が取れなければ、コードを AST 正規化した文字列をキーにする（弱いが無いよりよい）
- 出力ベクトルを結合し `sha1` 先頭 16 桁をキーにする（**振る舞い等価性ハッシュ**）
- 実行は既存 `run_python(code, timeout=POOL_EXEC_TIMEOUT, stdout_only=True)` を使う

### 3.2 `pool(question, candidates)` — 遅延束縛プーリング

```python
class PoolResult:
    answer: str            # 採用された答え（提示用テキスト）
    key: str | None        # 比較キー（ハード時のみ）
    kind: str              # "hard" | "soft"
    agreement: float|None  # 勝者票数 / 有効票数
    n_valid: int
    n_total: int
    confirmed: bool
    classes: list          # [[key, votes], ...] 降順
```

```python
def pool(question, candidates):
    keys  = [extract_key(c, question) for c in candidates]
    valid = [(k, c) for k, c in zip(keys, candidates) if k is not None]
    if len(valid) >= POOL_MIN_KEYS:
        # ハードプーリング — 既存 vote_answers / answers_equivalent を再利用
        ...
    else:
        # ソフトプーリング — 既存 aggregate を再利用
        ...
```

**確定条件はハードプーリングで既存 SC と完全に同一にする**
（全会一致 ∧ n≥`SC_MIN_VOTES`、または n≥4 ∧ 過半数）。比較可能性のためここは変えない。

`kind` が混在した場合（例: math キー 3 本 + exec キー 2 本）は
**多数派の kind だけを有効票とする**。異種キーを同一空間で投票させない。

ソフトプーリングは 1 本しか出ないため票の概念がなく、`confirmed=True` として扱う。

### 3.3 `decompose(question, parent=None, depth=0)` — 段分解

```python
def decompose(question, parent=None, depth=0) -> list[dict]:
    """推論の段を返す。段が取れなければ [{"goal": question}] を返す（縮退）。"""
```

- 既存 `_enumerate_strategies` と**同型**に実装する。強モデル
  （`ARBITER_MODEL` → 無ければ `REASONING_MODELS[0]`）に JSON スキーマ拘束で段の列を出させる
- **分解自体を `DECOMP_VOTES`（既定 3）本引いて投票する。** 段数が最頻の分解を採用する。
  分解の誤りが全段を巻き添えにするリスクへの主要な緩和策
- `depth >= DECOMP_MAX_DEPTH` なら分解せず `[{"goal": question}]` を返す
- 得られた段が 2 未満なら `[{"goal": question}]` を返す ← **これが分岐なしの縮退**
- `parent` が与えられた場合は親の文脈を含めて再分解する（階層分解）

スキーマ:
```json
{"steps": [{"goal": "この段で確定させること", "depends_on": [0, 1]}]}
```

`depends_on` は因果性の明示。今回は逐次実行でよい（並列化は将来の余地）。

### 3.4 `sample_step(step, question, ctx, n, models=None)` — 段ソルバ

```python
def sample_step(step, question, ctx, n, models=None) -> list[str]:
    """1 段を n 本サンプリング。models 省略時は REASONING_MODELS（＝重み共有）。"""
```

プロンプト構成（**残差結合を含む**）:

```
[system] 既存 SC_PROMPT_MATH を汎用化したもの（math 前提の文言を外す）
[user]
  ## 元の問題                    ← ★残差結合（FUGU_RESIDUAL）
  {question}

  ## ここまでに確定した内容        ← 因果性: 段 1..k-1 のみ。k 以降を漏らさないこと
  {ctx を整形}

  ## 今回のあなたの担当
  {step["goal"]}
  この段で確定させるべきことだけを出力し、最後に結論を 1 行で示すこと。
```

- 並列化は既存 `SC_PARALLEL` / `SC_WORKERS` と同じ `ThreadPoolExecutor` を使い、
  **回収は投入順に固定**する（決定性の維持。既存 `solve_verifiable` と同じ作法）
- 既存 `SC_STRATIFY`（S3C 戦略層化）が有効なら段内でも適用する

### 3.5 `compose(question, steps, ctx)` — 合成

確定した段の結論を統合して最終解答を作る。`aggregate` を流用してよいが、
プロンプトには**原問題と全段の結論**を渡す（残差結合）。
合成後に `extract_key` でキーが取れれば、それを最終キーとする。

### 3.6 `verify(question, answer, ctx)` — 検証

既存の検証機構を**全経路で**使えるようにする。

- `critique(question, answer)` — Critic
- `second_opinion(question, answer)` — 別系譜チェック
- **実行検証** — `answer` にコードが含まれれば `code_check(answer)` を通す
  （現状 PoT は SC 経路にしか繋がっていない。**これを全経路へ配線する**）

### 3.7 `fugu_solve(question, history=None, plan=None)` — 新しい入口

上記を束ねる。既存 `fugu_answer` は**そのまま残す**（比較用）。

早期終了（`single` モードの置き換え・`FUGU_EARLY_UNANIMOUS`）:
- 第 1 バッチで**全候補のキーが一致**したら、その段を即確定する
- 段が 1 つしかなく、かつ全会一致ならその場で返す
- **入力の分類ではなく出力の性質**なので、分岐を持ち込まずに済む

---

## 4. フラグ

```python
# ---- 統合 MoA ----
FUGU_UNIFIED          = False   # fugu_solve 経路を使う（マスタスイッチ）
FUGU_DECOMPOSE        = True    # 段分解（UNIFIED 内で有効）
FUGU_HIERARCHICAL     = True    # 割れた段の再分解（dilation）
FUGU_RESIDUAL         = True    # 原問題を全層へ直結
FUGU_EXEC_KEY         = True    # 実行等価性によるキー抽出
FUGU_EARLY_UNANIMOUS  = True    # 全会一致での早期確定

# ---- パラメータ ----
POOL_MIN_KEYS         = 3       # ハードプーリングに必要な有効キー数
POOL_EXEC_TIMEOUT     = 20      # exec キー生成の実行タイムアウト秒
DECOMP_VOTES          = 3       # 分解自体のサンプル数
DECOMP_MAX_STEPS      = 8       # 段数の上限
DECOMP_MAX_DEPTH      = 2       # 階層再分解の深さ上限
STEP_VOTES_INIT       = 4       # 1 段あたり初期票数
STEP_VOTES_STEP       = 2       # 追加票の単位
STEP_VOTES_MAX        = 12      # 1 段あたり上限
STEP_BEAM             = 2       # 確定できない段で保持する候補数
```

**`FUGU_UNIFIED=False` のとき、既存の `solve_verifiable` / `fugu_answer` の挙動が
1 バイトも変わらないこと。** これは完了条件に含まれる。

---

## 5. 既存コードとの関係

### 触らない（読み取り・再利用のみ）

`ask` / `_ask_nim` / `apply_nim_profile` / `vote_answers` / `answers_equivalent` /
`extract_final_answer` / `extract_code` / `run_python` / `code_check` / `aggregate` /
`get_proposals` / `critique` / `second_opinion` / `verify_single` / `conduct` /
`solve_verifiable` / `fugu_answer` / 既存 `SC_*` 定数

### 追加のみ

- §3 の新関数
- §4 のフラグ
- `bench_fugu.py` の `CONFIGS` への新エントリ（§6）

---

## 6. bench_fugu.py への追加

各 runner は既存 `run_sc3` / `run_sc6` / `run_sc7` と同じく
**フラグを保存 → 設定 → `finally` で復元**する作法に従うこと。

| config 名 | 設定 | 目的 |
|---|---|---|
| `u@nim` | UNIFIED + 全機能 ON | 本命 |
| `u-nodecomp@nim` | DECOMPOSE=False | 段分解の寄与 |
| `u-nohier@nim` | HIERARCHICAL=False | 階層分解の寄与 |
| `u-nores@nim` | RESIDUAL=False | 残差結合の寄与 |
| `u-noexec@nim` | EXEC_KEY=False | 実行等価キーの寄与 |
| `u-noearly@nim` | EARLY_UNANIMOUS=False | 早期確定の寄与（主に速度） |

---

## 7. オフラインテスト（API 不要・必須）

`test_nim_offline.py` に倣い、モデル呼び出しをモックして検証する。
**既存の 51 件を壊さないこと。**

1. `extract_key` — math / mcq / exec の各パスが発火し、優先順位が守られる
2. `extract_key` — キーが取れない入力で `None` を返す
3. `pool` — 有効キー ≥ `POOL_MIN_KEYS` でハード、未満でソフトに落ちる
4. `pool` — 確定条件が既存 SC と一致する（全会一致∧n≥3 / n≥4∧過半数）
5. `pool` — kind 混在時に多数派 kind のみが有効票になる
6. `decompose` — 段が 2 未満のとき縮退する
7. `decompose` — `depth >= DECOMP_MAX_DEPTH` で分解しない
8. `decompose` — 分解投票で段数が最頻のものを選ぶ
9. `sample_step` — `FUGU_RESIDUAL=True` でプロンプトに原問題が含まれる
10. `sample_step` — ctx に段 k 以降が漏れていない（因果性）
11. 並列時に回収順が投入順で決定的
12. exec キー — 実装の異なる 2 コードが同じ出力なら同一キーになる
13. **`FUGU_UNIFIED=False` のとき既存経路が一切変わらない**（最重要）

---

## 8. 計測とアブレーション

実装完了後に **1 回だけ**回す。段階を分けない代わりに、**寄与の切り分けはここで行う。**

### 8.1 本測定（フル）

```
u@nim × {aime24, aime25, aime26}      … 90 問
u@nim × math500  --limit 50
u@nim × humaneval --limit 30
u@nim × jmmlu    --limit 40
```

比較対象: `sc@nim`（AIME / math500）、`coder@nim`（humaneval）、`fugu@nim`（jmmlu）

### 8.2 アブレーション（診断サブセット 28 問）

**全 90 問は回さない。** `--ids` で以下に絞る。

- `sc@nim` で失敗した 15 問
- 未解決 3 問（`aime24-2024-I-12` / `aime24-2024-II-15` / `aime25-13`）
- **回帰検出用**に `sc@nim` で 40–60 req で正解した 10 問

```
{u-nodecomp, u-nohier, u-nores, u-noexec, u-noearly}@nim × 上記 28 問
```

### 8.3 コスト管理

推定: 本測定 約 5,000 req ＋ アブレーション 約 4,200 req ＝ **約 9,200 req**。
累計が 10,407 req なので、およそ倍になる。
`nim_usage.json` で消費を監視し、必要なら `FUGU_NIM_BUDGET` で上限を設定すること
（到達時 `SystemExit(42)` で安全停止する）。

### 8.4 報告

`fugu_bench/analysis/UNIFIED_RESULTS.md` に以下を含める。

- 全ベンチの `u@nim` vs 既存構成のスコア表
- **req/問** と **req/救済1問**（精度だけでなく効率を必ず併記）
- アブレーション表 — 各機能を切ったときのスコア差と req 差
- **回帰リスト** — 既存構成で正解していたのに `u@nim` で落とした問題を全件
- 想定外の所見

---

## 9. 完了条件

1. `FUGU_UNIFIED=False` で既存の全 config が**ビット同一の挙動**を示す
2. オフラインテストが全て緑（既存 51 件 ＋ 新規 13 件）
3. `u@nim` が全 4 ベンチで完走し、結果 JSONL が生成されている
4. アブレーション 5 構成が診断サブセット 28 問で完走している
5. 報告に**回帰リストが含まれている**

**スコアが既存を上回ることは完了条件に含めない。**
下回った場合は、その事実と原因の分析を報告することが完了条件である。
DeepConf を「異種混成では有害」と記録して構成ごと残したのと同じ扱いにする。

---

## 10. 禁止事項

- ❌ 既存関数の書き換え（`solve_verifiable` / `fugu_answer` / `vote_answers` 等）
- ❌ 既存フラグの既定値変更
- ❌ 既存 config の削除・リネーム
- ❌ 結果 JSONL / ログの削除・上書き
- ❌ `if task_type == ...` による経路分岐を新経路に持ち込む
- ❌ 悪い結果を報告しない、または良く見えるよう構成を後から調整する
- ❌ git commit / push

---

## 付録 A — 事前診断（API 呼び出しゼロ・任意・実装と並行可）

**実装をブロックしない。** ただし分解器（§3.3）のプロンプト設計に直接効くため、
先に、または並行して回しておくと精度が上がる可能性がある。

### A.1 目的

既存の失敗トレースを分類し、**誤りが段に局在しているか**を確認する。
局在していれば段別プーリングが効く見込みが立ち、
していなければ分解の粒度を粗く取るべき、という設計判断の材料になる。

### A.2 データ

**結果 JSONL** `fugu_bench/results/<dataset>__<config>.jsonl`
→ `id` / `correct` / `got` / `expected` / `seconds` / `nim_requests` / `answer_text`（代表トレース全文）

**実行ログ** `fugu_bench/logs/<dataset>__<config>.log`
→ **全サンプルの投票結果が残っている。** 書式（実測済み・逐語）:

```
=== [2/30] aime26-15 (aime26/sc@nim) ===
   [SC 1] deepseek-ai/deepseek-v4-pro (CoT) -> 1
   [SC 3] deepseek-ai/deepseek-v4-pro (CoT) -> 14
   [SC 4] openai/gpt-oss-120b (CoT) -> (抽出失敗)
   [SC] 確定: 1  (票 3/4, サンプル計 7)
    -> NG got=1 expected=83 (165.7s)  nim_req=7 (累計 14)
```

パース対象:

| パターン | 抽出するもの |
|---|---|
| `^=== \[\d+/\d+\] (\S+) \(` | 問題 ID |
| `^\s+\[SC (\d+)\] (\S+) \((CoT\|PoT)\) -> (.+)$` | サンプル番号 / モデル / 種別 / 答え |
| `^\s+\[SC\] 票が割れています (\[.*\]) → 追加サンプリング` | その時点の票分布 |
| `^\s+\[SC\] 確定: (.+?)\s+\(票 (\d+)/(\d+), サンプル計 (\d+)\)` | 確定値と票数 |
| `^\s+-> (OK\|NG) got=(.*?) expected=(.*?) \(([\d.]+)s\)\s+nim_req=(\d+)` | 判定・時間・req 数 |

### A.3 分類

**一次分類（ログから機械的に決まる）**

| ラベル | 条件 | 含意 |
|---|---|---|
| `D_near_miss` | 正解が票の中に出現したが多数決で負けた | 集約の改善で救える |
| `E_absent` | **正解が全サンプルのどこにも出現しない** | 集約では原理的に救えない。生成側の問題 |
| `C_vote_starved` | 抽出失敗 ≥ 有効票、または有効票 < 3 | 系統が実質的に減っている |

**二次分類（`E_absent` のみ・トレース精読）**

| ラベル | 条件 | 分解への含意 |
|---|---|---|
| `E1_same_step` | 複数トレースが**同じ段**で同じ誤り | 段を細かく切る価値が高い |
| `E2_diff_step` | トレースごとに違う段で転ぶ | 各段の精度自体が低い。票数を厚めに |
| `E3_diff_approach` | 解法の選び方が分かれ段の対応が取れない | 分解より S3C 戦略層化を優先 |
| `E4_no_steps` | 段に分解できない | 分解を縮退させる（`[question]`） |

### A.4 併記すべき所見

- **系統の実質的欠落** — 特定モデルの抽出失敗の偏り。
  例: `aime26-15` と `aime26-11` では `openai/gpt-oss-120b` が 3/3 抽出失敗しており、
  「3 系統 SC」が実質「deepseek 1 系統 + PoT」で回っていた可能性がある。
  **これが広範なら、実装より先に直すべきはこちらである。**
- **PoT の寄与** — PoT 票が正解と一致した / 誤答に加担した頻度
- **追加サンプリングの効果** — 割れた後に分布が改善したか、同じ誤答が積み増されただけか

### A.5 出力

`fugu_bench/analysis/failure_classification.json`（1 問 1 レコード）と
`fugu_bench/analysis/FAILURE_ANALYSIS.md`（集計と所見）。

分類には**必ずログまたはトレースからの逐語引用を添える**。
判断がつかないものは `unknown` とし、無理に分類しない。

### A.6 この診断での禁止事項

- ❌ LLM API を呼んで問題を解き直す（この診断は完全にオフライン）
- ❌ results / logs / *.py を変更する
- ❌ 根拠引用なしに分類ラベルを付ける
