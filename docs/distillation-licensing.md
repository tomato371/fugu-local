# 蒸留データの法的条件 — NIM 経由で得たトレースを学習に使えるか

調査日 **2026-08-20**。対象は `FUGU_TRACE_DIR` が保存する SC トレース(= NVIDIA NIM 無料枠
経由で得たモデル出力)を、ファインチューニング/蒸留の教師データに使う場合の条件。

**これは法律助言ではない。** 一次文書のどこに何が書いてあるかの整理であり、商用化の判断は
弁護士に相談すること。規約は改訂されるので、実際に学習を回す前に日付とバージョンを確認する。

---

## 0. 結論(先に読む)

| やりたいこと | 現状の可否 | 効いている条項 |
|---|---|---|
| ベンチマークで測る・評価する | ✅ 問題なし | Trial ToS §1.4「内部のテストと評価」そのもの |
| 手元でトレースを溜める | ✅ 問題なし | 出力の所有権は利用者側 (§6.3) |
| トレースで手元のモデルを微調整し、自分で使う | ⚠️ 条件つき | §4.12(競合品の開発)と各モデルライセンス |
| **データセットを公開・配布する** | ❌ **無料枠では不可** | §4.2「Generated Content を他者に配布しない」 |
| 微調整したモデルをサービスとして提供する | ❌ 無料枠では不可 / 要 Subscription | §1.2「本番利用不可」+ §4.12 |

**最重要**: 障害は「蒸留禁止条項」ではない。NIM 無料枠の規約が **(a) 出力の再配布を禁じ、
(b) 本番利用を禁じ、(c) API と競合する製品の開発に出力を使うことを禁じている**点にある。
モデル側のライセンス(下記2)はむしろ寛容で、蒸留そのものを禁じているものは1つもない。

**回避策は「経路を変える」こと。** 同じモデルの多くはオープンウェイトなので、
**ローカル(Ollama / vLLM)で生成したトレース**なら NVIDIA の API 規約は一切かからず、
モデルライセンスだけを守ればよい。公開データセットを作るつもりなら、この経路一択。

---

## 1. NVIDIA API Trial Terms of Service(すべてのトレースがここを通る)

出典: [NVIDIA API TRIAL TERMS OF SERVICE](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf)(PDF・全9頁を直接確認)

| 条項 | 内容 | この計画への影響 |
|---|---|---|
| §1.2 / §1.4 | トライアル利用は「限定的なトライアル目的のみ」、Generated Content を本番で使わない。Subscription を買わない限り **internal testing and evaluation purposes** に限る | ベンチ測定は完全に想定内。**微調整済みモデルの運用は範囲外** |
| §2.2 | 入出力に対し、セッション中サービス提供のため NVIDIA へ利用許諾を与える | 保存済みトレースの手元利用には影響しない |
| §3.3 | NVIDIA は入出力を **自社製品(AI モデルを含む)の改善に利用しうる** | 秘匿したい問題文を投げない判断材料 |
| **§4.2** | **Generated Content の一部たりとも他者へ複製・販売・再配布・利用可能化しない** | **トレース/SFT データセットの公開は不可**。HF へ上げる等は明確に対象 |
| **§4.12** | API Service(Generated Content を含む)を、**当該 API Service と競合する製品・サービスの開発や改善に使わない** | 「蒸留したモデルを推論/オーケストレーション API として出す」が最も危ない。個人研究とは別問題 |
| §6.3 | 付随するモデルライセンスに別段の定めがない限り、**Generated Content は利用者のもの** | 出力の所有権自体は問題にならない |

**存在しない条項**: 「出力を学習に使ってはならない」という一般的な禁止は**無い**。
(この点、二次情報には「clause 2.D で明示的に禁止」等の記述が流通しているが、
実際の PDF に該当条項も該当文言も存在しない。原文に当たること。)

---

## 2. モデル側のライセンス(§6.3 経由で効く)

現在の布陣(2026-08 実測: `nvidia/nemotron-3-ultra-550b-a55b`, `openai/gpt-oss-120b`,
`minimaxai/minimax-m3`、指揮者 `google/gemma-4-31b-it`)。

### nvidia/nemotron-3-ultra — NVIDIA Open Model License ✅ 最も寛容
- 合成データ生成による **Derivative Model の作成・配布を明示的に許可**。
- 出力に NVIDIA は所有権を主張しない。
- 義務: Trustworthy AI terms の遵守、ガードレールの回避禁止。
- 出典: [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license)

### openai/gpt-oss-120b — Apache 2.0 ✅ 寛容
- Apache 2.0 は出力に条件を課さない。別途 [usage policy](https://huggingface.co/openai/gpt-oss-120b/blob/main/USAGE_POLICY) の遵守のみ。

### minimaxai/minimax-m3 — MiniMax Community License ⚠️ 商用に条件
- 商用利用時は **「Built with MiniMax M3」の明示**が必要(サイト/UI/ドキュメント等)。
- 年商 **2,000万USD 超**の製品は**事前の書面許諾**、それ未満は一度の通知。
- post-training / fine-tuning 済みの提供も「商用利用」の定義に含まれる = 蒸留モデルを商用に出すなら上記が発動。
- 出典: [LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/LICENSE)

### google/gemma-4-31b-it — Gemma Terms of Use ⚠️ 義務が伝播する
- Google は**出力に権利を主張しない**が、**出力を使って学習したモデルは "Model Derivative"** と定義される。
- Model Derivative には Prohibited Use Policy と**使用制限の告知義務が伝播**し、配布時は規約の写しを同梱する必要がある。
- 出典: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

> **運用上の帰結**: 指揮者(gemma-4)の出力を学習データに混ぜると、
> 蒸留モデル全体に Gemma の条項が伝播する。`_sc_trace_dump` が保存するのは
> **SC プロポーザのサンプルだけ**で指揮者の計画 JSON は入らないため、既定では混ざらない。
> 意図的に指揮者の出力を足すときだけ、この伝播を引き受ける判断になる。

`traces_to_sft.py --exclude-model` で系統ごと機械的に落とせる。例:

```bash
python traces_to_sft.py --out sft.jsonl --exclude-model "gemma|minimax"
```

---

## 3. Sakana AI の TRINITY / Conductor(2026-08-20 追記)

- 論文はどちらも公開・ICLR 2026 採択:
  [TRINITY: An Evolved LLM Coordinator](https://arxiv.org/abs/2512.04695) /
  [Learning to Orchestrate Agents in Natural Language with the Conductor](https://arxiv.org/abs/2512.04388)。
- 一方 **Sakana Fugu 製品(重みとルーティング実装)は非公開**。[SakanaAI/fugu](https://github.com/SakanaAI/fugu)
  はコンフィグ・デモ・ドキュメントとテクニカルレポートで、学習済み係数は含まれない。
- したがって本リポジトリの実装は **論文記載の機構の再実装**であり、Sakana の成果物を
  取り込んだものではない(重みもコードも使っていない)。論文のアイデアの利用は
  学術的な引用の作法に従う。

---

## 4. 次にやるべきこと

1. **公開を視野に入れるならローカル生成に切り替える。** Ollama/vLLM で同じ布陣を回せば
   §4.2・§4.12 は無関係になり、モデルライセンスだけの世界になる(nemotron と gpt-oss は
   その時点でほぼ無条件)。8GB VRAM では 550B は動かないので、ここが DGX Spark の是非に直結する。
2. **商用化を考える時点で弁護士へ。** 特に §4.12「競合サービス」の解釈は、
   オーケストレーション API を出す構想と正面から当たる。
3. **規約の日付を記録する。** 本書は 2026-08-20 時点。学習を回す直前に再確認すること。
