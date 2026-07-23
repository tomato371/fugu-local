"""モデル呼び出しなしの高速回帰テスト。Ollama 不要・数秒で完走する。
実行: python test_fugu_offline.py
fugu_local / eval_fugu の純粋ロジック（プラン検証・JSON抽出・思考除去・言語判定・
アグリゲータのフォールバック・採点関数）を合成入力で検証する。
"""
import contextlib
import io
import json
import sys
import types
import urllib.error
import urllib.request

import fugu_local as f
import eval_fugu as e
import bench_queue as bq

_FAILS = []


def check(name, cond):
    print(f"[{'OK' if cond else 'NG'}] {name}")
    if not cond:
        _FAILS.append(name)


# ---------- extract_json ----------
check("json: 素のJSON", f.extract_json('{"a": 1}') == {"a": 1})
check("json: コードフェンス", f.extract_json('x\n```json\n{"a": 1}\n```\ny') == {"a": 1})
check("json: think混入", f.extract_json('<think>ignore {"b":9}</think>{"a": 1}') == {"a": 1})
check("json: 地の文に埋没", f.extract_json('The plan is {"a": 1} as follows') == {"a": 1})
check("json: 抽出不能はNone", f.extract_json("no json here") is None)
check("json: 空はNone", f.extract_json("") is None)
# 2026-07-22: 貪欲re.searchの over-capture 回帰防止（先頭オブジェクト消失バグ）
check("json: 末尾に余分な波括弧があっても先頭を抽出",
      f.extract_json('Sure! {"mode":"single"} note {x}') == {"mode": "single"})
check("json: 先行する集合記法{1,2,3}に惑わされない",
      f.extract_json('The set {1,2,3} then {"a": 1}') == {"a": 1})
check("json: 2つ目の有効オブジェクトがあっても最初を返す",
      f.extract_json('{"a": 1} and also {"b": 2}') == {"a": 1})
check("json: 文字列値中の}に惑わされない(深さカウントの文字列認識)",
      f.extract_json('x {"s": "a}b", "n": 2} y') == {"s": "a}b", "n": 2})
check("json: 閉じない{単体はクラッシュせずNone",
      f.extract_json("prefix { unbalanced no closing brace") is None)

# ---------- strip_think ----------
check("strip: think除去", f.strip_think("<think>x</think>answer") == "answer")
check("strip: thinking除去", f.strip_think("<THINKING>x</THINKING>ans") == "ans")
check("strip: 対象なしは素通し", f.strip_think("plain") == "plain")
check("strip: None耐性", f.strip_think(None) is None)
# 2026-07-22: num_predict 打ち切りで閉じタグの無い '<think>...' が丸ごと
# 「回答」として漏れる既知の失敗モード（gotcha #2 / #7）の回帰防止。
check("strip: 閉じタグ無しのthinkは末尾まで丸ごと除去",
      f.strip_think("<think>Let me compute... maybe 17... no, 42").strip() == "")
check("strip: 開始タグより前のテキストは保持",
      f.strip_think("answer is 5 <think>double-checking then cut off") == "answer is 5")
check("strip: 閉じタグ無しのTHINKINGも大小文字問わず除去",
      f.strip_think("<THINKING>still going with no closer").strip() == "")
check("strip: 対応の取れた既存ペアは従来通り除去(回帰)",
      f.strip_think("<think>x</think>answer") == "answer")
check("strip: 孤立した</think>閉じタグ単体は無視される",
      f.strip_think("no opener here </think> tail") == "no opener here </think> tail")
check("final_answer: 打ち切りthinkの中間値を誤って投票しない(E2E)",
      f.extract_final_answer(
          "<think>...intermediate value 17 then output cut off", "math") is None)

# ---------- validate_plan（新スキーマ: mode single|moa / selected_proposers 他） ----------
f.PROPOSERS = ["qwen3:4b", "phi4-mini", "gemma4:e2b-it-qat"]
f.AGGREGATOR = "deepseek-r1:7b"
f.CONDUCTOR = "qwen3:4b"

p = f.validate_plan({"mode": "moa",
                     "selected_proposers": ["qwen3:4b", "存在しないモデル"],
                     "rounds": 99, "use_image_generation": False,
                     "search_required": True})
check("plan: rounds を MAX_ROUNDS に丸める", p["rounds"] == f.MAX_ROUNDS)
check("plan: 未知プロポーザーを除外", p["selected_proposers"] == ["qwen3:4b"])
check("plan: search_required を反映", p["search_required"] is True)
check("plan: 不正mode は moa",
      f.validate_plan({"mode": "weird", "selected_proposers": ["qwen3:4b"]})["mode"] == "moa")
check("plan: dict以外はフォールバック", f.validate_plan(None).get("_fallback") is True)
check("plan: single は先頭1体のみ",
      f.validate_plan({"mode": "single",
                       "selected_proposers": ["phi4-mini", "qwen3:4b"]})["selected_proposers"]
      == ["phi4-mini"])
check("plan: 空selected は既定へフォールバック",
      len(f.validate_plan({"mode": "moa", "selected_proposers": []})["selected_proposers"]) >= 1)
check("plan: use_image_generation は mode を強制しない(非排他)",
      f.validate_plan({"mode": "moa", "selected_proposers": ["qwen3:4b"],
                       "use_image_generation": True})["mode"] == "moa")
check("plan: 画像生成フラグを反映",
      f.validate_plan({"use_image_generation": True,
                       "selected_proposers": ["qwen3:4b"]})["use_image_generation"] is True)
check("plan: image_only を反映(use_image_generation と整合時)",
      f.validate_plan({"image_only": True, "use_image_generation": True,
                       "selected_proposers": []})["image_only"] is True)
check("plan: make_pptx を反映し image_only を無効化",
      (lambda p: p["make_pptx"] is True and p["image_only"] is False)(
          f.validate_plan({"make_pptx": True, "image_only": True,
                           "selected_proposers": ["qwen3:4b"]})))
check("plan: use_image_generation=False なら image_only を強制的に無効化(矛盾解消)",
      f.validate_plan({"image_only": True, "use_image_generation": False,
                       "selected_proposers": []})["image_only"] is False)
check("plan: 矛盾したmath plan で SC投票ゲートの3フラグが全てFalseになる",
      (lambda p: p["image_only"] is False and p["make_pptx"] is False
       and p["use_image_generation"] is False)(
          f.validate_plan({"task_type": "math", "image_only": True,
                           "use_image_generation": False,
                           "selected_proposers": []})))
check("plan: 矛盾解消後は selected_proposers が通常moa分岐(image panelでない)",
      len(f.validate_plan({"task_type": "math", "image_only": True,
                           "use_image_generation": False,
                           "selected_proposers": []})["selected_proposers"]) >= 2)

# ---------- ペルソナ解決（selected_proposers のペルソナ名→実モデル） ----------
_op_persona = f.PROPOSERS
f.PROPOSERS = ["gpt-oss:20b", "qwen3-coder:30b", "gemma4:26b", "qwen3.6:35b"]
check("persona: 'Proposer A' → gpt-oss", f._resolve_proposer("Proposer A") == "gpt-oss:20b")
check("persona: 緩い 'a' → gpt-oss", f._resolve_proposer("a") == "gpt-oss:20b")
check("persona: モデル名直指定を許容", f._resolve_proposer("qwen3.6:35b") == "qwen3.6:35b")
check("persona: 未知は None", f._resolve_proposer("Proposer Z") is None)
check("persona: validate がペルソナ名を実モデルへ解決",
      f.validate_plan({"mode": "moa",
                       "selected_proposers": ["Proposer C", "Proposer D"]})["selected_proposers"]
      == ["gemma4:26b", "qwen3.6:35b"])
f.PROPOSERS = ["gpt-oss:20b", "qwen3-coder:30b", "phi4"]  # gemma4:26b 未導入シナリオ
check("persona: 未導入モデルのペルソナは None", f._resolve_proposer("Proposer C") is None)
check("persona: validate は未導入ペルソナを除外",
      f.validate_plan({"mode": "moa",
                       "selected_proposers": ["Proposer C", "Proposer B"]})["selected_proposers"]
      == ["qwen3-coder:30b"])

# ---------- 精度ガードレール（code/proof を single→moa へ格上げ） ----------
f.PROPOSERS = ["gpt-oss:20b", "qwen3-coder:30b", "gemma4:26b", "phi4"]


def _single_plan():
    return {"mode": "single", "selected_proposers": ["gpt-oss:20b"], "rounds": 1,
            "use_image_generation": False, "search_required": False,
            "reason": "r", "_fallback": False}


check("guard: コード質問は moa へ格上げ",
      f._apply_accuracy_guardrails("Pythonで実装して", _single_plan())["mode"] == "moa")
check("guard: 証明質問は moa へ格上げ",
      f._apply_accuracy_guardrails("背理法で証明せよ", _single_plan())["mode"] == "moa")
check("guard: 格上げ時は複数体を割当",
      len(f._apply_accuracy_guardrails("コードを書いて", _single_plan())["selected_proposers"]) >= 2)
check("guard: 平易な質問は single のまま",
      f._apply_accuracy_guardrails("日本の首都は？", _single_plan())["mode"] == "single")
check("guard: image_only は格上げ対象外",
      f._apply_accuracy_guardrails(
          "コードを実装して",
          {"mode": "single", "selected_proposers": [],
           "image_only": True})["mode"] == "single")
check("guard: イラスト付き(image_only=False)のコードは格上げ",
      f._apply_accuracy_guardrails(
          "コードを実装して",
          {"mode": "single", "selected_proposers": ["gpt-oss:20b"],
           "use_image_generation": True, "image_only": False})["mode"] == "moa")

# ---------- スライド分解（PowerPoint 用） ----------
_slides = f._parse_slides("## 概要\n- 要点1\n- 要点2\n\n## 詳細\n本文の段落です。\n1. 手順A\n2. 手順B")
check("pptx: 見出しでスライド分割", len(_slides) == 2)
check("pptx: 箇条書き記号を除去", _slides[0]["bullets"] == ["要点1", "要点2"])
check("pptx: タイトルは見出し由来", _slides[1]["title"] == "詳細")
check("pptx: 見出し無しは概要1枚",
      len(f._parse_slides("ただの文章その1。\nその2。")) == 1)

# 2026-07-23: 未対応(奇数個)の```フェンスが in_code を文書末尾まで固定し、
# 以降の見出しが新規スライドを作れず literal な "## ..." 箇条書きとして
# 1枚に押し込まれてしまう回帰の防止（extract_boxed iter11 / strip_think
# iter16 / _save_as_html iter37 と同種のバグクラス）。

# 回帰: 対になったコードブロック内の '# コメント' は見出しに昇格せず、
# コード本文も強調記号除去されない（従来通り）。
_bal_code = f._parse_slides(
    "## セクション1\n```python\n# これはコメント\nx = 1\n```\n## セクション2\n本文")
check("pptx回帰: 対になったコードフェンス内の#コメントは見出しに昇格しない",
      len(_bal_code) == 2
      and _bal_code[0]["title"] == "セクション1"
      and "# これはコメント" in _bal_code[0]["bullets"]
      and "x = 1" in _bal_code[0]["bullets"]
      and _bal_code[1]["title"] == "セクション2")

# 回帰: 対になったコードブロック2個(フェンス4本)+見出し混在は従来通り分割される。
_bal_two = f._parse_slides(
    "## A\n```python\n# a\n```\n## B\n```python\n# b\n```\n## C\n末尾の文")
check("pptx回帰: 対になったコードブロック2個でも見出し分割は不変",
      [s["title"] for s in _bal_two] == ["A", "B", "C"]
      and "# a" in _bal_two[0]["bullets"] and "# b" in _bal_two[1]["bullets"])

# 新規: 閉じ忘れの奇数フェンスの後に来る見出しは、literalな"## ..."箇条書きに
# ならず、新規スライドの見出しとして正しく分割される。
# (フェンス直後の本文行はあえて '#' で始めない。中断されたコードの地の文が
#  たまたま '#' で始まれば新仕様では見出し候補になるのは意図通りだが、
#  ここでは「フェンス後も後続の本当の見出しでちゃんと分割される」ことだけを
#  単純な形で確認する。)
_unterminated = f._parse_slides(
    "## 導入\n本文1\n```python\nx = 1\ny = 2\n## 結果\n本文2\n## まとめ\n本文3")
check("pptx新規: 閉じ忘れフェンス後の見出しは新規スライドになる",
      [s["title"] for s in _unterminated] == ["導入", "結果", "まとめ"])
check("pptx新規: 閉じ忘れフェンス後の見出しは#が除去されタイトルになる(literal箇条書きではない)",
      all("#" not in s["title"] for s in _unterminated)
      and not any(b.startswith("##") for s in _unterminated for b in s["bullets"]))
check("pptx新規: 閉じ忘れフェンスでも本文は失われない",
      "本文1" in _unterminated[0]["bullets"]
      and "本文2" in _unterminated[1]["bullets"]
      and "本文3" in _unterminated[2]["bullets"])

# 新規: 混在ケース(先に対になったブロックがあり、末尾だけ閉じ忘れ)でも、
# 先行する対になったブロックはコードモードを維持し#コメントを昇格させない。
_mixed = f._parse_slides(
    "## 前半\n```python\n# 対になっているコメント\n```\n## 後半\n```python\nx = 1\n## 次\n本文")
check("pptx混在: 先行する対になったブロックは見出し昇格させない",
      _mixed[0]["title"] == "前半" and "# 対になっているコメント" in _mixed[0]["bullets"])
check("pptx混在: 末尾の閉じ忘れフェンス後の見出しは新規スライドになる",
      [s["title"] for s in _mixed] == ["前半", "後半", "次"])

check("pptx: deck_title は短い質問を採用", f._deck_title("犬の紹介", _slides) == "犬の紹介")
# 2026-07-22: 空白のみの質問は if question で truthy のまま素通りし、
# strip()後に splitlines() が [] を返して [0] が IndexError になっていた回帰。
check("pptx: deck_title 空白のみ質問+スライド無しは既定値",
      f._deck_title("   ", []) == "プレゼンテーション")
check("pptx: deck_title 空白のみ質問はスライドタイトルへフォールバック",
      f._deck_title("\n\n", [{"title": "概要", "bullets": []}]) == "概要")
check("pptx: deck_title 空白のみ質問+無題スライドは既定値",
      f._deck_title("\t \n", [{"title": "", "bullets": []}]) == "プレゼンテーション")
check("pptx: deck_title 複数行質問は先頭行を採用",
      f._deck_title("\n  タイトル行\n本文\n", []) == "タイトル行")
check("pptx: deck_title 40字超はスライドタイトルへフォールバック",
      f._deck_title("あ" * 41, [{"title": "見出し", "bullets": []}]) == "見出し")
check("pptx: deck_title 空文字はスライドタイトルへフォールバック",
      f._deck_title("", [{"title": "見出し", "bullets": []}]) == "見出し")
check("pptx: deck_title None質問は既定値",
      f._deck_title(None, []) == "プレゼンテーション")

# ---------- 出力形態ルーティングガードレール ----------
f.PROPOSERS = ["gpt-oss:20b", "qwen3-coder:30b", "gemma4:26b", "phi4"]


def _base_plan():
    return {"mode": "single", "selected_proposers": ["gpt-oss:20b"], "rounds": 1,
            "use_image_generation": False, "image_only": False, "make_pptx": False,
            "search_required": False, "reason": "r", "_fallback": False}


_r = f._apply_routing_guardrails("機械学習入門のスライドを作って", _base_plan())
check("route: スライド→make_pptx+moa", _r["make_pptx"] is True and _r["mode"] == "moa")
_r = f._apply_routing_guardrails("かわいい柴犬のイラストを描いて", _base_plan())
check("route: イラストのみ→image_only", _r["use_image_generation"] is True and _r["image_only"] is True)
_r = f._apply_routing_guardrails("PINN洪水モデルを説明して図も作って", _base_plan())
check("route: 説明+図→イラスト付き(image_only=False)",
      _r["use_image_generation"] is True and _r["image_only"] is False)
_r = f._apply_routing_guardrails("日本の首都は？", _base_plan())
check("route: 通常質問は据え置き",
      _r["make_pptx"] is False and _r["use_image_generation"] is False)
f.PROPOSERS = _op_persona

# ---------- 自己一貫性投票（答え抽出・正規化・同値判定・投票） ----------
check("sc: boxed 抽出", f.extract_boxed("thus \\boxed{42}") == "42")
check("sc: boxed 入れ子", f.extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}")
check("sc: boxed 最後を採用", f.extract_boxed("\\boxed{1} then \\boxed{2}") == "2")
check("sc: boxed 無しは None", f.extract_boxed("no box") is None)
# 2026-07-22: \boxed{ が閉じられないまま出力が打ち切られた場合（thinking モデルの
# num_predict 打ち切り等、gotcha #2 の既知の失敗モード）は、切れた残骸を答えとして
# 返さず None（無投票）を返すことを検証する。
check("sc: boxed 未閉じは None（打ち切り）",
      f.extract_boxed("thus \\boxed{42 and then the response was cut off") is None)
check("sc: boxed 未閉じ・入れ子未対応も None",
      f.extract_boxed("\\boxed{\\frac{1}{2") is None)
check("sc: boxed 閉じ括弧後に散文があっても正しく抽出",
      f.extract_boxed("\\boxed{7} because it is prime") == "7")
check("sc: boxed 二重入れ子", f.extract_boxed("\\boxed{\\boxed{5}}") == "5")

# 2026-07-22 (iteration 12): 先に確定した \boxed{回答} があり、後続の
# \boxed{...} だけが打ち切られている場合は、手前の閉じた票を救出する
# （iteration 11 / gotcha #2, #7 参照。詳細は extract_boxed 本体のコメント）。
check("sc: boxed 後続が未閉じでも手前の確定票を救出",
      f.extract_boxed("\\boxed{42} then \\boxed{the next attempt got cut off") == "42")
check("sc: boxed 後続が入れ子ごと未閉じでも手前の確定票を救出",
      f.extract_boxed("\\boxed{7} ... \\boxed{\\frac{1}{2") == "7")

# 2026-07-22 (iteration 25, iteration 11/23 の続き): 末尾の \boxed{} が閉じては
# いるが中身が空/空白のみの場合は「無投票」として扱い、手前にある閉じた非空の
# 票まで遡って救出する（gotcha #2, #7 参照。詳細は extract_boxed 本体のコメント）。
# これをしないと extract_final_answer の math 分岐が None を受けて散文中の
# 数値を拾うフォールバックへ落ち、無投票のはずが誤投票に変わってしまう。
check("sc: boxed 末尾が空でも手前の確定票を救出",
      f.extract_boxed("\\boxed{5} then \\boxed{}") == "5")
check("sc: boxed 末尾が空白のみでも手前の確定票を救出",
      f.extract_boxed("\\boxed{42} then \\boxed{ }") == "42")
check("sc: boxed 単独の空は None（救出対象なし）",
      f.extract_boxed("\\boxed{}") is None)
check("sc: boxed 単独の空白のみも None（救出対象なし）",
      f.extract_boxed("\\boxed{ }") is None)
check("sc: boxed 末尾が空でも last-wins は非空同士で維持",
      f.extract_boxed("\\boxed{1} then \\boxed{2}") == "2")
check("sc: boxed 末尾空スキップ後もextract_final_answerが散文の数値を誤採用しない",
      f.extract_final_answer("\\boxed{5} then the loop ran 10 times \\boxed{}", "math") == "5")

check("sc: 正規化 全角→半角", f.normalize_answer("１２３") == "123")
check("sc: 正規化 桁区切り除去", f.normalize_answer("12,345") == "12345")
check("sc: 正規化 空白入り桁区切り", f.normalize_answer("11,\\! 111,\\! 111,\\! 100") == "11111111100")
check("sc: 正規化 前置き除去", f.normalize_answer("Answer: 700") == "700")
check("sc: 正規化 text外殻", f.normalize_answer("\\text{391}") == "391")
# 2026-07-22: _FW_TRANS 拡張分（Unicode MINUS SIGN / 全角句点・読点・スラッシュ）の
# 正規化。CJK 寄りのプロポーザ (qwen/gemma 系) がこれらを出力し、正規化しないと
# vote_answers で本来同値な答えが2系統の票に割れてしまう（詳細は _FW_TRANS 定義部の
# コメント参照）。
check("sc: 正規化 U+2212マイナス", f.normalize_answer("−5") == "-5")
check("sc: 正規化 全角ハイフンマイナス", f.normalize_answer("－5") == "-5")
check("sc: 正規化 全角数字+全角句点(小数)", f.normalize_answer("３．１４") == "3.14")
check("sc: 正規化 全角スラッシュ(分数)", f.normalize_answer("１／２") == "1/2")
check("sc: 正規化 全角カンマ 桁区切り", f.normalize_answer("1，234") == "1234")
# 回帰: 既存ASCII表記は一切変わらないこと
check("sc: 正規化 ASCII -5 不変", f.normalize_answer("-5") == "-5")
check("sc: 正規化 ASCII 1/2 不変", f.normalize_answer("1/2") == "1/2")
check("sc: 正規化 ASCII 1,234 不変", f.normalize_answer("1,234") == "1234")
# 2026-07-22: 末尾カンマ除去（extract_final_answer の数値抽出正規表現 [\d,]* が
# 桁区切りでない末尾カンマまで貪欲に飲み込む問題への対処、normalize_answer 側のコメント参照）
check("sc: 正規化 末尾カンマ除去", f.normalize_answer("42,") == "42")
check("sc: 正規化 桁区切り+末尾カンマ除去", f.normalize_answer("1234,") == "1234")

check("sc: 抽出 boxed優先", f.extract_final_answer("答えは 5 です。\\boxed{7}") == "7")
check("sc: 抽出 答え宣言", f.extract_final_answer("計算すると、答えは 700 円です") == "700")
check("sc: 抽出 最後の数値", f.extract_final_answer("17 * 23 = 391") == "391")
check("sc: 抽出 無しは None", f.extract_final_answer("わかりません") is None)
# 2026-07-22: 末尾カンマを伴う抽出（宣言分岐・最後の数値フォールバックの両方）
check("sc: 抽出 最後の数値 末尾カンマ",
      f.extract_final_answer("so in total we get 42,", "math") == "42")
check("sc: 抽出 答え宣言 桁区切り+末尾カンマ",
      f.extract_final_answer("the final answer is 1,234,", "math") == "1234")
check("sc: 抽出 最後の数値(boxedなし) 末尾カンマ",
      f.extract_final_answer("17 * 23 = 391,", "math") == "391")
# 2026-07-22: 最後の数値フォールバックの符号クラスに Unicode マイナス(U+2212)/
# 全角ハイフンマイナス(U+FF0D)を追加した回帰確認。\boxed{} も「答え」宣言もない
# 終端数値のみのケースで、CJK プロポーザが出しがちな全角/Unicode 符号付き負数が
# 正の値として誤投票されないことを検証する（extract_final_answer 内のコメント参照）。
check("sc: 抽出 最後の数値 U+2212マイナス(boxed/宣言なし)",
      f.extract_final_answer("計算の結果は −5", "math") == "-5")
# 注: 「答え/正解/answer」を含む文言だと宣言ブランチ(2318行目)が先に拾ってしまい
# ここで検証したい「最後の数値フォールバック」に到達しないため、あえてそれらの
# キーワードを含まない文言を使う。
check("sc: 抽出 最後の数値 全角ハイフンマイナス(boxed/宣言なし)",
      f.extract_final_answer("結論としては、最終的な値は －5である", "math") == "-5")
check("sc: 抽出 最後の数値 U+2212マイナスとASCIIの投票クラス一致",
      f.answers_equivalent(f.extract_final_answer("最終値は −5", "math"), "-5"))
# 2026-07-22: 最後の数値フォールバック（および宣言ブランチの数値部抽出）の整数部
# 文字クラスを [\d,]* から「桁区切りとして妥当なカンマのみ許容」に厳格化した回帰確認
# （iteration 13/22/24 と同じ抽出経路の姉妹修正、fugu_local.py 側のコメント参照）。
# \boxed{} も「答え/正解/answer」宣言も無く、桁区切りとして不正なカンマ区切りの
# 数値列で終わる文では、1トークンに誤結合された "1,2,3" ではなく最後の数値のみを拾う。
check("sc: 抽出 最後の数値 不正なカンマ区切り列は結合されない",
      f.extract_final_answer("the roots are 1,2,3", "math") == "3")
check("sc: 抽出 最後の数値 座標のカンマ区切りは結合されない",
      f.extract_final_answer("the point is (1,2)", "math") == "2")
check("sc: 抽出 最後の数値 不正カンマ区切り列がそのまま誤投票票にならない",
      f.answers_equivalent(f.extract_final_answer("the roots are 1,2,3", "math"), "3"))
# 桁区切りとして正当なカンマ(3桁区切り)は引き続き1トークンとして丸ごと拾う回帰確認。
check("sc: 抽出 最後の数値(boxed/宣言なし) 桁区切り1234",
      f.extract_final_answer("in total we counted up to 1,234", "math") == "1234")
check("sc: 抽出 最後の数値(boxed/宣言なし) 桁区切り1234567",
      f.extract_final_answer("in total we counted up to 1,234,567", "math") == "1234567")
check("sc: 抽出 最後の数値(boxed/宣言なし) 桁区切り12345",
      f.extract_final_answer("in total we counted up to 12,345", "math") == "12345")
check("sc: mcq boxed", f.extract_final_answer("\\boxed{B}", "mcq") == "B")
check("sc: mcq 宣言", f.extract_final_answer("正解は (C) です", "mcq") == "C")
check("sc: mcq 無しは None", f.extract_final_answer("どれも違う", "mcq") is None)
check("sc: mcq boxed 散文混じりは先頭文字",
      f.extract_final_answer("reasoning...\\boxed{C, because it is the largest}", "mcq") == "C")
check("sc: mcq boxed 散文のみは誤答せず None",
      f.extract_final_answer("\\boxed{None of the above}", "mcq") is None)
check("sc: mcq boxed 括弧付き先頭文字", f.extract_final_answer("\\boxed{(A)}", "mcq") == "A")
check("sc: mcq boxed text外殻付き先頭文字", f.extract_final_answer("\\boxed{\\text{D}}", "mcq") == "D")
check("sc: mcq boxed 選択肢+本文", f.extract_final_answer("\\boxed{A) 5}", "mcq") == "A")
check("sc: mcq 宣言 ディストラクタ言及に釣られない",
      f.extract_final_answer(
          "The correct answer is B. Note that answer A was a common distractor.",
          "mcq") == "B")
check("sc: mcq 宣言 訂正で文字が競合したら棄権",
      f.extract_final_answer(
          "The answer is B; oh wait, the answer: A", "mcq") is None)
check("sc: mcq 宣言 同一文字の繰り返しは誤棄権しない",
      f.extract_final_answer(
          "The answer is D. Restating: the answer is D.", "mcq") == "D")

# 2026-07-22: iteration 28 — math 宣言ブランチにも iteration 26 の MCQ 修正
# （複数宣言が競合したら無投票=None）を対称に適用した回帰確認。
# 注: 宣言抽出の捕獲グループ ([^\n]{1,60}) は改行を跨がず貪欲マッチするため、同一行に
# 複数の「answer is」があると最初のマッチが行末まで飲み込み2件目の宣言が独立して
# 検出されない。複数宣言を意図的に分離検出させるため、ここでは改行で区切って書く
# （実際の LLM 出力でも言い直し・訂正は改行/文区切りを伴うことが多い）。
check("sc: 抽出 答え宣言 訂正で数値が競合したら棄権",
      f.extract_final_answer(
          "The answer is 5.\nOn second thought, the answer is 7.", "math") is None)
_eq_restate = f.extract_final_answer(
    "The answer is 1/2.\nEquivalently, the answer is 0.5.", "math")
check("sc: 抽出 答え宣言 同値な言い直しは棄権しない",
      _eq_restate is not None and f.answers_equivalent(_eq_restate, "0.5"))
check("sc: 抽出 答え宣言 同一値の繰り返しは誤棄権しない",
      f.extract_final_answer(
          "The answer is 42.\nRestating: the answer is 42.", "math") == "42")
check("sc: 抽出 答え宣言 空の宣言候補は競合と数えない",
      f.extract_final_answer(
          "The answer is .\nThe answer is 9.", "math") == "9")

check("sc: 同値 完全一致", f.answers_equivalent("42", "42"))
check("sc: 同値 分数=小数", f.answers_equivalent("1/2", "0.5"))
check("sc: 同値 桁区切り", f.answers_equivalent("12,345", "12345"))
check("sc: 非同値", not f.answers_equivalent("41", "42"))
check("sc: 空は非同値", not f.answers_equivalent("", "42"))

# 2026-07-22: Unicode マイナス/全角スラッシュの同値判定が na.lower()/Fraction の
# 高速パスだけで完結し、math_verify に頼らないことを検証する。math_verify を
# 「呼ばれたら必ず例外」なスタブに差し替えても answers_equivalent が True を
# 返せることを確認し、フォールバック依存になっていないことを保証する。
def _mv_must_not_be_called(*_a, **_kw):
    raise RuntimeError("math_verify should not be needed for these fast-path cases")


_fake_math_verify = types.ModuleType("math_verify")
_fake_math_verify.parse = _mv_must_not_be_called
_fake_math_verify.verify = _mv_must_not_be_called
_orig_math_verify_mod = sys.modules.get("math_verify")
sys.modules["math_verify"] = _fake_math_verify
try:
    check("sc: 同値 U+2212マイナス（math_verify不要）", f.answers_equivalent("−5", "-5"))
    check("sc: 同値 全角スラッシュ分数（math_verify不要）", f.answers_equivalent("１／２", "1/2"))
finally:
    if _orig_math_verify_mod is not None:
        sys.modules["math_verify"] = _orig_math_verify_mod
    else:
        del sys.modules["math_verify"]

# 2026-07-22: math_verify フォールバック分岐そのもの（fugu_local.py ~L2433-2444）の
# 直接カバレッジ。上のテストは「フォールバックに頼らない」ことの証明であり、フォールバック
# 分岐自体には一度も入っていない。ここでは高速パス（na.lower()一致 / Fraction一致）を
# 意図的に迂回する入力（\frac{1}{2} vs 0.5、どちらも正規化後は非空・lower()不一致・
# Fraction変換失敗）を使い、記録スタブで (1) parse/verify が実際に呼ばれたこと、
# (2) gotcha #6 の parsing_timeout=None / timeout_seconds=None が渡っていること
# （Windows で math_verify 既定タイムアウトがハンドルエラーを撒く実測不具合への回帰防止）、
# (3) verify の戻り値がそのまま answers_equivalent の戻り値になること、
# (4) parse/verify が例外を送出しても except で握り潰され False になり例外が外に漏れない
# ことを検証する。


def _make_recording_math_verify(verify_result, raise_in):
    """呼び出しを記録する math_verify スタブモジュールを生成する。
    raise_in: "none" | "parse" | "verify" — 該当関数が呼ばれたら例外を送出する。"""
    calls = {"parse_args": [], "parse_kwargs": [], "verify_args": [], "verify_kwargs": []}

    def _parse(expr, **kwargs):
        calls["parse_args"].append(expr)
        calls["parse_kwargs"].append(kwargs)
        if raise_in == "parse":
            raise RuntimeError("boom in parse (stub)")
        return ("parsed", expr)

    def _verify(parsed_a, parsed_b, **kwargs):
        calls["verify_args"].append((parsed_a, parsed_b))
        calls["verify_kwargs"].append(kwargs)
        if raise_in == "verify":
            raise RuntimeError("boom in verify (stub)")
        return verify_result

    mod = types.ModuleType("math_verify")
    mod.parse = _parse
    mod.verify = _verify
    return mod, calls


def _run_with_math_verify_stub(verify_result, raise_in, body):
    """math_verify を記録スタブに差し替えて body(calls) を実行し、必ず元に戻す
    （L341-353 と同じ swap-and-restore パターン）。"""
    mod, calls = _make_recording_math_verify(verify_result, raise_in)
    orig = sys.modules.get("math_verify")
    sys.modules["math_verify"] = mod
    try:
        body(calls)
    finally:
        if orig is not None:
            sys.modules["math_verify"] = orig
        else:
            del sys.modules["math_verify"]


# 高速パスを迂回する入力ペア（正規化後も非空・lower()不一致・Fraction変換失敗）
_MV_A, _MV_B = r"\frac{1}{2}", "0.5"


def _t_mv_verify_true(calls):
    result = f.answers_equivalent(_MV_A, _MV_B)
    check("sc: math_verifyフォールバック parseが実呼出しされる", len(calls["parse_args"]) >= 2)
    check("sc: math_verifyフォールバック verifyが実呼出しされる", len(calls["verify_args"]) == 1)
    check("sc: math_verifyフォールバック parsing_timeout=None (gotcha#6)",
          len(calls["parse_kwargs"]) >= 2
          and all(kw.get("parsing_timeout", "MISSING") is None for kw in calls["parse_kwargs"]))
    check("sc: math_verifyフォールバック timeout_seconds=None (gotcha#6)",
          len(calls["verify_kwargs"]) == 1
          and all(kw.get("timeout_seconds", "MISSING") is None for kw in calls["verify_kwargs"]))
    check("sc: math_verifyフォールバック verify=Trueを伝播", result is True)


_run_with_math_verify_stub(True, "none", _t_mv_verify_true)


def _t_mv_verify_false(calls):
    result = f.answers_equivalent(_MV_A, _MV_B)
    check("sc: math_verifyフォールバック verify=Falseを伝播", result is False)


_run_with_math_verify_stub(False, "none", _t_mv_verify_false)


def _t_mv_parse_raises(calls):
    result = f.answers_equivalent(_MV_A, _MV_B)
    check("sc: math_verifyフォールバック parse例外はFalseに握り潰す（例外は漏れない）",
          result is False)


_run_with_math_verify_stub(None, "parse", _t_mv_parse_raises)


def _t_mv_verify_raises(calls):
    result = f.answers_equivalent(_MV_A, _MV_B)
    check("sc: math_verifyフォールバック verify例外はFalseに握り潰す（例外は漏れない）",
          result is False)


_run_with_math_verify_stub(None, "verify", _t_mv_verify_raises)

# math_verify の差し替えが確実に元へ復元されていること（スタブ混入なら False に化けるはずの
# 高速パスが正常動作することで間接確認する）
check("sc: math_verifyスタブ解除後も高速パスが正常動作（sys.modules復元確認）",
      f.answers_equivalent("42", "42") and not f.answers_equivalent("41", "42"))

_top, _cnt, _cls = f.vote_answers(["42", "42", "41", "0.5", "1/2", None, ""])
check("sc: 投票 最多クラス", _top == "42" and _cnt == 2)
check("sc: 投票 同値クラス集約", any(c[1] == 2 and f.answers_equivalent(c[0], "0.5") for c in _cls))
check("sc: 投票 空リスト", f.vote_answers([]) == (None, 0, []))

# ---------- vote_answers（tie安定性・代表選定・None/空フィルタ・件数降順） ----------
# gotcha #7 の自己整合性投票（solve_verifiable/_arbitrate）が構造的に依拠する契約を
# 直接ロックする。iteration 32 (answers_equivalent) / iteration 52 (_sc_sample) の
# 姉妹カバレッジ。全て answers_equivalent の高速パス（normalize一致/Fraction/_FW_TRANS）
# のみで解決する組み合わせを使い、math_verify の実体（サブプロセス/ライブラリ有無）には
# 依存しない。

# (a) 同数タイは「先出現が勝つ」: classes.sort(key=lambda c: -c[1]) は Python の安定
#     ソートなので、件数が同じクラス同士は出現順（挿入順）が保たれる。solve_verifiable の
#     拮抗判定 classes[0][1]==classes[1][1] や、_arbitrate が None を返した際の
#     先出現フォールバックは、この安定性に構造的に依存している。Counter や非安定ソートへの
#     リファクタはここを静かに壊しうる。
_top_a1, _cnt_a1, _cls_a1 = f.vote_answers(["7", "3", "3", "7"])
check("vote: 同数タイは先出現(7)が勝つ",
      _top_a1 == "7" and _cnt_a1 == 2 and _cls_a1 == [["7", 2], ["3", 2]])

_top_a2, _cnt_a2, _cls_a2 = f.vote_answers(["3", "7", "7", "3"])
check("vote: 入力順を逆にすると先出現(3)が勝つ(値の大小ではない)",
      _top_a2 == "3" and _cnt_a2 == 2 and _cls_a2 == [["3", 2], ["7", 2]])

# (b) 統合クラスの代表文字列は「最初に見た表記」。Fraction 高速パスで 1/2 と 0.5 が
#     同一クラスに集約されても、代表（res['votes'] のキー/_arbitrate の候補ラベルに使われる）
#     は後から来た '0.5' ではなく先出の '1/2' のまま。
_top_b, _cnt_b, _cls_b = f.vote_answers(["1/2", "0.5", "0.5"])
check("vote: 統合クラスの代表は最初に見た表記(1/2)のまま(0.5にならない)",
      (_top_b, _cnt_b, _cls_b) == ("1/2", 3, [["1/2", 3]]))

# (c) normalize_answer / _FW_TRANS 高速パス経由の統合（全角マイナス U+2212 → 半角ハイフン）。
_top_c, _cnt_c, _cls_c = f.vote_answers(["-5", "−5"])
check("vote: 全角マイナス(U+2212)と半角ハイフンの'-5'は同一クラスに統合される",
      (_top_c, _cnt_c, _cls_c) == ("-5", 2, [["-5", 2]]))

# (d) None/空文字は集計から除外される。
_top_d1, _cnt_d1, _cls_d1 = f.vote_answers(["5", None, "", "5"])
check("vote: Noneと空文字は集計から除外される",
      _top_d1 == "5" and _cnt_d1 == 2 and len(_cls_d1) == 1)
check("vote: 全てNone/空文字なら(None, 0, [])",
      f.vote_answers([None, "", None]) == (None, 0, []))

# (e) classes は件数降順で返る。入力の先頭に出た低頻度クラスも件数順どおり後ろに回る。
_top_e, _cnt_e, _cls_e = f.vote_answers(["5", "9", "9", "9", "2", "2"])
check("vote: classesは件数降順(入力先頭の低頻度クラスは後ろに回る)",
      _top_e == "9" and _cnt_e == 3
      and _cls_e == [["9", 3], ["2", 2], ["5", 1]])

# ---------- solve_verifiable（ask をモックして適応サンプリングを検証） ----------
_orig_ask2 = f.ask
_orig_props2 = f.PROPOSERS
_orig_reasoning = f.REASONING_MODELS
_orig_cheap = f.SC_CHEAP_VOTES
_orig_pot = f.SC_POT
_sc_calls = []


def _fake_sc_ask(model, messages, temperature, think=None, fmt=None,
                 label=None, num_predict=None, num_ctx=None):
    _sc_calls.append(model)
    return f"reasoning...\n\\boxed{{{'42' if len(_sc_calls) % 2 else '42'}}}"


try:
    f.PROPOSERS = ["m1", "m2"]
    f.REASONING_MODELS = ["m1", "m2"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ask = _fake_sc_ask
    _res = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
check("sc: 全会一致で早期確定", _res is not None and _res["answer"] == "42")
check("sc: 初回バッチのみで停止", len(_sc_calls) == f.SC_INITIAL)
check("sc: モデルを交互に使う", set(_sc_calls) == {"m1", "m2"})
# 2026-07-22 回帰: 拮抗/裁定が一切発生しないこのパスでは、今回の裁定後cnt/votes再計算
# 修正の影響を受けず、votes/n_samples が従来どおり返ることを明示的に確認する。
check("sc: 拮抗なし(全会一致)のvotes/n_samplesは修正の影響を受けない(不変)",
      _res is not None and _res["votes"] == {"42": f.SC_INITIAL}
      and _res["n_samples"] == f.SC_INITIAL)

# 票が割れるケース: 第1バッチで拮抗 → 追加サンプリング後に過半数で確定。
# バッチ化により第1バッチ(SC_INITIAL)は m1 まとめ→m2 まとめの順。call index で答えを固定し、
# 第1バッチを均等割り(過半数なし)にして 2 バッチ目で決着させる。
_sc_calls.clear()
_seq = (["\\boxed{1}"] * (f.SC_INITIAL // 2) + ["\\boxed{2}"] * (f.SC_INITIAL - f.SC_INITIAL // 2)
        + ["\\boxed{1}"] * 100)   # 第1バッチは均等、以降は 1 が積み上がる


def _fake_sc_ask2(model, messages, temperature, think=None, fmt=None,
                  label=None, num_predict=None, num_ctx=None):
    _sc_calls.append(model)
    idx = len(_sc_calls) - 1
    return _seq[idx] if idx < len(_seq) else "\\boxed{1}"


try:
    f.PROPOSERS = ["m1", "m2"]
    f.REASONING_MODELS = ["m1", "m2"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ask = _fake_sc_ask2
    _res2 = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
check("sc: 割れたら追加サンプリング", len(_sc_calls) > f.SC_INITIAL)
check("sc: 追加後に過半数で確定", _res2 is not None and _res2["answer"] == "1")

check("sc: SC_MIN_VOTES 定数", f.SC_MIN_VOTES == 3)

# 疑似全会一致ガード: 第1バッチで抽出成功が1票だけ（他は thinking打ち切り/boxed無しで
# 抽出失敗）だと、旧ロジックでは cnt(1)==n(1) で「全会一致」扱いになり k=1 で確定して
# しまっていた（2026-07-21 に発見・修正）。SC_MIN_VOTES 導入後は n<3 の全会一致では
# 確定させず、add_batch(SC_STEP) で追加サンプリングされることを検証する。
_orig_min_votes = f.SC_MIN_VOTES
_sc_calls.clear()
_seq3 = (["\\boxed{42}"] + ["すみません、答えが導けませんでした。"] * 5
         + ["\\boxed{42}"] * 100)  # 第1バッチ: 1票のみ抽出成功、以降は追加分がすべて42に収束


def _fake_sc_ask3(model, messages, temperature, think=None, fmt=None,
                  label=None, num_predict=None, num_ctx=None):
    _sc_calls.append(model)
    idx = len(_sc_calls) - 1
    return _seq3[idx] if idx < len(_seq3) else "\\boxed{42}"


try:
    f.PROPOSERS = ["m1", "m2"]
    f.REASONING_MODELS = ["m1", "m2"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ask = _fake_sc_ask3
    _res3 = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
    f.SC_MIN_VOTES = _orig_min_votes
check("sc: n<SC_MIN_VOTES の疑似全会一致では確定しない(追加サンプリング)",
      len(_sc_calls) > f.SC_INITIAL)
check("sc: 追加サンプリング後に正しく確定", _res3 is not None and _res3["answer"] == "42")

# 抽出成功が一度もない場合: 全バッチで n=0 のまま SC_MAX に到達し、無限ループせず
# None を返して MoA フォールバックへ委ねることを検証する（打ち切り自体は既存ロジック）。
_sc_calls.clear()


def _fake_sc_ask_noextract(model, messages, temperature, think=None, fmt=None,
                           label=None, num_predict=None, num_ctx=None):
    _sc_calls.append(model)
    return "考え中ですが、最終的な答えを出せませんでした。"


try:
    f.PROPOSERS = ["m1", "m2"]
    f.REASONING_MODELS = ["m1", "m2"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ask = _fake_sc_ask_noextract
    _res4 = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
    f.SC_MIN_VOTES = _orig_min_votes
check("sc: 抽出0票が続いてもハングせず終了", len(_sc_calls) > 0)
check("sc: 抽出0票なら None を返す(MoAへフォールバック)", _res4 is None)

check("sc: 拮抗なし時は _arbitrate 未使用・勝者サンプルの本文を採用(従来通り)",
      _res is not None and _res["text"].startswith("reasoning..."))

# ---------- SC_MIN_VOTES の床を最終returnにも適用（SC_MAX消化パス） ----------
# 2026-07-21: ループ内の早期確定条件（cnt==n and n>=SC_MIN_VOTES / n>=4 and cnt*2>n）は
# while ループの break だけを守っており、SC_MAX 消化で抜けた最終 return には床が
# 掛かっていなかった。thinking打ち切りで __ERROR__、PoT失敗、\boxed{}欠落などにより
# ほとんどのサンプルが抽出失敗すると、1〜2票しか無い「勝者」がそのまま確定扱いで返る
# 疑似全会一致バグが再現する。ここでは終始 2 サンプルしか抽出成功しない（残りは全て
# 抽出不能）状況を作り、SC_MAX に到達して None（MoA フォールバック）になること、かつ
# 無限ループせず打ち切られることを検証する。
_orig_min_votes2 = f.SC_MIN_VOTES
_sc_calls.clear()
_seq5 = (["\\boxed{42}", "\\boxed{42}"]
         + ["すみません、答えが導けませんでした。"] * 60)  # 以降は一切抽出できない


def _fake_sc_ask5(model, messages, temperature, think=None, fmt=None,
                  label=None, num_predict=None, num_ctx=None):
    _sc_calls.append(model)
    idx = len(_sc_calls) - 1
    return _seq5[idx] if idx < len(_seq5) else "すみません、答えが導けませんでした。"


try:
    f.PROPOSERS = ["m1", "m2"]
    f.REASONING_MODELS = ["m1", "m2"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ask = _fake_sc_ask5
    _res5 = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
    f.SC_MIN_VOTES = _orig_min_votes2
check("sc: SC_MAX消化まで無限ループせず打ち切られる(有限回で終了)",
      0 < len(_sc_calls) <= f.SC_MAX + 10)
check("sc: 最終return(SC_MAX消化パス)でも SC_MIN_VOTES 未満の勝者は None(床が効く)",
      _res5 is None)

# 境界値の回帰ガード: SC_MAX 消化パスでも勝者票数がちょうど SC_MIN_VOTES(3) に達して
# いれば床は発火せず、通常どおり dict を返さねばならない（floor の over-fire 防止）。
# 早期break条件（unanimous/majority）はどの中間状態でも満たさないよう票を分散させ、
# 最終的に SC_MAX 消化で抜けた時点で初めて勝者(42)が3票に達するようにしてある。
_orig_min_votes3 = f.SC_MIN_VOTES
_sc_calls.clear()
_seq6 = ["\\boxed{7}", "\\boxed{9}"]                      # batch1: idx0-1 (残り idx2-5 は抽出不能)
_seq6 += ["error"] * 4
_seq6 += ["\\boxed{42}", "\\boxed{42}", "\\boxed{7}", "error"]   # batch2: idx6-9
_seq6 += ["\\boxed{42}", "error", "error", "error"]              # batch3: idx10-13
_seq6 += ["error"] * 4                                             # batch4: idx14-17
_seq6 += ["error"] * 4                                             # batch5: idx18-21
_seq6 += ["error"] * 20                                            # 余裕分


def _fake_sc_ask6(model, messages, temperature, think=None, fmt=None,
                  label=None, num_predict=None, num_ctx=None):
    _sc_calls.append(model)
    idx = len(_sc_calls) - 1
    return _seq6[idx] if idx < len(_seq6) else "error"


try:
    f.PROPOSERS = ["m1", "m2"]
    f.REASONING_MODELS = ["m1", "m2"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ask = _fake_sc_ask6
    _res6 = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
    f.SC_MIN_VOTES = _orig_min_votes3
check("sc: SC_MAX消化パスでも勝者票数=SC_MIN_VOTESなら床は発火しない(通常確定)",
      _res6 is not None and _res6["answer"] == "42")
check("sc: 床が過剰発火していない場合はvotes/n_samplesも通常どおり返る",
      _res6 is not None and _res6["votes"].get("42") == 3 and _res6["n_samples"] == 22)

# ---------- 拮抗時の裁定（_arbitrate）----------
# 上位2クラスが同数で並ぶと _arbitrate が呼ばれる。かつては裁定役の答えだけを採用し、
# 本文(res['text'])は SAMPLE プールから _representative_text で再選出していたため、
# 裁定で数値が変わったり第三の答えに覆ったりすると、本文が敗者側候補の主張のまま
# 残る内部矛盾があった（2026-07-21 発見・修正）。ここでは _arbitrate 自身の解答
# テキストが res['text'] として使われることを検証する。
_orig_installed = f.installed_models
_orig_arbiter_model = f.ARBITER_MODEL


def _fake_installed_m1m2():
    return ["m1", "m2"]


# ケース1: 拮抗 → 裁定役が既存候補の一方(1)を支持。本文は裁定役自身の推論であること
# （敗者候補(2)の本文であってはならない）。
_arb_calls = []


def _fake_ask_arb_pick_existing(model, messages, temperature, think=None, fmt=None,
                                label=None, num_predict=None, num_ctx=None):
    _arb_calls.append((label, model))
    if label == "arbiter":
        return ("ARBITER_REASONING: candidate B miscalculates in step 2; "
                 "re-solving from scratch gives \\boxed{1}")
    idx = len(_arb_calls) - 1
    ans = "1" if idx % 2 == 0 else "2"
    return f"sc reasoning candidate {ans}\n\\boxed{{{ans}}}"


try:
    f.PROPOSERS = ["m1", "m2"]
    f.REASONING_MODELS = ["m1", "m2"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ARBITER_MODEL = None
    f.installed_models = _fake_installed_m1m2
    f.ask = _fake_ask_arb_pick_existing
    _res_arb1 = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
    f.ARBITER_MODEL = _orig_arbiter_model
    f.installed_models = _orig_installed
check("arb: 票が同数で拮抗 → 裁定役が呼ばれる", any(lab == "arbiter" for lab, _m in _arb_calls))
check("arb: 裁定役の答えを採用", _res_arb1 is not None and _res_arb1["answer"] == "1")
check("arb: 本文は裁定役自身の推論(敗者候補の本文ではない)",
      _res_arb1 is not None and "ARBITER_REASONING" in _res_arb1["text"]
      and "sc reasoning candidate" not in _res_arb1["text"])
# 2026-07-22: 裁定役が既存の票クラス('1')をそのまま支持したケースでは votes 辞書は
# 従来どおり(裁定前の同値クラス集計そのまま)で、answer は必ずそのキーとして存在すること。
check("arb: 裁定役が既存候補を採用した場合、votesにanswerがキーとして存在し票数は正しい",
      _res_arb1 is not None and _res_arb1["answer"] in _res_arb1["votes"]
      and _res_arb1["votes"]["1"] == _res_arb1["votes"]["2"])

# ケース2: 裁定役が両候補と異なる第三の答えを提示 → 本文が敗者側候補の主張になって
# はいけない（旧ロジックのバグ: _representative_text が第三の答えと同値のサンプルを
# 見つけられず「最長サンプル」＝どちらかの敗者の本文にフォールバックしていた）。
_arb_calls2 = []


def _fake_ask_arb_new_answer(model, messages, temperature, think=None, fmt=None,
                             label=None, num_predict=None, num_ctx=None):
    _arb_calls2.append((label, model))
    if label == "arbiter":
        return ("ARBITER_REASONING_NEW: both candidates share the same wrong "
                 "assumption; the correct value is \\boxed{3}")
    idx = len(_arb_calls2) - 1
    ans = "1" if idx % 2 == 0 else "2"
    return f"sc reasoning candidate {ans}\n\\boxed{{{ans}}}"


try:
    f.PROPOSERS = ["m1", "m2"]
    f.REASONING_MODELS = ["m1", "m2"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ARBITER_MODEL = None
    f.installed_models = _fake_installed_m1m2
    f.ask = _fake_ask_arb_new_answer
    _res_arb2 = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
    f.ARBITER_MODEL = _orig_arbiter_model
    f.installed_models = _orig_installed
check("arb: 裁定役が出した第三の答えを採用", _res_arb2 is not None and _res_arb2["answer"] == "3")
check("arb: 本文は裁定役の推論(第三の答え。敗者候補の主張ではない)",
      _res_arb2 is not None and "ARBITER_REASONING_NEW" in _res_arb2["text"]
      and "sc reasoning candidate" not in _res_arb2["text"])
# 2026-07-22 本修正の回帰テスト: 裁定役が票の無い第三の答え('3')を採用した場合、
# res['votes'] にその答えが「真の票数(0票)」で必ずキーとして載ること。また旧トップ
# ('1'または'2'、どちらも拮抗していた同数票)の票数が、そのまま裁定結果('3')の
# 票数であるかのように誤って表示されてはならない。
check("arb: 第三の答え採用時、votesにanswerが0票としてキーで存在する",
      _res_arb2 is not None and _res_arb2["votes"].get("3") == 0)
check("arb: 第三の答え採用時、敗者候補('1'/'2')の票数が勝者の票数として流用されていない",
      _res_arb2 is not None and _res_arb2["votes"]["1"] == _res_arb2["votes"]["2"]
      and _res_arb2["votes"]["1"] > 0
      and _res_arb2["votes"]["3"] != _res_arb2["votes"]["1"])

# ケース3 (2026-07-22 本修正): 裁定役が既存の拮抗クラスの一つと数学的に同値だが、
# 書き方だけ異なる文字列を返す（例: クラス代表 '1/2' に対し裁定役は小数表記 '0.5' を
# 提示。分数⇄小数の書き直しは裁定役がよくやる）。旧コードは
# 「match is None or match[0] != top」を一括りに扱っていたため、match が見つかって
# いても新規クラス [top, cnt] を無条件追加してしまい、同じ票が '1/2' と '0.5' の
# 二つのキーに二重計上されていた（sum(votes.values()) が実際の有効票数を超える）。
_arb_calls3b = []


def _fake_ask_arb_equivalent_diff_string(model, messages, temperature, think=None, fmt=None,
                                          label=None, num_predict=None, num_ctx=None):
    _arb_calls3b.append((label, model))
    if label == "arbiter":
        return ("ARBITER_REASONING_EQUIV: both are the same value; the correct "
                 "final answer is \\boxed{0.5}")
    idx = len(_arb_calls3b) - 1
    ans = "1/2" if idx % 2 == 0 else "2"
    return f"sc reasoning candidate {ans}\n\\boxed{{{ans}}}"


try:
    f.PROPOSERS = ["m1", "m2"]
    f.REASONING_MODELS = ["m1", "m2"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ARBITER_MODEL = None
    f.installed_models = _fake_installed_m1m2
    f.ask = _fake_ask_arb_equivalent_diff_string
    _res_arb_eq = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
    f.ARBITER_MODEL = _orig_arbiter_model
    f.installed_models = _orig_installed

_valid_votes_eq = sum(1 for lab, _m in _arb_calls3b if lab != "arbiter")
check("arb-eq: 裁定役が同値・別表記('0.5')を返す → answer はその文字列そのもの",
      _res_arb_eq is not None and _res_arb_eq["answer"] == "0.5")
check("arb-eq: votes に '0.5' が一度だけ載り、旧代表 '1/2' キーは残らない(二重計上なし)",
      _res_arb_eq is not None and _res_arb_eq["votes"].get("0.5") is not None
      and "1/2" not in _res_arb_eq["votes"])
check("arb-eq: '0.5' の票数は '2' クラスと同じ(旧'1/2'の真の票数)で0ではない",
      _res_arb_eq is not None and _res_arb_eq["votes"]["0.5"] == _res_arb_eq["votes"]["2"]
      and _res_arb_eq["votes"]["0.5"] > 0)
check("arb-eq: 票の合計が実際の有効票数と一致する(水増しなし)",
      _res_arb_eq is not None and sum(_res_arb_eq["votes"].values()) == _valid_votes_eq)

# ---------- 2026-07-22: N択拮抗で _arbitrate が classes[:2] に打ち切らないこと ----------
# 上位2クラスが同数(classes[0][1]==classes[1][1])で拮抗判定が発火する状況は、実際には
# 3クラス以上が同数タイになるケース(例: 票数 [k,k,k])も含みうる。従来の _arbitrate は
# classes[:2] で常に先頭2クラスしか裁定役に見せておらず、3番目以降の同数クラス
# （それが正解かもしれない）が黙って握りつぶされていた。ここでは 3モデル×均等分配で
# 常に3クラスが同数のまま SC_MAX まで積み上がる状況を作り、裁定役に3候補全てが
# 提示されること、かつ裁定役が(既存候補の一つである)第3クラスの答えを採用した場合に
# 事後の票数再集計(recount)が正しく合成されることを検証する。
_orig_installed3 = f.installed_models
_orig_arbiter_model3 = f.ARBITER_MODEL
_arb3_prompts = []
_sc3_idx = [0]


def _fake_installed_m1m2m3():
    return ["m1", "m2", "m3"]


def _fake_ask_arb_3way(model, messages, temperature, think=None, fmt=None,
                       label=None, num_predict=None, num_ctx=None):
    if label == "arbiter":
        _arb3_prompts.append(messages[0]["content"])
        # 裁定役は「票の無い新答え」ではなく、拮抗していた3クラスのうち3番目(='3')を支持する。
        return "ARBITER_REASONING_3WAY: candidate C is correct after re-derivation \\boxed{3}"
    idx = _sc3_idx[0]
    _sc3_idx[0] += 1
    ans = str((idx % 3) + 1)   # '1','2','3' を均等に繰り返す → 常に3クラス同数タイ
    return f"sc reasoning candidate {ans}\n\\boxed{{{ans}}}"


try:
    f.PROPOSERS = ["m1", "m2", "m3"]
    f.REASONING_MODELS = ["m1", "m2", "m3"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ARBITER_MODEL = None
    f.installed_models = _fake_installed_m1m2m3
    f.ask = _fake_ask_arb_3way
    _res_arb3 = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
    f.ARBITER_MODEL = _orig_arbiter_model3
    f.installed_models = _orig_installed3

check("arb3: 3択で拮抗 → 裁定役が呼ばれる", len(_arb3_prompts) > 0)
_arb3_prompt = _arb3_prompts[0] if _arb3_prompts else ""
check("arb3: 裁定役プロンプトに候補A(final answer: 1)が含まれる",
      "Candidate A" in _arb3_prompt and "final answer: 1" in _arb3_prompt)
check("arb3: 裁定役プロンプトに候補B(final answer: 2)が含まれる",
      "Candidate B" in _arb3_prompt and "final answer: 2" in _arb3_prompt)
check("arb3: 裁定役プロンプトに候補C(final answer: 3)が含まれる(3番目のタイ候補が握りつぶされていない)",
      "Candidate C" in _arb3_prompt and "final answer: 3" in _arb3_prompt)
check("arb3: プロンプト文言が候補数に応じている(count-agnostic)",
      "3 candidate solutions disagree" in _arb3_prompt
      and "Two candidate solutions disagree" not in _arb3_prompt)
check("arb3: 裁定役が採用した第3候補('3')がresの答えになる",
      _res_arb3 is not None and _res_arb3["answer"] == "3")
check("arb3: votes再集計により'3'の真の票数(既存タイ候補の実票数)が反映される",
      _res_arb3 is not None and _res_arb3["votes"].get("3") is not None
      and _res_arb3["votes"]["3"] == _res_arb3["votes"]["1"] == _res_arb3["votes"]["2"]
      and _res_arb3["votes"]["3"] > 0)

# ---------- 2026-07-22: ARBITRATE_MAX_CANDIDATES による上限保護(病的な多択タイ) ----------
# num_ctx(8192/16384に固定)を溢れさせないよう、同数タイが上限を超える場合は上限件数
# のみ提示し、超過分は黙って捨てずログに出す。_arbitrate を直接叩いて検証する
# （solve_verifiable 経由でここまで多くの均等クラスを作るのは非現実的なため）。
_cap_samples = [{"answer": str(i), "text": f"reasoning for {i}", "model": "m1", "pot": False}
                for i in range(1, 6)]                       # 1..5 の5クラス、全て同数(2票)
_cap_classes = [[str(i), 2] for i in range(1, 6)]
_orig_installed_cap = f.installed_models
_orig_arbiter_model_cap = f.ARBITER_MODEL
_orig_reasoning_cap = f.REASONING_MODELS
_orig_props_cap = f.PROPOSERS
_cap_prompts = []


def _fake_ask_cap(model, messages, temperature, think=None, fmt=None,
                  label=None, num_predict=None, num_ctx=None):
    if label == "arbiter":
        _cap_prompts.append(messages[0]["content"])
        return "ARBITER_REASONING_CAP \\boxed{1}"
    return "\\boxed{1}"


_cap_stdout = io.StringIO()
try:
    f.PROPOSERS = ["m1"]
    f.REASONING_MODELS = ["m1"]
    f.ARBITER_MODEL = None
    f.installed_models = lambda: ["m1"]
    f.ask = _fake_ask_cap
    with contextlib.redirect_stdout(_cap_stdout):
        _cap_result = f._arbitrate("test question", "math", _cap_samples, _cap_classes)
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props_cap
    f.REASONING_MODELS = _orig_reasoning_cap
    f.ARBITER_MODEL = _orig_arbiter_model_cap
    f.installed_models = _orig_installed_cap

_cap_prompt = _cap_prompts[0] if _cap_prompts else ""
check("cap: 5択タイでも上限(ARBITRATE_MAX_CANDIDATES=4)件しか提示されない",
      sum(f"final answer: {i}" in _cap_prompt for i in range(1, 6)) == f.ARBITRATE_MAX_CANDIDATES)
check("cap: 上限超過分(5)は黙って捨てずログに出力される",
      "5" in _cap_stdout.getvalue() and "提示されません" in _cap_stdout.getvalue())
check("cap: 上限保護時も有効な(answer, text)タプルを返す",
      _cap_result is not None and _cap_result[0] == "1")

# ---------- 2026-07-22: _arbitrate が ask() の __ERROR__ センチネルを誤って
# 数値解答として採用しないこと ----------
# ask() は失敗時 '__ERROR__: HTTP Error 500: Internal Server Error {...}' のような
# 文字列を返す（line ~1079）。旧 _arbitrate はこれをチェックせず strip_think →
# extract_final_answer に渡していたため、math タスクの最終数値フォールバック
# （line ~2299, `nums = re.findall(...)`）がエラーメッセージ中の '500'/'429' を
# 「裁定役の最終解答」として誤採用し、拮抗投票がでっち上げの自信満々な数値に化けて
# いた（_sc_sample=iter4, ask()自体=iter9, _critic_judge/second_opinion=iter15 で
# 直した同種バグの兄弟ケース）。ここでは (a) solve_verifiable 経由で拮抗した全裁定役が
# エラーになっても最終結果にエラー文字列/誤答が漏れないこと、(b) _arbitrate を直接叩いて
# チェーンの先頭がエラーでも次の裁定役へフォールバックすること、(c) 全裁定役がエラーなら
# _arbitrate が None を返すこと、の3点を検証する。

# (a) solve_verifiable レベル: ARBITER_MODEL 無し・REASONING_MODELS=PROPOSERS=[m1,m2] の
# 従来ケース1と同じ拮抗を作り、裁定役(m1もm2も)が毎回 __ERROR__ を返す状況。
_arb_err_calls = []


def _fake_ask_arb_error_only(model, messages, temperature, think=None, fmt=None,
                              label=None, num_predict=None, num_ctx=None):
    _arb_err_calls.append((label, model))
    if label == "arbiter":
        return "__ERROR__: HTTP Error 500: Internal Server Error"
    idx = len(_arb_err_calls) - 1
    ans = "1" if idx % 2 == 0 else "2"
    return f"sc reasoning candidate {ans}\n\\boxed{{{ans}}}"


try:
    f.PROPOSERS = ["m1", "m2"]
    f.REASONING_MODELS = ["m1", "m2"]
    f.SC_CHEAP_VOTES = 0
    f.SC_POT = False
    f.ARBITER_MODEL = None
    f.installed_models = _fake_installed_m1m2
    f.ask = _fake_ask_arb_error_only
    _res_arb_err = f.solve_verifiable("test question", "math")
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props2
    f.REASONING_MODELS = _orig_reasoning
    f.SC_CHEAP_VOTES = _orig_cheap
    f.SC_POT = _orig_pot
    f.ARBITER_MODEL = _orig_arbiter_model
    f.installed_models = _orig_installed
check("arb-err: 拮抗した全裁定役が__ERROR__を返してもanswerに'500'が誤採用されない",
      _res_arb_err is None or _res_arb_err["answer"] != "500")
check("arb-err: 拮抗した全裁定役が__ERROR__を返してもtextにエラー文字列が漏れない",
      _res_arb_err is None or "__ERROR__" not in _res_arb_err["text"])

# (b) _arbitrate 直接: ARBITER_MODEL(裁定役1番手)が__ERROR__、REASONING_MODELS の
# フォールバック(2番手)が有効な \boxed 解答を返す → チェーンを進めてその有効解答を
# 採用すること（エラーで止まって None になったり、エラー文中の数値を拾ったりしない）。
_orig_installed_e2 = f.installed_models
_orig_arbiter_model_e2 = f.ARBITER_MODEL
_orig_reasoning_e2 = f.REASONING_MODELS
_orig_props_e2 = f.PROPOSERS


def _fake_ask_arb_chain_fallback(model, messages, temperature, think=None, fmt=None,
                                  label=None, num_predict=None, num_ctx=None):
    assert label == "arbiter"
    if model == "arb_big":
        return "__ERROR__: HTTP Error 500: Internal Server Error"
    return "ARBITER_REASONING_FALLBACK: re-derived correctly \\boxed{7}"


_fb_samples = [{"answer": "1", "text": "reasoning for 1", "model": "m1", "pot": False},
               {"answer": "2", "text": "reasoning for 2", "model": "m1", "pot": False}]
_fb_classes = [["1", 2], ["2", 2]]
try:
    f.PROPOSERS = ["m1"]
    f.REASONING_MODELS = ["arb_big", "m1"]
    f.ARBITER_MODEL = "arb_big"
    f.installed_models = lambda: ["arb_big", "m1"]
    f.ask = _fake_ask_arb_chain_fallback
    _e2_result = f._arbitrate("test question", "math", _fb_samples, _fb_classes)
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props_e2
    f.REASONING_MODELS = _orig_reasoning_e2
    f.ARBITER_MODEL = _orig_arbiter_model_e2
    f.installed_models = _orig_installed_e2
check("arb-err: 先頭裁定役が__ERROR__ → 次の裁定役の有効な\\boxed解答へフォールバック",
      _e2_result is not None and _e2_result[0] == "7")
check("arb-err: フォールバック採用時の本文は次裁定役自身の推論(エラー文ではない)",
      _e2_result is not None and "ARBITER_REASONING_FALLBACK" in _e2_result[1]
      and "__ERROR__" not in _e2_result[1])

# (c) _arbitrate 直接: チェーン全員が__ERROR__(しかも数字入り)を返す → None を返し、
# 誤った数値タプルをでっち上げないこと。
_orig_installed_e3 = f.installed_models
_orig_arbiter_model_e3 = f.ARBITER_MODEL
_orig_reasoning_e3 = f.REASONING_MODELS
_orig_props_e3 = f.PROPOSERS


def _fake_ask_arb_all_error(model, messages, temperature, think=None, fmt=None,
                             label=None, num_predict=None, num_ctx=None):
    assert label == "arbiter"
    if model == "arb_big":
        return "__ERROR__: HTTP Error 500: Internal Server Error"
    return "__ERROR__: HTTP Error 429: Too Many Requests"


try:
    f.PROPOSERS = ["m1"]
    f.REASONING_MODELS = ["arb_big", "m1"]
    f.ARBITER_MODEL = "arb_big"
    f.installed_models = lambda: ["arb_big", "m1"]
    f.ask = _fake_ask_arb_all_error
    _e3_result = f._arbitrate("test question", "math", _fb_samples, _fb_classes)
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props_e3
    f.REASONING_MODELS = _orig_reasoning_e3
    f.ARBITER_MODEL = _orig_arbiter_model_e3
    f.installed_models = _orig_installed_e3
check("arb-err: 全裁定役が__ERROR__ → _arbitrate は None を返す(数値をでっち上げない)",
      _e3_result is None)

# ---------- 2026-07-22: _arbitrate プロンプト本文の候補数非依存化(iteration 16の続き) ----------
# iteration 16 はヘッダー行("{len(reps)} candidate solutions disagree:")を候補数に
# 応じた表現に直したが、本文の指示文 "Carefully check both, find the flaw in the wrong
# one" は2択決め打ちのまま残っていた。3択/4択タイでは候補が3-4件出るのに「both」
# 「the wrong one」(単数)と言われ、3件目以降の精査が軽視されるリスクがある。ここでは
# _arbitrate を直接叩き、(a) 2択では従来と同じ意図(両候補を精査し\boxed{}で単一の正解を
# 出す)を保ったまま候補数非依存の文言になっていること、(b) 3択では "check both" や
# 単数形の「the wrong one」という誤解を招く表現が出ないこと、(c) いずれもヘッダー行と
# \boxed{} 指示は変更されておらず、有効な(answer, text)タプル/Noneの契約を保つこと、を検証する。
_orig_installed_body2 = f.installed_models
_orig_arbiter_model_body2 = f.ARBITER_MODEL
_orig_reasoning_body2 = f.REASONING_MODELS
_orig_props_body2 = f.PROPOSERS
_body2_prompts = []


def _fake_ask_body2(model, messages, temperature, think=None, fmt=None,
                     label=None, num_predict=None, num_ctx=None):
    if label == "arbiter":
        _body2_prompts.append(messages[0]["content"])
        return "ARBITER_REASONING_BODY2 \\boxed{1}"
    return "\\boxed{1}"


_body2_samples = [{"answer": "1", "text": "reasoning for 1", "model": "m1", "pot": False},
                  {"answer": "2", "text": "reasoning for 2", "model": "m1", "pot": False}]
_body2_classes = [["1", 2], ["2", 2]]
try:
    f.PROPOSERS = ["m1"]
    f.REASONING_MODELS = ["m1"]
    f.ARBITER_MODEL = None
    f.installed_models = lambda: ["m1"]
    f.ask = _fake_ask_body2
    _body2_result = f._arbitrate("test question", "math", _body2_samples, _body2_classes)
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props_body2
    f.REASONING_MODELS = _orig_reasoning_body2
    f.ARBITER_MODEL = _orig_arbiter_model_body2
    f.installed_models = _orig_installed_body2

_body2_prompt = _body2_prompts[0] if _body2_prompts else ""
check("arb-body2: 2択のヘッダー行は従来通り('2 candidate solutions disagree:')",
      "2 candidate solutions disagree:" in _body2_prompt)
check("arb-body2: 2択でも本文は各候補の精査を指示している(same intent as 'check both')",
      "each candidate" in _body2_prompt)
check("arb-body2: \\boxed{} による単一最終解答の指示は維持されている",
      "\\boxed{}" in _body2_prompt)
check("arb-body2: 有効な(answer, text)タプルを返す",
      _body2_result is not None and _body2_result[0] == "1")

_orig_installed_body3 = f.installed_models
_orig_arbiter_model_body3 = f.ARBITER_MODEL
_orig_reasoning_body3 = f.REASONING_MODELS
_orig_props_body3 = f.PROPOSERS
_body3_prompts = []


def _fake_ask_body3(model, messages, temperature, think=None, fmt=None,
                     label=None, num_predict=None, num_ctx=None):
    if label == "arbiter":
        _body3_prompts.append(messages[0]["content"])
        return "ARBITER_REASONING_BODY3 \\boxed{1}"
    return "\\boxed{1}"


_body3_samples = [{"answer": str(i), "text": f"reasoning for {i}", "model": "m1", "pot": False}
                  for i in (1, 2, 3)]
_body3_classes = [["1", 2], ["2", 2], ["3", 2]]
try:
    f.PROPOSERS = ["m1"]
    f.REASONING_MODELS = ["m1"]
    f.ARBITER_MODEL = None
    f.installed_models = lambda: ["m1"]
    f.ask = _fake_ask_body3
    _body3_result = f._arbitrate("test question", "math", _body3_samples, _body3_classes)
finally:
    f.ask = _orig_ask2
    f.PROPOSERS = _orig_props_body3
    f.REASONING_MODELS = _orig_reasoning_body3
    f.ARBITER_MODEL = _orig_arbiter_model_body3
    f.installed_models = _orig_installed_body3

_body3_prompt = _body3_prompts[0] if _body3_prompts else ""
check("arb-body3: 3択のヘッダー行は候補数に応じている('3 candidate solutions disagree:')",
      "3 candidate solutions disagree:" in _body3_prompt)
check("arb-body3: 3択の本文に2択決め打ちの'check both'が残っていない",
      "check both" not in _body3_prompt.lower())
check("arb-body3: 3択の本文に単数形決め打ちの'the wrong one'が残っていない",
      "the wrong one" not in _body3_prompt.lower())
check("arb-body3: 3択の本文は候補数非依存の表現('each candidate'/'incorrect one(s)')になっている",
      "each candidate" in _body3_prompt and "incorrect one" in _body3_prompt)
check("arb-body3: \\boxed{} による単一最終解答の指示は維持されている",
      "\\boxed{}" in _body3_prompt)
check("arb-body3: 有効な(answer, text)タプルを返す",
      _body3_result is not None and _body3_result[0] == "1")

# ---------- task_type ガードレール ----------
def _tt(q, declared=""):
    return f._apply_tasktype_guardrails(q, {"task_type": declared})["task_type"]


check("tt: AIME風は math", _tt("Find the number of ordered pairs...") == "math")
check("tt: 日本語計算は math", _tt("1000円の3割引の支払額を求めよ") == "math")
check("tt: 選択肢列挙は mcq", _tt("正しいものを選べ\nA) foo\nB) bar") == "mcq")
check("tt: which of the following は mcq", _tt("Which of the following is true?") == "mcq")
check("tt: コードは code", _tt("フィボナッチ関数を実装して") == "code")
check("tt: 証明は math にしない", _tt("3連続整数の積が6の倍数であることを証明して求めよ") != "math")
check("tt: Conductor申告を尊重", _tt("こんにちは", "chat") == "chat")
check("tt: 不明シグナルは chat", _tt("よろしくね", "") == "chat")

# --- 誤申告レスキュー: _apply_tasktype_guardrails は solve_verifiable(自己一貫性投票、
# gotcha #7) への入口ゲート。小型 Conductor が math/mcq 問題を誤って chat/knowledge 等に
# 分類しても、確実なキーワードシグナルがあれば強制的に正しい task_type へ補正し、投票
# 経路を失わないことを検証する（Conductor申告のみを尊重する既存チェックの逆側）。
check("tt: 誤申告(chat)でも強い math シグナルで math へレスキュー",
      _tt("Find the remainder when 7^100 is divided by 13.", "chat") == "math")
check("tt: 誤申告(knowledge)でも選択肢列挙シグナルで mcq へレスキュー",
      _tt("正しいものを選べ\nA) foo\nB) bar", "knowledge") == "mcq")

# --- シグナル優先順位: 実装コードは if/elif mcq -> code -> math の順で判定されるため、
# mcq シグナルが最優先、次いで code、最後に math。複数シグナルが同居する問題での
# 優先順位を固定する。
check("tt: code と math シグナルが同居 -> elif順で code が勝つ",
      _tt("フィボナッチ関数を実装して。その関数の余りを求めよ。") == "code")
check("tt: mcq と code シグナルが同居 -> mcq が最優先",
      _tt("次のうち、コードの実装として正しいものを選べ\nA) foo\nB) bar") == "mcq")

# --- 自由記述デモーションのスコープ: t == "math" のときのみ証明/説明系シグナルで
# knowledge へ格下げされる（＝投票に回さない）。declared="mcq" など math 以外に確定した
# 場合はこのデモーション条件に入らず、証明系ワードが含まれていても格下げされないことを
# 確認する。
check("tt: 申告math + 自由記述シグナル(シグナル一致なし) -> knowledge へ格下げ",
      _tt("なぜ空は青いのか", "math") == "knowledge")
check("tt: 申告mcq + 自由記述ワードが含まれていても mcq のまま(格下げされない)",
      _tt("なぜ空は青いか説明して", "mcq") == "mcq")

check("tt: validate が task_type を保持",
      f.validate_plan({"mode": "moa", "selected_proposers": [],
                       "task_type": "math"})["task_type"] == "math")
check("tt: validate が不正 task_type を空へ",
      f.validate_plan({"mode": "moa", "selected_proposers": [],
                       "task_type": "quiz"})["task_type"] == "")

# ---------- MODEL_CONFIG 解決 ----------
check("cfg: 既知モデルの num_ctx", f.model_cfg("gpt-oss:20b", "num_ctx") == 16384)
check("cfg: 未知モデルは default", f.model_cfg("nonexistent", "num_ctx", 8192) == 8192)
check("cfg: think 段階指定", f.model_cfg("gpt-oss:20b", "think") == "high")

# ---------- is_installed（インストール済み判定：厳密タグ一致） ----------
# resolve_models() が DESIRED_PROPOSERS を採否判定し、_arbitrate() が ARBITER_MODEL の
# 起用可否を判定する土台。docstring の通り、旧 startswith 実装は 'qwen3:4b' が
# 'qwen3:4b-instruct' に誤ヒットするバグを持っていたため厳密一致へ変更され、
# タグ無し指定のときだけ ':latest' を許容する例外が残された。この判定を誤ると
# 未導入モデルを誤って起用/裁定役に据えたり、導入済みの正規プロポーザーを
# 黙って除外したりして、精度優先のアンサンブル構成が静かに壊れる。
check("inst: 厳密タグ一致で導入判定", f.is_installed("qwen3:4b", ["qwen3:4b"]) is True)
check("inst: 旧startswithの誤検知(タグ違い)は拒否",
      f.is_installed("qwen3:4b", ["qwen3:4b-instruct"]) is False)
check("inst: タグ無し指定は :latest 導入を許容",
      f.is_installed("qwen3", ["qwen3:latest"]) is True)
check("inst: タグ無し指定はタグ無し導入も許容",
      f.is_installed("qwen3", ["qwen3"]) is True)
check("inst: タグ無し指定は任意のタグ付き導入には一致しない",
      f.is_installed("qwen3", ["qwen3:4b"]) is False)
check("inst: タグ付き指定はタグ無し導入では満たされない(非対称)",
      f.is_installed("qwen3:latest", ["qwen3"]) is False)
check("inst: 空リストは未導入", f.is_installed("gpt-oss:20b", []) is False)
check("inst: 無関係な導入リストのみでは未導入",
      f.is_installed("gpt-oss:20b", ["phi4", "gemma4:26b"]) is False)

# ---------- 大VRAMプロファイル ----------
_hv_saved = (dict(f.MODEL_CONFIG), f.PARALLEL_PROPOSERS, f.SC_INITIAL, f.SC_MAX,
             f.SC_CHEAP_VOTES, f.MODEL_NUM_CTX)
try:
    _cfg_snapshot = {m: dict(c) for m, c in f.MODEL_CONFIG.items()}
    f.MODEL_CONFIG = {m: dict(c) for m, c in _cfg_snapshot.items()}
    f.apply_high_vram_profile()
    check("hv: 並列ON", f.PARALLEL_PROPOSERS is True)
    check("hv: SC上限を引き上げ", f.SC_MAX >= 40)
    check("hv: num_ctx拡大", f.model_cfg("gpt-oss:20b", "num_ctx") == 65536)
    check("hv: 安価票を有効化", f.SC_CHEAP_VOTES >= 8)
finally:
    (f.MODEL_CONFIG, f.PARALLEL_PROPOSERS, f.SC_INITIAL, f.SC_MAX,
     f.SC_CHEAP_VOTES, f.MODEL_NUM_CTX) = _hv_saved
_orig_pt = f.PROPOSER_THINK
try:
    f.PROPOSER_THINK = None
    check("cfg: think解決 グローバルNoneは設定値", f.proposer_think_for("gpt-oss:20b") == "high")
    f.PROPOSER_THINK = False
    check("cfg: think解決 グローバル優先", f.proposer_think_for("gpt-oss:20b") is False)
finally:
    f.PROPOSER_THINK = _orig_pt

# ---------- use_jp_aggregator ----------
check("jp: ひらがな", f.use_jp_aggregator("これはテストです"))
check("jp: カタカナ", f.use_jp_aggregator("テスト"))
check("jp: 漢字のみ(旧版の取りこぼし)", f.use_jp_aggregator("東京都の人口密度?"))
check("jp: 英語はFalse", not f.use_jp_aggregator("What is the capital of France?"))
check("jp: 空/None耐性", not f.use_jp_aggregator("") and not f.use_jp_aggregator(None))

# ---------- pick_aggregator（統合役の選定ルーティング） ----------
# has_code / 日本語 / それ以外 の各分岐と優先順位（コード > 日本語 > 推論型既定）を
# module globals をセンチネル値へ差し替えて検証する。JP_AGGREGATOR_STRONG / JP_AGGREGATOR /
# AGGREGATOR_REASONING / AGGREGATOR / CONDUCTOR / PROPOSERS はすべて try/finally で
# 復元し、後続チェック（aggregate 経由で pick_aggregator を間接的に叩く箇所を含む）に
# 汚染したグローバルを見せないようにする。
# 判定の曖昧さを避けるため、日本語入力はひらがな/カタカナのみ、非日本語入力はASCIIのみを使う
# （use_jp_aggregator の docstring が指摘する「漢字だけでは日中判別不能」問題を踏まない）。
_JP_TEXT = "これはテストです"   # ひらがな -> use_jp_aggregator は True
_EN_TEXT = "What is this?"    # ASCIIのみ -> use_jp_aggregator は False

_pa_orig_jp_strong = f.JP_AGGREGATOR_STRONG
_pa_orig_jp = f.JP_AGGREGATOR
_pa_orig_agg_reasoning = f.AGGREGATOR_REASONING
_pa_orig_agg = f.AGGREGATOR
_pa_orig_conductor = f.CONDUCTOR
_pa_orig_proposers = f.PROPOSERS
try:
    f.AGGREGATOR = "SENTINEL-AGGREGATOR-CODE"
    f.AGGREGATOR_REASONING = "SENTINEL-AGGREGATOR-REASONING"
    f.JP_AGGREGATOR_STRONG = "SENTINEL-JP-STRONG"
    f.JP_AGGREGATOR = "SENTINEL-JP-BASE"
    f.CONDUCTOR = "SENTINEL-CONDUCTOR"

    # (1) has_code=True は言語に関係なく常に AGGREGATOR
    f.PROPOSERS = []
    check("pick_agg: has_code優先(日本語入力でもAGGREGATOR)",
          f.pick_aggregator(_JP_TEXT, has_code=True) == f.AGGREGATOR)
    check("pick_agg: has_code優先(英語入力でもAGGREGATOR)",
          f.pick_aggregator(_EN_TEXT, has_code=True) == f.AGGREGATOR)

    # (2) 日本語 + JP_AGGREGATOR_STRONG が導入済み(PROPOSERSに存在) -> STRONGを返す
    f.PROPOSERS = [f.JP_AGGREGATOR_STRONG]
    check("pick_agg: 日本語+強JPモデル導入済みはSTRONGを返す",
          f.pick_aggregator(_JP_TEXT, has_code=False) == f.JP_AGGREGATOR_STRONG)

    # (3a) 日本語 + STRONGがfalsy(未設定) + JP_AGGREGATORがPROPOSERSに存在 -> JP_AGGREGATORを返す
    f.JP_AGGREGATOR_STRONG = None
    f.PROPOSERS = [f.JP_AGGREGATOR]
    check("pick_agg: 日本語+STRONG未設定はJP_AGGREGATORへ(導入済み)",
          f.pick_aggregator(_JP_TEXT, has_code=False) == f.JP_AGGREGATOR)

    # (3b) STRONGが値を持っていてもPROPOSERS未導入なら同様にJP_AGGREGATORへ落ちる
    f.JP_AGGREGATOR_STRONG = "SENTINEL-JP-STRONG"
    f.PROPOSERS = [f.JP_AGGREGATOR]  # STRONGはPROPOSERSに含まれない=未導入扱い
    check("pick_agg: 強JPモデルがPROPOSERS未導入ならJP_AGGREGATORへ",
          f.pick_aggregator(_JP_TEXT, has_code=False) == f.JP_AGGREGATOR)

    # (4) 日本語 + JP_AGGREGATORはPROPOSERS未導入だがCONDUCTORと一致 -> JP_AGGREGATORを返す
    f.JP_AGGREGATOR_STRONG = None
    f.PROPOSERS = ["SENTINEL-OTHER-MODEL"]
    f.CONDUCTOR = f.JP_AGGREGATOR
    check("pick_agg: JP_AGGREGATOR未導入でもCONDUCTORと一致すれば採用",
          f.pick_aggregator(_JP_TEXT, has_code=False) == f.JP_AGGREGATOR)

    # (5) 日本語 + JP系条件を何も満たさない -> 最終フォールバックのAGGREGATOR
    #     （現行docstring通りの意図的フォールバックであり、qwen3系のAGGREGATORも
    #     deepseek-r1の言語混入問題を踏まない選択なので、これは特性確認であってバグ報告ではない）
    f.CONDUCTOR = "SENTINEL-CONDUCTOR-OTHER"
    check("pick_agg: 日本語でJP系条件すべて不成立ならAGGREGATOR(特性確認)",
          f.pick_aggregator(_JP_TEXT, has_code=False) == f.AGGREGATOR)

    # (6) 非日本語 + AGGREGATOR_REASONINGが導入済み -> AGGREGATOR_REASONINGを返す
    f.PROPOSERS = [f.AGGREGATOR_REASONING]
    check("pick_agg: 非日本語は導入済みのAGGREGATOR_REASONINGへ",
          f.pick_aggregator(_EN_TEXT, has_code=False) == f.AGGREGATOR_REASONING)

    # (7) 非日本語 + AGGREGATOR_REASONINGが未導入 -> AGGREGATORへフォールバック
    f.PROPOSERS = ["SENTINEL-OTHER-MODEL"]
    check("pick_agg: 非日本語でAGGREGATOR_REASONING未導入はAGGREGATORへ",
          f.pick_aggregator(_EN_TEXT, has_code=False) == f.AGGREGATOR)

    # (8) 相互作用: 日本語だがhas_code=True -> コードがJP系より優先されAGGREGATORを返す
    f.JP_AGGREGATOR_STRONG = "SENTINEL-JP-STRONG"
    f.PROPOSERS = [f.JP_AGGREGATOR_STRONG, f.JP_AGGREGATOR]  # JP条件は満たせる状態にしておく
    f.CONDUCTOR = f.JP_AGGREGATOR
    check("pick_agg: 日本語でもhas_code=TrueならAGGREGATOR(コード優先)",
          f.pick_aggregator(_JP_TEXT, has_code=True) == f.AGGREGATOR)
finally:
    f.JP_AGGREGATOR_STRONG = _pa_orig_jp_strong
    f.JP_AGGREGATOR = _pa_orig_jp
    f.AGGREGATOR_REASONING = _pa_orig_agg_reasoning
    f.AGGREGATOR = _pa_orig_agg
    f.CONDUCTOR = _pa_orig_conductor
    f.PROPOSERS = _pa_orig_proposers

# ---------- aggregate のフォールバック（ask をモンキーパッチ） ----------
_orig_ask = f.ask
_ask_log = []


def _fake_ask_empty(model, messages, temperature, think=None, fmt=None,
                    label=None, num_predict=None):
    """アグリゲータ/再統合/Criticすべて空返答 → 保険2の最終分岐(最長の提案)まで落ちる。
    ※Critic は extract_json 失敗時 ok=True 既定なので、実際は最初の提案が返る。"""
    _ask_log.append((label, model, think))
    return ""


f.ask = _fake_ask_empty
try:
    out = f.aggregate("Q?", [("m1", "short"), ("m2", "much longer answer")])
finally:
    f.ask = _orig_ask
_agg_calls = [(m, th) for lab, m, th in _ask_log if lab == "aggregator"]
check("agg: 全滅時も提案のどれかを返す(空にしない)", out in ("short", "much longer answer"))
check("agg: 再統合(保険1)が試行されている", len(_agg_calls) == 2)
check("agg: 保険1は JP_AGGREGATOR + think=False で再統合",
      _agg_calls[1] == (f.JP_AGGREGATOR, False))

_ask_log.clear()


def _fake_ask_ok(model, messages, temperature, think=None, fmt=None,
                 label=None, num_predict=None):
    _ask_log.append((label, model, think))
    return "aggregated!"


f.ask = _fake_ask_ok
try:
    out = f.aggregate("Q?", [("m1", "a"), ("m2", "b")])
finally:
    f.ask = _orig_ask
check("agg: 正常時は統合結果を返す", out == "aggregated!")
check("agg: 正常時は1回だけ呼ぶ", len(_ask_log) == 1)

# エラー提案しかない場合
check("agg: 全プロポーザー失敗は__ERROR__",
      f.aggregate("Q?", [("m1", "__ERROR__: x")]).startswith("__ERROR__"))

# ---------- agg: 保険2(insurance-2)フォールバックが [Execution check: ...] を漏らさない ----------
# 2026-07-22: aggregate() は以前、コード付き提案に実行結果タグを付ける際に `good` 自体を
# タグ付き版で上書きしていた。保険2(統合失敗時に good から直接返す)経路は「主アグリゲータ」と
# 「JP_AGGREGATOR(think=False)再統合」の両方が空/エラーを返す場合に到達する
# (2026-07-04 の空返答実測に基づく既知の実運用経路)。このタグ/生トレースバックが
# ユーザー向け回答にそのまま漏れていたバグの回帰テスト。


def _fake_ask_always_empty(model, messages, temperature, think=None, fmt=None,
                            label=None, num_predict=None):
    """主アグリゲータ・再統合(保険1)ともに空/エラーを返し、保険2まで必ず落とす。"""
    return "" if label != "force_error" else "__ERROR__"


_orig_code_execution = f.CODE_EXECUTION
_orig_critique = f.critique
f.CODE_EXECUTION = True

_code_ans = "Here you go:\n\n```python\nprint(2 + 2)\n```\n"
_prose_ans = "This is a plain prose answer with no code block at all, just text."

try:
    # --- critique が最初の合格案をそのまま採用するケース ---
    f.ask = _fake_ask_always_empty
    f.critique = lambda question, answer: (True, "")
    out = f.aggregate("Q?", [("m1", _code_ans), ("m2", _prose_ans)])
    check("agg: 保険2(critique採用)はコード本文を含む", "print(2 + 2)" in out)
    check("agg: 保険2(critique採用)は[Execution check:]タグを漏らさない",
          "[Execution check:" not in out)

    # --- critique が全案を却下し、最長案(max fallback)まで落ちるケース ---
    # コード付き案をわざと最長にして、タグ付け前の `good`(クリーン版)から
    # 選ばれることを検証する。
    _code_ans_long = _code_ans + ("x" * 200)
    f.critique = lambda question, answer: (False, "no good")
    out2 = f.aggregate("Q?", [("m1", _prose_ans), ("m2", _code_ans_long)])
    check("agg: 保険2(最長fallback)は最長案(コード付き)を返す", "print(2 + 2)" in out2)
    check("agg: 保険2(最長fallback)は[Execution check:]タグを漏らさない",
          "[Execution check:" not in out2)
finally:
    f.ask = _orig_ask
    f.critique = _orig_critique
    f.CODE_EXECUTION = _orig_code_execution

# ---------- agg: 正常時、アグリゲータへの user プロンプトにはタグが残っていること ----------
# (AGGREGATOR_SYS ルール6はこのタグを判断材料にするため、アグリゲータ自身が見る
#  プロンプトからタグを消してはいけない。good を汚さない修正がここを壊していないことの回帰確認)
_captured_user = []


def _fake_ask_capture(model, messages, temperature, think=None, fmt=None,
                       label=None, num_predict=None):
    for msg in messages:
        if msg.get("role") == "user":
            _captured_user.append(msg.get("content", ""))
    return "aggregated!"


f.CODE_EXECUTION = True
f.ask = _fake_ask_capture
try:
    out3 = f.aggregate("Q?", [("m1", _code_ans), ("m2", _prose_ans)])
finally:
    f.ask = _orig_ask
    f.CODE_EXECUTION = _orig_code_execution
check("agg: 正常系はアグリゲータ出力をそのまま返す", out3 == "aggregated!")
check("agg: アグリゲータへのプロンプトには[Execution check: PASSED]が残る",
      any("[Execution check: PASSED]" in u for u in _captured_user))

# ---------- get_proposals の多様性（先頭はドラフト無しで新規回答） ----------
_seen_refs = []


def _fake_proposal(model, question, reference, issue=None, history=None):
    _seen_refs.append(reference)
    return model, "ans"


_orig_gsp = f.get_single_proposal
f.get_single_proposal = _fake_proposal
try:
    f.PARALLEL_PROPOSERS = False
    f.get_proposals(["m1", "m2", "m3"], "Q?", reference="draft", issue="x")
finally:
    f.get_single_proposal = _orig_gsp
check("prop: ラウンド2の先頭は新規回答(reference=None)", _seen_refs[0] is None)
check("prop: 2体目以降はドラフト改善", _seen_refs[1] == "draft" and _seen_refs[2] == "draft")

# ---------- get_proposals: PARALLEL_PROPOSERS=True でも多様性契約が保たれる ----------
# 2026-07-23: 並列分岐(get_single_proposalをThreadPoolExecutorで実行)でも、
# jobs[0](=models[0])だけがreference=Noneで呼ばれる契約が崩れていないことを確認する。
_seen_refs_parallel = {}


def _fake_proposal_parallel(model, question, reference, issue=None, history=None):
    _seen_refs_parallel[model] = reference
    return model, "ans"


_orig_gsp_par = f.get_single_proposal
_orig_parallel_flag_1 = f.PARALLEL_PROPOSERS
f.get_single_proposal = _fake_proposal_parallel
try:
    f.PARALLEL_PROPOSERS = True
    f.get_proposals(["m1", "m2", "m3"], "Q?", reference="draft", issue="x")
finally:
    f.get_single_proposal = _orig_gsp_par
    f.PARALLEL_PROPOSERS = _orig_parallel_flag_1
check("prop(parallel): 先頭(models[0])は新規回答(reference=None)",
      _seen_refs_parallel.get("m1") is None)
check("prop(parallel): 2体目以降はドラフト改善",
      _seen_refs_parallel.get("m2") == "draft" and _seen_refs_parallel.get("m3") == "draft")

# ---------- get_proposals: PARALLEL_PROPOSERS=True は completion順ではなく submission順で返す ----------
# 2026-07-23: as_completed()の完了順ではなく、futsリスト(=jobs=投入順)の順序で
# 結果を集めることを確認する。最初に投入するm1のジョブをthreading.Eventで
# 意図的にブロックし、最後に投入するm3が先に完了してイベントをセットするよう
# 強制することで、完了順(m3,m2,m1相当)とsubmission順(m1,m2,m3)が確実に食い違う
# 状況を作る。さらに、m1が実際にイベントを受信できた(timeoutしなかった)ことを
# 確認することで、「全ジョブ投入 → その後に結果収集」という並行性
# (apply_high_vram_profileが有効化する96GB構成でのwall-clock短縮の前提)が
# 壊れていないことも同時に検証する。
import threading as _threading

_order_flags = {}
_order_event = _threading.Event()


def _fake_proposal_ordered(model, question, reference, issue=None, history=None):
    if model == "m1":
        # 提出順では最初のジョブだが、m3がイベントをセットするまで完了しない
        # → 完了順ではm1が最後になる。もし「全ジョブ投入前に結果収集」に
        # 退行していればm3のジョブがまだ投入されておらずここでtimeoutする。
        _order_flags["m1_got_event"] = _order_event.wait(timeout=5)
    elif model == "m3":
        _order_event.set()
    return model, f"ans-{model}"


_orig_gsp_ord = f.get_single_proposal
_orig_parallel_flag_2 = f.PARALLEL_PROPOSERS
f.get_single_proposal = _fake_proposal_ordered
try:
    f.PARALLEL_PROPOSERS = True
    _out_ordered = f.get_proposals(["m1", "m2", "m3"], "Q?")
finally:
    f.get_single_proposal = _orig_gsp_ord
    f.PARALLEL_PROPOSERS = _orig_parallel_flag_2
check("prop(parallel): 全ジョブ投入後に収集される(m1がイベントを受信できた)",
      _order_flags.get("m1_got_event") is True)
check("prop(parallel): 完了順(m1が最後)ではなくsubmission順(m1,m2,m3)で返る",
      _out_ordered == [("m1", "ans-m1"), ("m2", "ans-m2"), ("m3", "ans-m3")])

# ---------- コード実行検証 ----------
check("code: python フェンス抽出", f.extract_code("x\n```python\nprint(1)\n```\ny") == "print(1)\n")
check("code: タグ無しフェンスも拾う", f.extract_code("```\nx = 1\n```") == "x = 1\n")
check("code: コード無しは None", f.extract_code("no code here") is None)

# 2026-07-22 回帰: 非python フェンス(```json 等)が先行しても、その閉じフェンスを
# 開始フェンスと誤認してブロック間のプロースを「コード」として誤抽出しないこと。
check(
    "code: jsonブロックの後のpythonブロックを正しく抽出",
    f.extract_code(
        "Here:\n```json\n{\"a\": 1}\n```\nNow code:\n```python\nprint(2+2)\n```"
    ) == "print(2+2)\n",
)
check(
    "code: text/outputブロックの後のpythonブロックを正しく抽出",
    f.extract_code(
        "```text\nsome output\n```\n説明\n```output\nmore output\n```\n"
        "```python\nprint(3+3)\n```"
    ) == "print(3+3)\n",
)
check(
    "code: 非pythonブロックのみなら None",
    f.extract_code("```json\n{\"a\": 1}\n```") is None,
)

# 2026-07-22 回帰: extract_code の言語タグ比較を「info string 全体」ではなく
# 「最初の空白区切りトークン」で行うよう修正（CommonMark の info string 仕様。
# iteration 7 のブロック選択修正、iteration 18 の _extract_code_for_output 修正に
# 続く、同じ誤抽出クラスの3件目）。装飾/メタデータ付きの python フェンスが誤って
# 読み飛ばされないことを確認する。
check(
    "code: python+装飾タグ({.line-numbers})も正しく抽出",
    f.extract_code("```python {.line-numbers}\nprint(1)\n```") == "print(1)\n",
)
check(
    "code: python+装飾タグ(title=...)も正しく抽出",
    f.extract_code("```python title=\"sol.py\"\nx = 1\n```") == "x = 1\n",
)
check(
    "code: py3タグは受理集合外のためNone(広げすぎていないことの確認)",
    f.extract_code("```py3\ncode\n```") is None,
)
check(
    "code: 先行する装飾非pythonフェンス(json {.foo})の後のpythonブロックを正しく抽出",
    f.extract_code(
        "```json {.foo}\n{\"a\":1}\n```\n```python\ncode\n```"
    ) == "code\n",
)
check(
    "code: ハイフン複合タグ(python-repl)は空白区切りが無いため受理されない",
    f.extract_code("```python-repl\ncode\n```") is None,
)

# 2026-07-22: _extract_code_for_output (_save_as_code が使うファイル出力用抽出) に
# iteration-7 の extract_code と同じ誤抽出クラスの修正を適用した回帰テスト。
check(
    "code_out: jsonブロックの後のbareフェンスからpythonを正しく抽出",
    f._extract_code_for_output(
        "```json\n{\"a\": 1}\n```\n```\ndef f():\n    return 42\n```", ".py"
    ) == "def f():\n    return 42\n",
)
check(
    "code_out: textブロック+ブロック間プロースの後のbareフェンスを正しく抽出",
    f._extract_code_for_output(
        "```text\nsome output\n```\n説明のプロース\n```\ndef g():\n    return 1\n```",
        ".py",
    ) == "def g():\n    return 1\n",
)
check(
    "code_out: python3タグ単体ブロックを抽出",
    f._extract_code_for_output("```python3\nprint('hi')\n```", ".py")
    == "print('hi')\n",
)
check(
    "code_out: jsonブロックに続くpython3ブロックを正しく抽出",
    f._extract_code_for_output(
        "```json\n{\"x\": 1}\n```\n```python3\nprint('hi')\n```", ".py"
    ) == "print('hi')\n",
)
check(
    "code_out: 単一pythonブロック(回帰・従来通り)",
    f._extract_code_for_output("```python\nprint(1)\n```", ".py") == "print(1)\n",
)
check(
    "code_out: 単一bareフェンス(回帰・従来通り)",
    f._extract_code_for_output("```\nx = 1\n```", ".py") == "x = 1\n",
)
check(
    "code_out: フェンス無しはマークダウン見出し除去にフォールバック(回帰・従来通り)",
    f._extract_code_for_output("# Title\nSome text\n# Another\nMore text", ".py")
    == "Some text\nMore text",
)
check(
    "code_out: 唯一のフェンスが別言語(c)の実コードなら保守的に採用(スキップリストで飲み込まない)",
    f._extract_code_for_output("```c\nint main(){return 0;}\n```", ".py")
    == "int main(){return 0;}\n",
)

# 2026-07-23 回帰: フェンス無しフォールバックが l.startswith("#") で '#' 始まりの
# 行を無条件に全削除していたため、--out file.<ext> で保存される素の(フェンス無し)
# コードに含まれる #include/#define/#pragma、シェバン行、Rust属性が「見出し」
# 扱いされて誤って削除されていたバグの修正確認 (iteration 7/18/28/29 が対処した
# フェンス選択側の兄弟分岐)。CommonMark の ATX heading (^#{1,6}(?:\s|$)) にのみ
# 一致する行を除去するよう修正した。
check(
    "code_out: フェンス無しのC #includeは見出し扱いされず保持される",
    "#include <stdio.h>"
    in f._extract_code_for_output(
        "#include <stdio.h>\nint main(void){return 0;}", ".c"
    ),
)
check(
    "code_out: フェンス無しの#define/#pragma/シェバン/Rust属性はすべて保持される",
    f._extract_code_for_output(
        "#!/usr/bin/env python\n"
        "#define X 1\n"
        "#pragma once\n"
        "#[derive(Debug)]\n"
        "code_line()",
        ".py",
    )
    == "#!/usr/bin/env python\n#define X 1\n#pragma once\n#[derive(Debug)]\ncode_line()",
)
check(
    "code_out: 7個以上の連続'#'はATX見出しとして無効なため保持される",
    "####### note"
    in f._extract_code_for_output("####### note\ncode_line()", ".py"),
)
check(
    "code_out: フェンス無しでも真のATX見出し(# Title/## Section)は従来通り除去される",
    f._extract_code_for_output(
        "# Title\ncode_a()\n## Section\ncode_b()", ".py"
    )
    == "code_a()\ncode_b()",
)

# 2026-07-22: iteration 28 の extract_code 修正 (info string は最初の空白区切り
# トークンのみを言語タグとする) を _extract_code_for_output にも追随適用した
# 回帰・新規テスト。装飾付き info string (```python title="sol.py" や
# ```python {.line-numbers}) が tier-1 の対象言語一致から漏れたり、装飾付き
# 非コードフェンス(```json {.line-numbers})が _NON_CODE_TAGS と不一致になって
# tier-3 で誤って採用されたりしないことを確認する。
check(
    "code_out: title=属性で装飾されたpythonフェンスもtier-1(対象言語一致)で抽出",
    f._extract_code_for_output(
        "```python title=\"sol.py\"\nprint(1)\n```", ".py"
    ) == "print(1)\n",
)
check(
    "code_out: {.line-numbers}属性で装飾されたpythonフェンスもtier-1で抽出",
    f._extract_code_for_output(
        "```python {.line-numbers}\nprint(2)\n```", ".py"
    ) == "print(2)\n",
)
check(
    "code_out: 装飾付き非コードフェンス(json {.foo})はNON_CODE_TAGSとして読み飛ばされ、後続pythonを抽出",
    f._extract_code_for_output(
        "```json {.foo}\n{\"a\":1}\n```\n\n```python\nprint(3)\n```", ".py"
    ) == "print(3)\n",
)

# 2026-07-23: _CODE_EXTENSIONS (25拡張子) と _extract_code_for_output 内 lang_map
# (13拡張子のみ) の同期漏れ修正の回帰・新規テスト。lang_map に無い残り12拡張子
# (.jsx .tsx .mjs .h .hpp .kt .swift .php .bat .ps1 .m .jl) は langs が空集合に
# なり tier-1 (対象言語タグ一致) が絶対に発火せず、非対象言語のブロック
# (例: 使い方説明の```bash)が先行すると tier-3 でそれを誤って採用してしまう
# (iteration 7/18/28/29/56 と同じブロック誤選択バグクラス)。実例として挙げられた
# 「```bash の npm install の後に本命の```jsx」で、修正前は 'npm install\n' が
# 返っていたことを確認しつつ、12拡張子それぞれの代表ケースで tier-1 が正しく
# 発火することを検証する。
check(
    "code_out: 先行するbashブロックの後のjsxブロックを正しく抽出(修正前はnpm installを誤抽出)",
    "export default"
    in f._extract_code_for_output(
        "```bash\nnpm install\n```\n"
        "```jsx\nexport default function App(){return null}\n```",
        ".jsx",
    )
    and "npm install"
    not in f._extract_code_for_output(
        "```bash\nnpm install\n```\n"
        "```jsx\nexport default function App(){return null}\n```",
        ".jsx",
    ),
)
check(
    "code_out: 先行するbashブロックの後のtsx(tsxタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```bash\nnpm install\n```\n```tsx\nconst x: number = 1;\n```", ".tsx"
    ) == "const x: number = 1;\n",
)
check(
    "code_out: 先行するbashブロックの後のtsx(typescriptタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```bash\nnpm install\n```\n```typescript\nconst y: number = 2;\n```", ".tsx"
    ) == "const y: number = 2;\n",
)
check(
    "code_out: 先行するshブロックの後のmjs(javascriptタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```sh\nnode -v\n```\n```javascript\nexport const z = 1;\n```", ".mjs"
    ) == "export const z = 1;\n",
)
check(
    "code_out: 先行するbashブロックの後のh(cタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```bash\ngcc --version\n```\n```c\nint add(int a, int b){return a+b;}\n```",
        ".h",
    ) == "int add(int a, int b){return a+b;}\n",
)
check(
    "code_out: 先行するbashブロックの後のhpp(cppタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```bash\ncmake .\n```\n```cpp\nint add(int a,int b){return a+b;}\n```",
        ".hpp",
    ) == "int add(int a,int b){return a+b;}\n",
)
check(
    "code_out: 先行するbashブロックの後のphp(phpタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```bash\ncomposer install\n```\n```php\necho 1;\n```", ".php"
    ) == "echo 1;\n",
)
check(
    "code_out: 先行するbashブロックの後のkt(kotlinタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```bash\ngradle build\n```\n```kotlin\nfun main() {}\n```", ".kt"
    ) == "fun main() {}\n",
)
check(
    "code_out: 先行するbashブロックの後のswift(swiftタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```bash\nswift build\n```\n```swift\nprint(\"hi\")\n```", ".swift"
    ) == "print(\"hi\")\n",
)
check(
    "code_out: 先行するshブロックの後のps1(powershellタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```sh\necho setup\n```\n```powershell\nGet-Process\n```", ".ps1"
    ) == "Get-Process\n",
)
check(
    "code_out: 先行するbashブロックの後のbat(batタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```bash\necho setup\n```\n```bat\n@echo off\n```", ".bat"
    ) == "@echo off\n",
)
check(
    "code_out: 先行するbashブロックの後のjl(juliaタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```bash\njulia --version\n```\n```julia\nprintln(\"hi\")\n```", ".jl"
    ) == "println(\"hi\")\n",
)
check(
    "code_out: 先行するbashブロックの後のm(matlabタグ)ブロックを正しく抽出",
    f._extract_code_for_output(
        "```bash\necho setup\n```\n```matlab\ndisp('hi')\n```", ".m"
    ) == "disp('hi')\n",
)

# 回帰: 新規マッピング拡張子でも単一ブロック/bareフェンス/フェンス無しATXフォール
# バックは従来のtier-2/tier-3/フェンス無し挙動のまま変わらないことを確認する。
check(
    "code_out: phpの単一ブロック(回帰・従来通り)",
    f._extract_code_for_output("```php\necho 1;\n```", ".php") == "echo 1;\n",
)
check(
    "code_out: jsx向けでもbareフェンス(タグ無し)はtier-2で従来通り抽出",
    f._extract_code_for_output("```\nconst a = 1;\n```", ".jsx") == "const a = 1;\n",
)
check(
    "code_out: swift向けでもフェンス無しはATX見出し除去フォールバックに従来通り落ちる",
    f._extract_code_for_output(
        "# Title\nlet a = 1\n## Section\nlet b = 2", ".swift"
    ) == "let a = 1\nlet b = 2",
)

# End-to-end: _save_as_code (--out <newly-mapped-suffix> の実書き込み経路) でも
# tier-1 が発火し、先行するbashブロックではなく対象言語ブロックが書き出される
# ことを一時ファイルへの実書き込みで確認する。Ollama/ネットワーク呼び出しは無い。
import tempfile as _cfo_tempfile
import pathlib as _cfo_pathlib
with _cfo_tempfile.TemporaryDirectory() as _cfo_dir:
    _cfo_root = _cfo_pathlib.Path(_cfo_dir)
    _cfo_out = _cfo_root / "app.tsx"
    f._save_as_code(
        _cfo_out,
        "```bash\nnpm install\n```\n```typescript\nexport const x: number = 1;\n```",
    )
    _cfo_content = _cfo_out.read_text(encoding="utf-8")
    check(
        "code_out/_save_as_code: .tsxへの実書き込みは先行bashではなく対象言語ブロックになる",
        _cfo_content == "export const x: number = 1;\n\n"
        and "npm install" not in _cfo_content,
    )

ok, out = f.run_python("print('hello_runner')")
check("code: 実行成功", ok and "hello_runner" in out)
ok, out = f.run_python("raise ValueError('boom')")
check("code: 例外を検知して traceback を返す", (not ok) and "boom" in out)
ok, out = f.run_python("while True:\n    pass", timeout=2)
check("code: 無限ループはタイムアウト", (not ok) and "TIMEOUT" in out)

# 2026-07-22 回帰: run_python は子プロセスの stdin を DEVNULL にするので、
# input() を呼ぶコードはタイムアウトまでハングせず即座に EOFError で失敗する
# （親の stdin を継承していた旧挙動では repl() の対話入力を子に奪われたり、
#   TTY/pipe/closed の違いで非決定的に振る舞ったりしていた）。
# timeout は「タイムアウトまで待っていない」ことを示すため意図的に長め(30s)にする。
ok_input, out_input = f.run_python("data = input()\nprint(data)", timeout=30)
check(
    "code: input()はDEVNULL stdinによりEOFErrorで即失敗しTIMEOUTしない",
    (not ok_input) and ("EOFError" in out_input) and ("TIMEOUT" not in out_input),
)

# regression guard: DEVNULL stdin が通常コードの成功経路を邪魔しないこと
ok_normal, out_normal = f.run_python("print('no_input_needed_here')")
check(
    "code: input()を使わない通常コードはDEVNULL化後も正常に成功",
    ok_normal and "no_input_needed_here" in out_normal,
)

# regression guard: stdout_only=True も DEVNULL化後、成功時はstdoutのみを返す(iteration 4挙動)
ok_normal_only, out_normal_only = f.run_python("print('clean_stdout_only')", stdout_only=True)
check(
    "code: stdout_only=Trueの成功経路はDEVNULL化後もstdoutのみ",
    ok_normal_only and out_normal_only.strip() == "clean_stdout_only",
)

# stdout_only: 既定(False)は stdout+stderr 結合のまま(バイトレベルで不変)。
# stdout_only=True かつ成功時は stdout のみ返し、stderr の警告文で末尾行が汚染されない。
_warn_code = (
    "import sys\n"
    "print('a warning', file=sys.stderr)\n"
    "print('42')\n"
)
ok_default, out_default = f.run_python(_warn_code)
check("code: stdout_only既定Falseはstderrも含む", ok_default and "a warning" in out_default
      and "42" in out_default)
ok_only, out_only = f.run_python(_warn_code, stdout_only=True)
check("code: stdout_only=Trueはstdoutのみ・最終行が正しい値",
      ok_only and "a warning" not in out_only and out_only.splitlines()[-1].strip() == "42")

# stdout_only=True でも失敗時(returncode!=0)は traceback 込みの結合出力を返す
# （code-repair loop がエラー内容を見えるようにするための回帰ガード）。
ok_fail_only, out_fail_only = f.run_python("raise ValueError('boom_stdout_only')", stdout_only=True)
check("code: stdout_only=Trueでも失敗時はtraceback付き結合出力",
      (not ok_fail_only) and "boom_stdout_only" in out_fail_only)

# _sc_sample の PoT 分岐: 生成コードが正しい答えを stdout に、警告を stderr に出すケースで、
# run_python(stdout_only=True) により警告文が投票を汚染しないことを確認する
# （2026-07-21 修正の回帰ガード。修正前は out.splitlines()[-1] が stderr の警告行になり得た）。
_orig_ask_pot = f.ask


def _fake_ask_pot(model, messages, temperature, think=None, fmt=None,
                   label=None, num_predict=None, num_ctx=None):
    return (
        "考え方の説明です。\n```python\n"
        "import sys\n"
        "print('RuntimeWarning: something noisy', file=sys.stderr)\n"
        "print(7)\n"
        "```\n"
    )


try:
    f.ask = _fake_ask_pot
    _pot_ans, _pot_text = f._sc_sample("m1", "1+2+4=?", "math", pot=True)
finally:
    f.ask = _orig_ask_pot
check("sc: PoT stdout_onlyで警告行ではなく印字された答えが投票になる",
      _pot_ans == f.normalize_answer("7"))

# ---------- _sc_sample: __ERROR__ センチネルは投票を汚染せずそのまま素通しする (2026-07-23) ----------
# ask() は失敗時 '__ERROR__: HTTP Error 500: Internal Server Error' のような文字列を返す
# (line ~1079)。_sc_sample の L2558 (`if raw.startswith("__ERROR__"): return None, raw`) は
# これを extract_final_answer/PoT実行より前段で弾いている。弾かなければ math の最終数値
# フォールバック(extract_final_answer, line ~2497)がエラーメッセージ中の '500' を確信ありの
# 1票として誤採用しうる。同種ガードは ask()自身(iter9)・_critic_judge/second_opinion(iter15)・
# _arbitrate(iter20)では回帰確認済みだったが、自己一貫性投票(gotcha#7)の最小単位である
# _sc_sample 自身は iteration 4 で PoT の happy path しか検証されておらず、非PoT分岐・mcq分岐・
# PoTのガード早期リターン(サブプロセス不起動)は無防備だった。ここで塞ぐ。
_orig_ask_scerr = f.ask
_orig_extract_code_scerr = f.extract_code
_orig_run_python_scerr = f.run_python

try:
    # (1) 非PoT math: __ERROR__ は数値 '500' に化けず、素の (None, raw) を返す
    f.ask = lambda *a, **k: "__ERROR__: HTTP Error 500: Internal Server Error"
    _ans_math_err, _text_math_err = f._sc_sample("m1", "1+1=?", "math", pot=False)
    check("sc-err: 非PoT mathの__ERROR__は'500'を誤採用せずanswer=None",
          _ans_math_err is None)
    check("sc-err: 非PoT mathの__ERROR__はtextに生のエラー文字列をそのまま返す",
          _text_math_err == "__ERROR__: HTTP Error 500: Internal Server Error")

    # (2) 非PoT mcq: エラー文中にA-E文字や数字が含まれていても選択肢を誤採用しない
    f.ask = lambda *a, **k: "__ERROR__: HTTP Error 400 for model A"
    _ans_mcq_err, _text_mcq_err = f._sc_sample("m1", "Which is correct?", "mcq", pot=False)
    check("sc-err: 非PoT mcqの__ERROR__は選択肢文字/数字を誤採用せずanswer=None",
          _ans_mcq_err is None)

    # (3) PoT math: ガードが extract_code/run_python より前段で早期returnし、
    # サブプロセスが一切起動されないことを証明する(到達すればAssertionErrorで即検知)。
    _pot_err_touched = []

    def _extract_code_forbidden_scerr(*a, **kw):
        _pot_err_touched.append("extract_code")
        raise AssertionError("__ERROR__ガード後にextract_codeへ到達してはならない")

    def _run_python_forbidden_scerr(*a, **kw):
        _pot_err_touched.append("run_python")
        raise AssertionError("__ERROR__ガード後にrun_pythonへ到達してはならない(subprocess起動)")

    f.extract_code = _extract_code_forbidden_scerr
    f.run_python = _run_python_forbidden_scerr
    f.ask = lambda *a, **k: "__ERROR__: HTTP Error 500: Internal Server Error"
    _ans_pot_err, _text_pot_err = f._sc_sample("m1", "1+1=?", "math", pot=True)
    check("sc-err: PoT分岐でも__ERROR__は(None, raw)を返す",
          _ans_pot_err is None
          and _text_pot_err == "__ERROR__: HTTP Error 500: Internal Server Error")
    check("sc-err: PoT分岐の__ERROR__はextract_code/run_pythonへ到達しない(subprocess不起動)",
          not _pot_err_touched)
finally:
    f.ask = _orig_ask_scerr
    f.extract_code = _orig_extract_code_scerr
    f.run_python = _orig_run_python_scerr

# regression: 正常な非PoT math応答は従来どおり\boxed{}から答えが採れる(ガードの過検知なし)
try:
    f.ask = lambda *a, **k: "計算しました。\\boxed{42}"
    _ans_ok_scerr, _text_ok_scerr = f._sc_sample("m1", "6*7=?", "math", pot=False)
finally:
    f.ask = _orig_ask_scerr
check("sc-err: 通常応答は__ERROR__ガードに過検知されず'42'を採票",
      _ans_ok_scerr == "42" and _text_ok_scerr == "計算しました。\\boxed{42}")

# ---------- _sc_sample: PoT分岐の4つの棄却ガードは無投票を返す (2026-07-23) ----------
# _sc_sample の PoT 分岐 (line ~2587-2598) には「怪しい PoT サンプルを黙って捨て、
# SC投票を汚染しない」ための4つの早期returnガードがある:
#   (a) extract_code(text) が None（```python フェンス無し）
#   (b) run_python の ok が False（実行時エラー=トレースバック）
#   (c) 実行はできたが stdout が空
#   (d) stdout 最終行が80文字を超える（自由記述の暴走を数値解答として誤採用しない）
# いずれも (None, text) を返し無投票となる契約（「無投票 > 誤投票」の精度優先方針、
# gotcha#7）。iteration 4 は PoT の happy path のみ、iteration 52 は __ERROR__ センチ
# ネルのみを検証しており（52自身のコメントが認める通り）、この4ガードはこれまで無防備
# だった。ここでは f.ask のみをモックして _sc_sample を直接呼ぶ（モデル/ネットワーク
# 呼び出しは一切発生しない）。(b)(c)(d) は run_python に本物の軽量・決定的なローカル
# subprocess（1/0 や print(...) 程度）を実際に実行させる — iteration 4/46/52 と同じ流儀。
_orig_ask_potguard = f.ask

# (a) コードフェンス無し: extract_code が None を返し、run_python は起動されない
#     （到達すれば forbidden 関数が AssertionError を送出し、即座に検知できる）。
_pot_nocode_touched = []


def _run_python_forbidden_potguard_a(*a, **kw):
    _pot_nocode_touched.append("run_python")
    raise AssertionError("コードフェンス無しなのにrun_pythonへ到達してはならない(subprocess起動)")


_orig_run_python_potguard_a = f.run_python
try:
    f.run_python = _run_python_forbidden_potguard_a
    f.ask = lambda *a, **k: "説明だけの回答です。コードブロックはありません。答えはたぶん42。"
    _ans_nocode, _text_nocode = f._sc_sample("m1", "1+1=?", "math", pot=True)
finally:
    f.ask = _orig_ask_potguard
    f.run_python = _orig_run_python_potguard_a
check("sc-pot: (a)コードフェンス無し→extract_code=Noneでanswer=None(無投票)",
      _ans_nocode is None)
check("sc-pot: (a)コードフェンス無し→run_pythonへ到達しない(subprocess不起動)",
      not _pot_nocode_touched)

# (b) 実行時エラー(ZeroDivisionError): run_python の ok=False → answer=None。
#     注意: ok=False 分岐の _sc_sample は「text + run_python出力」を連結せず、素の
#     text（LLMが書いたコード込みの生テキスト）をそのまま返す契約（連結は成功時のみ）。
#     よって「tracebackの数字が誤ってanswerに混入しない」ことは _sc_sample の戻り値
#     answer が None であること自体で保証される。ここではまず run_python を直接呼び、
#     1/0 の実行が本当に失敗し(ok=False)、その出力(traceback)に数字が実在すること
#     （テスト前提の健全性確認＝ガードが無ければ数字が採用され得た状況であること）を
#     確認したうえで、_sc_sample がそれを answer=None として無投票に倒すことを検証する。
_ok_sanity_execfail, _out_sanity_execfail = f.run_python("1 / 0", stdout_only=True)
check("sc-pot: (b)前提確認: 1/0の実行は失敗し(ok=False)traceback出力に数字が実在する",
      (not _ok_sanity_execfail) and any(ch.isdigit() for ch in _out_sanity_execfail))
try:
    f.ask = lambda *a, **k: "計算コードです。\n```python\n1 / 0\n```\n"
    _ans_execfail, _text_execfail = f._sc_sample("m1", "1/0 は何？", "math", pot=True)
finally:
    f.ask = _orig_ask_potguard
check("sc-pot: (b)実行失敗(ZeroDivisionError, ok=False)はtraceback中の数字を誤採用せずanswer=None",
      _ans_execfail is None)

# (c) 実行は成功するが stdout が空: ok=True だが out=="" → answer=None
try:
    f.ask = lambda *a, **k: "何も出力しないコードです。\n```python\nx = 1 + 1\n```\n"
    _ans_emptyout, _text_emptyout = f._sc_sample("m1", "1+1=?", "math", pot=True)
finally:
    f.ask = _orig_ask_potguard
check("sc-pot: (c)stdoutが空(print無し)はanswer=None(無投票)",
      _ans_emptyout is None)

# (d) stdout 最終行が80文字を超える: 自由記述の暴走を「解答」として誤採用しない
_long_line = "9" * 90
try:
    f.ask = lambda *a, **k: "長い出力のコードです。\n```python\nprint('" + _long_line + "')\n```\n"
    _ans_toolong, _text_toolong = f._sc_sample("m1", "何か計算して", "math", pot=True)
finally:
    f.ask = _orig_ask_potguard
check("sc-pot: (d)前提確認: 生成した最終行は80文字を超える",
      len(_long_line) > 80)
check("sc-pot: (d)stdout最終行が80文字超はanswer=None(無投票、暴走出力を誤採用しない)",
      _ans_toolong is None)

# regression: 短い答えを最終行に印字 + 無関係なstderr警告、という正常系は上記4ガード
# 追加後も従来通り採票される（過検知が無いことの確認）。stdout_only=Trueによるstderr
# 除去自体は2026-07-21の別テスト（line ~2028）で既に確認済みだが、ここでは本セクション
# 自身のモック配線内でも happy path が壊れていないことを直接ロックする。
try:
    f.ask = lambda *a, **k: (
        "考え方の説明です。\n```python\n"
        "import sys\n"
        "print('DeprecationWarning: ignore me', file=sys.stderr)\n"
        "print(99)\n"
        "```\n"
    )
    _ans_happy_potguard, _text_happy_potguard = f._sc_sample("m1", "regression", "math", pot=True)
finally:
    f.ask = _orig_ask_potguard
check("sc-pot: regression: 短い答え+無関係なstderr警告は従来通り採票される(4ガード追加後も過検知なし)",
      _ans_happy_potguard == f.normalize_answer("99"))

# ---------- _representative_text: SC勝者クラスの代表テキスト選出 (2026-07-23) ----------
# solve_verifiable の最終return直前(line ~2788)でres['text']を決める、自己一貫性投票
# 経路(gotcha#7)で最後まで直接テストされていなかったヘルパー。選出順序は
# 「CoT(pot=False)優先 → 無ければ最初に一致したPoTサンプル → それも無ければ全サンプル中
# 最長」。一致判定は文字列==ではなくanswers_equivalent経由（表記違いの同値も拾う）。
# iteration 2 のchangelogで「代表テキストが敗者候補の主張になりうる」懸念が挙がって
# いた箇所（_arbitrateのrep選出とは別に、_representative_text自身の契約）を直接ロックする。
def _rt_sample(answer, text, pot=False, model="m1"):
    return {"answer": answer, "text": text, "pot": pot, "model": model}


# (1) CoT優先: リスト上で先に出てくるPoT一致サンプルより、後から出てくるCoT一致サンプルの
#     テキストを返す（"pot でない"優先はリストの並び順に依存しないことを証明する）。
_rt1 = [
    _rt_sample("42", "POT_TEXT_EARLIER", pot=True),
    _rt_sample("42", "COT_TEXT_LATER", pot=False),
]
check("rt: CoT優先(先行PoT一致より後続CoT一致を採用、順序非依存)",
      f._representative_text(_rt1, "42") == "COT_TEXT_LATER")

# (2) PoTフォールバック: CoT一致が一つも無い場合は最初に一致したPoTサンプルのテキストを返す
_rt2 = [
    _rt_sample("7", "POT_TEXT_NOMATCH", pot=True),      # 不一致(勝者は42)
    _rt_sample("42", "POT_TEXT_FIRST_MATCH", pot=True),
    _rt_sample("42", "POT_TEXT_SECOND_MATCH", pot=True),
]
check("rt: CoT一致無し→最初に一致したPoTサンプルのテキストを返す",
      f._representative_text(_rt2, "42") == "POT_TEXT_FIRST_MATCH")

# (3) 同値判定: 勝者'0.5'に対しサンプルの答えが'1/2'(Fractionによる高速パス、math_verifyは
#     不要)でも一致とみなし、文字列==ではなくanswers_equivalent経由でマッチしていることを示す
_rt3 = [_rt_sample("1/2", "FRACTION_MATCH_TEXT", pot=False)]
check("rt: 同値判定(1/2 と 0.5、Fraction高速パス)でマッチしテキストを返す",
      f._representative_text(_rt3, "0.5") == "FRACTION_MATCH_TEXT")

# (4) 敗者除外: 勝者と一致するサンプルのテキストが返り、別の敗者候補のテキストは
#     決して返らない(iteration 2 changelogの「敗者候補の主張になりうる」懸念の直接ガード)
_rt4 = [
    _rt_sample("7", "LOSER_TEXT", pot=False),
    _rt_sample("42", "WINNER_TEXT", pot=False),
]
_rt4_result = f._representative_text(_rt4, "42")
check("rt: 勝者一致サンプルのテキストを返す(敗者除外 1/2)", _rt4_result == "WINNER_TEXT")
check("rt: 敗者候補のテキストは返らない(敗者除外 2/2)", _rt4_result != "LOSER_TEXT")

# (5) 無一致フォールバック: どのサンプルも勝者と同値でない場合は全サンプル中最長の
#     テキストを返す(max-by-len分岐)。空リストは""を返す。
_rt5 = [
    _rt_sample("7", "short", pot=False),
    _rt_sample("9", "much much longer non-matching text", pot=False),
]
check("rt: 無一致→全サンプル中最長のテキストにフォールバック",
      f._representative_text(_rt5, "999") == "much much longer non-matching text")
check("rt: 空サンプルリストは空文字列を返す", f._representative_text([], "42") == "")

# (6) answer=Noneのサンプルは一致判定の対象から除外される(最長でも誤って採用されない)。
#     ただし全体が無一致で最長フォールバックに落ちた場合は、その候補にもなりうる
#     (max-by-len分岐はanswerの有無を見ないため)。
_rt6a = [
    _rt_sample(None, "NONE_ANSWER_BUT_LONGEST_TEXT_AAAAAAAAAAAAAAAA", pot=False),
    _rt_sample("42", "SHORT_MATCH_TEXT", pot=False),
]
check("rt: answer=Noneのサンプルは一致判定から除外(最長でも採用されない)",
      f._representative_text(_rt6a, "42") == "SHORT_MATCH_TEXT")

_rt6b = [
    _rt_sample(None, "NONE_ANSWER_LONGEST_FALLBACK_TEXT_AAAAAAAAAAAAAA", pot=False),
    _rt_sample("7", "short_nomatch", pot=False),
]
check("rt: 無一致時に限り、answer=Noneのサンプルも最長フォールバックの対象になりうる",
      f._representative_text(_rt6b, "999") == "NONE_ANSWER_LONGEST_FALLBACK_TEXT_AAAAAAAAAAAAAA")

check("code: code_check 正常コードは None", f.code_check("```python\nprint(1)\n```") is None)
_issue = f.code_check("```python\n1/0\n```")
check("code: code_check 失敗はエラー要約", _issue is not None and "ZeroDivision" in _issue)
check("code: コード無し回答は None", f.code_check("plain text answer") is None)

# 2026-07-22 回帰: 先行する非pythonブロック(```json)があっても code_check が
# ブロック間のプロースではなく実物の python を検証すること（見せかけの失敗を防ぐ）。
_issue_leading_json = f.code_check(
    "```json\n{\"note\": \"ignore me\"}\n```\n```python\n1/0\n```"
)
check(
    "code: 先行jsonブロックがあってもpythonの失敗を正しく検知",
    _issue_leading_json is not None and "ZeroDivision" in _issue_leading_json,
)
check(
    "code: 先行jsonブロック+正しいpythonはNone",
    f.code_check(
        "```json\n{\"note\": \"ignore me\"}\n```\n```python\nprint(1)\n```"
    ) is None,
)

# 2026-07-22 回帰: 先行する装飾付き非codeブロック(```json {.foo})があっても、
# code_check が info string の最初のトークンで正しく python ブロックを見つけ、
# 見せかけの実行失敗("code execution FAILED")を報告しないこと。
check(
    "code: 先行装飾jsonブロック+正しいpythonはNone(見せかけの失敗なし)",
    f.code_check(
        "```json {.foo}\n{\"note\": \"ignore me\"}\n```\n```python\nprint(1)\n```"
    ) is None,
)
_issue_leading_decorated_json = f.code_check(
    "```json {.foo}\n{\"note\": \"ignore me\"}\n```\n```python\n1/0\n```"
)
check(
    "code: 先行装飾jsonブロックがあってもpythonの失敗を正しく検知",
    _issue_leading_decorated_json is not None
    and "ZeroDivision" in _issue_leading_decorated_json,
)

_good_fib = ("説明します。\n```python\n"
             "def fib(n):\n"
             "    a, b = 1, 1\n"
             "    for _ in range(n - 1):\n"
             "        a, b = b, a + b\n"
             "    return a\n\n"
             "assert fib(10) == 55\n"
             "```")
_bad_fib = "```python\ndef fib(n):\n    return n\n```"
check("eval: fib 正解コード→OK", e.grade_code_fib(_good_fib) is True)
check("eval: fib 誤りコード→NG", e.grade_code_fib(_bad_fib) is False)
check("eval: コード無し回答→NG", e.grade_code_fib("fib(10)は55です") is False)

# ---------- eval_fugu の採点 ----------
check("eval: has_num 境界", e.has_num("answer is 391.", "391") and not e.has_num("3910", "391"))
check("eval: has_num 部分数字を弾く", not e.has_num("17 and 13", "7"))
check("eval: batball 0.05のみ→OK", e.grade_batball("The ball costs $0.05.") is True)
check("eval: batball 0.05なし→NG", e.grade_batball("The bat costs $1.05.") is False)

# ---------- second_opinion のバイアス対策（PROPOSERS から除外したケース） ----------
_orig_proposers = f.PROPOSERS
_orig_second_opinion_model = f.SECOND_OPINION_MODEL
_orig_disabled_flag = f._SECOND_OPINION_DISABLED
try:
    f.PROPOSERS = ["qwen3:4b"]  # phi4-mini を除外
    f.SECOND_OPINION_MODEL = "phi4-mini"
    f._SECOND_OPINION_DISABLED = False
    ok, issue = f.second_opinion("test", "test answer")
    check("so: PROPOSERS外のモデルは ok=True で即返す", ok is True and issue == "")
    check("so: 無効化フラグがセットされる", f._SECOND_OPINION_DISABLED is True)
finally:
    f.PROPOSERS = _orig_proposers
    f.SECOND_OPINION_MODEL = _orig_second_opinion_model
    f._SECOND_OPINION_DISABLED = _orig_disabled_flag

# ---------- _critic_judge / second_opinion: __ERROR__ センチネルは ok=False (2026-07-22) ----------
# ask() が通信/モデル失敗で '__ERROR__:...' を返したとき、extract_json は None になり
# 旧実装は p.get("ok", True) で黙って ok=True（審査合格）にしてしまっていた。
# critic 呼び出し自体が失敗しているだけなのに「回答は問題なし」と誤判定するのは
# 精度優先の方針に反するため、__ERROR__ センチネルだけを ok=False に反転させる。
# 一方、空文字や非JSONの地の文（gpt-oss:20b の think 予算切れ等）は既存どおり
# ok=True 既定を維持する必要があり、それも合わせて回帰確認する。
_orig_ask = f.ask
try:
    f.ask = lambda *a, **k: "__ERROR__: simulated transport failure"
    ok, issue = f._critic_judge("q", "a", think=False)
    check("critic: __ERROR__センチネル(think=False)はok=False", ok is False and bool(issue))
    ok, issue = f._critic_judge("q", "a", think=True)
    check("critic: __ERROR__センチネル(think=True)もok=False", ok is False and bool(issue))

    f.ask = lambda *a, **k: ""
    ok, issue = f._critic_judge("q", "a", think=False)
    check("critic: 空文字は既定どおりok=True(gpt-oss think予算切れ対策を維持)", ok is True)

    f.ask = lambda *a, **k: "Looks fine to me, no issues here."
    ok, issue = f._critic_judge("q", "a", think=False)
    check("critic: 非JSONの地の文も既定どおりok=True", ok is True)
finally:
    f.ask = _orig_ask

# ---------- critique(): 2段階エスカレーションの直接テスト (iter 58) ----------
# critique() は fugu_answer() の MoA 継続判定(need_more = not ok, fugu_local.py:2929)と
# aggregate() の保険2採用判定(fugu_local.py:2245)の両方を左右するが、既存テストは
# critique 自体を lambda で丸ごと差し替えて使っており(前掲の agg 保険2テスト参照)、
# 本物の2段階ロジック(1段目: think=False+スキーマで高速判定、ok ならそこで確定。
# 2段目: 1段目が NG のときだけ think=True で再検算し、その結果を最終判定とする)は
# 一度も直接実行されていなかった。この2段階構成は 2026-07-03 のフル評価で実測された
# 偽エスカレーション(think=False critic が正答 '700' を誤って NG にし 310秒浪費)への
# 対策であり、以下の不変条件を直接ロックする:
#   (1) think=False が ok なら think=True は一切呼ばずに (True, "") で確定(高速パス短絡)
#   (2) think=False が NG のときだけ think=True 再検算が発火し、その結果(issue含む)が
#       そのまま最終判定になる(権威は think=True 側であって think=False の issue ではない)
#   (3) think=True 側が _critic_judge の __ERROR__ センチネル形(ok=False, "critic call
#       failed: ...")を返した場合でも、critique() は黙って ok=True にせず reject する
# f._critic_judge をモックし、think 引数(キーワード呼び出し)で分岐・記録することで検証する。
# ask() 自体は一切呼ばない(_critic_judge の一段上を差し替えるため、実運用の think=False/True
# ラベル分岐が critique() 側で正しく起きているかだけを見る)。
_orig_critic_judge = f._critic_judge


def _make_critique_judge(ok1, issue1, ok2=False, issue2="", calls=None):
    """think 引数(bool化)で呼び出しを分岐・記録するモック。calls に think 値を記録する。"""
    if calls is None:
        calls = []

    def _judge(question, answer, think=False):
        calls.append(bool(think))
        if think:
            return ok2, issue2
        return ok1, issue1
    _judge.calls = calls
    return _judge


try:
    # (1) think=False が ok → 即 (True, "") で確定し、think=True は呼ばれない(高速パス短絡)。
    _calls = []
    f._critic_judge = _make_critique_judge(ok1=True, issue1="unused fast-path issue",
                                            calls=_calls)
    ok, issue = f.critique("2+2?", "4")
    check("critique: think=False合格は(True,'')で確定", ok is True and issue == "")
    check("critique: think=False合格時はthink=Trueを呼ばない(高速パス短絡)",
          _calls == [False])

    # (2a) think=False NG → think=True 再検算が発火し、think=True 側が ok なら採用。
    _calls = []
    f._critic_judge = _make_critique_judge(ok1=False, issue1="fast doubt",
                                            ok2=True, issue2="", calls=_calls)
    ok, issue = f.critique("700は正しいか?", "700")
    check("critique: think=False NGでthink=True再検算が発火する", _calls == [False, True])
    check("critique: think=True再検算がokなら(True,'')で採用", ok is True and issue == "")

    # (2b) think=False NG → think=True も NG。最終issueはthink=True側(issue3)がそのまま
    #      使われ、think=False側のissue1は使われない(権威は再検算側であることの確認)。
    _calls = []
    f._critic_judge = _make_critique_judge(ok1=False, issue1="fast doubt",
                                            ok2=False, issue2="authoritative issue",
                                            calls=_calls)
    ok, issue = f.critique("q", "a")
    check("critique: think=True再検算もNGならreject(False)", ok is False)
    check("critique: 最終issueはthink=True側(権威)がそのまま使われる",
          issue == "authoritative issue")

    # (3) think=True 側が _critic_judge の __ERROR__ センチネル形(ok=False, "critic call
    #     failed: ...")を返す場合。critic 呼び出し自体の失敗であって「回答に問題なし」
    #     ではないため、critique() はこれを ok=True に握りつぶさず reject のまま返す必要がある。
    _calls = []
    f._critic_judge = _make_critique_judge(
        ok1=False, issue1="fast doubt",
        ok2=False, issue2="critic call failed: __ERROR__: simulated transport failure",
        calls=_calls)
    ok, issue = f.critique("q", "a")
    check("critique: think=True側がcritic呼び出し失敗センチネルでもok=Trueに握りつぶさない",
          ok is False)
finally:
    f._critic_judge = _orig_critic_judge

_orig_proposers = f.PROPOSERS
_orig_second_opinion_model = f.SECOND_OPINION_MODEL
_orig_disabled_flag = f._SECOND_OPINION_DISABLED
_orig_ask = f.ask
try:
    # SECOND_OPINION_MODEL を PROPOSERS に含めて「有効」経路を通す。
    f.SECOND_OPINION_MODEL = "phi4-mini"
    f.PROPOSERS = ["phi4-mini", "qwen3:4b"]
    f._SECOND_OPINION_DISABLED = False

    f.ask = lambda *a, **k: "__ERROR__: simulated transport failure"
    ok, issue = f.second_opinion("q", "a")
    check("so: __ERROR__センチネルはok=False", ok is False and bool(issue))

    f.ask = lambda *a, **k: ""
    ok, issue = f.second_opinion("q", "a")
    check("so: 空文字は既定どおりok=True", ok is True)

    f.ask = lambda *a, **k: "Looks fine to me, no issues here."
    ok, issue = f.second_opinion("q", "a")
    check("so: 非JSONの地の文も既定どおりok=True", ok is True)

    # 無効化パス(SECOND_OPINION_MODEL not in PROPOSERS)は ask を一切呼ばずに (True, "") を返す。
    _calls = []
    f.PROPOSERS = ["qwen3:4b"]  # phi4-mini を除外
    f.ask = lambda *a, **k: _calls.append(1) or "__ERROR__: should not be reached"
    ok, issue = f.second_opinion("q", "a")
    check("so: 無効化パスはaskを呼ばずok=True", ok is True and issue == "" and not _calls)
finally:
    f.PROPOSERS = _orig_proposers
    f.SECOND_OPINION_MODEL = _orig_second_opinion_model
    f._SECOND_OPINION_DISABLED = _orig_disabled_flag
    f.ask = _orig_ask

# ---------- verify_single: think=True 最終審判の __ERROR__ は MoA へエスカレーション (2026-07-22) ----------
# verify_single は高速チェックのどちらかが疑義を出したときだけ think=True 再検算を
# 最終審判にする。その think=True 呼び出し自体が __ERROR__ で失敗した場合、_critic_judge の
# 修正により ok=False になるはずで、verify_single はそれを受けて False を返し MoA パネルへの
# 格上げを引き起こす（黙って True を返して壊れた回答を採用しない）ことを確認する。
_orig_proposers = f.PROPOSERS
_orig_second_opinion_model = f.SECOND_OPINION_MODEL
_orig_disabled_flag = f._SECOND_OPINION_DISABLED
_orig_ask = f.ask
try:
    # second_opinion を無効化パスに固定し、think=True 最終審判の挙動だけを見る。
    f.PROPOSERS = ["qwen3:4b"]
    f.SECOND_OPINION_MODEL = "phi4-mini"
    f._SECOND_OPINION_DISABLED = False

    def _fake_ask_escalate(model, messages, temperature, think=None, fmt=None,
                            label=None, num_predict=None, num_ctx=None):
        if think:
            return "__ERROR__: simulated transport failure"
        return json.dumps({"ok": False, "issue": "fast check flagged"})
    f.ask = _fake_ask_escalate
    ok, issue = f.verify_single("2+2?", "4")
    check("verify_single: think=True最終審判の__ERROR__はMoAへ格上げ(ok=False)",
          ok is False and bool(issue))

    def _fake_ask_control(model, messages, temperature, think=None, fmt=None,
                           label=None, num_predict=None, num_ctx=None):
        if think:
            return json.dumps({"ok": True, "issue": ""})
        return json.dumps({"ok": False, "issue": "fast check flagged"})
    f.ask = _fake_ask_control
    ok, issue = f.verify_single("2+2?", "4")
    check("verify_single: think=True最終審判が正常なJSONなら採用(control)",
          ok is True and issue == "")
finally:
    f.PROPOSERS = _orig_proposers
    f.SECOND_OPINION_MODEL = _orig_second_opinion_model
    f._SECOND_OPINION_DISABLED = _orig_disabled_flag
    f.ask = _orig_ask

# ---------- verify_single: second_opinion 有効時（自己評価バイアス対策 seam, 2026-07-23）----------
# 上の 2 ブロック（second_opinion のバイアス対策 / verify_single の think=True 最終審判）は
# どちらも PROPOSERS=["qwen3:4b"] のみで SECOND_OPINION_MODEL(既定 gpt-oss:20b、ここでは
# phi4-mini) を含めていない。そのため second_opinion() は毎回 (True, "") を即返して
# _SECOND_OPINION_DISABLED=True に自己ラッチするだけで、ask を一度も呼ばない
# 「独立チェック無効」分岐しか通っていなかった。verify_single の目玉である
# 「2系統独立チェック（qwen3 think=False の自己批評 と 別モデル phi4-mini の独立チェック）を
# 両方揃えたときの高速一致即採用」と「片方だけが疑義を出したら think=True 最終審判に
# 格上げする自己評価バイアス対策そのもの」は一度も検証されていなかった。
# ここでは PROPOSERS に SECOND_OPINION_MODEL を含めて「有効」経路を通し、label
# ("critic"/"critic2") と think の組み合わせで 3 種類の呼び出し
# （fast self-critic=critic/think=False, 独立second opinion=critic2, think=True最終審判=
# critic/think=True）を区別してモックする。fast self-critic と second_opinion はどちらも
# think を渡さない（falsy）ため、think だけでは区別できず label 分岐が必須になる点に注意。
_orig_proposers = f.PROPOSERS
_orig_second_opinion_model = f.SECOND_OPINION_MODEL
_orig_disabled_flag = f._SECOND_OPINION_DISABLED
_orig_ask = f.ask


def _make_vs_ask(calls, ok1, issue1, ok2, issue2, ok3=False, issue3=""):
    """label と think(bool化) で 3種の呼び出しを切り分けて記録するモック。
    calls には (label, think) の実呼び出しログを積む（呼ばれた事実自体をテストで検証する）。"""
    def _ask(model, messages, temperature, think=None, fmt=None, label=None,
             num_predict=None, num_ctx=None):
        calls.append((label, bool(think)))
        if label == "critic" and think:
            return json.dumps({"ok": ok3, "issue": issue3})
        if label == "critic2":
            return json.dumps({"ok": ok2, "issue": issue2})
        return json.dumps({"ok": ok1, "issue": issue1})
    return _ask


try:
    f.PROPOSERS = ["phi4-mini", "qwen3:4b"]  # SECOND_OPINION_MODEL を含める→有効経路
    f.SECOND_OPINION_MODEL = "phi4-mini"

    # (A) ok1=True かつ ok2=True: 高速2系統が一致→即採用。think=True 最終審判は
    #     呼ばれないはず（高速パスの短絡が壊れていないことをロックする）。
    f._SECOND_OPINION_DISABLED = False
    _calls = []
    f.ask = _make_vs_ask(_calls, ok1=True, issue1="", ok2=True, issue2="")
    ok, issue = f.verify_single("2+2?", "4")
    check("vs(有効) A: ok1=True/ok2=True→即採用", ok is True and issue == "")
    check("vs(有効) A: second_opinionが実際に呼ばれた(critic2が無効化せず発火)",
          any(lbl == "critic2" for lbl, _th in _calls))
    check("vs(有効) A: 高速一致時はthink=True最終審判を呼ばない(短絡ロック)",
          not any(lbl == "critic" and th for lbl, th in _calls))

    # (B) ok1=True(fast self-critic は合格) だが ok2=False(独立モデルが疑義)。
    #     自己評価バイアス対策の本丸：片方だけの疑義でも think=True 最終審判へ格上げされる。
    #   (B1) 最終審判が ok→採用(True)。
    f._SECOND_OPINION_DISABLED = False
    _calls = []
    f.ask = _make_vs_ask(_calls, ok1=True, issue1="", ok2=False, issue2="second opinion doubt",
                          ok3=True, issue3="")
    ok, issue = f.verify_single("2+2?", "4")
    check("vs(有効) B1: ok1=True/ok2=False→think=True最終審判が呼ばれる",
          any(lbl == "critic" and th for lbl, th in _calls))
    check("vs(有効) B1: 最終審判okなら採用(True)", ok is True and issue == "")

    #   (B2) 最終審判もNG→不採用(False)。理由は最終審判(issue3)が権威として使われる。
    f._SECOND_OPINION_DISABLED = False
    _calls = []
    f.ask = _make_vs_ask(_calls, ok1=True, issue1="", ok2=False, issue2="second opinion doubt",
                          ok3=False, issue3="final check: real issue found")
    ok, issue = f.verify_single("2+2?", "4")
    check("vs(有効) B2: 最終審判もNGなら不採用(False)でissue非空",
          ok is False and bool(issue))
    check("vs(有効) B2: 不採用理由はthink=True最終審判が権威(issue3を採用)",
          issue == "final check: real issue found")

    # (C) ok1=False(fast self-critic が疑義) だが ok2=True(独立モデルは合格)。
    #     こちらの片方だけの疑義でも think=True 最終審判へ格上げされることを確認する。
    #     最終審判がNGでissue3が空文字のときは doubt(=issue1) にフォールバックする。
    f._SECOND_OPINION_DISABLED = False
    _calls = []
    f.ask = _make_vs_ask(_calls, ok1=False, issue1="fast critic doubt", ok2=True, issue2="",
                          ok3=False, issue3="")
    ok, issue = f.verify_single("2+2?", "4")
    check("vs(有効) C: ok1=False/ok2=True→think=True最終審判が呼ばれる(doubtはissue1側)",
          any(lbl == "critic" and th for lbl, th in _calls))
    check("vs(有効) C: 最終審判issue3が空ならdoubt(issue1)にフォールバック",
          ok is False and issue == "fast critic doubt")
finally:
    f.PROPOSERS = _orig_proposers
    f.SECOND_OPINION_MODEL = _orig_second_opinion_model
    f._SECOND_OPINION_DISABLED = _orig_disabled_flag
    f.ask = _orig_ask

# ---------- bench_queue: 異常終了コード分類（gotcha 8 再発防止, 2026-07-21）----------
# job 4 (math500/sc+pot) が rc=1073807364 で落ちた際、旧実装は成功/失敗/クラッシュを
# 区別せず、以降のジョブが rc=3221226091 で連鎖即死してもキューは気づかず
# 「正常終了」の top-level status を書いていた。classify_exit_code はその判定を
# 担う純粋関数（I/O無し・import 時に副作用も無い）で、ここではモデル呼び出しも
# subprocess も一切使わずオフラインで検証する。
check("bq: rc=0 は ok", bq.classify_exit_code(0) == "ok")
check("bq: 通常の Python 失敗(rc=1)は error(crashではない)", bq.classify_exit_code(1) == "error")
check("bq: 通常の Python 失敗(rc=2)は error", bq.classify_exit_code(2) == "error")
check("bq: 実際に遭遇した job4 の異常終了コードは crash",
      bq.classify_exit_code(1073807364) == "crash")
check("bq: 連鎖即死した後続ジョブの異常終了コードも crash",
      bq.classify_exit_code(3221226091) == "crash")
check("bq: 負のシグナル終了コードも crash", bq.classify_exit_code(-1) == "crash")
check("bq: 閾値未満の巨大値は crash 扱いしない(境界)", bq.classify_exit_code(0x40000000 - 1) == "error")
check("bq: 閾値ちょうどは crash(境界)", bq.classify_exit_code(0x40000000) == "crash")

# main() のループを直接叩かず、同じ分類ロジックを模擬ジョブ列に適用して、
# crash が発生したジョブの status が 'ok' でなく、かつ全体の総合結果
# (status['ok'] 相当) が「クリーンな成功」を主張しないことを確認する。
_sim_rcs = [0, 0, 1073807364, 3221226091]
_sim_jobs = [{"n": i + 1, "rc": rc, "status": bq.classify_exit_code(rc)}
             for i, rc in enumerate(_sim_rcs)]
check("bq: crash ジョブの status は 'ok' ではない", _sim_jobs[2]["status"] != "ok")
_sim_crashed = [j for j in _sim_jobs if j["status"] == "crash"]
_sim_failed = [j for j in _sim_jobs if j["status"] == "error"]
_sim_overall_ok = not _sim_crashed and not _sim_failed
check("bq: gotcha8同型シナリオでは総合結果が失敗を示す(クリーン成功を主張しない)",
      _sim_overall_ok is False and len(_sim_crashed) >= 1)

# bench_queue の import 自体が副作用を持たないこと（main() は __main__ ガード下）の
# 簡易確認: モジュールに main はあるが、import 時点でジョブが実行されていないこと
# （_sim_jobs はテスト側のダミーであり QUEUE の長さと無関係）。
check("bq: import時点でQUEUEは未実行のジョブ一覧のまま(副作用なし)",
      hasattr(bq, "QUEUE") and hasattr(bq, "main") and hasattr(bq, "classify_exit_code"))


# ---------- ask(): think-strip リトライがループ末尾で握り潰されない (gotcha 3 関連バグ修正) ----------
# 旧実装は「thinking非対応」400 を検知したら payload から think を pop して
# for attempt in (1, 2) の次のイテレーションに continue するだけだった。
# attempt=1 が一過性の500（ロード直後によくある）で、attempt=2 で初めて
# 「thinking非対応」400 が出ると、continue しても for ループは尽きており、
# 組み直したリクエストは一度も送信されずに __ERROR__: think_stripped_retry が
# そのまま最終戻り値になっていた（SC投票/提案が黙って1票失われる = 精度低下）。
# ここでは urllib.request.urlopen と time.sleep のみをモックし、実際の
# Ollama/ネットワーク呼び出しは一切発生させずに検証する。


class _FakeHTTPResponse:
    """urllib.request.urlopen が返す `with ... as r:` 用の最小モック。"""

    def __init__(self, body_bytes):
        self._body = body_bytes

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._body


def _http_error(code, body_text):
    return urllib.error.HTTPError(
        f"{f.OLLAMA_URL}/api/chat", code, "mock error", {},
        io.BytesIO(body_text.encode("utf-8")),
    )


def _make_fake_urlopen(steps, calls_log):
    """steps: [("error", code, body_text), ...] または [("ok", content_text), ...] のリスト。
    定義された手数を超えて呼ばれたら AssertionError にする(無限ループを検知するため)。
    呼び出しごとに送信 payload(dict) を calls_log に積む。"""
    state = {"i": 0}

    def _fake_urlopen(req, timeout=None):
        i = state["i"]
        state["i"] += 1
        if i >= len(steps):
            raise AssertionError(f"unexpected extra urlopen call #{i + 1} (bounded-loop violation)")
        calls_log.append(json.loads(req.data.decode("utf-8")))
        step = steps[i]
        if step[0] == "error":
            raise _http_error(step[1], step[2])
        return _FakeHTTPResponse(json.dumps({"message": {"content": step[1]}}).encode("utf-8"))

    return _fake_urlopen


_orig_urlopen = urllib.request.urlopen
_orig_sleep = f.time.sleep

# --- シナリオ1: attempt1=一過性500, attempt2=thinking非対応400 → 組み直しリクエストが
#     独自に送信され、最終応答が失われない(バグ再現シナリオそのもの) ---
_calls1 = []
try:
    f.time.sleep = lambda s: None  # 一過性リトライの sleep(2) を待たない
    urllib.request.urlopen = _make_fake_urlopen(
        [("error", 500, "internal error, model loading"),
         ("error", 400, "this model does not support thinking"),
         ("ok", "the real final answer")],
        _calls1,
    )
    _r1 = f.ask("m1", [{"role": "user", "content": "hi"}], 0.7, think=True)
    check("ask: 500→thinking400→success で最終応答が失われない",
          _r1 == "the real final answer")
    check("ask: __ERROR__を返さない(think_stripped_retryで確定終了しない)",
          not str(_r1).startswith("__ERROR__"))
    check("ask: 成功リクエストのpayloadにthinkキーが残っていない",
          "think" not in _calls1[-1])
finally:
    urllib.request.urlopen = _orig_urlopen
    f.time.sleep = _orig_sleep

# --- シナリオ2: attempt1で即thinking非対応400(既存の主要ケース)→引き続き成功する回帰確認 ---
_calls2 = []
try:
    f.time.sleep = lambda s: None
    urllib.request.urlopen = _make_fake_urlopen(
        [("error", 400, "this model does not support thinking"),
         ("ok", "stripped retry answer")],
        _calls2,
    )
    _r2 = f.ask("m1", [{"role": "user", "content": "hi"}], 0.7, think=True)
    check("ask: 初回でthinking400→即座のstrip再送で成功(既存ケースの回帰なし)",
          _r2 == "stripped retry answer")
    check("ask: シナリオ2でも成功payloadにthinkキーが残っていない",
          "think" not in _calls2[-1])
finally:
    urllib.request.urlopen = _orig_urlopen
    f.time.sleep = _orig_sleep

# --- シナリオ3: 毎回thinking非対応400 → think はpop済みなので分岐は高々1回のみ発火し、
#     有限回のurlopen呼び出しで(無限ループせず)__ERROR__を返す ---
_calls3 = []
try:
    f.time.sleep = lambda s: None
    urllib.request.urlopen = _make_fake_urlopen(
        [("error", 400, "this model does not support thinking"),
         ("error", 400, "this model does not support thinking")],
        _calls3,
    )
    _r3 = f.ask("m1", [{"role": "user", "content": "hi"}], 0.7, think=True)
    check("ask: 毎回thinking400でも有限回(<=2回)のurlopen呼び出しで打ち切り(無限ループなし)",
          len(_calls3) <= 2)
    check("ask: 毎回thinking400なら最終的に__ERROR__を返す", str(_r3).startswith("__ERROR__"))
finally:
    urllib.request.urlopen = _orig_urlopen
    f.time.sleep = _orig_sleep

# --- シナリオ4: 通常の一過性失敗(thinking非対応ではない500が2連続)は従来通り
#     ちょうど2回試行してsleep(2)を1回だけ挟み__ERROR__を返す(一過性リトライ予算は不変) ---
_calls4 = []
try:
    f.time.sleep = lambda s: None
    urllib.request.urlopen = _make_fake_urlopen(
        [("error", 500, "internal error"),
         ("error", 500, "internal error")],
        _calls4,
    )
    _r4 = f.ask("m1", [{"role": "user", "content": "hi"}], 0.7, think=True)
    check("ask: 通常の一過性失敗(500,500)はちょうど2回試行して__ERROR__",
          len(_calls4) == 2 and str(_r4).startswith("__ERROR__"))
finally:
    urllib.request.urlopen = _orig_urlopen
    f.time.sleep = _orig_sleep

# --- gotcha #1 / #2 回帰: /api/chat 固定 & options.num_ctx 常時pin ---
# 既存の _make_fake_urlopen は payload(dict) だけを calls_log に積む設計で、既存の
# _calls1..4 のインデックス方法(payload dict として直接参照)を変えると回帰するため
# ここでは触らない。URL も検証したいこのセクション専用に別のフェイク urlopen を用意する
# (calls_log の各要素は {"full_url":.., "payload":..} の dict)。


def _make_fake_urlopen_url(steps, calls_log):
    """_make_fake_urlopen と同じ挙動だが、送信 payload に加えて req.full_url も記録する。"""
    state = {"i": 0}

    def _fake_urlopen(req, timeout=None):
        i = state["i"]
        state["i"] += 1
        if i >= len(steps):
            raise AssertionError(f"unexpected extra urlopen call #{i + 1} (bounded-loop violation)")
        calls_log.append({
            "full_url": req.full_url,
            "payload": json.loads(req.data.decode("utf-8")),
        })
        step = steps[i]
        if step[0] == "error":
            raise _http_error(step[1], step[2])
        return _FakeHTTPResponse(json.dumps({"message": {"content": step[1]}}).encode("utf-8"))

    return _fake_urlopen


# シナリオ5: 通常成功呼び出し(think/num_predict/fmt すべて未指定・未知モデル) →
# native /api/chat を叩き(/v1 は使わない)、options.num_ctx が既定値で必ず pin される。
_calls5 = []
try:
    f.time.sleep = lambda s: None
    urllib.request.urlopen = _make_fake_urlopen_url(
        [("ok", "plain answer")],
        _calls5,
    )
    _r5 = f.ask("m-unknown-nonthinking", [{"role": "user", "content": "hi"}], 0.7)
    _url5 = _calls5[-1]["full_url"]
    _opts5 = _calls5[-1]["payload"].get("options", {})
    check("ask: 通常呼び出しは /api/chat を叩く(gotcha#1)", _url5.endswith("/api/chat"))
    check("ask: 通常呼び出しで /v1 エンドポイントは使わない(gotcha#1)", "/v1" not in _url5)
    check("ask: think/num_predict/fmt が全てNoneでもoptions.num_ctxは省略されない(gotcha#2)",
          "num_ctx" in _opts5 and _opts5["num_ctx"])
    check("ask: 未知モデルはMODEL_NUM_CTXが既定値になる",
          _opts5["num_ctx"] == f.MODEL_NUM_CTX)
finally:
    urllib.request.urlopen = _orig_urlopen
    f.time.sleep = _orig_sleep

# シナリオ6: MODEL_CONFIG に登録された思考モデル(gpt-oss:20b)は num_ctx=16384 が
# model_cfg 由来で pin される(8192 のままでは思考が truncate される既知不具合の回帰防止)。
_calls6 = []
try:
    f.time.sleep = lambda s: None
    urllib.request.urlopen = _make_fake_urlopen_url(
        [("ok", "thinking model answer")],
        _calls6,
    )
    _r6 = f.ask("gpt-oss:20b", [{"role": "user", "content": "hi"}], 0.7)
    _opts6 = _calls6[-1]["payload"].get("options", {})
    _expected_ctx6 = f.model_cfg("gpt-oss:20b", "num_ctx", f.MODEL_NUM_CTX)
    check("ask: gpt-oss:20b(思考モデル)はMODEL_CONFIG由来のnum_ctxになる",
          _opts6["num_ctx"] == _expected_ctx6 == 16384)
finally:
    urllib.request.urlopen = _orig_urlopen
    f.time.sleep = _orig_sleep

# シナリオ7: 明示的な num_ctx=... 引数はモデル既定値より優先される。
_calls7 = []
try:
    f.time.sleep = lambda s: None
    urllib.request.urlopen = _make_fake_urlopen_url(
        [("ok", "explicit ctx answer")],
        _calls7,
    )
    _r7 = f.ask("gpt-oss:20b", [{"role": "user", "content": "hi"}], 0.7, num_ctx=12345)
    _opts7 = _calls7[-1]["payload"].get("options", {})
    check("ask: 明示的なnum_ctx引数はモデル既定値より優先される",
          _opts7["num_ctx"] == 12345)
finally:
    urllib.request.urlopen = _orig_urlopen
    f.time.sleep = _orig_sleep

# シナリオ8(重要): think-strip再送パス(500→thinking非対応400→success)でも、
# 最終的に再送されるリクエストが num_ctx pin と /api/chat エンドポイントの両方を
# 維持していること(L1054のpayload再構築で options.num_ctx や URL が失われていないか)。
_calls8 = []
try:
    f.time.sleep = lambda s: None
    urllib.request.urlopen = _make_fake_urlopen_url(
        [("error", 500, "internal error, model loading"),
         ("error", 400, "this model does not support thinking"),
         ("ok", "final answer after think strip")],
        _calls8,
    )
    _r8 = f.ask("gpt-oss:20b", [{"role": "user", "content": "hi"}], 0.7, think=True)
    _expected_ctx8 = f.model_cfg("gpt-oss:20b", "num_ctx", f.MODEL_NUM_CTX)
    _final_call8 = _calls8[-1]
    _final_opts8 = _final_call8["payload"].get("options", {})
    check("ask: think-strip再送でも最終応答が失われない",
          _r8 == "final answer after think strip")
    check("ask: think-strip再送の最終リクエストもoptions.num_ctxを維持する(gotcha#2)",
          "num_ctx" in _final_opts8 and _final_opts8["num_ctx"] == _expected_ctx8)
    check("ask: think-strip再送の最終リクエストも/api/chatを維持する(gotcha#1)",
          _final_call8["full_url"].endswith("/api/chat") and "/v1" not in _final_call8["full_url"])
    check("ask: think-strip再送の最終リクエストはthinkキーが除去されている",
          "think" not in _final_call8["payload"])
finally:
    urllib.request.urlopen = _orig_urlopen
    f.time.sleep = _orig_sleep

# ---------- fugu_answer: SC結果のユーザー提示 ----------
# fugu_answer() の自己一貫性投票(SC)結果 → ユーザー提示の接合点（行 2602-2615 付近）の回帰テスト。
# solve_verifiable の戻り値と、本文から実際に抽出した答え(extract_final_answer/answers_equivalent
# は本物をそのまま使う)がずれた場合にのみ「(自己一貫性投票による最終解答: X)」を付記し、
# 一致すれば本文をそのまま返す。None ならMoA(合議)へフォールバックする、という分岐を検証する。
# validate_plan() 済みの明示プランを渡して conduct() を経由させない。

_orig_sc_enabled = f.SC_ENABLED
_orig_solve_verifiable = f.solve_verifiable
_orig_get_proposals = f.get_proposals
_orig_aggregate = f.aggregate


def _validated_plan(task_type, mode="moa", rounds=1):
    # default_plan()/validate_plan() が生成する形と同じキー構成の明示プラン。
    return {
        "mode": mode,
        "task_type": task_type,
        "selected_proposers": ["m1", "m2", "m3"],
        "rounds": rounds,
        "use_image_generation": False,
        "image_only": False,
        "make_pptx": False,
        "search_required": False,
        "reason": "test",
        "_fallback": False,
    }


def _make_moa_forbidden(touched_list):
    """SCが成功した経路ではget_proposals/aggregateへ絶対に到達してはならないことを
    検証するための番人。呼ばれたら記録した上で必ず例外を送出する。"""
    def _get_proposals_forbidden(*a, **kw):
        touched_list.append(True)
        raise AssertionError("SC成功時はMoA(get_proposals)へ到達してはならない")

    def _aggregate_forbidden(*a, **kw):
        touched_list.append(True)
        raise AssertionError("SC成功時はMoA(aggregate)へ到達してはならない")

    return _get_proposals_forbidden, _aggregate_forbidden


# --- Case A: 本文の結論(裁定等で差し替わった\boxed)が投票結果と食い違う → 明示注記が付く ---
_moa_touched_a = []
_get_proposals_never_a, _aggregate_never_a = _make_moa_forbidden(_moa_touched_a)

try:
    f.SC_ENABLED = True
    f.solve_verifiable = lambda question, task_type, history=None: {
        "answer": "5",
        "text": "途中式…裁定により \\boxed{7} に差し替え。",
        "votes": {"7": 2, "5": 1},
        "n_samples": 3,
    }
    f.get_proposals = _get_proposals_never_a
    f.aggregate = _aggregate_never_a
    with contextlib.redirect_stdout(io.StringIO()):
        _ans_a = f.fugu_answer("2+3は?", plan=_validated_plan("math"))
finally:
    f.SC_ENABLED = _orig_sc_enabled
    f.solve_verifiable = _orig_solve_verifiable
    f.get_proposals = _orig_get_proposals
    f.aggregate = _orig_aggregate

check("fugu_answer: 本文の結論と投票結果が食い違う場合は明示注記を付す",
      "(自己一貫性投票による最終解答: 5)" in _ans_a)
check("fugu_answer: 食い違いケースでも元の本文はそのまま含まれる",
      "裁定により \\boxed{7} に差し替え。" in _ans_a)
check("fugu_answer: 食い違いケースはSC経路で返りMoAへ到達しない", not _moa_touched_a)

# --- Case B: 本文の結論(\boxedの答え)が投票結果と一致 → 注記なしでそのまま返す（math） ---
_moa_touched_b1 = []
_get_proposals_never_b1, _aggregate_never_b1 = _make_moa_forbidden(_moa_touched_b1)
try:
    f.SC_ENABLED = True
    f.solve_verifiable = lambda question, task_type, history=None: {
        "answer": "42",
        "text": "計算の結果、\\boxed{42} である。",
        "votes": {"42": 3},
        "n_samples": 3,
    }
    f.get_proposals = _get_proposals_never_b1
    f.aggregate = _aggregate_never_b1
    with contextlib.redirect_stdout(io.StringIO()):
        _ans_b1 = f.fugu_answer("6*7は?", plan=_validated_plan("math"))
finally:
    f.SC_ENABLED = _orig_sc_enabled
    f.solve_verifiable = _orig_solve_verifiable
    f.get_proposals = _orig_get_proposals
    f.aggregate = _orig_aggregate

check("fugu_answer: 本文とSC結果(math)が一致すれば注記なし",
      "自己一貫性投票による最終解答" not in _ans_b1)
check("fugu_answer: 一致ケース(math)は本文をそのまま返す",
      _ans_b1 == "計算の結果、\\boxed{42} である。")
check("fugu_answer: 一致ケース(math)もMoAへ到達しない", not _moa_touched_b1)

# --- Case B': mcq版（選択肢文字が一致） ---
_moa_touched_b2 = []
_get_proposals_never_b2, _aggregate_never_b2 = _make_moa_forbidden(_moa_touched_b2)
try:
    f.SC_ENABLED = True
    f.solve_verifiable = lambda question, task_type, history=None: {
        "answer": "C",
        "text": "検討の結果、\\boxed{C} が正解。",
        "votes": {"C": 3},
        "n_samples": 3,
    }
    f.get_proposals = _get_proposals_never_b2
    f.aggregate = _aggregate_never_b2
    with contextlib.redirect_stdout(io.StringIO()):
        _ans_b2 = f.fugu_answer("次のうち正しいものは?", plan=_validated_plan("mcq"))
finally:
    f.SC_ENABLED = _orig_sc_enabled
    f.solve_verifiable = _orig_solve_verifiable
    f.get_proposals = _orig_get_proposals
    f.aggregate = _orig_aggregate

check("fugu_answer: 本文とSC結果(mcq選択肢)が一致すれば注記なし",
      "自己一貫性投票による最終解答" not in _ans_b2)
check("fugu_answer: 一致ケース(mcq)は本文をそのまま返す",
      _ans_b2 == "検討の結果、\\boxed{C} が正解。")
check("fugu_answer: 一致ケース(mcq)もMoAへ到達しない", not _moa_touched_b2)

# --- Case C: solve_verifiable が None(投票不成立) → MoAへフォールスルーする ---
# plan["rounds"]=MAX_ROUNDS にして、r>=limit のブレークが「計画分残っているか」判定より
# 先に効くようにし、critique()（本物のask呼び出しを要する）へ到達せずに済ませる
# （このテストが検証したいのはSC→MoAへの委譲そのものであり、MoAの反復打ち切りロジック自体は
# 既存の他テストが担保している）。
_MOA_SENTINEL = "MOA_FALLBACK_SENTINEL: 単体/合議側で生成された最終回答"
_get_proposals_calls_c = []
_aggregate_calls_c = []


def _fake_get_proposals_c(models, question, reference=None, issue=None, history=None):
    _get_proposals_calls_c.append((tuple(models), reference, issue))
    return [(m, "dummy proposal (SCフォールバック検証用ダミー)") for m in models]


def _fake_aggregate_c(question, proposals):
    _aggregate_calls_c.append(len(proposals))
    return _MOA_SENTINEL


try:
    f.SC_ENABLED = True
    f.solve_verifiable = lambda question, task_type, history=None: None
    f.get_proposals = _fake_get_proposals_c
    f.aggregate = _fake_aggregate_c
    with contextlib.redirect_stdout(io.StringIO()):
        _ans_c = f.fugu_answer(
            "解けない問題?", plan=_validated_plan("math", mode="moa", rounds=f.MAX_ROUNDS))
finally:
    f.SC_ENABLED = _orig_sc_enabled
    f.solve_verifiable = _orig_solve_verifiable
    f.get_proposals = _orig_get_proposals
    f.aggregate = _orig_aggregate

check("fugu_answer: SCがNoneならMoA(get_proposals/aggregate)へフォールスルーする",
      _ans_c == _MOA_SENTINEL)
check("fugu_answer: フォールバック時は実際にget_proposals/aggregateが呼ばれる",
      len(_get_proposals_calls_c) >= 1 and len(_aggregate_calls_c) >= 1)

# ---------- _load_rag_chunks: '[' 始まりの過剰フィルタ回帰防止 (2026-07-22) ----------
# _read_excel/_read_pptx の成功時出力（"[Sheet: ...]" / "[Slide 1]"）が
# text.startswith("[") だけで誤スキップされ、RAGから丸ごと欠落していたバグの回帰テスト。
# ライブラリ未インストール通知（1行・pip install を含む）だけが正しくスキップされることも検証。
# ローカル一時ファイルのみを使用し、Ollama/ネットワーク/bench呼び出しは一切行わない。
import tempfile as _tempfile
import os as _os

check("_is_lib_missing_notice: PDF未インストール通知はTrue",
      f._is_lib_missing_notice(
          "[PDF: foo.pdf — テキスト抽出には pdfplumber or pypdf が必要: pip install pdfplumber]"))
check("_is_lib_missing_notice: DOCX未インストール通知はTrue",
      f._is_lib_missing_notice("[DOCX: foo.docx — python-docx が必要: pip install python-docx]"))
check("_is_lib_missing_notice: Excel未インストール通知はTrue",
      f._is_lib_missing_notice("[Excel: foo.xlsx — openpyxl or pandas が必要: pip install openpyxl]"))
check("_is_lib_missing_notice: PPTX未インストール通知はTrue",
      f._is_lib_missing_notice("[PPTX: foo.pptx — python-pptx が必要: pip install python-pptx]"))
check("_is_lib_missing_notice: Excel成功時の'[Sheet: ...]'はFalse",
      not f._is_lib_missing_notice("[Sheet: Sheet1]\nA\tB\n1\t2"))
check("_is_lib_missing_notice: PPTX成功時の'[Slide 1]'はFalse",
      not f._is_lib_missing_notice("[Slide 1]\nこんにちは"))
check("_is_lib_missing_notice: JSON配列先頭'[1, 2, 3]'はFalse",
      not f._is_lib_missing_notice("[1, 2, 3]"))
check("_is_lib_missing_notice: Markdownリンク'[link](url)'はFalse",
      not f._is_lib_missing_notice("[link](url)\n本文がここに続く"))
check("_is_lib_missing_notice: 空文字はFalse", not f._is_lib_missing_notice(""))

with _tempfile.TemporaryDirectory() as _rag_dir:
    import pathlib as _pathlib
    _rag_root = _pathlib.Path(_rag_dir)

    # 成功したExcel抽出を模したテキストファイル（本物のopenpyxl/pandas呼び出しは不要、
    # read_file_text の出力形状だけを .txt として直接再現して検証する）
    (_rag_root / "sheet_like.txt").write_text(
        "[Sheet: Sheet1]\n" + ("data\t" * 5 + "\n") * 50, encoding="utf-8")
    # 成功したPPTX抽出を模したテキストファイル
    (_rag_root / "slide_like.txt").write_text(
        "[Slide 1]\n" + ("スライド本文の内容です。" * 20 + "\n") * 20, encoding="utf-8")
    # JSON配列（トップレベルが '[' で始まる正当なドキュメント）
    (_rag_root / "array.json").write_text(
        json.dumps(list(range(200))), encoding="utf-8")
    # ブラケットリンクで始まる正当なMarkdown
    (_rag_root / "note.md").write_text(
        "[link](https://example.com)\n" + ("本文テキストです。" * 20 + "\n") * 20,
        encoding="utf-8")
    # 本物のライブラリ未インストール通知そのもの（1行）をそのまま模したファイル
    (_rag_root / "notice_like.txt").write_text(
        "[Excel: dummy.xlsx — openpyxl or pandas が必要: pip install openpyxl]",
        encoding="utf-8")
    # 通常のブラケット無しテキスト（既存挙動のバイト単位不変性チェック用）
    _plain_text = ("普通の本文テキストです。" * 30 + "\n") * 10
    (_rag_root / "plain.txt").write_text(_plain_text, encoding="utf-8")

    _rag_chunks = f._load_rag_chunks([str(_rag_root)])
    _rag_by_file = {}
    for _fp, _chunk in _rag_chunks:
        _rag_by_file.setdefault(_os.path.basename(_fp), []).append(_chunk)

    check("RAG: 成功Excel様('[Sheet: ...]')出力がチャンク化される",
          len(_rag_by_file.get("sheet_like.txt", [])) >= 1)
    check("RAG: 成功PPTX様('[Slide 1]')出力がチャンク化される",
          len(_rag_by_file.get("slide_like.txt", [])) >= 1)
    check("RAG: JSON配列('[1, 2, ...']')がチャンク化される",
          len(_rag_by_file.get("array.json", [])) >= 1)
    check("RAG: ブラケットリンクMarkdownがチャンク化される",
          len(_rag_by_file.get("note.md", [])) >= 1)
    check("RAG: ライブラリ未インストール通知そのものはチャンク化されない",
          "notice_like.txt" not in _rag_by_file)

    # チャンク分割/オーバーラップ計算が非ブラケットテキストで従来通りバイト一致すること
    _expected_plain_chunks = []
    _start = 0
    while _start < len(_plain_text):
        _end = _start + f.RAG_CHUNK_CHARS
        _expected_plain_chunks.append(_plain_text[_start:_end])
        _start += f.RAG_CHUNK_CHARS - f.RAG_CHUNK_OVERLAP
    check("RAG: 通常テキストのチャンク分割はバイト単位で従来通り",
          _rag_by_file.get("plain.txt", []) == _expected_plain_chunks)

# ---------- _load_rag_chunks: 1ファイルの読み込み例外でRAG全体が落ちない (2026-07-22 / iter42) ----------
# read_file_text(fp) を裸で呼んでいたため、破損/未対応ファイル1件がImportError以外の例外を
# 送出すると _load_rag_chunks -> _get_rag_chunks -> rag_search -> build_context まで伝播し、
# 質問のたびにRAGコンテキストが丸ごと失われていた。ここでは1ファイル単位に例外を隔離し、
# 他ファイルは正常にチャンク化されることを検証する。iter41のgraceful-degradation方針を踏襲。

# (1) read_file_textをmonkeypatchし、片方のファイルパスだけ例外を送出させる。
#     実ファイルは一時ディレクトリに置き、read_file_textをすり替えるだけで
#     Ollama/ネットワーク呼び出しは一切行わない。
_orig_read_file_text = f.read_file_text
with _tempfile.TemporaryDirectory() as _rag_dir2:
    _rag_root2 = _pathlib.Path(_rag_dir2)
    _bad_fp = _rag_root2 / "corrupt.xlsx"
    _good_fp = _rag_root2 / "good.txt"
    _bad_fp.write_bytes(b"not a real xlsx file, just garbage bytes")
    _good_text = ("これは正常に読めるファイルの本文です。" * 10 + "\n") * 5
    _good_fp.write_text(_good_text, encoding="utf-8")

    def _fake_read_file_text(path):
        if _pathlib.Path(path).name == "corrupt.xlsx":
            raise ValueError("simulated corrupt file read failure")
        return _orig_read_file_text(path)

    try:
        f.read_file_text = _fake_read_file_text
        _rag_chunks2 = f._load_rag_chunks([str(_rag_root2)])
    finally:
        f.read_file_text = _orig_read_file_text

    check("_load_rag_chunks: 1ファイルの読み込み例外で全体が例外送出しない(到達できていること自体が検証)",
          True)
    _rag_by_file2 = {}
    for _fp2, _chunk2 in _rag_chunks2:
        _rag_by_file2.setdefault(_os.path.basename(_fp2), []).append((_fp2, _chunk2))
    check("_load_rag_chunks: 読み込み失敗したファイルのチャンクは一切含まれない",
          "corrupt.xlsx" not in _rag_by_file2)
    check("_load_rag_chunks: 正常ファイルのチャンクは含まれる",
          len(_rag_by_file2.get("good.txt", [])) >= 1)
    _expected_good_chunks2 = []
    _start2 = 0
    while _start2 < len(_good_text):
        _end2 = _start2 + f.RAG_CHUNK_CHARS
        _expected_good_chunks2.append((str(_good_fp), _good_text[_start2:_end2]))
        _start2 += f.RAG_CHUNK_CHARS - f.RAG_CHUNK_OVERLAP
    check("_load_rag_chunks: 正常ファイルの(パス,チャンク)タプルが正しい",
          _rag_by_file2.get("good.txt", []) == _expected_good_chunks2)

# (2) ディレクトリ内が「読み込みに失敗するファイルのみ」の場合、空リストを返し例外を送出しない。
with _tempfile.TemporaryDirectory() as _rag_dir3:
    _rag_root3 = _pathlib.Path(_rag_dir3)
    (_rag_root3 / "onlybad.xlsx").write_bytes(b"garbage garbage garbage")

    def _always_fail_read_file_text(path):
        raise RuntimeError("simulated total read failure")

    try:
        f.read_file_text = _always_fail_read_file_text
        _rag_chunks3 = f._load_rag_chunks([str(_rag_root3)])
    finally:
        f.read_file_text = _orig_read_file_text

    check("_load_rag_chunks: 全ファイルが読み込み失敗するディレクトリでは空リストを返す",
          _rag_chunks3 == [])

# (3) 回帰: 正常に読めるファイルのみのディレクトリでは、変更前と完全に同一のチャンク出力
#     (境界・オーバーラップ・順序含め)になること。
with _tempfile.TemporaryDirectory() as _rag_dir4:
    _rag_root4 = _pathlib.Path(_rag_dir4)
    _text_a4 = ("ファイルAの本文テキストです。" * 15 + "\n") * 8
    _text_b4 = ("File B plain ascii content line. " * 20 + "\n") * 6
    (_rag_root4 / "a_file.txt").write_text(_text_a4, encoding="utf-8")
    (_rag_root4 / "b_file.md").write_text(_text_b4, encoding="utf-8")

    _rag_chunks4 = f._load_rag_chunks([str(_rag_root4)])

    _expected_chunks4 = []
    for _fname4, _text4 in sorted([("a_file.txt", _text_a4), ("b_file.md", _text_b4)]):
        _fp4 = str(_rag_root4 / _fname4)
        _start4 = 0
        while _start4 < len(_text4):
            _end4 = _start4 + f.RAG_CHUNK_CHARS
            _expected_chunks4.append((_fp4, _text4[_start4:_end4]))
            _start4 += f.RAG_CHUNK_CHARS - f.RAG_CHUNK_OVERLAP
    check("_load_rag_chunks: 正常ファイルのみの場合はチャンク出力が変更前とバイト単位で完全一致(境界/オーバーラップ/順序)",
          _rag_chunks4 == _expected_chunks4)

# (4) 任意: 実際のリーダー例外経路の検証（monkeypatchではなく、本物の壊れた.xlsxを
#     _read_excel に読ませて例外を発生させる）。openpyxlが利用可能な環境でのみ実施。
try:
    import openpyxl as _openpyxl_probe2  # noqa: F401
    _HAS_OPENPYXL2 = True
except Exception:
    _HAS_OPENPYXL2 = False

if _HAS_OPENPYXL2:
    with _tempfile.TemporaryDirectory() as _rag_dir5:
        _rag_root5 = _pathlib.Path(_rag_dir5)
        # 本物の壊れた.xlsx（ZIP/XMLとして無効なゴミバイト列）
        (_rag_root5 / "broken.xlsx").write_bytes(b"\x00\x01\x02not a zip or xlsx file at all\xff\xfe")
        _good_text5 = "This is a genuinely readable plain text file for RAG.\n" * 30
        (_rag_root5 / "readable.txt").write_text(_good_text5, encoding="utf-8")

        _rag_chunks5 = f._load_rag_chunks([str(_rag_root5)])
        _rag_by_file5 = {}
        for _fp5, _chunk5 in _rag_chunks5:
            _rag_by_file5.setdefault(_os.path.basename(_fp5), []).append(_chunk5)

        check("_load_rag_chunks: 本物の破損.xlsx(実リーダー例外)はスキップされる",
              "broken.xlsx" not in _rag_by_file5)
        check("_load_rag_chunks: 破損.xlsxと同居する正常な.txtは読み込まれる",
              len(_rag_by_file5.get("readable.txt", [])) >= 1)
else:
    print("   [SKIP] openpyxl未インストールのため実.xlsx破損読み込みテストをスキップ")

# ---------- read_file_text: ディスパッチ例外を握りつぶして""を返す (2026-07-23 / iter53) ----------
# _read_pdf/_read_docx/_read_excel/_read_pptx/_read_html は import 文と実パース処理を
# 同じ try/except ImportError で包んでいるため、ライブラリ自体は入っているがファイルが
# 壊れている場合の非ImportError例外（zipfile.BadZipFile, PDFSyntaxError等）が
# read_file_text の外まで素通しで伝播していた。main()の--file呼び出し
# （`read_file_text(fp).strip()`）は_load_rag_chunksと違って無防備だったため、
# 壊れたOffice/PDFファイルを渡すとCLI全体がクラッシュしていた（iter42はRAG経路のみ保護、
# 全リーダー書き換えを試みたiter51は行き詰まった）。ここではread_file_text自身が
# 例外を握りつぶして""を返すことを、個々のリーダーをmonkeypatchして直接検証する
# （実ライブラリ・実ファイル・Ollama呼び出しは一切不要）。
import zipfile as _zipfile

_orig_read_pdf2 = f._read_pdf
_orig_read_excel2 = f._read_excel


def _rft_raise_value_error(path):
    raise ValueError("simulated corrupt PDF parse failure")


def _rft_raise_bad_zip(path):
    raise _zipfile.BadZipFile("simulated corrupt xlsx (not a zip file)")


try:
    f._read_pdf = _rft_raise_value_error
    with contextlib.redirect_stdout(io.StringIO()) as _rft_out1:
        _rft_result1 = f.read_file_text(_pathlib.Path("dummy_corrupt.pdf"))
finally:
    f._read_pdf = _orig_read_pdf2

check("read_file_text: _read_pdfが非ImportError例外(ValueError)を送出しても伝播せず\"\"を返す",
      _rft_result1 == "")
check("read_file_text: PDF読み込み失敗時に警告(ファイル名+例外型)を表示する",
      "dummy_corrupt.pdf" in _rft_out1.getvalue() and "ValueError" in _rft_out1.getvalue())

try:
    f._read_excel = _rft_raise_bad_zip
    with contextlib.redirect_stdout(io.StringIO()) as _rft_out2:
        _rft_result2 = f.read_file_text(_pathlib.Path("dummy_corrupt.xlsx"))
finally:
    f._read_excel = _orig_read_excel2

check("read_file_text: _read_excelがBadZipFileを送出しても伝播せず\"\"を返す(ディスパッチ全体の保護を確認)",
      _rft_result2 == "")
check("read_file_text: Excel読み込み失敗時に警告(ファイル名+例外型)を表示する",
      "dummy_corrupt.xlsx" in _rft_out2.getvalue() and "BadZipFile" in _rft_out2.getvalue())

# 回帰: 正常なテキストファイル/_BINARY_SKIP拡張子は従来通りの挙動を維持
with _tempfile.TemporaryDirectory() as _rft_dir:
    _rft_root = _pathlib.Path(_rft_dir)
    _rft_txt_content = "これは通常のテキストファイルです。\nsecond line.\n"
    (_rft_root / "plain.txt").write_text(_rft_txt_content, encoding="utf-8")
    (_rft_root / "note.md").write_text(_rft_txt_content, encoding="utf-8")
    check("read_file_text: 通常の.txtファイルはバイト単位で内容が一致(回帰)",
          f.read_file_text(_rft_root / "plain.txt") == _rft_txt_content)
    check("read_file_text: 通常の.mdファイルはバイト単位で内容が一致(回帰)",
          f.read_file_text(_rft_root / "note.md") == _rft_txt_content)

    (_rft_root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\nnot a real png")
    check("read_file_text: _BINARY_SKIP拡張子(.png)は従来通り\"\"を返す(回帰)",
          f.read_file_text(_rft_root / "image.png") == "")

# ライブラリ未インストール通知文字列（例外ではなくリーダーが正常returnした場合）は
# ""に変換されず、そのままバイト単位で素通しされること（_is_lib_missing_notice判定は
# 呼び出し側=_load_rag_chunksの仕事であり、read_file_text自体は加工しない）。
_rft_notice_text = "[Excel: dummy.xlsx — openpyxl or pandas が必要: pip install openpyxl]"


def _rft_return_notice(path):
    return _rft_notice_text


try:
    f._read_excel = _rft_return_notice
    _rft_result3 = f.read_file_text(_pathlib.Path("dummy_notice.xlsx"))
finally:
    f._read_excel = _orig_read_excel2

check("read_file_text: リーダーがpip install通知文字列を正常returnした場合はそのまま素通し(\"\"化されない)",
      _rft_result3 == _rft_notice_text)

# ---------- _tokenize / _score_chunk: 現行挙動の直接検証 ----------
check("_tokenize: ASCII+CJK混在を別トークンに分割",
      f._tokenize("PINNについて") == {"pinn", "について"})
check("_score_chunk: 空チャンクは0.0",
      f._score_chunk({"apple"}, "") == 0.0)
check("_score_chunk: 重複トークンありは正のスコア",
      f._score_chunk({"apple"}, "apple pie recipe") > 0.0)
check("_score_chunk: 重複トークンなしは0.0",
      f._score_chunk({"apple"}, "zebra mountain train") == 0.0)

# ---------- rag_search: score>0のみ抽出（2026-07-22） ----------
# 以前は top = scored[:top_k] のうち先頭(best)が0でなければ丸ごと返しており、
# top_k内にキーワード的に無関係(score==0)なチャンクが混ざっていても
# そのままプロンプトへ注入されていた（rag_search ~L855-864）。
# score>0のみへの絞り込みが「best以外は素通し」の回帰を起こしていないか、
# また「関連チャンクを取りこぼしていない」かを、Ollama/ネットワーク一切なしで検証する。


_orig_get_rag_chunks = f._get_rag_chunks
try:
    # 1) queryは1件だけにマッチ、他はスコア0。top_k=2(>=2件)でも
    #    無関係チャンクの本文が混入しないこと。
    _chunks_partial = [
        ("dir/apple.txt", "This chunk talks about apple pie and recipe details."),
        ("dir/zebra.txt", "Completely unrelated content about zebras and mountains."),
        ("dir/cars.txt", "Another unrelated text about cars and trains."),
    ]
    f._get_rag_chunks = lambda dirs: _chunks_partial
    _res_partial = f.rag_search("apple recipe", dirs=["dummy"], top_k=2)
    check("rag_search: マッチしたチャンクのSourceが含まれる",
          "[Source: apple.txt]" in _res_partial)
    check("rag_search: スコア0チャンクの本文(zebra)は混入しない",
          "zebra" not in _res_partial and "mountains" not in _res_partial)
    check("rag_search: スコア0チャンクの本文(cars)は混入しない",
          "cars" not in _res_partial and "trains" not in _res_partial)

    # 2) 複数チャンクが全てスコア>0 -> 従来通りtop_k件まで降順・書式そのまま。
    # 注意: _score_chunk は _tokenize(chunk) を「集合」として重複除去してから
    # overlap/sqrt(len)+1 を計算するため、同じ語の反復回数はスコアに影響しない。
    # トークン集合の重複数と集合サイズを変えて意図的にスコアを分ける
    # （one: overlap2/len2≈82.8 > three: overlap1/len1=50.0 > two: overlap1/len2≈41.4）。
    # わざと入力順とスコア降順を食い違わせ、sort が実際に効いていることを検証する。
    _chunks_all_match = [
        ("dir/two.txt", "apple only"),      # overlap=1({apple}), len=2 -> ~41.4
        ("dir/three.txt", "recipe"),        # overlap=1({recipe}), len=1 -> 50.0
        ("dir/one.txt", "apple recipe"),    # overlap=2({apple,recipe}), len=2 -> ~82.8
    ]
    f._get_rag_chunks = lambda dirs: _chunks_all_match
    _res_all = f.rag_search("apple recipe", dirs=["dummy"], top_k=3)
    _expected_all = (
        "## Relevant Document Context (RAG)\n\n"
        "[Source: one.txt]\napple recipe"
        "\n\n---\n\n"
        "[Source: three.txt]\nrecipe"
        "\n\n---\n\n"
        "[Source: two.txt]\napple only"
    )
    check("rag_search: 全チャンクscore>0ならtop_k件・降順・書式が従来通り",
          _res_all == _expected_all)

finally:
    f._get_rag_chunks = _orig_get_rag_chunks

# best(=全件)が0スコアになるクエリで再検証(明示的に独立したtry/finallyで実施)
try:
    f._get_rag_chunks = lambda dirs: [
        ("dir/zebra.txt", "Completely unrelated content about zebras and mountains."),
        ("dir/cars.txt", "Another unrelated text about cars and trains."),
    ]
    check("rag_search: best(=全件)がscore0なら空文字（既存契約を維持）",
          f.rag_search("apple recipe", dirs=["dummy"], top_k=2) == "")
finally:
    f._get_rag_chunks = _orig_get_rag_chunks

# 4) 空dirs / 空チャンクリストは従来通り空文字
check("rag_search: dirsが空なら空文字",
      f.rag_search("apple", dirs=[]) == "")
try:
    f._get_rag_chunks = lambda dirs: []
    check("rag_search: チャンクリストが空なら空文字",
          f.rag_search("apple", dirs=["dummy"]) == "")
finally:
    f._get_rag_chunks = _orig_get_rag_chunks

# ---------- _save_as_html: コードフェンスの開始/終了タグ整合性 (2026-07-22) ----------
# 旧実装は開始/終了 ``` フェンスを両方とも "<pre><code>" にマップし、
# "</code></pre>" を一度も出力しないため <pre><code>...<pre><code> という
# 入れ子・未クローズの不整合HTMLになり、さらにコード本文が <br> 付きの
# 通常行として扱われて整形が崩れていた（L3135-3140）。ローカル一時ファイル
# への書き込み・読み戻しのみで検証し、Ollama/ネットワーク呼び出しは一切ない。
import pathlib as _html_pathlib

with _tempfile.TemporaryDirectory() as _html_dir:
    _html_root = _html_pathlib.Path(_html_dir)

    # (a) 単一のpythonフェンス付きコードブロック
    _html_out_a = _html_root / "a.html"
    _answer_a = "before\n```python\nx = 1\nprint(x)\n```\nafter"
    f._save_as_html(_html_out_a, "q1", _answer_a, 1.23)
    _content_a = _html_out_a.read_text(encoding="utf-8")
    check("_save_as_html: <pre><code>は正確に1回出現",
          _content_a.count("<pre><code>") == 1)
    check("_save_as_html: </code></pre>は正確に1回出現",
          _content_a.count("</code></pre>") == 1)
    check("_save_as_html: コード本文はescapeされている",
          "x = 1" in _content_a and "print(x)" in _content_a)
    _code_body_a = _content_a.split("<pre><code>", 1)[1].split("</code></pre>", 1)[0]
    check("_save_as_html: コード本文内に<br>が混入しない",
          "<br>" not in _code_body_a)

    # (b) プレーンテキストのみの回答: <br>維持・<pre>は出現しない
    _html_out_b = _html_root / "b.html"
    f._save_as_html(_html_out_b, "q2", "line1\nline2\nline3", 0.5)
    _content_b = _html_out_b.read_text(encoding="utf-8")
    check("_save_as_html: プレーンテキストは<br>で改行が維持される",
          _content_b.count("<br>") == 3)
    check("_save_as_html: プレーンテキストのみでは<pre>が出現しない",
          "<pre>" not in _content_b)

    # (c) 未終端フェンス（```が奇数個）でもバランスの取れた閉じタグになる
    _html_out_c = _html_root / "c.html"
    _answer_c = "intro\n```python\nx = 1\ny = 2\n"  # 閉じフェンスなし
    f._save_as_html(_html_out_c, "q3", _answer_c, 0.1)
    _content_c = _html_out_c.read_text(encoding="utf-8")
    check("_save_as_html: 未終端フェンスでも<pre><code>と</code></pre>の個数が一致",
          _content_c.count("<pre><code>") == _content_c.count("</code></pre>") == 1)

    # (d) 同一パスへ2回保存 -> 単一<body>へマージされ、両方のコードブロックがバランス
    _html_out_d = _html_root / "d.html"
    f._save_as_html(_html_out_d, "q4a", "```\ncode block one\n```", 0.1)
    f._save_as_html(_html_out_d, "q4b", "```\ncode block two\n```", 0.2)
    _content_d = _html_out_d.read_text(encoding="utf-8")
    check("_save_as_html: 2回保存しても<body>は1つにマージされる",
          _content_d.count("<body>") == 1 and _content_d.count("</body>") == 1)
    check("_save_as_html: 2回保存後も<pre><code>/</code></pre>の個数が一致(各2件)",
          _content_d.count("<pre><code>") == 2 and _content_d.count("</code></pre>") == 2)
    check("_save_as_html: 2回保存後も両方の回答本文が含まれる",
          "code block one" in _content_d and "code block two" in _content_d)

    # 回帰: コード本文中の '<' '&' がHTMLエスケープされ、生の '<b' 等が漏れない
    _html_out_e = _html_root / "e.html"
    _answer_e = "```\na < b && c\n```"
    f._save_as_html(_html_out_e, "q5", _answer_e, 0.1)
    _content_e = _html_out_e.read_text(encoding="utf-8")
    check("_save_as_html: コード本文の'<'/'&'がescapeされている",
          "a &lt; b &amp;&amp; c" in _content_e)
    check("_save_as_html: 生の(未escape)コード本文は出力に含まれない",
          "a < b && c" not in _content_e)

# ---------- _save_as_markdown/_save_as_text/_save_as_html: 既存ファイルが非UTF-8でも
# クラッシュしない (2026-07-23) ----------
# 3関数とも既存 --out ファイルの読み戻しに encoding="utf-8" のみを渡しており
# errors ハンドラが無かった（L3158/L3167/L3212）。このマシンのコンソールが
# cp932 であること(既知の落とし穴 #4)に起因し、既存ファイルが cp932/Shift_JIS
# 等の非UTF-8バイト列を含む場合に read_text が UnicodeDecodeError を送出し、
# _save_answer_to_file 経由で保存ステップ全体が異常終了して、計算し終えた
# 回答が失われていた。これは iteration 41-44 の _save_as_excel/_docx/_pdf
# 修正（保存を絶対にクラッシュさせず回答を失わない）と同じバグクラス。
# errors="replace" を既存内容の読み戻しにのみ追加し、書き込み側の
# encoding="utf-8" とUTF-8正常系の追記/マージ挙動は不変であることを検証する。
# ローカル一時ファイルの読み書きのみで検証し、Ollama/ネットワーク/bench呼び出しは一切ない。
with _tempfile.TemporaryDirectory() as _sv_dir:
    _sv_root = _html_pathlib.Path(_sv_dir)

    # --- _save_as_markdown: 既存ファイルがcp932(日本語)でも例外を送出しない ---
    _md_bad = _sv_root / "bad.md"
    _md_bad.write_bytes("日本語の既存メモ".encode("cp932"))
    try:
        f._save_as_markdown(_md_bad, "q_bad_md", "新しい回答md", 1.0, "")
        _md_bad_exc = None
    except Exception as _exc:
        _md_bad_exc = _exc
    check("_save_as_markdown: 既存ファイルがcp932でも例外を送出しない",
          _md_bad_exc is None)
    if _md_bad_exc is None:
        _md_bad_content = _md_bad.read_text(encoding="utf-8")
        check("_save_as_markdown: 非UTF-8既存ファイルでも新規回答が保存される",
              "新しい回答md" in _md_bad_content)
        try:
            _md_bad.read_text(encoding="utf-8")
            _md_bad_reread_ok = True
        except UnicodeDecodeError:
            _md_bad_reread_ok = False
        check("_save_as_markdown: 保存後ファイルはutf-8として問題なく再読込できる(残留未デコードバイトなし)",
              _md_bad_reread_ok)

    # --- _save_as_text: 既存ファイルが不正単独バイト列でも例外を送出しない ---
    _txt_bad = _sv_root / "bad.txt"
    _txt_bad.write_bytes(b"\x93\xfa\xff")
    try:
        f._save_as_text(_txt_bad, "q_bad_txt", "新しい回答txt", 0.5)
        _txt_bad_exc = None
    except Exception as _exc:
        _txt_bad_exc = _exc
    check("_save_as_text: 既存ファイルが不正単独バイトでも例外を送出しない",
          _txt_bad_exc is None)
    if _txt_bad_exc is None:
        _txt_bad_content = _txt_bad.read_text(encoding="utf-8")
        check("_save_as_text: 非UTF-8既存ファイルでも新規回答が保存される",
              "新しい回答txt" in _txt_bad_content)
        try:
            _txt_bad.read_text(encoding="utf-8")
            _txt_bad_reread_ok = True
        except UnicodeDecodeError:
            _txt_bad_reread_ok = False
        check("_save_as_text: 保存後ファイルはutf-8として問題なく再読込できる(残留未デコードバイトなし)",
              _txt_bad_reread_ok)

    # --- _save_as_html: 既存ファイルが非UTF-8(cp932)でも例外を送出しない ---
    _html_bad = _sv_root / "bad.html"
    _html_bad.write_bytes("<body>日本語</body>".encode("cp932"))
    try:
        f._save_as_html(_html_bad, "q_bad_html", "新しい回答html", 0.7)
        _html_bad_exc = None
    except Exception as _exc:
        _html_bad_exc = _exc
    check("_save_as_html: 既存ファイルがcp932でも例外を送出しない",
          _html_bad_exc is None)
    if _html_bad_exc is None:
        _html_bad_content = _html_bad.read_text(encoding="utf-8")
        check("_save_as_html: 非UTF-8既存ファイルでも新規回答が保存される",
              "新しい回答html" in _html_bad_content)
        try:
            _html_bad.read_text(encoding="utf-8")
            _html_bad_reread_ok = True
        except UnicodeDecodeError:
            _html_bad_reread_ok = False
        check("_save_as_html: 保存後ファイルはutf-8として問題なく再読込できる(残留未デコードバイトなし)",
              _html_bad_reread_ok)

    # --- 回帰: 既存ファイルが無い/有効UTF-8の場合は従来通りbyte-for-byte一致 ---

    # _save_as_markdown: 有効UTF-8の既存ファイルへの追記が従来通り
    _md_good = _sv_root / "good.md"
    _md_existing = "# 既存メモ\n\n有効なUTF-8の既存内容 täüst 日本語。\n\n"
    _md_good.write_text(_md_existing, encoding="utf-8")
    f._save_as_markdown(_md_good, "q_good_md", "有効な既存回答md", 2.5, "")
    _md_good_content = _md_good.read_text(encoding="utf-8")
    _md_ts_m = f.re.search(r"## Q \(([^)]+)\)", _md_good_content)
    check("_save_as_markdown: 追記後にtsが検出できる", _md_ts_m is not None)
    if _md_ts_m:
        _md_ts = _md_ts_m.group(1)
        _md_expected_block = (f"## Q ({_md_ts})\n\nq_good_md\n\n"
                              f"## A\n\n有効な既存回答md\n\n*所要: 2.5s*\n\n---\n\n")
        check("_save_as_markdown: 有効UTF-8既存ファイルへの追記はbyte-for-byte従来通り",
              _md_good_content == _md_existing + _md_expected_block)

    # _save_as_text: 有効UTF-8の既存ファイルへの追記が従来通り
    _txt_good = _sv_root / "good.txt"
    _txt_existing = "有効なUTF-8の既存内容 täüst 日本語。\n\n"
    _txt_good.write_text(_txt_existing, encoding="utf-8")
    f._save_as_text(_txt_good, "q_good_txt", "有効な既存回答txt", 1.5)
    _txt_good_content = _txt_good.read_text(encoding="utf-8")
    _txt_ts_m = f.re.search(r"^\[([^\]]+)\]", _txt_good_content[len(_txt_existing):])
    check("_save_as_text: 追記後にtsが検出できる", _txt_ts_m is not None)
    if _txt_ts_m:
        _txt_ts = _txt_ts_m.group(1)
        _txt_expected_block = (f"[{_txt_ts}]\nQ: q_good_txt\n\nA:\n有効な既存回答txt\n\n"
                               f"(所要 1.5s)\n{'='*60}\n\n")
        check("_save_as_text: 有効UTF-8既存ファイルへの追記はbyte-for-byte従来通り",
              _txt_good_content == _txt_existing + _txt_expected_block)

    # _save_as_html: 有効UTF-8の既存ファイル(2回保存の<body>マージ)が従来通り
    _html_good = _sv_root / "good.html"
    f._save_as_html(_html_good, "q_good_html1", "有効な既存回答html1", 0.3)
    _html_good_first = _html_good.read_text(encoding="utf-8")
    f._save_as_html(_html_good, "q_good_html2", "有効な既存回答html2", 0.4)
    _html_good_content = _html_good.read_text(encoding="utf-8")
    _html_first_body_m = f.re.search(r"<body>\n(.*)</body>", _html_good_first, f.re.DOTALL)
    _html_ts2_m = f.re.search(r"<h2>Q <small>\(([^)]+)\)</small></h2>\n<p>q_good_html2</p>",
                              _html_good_content)
    check("_save_as_html: 1回目保存のbodyが抽出できる", _html_first_body_m is not None)
    check("_save_as_html: 2回目保存のtsが検出できる", _html_ts2_m is not None)
    if _html_first_body_m and _html_ts2_m:
        _html_first_body = _html_first_body_m.group(1)
        _html_ts2 = _html_ts2_m.group(1)
        _html_second_body = (f"<h2>Q <small>({_html_ts2})</small></h2>\n<p>q_good_html2</p>\n"
                             f"<h2>A</h2>\n<p>有効な既存回答html2<br></p>\n"
                             f"<hr><p><small>所要: 0.4s</small></p>\n")
        # 実装(L3210-3218)は既存<body>...</body>間の中身をそのまま次回の
        # existing_body として引き継ぐ。1回目保存時点の内容は
        # "<body>\n" + body1 + "</body></html>" であり、<body>と</body>の
        # 間は "\n" + body1 なので、2回目保存の既存部分には <body> 直後の
        # 改行が引き継がれ "\n\n" + body1 になる（既存の仕様・quirkであり
        # 今回の修正では変更していない）。
        _html_expected = (f"<!DOCTYPE html>\n<html lang='ja'><head>"
                          f"<meta charset='UTF-8'><title>Fugu Output</title></head>\n"
                          f"<body>\n\n{_html_first_body}{_html_second_body}</body></html>")
        check("_save_as_html: 有効UTF-8既存ファイルへの2回目保存(<body>マージ)はbyte-for-byte従来通り",
              _html_good_content == _html_expected)

# ---------- research_search: 反復リサーチループ (dedup / ラウンド上限 / 早期終了) ----------
# research_search は「Conductor/proposer 全員に注入される権威コンテキスト」を作る
# 精度クリティカルな経路だが、これまでオフラインテストが皆無だった。
# f._search_raw と f.ask の両方をモックし、実ネットワーク/Ollama呼び出しを一切発生させずに
# 分岐(Source URL重複排除・大小無視のクエリ重複排除・sufficient判定・空queries早期終了・
# SEARCH_MAX_ROUNDS上限)を検証する。extract_json は本物をそのまま使う(ask()の戻り値の
# 生文字列だけをモックする)。
#
# さらに、万一モックが外れて本物の _search_raw/ask 経由で実ネットワークに落ちないことを
# 保証するため、urllib.request.urlopen と f.subprocess.run も「呼ばれたら即座に
# AssertionError」の番人(センチネル)に差し替える(gotcha #8 の「bounded-loop違反は例外で
# 可視化する」流儀を踏襲)。

_orig_search_raw_rs = f._search_raw
_orig_ask_rs = f.ask
_orig_urlopen_rs = urllib.request.urlopen
_orig_subprocess_run_rs = f.subprocess.run


def _rs_no_network_urlopen(*a, **k):
    raise AssertionError("research_search: モック漏れで実urlopen(ネットワーク)が呼ばれた")


def _rs_no_subprocess_run(*a, **k):
    raise AssertionError("research_search: モック漏れで実subprocess.runが呼ばれた")


def _rs_search_factory(mapping, calls_log, max_calls):
    """mapping: {query: [item, ...]}。想定回数(max_calls)を超えたら例外(上限違反を可視化)。"""
    def _fake(query, max_results=None):
        calls_log.append(query)
        if len(calls_log) > max_calls:
            raise AssertionError(
                f"research_search: 想定回数({max_calls})を超えて_search_rawが呼ばれた"
                "(SEARCH_MAX_ROUNDS違反疑い)")
        return list(mapping.get(query, []))
    return _fake


def _rs_ask_factory(responses, calls_log):
    """responses: 十分性判定として順に返すdictのリスト。想定回数を超えたら例外。"""
    def _fake(model, messages, temperature, think=None, fmt=None, label=None,
              num_predict=None, num_ctx=None):
        calls_log.append(messages)
        idx = len(calls_log) - 1
        if idx >= len(responses):
            raise AssertionError(
                "research_search: 想定回数を超えてask()が呼ばれた(早期終了/ラウンド上限違反疑い)")
        return json.dumps(responses[idx])
    return _fake


try:
    urllib.request.urlopen = _rs_no_network_urlopen
    f.subprocess.run = _rs_no_subprocess_run

    # --- (A) Source-URL重複排除 と Sourceなし項目の先頭80文字重複排除 ---
    _itemA1 = "[T1]\nBody1 for source dedup test\nSource: http://example.com/a"
    _itemA2 = "No-source item padding text to reach eighty chars exactly xxxxxxxxxxxxxxxxxxxxxxxxx"
    _itemA3 = "[T3]\nBody3 brand new item\nSource: http://example.com/b"
    _searchA_calls = []
    _askA_calls = []
    try:
        f._search_raw = _rs_search_factory(
            {"Q_DEDUP": [_itemA1, _itemA2],
             "Q_DEDUP_R2": [_itemA1, _itemA2, _itemA3]},  # R2で同じ2件+新規1件を返す
            _searchA_calls, max_calls=2)
        f.ask = _rs_ask_factory(
            [{"sufficient": False, "missing": "x", "queries": ["Q_DEDUP_R2"]},
             {"sufficient": True, "missing": "", "queries": []}],
            _askA_calls)
        _resA = f.research_search("Q_DEDUP")
        check("research_search: Source URL重複は2巡目で再注入されない",
              _resA.count("http://example.com/a") == 1)
        check("research_search: Sourceなし項目は先頭80文字一致で重複排除される",
              _resA.count("No-source item padding text to reach eighty chars exactly") == 1)
        check("research_search: 新規Source項目はきちんと追加される",
              "http://example.com/b" in _resA)
        check("research_search: (A)_search_rawはR1・R2の2回のみ呼ばれる",
              _searchA_calls == ["Q_DEDUP", "Q_DEDUP_R2"])
    finally:
        f._search_raw = _orig_search_raw_rs
        f.ask = _orig_ask_rs

    # --- (B) 実行済みクエリの大小無視の重複排除(同一クエリの再検索防止) ---
    _itemB1 = "[B1]\nfirst round body\nSource: http://example.com/r1"
    _itemB2 = "[B2]\nsecond round body\nSource: http://example.com/r2"
    _searchB_calls = []
    _askB_calls = []
    try:
        f._search_raw = _rs_search_factory(
            {"Case Test Query": [_itemB1], "New Angle Query": [_itemB2]},
            _searchB_calls, max_calls=2)
        f.ask = _rs_ask_factory(
            [{"sufficient": False, "missing": "x",
              "queries": ["CASE TEST QUERY", "New Angle Query"]},  # 1つ目は既実行クエリの大小違い
             {"sufficient": True, "missing": "", "queries": []}],
            _askB_calls)
        _resB = f.research_search("Case Test Query")
        check("research_search: 既実行クエリは大小無視で再検索されない",
              _searchB_calls == ["Case Test Query", "New Angle Query"])
        check("research_search: 重複クエリをスキップしつつ新規クエリの結果は反映される",
              "http://example.com/r1" in _resB and "http://example.com/r2" in _resB)
    finally:
        f._search_raw = _orig_search_raw_rs
        f.ask = _orig_ask_rs

    # --- (C) sufficient=true で即座に早期終了(以降のラウンドの検索が発生しない) ---
    _itemC1 = "[C1]\nsufficient stop body\nSource: http://example.com/c1"
    _searchC_calls = []
    _askC_calls = []
    try:
        f._search_raw = _rs_search_factory({"Suff Stop Query": [_itemC1]},
                                            _searchC_calls, max_calls=1)
        f.ask = _rs_ask_factory([{"sufficient": True, "missing": "", "queries": []}],
                                 _askC_calls)
        _resC = f.research_search("Suff Stop Query")
        check("research_search: sufficient=trueで即座に早期終了(検索は1ラウンドのみ)",
              _searchC_calls == ["Suff Stop Query"])
        check("research_search: sufficient=trueなら判定は1回のみ呼ばれる",
              len(_askC_calls) == 1)
        check("research_search: 早期終了時もそのラウンドの結果は反映される",
              "http://example.com/c1" in _resC)
    finally:
        f._search_raw = _orig_search_raw_rs
        f.ask = _orig_ask_rs

    # --- (D) 空/欠落のqueriesで早期終了(以降のラウンドの検索が発生しない) ---
    _itemD1 = "[D1]\nempty queries stop body\nSource: http://example.com/d1"
    _searchD_calls = []
    _askD_calls = []
    try:
        f._search_raw = _rs_search_factory({"Empty Queries Query": [_itemD1]},
                                            _searchD_calls, max_calls=1)
        f.ask = _rs_ask_factory(
            [{"sufficient": False, "missing": "still missing", "queries": []}],
            _askD_calls)
        _resD = f.research_search("Empty Queries Query")
        check("research_search: 空queriesリストで早期終了(検索は1ラウンドのみ)",
              _searchD_calls == ["Empty Queries Query"])
        check("research_search: 空queries早期終了時もそのラウンドの結果は反映される",
              "http://example.com/d1" in _resD)
    finally:
        f._search_raw = _orig_search_raw_rs
        f.ask = _orig_ask_rs

    _itemD2 = "[D2]\nmissing queries key stop body\nSource: http://example.com/d2"
    _searchD2_calls = []
    _askD2_calls = []
    try:
        f._search_raw = _rs_search_factory({"Missing Key Query": [_itemD2]},
                                            _searchD2_calls, max_calls=1)
        # "queries" キー自体が欠落したJSON(j.get("queries") が None になるケース)
        f.ask = _rs_ask_factory([{"sufficient": False, "missing": "still missing"}],
                                 _askD2_calls)
        _resD2 = f.research_search("Missing Key Query")
        check("research_search: queriesキー欠落でも早期終了する(検索は1ラウンドのみ)",
              _searchD2_calls == ["Missing Key Query"])
    finally:
        f._search_raw = _orig_search_raw_rs
        f.ask = _orig_ask_rs

    # --- (E) SEARCH_MAX_ROUNDS上限: 常にsufficient=falseかつ新規クエリでも上限で打ち切り ---
    _searchE_calls = []
    _askE_calls = []
    try:
        f._search_raw = _rs_search_factory(
            {"Bound Round Query": ["[E1]\nr1\nSource: http://example.com/e1"],
             "Bound Q2": ["[E2]\nr2\nSource: http://example.com/e2"],
             "Bound Q3": ["[E3]\nr3\nSource: http://example.com/e3"]},
            _searchE_calls, max_calls=f.SEARCH_MAX_ROUNDS)
        f.ask = _rs_ask_factory(
            [{"sufficient": False, "missing": "x", "queries": ["Bound Q2"]},
             {"sufficient": False, "missing": "y", "queries": ["Bound Q3"]}],
            _askE_calls)  # ちょうど MAX_ROUNDS-1 回分しか用意しない(最終ラウンドは判定なし)
        _resE = f.research_search("Bound Round Query")
        check("research_search: 新規クエリが尽きなくてもSEARCH_MAX_ROUNDSでちょうど打ち切り",
              len(_searchE_calls) == f.SEARCH_MAX_ROUNDS == 3)
        check("research_search: 最終ラウンドでは十分性判定(ask)を呼ばない",
              len(_askE_calls) == f.SEARCH_MAX_ROUNDS - 1)
        check("research_search: 上限到達までの全ラウンドの結果が反映される",
              all(u in _resE for u in ("http://example.com/e1", "http://example.com/e2",
                                        "http://example.com/e3")))
    finally:
        f._search_raw = _orig_search_raw_rs
        f.ask = _orig_ask_rs

    # --- (F) 全ラウンドで結果ゼロ -> 空文字を返す ---
    _searchF_calls = []
    _askF_calls = []
    try:
        f._search_raw = _rs_search_factory({}, _searchF_calls, max_calls=f.SEARCH_MAX_ROUNDS)
        f.ask = _rs_ask_factory(
            [{"sufficient": False, "missing": "x", "queries": ["F Round2"]},
             {"sufficient": False, "missing": "y", "queries": ["F Round3"]}],
            _askF_calls)
        _resF = f.research_search("F Round1")
        check("research_search: 全ラウンドで結果ゼロなら空文字を返す", _resF == "")
        check("research_search: (F)検索はSEARCH_MAX_ROUNDS回実施された上での空文字",
              len(_searchF_calls) == f.SEARCH_MAX_ROUNDS)
    finally:
        f._search_raw = _orig_search_raw_rs
        f.ask = _orig_ask_rs

    # --- (G) 結果が1件でもあれば日付入りフレッシュネスヘッダーが付与される ---
    _searchG_calls = []
    _askG_calls = []
    try:
        f._search_raw = _rs_search_factory(
            {"Header Query": ["[G1]\nheader body\nSource: http://example.com/g1"]},
            _searchG_calls, max_calls=1)
        f.ask = _rs_ask_factory([{"sufficient": True, "missing": "", "queries": []}],
                                 _askG_calls)
        _resG = f.research_search("Header Query")
        _expected_date_g = f.time.strftime("%Y-%m-%d")
        check("research_search: 結果ありなら取得日入りヘッダーが付与される",
              _resG.startswith(f"## Web Search Results (取得日: {_expected_date_g})"))
        check("research_search: ヘッダーに本文(検索結果)が続く",
              "http://example.com/g1" in _resG)
    finally:
        f._search_raw = _orig_search_raw_rs
        f.ask = _orig_ask_rs

    # --- (H) 2026-07-22 修正済み挙動の固定化: 先頭(唯一の)結果が SEARCH_CONTEXT_CHARS を
    #     超えていても、body を空文字のまま break せず、先頭結果を上限まで切り詰めて必ず
    #     注入する(精度優先。sufficient=True と判定された唯一の具体的事実を黙って
    #     落とさない)。旧挙動(body="")はイテレーション38で特性テストとして固定されていたが、
    #     イテレーション39で修正した。
    _hugeH = "[Huge]\n" + ("Z" * (f.SEARCH_CONTEXT_CHARS + 500)) + "\nSource: http://example.com/huge"
    _searchH_calls = []
    _askH_calls = []
    try:
        f._search_raw = _rs_search_factory({"Huge Item Query": [_hugeH]},
                                            _searchH_calls, max_calls=1)
        f.ask = _rs_ask_factory([{"sufficient": True, "missing": "", "queries": []}],
                                 _askH_calls)
        _resH = f.research_search("Huge Item Query")
        _headerH = f"## Web Search Results (取得日: {f.time.strftime('%Y-%m-%d')})"
        _bodyH = _resH[len(_headerH):] if _resH.startswith(_headerH) else _resH
        check("research_search: 先頭項目がSEARCH_CONTEXT_CHARS超過でもheaderが付与される",
              _resH.startswith(_headerH))
        check("research_search: 先頭項目がSEARCH_CONTEXT_CHARS超過でもbodyが空にならない"
              "(切り詰めてでも必ず注入)",
              _bodyH.strip() != "")
        check("research_search: 切り詰められたbodyは先頭結果のプレフィックスを含む",
              _hugeH[:200] in _resH)
        check("research_search: 切り詰められたbodyの全文Sourceまでは含まれない"
              "(SEARCH_CONTEXT_CHARSで打ち切られている)",
              "http://example.com/huge" not in _resH)
    finally:
        f._search_raw = _orig_search_raw_rs
        f.ask = _orig_ask_rs

finally:
    f._search_raw = _orig_search_raw_rs
    f.ask = _orig_ask_rs
    urllib.request.urlopen = _orig_urlopen_rs
    f.subprocess.run = _orig_subprocess_run_rs

# ---------- _save_as_excel: XML不正制御文字によるIllegalCharacterError耐性 (2026-07-22) ----------
# openpyxl の ws.append() は XML 1.0 で禁止された制御文字(0x00-0x08/0x0B/0x0C/0x0E-0x1F)を
# 含むセルに対して openpyxl.utils.exceptions.IllegalCharacterError（ImportErrorではない
# 素のException）を送出する。従来コードは except ImportError しか捕捉しておらず、
# LLM回答に混入したフォームフィード(\x0c)/ANSIエスケープ(\x1b)/NUL(\x00)等が原因で
# _save_answer_to_file 全体が異常終了していた。ここでは openpyxl の有無を検出し、
# 両方の分岐（サニタイズして.xlsx生成 / 未インストール時の.csvフォールバック）を検証する。
# 本物のOllama/ネットワーク呼び出しは一切行わない。
import pathlib as _pathlib_xlsx

try:
    import openpyxl as _openpyxl_probe
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False

with _tempfile.TemporaryDirectory() as _xlsx_dir:
    _xlsx_root = _pathlib_xlsx.Path(_xlsx_dir)

    def _trim_trailing_none(_row):
        # openpyxl の iter_rows() はシート全体の最大列数まで各行を None で
        # パディングして返す(この関数の変更とは無関係の既存仕様)。比較用に
        # 末尾の None パディングだけを取り除く。
        _r = list(_row)
        while _r and _r[-1] is None:
            _r.pop()
        return _r

    if _HAS_OPENPYXL:
        # 制御文字はわざと文字列の「途中」に置く。ただし \x0b/\x0c/\x1c-\x1e は
        # Python の str.splitlines() 自体が改行境界として解釈し、answer.splitlines()
        # の時点でセル文字列に残らず消費されてしまう(この関数の既存仕様であり、
        # 今回の修正対象外)。そのため実際にセルへ到達し検証可能な、XML不正かつ
        # splitlines非対象の制御文字 \x1b(ESC)/\x00(NUL) を用いる。
        _illegal_answer = "name,age\nAli\x1bce,30\nB\x00ob,25"
        _out_illegal = _xlsx_root / "illegal.xlsx"
        _exc = None
        try:
            f._save_as_excel(_out_illegal, _illegal_answer)
        except Exception as _e:
            _exc = _e
        check("_save_as_excel: XML不正制御文字混入でも例外を送出しない(IllegalCharacterError回帰)",
              _exc is None)
        check("_save_as_excel: 制御文字混入時も.xlsxファイルが生成される", _out_illegal.exists())

        if _exc is None and _out_illegal.exists():
            _wb_illegal = _openpyxl_probe.load_workbook(str(_out_illegal))
            _rows_illegal = [_trim_trailing_none(row) for row in _wb_illegal.active.iter_rows(values_only=True)]
            check("_save_as_excel: 制御文字は除去されつつ実データ(表の中身)は保持される",
                  _rows_illegal == [["name", "age"], ["Alice", "30"], ["Bob", "25"]])

        # 制御文字を含まない通常回答は、列分割(re.split(r"[,\t|]", line))が
        # 従来とバイト単位で同一であること。
        _clean_answer = "a,b\tc|d\nx, y , z"
        _out_clean = _xlsx_root / "clean.xlsx"
        f._save_as_excel(_out_clean, _clean_answer)
        _wb_clean = _openpyxl_probe.load_workbook(str(_out_clean))
        _rows_clean = [_trim_trailing_none(row) for row in _wb_clean.active.iter_rows(values_only=True)]
        _expected_clean = [
            [c.strip() for c in f.re.split(r"[,\t|]", "a,b\tc|d")],
            [c.strip() for c in f.re.split(r"[,\t|]", "x, y , z")],
        ]
        check("_save_as_excel: 制御文字なしの通常回答は従来通り列分割される(既存挙動不変)",
              _rows_clean == _expected_clean)
    else:
        # openpyxl 未インストール環境: 既存の.csvフォールバック(拡張子/メッセージ/戻り値)が
        # 従来通り機能すること。
        _out_missing = _xlsx_root / "missing.xlsx"
        _result_missing = f._save_as_excel(_out_missing, "a,b\nc,d")
        check("_save_as_excel: openpyxl未インストール時は.csvへフォールバックする",
              _result_missing == _out_missing.with_suffix(".csv"))
        check("_save_as_excel: フォールバック時の.csvファイルが実際に書かれる",
              _result_missing is not None and _result_missing.exists())

# ---------- _save_as_docx: XML不正制御文字によるValueError耐性 (2026-07-22) ----------
# python-docx (lxml) の add_paragraph/add_heading は XML 1.0 で禁止された制御文字
# (0x00-0x08/0x0B/0x0C/0x0E-0x1F) を含む文字列を渡されると ValueError を送出する。
# 従来コードは except ImportError しか捕捉しておらず、question をそのまま
# add_paragraph に渡していた事もあり、LLM回答/questionに混入したANSIエスケープ
# (\x1b)やNUL(\x00)等が原因で _save_answer_to_file 全体が異常終了していた
# （iteration 41 の _save_as_excel の IllegalCharacterError 修正と同じバグクラス）。
# ここでは python-docx の有無を検出し、サニタイズして.docx生成できること・
# 未インストール時は既存通り.mdへフォールバックすること・ビルド/保存失敗時も
# .mdへ安全に降格することを検証する。本物のOllama/ネットワーク呼び出しは一切行わない。
import pathlib as _pathlib_docx

try:
    import docx as _docx_probe
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False

with _tempfile.TemporaryDirectory() as _docx_dir:
    _docx_root = _pathlib_docx.Path(_docx_dir)

    if _HAS_DOCX:
        # 制御文字はわざと単語の間の空白の位置に置く。除去後も単語同士がくっつかず
        # 実データ(本文)が読み取れることを確認する。
        _illegal_question = "Hello \x1bWorld \x00Test?"
        _illegal_answer = "Answer \x1bline one.\n\n```python\nprint(\x001)\n```\n\nFinal \x00line."
        _out_illegal_docx = _docx_root / "illegal.docx"
        _exc_docx = None
        try:
            _ret_illegal = f._save_as_docx(_out_illegal_docx, _illegal_question, _illegal_answer, 0.42)
        except Exception as _e:
            _exc_docx = _e
            _ret_illegal = None
        check("_save_as_docx: XML不正制御文字混入でも例外を送出しない(ValueError回帰)",
              _exc_docx is None)
        check("_save_as_docx: 制御文字混入時も.docxファイルが生成される", _out_illegal_docx.exists())
        check("_save_as_docx: 制御文字混入でも成功時は.docxパス(None)を返す(戻り値契約維持)",
              _ret_illegal is None)

        if _exc_docx is None and _out_illegal_docx.exists():
            _doc_illegal = _docx_probe.Document(str(_out_illegal_docx))
            _texts_illegal = [p.text for p in _doc_illegal.paragraphs]
            _all_text_illegal = "\n".join(_texts_illegal)
            check("_save_as_docx: 制御文字(ESC/NUL)は本文から除去される",
                  "\x1b" not in _all_text_illegal and "\x00" not in _all_text_illegal)
            check("_save_as_docx: 制御文字除去後もquestion本文の単語は保持される",
                  "Hello World Test?" in _texts_illegal)
            check("_save_as_docx: 制御文字除去後もanswer本文の単語は保持される",
                  "Answer line one." in _texts_illegal and "Final line." in _texts_illegal)
            check("_save_as_docx: 制御文字除去後もコードブロック本文は保持される",
                  "print(1)" in _texts_illegal)

        # 制御文字を含まない通常回答は、コードフェンス解析・見出し構造・所要時間行が
        # 従来通り生成されること(既存挙動不変の回帰確認)。
        _clean_question = "Plain question"
        _clean_answer = "Intro line\n\n```python\nprint('hi')\n```\n\nOutro line"
        _out_clean_docx = _docx_root / "clean.docx"
        _ret_clean = f._save_as_docx(_out_clean_docx, _clean_question, _clean_answer, 2.5)
        check("_save_as_docx: 制御文字なしの通常回答も成功時はNoneを返す", _ret_clean is None)
        _doc_clean = _docx_probe.Document(str(_out_clean_docx))
        _paras_clean = list(_doc_clean.paragraphs)
        _texts_clean = [p.text for p in _paras_clean]
        _styles_clean = [p.style.name for p in _paras_clean]
        check("_save_as_docx: Q見出しが先頭に生成される(既存挙動不変)",
              len(_texts_clean) > 0 and _texts_clean[0].startswith("Q (") and _styles_clean[0] == "Heading 1")
        check("_save_as_docx: question本文がQ見出しの直後に生成される(既存挙動不変)",
              len(_texts_clean) > 1 and _texts_clean[1] == "Plain question")
        check("_save_as_docx: A見出しが生成される(既存挙動不変)",
              len(_texts_clean) > 2 and _texts_clean[2] == "A" and _styles_clean[2] == "Heading 1")
        check("_save_as_docx: コードフェンス本文はNo Spacingスタイルの段落になる(既存挙動不変)",
              "print('hi')" in _texts_clean and
              _styles_clean[_texts_clean.index("print('hi')")] == "No Spacing")
        check("_save_as_docx: 所要時間の行が末尾に生成される(既存挙動不変)",
              _texts_clean[-1] == "所要: 2.5s")

        # python-docx が未インストールの場合の分岐: sys.modules['docx'] を None に
        # することで `import docx` に ImportError を送出させる（実インストール状態を
        # 変更せずに未インストール環境を模擬する標準的な手法）。
        _orig_docx_mod = sys.modules.get("docx")
        sys.modules["docx"] = None
        try:
            _out_missing_docx = _docx_root / "missing.docx"
            _ret_missing = f._save_as_docx(_out_missing_docx, "Q?", "A.", 1.0)
            check("_save_as_docx: python-docx未インストール時は.mdへフォールバックする",
                  _ret_missing == _out_missing_docx.with_suffix(".md"))
            check("_save_as_docx: フォールバック時の.mdファイルが実際に書かれる",
                  _ret_missing is not None and _ret_missing.exists())
        finally:
            if _orig_docx_mod is not None:
                sys.modules["docx"] = _orig_docx_mod
            else:
                del sys.modules["docx"]

        # ビルド/保存自体が失敗するケース(IllegalXml以外の残存エラーも含む)を
        # docx.document.Document.save をモンキーパッチして模擬し、例外が
        # 外へ漏れずに.mdへ降格することを確認する。
        import docx.document as _docx_document_mod
        _orig_save_method = _docx_document_mod.Document.save

        def _boom_save(self, *_a, **_kw):
            raise RuntimeError("simulated docx save failure")

        _docx_document_mod.Document.save = _boom_save
        try:
            _out_fail_docx = _docx_root / "fail.docx"
            _exc_fail = None
            try:
                _ret_fail = f._save_as_docx(_out_fail_docx, "Q?", "A.", 1.0)
            except Exception as _e:
                _exc_fail = _e
                _ret_fail = None
            check("_save_as_docx: 保存失敗時も例外は外へ伝播しない", _exc_fail is None)
            check("_save_as_docx: 保存失敗時は.mdへフォールバックする",
                  _ret_fail == _out_fail_docx.with_suffix(".md"))
            check("_save_as_docx: 保存失敗フォールバック時の.mdファイルが実際に書かれる",
                  _ret_fail is not None and _ret_fail.exists())
        finally:
            _docx_document_mod.Document.save = _orig_save_method
    else:
        # python-docx 未インストール環境: 既存の.mdフォールバック(拡張子/戻り値)が
        # 従来通り機能すること。
        _out_missing_docx = _docx_root / "missing.docx"
        _result_missing_docx = f._save_as_docx(_out_missing_docx, "Q?", "A.", 1.0)
        check("_save_as_docx: python-docx未インストール時は.mdへフォールバックする",
              _result_missing_docx == _out_missing_docx.with_suffix(".md"))
        check("_save_as_docx: フォールバック時の.mdファイルが実際に書かれる",
              _result_missing_docx is not None and _result_missing_docx.exists())

# ---------- _save_as_pdf: fpdf2 ビルド/出力失敗時の .md フォールバック耐性 (2026-07-22) ----------
# fpdf2 には 'DejaVu' という名前で事前登録された組み込み Unicode フォントは無く、
# set_font("DejaVu") は FPDFException となり Helvetica (コアlatin-1フォント) へ
# 縮退する。既定言語である日本語などの非ASCII文字を multi_cell に渡すと
# FPDFUnicodeEncodingException（ImportErrorではない素のException）が送出される。
# 従来コードは except ImportError しか捕捉しておらず、_save_as_pdf の呼び出し元
# _save_answer_to_file 自体にもガードが無いため、回答保存ステップ全体が
# 異常終了していた（iteration 41 の _save_as_excel の IllegalCharacterError 修正、
# iteration 43 の _save_as_docx の制御文字 ValueError 修正と同じバグクラス）。
# ここでは fpdf2 の有無を検出し、(1) 日本語などUnicodeを含む通常呼び出しが例外を
# 送出せず実ファイルを生成すること、(2) ビルド/出力自体が失敗しても.mdへ安全に
# 降格すること、(3) fpdf2未インストール時の既存.mdフォールバック(メッセージ/戻り値)
# が変わらないことを検証する。本物のOllama/ネットワーク呼び出しは一切行わない。
import pathlib as _pathlib_pdf

try:
    import fpdf as _fpdf_probe
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False

with _tempfile.TemporaryDirectory() as _pdf_dir:
    _pdf_root = _pathlib_pdf.Path(_pdf_dir)

    if _HAS_FPDF:
        # 日本語(Unicode)を含む question/answer で呼んでも例外が伝播せず、
        # 実ファイル(.pdf または .md フォールバック)が生成されること。
        _ja_question = "日本語の質問です。テスト？"
        _ja_answer = "日本語の回答です。\n改行を含む本文。"
        _out_ja = _pdf_root / "ja.pdf"
        _exc_ja = None
        try:
            _ret_ja = f._save_as_pdf(_out_ja, _ja_question, _ja_answer, 1.23)
        except Exception as _e:
            _exc_ja = _e
            _ret_ja = None
        check("_save_as_pdf: 日本語/Unicode本文でも例外を送出しない(FPDFUnicodeEncodingException回帰)",
              _exc_ja is None)
        _produced_ja = _out_ja.exists() or _out_ja.with_suffix(".md").exists()
        check("_save_as_pdf: 日本語/Unicode本文でも実ファイル(.pdfまたは.md)が生成される",
              _produced_ja)

        # ビルド/出力自体が失敗するケース(Unicode以外の残存エラーも含む)を
        # fpdf.FPDF.output をモンキーパッチして模擬し、例外が外へ漏れずに
        # .md へ降格することを確認する。
        _orig_output_method = _fpdf_probe.FPDF.output

        def _boom_output(self, *_a, **_kw):
            raise RuntimeError("simulated pdf output failure")

        _fpdf_probe.FPDF.output = _boom_output
        try:
            _out_fail_pdf = _pdf_root / "fail.pdf"
            _exc_fail_pdf = None
            try:
                _ret_fail_pdf = f._save_as_pdf(_out_fail_pdf, "Q?", "A.", 1.0)
            except Exception as _e:
                _exc_fail_pdf = _e
                _ret_fail_pdf = None
            check("_save_as_pdf: 保存失敗時も例外は外へ伝播しない", _exc_fail_pdf is None)
            check("_save_as_pdf: 保存失敗時は.mdへフォールバックする",
                  _ret_fail_pdf == _out_fail_pdf.with_suffix(".md"))
            check("_save_as_pdf: 保存失敗フォールバック時の.mdファイルが実際に書かれる",
                  _ret_fail_pdf is not None and _ret_fail_pdf.exists())
        finally:
            _fpdf_probe.FPDF.output = _orig_output_method
    else:
        # fpdf2 が本当に存在しない環境: 既存の.mdフォールバック(拡張子/戻り値)が
        # 従来通り機能すること(iteration 41 の else分岐スタイルを踏襲)。
        _out_missing_pdf_real = _pdf_root / "missing_real.pdf"
        _result_missing_pdf_real = f._save_as_pdf(_out_missing_pdf_real, "Q?", "A.", 1.0)
        check("_save_as_pdf: fpdf2未インストール環境では.mdへフォールバックする",
              _result_missing_pdf_real == _out_missing_pdf_real.with_suffix(".md"))
        check("_save_as_pdf: フォールバック時の.mdファイルが実際に書かれる(未インストール環境)",
              _result_missing_pdf_real is not None and _result_missing_pdf_real.exists())

    # 回帰ガード: fpdf2 が実際にインストールされている環境でも sys.modules['fpdf']
    # を None にすることで `from fpdf import FPDF` に ImportError を送出させ
    # （実インストール状態を変更せずに未インストール環境を模擬する標準的な手法）、
    # 既存の ImportError フォールバック分岐(メッセージ/戻り値)が変わっていない
    # ことを検証する。
    _orig_fpdf_mod = sys.modules.get("fpdf")
    sys.modules["fpdf"] = None
    try:
        _out_missing_pdf = _pdf_root / "missing.pdf"
        _ret_missing_pdf = f._save_as_pdf(_out_missing_pdf, "Q?", "A.", 1.0)
        check("_save_as_pdf: fpdf2未インストール時は.mdへフォールバックする(既存メッセージ/戻り値不変)",
              _ret_missing_pdf == _out_missing_pdf.with_suffix(".md"))
        check("_save_as_pdf: フォールバック時の.mdファイルが実際に書かれる",
              _ret_missing_pdf is not None and _ret_missing_pdf.exists())
    finally:
        if _orig_fpdf_mod is not None:
            sys.modules["fpdf"] = _orig_fpdf_mod
        else:
            del sys.modules["fpdf"]

# ---------- _trim_history: 直近ペアを消さない (2026-07-23回帰) ----------
# 背景: ask_fugu は直近の [user, assistant] ペアを _HISTORY に追記した「直後」に
# _trim_history を呼ぶ。旧ガード `len(history) >= 2` だと、履歴が直近ペア1組
# (長さ2) まで削られた状態でもまだ MAX_HISTORY_CHARS 超過ならループに入り、
# pop(0) を2回叩いて直近ペアごと消してしまい、履歴が完全に空になっていた
# (直後に "[会話履歴: 0 往復保持中]" と出る既知の再現手順)。
# `len(history) > 2` に直したことで、削除対象が最後の1ペアだけになった時点で
# 必ず停止し、多ターン対話の文脈(精度に直結)を失わないことをここで検証する。
_orig_max_hist_chars = f.MAX_HISTORY_CHARS

# (1) 予算内なら無加工でそのまま(同一オブジェクト・同一順序・同一長)
_hist_small = [{"role": "user", "content": "hello"},
               {"role": "assistant", "content": "world"}]
_hist_small_ids_before = [id(m) for m in _hist_small]
f._trim_history(_hist_small)
check("trim_history: 予算内は要素数不変(無加工)", len(_hist_small) == 2)
check("trim_history: 予算内は同一オブジェクト・順序も不変(無加工)",
      [id(m) for m in _hist_small] == _hist_small_ids_before)

# (2) バグ再現(主目的): 直近1ペアだけの状態で単独で予算超過しても消えない
try:
    f.MAX_HISTORY_CHARS = 4000
    _hist_one_pair = [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": "X" * 6000}]
    f._trim_history(_hist_one_pair)
    check("trim_history: 直近1ペアが単独で予算超過しても消えない(バグ再現)",
          len(_hist_one_pair) == 2)
    check("trim_history: 残るのは直近のuser/assistantペアそのもの(内容不変)",
          _hist_one_pair[0]["role"] == "user"
          and _hist_one_pair[0]["content"] == "q"
          and _hist_one_pair[1]["role"] == "assistant"
          and _hist_one_pair[1]["content"] == "X" * 6000)
finally:
    f.MAX_HISTORY_CHARS = _orig_max_hist_chars

# (3) 回帰: 複数ペアが予算超過なら古いペアから先頭ごと削除され、
#     生き残りは元リストの末尾(tail)と一致する(順序維持・ペア単位で削除)
try:
    f.MAX_HISTORY_CHARS = 250
    _pair1 = [{"role": "user", "content": "u" * 50},
              {"role": "assistant", "content": "a" * 50}]
    _pair2 = [{"role": "user", "content": "u" * 50},
              {"role": "assistant", "content": "a" * 50}]
    _pair3 = [{"role": "user", "content": "u" * 50},
              {"role": "assistant", "content": "a" * 50}]
    _hist_multi = _pair1 + _pair2 + _pair3
    _hist_multi_expected_tail = _pair2 + _pair3
    f._trim_history(_hist_multi)
    check("trim_history: 予算超過時は古いペアが先頭からペア単位で削除される",
          _hist_multi == _hist_multi_expected_tail)
    check("trim_history: 生き残りは元リストの末尾と一致する(順序維持)",
          len(_hist_multi) == 4 and _hist_multi[0]["content"] == "u" * 50)
finally:
    f.MAX_HISTORY_CHARS = _orig_max_hist_chars

# (4) 全ペアが予算超過でも直近の1ペアだけは必ず残る(先頭ペアのみ削除)
try:
    f.MAX_HISTORY_CHARS = 4000
    _big_pair1 = [{"role": "user", "content": "u" * 3000},
                  {"role": "assistant", "content": "a" * 3000}]
    _big_pair2 = [{"role": "user", "content": "u" * 3000},
                  {"role": "assistant", "content": "a" * 3000}]
    _hist_two_big = _big_pair1 + _big_pair2
    f._trim_history(_hist_two_big)
    check("trim_history: 2ペアとも予算超過でも直近ペアのみ残る(先頭ペア削除)",
          _hist_two_big == _big_pair2)
finally:
    f.MAX_HISTORY_CHARS = _orig_max_hist_chars

# (5) 一般化: 極小予算・様々なペア数でも空にならず、例外を送出せず完走する(ハングしない)
try:
    f.MAX_HISTORY_CHARS = 1
    _no_hang_exc = None
    try:
        for _n_pairs in (1, 2, 3, 5):
            _h = []
            for _i in range(_n_pairs):
                _h.append({"role": "user", "content": "u" * 500})
                _h.append({"role": "assistant", "content": "a" * 500})
            f._trim_history(_h)
            check(f"trim_history: 予算1でも空にならない(ペア数={_n_pairs})",
                  len(_h) >= 2)
    except Exception as _e:
        _no_hang_exc = _e
    check("trim_history: 極小予算でも例外を送出せず完走する(ハング/クラッシュなし)",
          _no_hang_exc is None)
finally:
    f.MAX_HISTORY_CHARS = _orig_max_hist_chars

print()
if _FAILS:
    print(f"FAILED: {len(_FAILS)} 件 -> {_FAILS}")
    raise SystemExit(1)
print("ALL PASSED")
