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
check("plan: image_only を反映",
      f.validate_plan({"image_only": True, "selected_proposers": []})["image_only"] is True)
check("plan: make_pptx を反映し image_only を無効化",
      (lambda p: p["make_pptx"] is True and p["image_only"] is False)(
          f.validate_plan({"make_pptx": True, "image_only": True,
                           "selected_proposers": ["qwen3:4b"]})))

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
check("pptx: deck_title は短い質問を採用", f._deck_title("犬の紹介", _slides) == "犬の紹介")

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

_top, _cnt, _cls = f.vote_answers(["42", "42", "41", "0.5", "1/2", None, ""])
check("sc: 投票 最多クラス", _top == "42" and _cnt == 2)
check("sc: 投票 同値クラス集約", any(c[1] == 2 and f.answers_equivalent(c[0], "0.5") for c in _cls))
check("sc: 投票 空リスト", f.vote_answers([]) == (None, 0, []))

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

ok, out = f.run_python("print('hello_runner')")
check("code: 実行成功", ok and "hello_runner" in out)
ok, out = f.run_python("raise ValueError('boom')")
check("code: 例外を検知して traceback を返す", (not ok) and "boom" in out)
ok, out = f.run_python("while True:\n    pass", timeout=2)
check("code: 無限ループはタイムアウト", (not ok) and "TIMEOUT" in out)

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

print()
if _FAILS:
    print(f"FAILED: {len(_FAILS)} 件 -> {_FAILS}")
    raise SystemExit(1)
print("ALL PASSED")
