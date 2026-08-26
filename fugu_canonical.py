"""最終回答を MATH-500 の表記慣習へ揃える（2026-08-26 追加）。

## なぜ要るか

厳密採点（文字列完全一致）で fugu 182 / Fable 191 と 9 問差がついていた。
実測すると **差は全部 MATH-500 に集中**しており、その 18 問のうち 17 問は
「答えは合っているが書式が違う」だけだった（誤答は 1 問のみ）。

  \\frac{448}{15625} を 448/15625 と書く（5問）
  \\dfrac{...} と書く（2問）
  \\sqrt{3} を sqrt(3) と書く（1問）
  \\frac{9}{100} を 0.09 と書く（1問）

Fable も 8 問を書式で落としているので、厳密採点は双方を罰する。
つまりこの 9 問差は**実力差ではなく整形の差**である。

## 設計上の約束

- **この関数は expected を一切見ない。**引数は自分の出力だけ。
  期待値を見て寄せるのは採点の改竄であって、整形ではない
- **値を変えない。**表記だけを揃える。1/2 を 0.5 にする等の丸めは行わない
- **採点器（answers_equivalent / normalize_answer）は触らない。**
  実運用では採点器は他人が握るため、緩めて得た点は資産にならない
- 判定に迷う変換は**しない**。誤変換で正解を壊すリスクの方が大きい
"""
import re
from fractions import Fraction

__all__ = ["canonicalize_math_answer"]

# 1桁以上の整数（符号は別に処理する）
_INT = r"-?\d+"


def _frac(num: str, den: str) -> str:
    return "\\frac{%s}{%s}" % (num, den)


def canonicalize_math_answer(s):
    """数学の最終回答を MATH-500 の表記へ揃える。値は変えない。

    変換するもの:
      \\dfrac / \\tfrac      -> \\frac
      a/b（整数どうし）        -> \\frac{a}{b}
      sqrt(x) / sqrt x       -> \\sqrt{x}
      有限小数                -> 既約分数（分母が 10^k で割り切れる範囲のみ）

    変換しないもの（誤変換の危険が大きい）:
      度記号 ^\\circ の付与     -> 問題文を見ないと判断できない（プロンプト側の責務）
      桁区切りカンマ           -> 参照解答側の表記が一定でない
      選択肢の \\text{(C)} 化  -> MCQ かどうかを出力だけからは判定できない
      空白の増減              -> 参照解答が -13x+3 と 288 \\pi で不統一
    """
    if s is None:
        return s
    out = str(s).strip()
    if not out:
        return out

    # 1) \dfrac / \tfrac は \frac の表示違い。値は同じ
    out = out.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # 2) ASCII の sqrt(...) / sqrt N を LaTeX へ
    out = re.sub(r"\bsqrt\s*\(\s*([^()]+?)\s*\)", r"\\sqrt{\1}", out)
    out = re.sub(r"\bsqrt\s+(\d+)\b", r"\\sqrt{\1}", out)

    # 3) 全体が「整数/整数」の素の分数なら LaTeX 分数へ
    m = re.fullmatch(r"\s*(%s)\s*/\s*(\d+)\s*" % _INT, out)
    if m:
        num, den = m.group(1), m.group(2)
        if den != "0":
            if num.startswith("-"):
                out = "-" + _frac(num[1:], den)
            else:
                out = _frac(num, den)
        return out

    # 4) 全体が有限小数なら既約分数へ（値は厳密に等しい）
    m = re.fullmatch(r"\s*(-?)(\d+)\.(\d+)\s*", out)
    if m:
        sign, ip, fp = m.group(1), m.group(2), m.group(3)
        if len(fp) <= 6:                      # 桁が長いものは触らない
            fr = Fraction(int(ip + fp), 10 ** len(fp))
            if fr.denominator != 1:
                body = _frac(str(fr.numerator), str(fr.denominator))
                out = (sign + body) if sign else body
    return out
