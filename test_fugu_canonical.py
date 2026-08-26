"""fugu_canonical.canonicalize_math_answer のテスト（2026-08-26 追加）。

実データ（math500 の厳密採点で落ちた18問）から起こしたケースを含む。
"""
import pytest

from fugu_canonical import canonicalize_math_answer as C


@pytest.mark.parametrize("src,want", [
    # --- 実データで救済できたもの ---
    ("-35/9",                 "-\\frac{35}{9}"),
    ("448/15625",             "\\frac{448}{15625}"),
    ("3840/289",              "\\frac{3840}{289}"),
    ("-24/25",                "-\\frac{24}{25}"),
    ("\\dfrac{\\sqrt{21}}{5}", "\\frac{\\sqrt{21}}{5}"),
    ("-\\dfrac{\\pi}{6}",     "-\\frac{\\pi}{6}"),
    ("-sqrt(3)",              "-\\sqrt{3}"),
    ("0.09",                  "\\frac{9}{100}"),
    # --- 値を変えないこと ---
    ("\\frac{1}{2}",          "\\frac{1}{2}"),
    ("902",                   "902"),
    ("-13x + 3",              "-13x + 3"),     # 空白は触らない（参照側が不統一）
    ("30",                    "30"),           # 度記号は付けない（問題文が要る）
    ("C",                     "C"),            # MCQ 化しない
    ("11111111100",           "11111111100"),  # 桁区切りにしない
    ("288\\pi",               "288\\pi"),
    ("[-2,7]",                "[-2,7]"),
    # --- 変換の周辺 ---
    ("\\tfrac{3}{4}",         "\\frac{3}{4}"),
    ("sqrt(2)",               "\\sqrt{2}"),
    ("sqrt 5",                "\\sqrt{5}"),
    ("2.5",                   "\\frac{5}{2}"),
    ("1.0",                   "1.0"),          # 分母1になるものは触らない
    ("0/5",                   "\\frac{0}{5}"),
])
def test_cases(src, want):
    assert C(src) == want


def test_none_and_empty():
    assert C(None) is None
    assert C("") == ""
    assert C("   ") == ""


def test_zero_division_is_left_alone():
    assert C("3/0") == "3/0"


def test_long_decimal_is_left_alone():
    assert C("0.1234567") == "0.1234567"


def test_idempotent():
    for s in ["-35/9", "\\dfrac{1}{2}", "0.09", "-sqrt(3)", "902"]:
        once = C(s)
        assert C(once) == once, s


def test_never_reads_expected():
    """引数は1つだけ。期待値を渡す口が無いことをシグネチャで保証する。"""
    import inspect
    sig = inspect.signature(C)
    assert len(sig.parameters) == 1
