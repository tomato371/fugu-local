"""fugu_artifacts のオフラインテスト(Gradio import なし・純ロジックのみ)。"""
from fugu_artifacts import (
    EMPTY_PREVIEW,
    Artifact,
    build_canvas,
    diff_versions,
    extract_artifacts,
    render_preview_html,
    suggest_filename,
)

HTML_ANSWER = (
    "こちらがページです:\n\n```html\n<!doctype html>\n<html><body><h1>Hi</h1>"
    "</body></html>\n```\n以上です。"
)


# ------------------------------------------------------------------ extract

def test_extract_fenced_html():
    arts = extract_artifacts(HTML_ANSWER)
    assert len(arts) == 1
    assert arts[0].kind == "html"
    assert "<h1>Hi</h1>" in arts[0].code


def test_extract_multiple_blocks_in_order():
    text = "```python\nprint(1)\n```\ntext\n```js\nalert(1)\n```"
    arts = extract_artifacts(text)
    assert [a.kind for a in arts] == ["python", "javascript"]


def test_extract_svg_by_content_without_lang_tag():
    arts = extract_artifacts("```\n<svg viewBox='0 0 1 1'></svg>\n```")
    assert arts[0].kind == "svg"


def test_extract_bare_html_without_fence():
    arts = extract_artifacts("Here you go: <html><body>x</body></html> enjoy")
    assert len(arts) == 1
    assert arts[0].kind == "html"
    assert arts[0].code.startswith("<html>")


def test_extract_bare_svg_without_fence():
    arts = extract_artifacts("figure: <svg width='1'><rect/></svg> done")
    assert arts[0].kind == "svg"


def test_extract_prefers_fences_over_bare():
    text = "```python\nx = 1\n```\nand <svg></svg>"  # 裸SVGは閉じタグ付きだが fence 優先
    arts = extract_artifacts(text + "</svg>")
    assert [a.kind for a in arts] == ["python"]


def test_extract_nothing():
    assert extract_artifacts("plain prose only") == []
    assert extract_artifacts("") == []


def test_extract_skips_empty_blocks():
    assert extract_artifacts("```python\n\n```") == []


def test_extract_markdown_and_unknown_lang():
    arts = extract_artifacts("```md\n# t\n```\n```rust\nfn main(){}\n```")
    assert [a.kind for a in arts] == ["markdown", "rust"]


# ------------------------------------------------------------------ preview

def test_preview_html_uses_sandboxed_iframe_srcdoc():
    out = render_preview_html(Artifact(kind="html", code="<h1>x</h1>"))
    assert out.startswith("<iframe sandbox=")
    assert "srcdoc=" in out
    assert "&lt;h1&gt;" in out  # srcdoc 内は escape 済み


def test_preview_svg_wrapped_in_html():
    out = render_preview_html(Artifact(kind="svg", code="<svg></svg>"))
    assert "<iframe" in out
    assert "&lt;svg&gt;" in out


def test_preview_code_falls_back_to_escaped_pre():
    out = render_preview_html(Artifact(kind="python", code="print('<x>')"))
    assert out.startswith("<pre")
    assert "&lt;x&gt;" in out
    assert "<iframe" not in out


# ------------------------------------------------------------------ diff

def test_diff_versions_reports_changes():
    diff = diff_versions("a\nb\n", "a\nc\n")
    assert "-b" in diff and "+c" in diff
    assert "--- previous" in diff and "+++ current" in diff


def test_diff_versions_empty_when_identical():
    assert diff_versions("same\n", "same\n") == ""


# ------------------------------------------------------------------ filename

def test_suggest_filename_by_kind():
    assert suggest_filename(Artifact(kind="html", code="x")) == "artifact-1.html"
    assert suggest_filename(Artifact(kind="python", code="x"), index=2) == "artifact-2.py"
    assert suggest_filename(Artifact(kind="rust", code="x")) == "artifact-1.txt"


# ------------------------------------------------------------------ build_canvas

def test_build_canvas_without_artifact():
    view = build_canvas("plain prose")
    assert view["has_artifact"] is False
    assert view["preview_html"] == EMPTY_PREVIEW
    assert view["code"] == "" and view["diff"] == "" and view["filename"] == ""


def test_build_canvas_with_html_answer():
    view = build_canvas(HTML_ANSWER)
    assert view["has_artifact"] is True
    assert "<iframe" in str(view["preview_html"])
    assert view["kind"] == "html"
    assert view["filename"] == "artifact-1.html"


def test_build_canvas_prefers_previewable_artifact():
    text = "```python\nprint(1)\n```\n```html\n<html><body>x</body></html>\n```"
    view = build_canvas(text)
    assert view["kind"] == "html"  # 2番目でも html/svg を主要として選ぶ


def test_build_canvas_diff_against_previous_version():
    view = build_canvas("```python\nprint(2)\n```", prev_code="print(1)\n")
    assert "-print(1)" in str(view["diff"])
    assert "+print(2)" in str(view["diff"])


def test_build_canvas_no_diff_without_previous():
    assert build_canvas("```python\nx=1\n```")["diff"] == ""
