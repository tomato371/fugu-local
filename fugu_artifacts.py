# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_artifacts — Canvas/Artifacts 用の純ロジック (Doc B Phase 4)。

回答テキストから「アーティファクト」(HTML/SVG/コードブロック等の成果物)を検出し、
右ペイン Canvas 用の Live Preview HTML・バージョン間 diff・エクスポート名を
組み立てる。**Gradio には一切依存しない**(Gradio バージョンドリフト対策:
ロジックはここに隔離し、fugu_web.py は安定コンポーネントの配線だけを持つ。
テストも Gradio import なしで全て回る)。

- :func:`extract_artifacts` — fenced code block と裸の HTML/SVG を検出。
- :func:`render_preview_html` — iframe ``srcdoc``(サンドボックス付き)の
  プレビュー HTML を生成。非レンダラブル種別は escape 済み ``<pre>`` に落とす。
- :func:`diff_versions` — difflib unified diff。
- :func:`build_canvas` — fugu_web が1回呼ぶだけで済む集約関数。
"""
from __future__ import annotations

import difflib
import html as _html
import re
from dataclasses import dataclass
from typing import Dict, List

#: fenced code block(言語タグは任意)。
_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
#: fence が無い回答向け: 裸の HTML 文書 / SVG。
_BARE_HTML_RE = re.compile(r"<(?:!doctype\s+html|html)\b.*?</html\s*>",
                           re.IGNORECASE | re.DOTALL)
_BARE_SVG_RE = re.compile(r"<svg\b.*?</svg\s*>", re.IGNORECASE | re.DOTALL)

#: プレビュー無し時に Canvas に出すプレースホルダ。
EMPTY_PREVIEW = (
    "<div style='color:#888;padding:1.5em;font-family:sans-serif'>"
    "アーティファクトはまだありません</div>"
)

_EXT: Dict[str, str] = {
    "html": ".html", "svg": ".svg", "python": ".py",
    "javascript": ".js", "markdown": ".md",
}


@dataclass
class Artifact:
    kind: str   # "html" / "svg" / "python" / "javascript" / "markdown" / その他
    code: str
    lang: str = ""  # fence の生言語タグ(検出補助・表示用)


def _detect_kind(lang: str, code: str) -> str:
    lang = (lang or "").lower()
    body = code.lstrip().lower()
    if lang in ("html", "htm") or body.startswith(("<!doctype", "<html")):
        return "html"
    if lang == "svg" or body.startswith("<svg"):
        return "svg"
    if lang in ("python", "py"):
        return "python"
    if lang in ("javascript", "js", "jsx", "typescript", "ts", "tsx"):
        return "javascript"
    if lang in ("markdown", "md"):
        return "markdown"
    return lang or "code"


def extract_artifacts(text: str) -> List[Artifact]:
    """回答テキストからアーティファクトを検出する。

    fenced code block があればそれら(出現順)。無ければ裸の HTML 文書 / SVG を
    探す(モデルが fence 無しで丸ごと HTML を返すケースの救済)。何も無ければ空。
    """
    artifacts: List[Artifact] = []
    for m in _FENCE_RE.finditer(text or ""):
        lang, code = m.group(1), m.group(2).strip()
        if code:
            artifacts.append(Artifact(kind=_detect_kind(lang, code), code=code, lang=lang))
    if artifacts:
        return artifacts
    for pattern, kind in ((_BARE_HTML_RE, "html"), (_BARE_SVG_RE, "svg")):
        m = pattern.search(text or "")
        if m:
            return [Artifact(kind=kind, code=m.group(0).strip())]
    return []


def render_preview_html(artifact: Artifact, height: int = 480) -> str:
    """アーティファクトの Live Preview HTML を返す。

    html/svg は sandbox 付き iframe の ``srcdoc`` に escape して埋め込む
    (親ページの DOM/セッションに触れない)。それ以外は escape 済み ``<pre>``。
    """
    if artifact.kind in ("html", "svg"):
        doc = artifact.code if artifact.kind == "html" else (
            f"<html><body style='margin:0'>{artifact.code}</body></html>")
        return (
            f"<iframe sandbox=\"allow-scripts\" srcdoc=\"{_html.escape(doc, quote=True)}\" "
            f"style=\"width:100%;height:{height}px;border:1px solid #ddd;"
            f"border-radius:6px;background:#fff\"></iframe>"
        )
    return (
        f"<pre style='max-height:{height}px;overflow:auto;padding:1em'>"
        f"<code>{_html.escape(artifact.code)}</code></pre>"
    )


def diff_versions(old: str, new: str, fromfile: str = "previous",
                  tofile: str = "current") -> str:
    """2バージョン間の unified diff(差分なしなら空文字列)。"""
    lines = difflib.unified_diff(
        (old or "").splitlines(keepends=True),
        (new or "").splitlines(keepends=True),
        fromfile=fromfile, tofile=tofile,
    )
    return "".join(lines)


def suggest_filename(artifact: Artifact, index: int = 1) -> str:
    """エクスポート用ファイル名(kind から拡張子を決める)。"""
    return f"artifact-{index}{_EXT.get(artifact.kind, '.txt')}"


def build_canvas(answer_text: str, prev_code: str = "") -> Dict[str, object]:
    """回答テキストから Canvas 表示一式を組み立てる(fugu_web が呼ぶ集約点)。

    主要アーティファクトはプレビュー可能な html/svg を優先し、無ければ先頭の
    ブロック。``prev_code``(前回の主要アーティファクト)があれば diff も返す。

    Returns:
        dict: has_artifact / preview_html / code / kind / diff / filename
    """
    artifacts = extract_artifacts(answer_text or "")
    if not artifacts:
        return {"has_artifact": False, "preview_html": EMPTY_PREVIEW,
                "code": "", "kind": "", "diff": "", "filename": ""}
    primary = next((a for a in artifacts if a.kind in ("html", "svg")), artifacts[0])
    return {
        "has_artifact": True,
        "preview_html": render_preview_html(primary),
        "code": primary.code,
        "kind": primary.kind,
        "diff": diff_versions(prev_code, primary.code) if prev_code else "",
        "filename": suggest_filename(primary),
    }
