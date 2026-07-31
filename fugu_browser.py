"""fugu_browser — 差し替え可能なブラウザ自動化と Web ページ取得 (Doc B Phase 3)。

検索スニペットだけでは足りない「ページ本文」を取得する層。3段フォールバック:

1. :class:`PlaywrightBrowser` — JS レンダリング + スクリーンショット(要 playwright)。
2. :class:`HttpxSoupBrowser` — 静的 HTML の高品質テキスト抽出(要 httpx + bs4)。
3. :class:`UrllibBrowser` — stdlib のみ。常に利用可能な最終フォールバック。

:func:`get_browser` が導入済みの最上位実装を選ぶ。:func:`as_fetcher` は
fugu_rag.research (Doc B1) の ``fetch_fn`` 互換 callable を返し、
:func:`enrich_search_results` は fugu_local の検索結果(``Source: URL`` 行付き
スニペット)へページ本文の抜粋を追記する。フックは env フラグ ``FUGU_BROWSER=1``
のときだけ fugu_local から呼ばれる(既定経路は不変)。

オフラインテスト規約: Browser は Protocol。テストは FakeBrowser を注入し、
UrllibBrowser 自体は ``data:`` URL でネットワーク無しに検証する。
"""
from __future__ import annotations

import importlib.util
import os
import re
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, List, Optional, Protocol

DEFAULT_TIMEOUT = 20.0
USER_AGENT = "fugu-local/1.0 (local research; +https://github.com/tomato371)"

#: enrich が1回で取りに行くページ数と、1ページから注入する最大文字数。
#: num_ctx 8192 pin (gotcha #2) を検索コンテキストが圧迫しないための予算。
ENRICH_MAX_PAGES = 3
ENRICH_CHARS = 1200


@dataclass
class Page:
    """取得結果。text は本文テキスト、html は生 HTML(取れた場合のみ)。"""

    url: str
    text: str
    html: str = ""
    status: int = 0

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 400 and bool(self.text.strip())


class Browser(Protocol):
    """fetch はページを返すか例外を送出する。screenshot は成功時 True。"""

    def fetch(self, url: str, timeout: float = DEFAULT_TIMEOUT) -> Page:
        ...

    def screenshot(self, url: str, path: str, timeout: float = DEFAULT_TIMEOUT) -> bool:
        ...


# ------------------------------------------------------------------ HTML→テキスト

class _TextExtractor(HTMLParser):
    """script/style 等を除いた可視テキストを集める stdlib-only 抽出器。"""

    _SKIP = frozenset({"script", "style", "noscript", "template", "head"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def html_to_text(html_text: str) -> str:
    """HTML から可視テキストを抽出する(壊れた HTML でも例外を出さない)。"""
    parser = _TextExtractor()
    try:
        parser.feed(html_text or "")
        parser.close()
    except Exception:
        pass  # HTMLParser が稀に投げる壊れ入力はそこまでの parts で返す
    return "\n".join(parser.parts)


# ------------------------------------------------------------------ 実装3段

class PlaywrightBrowser:
    """headless Chromium で JS レンダリング済みページを取得(要 playwright)。"""

    name = "playwright"

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("playwright") is not None

    def fetch(self, url: str, timeout: float = DEFAULT_TIMEOUT) -> Page:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=USER_AGENT)
                resp = page.goto(url, timeout=timeout * 1000)
                html_text = page.content()
                text = page.inner_text("body")
                status = resp.status if resp else 200
            finally:
                browser.close()
        return Page(url=url, text=text, html=html_text, status=status)

    def screenshot(self, url: str, path: str, timeout: float = DEFAULT_TIMEOUT) -> bool:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=USER_AGENT)
                page.goto(url, timeout=timeout * 1000)
                page.screenshot(path=path, full_page=True)
            finally:
                browser.close()
        return True


class HttpxSoupBrowser:
    """httpx + BeautifulSoup による静的取得(JS なし・スクリーンショット不可)。"""

    name = "httpx"

    @staticmethod
    def available() -> bool:
        return (importlib.util.find_spec("httpx") is not None
                and importlib.util.find_spec("bs4") is not None)

    def fetch(self, url: str, timeout: float = DEFAULT_TIMEOUT) -> Page:
        import httpx
        from bs4 import BeautifulSoup

        resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": USER_AGENT})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        return Page(url=url, text=text, html=resp.text, status=resp.status_code)

    def screenshot(self, url: str, path: str, timeout: float = DEFAULT_TIMEOUT) -> bool:
        return False


class UrllibBrowser:
    """stdlib のみの最終フォールバック(常に利用可能・JS なし)。"""

    name = "urllib"

    @staticmethod
    def available() -> bool:
        return True

    def fetch(self, url: str, timeout: float = DEFAULT_TIMEOUT) -> Page:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read()
            headers = getattr(resp, "headers", None)
            charset = headers.get_content_charset() if headers is not None else None
            status = int(getattr(resp, "status", 200) or 200)
        html_text = raw.decode(charset or "utf-8", errors="replace")
        return Page(url=url, text=html_to_text(html_text), html=html_text, status=status)

    def screenshot(self, url: str, path: str, timeout: float = DEFAULT_TIMEOUT) -> bool:
        return False


#: フォールバック順(上位ほど高機能)。
_CHAIN = (PlaywrightBrowser, HttpxSoupBrowser, UrllibBrowser)


def get_browser(prefer: Optional[str] = None) -> Browser:
    """導入済みの最上位実装を返す。

    ``prefer``(または env ``FUGU_BROWSER_BACKEND``)で "playwright" / "httpx" /
    "urllib" を指名できる。指名先が未導入ならフォールバック順で次に進むため、
    常に必ず何かしらの Browser が返る(UrllibBrowser は常時利用可)。
    """
    prefer = prefer or os.environ.get("FUGU_BROWSER_BACKEND") or ""
    chain = sorted(_CHAIN, key=lambda cls: cls.name != prefer) if prefer else list(_CHAIN)
    for cls in chain:
        if cls.available():
            return cls()
    return UrllibBrowser()


def as_fetcher(browser: Optional[Browser] = None,
               max_chars: Optional[int] = None) -> Callable[[str], str]:
    """fugu_rag.research (B1) の ``fetch_fn`` 互換 callable を返す。

    失敗は例外のまま伝播する(B1 側の _gather が握って snippet に留める契約)。
    browser 未指定なら初回呼び出し時に :func:`get_browser` で解決する。
    """
    state = {"browser": browser}

    def fetch_fn(url: str) -> str:
        if state["browser"] is None:
            state["browser"] = get_browser()
        text = state["browser"].fetch(url).text
        return text[:max_chars] if max_chars else text

    return fetch_fn


# ------------------------------------------------------------------ 検索結果の増強

_SOURCE_RE = re.compile(r"Source: (\S+)")


def enrich_search_results(results: List[str], browser: Optional[Browser] = None,
                          max_pages: int = ENRICH_MAX_PAGES,
                          chars: int = ENRICH_CHARS) -> List[str]:
    """検索スニペットの ``Source: URL`` 先の本文抜粋を追記した新リストを返す。

    fugu_local の web_search / research_search から FUGU_BROWSER=1 のときだけ
    呼ばれる。元リストは変更しない。取得失敗・非HTTP URL・重複 URL・空本文は
    黙って読み飛ばす(検索コンテキストを絶対に壊さない)。max_pages 件まで・
    1ページ chars 文字まで(num_ctx 予算)。
    """
    if not results:
        return results
    if browser is None:
        browser = get_browser()
    out = list(results)
    fetched = 0
    seen = set()
    for i, item in enumerate(results):
        if fetched >= max_pages:
            break
        if not isinstance(item, str):
            continue
        m = _SOURCE_RE.search(item)
        if not m:
            continue
        url = m.group(1)
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        try:
            page = browser.fetch(url)
        except Exception:
            continue
        excerpt = page.text.strip()[:chars]
        if not excerpt:
            continue
        out[i] = f"{item}\n[Page content]\n{excerpt}"
        fetched += 1
    return out
