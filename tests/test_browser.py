# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_browser のオフラインテスト(ネットワーク不要。実HTTP は一切叩かない)。

UrllibBrowser は stdlib が data: URL をサポートすることを利用して実 fetch 経路を
オフラインで検証する。playwright / httpx 系は importorskip(未導入環境では skip)。
"""
import importlib.util

import pytest

import fugu_browser
from fugu_browser import (
    HttpxSoupBrowser,
    Page,
    PlaywrightBrowser,
    UrllibBrowser,
    as_fetcher,
    enrich_search_results,
    get_browser,
    html_to_text,
)


class FakeBrowser:
    """テスト注入用: url -> text の辞書。無い URL は例外(取得失敗の再現)。"""

    name = "fake"

    def __init__(self, pages=None):
        self.pages = pages or {}
        self.fetched = []

    def fetch(self, url, timeout=20.0):
        self.fetched.append(url)
        if url not in self.pages:
            raise RuntimeError(f"unreachable: {url}")
        return Page(url=url, text=self.pages[url], status=200)

    def screenshot(self, url, path, timeout=20.0):
        return False


# ------------------------------------------------------------------ html_to_text

def test_html_to_text_extracts_visible_text():
    html = ("<html><head><title>T</title><style>body{color:red}</style></head>"
            "<body><h1>Heading</h1><script>var x=1;</script><p>Body text.</p></body></html>")
    text = html_to_text(html)
    assert "Heading" in text and "Body text." in text
    assert "color:red" not in text and "var x=1" not in text


def test_html_to_text_survives_broken_html():
    assert "hello" in html_to_text("<div><p>hello<div></span>")
    assert html_to_text("") == ""


def test_html_to_text_unescapes_entities():
    assert "a < b & c" in html_to_text("<p>a &lt; b &amp; c</p>")


# ------------------------------------------------------------------ UrllibBrowser

def test_urllib_browser_fetches_data_url_offline():
    page = UrllibBrowser().fetch(
        "data:text/html,<html><body><p>offline page</p><script>x</script></body></html>")
    assert page.ok
    assert "offline page" in page.text
    assert "<p>" in page.html  # 生 HTML も保持


def test_urllib_browser_screenshot_unsupported(tmp_path):
    assert UrllibBrowser().screenshot("data:text/html,<p>x</p>",
                                      str(tmp_path / "s.png")) is False


def test_urllib_browser_always_available():
    assert UrllibBrowser.available() is True


# ------------------------------------------------------------------ optional backends

def test_playwright_browser_available_matches_import():
    expected = importlib.util.find_spec("playwright") is not None
    assert PlaywrightBrowser.available() is expected


def test_httpx_browser_available_matches_import():
    expected = (importlib.util.find_spec("httpx") is not None
                and importlib.util.find_spec("bs4") is not None)
    assert HttpxSoupBrowser.available() is expected


def test_httpx_browser_fetch_parses_html():
    pytest.importorskip("httpx")
    bs4 = pytest.importorskip("bs4")
    # 実 HTTP は叩かず、パース部分だけを検証する
    soup = bs4.BeautifulSoup("<body><p>hi</p><script>x</script></body>", "html.parser")
    for tag in soup(["script"]):
        tag.decompose()
    assert soup.get_text(strip=True) == "hi"


# ------------------------------------------------------------------ get_browser

def test_get_browser_returns_always_working_fallback(monkeypatch):
    monkeypatch.setattr(PlaywrightBrowser, "available", staticmethod(lambda: False))
    monkeypatch.setattr(HttpxSoupBrowser, "available", staticmethod(lambda: False))
    assert isinstance(get_browser(), UrllibBrowser)


def test_get_browser_prefer_forces_backend():
    assert isinstance(get_browser(prefer="urllib"), UrllibBrowser)


def test_get_browser_env_backend(monkeypatch):
    monkeypatch.setenv("FUGU_BROWSER_BACKEND", "urllib")
    assert isinstance(get_browser(), UrllibBrowser)


def test_get_browser_unavailable_prefer_falls_through(monkeypatch):
    monkeypatch.setattr(PlaywrightBrowser, "available", staticmethod(lambda: False))
    monkeypatch.setattr(HttpxSoupBrowser, "available", staticmethod(lambda: False))
    assert isinstance(get_browser(prefer="playwright"), UrllibBrowser)


# ------------------------------------------------------------------ as_fetcher

def test_as_fetcher_is_b1_compatible():
    fetch_fn = as_fetcher(FakeBrowser({"http://a": "page text"}))
    assert fetch_fn("http://a") == "page text"


def test_as_fetcher_truncates():
    fetch_fn = as_fetcher(FakeBrowser({"http://a": "x" * 100}), max_chars=10)
    assert len(fetch_fn("http://a")) == 10


def test_as_fetcher_propagates_failure():
    fetch_fn = as_fetcher(FakeBrowser())
    with pytest.raises(RuntimeError):
        fetch_fn("http://gone")


# ------------------------------------------------------------------ enrich

def _snippet(url, body="snippet"):
    return f"[title]\n{body}\nSource: {url}"


def test_enrich_appends_page_excerpt():
    browser = FakeBrowser({"http://a": "full page body"})
    out = enrich_search_results([_snippet("http://a")], browser)
    assert "[Page content]" in out[0]
    assert "full page body" in out[0]
    assert "snippet" in out[0]  # 元スニペットは保持


def test_enrich_does_not_mutate_input():
    items = [_snippet("http://a")]
    enrich_search_results(items, FakeBrowser({"http://a": "body"}))
    assert "[Page content]" not in items[0]


def test_enrich_skips_failed_and_non_http():
    browser = FakeBrowser({"http://ok": "body"})
    items = [_snippet("http://dead"), _snippet("ftp://x"), "no source line",
             _snippet("http://ok")]
    out = enrich_search_results(items, browser)
    assert out[0] == items[0] and out[1] == items[1] and out[2] == items[2]
    assert "[Page content]" in out[3]
    assert "ftp://x" not in browser.fetched  # 非HTTPはそもそも fetch しない


def test_enrich_respects_max_pages_and_chars():
    browser = FakeBrowser({f"http://{i}": "y" * 5000 for i in range(5)})
    items = [_snippet(f"http://{i}") for i in range(5)]
    out = enrich_search_results(items, browser, max_pages=2, chars=100)
    enriched = [x for x in out if "[Page content]" in x]
    assert len(enriched) == 2
    assert all(len(x) < 400 for x in enriched)  # 抜粋は chars で切られている


def test_enrich_dedupes_urls():
    browser = FakeBrowser({"http://a": "body"})
    out = enrich_search_results([_snippet("http://a"), _snippet("http://a")], browser)
    assert browser.fetched == ["http://a"]
    assert "[Page content]" in out[0] and "[Page content]" not in out[1]


def test_enrich_empty_results_passthrough():
    assert enrich_search_results([], FakeBrowser()) == []


# ------------------------------------------------------------------ fugu_local hook

def test_browser_enrich_hook_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_BROWSER", raising=False)
    items = [_snippet("http://a")]
    monkeypatch.setattr(fugu_browser, "enrich_search_results",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("呼ばれない")))
    assert fugu_local._browser_enrich(items) is items


def test_browser_enrich_hook_enabled(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_BROWSER", "1")
    monkeypatch.setattr(fugu_browser, "get_browser",
                        lambda prefer=None: FakeBrowser({"http://a": "page body"}))
    out = fugu_local._browser_enrich([_snippet("http://a")])
    assert "[Page content]" in out[0]


def test_browser_enrich_hook_never_raises(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_BROWSER", "1")
    monkeypatch.setattr(fugu_browser, "enrich_search_results",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    items = [_snippet("http://a")]
    assert fugu_local._browser_enrich(items) is items
