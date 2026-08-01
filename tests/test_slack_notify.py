"""notify_slack のオフラインテスト(ネット不要 — urlopen をフェイクに差し替える)。

既定(FUGU_SLACK_FULL 未設定)は従来どおり「1 通・先頭 500 字プレビュー」で
挙動不変、FUGU_SLACK_FULL=1 で全文を分割送信することを確認する。
"""
import json

import pytest

import fugu_local as f


@pytest.fixture()
def sent(monkeypatch):
    """Webhook への POST を捕捉する。戻り値のリストに text が積まれる。"""
    posts = []

    def fake_urlopen(req, timeout=None):
        posts.append(json.loads(req.data.decode("utf-8"))["text"])

    monkeypatch.setattr(f, "SLACK_WEBHOOK_URL", "http://hook.example/x")
    monkeypatch.setattr(f.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("FUGU_SLACK_FULL", raising=False)
    return posts


def test_no_webhook_means_no_post(sent, monkeypatch):
    monkeypatch.setattr(f, "SLACK_WEBHOOK_URL", "")
    f.notify_slack("q", "a", 1.0)
    assert sent == []


def test_default_is_single_truncated_preview(sent):
    f.notify_slack("質問です", "答" * 1000, 12.3)
    assert len(sent) == 1
    text = sent[0]
    assert ":white_check_mark:" in text and "質問です" in text
    # 500 字 + 省略記号で切れている(全文 1000 字は載らない)
    assert "答" * 500 + "…" in text
    assert "答" * 501 not in text


def test_error_answer_uses_x_icon(sent):
    f.notify_slack("q", "__ERROR__ boom", 1.0)
    assert len(sent) == 1 and sent[0].startswith(":x:")


def test_full_mode_short_answer_is_one_complete_message(sent, monkeypatch):
    monkeypatch.setenv("FUGU_SLACK_FULL", "1")
    f.notify_slack("q", "短い答え", 1.0)
    assert len(sent) == 1
    assert "短い答え" in sent[0] and "…" not in sent[0]


def test_full_mode_splits_long_answer_and_loses_nothing(sent, monkeypatch):
    monkeypatch.setenv("FUGU_SLACK_FULL", "1")
    body = "".join(f"{i % 10}" for i in range(f.SLACK_CHUNK_CHARS * 2 + 100))
    f.notify_slack("長い質問", body, 99.0)
    assert len(sent) == 3
    # ヘッダ(Q)は先頭の 1 通だけ、続きには通し番号が付く
    assert "長い質問" in sent[0] and "(1/3)" in sent[0]
    assert "長い質問" not in sent[1] and "(2/3 続き)" in sent[1]
    # 3 通の本文を繋ぐと元の全文になる(取りこぼしなし)
    rebuilt = ""
    for text in sent:
        rebuilt += text.split("\n", 2)[-1] if text.startswith(":") else \
            text.split("\n", 1)[1]
    assert body in rebuilt


def test_full_mode_caps_chunks_and_says_so(sent, monkeypatch):
    monkeypatch.setenv("FUGU_SLACK_FULL", "1")
    body = "x" * (f.SLACK_CHUNK_CHARS * (f.SLACK_MAX_CHUNKS + 2))
    f.notify_slack("q", body, 1.0)
    assert len(sent) == f.SLACK_MAX_CHUNKS
    assert "省略" in sent[-1]


def test_full_mode_stops_after_first_failure(sent, monkeypatch):
    monkeypatch.setenv("FUGU_SLACK_FULL", "1")
    calls = {"n": 0}

    def flaky_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise OSError("down")
        sent.append(json.loads(req.data.decode("utf-8"))["text"])

    monkeypatch.setattr(f.urllib.request, "urlopen", flaky_urlopen)
    f.notify_slack("q", "y" * (f.SLACK_CHUNK_CHARS * 3), 1.0)
    assert calls["n"] == 2 and len(sent) == 1   # 2 通目で失敗 → 3 通目は送らない
