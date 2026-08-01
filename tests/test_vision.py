# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""Vision 入力（images パススルー・直行ルーティング）のオフラインテスト。"""
from __future__ import annotations

import base64
import json

import pytest

import fugu_local


@pytest.fixture()
def fake_urlopen(monkeypatch):
    """urllib.request.urlopen を差し替えて /api/chat ペイロードを記録する。"""
    captured = {}

    class _Resp:
        status = 200

        def read(self):
            return json.dumps({"message": {"content": "a red panda"}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr(fugu_local.urllib.request, "urlopen", _fake)
    return captured


def test_ask_attaches_images_to_last_message(fake_urlopen):
    out = fugu_local.ask(
        "vision-model",
        [{"role": "system", "content": "s"}, {"role": "user", "content": "what?"}],
        0.2, images=["QUJD"])
    assert out == "a red panda"
    messages = fake_urlopen["payload"]["messages"]
    assert messages[-1]["images"] == ["QUJD"]
    assert "images" not in messages[0]
    # num_ctx は常に明示 pin（gotcha #2 の不変条件は vision でも維持）
    assert "num_ctx" in fake_urlopen["payload"]["options"]


def test_ask_does_not_mutate_caller_messages(fake_urlopen):
    messages = [{"role": "user", "content": "q"}]
    fugu_local.ask("m", messages, 0.2, images=["QUJD"])
    assert "images" not in messages[0]  # 呼び出し元のリストは不変


def test_ask_no_images_key_when_absent(fake_urlopen):
    fugu_local.ask("m", [{"role": "user", "content": "q"}], 0.2)
    assert "images" not in fake_urlopen["payload"]["messages"][-1]


def test_encode_image_file_roundtrip(tmp_path):
    raw = b"\x89PNG fake"
    p = tmp_path / "x.png"
    p.write_bytes(raw)
    assert base64.b64decode(fugu_local._encode_image_file(str(p))) == raw


def test_vision_answer_routes_to_vision_model(tmp_path, monkeypatch):
    p = tmp_path / "cat.png"
    p.write_bytes(b"imgbytes")
    recorded = {}

    def fake_ask(model, messages, temperature, **kw):
        recorded["model"] = model
        recorded["images"] = kw.get("images")
        recorded["question"] = messages[-1]["content"]
        return "a cat"

    monkeypatch.setattr(fugu_local, "ask", fake_ask)
    out = fugu_local._vision_answer("what is this?", [str(p)])
    assert out == "a cat"
    assert recorded["model"] == fugu_local.VISION_MODEL
    assert recorded["images"] == [base64.b64encode(b"imgbytes").decode("ascii")]
    assert recorded["question"] == "what is this?"


def test_vision_answer_missing_file_returns_error_sentinel():
    out = fugu_local._vision_answer("q", ["Z:/no/such/image.png"])
    assert out.startswith("__ERROR__")


def test_ask_fugu_images_bypasses_moa(monkeypatch):
    monkeypatch.setattr(fugu_local, "setup", lambda: True)
    monkeypatch.setattr(fugu_local, "_vision_answer",
                        lambda q, imgs: "vision says hi")
    conduct_called = {"n": 0}

    def spy_conduct(*a, **k):
        conduct_called["n"] += 1
        return {}, ""

    monkeypatch.setattr(fugu_local, "conduct", spy_conduct)
    result = fugu_local.ask_fugu("describe", images=["a.png"])
    assert result == "vision says hi"
    assert conduct_called["n"] == 0  # Conductor/MoA はバイパスされる


def test_askchat_passes_images(monkeypatch):
    import fugu_llm
    seen = {}

    def fake_ask(model, messages, temperature, **kw):
        seen["images"] = kw.get("images")
        return "ok"

    monkeypatch.setattr(fugu_local, "ask", fake_ask)
    fugu_llm.AskChat(model="m").complete("q", images=["QUJD"])
    assert seen["images"] == ["QUJD"]
