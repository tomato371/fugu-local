# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_llm — 新規モジュール向けの Chat プロトコルと fugu_local.ask() アダプタ。

fugu-rag の fugu_rag.llm.Chat と同一シグネチャの Protocol を定義することで、
構造的部分型により fugu-rag 側の OllamaChat / FakeChat もそのまま満たせる
（ハードな相互依存なし）。新機能モジュールは chat: Chat を引数で受け取り、
内部でクライアントを構築しないこと（オフラインテスト可能性の要）。

エラー規約: fugu_local.ask() は失敗を "__ERROR__: ..." センチネル文字列で返すが、
Chat プロトコルの世界では例外 (RuntimeError) に変換する。fugu_rag の呼び出し側は
chat.complete を try/except で囲む慣習のため、センチネルを漏らすと沈黙のまま
誤テキストとして下流に流れてしまう。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union

Format = Union[dict, str, None]


class Chat(Protocol):
    def complete(self, prompt: str, *, system: Optional[str] = None,
                 fmt: Format = None, temperature: float = 0.2,
                 images: Optional[Sequence[str]] = None) -> str:
        ...


class AskChat:
    """fugu_local.ask() を Chat プロトコルに適合させるアダプタ。

    fugu_local の import は complete() 呼び出し時まで遅延する — このモジュール自体の
    import は軽量なまま保ち、テストでは fugu_local を stub した後に使える。
    model=None なら CONDUCTOR → FALLBACK_MODEL の順で解決する。
    """

    def __init__(self, model: Optional[str] = None, label: Optional[str] = None,
                 think: Optional[Any] = None, num_predict: Optional[int] = None):
        self.model = model
        self.label = label
        self.think = think
        self.num_predict = num_predict

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 fmt: Format = None, temperature: float = 0.2,
                 images: Optional[Sequence[str]] = None) -> str:
        import fugu_local

        model = self.model or fugu_local.CONDUCTOR or fugu_local.FALLBACK_MODEL
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        out = fugu_local.ask(
            model, messages, temperature,
            think=self.think, fmt=fmt, label=self.label,
            num_predict=self.num_predict, images=images,
        )
        if out.startswith("__ERROR__"):
            raise RuntimeError(out)
        return out


class FakeChat:
    """オフラインテスト用の Chat 実装（fugu_rag.llm.FakeChat と同形）。

    responses: 先頭から順に返すスクリプト済み応答。
    default:   responses が尽きた後に返す固定応答。
    fn:        prompt を受けて応答を返す任意関数（最優先）。
    calls:     受け取った引数の記録（アサーション用）。
    """

    def __init__(self, responses: Optional[List[str]] = None,
                 default: Optional[str] = None,
                 fn: Optional[Callable[[str], str]] = None):
        self.responses = list(responses or [])
        self.default = default
        self.fn = fn
        self.calls: List[Dict[str, Any]] = []

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 fmt: Format = None, temperature: float = 0.2,
                 images: Optional[Sequence[str]] = None) -> str:
        self.calls.append({"prompt": prompt, "system": system,
                           "fmt": fmt, "temperature": temperature,
                           "images": tuple(images or ())})
        if self.fn is not None:
            return self.fn(prompt)
        if self.responses:
            return self.responses.pop(0)
        if self.default is not None:
            return self.default
        raise AssertionError("FakeChat: no scripted response left")
