"""fugu_core.memory — エピソード軌跡記憶 (Doc D Phase 1)。

サンドボックス自己デバッグ (A1) や自己進化検証 (C4) の顛末を :class:`Episode`
として永続化し、次の類似タスクで「過去の教訓」として fugu_answer に注入する。

- :class:`LexicalMemory` — stdlib のみ。JSONL 永続化 + 語彙重なり検索
  (日本語は文字 2-gram で重なりを取る)。常に利用可能なフォールバック。
- :class:`VectorMemory` — fugu_rag.vectorstore + 注入 Embedder による意味検索。
  fugu_rag 不在なら ImportError となり、:func:`get_memory` が Lexical に落ちる
  (D-2 guarded import)。store はテスト用に注入可能。
- :func:`lessons_for` — 検索結果を注入用テキストに整形。

フック(いずれも ``FUGU_MEMORY=1`` のときのみ・既定経路不変):
fugu_sandbox.run_with_self_debug / fugu_evolve.evaluator.verify が記録し、
fugu_local.fugu_answer が質問の前に教訓を付す。
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Protocol, Set

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Episode:
    """1回の試みの記録。lesson が次回への注入価値の本体。"""

    kind: str      # "sandbox" / "evolve" / "answer" など
    task: str      # 何をしようとしたか(質問・コードの要約)
    outcome: str   # "success" / "failure"
    lesson: str    # 次回に活かす教訓
    ts: str = ""   # ISO タイムスタンプ(record 時に補完)


class MemoryStore(Protocol):
    def record(self, episode: Episode) -> None: ...

    def search(self, query: str, k: int = 3) -> List[Episode]: ...


def _tokens(text: str) -> Set[str]:
    """語彙重なり用トークン。ASCII は単語、非 ASCII(日本語等)は文字 2-gram。"""
    text = (text or "").lower()
    tokens = set(_WORD.findall(text))
    for run in re.findall(r"[^\x00-\x7f]+", text):
        if len(run) == 1:
            tokens.add(run)
        tokens.update(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class LexicalMemory:
    """JSONL 永続化 + 語彙重なりスコアの stdlib-only 記憶。

    ``path=None`` ならメモリ内のみ(テスト・一時利用)。壊れた行は読み飛ばす
    (過去の書き込み事故で記憶全体を失わない)。
    """

    def __init__(self, path: Optional[str] = None,
                 now_fn: Optional[Callable[[], str]] = None):
        self.path = path
        self._now = now_fn or (lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.episodes: List[Episode] = []
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        self.episodes.append(Episode(
                            kind=str(obj["kind"]), task=str(obj["task"]),
                            outcome=str(obj["outcome"]), lesson=str(obj["lesson"]),
                            ts=str(obj.get("ts", "")),
                        ))
                    except Exception:
                        continue

    def record(self, episode: Episode) -> None:
        if not episode.ts:
            episode.ts = self._now()
        self.episodes.append(episode)
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")

    def search(self, query: str, k: int = 3) -> List[Episode]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scored = []
        for idx, ep in enumerate(self.episodes):
            score = len(query_tokens & _tokens(f"{ep.task} {ep.lesson}"))
            if score > 0:
                scored.append((score, -idx, ep))  # 同点は新しい方を先に
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [ep for _, _, ep in scored[:k]]


class VectorMemory:
    """fugu_rag.vectorstore による意味検索記憶(Embedder は必須注入)。

    保存・永続化は内部の :class:`LexicalMemory` に委譲し、ベクトル索引だけを
    重ねる。``store`` 未指定時は fugu_rag の VectorStore を guarded import する
    (不在なら ImportError → :func:`get_memory` が Lexical へフォールバック)。
    """

    def __init__(self, embedder, path: Optional[str] = None,
                 now_fn: Optional[Callable[[], str]] = None, store=None):
        if store is None:
            from fugu_rag.vectorstore import VectorStore  # guarded (D-2)
            store = VectorStore()
        self.embedder = embedder
        self.store = store
        self._inner = LexicalMemory(path=path, now_fn=now_fn)
        if self._inner.episodes:  # 既存 JSONL からベクトル索引を再構築
            vectors = self.embedder.embed(
                [self._text(ep) for ep in self._inner.episodes])
            for idx, vec in enumerate(vectors):
                self.store.add(str(idx), vec)

    @property
    def episodes(self) -> List[Episode]:
        return self._inner.episodes

    @staticmethod
    def _text(episode: Episode) -> str:
        return f"{episode.task} {episode.lesson}"

    def record(self, episode: Episode) -> None:
        self._inner.record(episode)
        vec = self.embedder.embed([self._text(episode)])[0]
        self.store.add(str(len(self._inner.episodes) - 1), vec)

    def search(self, query: str, k: int = 3) -> List[Episode]:
        if not self._inner.episodes:
            return []
        query_vec = self.embedder.embed([query])[0]
        hits = self.store.search(query_vec, k=k)
        return [self._inner.episodes[int(doc_id)]
                for doc_id, score in hits if score > 0]


def get_memory(path: Optional[str] = None, embedder=None) -> MemoryStore:
    """embedder があれば VectorMemory を試み、fugu_rag 不在なら Lexical に落ちる。"""
    if embedder is not None:
        try:
            return VectorMemory(embedder, path=path)
        except ImportError:
            pass
    return LexicalMemory(path=path)


#: プロセス共有の既定記憶(遅延生成)。テストは reset_default_memory で差し替える。
_DEFAULT: dict = {}


def get_default_memory() -> MemoryStore:
    """既定記憶。保存先は env ``FUGU_MEMORY_PATH``(既定 ``~/.fugu_memory.jsonl``)。

    embedder は使わない(常に Lexical)— 記憶注入はホットパスであり、埋め込み
    モデルの有無・GPU 状態に依存させない。意味検索が欲しい呼び出し側は
    :func:`get_memory` に embedder を渡して個別に構築する。
    """
    if "store" not in _DEFAULT:
        path = (os.environ.get("FUGU_MEMORY_PATH")
                or os.path.join(os.path.expanduser("~"), ".fugu_memory.jsonl"))
        _DEFAULT["store"] = LexicalMemory(path=path)
    return _DEFAULT["store"]


def reset_default_memory() -> None:
    """既定記憶のシングルトンを破棄する(テスト・パス切替用)。"""
    _DEFAULT.clear()


def lessons_for(memory: MemoryStore, query: str, k: int = 3) -> str:
    """query に関連する過去の教訓を注入用テキストに整形する(無関連なら空)。"""
    episodes = memory.search(query, k=k)
    if not episodes:
        return ""
    lines = ["## 過去エピソードからの教訓 (episodic memory)"]
    for ep in episodes:
        lines.append(f"- [{ep.kind}/{ep.outcome}] {ep.lesson}")
    return "\n".join(lines)
