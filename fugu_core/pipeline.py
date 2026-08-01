# Copyright 2026 fugu-local contributors
# SPDX-License-Identifier: Apache-2.0
"""fugu_core.pipeline — VRAM 予算内の投機的並行実行 (Doc D Phase 3 / D-4)。

Conductor の計画中に RAG/Web コンテキストを先読みするなど、独立な作業を
重ねて実時間を縮める。8GB VRAM では LLM の同時実行は増やせないため、
LLM を使うタスクは :func:`run_speculative` の ``asyncio.Semaphore``
(既定 ``max_parallel_llm=1``)で直列化し、I/O タスクだけを並行させる。

- :class:`SpecTask` / :class:`SpecResult` — タスクと結果(例外はタスク単位で
  隔離: 1つの失敗が他を殺さない)。
- :func:`run_speculative` — asyncio + ``run_in_executor`` の本体(D-4)。
- :func:`speculate` — 同期ファサード。実行中ループ内から呼ばれたら決定論的に
  逐次実行へ退避する(uvicorn 等の中でも壊れない)。
- :func:`collect_ready` — 成功結果だけを {name: value} で回収。
- :class:`Prefetch` — 同期コード(fugu_answer)から conduct と重ねるための
  1タスク・スレッド先行実行。外れてもよい投機なので失敗は None に落ちる。

フックは ``FUGU_SPECULATE=1`` のときだけ fugu_local(ask_fugu)が使う。
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SpecTask:
    """投機タスク。fn は引数なしに閉じた blocking callable。"""

    name: str
    fn: Callable[[], Any]
    uses_llm: bool = False  # True なら LLM セマフォ(既定=同時1)を取る


@dataclass
class SpecResult:
    name: str
    value: Any = None
    error: Optional[BaseException] = None
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


async def run_speculative(tasks: List[SpecTask], max_parallel_llm: int = 1,
                          max_workers: int = 4) -> Dict[str, SpecResult]:
    """タスク群を並行実行し {name: SpecResult} を返す(例外はタスク単位で隔離)。

    LLM タスクは Semaphore(max_parallel_llm) で絞る(8GB VRAM 既定=1)。
    純 I/O タスクは executor の ``max_workers`` まで自由に並行する。
    """
    loop = asyncio.get_running_loop()
    llm_sem = asyncio.Semaphore(max(1, max_parallel_llm))
    results: Dict[str, SpecResult] = {}

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:

        async def _one(task: SpecTask) -> None:
            start = time.perf_counter()
            try:
                if task.uses_llm:
                    async with llm_sem:
                        value = await loop.run_in_executor(pool, task.fn)
                else:
                    value = await loop.run_in_executor(pool, task.fn)
                results[task.name] = SpecResult(
                    name=task.name, value=value,
                    elapsed=time.perf_counter() - start)
            except Exception as exc:
                results[task.name] = SpecResult(
                    name=task.name, error=exc,
                    elapsed=time.perf_counter() - start)

        await asyncio.gather(*[_one(task) for task in tasks])
    return results


def speculate(tasks: List[SpecTask], max_parallel_llm: int = 1,
              max_workers: int = 4) -> Dict[str, SpecResult]:
    """同期ファサード。既に実行中のイベントループ内なら決定論的に逐次実行する。"""
    try:
        return asyncio.run(run_speculative(
            tasks, max_parallel_llm=max_parallel_llm, max_workers=max_workers))
    except RuntimeError:
        results: Dict[str, SpecResult] = {}
        for task in tasks:
            start = time.perf_counter()
            try:
                results[task.name] = SpecResult(
                    name=task.name, value=task.fn(),
                    elapsed=time.perf_counter() - start)
            except Exception as exc:
                results[task.name] = SpecResult(
                    name=task.name, error=exc,
                    elapsed=time.perf_counter() - start)
        return results


def collect_ready(results: Dict[str, SpecResult],
                  names: Optional[List[str]] = None) -> Dict[str, Any]:
    """成功した結果だけを {name: value} で返す(``names`` で絞り込み可)。"""
    return {r.name: r.value for r in results.values()
            if r.ok and (names is None or r.name in names)}


class Prefetch:
    """1タスクのスレッド先行実行 — 同期コードから conduct と重ねる最小形。

    投機なので「外れ」は許容: 実行が失敗した/間に合わなかった場合、
    :meth:`result` は None を返し、呼び出し側は従来の同期経路に落ちる。
    値としての None を返しうる fn には使わないこと(区別できない)。
    """

    def __init__(self, fn: Callable[[], Any]):
        self._value: Any = None
        self._error: Optional[BaseException] = None
        self._done = threading.Event()
        thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        thread.start()

    def _run(self, fn: Callable[[], Any]) -> None:
        try:
            self._value = fn()
        except BaseException as exc:  # 投機失敗はどんな例外でも「外れ」扱い
            self._error = exc
        finally:
            self._done.set()

    def result(self, timeout: Optional[float] = None) -> Any:
        """完了を待って値を返す。失敗・タイムアウトは None(=投機の外れ)。"""
        if not self._done.wait(timeout):
            return None
        if self._error is not None:
            return None
        return self._value
