"""fugu_core.pipeline のオフラインテスト(即時フェイクで順序/例外分離/並列度)。"""
import asyncio
import threading

from fugu_core.pipeline import (
    Prefetch,
    SpecResult,
    SpecTask,
    collect_ready,
    speculate,
)


# ------------------------------------------------------------------ speculate

def test_speculate_returns_all_results_by_name():
    tasks = [SpecTask("a", lambda: 1), SpecTask("b", lambda: "two")]
    results = speculate(tasks)
    assert set(results) == {"a", "b"}
    assert results["a"].value == 1 and results["b"].value == "two"
    assert all(r.ok for r in results.values())


def test_speculate_isolates_exceptions():
    def boom():
        raise ValueError("task down")

    results = speculate([SpecTask("bad", boom), SpecTask("good", lambda: 42)])
    assert results["good"].ok and results["good"].value == 42
    assert not results["bad"].ok
    assert isinstance(results["bad"].error, ValueError)


def test_llm_semaphore_serializes_llm_tasks():
    lock = threading.Lock()
    state = {"current": 0, "max": 0}

    def llm_task():
        with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        # わずかに保持して重なりの機会を作る(セマフォが無ければ max>1 になりうる)
        threading.Event().wait(0.02)
        with lock:
            state["current"] -= 1
        return "ok"

    tasks = [SpecTask(f"llm{i}", llm_task, uses_llm=True) for i in range(4)]
    results = speculate(tasks, max_parallel_llm=1, max_workers=4)
    assert all(r.ok for r in results.values())
    assert state["max"] == 1  # Semaphore(1) の保証 — 決定論的に同時1


def test_io_tasks_run_concurrently():
    # 3タスク全員がバリアに到達しないと進めない = 並行実行でなければタイムアウト
    barrier = threading.Barrier(3, timeout=5)

    def io_task():
        barrier.wait()
        return "passed"

    tasks = [SpecTask(f"io{i}", io_task) for i in range(3)]
    results = speculate(tasks, max_workers=3)
    assert [results[f"io{i}"].value for i in range(3)] == ["passed"] * 3


def test_speculate_inside_running_loop_degrades_to_sequential():
    async def main():
        return speculate([SpecTask("x", lambda: 7), SpecTask("y", lambda: 8)])

    results = asyncio.run(main())
    assert results["x"].value == 7 and results["y"].value == 8


# ------------------------------------------------------------------ collect_ready

def test_collect_ready_filters_failures_and_names():
    results = {
        "a": SpecResult("a", value=1),
        "b": SpecResult("b", error=RuntimeError("x")),
        "c": SpecResult("c", value=3),
    }
    assert collect_ready(results) == {"a": 1, "c": 3}
    assert collect_ready(results, names=["c"]) == {"c": 3}


# ------------------------------------------------------------------ Prefetch

def test_prefetch_returns_value():
    assert Prefetch(lambda: "ctx").result(timeout=5) == "ctx"


def test_prefetch_empty_string_is_a_valid_result():
    assert Prefetch(lambda: "").result(timeout=5) == ""


def test_prefetch_error_is_none():
    def boom():
        raise RuntimeError("prefetch down")
    assert Prefetch(boom).result(timeout=5) is None


def test_prefetch_timeout_is_none():
    release = threading.Event()
    handle = Prefetch(lambda: release.wait(5))
    assert handle.result(timeout=0.05) is None  # 間に合わない=外れ
    release.set()


# ------------------------------------------------------------------ fugu_local hook

def test_speculate_hook_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_SPECULATE", raising=False)
    assert fugu_local._speculate_context("q", False, None) is None


def test_speculate_hook_hit_returns_prefetched_context(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_SPECULATE", "1")
    calls = []

    def fake_build_context(question, use_search=False, rag_dirs=None):
        calls.append({"use_search": use_search})
        return "PREFETCHED"

    monkeypatch.setattr(fugu_local, "build_context", fake_build_context)
    resolve = fugu_local._speculate_context("q", False, None)
    assert resolve(False) == "PREFETCHED"
    assert calls == [{"use_search": False}]  # 取り直しは発生しない


def test_speculate_hook_miss_on_condition_change(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_SPECULATE", "1")
    monkeypatch.setattr(fugu_local, "build_context",
                        lambda question, use_search=False, rag_dirs=None: "CTX")
    resolve = fugu_local._speculate_context("q", False, None)
    assert resolve(True) is None  # Conductor が検索要と判断 → 投機の外れ


def test_speculate_hook_failure_is_miss(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_SPECULATE", "1")
    monkeypatch.setattr(
        fugu_local, "build_context",
        lambda question, use_search=False, rag_dirs=None:
        (_ for _ in ()).throw(RuntimeError("search down")))
    resolve = fugu_local._speculate_context("q", False, None)
    assert resolve(False) is None
