"""fugu_core.memory のオフラインテスト + FUGU_MEMORY=1 フックの検証。"""
import json
import math

from fugu_core.memory import (
    Episode,
    LexicalMemory,
    VectorMemory,
    get_default_memory,
    get_memory,
    lessons_for,
    reset_default_memory,
)


def _ep(task="task", lesson="lesson", kind="sandbox", outcome="failure"):
    return Episode(kind=kind, task=task, outcome=outcome, lesson=lesson)


# ------------------------------------------------------------------ Lexical

def test_lexical_record_stamps_ts_with_injected_clock():
    mem = LexicalMemory(now_fn=lambda: "2026-08-01T12:00:00")
    mem.record(_ep())
    assert mem.episodes[0].ts == "2026-08-01T12:00:00"


def test_lexical_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "mem.jsonl")
    mem = LexicalMemory(path=path, now_fn=lambda: "t1")
    mem.record(_ep(task="prime check failed", lesson="use sympy.isprime"))
    mem.record(_ep(task="timeout in loop", lesson="add num_ctx pin"))

    reloaded = LexicalMemory(path=path)
    assert len(reloaded.episodes) == 2
    assert reloaded.episodes[0].lesson == "use sympy.isprime"
    assert reloaded.episodes[0].ts == "t1"


def test_lexical_skips_corrupt_lines(tmp_path):
    path = tmp_path / "mem.jsonl"
    good = json.dumps({"kind": "k", "task": "t", "outcome": "o", "lesson": "l"})
    path.write_text("{broken\n" + good + "\n", encoding="utf-8")
    assert len(LexicalMemory(path=str(path)).episodes) == 1


def test_lexical_search_ranks_by_overlap():
    mem = LexicalMemory()
    mem.record(_ep(task="prime number check", lesson="use trial division"))
    mem.record(_ep(task="matrix multiply", lesson="use numpy"))
    hits = mem.search("how to check a prime number")
    assert [h.task for h in hits] == ["prime number check"]


def test_lexical_search_japanese_bigrams():
    mem = LexicalMemory()
    mem.record(_ep(task="素数判定に失敗", lesson="sympyを使う"))
    mem.record(_ep(task="行列積の計算", lesson="numpyを使う"))
    hits = mem.search("91は素数ですか")
    assert hits and hits[0].task == "素数判定に失敗"


def test_lexical_search_no_overlap_is_empty():
    mem = LexicalMemory()
    mem.record(_ep(task="alpha", lesson="beta"))
    assert mem.search("完全に無関係") == []
    assert mem.search("") == []


def test_lexical_search_prefers_recent_on_tie_and_caps_k():
    mem = LexicalMemory()
    for i in range(5):
        mem.record(_ep(task=f"prime attempt {i}", lesson="same overlap"))
    hits = mem.search("prime", k=2)
    assert len(hits) == 2
    assert hits[0].task == "prime attempt 4"  # 同点なら新しい方


# ------------------------------------------------------------------ Vector

class MiniStore:
    """fugu_rag.vectorstore.VectorStore と同じ API の最小コピー(注入用)。"""

    def __init__(self):
        self.vecs = {}

    def add(self, doc_id, vector):
        self.vecs[doc_id] = list(vector)

    def search(self, query_vec, k=5):
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0
        scored = [(doc_id, cos(query_vec, v)) for doc_id, v in self.vecs.items()]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:k]


class KeywordEmbedder:
    """固定語彙の bag-of-words 埋め込み(決定論的・依存ゼロ)。"""

    VOCAB = ("prime", "matrix", "timeout", "sandbox")

    def embed(self, texts):
        return [[float(w in t.lower()) for w in self.VOCAB] for t in texts]


def test_vector_memory_semantic_ranking(tmp_path):
    mem = VectorMemory(KeywordEmbedder(), path=str(tmp_path / "m.jsonl"),
                       store=MiniStore())
    mem.record(_ep(task="prime check", lesson="prime lesson"))
    mem.record(_ep(task="matrix multiply", lesson="matrix lesson"))
    hits = mem.search("prime question", k=1)
    assert [h.task for h in hits] == ["prime check"]


def test_vector_memory_rebuilds_index_from_jsonl(tmp_path):
    path = str(tmp_path / "m.jsonl")
    VectorMemory(KeywordEmbedder(), path=path, store=MiniStore()).record(
        _ep(task="timeout bug", lesson="pin num_ctx"))
    reloaded = VectorMemory(KeywordEmbedder(), path=path, store=MiniStore())
    assert len(reloaded.episodes) == 1
    assert reloaded.search("timeout", k=1)[0].lesson == "pin num_ctx"


def test_vector_memory_empty_search():
    mem = VectorMemory(KeywordEmbedder(), store=MiniStore())
    assert mem.search("anything") == []


# ------------------------------------------------------------------ get_memory / default

def test_get_memory_without_embedder_is_lexical():
    assert isinstance(get_memory(), LexicalMemory)


def test_get_memory_falls_back_when_fugu_rag_missing(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "fugu_rag", None)
    monkeypatch.setitem(sys.modules, "fugu_rag.vectorstore", None)
    assert isinstance(get_memory(embedder=KeywordEmbedder()), LexicalMemory)


def test_get_default_memory_honors_env_path(tmp_path, monkeypatch):
    path = str(tmp_path / "custom.jsonl")
    monkeypatch.setenv("FUGU_MEMORY_PATH", path)
    reset_default_memory()
    try:
        mem = get_default_memory()
        assert isinstance(mem, LexicalMemory) and mem.path == path
        assert get_default_memory() is mem  # シングルトン
    finally:
        reset_default_memory()


# ------------------------------------------------------------------ lessons_for

def test_lessons_for_formats_hits():
    mem = LexicalMemory()
    mem.record(_ep(task="prime check", lesson="use sympy", outcome="failure"))
    text = lessons_for(mem, "prime")
    assert text.startswith("## 過去エピソードからの教訓")
    assert "- [sandbox/failure] use sympy" in text


def test_lessons_for_empty_when_no_hits():
    assert lessons_for(LexicalMemory(), "anything") == ""


# ------------------------------------------------------------------ hooks

def _patched_default(monkeypatch):
    """既定記憶をメモリ内 LexicalMemory に差し替えて返す。"""
    from fugu_core import memory as memory_mod
    mem = LexicalMemory()
    monkeypatch.setattr(memory_mod, "get_default_memory", lambda: mem)
    return mem


def test_fugu_local_hook_disabled_by_default(monkeypatch):
    import fugu_local
    monkeypatch.delenv("FUGU_MEMORY", raising=False)
    assert fugu_local._memory_lessons("q") == "q"


def test_fugu_local_hook_prepends_lessons(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_MEMORY", "1")
    mem = _patched_default(monkeypatch)
    mem.record(_ep(task="prime check", lesson="use sympy"))
    out = fugu_local._memory_lessons("prime question")
    assert out.endswith("prime question")
    assert "use sympy" in out


def test_fugu_local_hook_no_hits_leaves_question(monkeypatch):
    import fugu_local
    monkeypatch.setenv("FUGU_MEMORY", "1")
    _patched_default(monkeypatch)
    assert fugu_local._memory_lessons("unrelated") == "unrelated"


def test_sandbox_hook_records_episode(monkeypatch):
    import fugu_sandbox
    monkeypatch.setenv("FUGU_MEMORY", "1")
    mem = _patched_default(monkeypatch)
    result, _code, attempts = fugu_sandbox.run_with_self_debug(
        "print('hello')", chat=None, max_retries=0)
    assert result.ok and attempts == 1
    assert len(mem.episodes) == 1
    assert mem.episodes[0].kind == "sandbox"
    assert mem.episodes[0].outcome == "success"


def test_evaluator_hook_records_episode(monkeypatch, tmp_path):
    monkeypatch.setenv("FUGU_MEMORY", "1")
    mem = _patched_default(monkeypatch)
    from fugu_llm import FakeChat
    from fugu_evolve.evaluator import verify
    from tests.test_evolve_evaluator import (
        BASELINE_NO_BENCH, PYTEST_OK, FakeSandbox, _ws)
    v = verify(_ws(tmp_path), FakeSandbox([PYTEST_OK]),
               FakeChat(default="unused"), BASELINE_NO_BENCH)
    assert v.verdict == "VERIFIED"
    assert len(mem.episodes) == 1
    assert mem.episodes[0].kind == "evolve"
    assert mem.episodes[0].outcome == "success"
