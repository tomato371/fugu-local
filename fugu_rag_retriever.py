"""Optional bridge: use the sibling project fugu-rag as a fugu-local retriever.

fugu-local's ``rag_bench`` defines ``Retriever = Callable[[str, Corpus], List[str]]``.
The sibling project **fugu-rag** (https://github.com/tomato371/fugu-rag) — an
adaptive, evaluation-driven, 100%-local RAG — exposes its whole pipeline through
exactly that contract via ``fugu_rag.adapter.as_retriever``. This module wires the
two together so fugu-rag can stand in for the built-in ``lexical_retriever`` inside
``rag_bench.evaluate`` (and, in turn, ``ask_fugu``).

fugu-rag is an **optional** dependency. If it is not importable, or no local Ollama
embedder is available, :func:`get_retriever` returns fugu-local's own dependency-free
``lexical_retriever`` — so this module never breaks the offline CI, and installing
fugu-rag alongside is a pure upgrade rather than a hard requirement.

    # benchmark the real fugu-rag pipeline with fugu-local's own metrics
    from rag_bench import evaluate
    from fugu_rag_retriever import get_retriever
    print(evaluate(golden, corpus, get_retriever(chat_model="qwen3:4b"), k=5))
"""
from __future__ import annotations

from typing import Optional

from rag_bench import Retriever, lexical_retriever


def fugu_rag_retriever(
    embed_model: str = "nomic-embed-text",
    chat_model: Optional[str] = None,
    k: int = 5,
) -> Optional[Retriever]:
    """Return a fugu-rag-backed retriever, or ``None`` if it can't be used here.

    Returns ``None`` when fugu-rag is not importable or no local embedding model is
    available, so callers can fall back. With ``chat_model`` set, the adaptive path
    (LLM routing + HyDE + rerank) is enabled; omit it for the fast heuristic-routed
    hybrid path. ``embed_model`` names the Ollama embedding model.
    """
    try:
        from fugu_rag.adapter import as_retriever
        from fugu_rag.llm import OllamaChat, OllamaEmbedder
    except ImportError:
        return None

    embedder = OllamaEmbedder(model=embed_model)
    if not embedder.available():
        return None
    chat = OllamaChat(model=chat_model) if chat_model else None
    return as_retriever(embedder, chat, k=k)


def get_retriever(
    embed_model: str = "nomic-embed-text",
    chat_model: Optional[str] = None,
    k: int = 5,
) -> Retriever:
    """The fugu-rag retriever if available, else fugu-local's lexical fallback."""
    retriever = fugu_rag_retriever(embed_model=embed_model, chat_model=chat_model, k=k)
    return retriever if retriever is not None else lexical_retriever
