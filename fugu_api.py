"""Minimal REST API for the Fugu orchestrator (FastAPI).

Endpoints
    GET  /health      -> {"status": "ok" | "ollama_unreachable", "ollama_url": "..."}
    POST /ask         -> {"answer": "...", "elapsed_seconds": 1.2}
                         body: {"question": "...", "use_search": false, "rag_dirs": null}
    GET  /ask/stream  -> Server-Sent Events: real-time pipeline events
                         (plan/proposals/aggregate/sandbox/critic/final), then the
                         answer and a `done` sentinel
    WS   /ws/ask      -> same event stream over WebSocket (send {"question": ...})
    POST /completion  -> inline code completion (single coder-model call, no MoA)
    POST /refactor    -> instructed rewrite + unified diff
    POST /test-run    -> sandboxed execution with optional self-debug retries

IDE endpoint schemas and curl examples: docs/api_ide.md

Run
    uvicorn fugu_api:app --host 0.0.0.0 --port 8000
Interactive docs
    http://localhost:8000/docs
"""
import asyncio
import difflib
import json
import os
import queue
import re
import threading
import time
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import fugu_local as fugu

app = FastAPI(
    title="Fugu-Local API",
    description="REST API for the dynamic Mixture-of-Agents orchestrator (100% local via Ollama).",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Is 91 a prime number?"])
    use_search: bool = Field(False, description="Inject DuckDuckGo web-search context")
    rag_dirs: Optional[List[str]] = Field(None, description="Local document dirs for RAG")
    thinking_budget: Optional[Literal["minimal", "low", "medium", "high",
                                      "ultra", "max", "auto"]] = Field(
        None, description="Extended-thinking depth, 6 levels + auto "
                          "(controls think mode, reflection count, and the "
                          "MoA round floor)")


class AskResponse(BaseModel):
    answer: str
    elapsed_seconds: float


@app.get("/health")
def health():
    """Report whether the Ollama backend is reachable."""
    up = fugu.server_up()
    return {"status": "ok" if up else "ollama_unreachable", "ollama_url": fugu.OLLAMA_URL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a question through the full dynamic-MoA pipeline."""
    t0 = time.time()
    # The thinking budget travels via the same env flag the CLI uses; scope it to
    # this request so one caller's budget never leaks into the next request.
    prev_budget = os.environ.get("FUGU_THINKING_BUDGET")
    if req.thinking_budget:
        os.environ["FUGU_THINKING_BUDGET"] = req.thinking_budget
    try:
        answer = fugu.ask_fugu(req.question, use_search=req.use_search, rag_dirs=req.rag_dirs)
    except Exception as exc:  # surface orchestrator errors as 500s
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if req.thinking_budget:
            if prev_budget is None:
                os.environ.pop("FUGU_THINKING_BUDGET", None)
            else:
                os.environ["FUGU_THINKING_BUDGET"] = prev_budget
    if answer is None:
        raise HTTPException(
            status_code=503,
            detail="Setup failed: Ollama unreachable or no models available.",
        )
    return AskResponse(answer=answer, elapsed_seconds=round(time.time() - t0, 2))


# ------------------------------------------------------------------ real-time streaming
# The orchestrator runs in a worker thread; fugu_local._emit events flow through
# a queue.Queue to the client. Single-flight by design (one GPU, one event sink):
# concurrent streams would interleave sinks, matching the existing web UI lock
# philosophy. `None` on the queue is the internal done sentinel.


def _run_streaming(question: str, use_search: bool, event_q: "queue.Queue") -> None:
    """Worker: register the event sink, run the pipeline, then signal completion.

    Puts ("answer", {...}) / ("error", {...}) before the final `None` sentinel so
    readers can simply drain the queue until None.
    """
    def sink(kind, data):
        event_q.put((kind, data))

    fugu.set_event_sink(sink)
    try:
        answer = fugu.ask_fugu(question, use_search=use_search)
        if answer is None:
            event_q.put(("error", {"detail": "Setup failed: Ollama unreachable "
                                             "or no models available."}))
        else:
            event_q.put(("answer", {"answer": answer}))
    except Exception as exc:
        event_q.put(("error", {"detail": str(exc)}))
    finally:
        fugu.set_event_sink(None)
        event_q.put(None)


def _sse_line(kind: str, data: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/ask/stream")
def ask_stream(question: str, use_search: bool = False):
    """Answer via the full pipeline, streaming internal events as SSE.

    Event kinds: plan / proposals / aggregate / sandbox / critic / final
    (from fugu_local._emit), then `answer` (or `error`) and a final `done`.
    """
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    def gen():
        event_q: "queue.Queue" = queue.Queue()
        worker = threading.Thread(
            target=_run_streaming, args=(question, use_search, event_q), daemon=True)
        worker.start()
        while True:
            item = event_q.get()
            if item is None:
                break
            yield _sse_line(item[0], item[1])
        yield _sse_line("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.websocket("/ws/ask")
async def ws_ask(websocket: WebSocket):
    """Same event stream over WebSocket. Client sends {"question": ..., "use_search"?}.

    Each event is a JSON object {"event": kind, "data": {...}}; the stream ends
    with {"event": "done"} and the socket is closed by the server.
    """
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        question = str(payload.get("question") or "").strip()
        if not question:
            await websocket.send_json({"event": "error",
                                       "data": {"detail": "question must not be empty"}})
            return
        event_q: "queue.Queue" = queue.Queue()
        worker = threading.Thread(
            target=_run_streaming,
            args=(question, bool(payload.get("use_search")), event_q), daemon=True)
        worker.start()
        loop = asyncio.get_running_loop()
        while True:
            # blocking queue.get を executor へ (D-4: asyncio + run_in_executor)
            item = await loop.run_in_executor(None, event_q.get)
            if item is None:
                break
            await websocket.send_json({"event": item[0], "data": item[1]})
        await websocket.send_json({"event": "done", "data": {}})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass  # クライアント切断済みなら閉じ損ねてよい


# ------------------------------------------------------------------ approval gate (E3)
# FUGU_REQUIRE_APPROVAL=1 のとき、sandbox 実行や evolve merge が
# `approval_required` イベント(SSE/WS)を発行してブロックする。ここはその解決口。


class ApprovalDecision(BaseModel):
    approve: bool = Field(..., description="true=実行を許可 / false=拒否")


@app.get("/approvals")
def approvals():
    """未決の承認要求 run_id 一覧(FUGU_REQUIRE_APPROVAL=1 のときのみ増える)。"""
    import fugu_approval
    return {"pending": fugu_approval.pending()}


@app.post("/approve/{run_id}")
def approve(run_id: str, decision: ApprovalDecision):
    """承認要求を解決する。未知・解決済みの run_id は 404。"""
    import fugu_approval
    if not fugu_approval.resolve(run_id, decision.approve):
        raise HTTPException(status_code=404,
                            detail=f"unknown or already-resolved run_id: {run_id}")
    return {"run_id": run_id, "approved": decision.approve}


# ------------------------------------------------------------------ IDE endpoints
# Single-model, low-latency surfaces for editor integration (VS Code / Cursor).
# No MoA, no Conductor — one coder-model call (or a pure sandbox run).

CODER_MODEL = "qwen3-coder:30b"


class CompletionRequest(BaseModel):
    prefix: str = Field(..., min_length=1, description="Code before the cursor")
    suffix: str = Field("", description="Code after the cursor (optional context)")
    language: str = Field("python", description="Language hint")
    max_tokens: int = Field(256, ge=1, le=2048, description="num_predict cap")


class CompletionResponse(BaseModel):
    completion: str
    elapsed_seconds: float


class RefactorRequest(BaseModel):
    code: str = Field(..., min_length=1)
    instruction: str = Field(..., min_length=1, examples=["extract the loop into a helper"])
    language: str = Field("python")


class RefactorResponse(BaseModel):
    refactored: str
    diff: str
    elapsed_seconds: float


class TestRunRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Script to execute")
    tests: Optional[str] = Field(
        None, description="Optional pytest module run against the code (TDC mode)")
    max_retries: int = Field(0, ge=0, le=5,
                             description="Self-debug attempts on failure (needs LLM)")
    timeout: float = Field(30.0, gt=0, le=600)


class TestRunResponse(BaseModel):
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    attempts: int
    code: str  # final (possibly self-debug-repaired) code


def _single_model_call(system: str, prompt: str, max_tokens: int) -> str:
    """One direct coder-model call through fugu.ask(); raises HTTP 502 on error."""
    if not fugu.setup():
        raise HTTPException(status_code=503,
                            detail="Setup failed: Ollama unreachable or no models available.")
    raw = fugu.ask(
        CODER_MODEL,
        [{"role": "system", "content": system},
         {"role": "user", "content": prompt}],
        0.2, label="ide", num_predict=max_tokens,
    )
    out = fugu.strip_think(raw)
    if out.startswith("__ERROR__"):
        raise HTTPException(status_code=502, detail=out)
    return out


#: Any-language fenced block (fugu_sandbox's regex only matches python/bash tags).
_FENCE_ANY_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)


def _clean_completion(out: str, prefix: str) -> str:
    """Normalize a completion reply to pure insertion text.

    Live finding 2026-08-01: qwen3-coder sometimes ignores the no-fence /
    no-repetition instruction and returns the whole function fenced. IDE
    clients need only the text to insert at the cursor, so strip one fenced
    wrapper and, if the reply restates the prefix verbatim, drop that echo.
    """
    m = _FENCE_ANY_RE.search(out)
    if m:
        out = m.group(1)
    if prefix and out.startswith(prefix):
        out = out[len(prefix):]
    return out.rstrip("\n")


@app.post("/completion", response_model=CompletionResponse)
def completion(req: CompletionRequest):
    """Inline code completion: continue the code at the cursor. No MoA."""
    t0 = time.time()
    system = (
        f"You are an inline {req.language} code completion engine. Continue the "
        "code exactly at the cursor. Output ONLY the inserted code — no fences, "
        "no explanations, no repetition of the prefix."
    )
    prompt = f"<prefix>\n{req.prefix}\n</prefix>\n<suffix>\n{req.suffix}\n</suffix>"
    out = _clean_completion(_single_model_call(system, prompt, req.max_tokens),
                            req.prefix)
    return CompletionResponse(completion=out,
                              elapsed_seconds=round(time.time() - t0, 2))


@app.post("/refactor", response_model=RefactorResponse)
def refactor(req: RefactorRequest):
    """Rewrite code per instruction and return the unified diff."""
    import fugu_sandbox

    t0 = time.time()
    system = (
        f"You are a {req.language} refactoring engine. Apply the instruction and "
        "return the COMPLETE rewritten code in ONE fenced code block. No prose."
    )
    prompt = f"Instruction: {req.instruction}\n\n```{req.language}\n{req.code}\n```"
    out = _single_model_call(system, prompt, 2048)
    refactored = fugu_sandbox.extract_code_block(out) or out.strip()
    diff = "".join(difflib.unified_diff(
        req.code.splitlines(keepends=True),
        (refactored + "\n").splitlines(keepends=True),
        fromfile="before", tofile="after",
    ))
    return RefactorResponse(refactored=refactored, diff=diff,
                            elapsed_seconds=round(time.time() - t0, 2))


@app.post("/test-run", response_model=TestRunResponse)
def test_run(req: TestRunRequest):
    """Execute code in the sandbox; optionally run pytest tests against it and
    self-debug on failure (max_retries > 0 requires a reachable model)."""
    import fugu_sandbox
    import fugu_tdc

    sandbox = fugu_sandbox.get_sandbox(timeout=req.timeout)  # Docker 稼働時は自動昇格
    if req.tests:
        result = fugu_tdc.run_tests(req.code, req.tests, sandbox=sandbox,
                                    timeout=req.timeout)
        final_code, attempts = req.code, 1
    elif req.max_retries > 0:
        import fugu_llm
        result, final_code, attempts = fugu_sandbox.run_with_self_debug(
            req.code, fugu_llm.AskChat(model=CODER_MODEL, label="ide-debug"),
            sandbox=sandbox, max_retries=req.max_retries, timeout=req.timeout)
    else:
        result = sandbox.run(req.code, timeout=req.timeout)
        final_code, attempts = req.code, 1
    return TestRunResponse(ok=result.ok, stdout=result.stdout, stderr=result.stderr,
                           exit_code=result.exit_code, timed_out=result.timed_out,
                           attempts=attempts, code=final_code)
