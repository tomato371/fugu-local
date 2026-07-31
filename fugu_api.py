"""Minimal REST API for the Fugu orchestrator (FastAPI).

Endpoints
    GET  /health      -> {"status": "ok" | "ollama_unreachable", "ollama_url": "..."}
    POST /ask         -> {"answer": "...", "elapsed_seconds": 1.2}
                         body: {"question": "...", "use_search": false, "rag_dirs": null}
    POST /completion  -> inline code completion (single coder-model call, no MoA)
    POST /refactor    -> instructed rewrite + unified diff
    POST /test-run    -> sandboxed execution with optional self-debug retries

IDE endpoint schemas and curl examples: docs/api_ide.md

Run
    uvicorn fugu_api:app --host 0.0.0.0 --port 8000
Interactive docs
    http://localhost:8000/docs
"""
import difflib
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
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
    try:
        answer = fugu.ask_fugu(req.question, use_search=req.use_search, rag_dirs=req.rag_dirs)
    except Exception as exc:  # surface orchestrator errors as 500s
        raise HTTPException(status_code=500, detail=str(exc))
    if answer is None:
        raise HTTPException(
            status_code=503,
            detail="Setup failed: Ollama unreachable or no models available.",
        )
    return AskResponse(answer=answer, elapsed_seconds=round(time.time() - t0, 2))


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
    out = _single_model_call(system, prompt, req.max_tokens)
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

    sandbox = fugu_sandbox.SubprocessSandbox(timeout=req.timeout)
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
