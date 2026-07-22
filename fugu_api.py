"""Minimal REST API for the Fugu orchestrator (FastAPI).

Endpoints
    GET  /health  -> {"status": "ok" | "ollama_unreachable", "ollama_url": "..."}
    POST /ask     -> {"answer": "...", "elapsed_seconds": 1.2}
                     body: {"question": "...", "use_search": false, "rag_dirs": null}

Run
    uvicorn fugu_api:app --host 0.0.0.0 --port 8000
Interactive docs
    http://localhost:8000/docs
"""
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
