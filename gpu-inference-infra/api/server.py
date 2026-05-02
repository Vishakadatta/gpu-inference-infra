"""GPU Inference Observatory — FastAPI backend.

Endpoints:
  GET  /api/health       — NIM connectivity check
  GET  /api/models       — list available NIM models (filtered, no Chinese-origin publishers)
  POST /api/infer        — single inference with TTFT measurement  (10 req/min per IP)
  POST /api/loadtest     — concurrent load test, returns aggregated metrics  (10 req/min per IP)
  GET  /                 — serves frontend/index.html

Environment variables (set in deploy/.env or Render dashboard):
  NVIDIA_API_KEY  — required for NIM hosted API
  NIM_BASE        — defaults to https://integrate.api.nvidia.com/v1
  NIM_MODEL       — default model, e.g. meta/llama-3.1-8b-instruct
  BACKEND         — nim-hosted | nim-container  (default: nim-hosted)
  NIM_HOST        — container host  (nim-container only, default: localhost)
  NIM_PORT        — container port  (nim-container only, default: 8000)
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Load .env if present (local dev) ─────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / "deploy" / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
_BACKEND   = os.environ.get("BACKEND", "nim-hosted")
_NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-8b-instruct")


def _nim_base() -> str:
    """Read NIM base URL fresh — same pattern as ModelCouncil."""
    if os.environ.get("BACKEND", "nim-hosted") == "nim-container":
        host = os.environ.get("NIM_HOST", "localhost")
        port = os.environ.get("NIM_PORT", "8000")
        return f"http://{host}:{port}/v1"
    return os.environ.get("NIM_BASE", "https://integrate.api.nvidia.com/v1").rstrip("/")


def _auth_header() -> str | None:
    """Read NVIDIA_API_KEY fresh on every call — same pattern as ModelCouncil.
    This ensures the key is always current whether loaded from env (Render)
    or from a .env file loaded after import (local dev)."""
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY", "")
    return f"Bearer {key}" if key else None

# ── Publisher blocklist (Chinese-origin models) ───────────────────────────────
# Publisher is the prefix before the first "/" in a NIM model ID.
_BLOCKED_PUBLISHERS: frozenset[str] = frozenset({
    "deepseek-ai", "deepseek",
    "qwen", "alibaba",
    "01-ai",
    "thudm", "zhipu-ai", "zhipuai", "glm",
    "minimax-ai", "minimaxai",
    "moonshot-ai", "moonshotai",
    "baichuan-inc", "baichuan",
    "z-ai",
    "internlm",
    "openbmb",
    "tiiuae",          # falcon (UAE, excluded per policy)
})


def _publisher_allowed(model_id: str) -> bool:
    """Return True if the model's publisher is not on the blocklist."""
    if "/" not in model_id:
        return True   # bare model names have no publisher prefix — keep them
    publisher = model_id.split("/")[0].lower()
    return publisher not in _BLOCKED_PUBLISHERS


# ── Prompt presets ────────────────────────────────────────────────────────────
PROMPTS = {
    "short":  "What is a GPU?",
    "medium": (
        "Explain in detail how a graphics processing unit works, "
        "including its architecture, how it differs from a CPU, "
        "and why it is useful for machine learning workloads."
    ),
    "long": (
        "Write a comprehensive technical overview of GPU computing. "
        "Cover: the history of GPU development, CUDA programming model, "
        "GPU memory hierarchy (global, shared, registers, L1/L2 cache), "
        "thread execution model (warps, blocks, grids), common optimisation "
        "techniques, comparison with CPUs for parallel workloads, and the "
        "role of GPUs in modern AI inference and training."
    ),
}

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="GPU Inference Observatory", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve frontend static files
_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ── Request / response models ─────────────────────────────────────────────────

class InferRequest(BaseModel):
    prompt:     str
    model:      Optional[str] = None
    max_tokens: int = 256


class LoadTestRequest(BaseModel):
    prompt_preset: str = "short"   # short | medium | long
    concurrency:   int = 4
    num_requests:  int = 20
    max_tokens:    int = 100
    model:         Optional[str] = None


# ── Core NIM call with TTFT measurement ──────────────────────────────────────

async def _call_nim_streaming(
    prompt: str,
    model: str,
    max_tokens: int,
) -> dict:
    """
    Call NIM with server-sent events (streaming=True) to measure TTFT accurately.

    Returns:
        answer          — full response text
        ttft_ms         — time from request sent to first token received (ms)
        total_latency_ms — time from request sent to last token received (ms)
        tokens_generated — approximate token count (SSE chunk count)
        tokens_per_second
        prompt_tokens   — from usage field on final chunk (if available)
        model_used      — actual model ID returned by NIM
    """
    auth = _auth_header()
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = auth

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    start         = time.perf_counter()
    ttft_ms       = None
    full_text     = ""
    token_count   = 0
    prompt_tokens = 0
    model_used    = model

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{_nim_base()}/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"NIM error {resp.status_code}: {body.decode()[:300]}",
                )

            async for raw_line in resp.aiter_lines():
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:]
                if data.strip() == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Capture usage from the final chunk (some NIM models send it)
                if obj.get("usage"):
                    prompt_tokens = obj["usage"].get("prompt_tokens", 0)

                choices = obj.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                text  = delta.get("content") or ""

                if text:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    full_text   += text
                    token_count += 1

                if obj.get("model"):
                    model_used = obj["model"]

    total_ms = (time.perf_counter() - start) * 1000
    tps = token_count / (total_ms / 1000) if total_ms > 0 else 0.0

    return {
        "answer":            full_text,
        "ttft_ms":           round(ttft_ms or 0.0, 1),
        "total_latency_ms":  round(total_ms, 1),
        "tokens_generated":  token_count,
        "prompt_tokens":     prompt_tokens,
        "tokens_per_second": round(tps, 1),
        "model_used":        model_used,
        "backend":           _BACKEND,
    }


# ── Single request (non-streaming) for load test ─────────────────────────────

async def _call_nim_single(
    prompt: str,
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    req_id: int,
) -> dict:
    auth = _auth_header()
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = auth

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    start = time.perf_counter()
    async with semaphore:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{_nim_base()}/chat/completions",
                    json=payload,
                    headers=headers,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                return {
                    "req_id": req_id,
                    "status": "error",
                    "latency_ms": round(elapsed_ms, 2),
                    "error": f"HTTP {resp.status_code}",
                }

            data   = resp.json()
            usage  = data.get("usage", {})
            tokens = usage.get("completion_tokens", 0)
            tps    = tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0

            return {
                "req_id":           req_id,
                "status":           "success",
                "latency_ms":       round(elapsed_ms, 2),
                "tokens_generated": tokens,
                "tokens_per_second": round(tps, 2),
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "req_id":     req_id,
                "status":     "error",
                "latency_ms": round(elapsed_ms, 2),
                "error":      str(e),
            }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index = _FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"status": "GPU Inference Observatory API", "docs": "/docs"})


@app.get("/api/health")
async def health():
    """Check NIM connectivity and measure round-trip latency."""
    auth = _auth_header()
    headers = {"Authorization": auth} if auth else {}
    base = _nim_base()

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if os.environ.get("BACKEND", "nim-hosted") == "nim-hosted":
                r = await client.get(f"{base}/models", headers=headers)
            else:
                r = await client.get(f"{base}/health/ready", headers=headers)
        latency_ms = (time.perf_counter() - start) * 1000
        ok = r.status_code == 200
        return {
            "status":      "ok" if ok else "degraded",
            "backend":     os.environ.get("BACKEND", "nim-hosted"),
            "base_url":    base,
            "model":       _NIM_MODEL,
            "latency_ms":  round(latency_ms, 1),
            "http_status": r.status_code,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)},
        )


@app.get("/api/models")
async def list_models():
    """Return allowed NIM models (Chinese-origin publishers filtered out)."""
    auth = _auth_header()
    headers = {"Authorization": auth} if auth else {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{_nim_base()}/models", headers=headers)
        r.raise_for_status()
        data   = r.json()
        models = [
            m["id"]
            for m in data.get("data", [])
            if _publisher_allowed(m.get("id", ""))
        ]
        # Put the configured default first if it is in the allowed list
        if _NIM_MODEL in models:
            models.remove(_NIM_MODEL)
            models.insert(0, _NIM_MODEL)
        return {"models": models, "default": _NIM_MODEL}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/infer")
@limiter.limit("10/minute")
async def infer(request: Request, req: InferRequest):
    """
    Run a single inference request and return the answer + infrastructure metrics.

    Rate-limited to 10 requests per minute per IP.

    Metrics returned:
      ttft_ms          — time to first token  (how long before words started arriving)
      total_latency_ms — full round-trip time
      tokens_generated — number of tokens in the response
      tokens_per_second — generation throughput
      prompt_tokens    — number of tokens in the prompt
    """
    model = req.model or _NIM_MODEL
    try:
        result = await _call_nim_streaming(req.prompt, model, req.max_tokens)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/loadtest")
@limiter.limit("10/minute")
async def loadtest(request: Request, req: LoadTestRequest):
    """
    Fire N concurrent requests and return latency statistics.

    Rate-limited to 10 requests per minute per IP.

    This is the infrastructure story: watch how TTFT and throughput degrade
    as concurrency increases.

    Returns per-request results + aggregate stats:
      avg_latency_ms, p50_ms, p95_ms, p99_ms, avg_tps, error_rate
    """
    if req.prompt_preset not in PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"prompt_preset must be one of {list(PROMPTS.keys())}",
        )
    if req.concurrency > 16:
        raise HTTPException(status_code=400, detail="concurrency max is 16")
    if req.num_requests > 50:
        raise HTTPException(status_code=400, detail="num_requests max is 50")

    model  = req.model or _NIM_MODEL
    prompt = PROMPTS[req.prompt_preset]
    sem    = asyncio.Semaphore(req.concurrency)

    start   = time.perf_counter()
    tasks   = [
        _call_nim_single(prompt, model, req.max_tokens, sem, i)
        for i in range(req.num_requests)
    ]
    results = await asyncio.gather(*tasks)
    total_s = time.perf_counter() - start

    successes = [r for r in results if r["status"] == "success"]
    errors    = [r for r in results if r["status"] == "error"]

    latencies = sorted(r["latency_ms"] for r in successes)
    n         = len(latencies)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(int(p / 100 * n), n - 1)
        return round(latencies[idx], 1)

    summary = {
        "total_requests":   req.num_requests,
        "successful":       len(successes),
        "errors":           len(errors),
        "error_rate_pct":   round(len(errors) / req.num_requests * 100, 1),
        "duration_s":       round(total_s, 2),
        "concurrency":      req.concurrency,
        "prompt_preset":    req.prompt_preset,
        "model":            model,
        "avg_latency_ms":   round(statistics.mean(latencies), 1) if latencies else 0,
        "p50_ms":           pct(50),
        "p95_ms":           pct(95),
        "p99_ms":           pct(99),
        "min_latency_ms":   round(latencies[0],  1) if latencies else 0,
        "max_latency_ms":   round(latencies[-1], 1) if latencies else 0,
        "avg_tps":          round(
            statistics.mean(r["tokens_per_second"] for r in successes), 1
        ) if successes else 0,
    }

    return {"summary": summary, "results": results}
