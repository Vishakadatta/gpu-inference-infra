#!/usr/bin/env python3
"""
loadtest.py — Concurrent load tester for NIM / vLLM inference servers.

Fires configurable concurrent requests and measures:
  - Per-request latency (end-to-end)
  - Time to first token (TTFT)  — streaming mode only
  - Tokens per second
  - Throughput (requests/sec)
  - Error rate

Backends:
  nim-hosted   — NVIDIA NIM hosted API  (default; requires NVIDIA_API_KEY)
  nim-container — NIM container on local GPU  (default URL: localhost:8000)
  vllm         — raw vLLM server  (legacy; no auth)

Usage:
  python3 loadtest/loadtest.py                              # NIM hosted, short prompts
  python3 loadtest/loadtest.py --backend nim-container      # self-hosted NIM container
  python3 loadtest/loadtest.py --concurrency 1 2 4 8 16    # sweep concurrency levels
  python3 loadtest/loadtest.py --prompt-length long         # long prompt stress test
"""

import asyncio
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import aiohttp

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)

# NIM hosted API base URL
NIM_HOSTED_BASE   = "https://integrate.api.nvidia.com/v1"
NIM_DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"

# ── Prompt templates ──────────────────────────────────────────────────────────
PROMPTS = {
    "short": "What is a GPU?",
    "medium": (
        "Explain in detail how a graphics processing unit works, including its "
        "architecture, how it differs from a CPU, and why it is useful for "
        "machine learning workloads."
    ),
    "long": (
        "Write a comprehensive technical overview of GPU computing. Cover: "
        "the history of GPU development, CUDA programming model, GPU memory "
        "hierarchy (global memory, shared memory, registers, L1/L2 cache), "
        "thread execution model (warps, blocks, grids), common optimisation "
        "techniques, comparison with CPU computing for parallel workloads, and "
        "the role of GPUs in modern AI inference and training. "
        "Include specific examples where relevant."
    ),
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    request_id:       int
    concurrency:      int
    prompt_length:    str
    latency_ms:       float
    ttft_ms:          float        # 0.0 if streaming not used
    tokens_generated: int
    tokens_per_second: float
    status:           str          # "success" or "error"
    error_message:    str = ""
    model_used:       str = ""


# ── Single request (non-streaming, measures end-to-end latency) ──────────────

async def send_request(
    session:       aiohttp.ClientSession,
    request_id:    int,
    concurrency:   int,
    prompt:        str,
    prompt_length: str,
    base_url:      str,
    model_name:    str,
    max_tokens:    int,
    auth_header:   Optional[str],
) -> RequestResult:

    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }

    start = time.perf_counter()

    try:
        async with session.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000

            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    request_id=request_id, concurrency=concurrency,
                    prompt_length=prompt_length, latency_ms=elapsed_ms,
                    ttft_ms=0.0, tokens_generated=0, tokens_per_second=0.0,
                    status="error",
                    error_message=f"HTTP {resp.status}: {body[:200]}",
                )

            data = await resp.json()
            usage    = data.get("usage", {})
            tokens   = usage.get("completion_tokens", 0)
            tps      = tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
            model_id = data.get("model", model_name)

            return RequestResult(
                request_id=request_id, concurrency=concurrency,
                prompt_length=prompt_length, latency_ms=round(elapsed_ms, 2),
                ttft_ms=0.0, tokens_generated=tokens,
                tokens_per_second=round(tps, 2),
                status="success", model_used=model_id,
            )

    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            request_id=request_id, concurrency=concurrency,
            prompt_length=prompt_length, latency_ms=elapsed_ms,
            ttft_ms=0.0, tokens_generated=0, tokens_per_second=0.0,
            status="error", error_message="Request timed out (120s)",
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            request_id=request_id, concurrency=concurrency,
            prompt_length=prompt_length, latency_ms=elapsed_ms,
            ttft_ms=0.0, tokens_generated=0, tokens_per_second=0.0,
            status="error", error_message=str(e),
        )


# ── Concurrency-level runner ──────────────────────────────────────────────────

async def run_concurrency_level(
    concurrency:  int,
    prompt:       str,
    prompt_length: str,
    base_url:     str,
    model_name:   str,
    max_tokens:   int,
    num_requests: int,
    auth_header:  Optional[str],
) -> List[RequestResult]:

    print(
        f"\n  Concurrency {concurrency}: "
        f"sending {num_requests} requests…", end="", flush=True
    )

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def limited(req_id: int) -> RequestResult:
            async with semaphore:
                return await send_request(
                    session, req_id, concurrency, prompt, prompt_length,
                    base_url, model_name, max_tokens, auth_header,
                )

        results = await asyncio.gather(*[limited(i) for i in range(num_requests)])

    successes = [r for r in results if r.status == "success"]
    errors    = [r for r in results if r.status == "error"]

    if successes:
        avg = sum(r.latency_ms for r in successes) / len(successes)
        print(
            f" done.  {len(successes)}/{num_requests} ok  "
            f"avg latency: {avg:.0f} ms"
        )
    else:
        print(f" done.  ALL FAILED ({len(errors)} errors)")

    return list(results)


# ── Health check ──────────────────────────────────────────────────────────────

async def check_server(base_url: str, backend: str, auth_header: Optional[str]) -> bool:
    """Verify the server is reachable before starting the load test."""
    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header

    # NIM exposes /v1/health/ready; hosted API exposes /v1/models
    if backend == "nim-hosted":
        check_url = f"{base_url}/models"
    else:
        check_url = f"{base_url}/health/ready"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                check_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status not in (200, 503):
                    print(f"ERROR: Server check returned HTTP {resp.status}")
                    return False
                return True
    except Exception as e:
        print(f"ERROR: Cannot reach server at {base_url}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Load test NIM / vLLM inference server"
    )
    parser.add_argument(
        "--backend",
        choices=["nim-hosted", "nim-container", "vllm"],
        default=None,
        help="Backend type (default: read from BACKEND env var, then 'nim-hosted')",
    )
    parser.add_argument(
        "--url",
        default=None,
        help=(
            "Server base URL. Defaults: "
            "nim-hosted → integrate.api.nvidia.com/v1, "
            "nim-container/vllm → http://localhost:8000/v1"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults: read from NIM_MODEL env var, then llama-3.1-8b-instruct",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NVIDIA API key. Defaults: read from NVIDIA_API_KEY env var.",
    )
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--prompt-length",
        choices=["short", "medium", "long"],
        default="short",
    )
    parser.add_argument("--max-tokens",          type=int, default=100)
    parser.add_argument("--requests-per-level",  type=int, default=20)
    parser.add_argument("--output",              default=None)
    args = parser.parse_args()

    # ── Resolve backend ──
    backend = args.backend or os.environ.get("BACKEND", "nim-hosted")

    # ── Resolve base URL ──
    if args.url:
        base_url = args.url.rstrip("/")
    elif backend == "nim-hosted":
        base_url = NIM_HOSTED_BASE
    else:
        base_url = "http://localhost:8000/v1"

    # ── Resolve model ──
    model = (
        args.model
        or os.environ.get("NIM_MODEL")
        or NIM_DEFAULT_MODEL
    )

    # ── Resolve auth ──
    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")
    if backend in ("nim-hosted", "nim-container") and not api_key:
        print(
            "ERROR: NVIDIA_API_KEY is required for NIM backends.\n"
            "  Run `make setup` first, or set NVIDIA_API_KEY in your environment."
        )
        sys.exit(1)
    auth_header = f"Bearer {api_key}" if api_key else None

    prompt = PROMPTS[args.prompt_length]

    # ── Server reachability check ──
    print(f"Checking server at {base_url}…")
    if not await check_server(base_url, backend, auth_header):
        sys.exit(1)

    print("=" * 56)
    print("  GPU Inference Observatory — Load Test")
    print("=" * 56)
    print(f"  Backend:     {backend}")
    print(f"  Server:      {base_url}")
    print(f"  Model:       {model}")
    print(f"  Prompt:      {args.prompt_length}  ({len(prompt)} chars)")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Req/level:   {args.requests_per_level}")

    all_results: List[RequestResult] = []
    test_start = time.time()

    for c in args.concurrency:
        results = await run_concurrency_level(
            concurrency=c,
            prompt=prompt,
            prompt_length=args.prompt_length,
            base_url=base_url,
            model_name=model,
            max_tokens=args.max_tokens,
            num_requests=args.requests_per_level,
            auth_header=auth_header,
        )
        all_results.extend(results)

    test_duration = time.time() - test_start

    # ── Save results ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = args.output or os.path.join(
        RESULTS_DIR,
        f"loadtest_{args.prompt_length}_{int(time.time())}.json",
    )

    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "backend": backend,
                "url": base_url,
                "model": model,
                "prompt_length": args.prompt_length,
                "max_tokens": args.max_tokens,
                "concurrency_levels": args.concurrency,
                "requests_per_level": args.requests_per_level,
            },
            "duration_seconds": round(test_duration, 2),
            "total_requests": len(all_results),
            "results": [asdict(r) for r in all_results],
        }, f, indent=2)

    print(f"\n  Results saved: {output_file}")
    print(f"  Total time:    {test_duration:.1f}s")
    print(f"\n  Run 'python3 loadtest/analyze.py {output_file}' for full analysis.")


if __name__ == "__main__":
    asyncio.run(main())
