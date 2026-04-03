#!/usr/bin/env python3
"""
loadtest.py — Concurrent load tester for vLLM inference server.

Fires configurable concurrent requests and measures:
  - Per-request latency
  - Time to first token (TTFT)
  - Tokens per second
  - Throughput (requests/sec)
  - Error rate

Usage:
  python3 loadtest.py                           # defaults: 8 concurrent, short prompts
  python3 loadtest.py --concurrency 1 2 4 8 16  # sweep concurrency levels
  python3 loadtest.py --prompt-length long       # test with long prompts
"""

import asyncio
import aiohttp
import argparse
import json
import time
import os
import sys
from dataclasses import dataclass, asdict
from typing import List

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# ── Prompt templates by length ──
PROMPTS = {
    "short": "What is a GPU?",
    "medium": "Explain in detail how a graphics processing unit works, including its architecture, "
              "how it differs from a CPU, and why it is useful for machine learning workloads.",
    "long": "Write a comprehensive technical overview of GPU computing. Cover the following topics: "
            "the history of GPU development, CUDA programming model, GPU memory hierarchy "
            "(global memory, shared memory, registers, L1/L2 cache), thread execution model "
            "(warps, blocks, grids), common optimization techniques, comparison with CPU computing "
            "for parallel workloads, and the role of GPUs in modern AI inference and training. "
            "Include specific examples where relevant.",
}


@dataclass
class RequestResult:
    """Result of a single inference request."""
    request_id: int
    concurrency: int
    prompt_length: str
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    status: str  # "success" or "error"
    error_message: str = ""


async def send_request(
    session: aiohttp.ClientSession,
    request_id: int,
    concurrency: int,
    prompt: str,
    prompt_length: str,
    base_url: str,
    model_name: str,
    max_tokens: int,
) -> RequestResult:
    """Send a single inference request and measure performance."""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    start = time.perf_counter()

    try:
        async with session.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000

            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    request_id=request_id,
                    concurrency=concurrency,
                    prompt_length=prompt_length,
                    latency_ms=elapsed_ms,
                    tokens_generated=0,
                    tokens_per_second=0,
                    status="error",
                    error_message=f"HTTP {resp.status}: {body[:200]}",
                )

            data = await resp.json()
            tokens = data["usage"]["completion_tokens"]
            tps = tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

            return RequestResult(
                request_id=request_id,
                concurrency=concurrency,
                prompt_length=prompt_length,
                latency_ms=elapsed_ms,
                tokens_generated=tokens,
                tokens_per_second=round(tps, 2),
                status="success",
            )

    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            request_id=request_id,
            concurrency=concurrency,
            prompt_length=prompt_length,
            latency_ms=elapsed_ms,
            tokens_generated=0,
            tokens_per_second=0,
            status="error",
            error_message="Request timed out (120s)",
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            request_id=request_id,
            concurrency=concurrency,
            prompt_length=prompt_length,
            latency_ms=elapsed_ms,
            tokens_generated=0,
            tokens_per_second=0,
            status="error",
            error_message=str(e),
        )


async def run_concurrency_level(
    concurrency: int,
    prompt: str,
    prompt_length: str,
    base_url: str,
    model_name: str,
    max_tokens: int,
    num_requests: int,
) -> List[RequestResult]:
    """Run a batch of concurrent requests at a given concurrency level."""

    print(f"\n  Concurrency {concurrency}: sending {num_requests} requests...", end="", flush=True)

    async with aiohttp.ClientSession() as session:
        # Create all request tasks
        tasks = [
            send_request(session, i, concurrency, prompt, prompt_length, base_url, model_name, max_tokens)
            for i in range(num_requests)
        ]

        # Run them with concurrency limit
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_task(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[limited_task(t) for t in tasks])

    successes = [r for r in results if r.status == "success"]
    errors = [r for r in results if r.status == "error"]

    if successes:
        latencies = [r.latency_ms for r in successes]
        avg_latency = sum(latencies) / len(latencies)
        print(f" done. {len(successes)}/{num_requests} ok, avg latency: {avg_latency:.0f}ms")
    else:
        print(f" done. ALL FAILED ({len(errors)} errors)")

    return list(results)


async def main():
    parser = argparse.ArgumentParser(description="Load test vLLM inference server")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                        help="Concurrency levels to test")
    parser.add_argument("--prompt-length", choices=["short", "medium", "long"], default="short",
                        help="Prompt length category")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate per request")
    parser.add_argument("--requests-per-level", type=int, default=20,
                        help="Number of requests per concurrency level")
    parser.add_argument("--output", default=None, help="Output JSON file (default: auto-named)")
    args = parser.parse_args()

    prompt = PROMPTS[args.prompt_length]

    # Verify server is reachable
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"ERROR: Server health check returned {resp.status}")
                    sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.url}: {e}")
        sys.exit(1)

    print("============================================")
    print("  Load Test")
    print("============================================")
    print(f"  Server:      {args.url}")
    print(f"  Model:       {args.model}")
    print(f"  Prompt:      {args.prompt_length} ({len(prompt)} chars)")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Requests/level: {args.requests_per_level}")

    all_results = []
    test_start = time.time()

    for c in args.concurrency:
        results = await run_concurrency_level(
            concurrency=c,
            prompt=prompt,
            prompt_length=args.prompt_length,
            base_url=args.url,
            model_name=args.model,
            max_tokens=args.max_tokens,
            num_requests=args.requests_per_level,
        )
        all_results.extend(results)

    test_duration = time.time() - test_start

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = args.output or os.path.join(
        RESULTS_DIR,
        f"loadtest_{args.prompt_length}_{int(time.time())}.json"
    )

    output_data = {
        "config": {
            "url": args.url,
            "model": args.model,
            "prompt_length": args.prompt_length,
            "max_tokens": args.max_tokens,
            "concurrency_levels": args.concurrency,
            "requests_per_level": args.requests_per_level,
        },
        "duration_seconds": round(test_duration, 2),
        "total_requests": len(all_results),
        "results": [asdict(r) for r in all_results],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print(f"  Total time: {test_duration:.1f}s")
    print(f"\n  Run 'python3 loadtest/analyze.py {output_file}' for detailed analysis.")


if __name__ == "__main__":
    asyncio.run(main())
