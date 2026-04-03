#!/bin/bash
# vLLM Configuration
# Each parameter is documented with what it controls and why the value was chosen.

# ──────────────────────────────────────────────
# Model Selection
# ──────────────────────────────────────────────
# Using Meta's Llama 3 8B Instruct — fits in 24GB VRAM at float16 (~16GB)
# Change this to serve a different model.
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

# ──────────────────────────────────────────────
# GPU Memory
# ──────────────────────────────────────────────
# What % of GPU VRAM vLLM is allowed to use.
# 0.85 = 85% of 24GB = ~20.4GB usable
# Remaining 15% is headroom for CUDA context, KV cache spikes, etc.
# Too high → OOM crash under load
# Too low  → wasted GPU capacity
export GPU_MEM_UTIL=0.85

# ──────────────────────────────────────────────
# Sequence Length
# ──────────────────────────────────────────────
# Max tokens per request (prompt + generated output combined).
# Higher = more memory per request, fewer concurrent requests possible.
# 4096 is a reasonable default for most use cases.
export MAX_MODEL_LEN=4096

# ──────────────────────────────────────────────
# Data Type
# ──────────────────────────────────────────────
# float16 = half precision. Uses half the VRAM of float32.
# Minimal quality loss for inference. Standard practice.
export DTYPE="float16"

# ──────────────────────────────────────────────
# Batching
# ──────────────────────────────────────────────
# Max number of requests vLLM will process simultaneously.
# Higher = better throughput but more VRAM usage.
# This is the primary tuning knob for throughput vs memory tradeoff.
# Start conservative, increase during load testing.
export MAX_NUM_SEQS=8

# ──────────────────────────────────────────────
# API Configuration
# ──────────────────────────────────────────────
export API_HOST="0.0.0.0"
export API_PORT=8000
