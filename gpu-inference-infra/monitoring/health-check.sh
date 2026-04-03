#!/bin/bash

# ────────────────────────────────────────────────────────
# health-check.sh
# Multi-layer health check for vLLM inference server.
#
# Checks:
#   1. Is the vLLM process responding?
#   2. Is GPU memory in expected range? (model loaded, not OOM)
#   3. Can it actually serve inference? (not just alive, but functional)
#   4. Is GPU temperature safe? (not thermally throttling)
#
# Exit codes:
#   0 = healthy
#   1 = unhealthy (with reason printed to stdout)
# ────────────────────────────────────────────────────────

VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORT="${VLLM_PORT:-8000}"
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"

# ── Check 1: Is vLLM responding? ──
if ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "FAIL: vLLM health endpoint unreachable at ${BASE_URL}/health"
    exit 1
fi

# ── Check 2: GPU memory in expected range? ──
if command -v nvidia-smi &> /dev/null; then
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')

    if [ -n "$MEM_USED" ] && [ -n "$MEM_TOTAL" ] && [ "$MEM_TOTAL" -gt 0 ]; then
        MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))

        # Model should be loaded — if memory is too low, model may have crashed
        if [ "$MEM_PCT" -lt 30 ]; then
            echo "FAIL: GPU memory ${MEM_PCT}% — model likely not loaded (expected >50%)"
            exit 1
        fi

        # If memory is critically high, OOM is imminent
        if [ "$MEM_PCT" -gt 97 ]; then
            echo "WARN: GPU memory ${MEM_PCT}% — OOM risk"
            exit 1
        fi
    fi
fi

# ── Check 3: Can it serve inference? ──
RESPONSE=$(curl -sf -m 15 "${BASE_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"test\", \"max_tokens\": 3}" 2>&1)

if [ $? -ne 0 ]; then
    echo "FAIL: Inference request failed or timed out (15s)"
    exit 1
fi

if ! echo "$RESPONSE" | grep -q "choices"; then
    echo "FAIL: Inference response malformed — no 'choices' field"
    exit 1
fi

# ── Check 4: GPU temperature safe? ──
if command -v nvidia-smi &> /dev/null; then
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')

    if [ -n "$TEMP" ] && [ "$TEMP" -gt 85 ]; then
        echo "WARN: GPU temperature ${TEMP}C — thermal throttling likely"
        exit 1
    fi
fi

# ── All checks passed ──
if command -v nvidia-smi &> /dev/null; then
    echo "OK | VRAM: ${MEM_USED}/${MEM_TOTAL} MiB (${MEM_PCT}%) | Temp: ${TEMP}C"
else
    echo "OK | GPU metrics unavailable (no nvidia-smi in container)"
fi
exit 0
