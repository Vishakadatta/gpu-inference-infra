#!/bin/bash

# ─────────────────────────────────────────────────────────────────
# health-check.sh
# Multi-layer health check — works for both NIM backends.
#
# BACKEND=nim-hosted     → checks NIM hosted API reachability + live inference
# BACKEND=nim-container  → checks container liveness, readiness, GPU, inference
# BACKEND=vllm           → legacy vLLM checks (VRAM + inference)
#
# Exit codes:
#   0 = healthy
#   1 = unhealthy (reason printed to stdout)
# ─────────────────────────────────────────────────────────────────

BACKEND="${BACKEND:-nim-hosted}"
NIM_HOST="${NIM_HOST:-localhost}"
NIM_PORT="${NIM_PORT:-8000}"
NIM_MODEL="${NIM_MODEL:-meta/llama-3.1-8b-instruct}"
NVIDIA_API_KEY="${NVIDIA_API_KEY:-}"
NIM_BASE="${NIM_BASE:-https://integrate.api.nvidia.com/v1}"

# ── Helper: curl with optional auth ──────────────────────────────
nim_curl() {
    local url="$1"
    shift
    if [ -n "$NVIDIA_API_KEY" ]; then
        curl -sf -H "Authorization: Bearer ${NVIDIA_API_KEY}" "$url" "$@"
    else
        curl -sf "$url" "$@"
    fi
}

# ═════════════════════════════════════════════════════════════════
# PATH A: NIM Hosted API
# ═════════════════════════════════════════════════════════════════
if [ "$BACKEND" = "nim-hosted" ]; then

    # Check 1: Can we reach the NIM API?
    if ! nim_curl "${NIM_BASE}/models" > /dev/null 2>&1; then
        echo "FAIL: Cannot reach NIM hosted API at ${NIM_BASE}"
        echo "      Check NVIDIA_API_KEY and network connectivity."
        exit 1
    fi
    echo "  [OK] NIM hosted API reachable"

    # Check 2: Can it serve inference?
    RESPONSE=$(nim_curl "${NIM_BASE}/chat/completions" \
        -m 20 \
        -H "Content-Type: application/json" \
        -d "{
              \"model\": \"${NIM_MODEL}\",
              \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
              \"max_tokens\": 3
            }" 2>&1)

    if [ $? -ne 0 ]; then
        echo "FAIL: Inference request failed or timed out (20s)"
        exit 1
    fi

    if ! echo "$RESPONSE" | grep -q '"choices"'; then
        echo "FAIL: Inference response malformed — no 'choices' field"
        echo "      Response: ${RESPONSE:0:300}"
        exit 1
    fi
    echo "  [OK] Inference serving  (model: ${NIM_MODEL})"
    echo "OK | backend: nim-hosted | model: ${NIM_MODEL}"
    exit 0
fi

# ═════════════════════════════════════════════════════════════════
# PATH B: NIM Container (self-hosted)
# ═════════════════════════════════════════════════════════════════
if [ "$BACKEND" = "nim-container" ]; then

    CONTAINER_BASE="http://${NIM_HOST}:${NIM_PORT}"

    # Check 1: Container liveness (independent of model load state)
    if ! curl -sf "${CONTAINER_BASE}/v1/health/live" > /dev/null 2>&1; then
        echo "FAIL: NIM container liveness probe failed at ${CONTAINER_BASE}/v1/health/live"
        echo "      Container may not be running. Try: docker ps | grep nim"
        exit 1
    fi
    echo "  [OK] NIM container live"

    # Check 2: Model readiness (model fully loaded into GPU VRAM)
    READY_RESP=$(curl -sf -w "%{http_code}" -o /dev/null \
        "${CONTAINER_BASE}/v1/health/ready" 2>&1)
    if [ "$READY_RESP" != "200" ]; then
        echo "WARN: NIM container not yet ready (HTTP ${READY_RESP}) — model still loading"
        echo "      Large models can take 2–5 minutes to load into GPU VRAM."
        echo "      Re-run health check once loading is complete."
        exit 1
    fi
    echo "  [OK] NIM container ready (model loaded into VRAM)"

    # Check 3: GPU memory in expected range (model loaded ≠ empty VRAM)
    if command -v nvidia-smi &> /dev/null; then
        MEM_USED=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits | head -1 | tr -d ' ')
        MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')

        if [ -n "$MEM_USED" ] && [ -n "$MEM_TOTAL" ] && [ "$MEM_TOTAL" -gt 0 ]; then
            MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))

            if [ "$MEM_PCT" -lt 20 ]; then
                echo "FAIL: GPU VRAM only ${MEM_PCT}% used — model likely not loaded"
                exit 1
            fi
            if [ "$MEM_PCT" -gt 97 ]; then
                echo "WARN: GPU VRAM ${MEM_PCT}% — OOM risk, consider reducing MAX_NUM_SEQS"
                exit 1
            fi
            echo "  [OK] GPU VRAM ${MEM_USED}/${MEM_TOTAL} MiB  (${MEM_PCT}%)"
        fi
    else
        echo "  [SKIP] nvidia-smi not available — skipping VRAM check"
    fi

    # Check 4: GPU temperature safe
    if command -v nvidia-smi &> /dev/null; then
        TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits \
               | head -1 | tr -d ' ')
        if [ -n "$TEMP" ] && [ "$TEMP" -gt 85 ]; then
            echo "WARN: GPU temperature ${TEMP}°C — thermal throttling likely"
            exit 1
        fi
        echo "  [OK] GPU temperature ${TEMP}°C"
    fi

    # Check 5: Can it actually serve inference?
    RESPONSE=$(nim_curl "${CONTAINER_BASE}/v1/chat/completions" \
        -m 20 \
        -H "Content-Type: application/json" \
        -d "{
              \"model\": \"${NIM_MODEL}\",
              \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
              \"max_tokens\": 3
            }" 2>&1)

    if [ $? -ne 0 ]; then
        echo "FAIL: Inference request failed or timed out (20s)"
        exit 1
    fi
    if ! echo "$RESPONSE" | grep -q '"choices"'; then
        echo "FAIL: Inference response malformed — no 'choices' field"
        exit 1
    fi
    echo "  [OK] Inference serving"

    if command -v nvidia-smi &> /dev/null; then
        echo "OK | VRAM: ${MEM_USED}/${MEM_TOTAL} MiB (${MEM_PCT}%) | Temp: ${TEMP}°C | model: ${NIM_MODEL}"
    else
        echo "OK | backend: nim-container | model: ${NIM_MODEL}"
    fi
    exit 0
fi

# ═════════════════════════════════════════════════════════════════
# PATH C: Legacy vLLM (kept for backward compatibility)
# ═════════════════════════════════════════════════════════════════
VLLM_BASE="http://${NIM_HOST}:${NIM_PORT}"

if ! curl -sf "${VLLM_BASE}/health" > /dev/null 2>&1; then
    echo "FAIL: vLLM health endpoint unreachable at ${VLLM_BASE}/health"
    exit 1
fi

if command -v nvidia-smi &> /dev/null; then
    MEM_USED=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits | head -1 | tr -d ' ')
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ -n "$MEM_USED" ] && [ -n "$MEM_TOTAL" ] && [ "$MEM_TOTAL" -gt 0 ]; then
        MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))
        if [ "$MEM_PCT" -lt 30 ]; then
            echo "FAIL: GPU VRAM ${MEM_PCT}% — model likely not loaded"
            exit 1
        fi
        if [ "$MEM_PCT" -gt 97 ]; then
            echo "WARN: GPU VRAM ${MEM_PCT}% — OOM risk"
            exit 1
        fi
    fi

    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ -n "$TEMP" ] && [ "$TEMP" -gt 85 ]; then
        echo "WARN: GPU temperature ${TEMP}°C — thermal throttling likely"
        exit 1
    fi
fi

RESPONSE=$(curl -sf -m 15 "${VLLM_BASE}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${NIM_MODEL}\", \"prompt\": \"test\", \"max_tokens\": 3}" 2>&1)
if [ $? -ne 0 ]; then
    echo "FAIL: vLLM inference request failed or timed out"
    exit 1
fi
if ! echo "$RESPONSE" | grep -q '"choices"'; then
    echo "FAIL: vLLM inference response malformed"
    exit 1
fi

echo "OK | vLLM | VRAM: ${MEM_USED}/${MEM_TOTAL} MiB (${MEM_PCT}%) | Temp: ${TEMP}°C"
exit 0
