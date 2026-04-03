#!/bin/bash
set -euo pipefail

# ────────────────────────────────────────────────────────
# deploy.sh
# Deploy vLLM inference server + monitoring stack.
# Assumes setup-gpu-node.sh has already been run.
# ────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="$PROJECT_DIR/deploy"

echo "============================================"
echo "  Deploying Inference Server"
echo "============================================"
echo ""

# ── Pre-flight checks ──
if [ ! -f "$DEPLOY_DIR/.env" ]; then
    echo "FATAL: deploy/.env not found."
    echo "Run: cp deploy/.env.example deploy/.env"
    echo "Then add your HuggingFace token."
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "FATAL: nvidia-smi not found. Run setup-gpu-node.sh first."
    exit 1
fi

# Source config for display
source "$DEPLOY_DIR/vllm-config.sh"

echo "Configuration:"
echo "  Model:         $MODEL_NAME"
echo "  GPU Mem Util:  $GPU_MEM_UTIL"
echo "  Max Seq Len:   $MAX_MODEL_LEN"
echo "  Max Batch:     $MAX_NUM_SEQS"
echo "  Dtype:         $DTYPE"
echo ""

# ── Record GPU state before deploy ──
echo "GPU state BEFORE deployment:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader
echo ""

# ── Start services ──
echo "Starting services..."
cd "$DEPLOY_DIR"
docker-compose up -d --build

# ── Wait for vLLM to load model ──
echo ""
echo "Waiting for vLLM to load model into GPU memory..."
echo "(This takes 1-5 minutes depending on model size and download speed)"
echo ""

MAX_WAIT=600  # 10 minutes — first run needs to download the model
ELAPSED=0
LAST_MEM=""

while ! curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo ""
        echo "FATAL: vLLM failed to start within ${MAX_WAIT}s"
        echo ""
        echo "Last 50 lines of vLLM logs:"
        docker-compose logs --tail=50 vllm
        exit 1
    fi

    # Show GPU memory progress
    CURRENT_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "?")
    if [ "$CURRENT_MEM" != "$LAST_MEM" ]; then
        echo "  [${ELAPSED}s] GPU memory: ${CURRENT_MEM} MiB"
        LAST_MEM="$CURRENT_MEM"
    fi
done

echo ""
echo "GPU state AFTER deployment:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader

# ── Verify inference works ──
echo ""
echo "Verifying inference..."
RESPONSE=$(curl -sf -m 30 http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL_NAME\", \"prompt\": \"Hello\", \"max_tokens\": 5}" 2>&1) || true

if echo "$RESPONSE" | grep -q "choices"; then
    echo "  Inference verified — server is generating text."
else
    echo "  WARN: Inference verification failed. Server may still be warming up."
    echo "  Response: $RESPONSE"
fi

echo ""
echo "============================================"
echo "  Server Ready"
echo "============================================"
echo ""
echo "  API:         http://localhost:8000/v1/completions"
echo "  Health:      http://localhost:8000/health"
echo "  Metrics:     http://localhost:8000/metrics"
echo "  Prometheus:  http://localhost:9090"
echo ""
echo "  Try it:  curl http://localhost:8000/v1/completions \\"
echo "             -H 'Content-Type: application/json' \\"
echo "             -d '{\"model\": \"$MODEL_NAME\", \"prompt\": \"What is a GPU?\", \"max_tokens\": 100}'"
echo ""
