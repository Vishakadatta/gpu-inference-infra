#!/bin/bash
set -euo pipefail

# ────────────────────────────────────────────────────────
# teardown.sh
# Clean shutdown of inference server and monitoring stack.
# ────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")/deploy"

echo "Shutting down inference server..."

cd "$DEPLOY_DIR"
docker-compose down

echo ""
echo "GPU state after teardown:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

echo ""
echo "Teardown complete."
echo ""
echo "Note: Model cache is preserved in Docker volume 'deploy_model-cache'."
echo "To remove cached models:  docker volume rm deploy_model-cache"
