#!/bin/bash

# ────────────────────────────────────────────────────────
# gpu-monitor.sh
# Logs GPU metrics to CSV every second.
# Run alongside load tests to capture GPU behavior.
#
# Usage:
#   ./gpu-monitor.sh                    # logs to results/gpu_metrics.csv
#   ./gpu-monitor.sh my_test_run.csv    # logs to results/my_test_run.csv
# ────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$(dirname "$SCRIPT_DIR")/results"
OUTPUT_FILE="${RESULTS_DIR}/${1:-gpu_metrics.csv}"

mkdir -p "$RESULTS_DIR"

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Cannot monitor GPU."
    exit 1
fi

# Write CSV header
echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib,mem_util_pct,temperature_c,power_draw_w" \
    > "$OUTPUT_FILE"

echo "GPU Monitor started. Writing to: $OUTPUT_FILE"
echo "Press Ctrl+C to stop."
echo ""

# Show live GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

trap 'echo ""; echo "Monitor stopped. $(wc -l < "$OUTPUT_FILE") data points collected."; exit 0' INT

while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,utilization.memory,temperature.gpu,power.draw \
        --format=csv,noheader,nounits >> "$OUTPUT_FILE"
    sleep 1
done
