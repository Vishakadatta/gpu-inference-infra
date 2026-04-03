# Project 1: GPU Inference Infrastructure

## What This Is

You deploy vLLM (an open-source inference engine) as a black box. You don't write the model code. You don't touch tokenization. You build everything AROUND it — the deployment, the monitoring, the profiling, the health checks, the automation, and the operational tooling. This is what an infrastructure engineer actually does.


## Your Role vs Not Your Role

```
YOUR JOB (Infrastructure):              NOT YOUR JOB (ML Engineering):
─────────────────────────────            ─────────────────────────────
Containerize vLLM for deployment         Write the model
Configure GPU resource allocation        Design the architecture
Automate deployment with scripts         Write tokenization code
Monitor GPU health and utilization       Tune model hyperparameters
Build health checks and alerting         Implement attention mechanisms
Profile performance under load           Train or fine-tune models
Find bottlenecks and tune configs        Write inference logic
Detect failures and handle recovery      Build the forward pass
Document operational runbooks
```

Think of it like your Corning work: you didn't write the vCU software. You managed its lifecycle, deployment, containers, networking, and health. Same thing here with vLLM.


## What You Need

- HuggingFace account (free) — to download models that vLLM serves
- GPU rental account (Lambda Labs, RunPod, or Vast.ai)
- Your local machine for writing scripts and Dockerfiles
- GitHub repo: `gpu-inference-infra`


## Phase 0: Understand the Landscape (Day 1)

No code. Just get the mental model right.

### What is vLLM?

An open-source inference engine built by UC Berkeley. It takes a model, loads it onto a GPU, and serves inference requests via an OpenAI-compatible HTTP API. It handles tokenization, batching, memory management, and the actual GPU computation internally. You don't need to understand HOW it does this. You need to understand how to OPERATE it.

It's like nginx or PostgreSQL — you don't write the HTTP parser or the query engine. You deploy it, configure it, monitor it, and keep it running.

### What does vLLM expose?

When running, vLLM gives you:
- An HTTP API at port 8000 (OpenAI-compatible)
- `POST /v1/completions` — send a prompt, get generated text
- `GET /health` — is the server alive
- `GET /metrics` — Prometheus-format performance metrics
- Configurable settings: max batch size, GPU memory utilization, tensor parallelism, quantization

### What GPU to rent and why

NVIDIA A10G (24GB VRAM):
- Cheapest GPU that can run a real model (Llama 3 8B in float16 uses ~16GB)
- Available on Lambda Labs (~$0.75/hr), RunPod (~$0.50/hr), Vast.ai (~$0.30/hr)
- Comes with CUDA pre-installed on most providers
- You'll need ~15-20 hours total across all phases = $5-15

### nvidia-smi — your most important tool

This is the command you'll run more than anything else. It shows:
```
+-----------------------------------------------------------------------------+
| GPU Name        | VRAM Used / Total | GPU Utilization | Temperature        |
|   A10G          | 432MiB / 24576MiB |       0%        |    32C             |
+-----------------------------------------------------------------------------+
```

After vLLM loads a model, VRAM jumps to ~16GB. During inference, GPU utilization spikes. Between requests, it drops. Your job is to understand these patterns and optimize them.


## Phase 1: Deploy vLLM Locally with Docker — No GPU (Days 2-3)

You write the deployment infrastructure on your local machine. vLLM won't actually serve on CPU (it requires GPU), but you get all the Docker and scripting work done so you're not debugging on rented GPU time.

### Step 1.1: Project structure

```
gpu-inference-infra/
├── deploy/
│   ├── Dockerfile                # vLLM container with your config
│   ├── docker-compose.yml        # Full stack: vLLM + monitoring
│   ├── vllm-config.sh            # vLLM launch parameters
│   └── .env.example              # Environment variables template
├── monitoring/
│   ├── gpu-monitor.sh            # nvidia-smi logging daemon
│   ├── health-check.sh           # Checks if vLLM is alive and responsive
│   ├── prometheus.yml            # Prometheus config to scrape vLLM metrics
│   └── alert-rules.yml           # Alert when GPU memory > 90%, utilization drops, etc.
├── loadtest/
│   ├── loadtest.py               # Python script to fire concurrent requests
│   └── analyze-results.py        # Parse load test output into summary stats
├── scripts/
│   ├── setup-gpu-node.sh         # Bootstrap a fresh GPU machine
│   ├── deploy.sh                 # One-command deploy
│   ├── teardown.sh               # Clean shutdown
│   └── rotate-model.sh           # Swap to a different model with zero downtime (stretch goal)
├── results/
│   └── (performance data goes here)
├── docs/
│   ├── architecture.md
│   ├── runbook.md                # Operational runbook: what to do when things break
│   └── tuning-guide.md           # What configs to change and why
├── README.md
└── Makefile                      # make deploy, make test, make monitor, make teardown
```

### Step 1.2: Dockerfile

```dockerfile
FROM vllm/vllm-openai:latest

# Your infra additions on top of the vLLM base image
# Health check script
COPY monitoring/health-check.sh /opt/health-check.sh
RUN chmod +x /opt/health-check.sh

# GPU monitor
COPY monitoring/gpu-monitor.sh /opt/gpu-monitor.sh
RUN chmod +x /opt/gpu-monitor.sh

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD /opt/health-check.sh
```

### Step 1.3: docker-compose.yml

```yaml
version: '3.8'
services:
  vllm:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_NAME=${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}
      - GPU_MEMORY_UTILIZATION=${GPU_MEM_UTIL:-0.85}
      - MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
    command: >
      --model ${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}
      --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85}
      --max-model-len ${MAX_MODEL_LEN:-4096}
      --dtype float16
    volumes:
      - model-cache:/root/.cache/huggingface
      - ./results:/opt/results
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert-rules.yml:/etc/prometheus/alert-rules.yml

volumes:
  model-cache:
```

### Step 1.4: vLLM launch configuration (vllm-config.sh)

This is where your infra decisions live. Each parameter matters:

```bash
#!/bin/bash

# Model to serve
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

# What % of GPU memory vLLM is allowed to use
# 0.85 = 85%. Leave headroom for CUDA overhead.
# Too high = OOM crash. Too low = wasted capacity.
export GPU_MEM_UTIL=0.85

# Maximum sequence length (prompt + response tokens)
# Higher = more memory per request. Lower = limits what users can ask.
export MAX_MODEL_LEN=4096

# Data type: float16 uses half the memory of float32
# with minimal quality loss for inference
export DTYPE="float16"

# Max concurrent requests vLLM will batch together
# Higher = better throughput but more memory. You'll tune this in Phase 3.
export MAX_NUM_SEQS=8
```

### Step 1.5: setup-gpu-node.sh

Bootstrap script for a fresh rented GPU machine:

```bash
#!/bin/bash
set -euo pipefail

echo "=== GPU Node Setup ==="

# Verify GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. This machine has no GPU drivers."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
fi

# Install NVIDIA Container Toolkit (lets Docker access GPU)
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# Install docker-compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "Installing docker-compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
      -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Verify GPU is accessible from Docker
echo "Verifying Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "=== Setup Complete ==="
echo "Run 'make deploy' to start the inference server."
```

### Step 1.6: deploy.sh

```bash
#!/bin/bash
set -euo pipefail

echo "=== Deploying Inference Server ==="

# Load config
source deploy/vllm-config.sh

# Pre-pull model to avoid timeout during docker-compose up
echo "Pre-caching model: $MODEL_NAME"
echo "This may take 5-10 minutes on first run..."

# Record GPU state before deployment
echo "GPU state BEFORE deployment:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# Start services
docker-compose -f deploy/docker-compose.yml up -d

# Wait for vLLM to load the model (this takes a while)
echo "Waiting for vLLM to load model into GPU memory..."
MAX_WAIT=300  # 5 minutes
ELAPSED=0
while ! curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM failed to start within ${MAX_WAIT}s"
        docker-compose -f deploy/docker-compose.yml logs vllm | tail -50
        exit 1
    fi
    echo "  Still loading... (${ELAPSED}s)"
done

echo ""
echo "GPU state AFTER deployment:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "=== Server Ready ==="
echo "API:        http://localhost:8000/v1/completions"
echo "Health:     http://localhost:8000/health"
echo "Metrics:    http://localhost:8000/metrics"
echo "Prometheus: http://localhost:9090"
```

### Step 1.7: Makefile

```makefile
.PHONY: setup deploy test monitor teardown

setup:
	bash scripts/setup-gpu-node.sh

deploy:
	bash scripts/deploy.sh

test:
	python3 loadtest/loadtest.py

monitor:
	bash monitoring/gpu-monitor.sh

health:
	bash monitoring/health-check.sh

teardown:
	bash scripts/teardown.sh

logs:
	docker-compose -f deploy/docker-compose.yml logs -f vllm

all: setup deploy test
```

### Checkpoint after Phase 1:
- All scripts written and tested (syntax, logic)
- Docker setup ready to go
- You have NOT spent any money on GPU yet
- Everything is in git


## Phase 2: Deploy on Real GPU (Days 4-6)

Now you rent a GPU and run it for real.

### Step 2.1: Rent the machine

Go to RunPod or Lambda Labs. Get an A10G with Ubuntu 22.04 + CUDA pre-installed. SSH in.

### Step 2.2: Clone your repo and run setup

```bash
git clone https://github.com/YOUR_USERNAME/gpu-inference-infra.git
cd gpu-inference-infra
make setup
```

If your setup script works first try on a fresh machine, that itself is impressive engineering. If it doesn't, fix it until it does. This is infra work.

### Step 2.3: Deploy

```bash
make deploy
```

Watch two things simultaneously (use tmux or two SSH sessions):

Terminal 1: `make deploy` running
Terminal 2: `watch -n 1 nvidia-smi`

You'll see VRAM climb from ~0 to ~16GB as the model loads. This is the moment you understand what "deploying a model" actually means at the hardware level.

### Step 2.4: Verify it works

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "prompt": "Explain what a GPU does in two sentences.",
    "max_tokens": 100
  }'
```

You should get a response in 1-3 seconds.

### Step 2.5: Start the GPU monitor

```bash
make monitor
```

This runs your gpu-monitor.sh in the background, logging GPU utilization, memory, and temperature every second to a CSV file. Keep this running through all your testing.

### Checkpoint after Phase 2:
- vLLM running on a real GPU, serving requests
- GPU monitor collecting data
- deploy.sh works end-to-end on a fresh machine


## Phase 3: Profile, Load Test, and Tune (Days 7-11)

This is the core infra engineering phase. This is where the project stops being "I deployed a thing" and becomes "I understand how to operate this thing."

### Step 3.1: Build the load tester (loadtest/loadtest.py)

Python script using asyncio + aiohttp that:
- Fires N concurrent requests at the server
- Varies concurrency: 1, 2, 4, 8, 16, 32, 64 concurrent users
- Varies prompt length: short (10 tokens), medium (100 tokens), long (500 tokens)
- Records per-request: latency, tokens generated, time to first token
- Outputs a JSON results file

### Step 3.2: Build the results analyzer (loadtest/analyze-results.py)

Python script that reads the JSON results and computes:
- Throughput: requests/second and tokens/second at each concurrency level
- Latency: p50, p95, p99 at each concurrency level
- GPU utilization: average and peak during each test run
- The breaking point: at what concurrency does the server start failing or degrading

Outputs a clean summary table and generates simple charts (matplotlib).

### Step 3.3: Run the load test matrix

Test these configurations by changing vLLM launch parameters:

```
Config A: gpu-memory-utilization=0.70, max-num-seqs=4
Config B: gpu-memory-utilization=0.85, max-num-seqs=8
Config C: gpu-memory-utilization=0.90, max-num-seqs=16
Config D: gpu-memory-utilization=0.95, max-num-seqs=32
```

For each config, run the full concurrency sweep. Record everything.

You're looking for:
- Which config gives best throughput?
- Which config has lowest latency?
- At what point does increasing max-num-seqs cause OOM?
- What's the GPU utilization at each setting?

### Step 3.4: Find the bottleneck

For each configuration, identify WHERE the bottleneck is:
- Is GPU compute maxed out? (utilization at 100%)
- Is GPU memory full? (VRAM at max, requests getting queued)
- Is the network the limit? (unlikely at this scale but check)
- Is it CPU-side preprocessing? (check CPU usage)

Write this up. "At concurrency 16 with max-num-seqs=8, GPU utilization was 94% and VRAM was at 21.2/24GB. The bottleneck was GPU memory — increasing to max-num-seqs=16 caused OOM at concurrency 32."

This is exactly what a Midjourney infra engineer would need to figure out when tuning their fleet.

### Step 3.5: Build the health check (monitoring/health-check.sh)

Not just "is the port open." A real health check:

```bash
#!/bin/bash

# Check 1: Is the process alive?
if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "FAIL: vLLM health endpoint unreachable"
    exit 1
fi

# Check 2: Is GPU memory in expected range?
MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))

if [ "$MEM_PCT" -lt 50 ]; then
    echo "WARN: GPU memory usage suspiciously low (${MEM_PCT}%). Model may not be loaded."
    exit 1
fi

if [ "$MEM_PCT" -gt 97 ]; then
    echo "WARN: GPU memory critically high (${MEM_PCT}%). OOM risk."
    exit 1
fi

# Check 3: Can it actually serve inference? (not just alive, but functional)
RESPONSE=$(curl -sf -m 10 http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Meta-Llama-3-8B-Instruct", "prompt": "test", "max_tokens": 5}')

if [ $? -ne 0 ]; then
    echo "FAIL: Inference request timed out or failed"
    exit 1
fi

# Check 4: Is GPU temperature safe?
TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
if [ "$TEMP" -gt 85 ]; then
    echo "WARN: GPU temperature ${TEMP}C — throttling likely"
    exit 1
fi

echo "OK: vLLM healthy | VRAM: ${MEM_PCT}% | Temp: ${TEMP}C"
exit 0
```

### Step 3.6: Write the alert rules (monitoring/alert-rules.yml)

Prometheus alert rules for:
- GPU utilization drops to 0 for >60 seconds (server crashed but container alive)
- GPU memory above 95% for >5 minutes (OOM imminent)
- GPU temperature above 83C (thermal throttling)
- vLLM request latency p99 above 10 seconds (degraded performance)
- Health check failures

### Checkpoint after Phase 3:
- Load test data across multiple configurations
- Clear identification of bottlenecks with evidence
- Tuning recommendations backed by real numbers
- Health checking that catches real failure modes
- Alerting rules for operational monitoring


## Phase 4: Document Everything (Days 12-14)

### README.md structure:

```markdown
# GPU Inference Infrastructure

One-command deployment, monitoring, and operational tooling
for GPU-based model serving using vLLM.

## Quick Start
  make setup && make deploy && make test

## What This Is
  (one paragraph: what it does, what problem it solves)

## Architecture
  (diagram showing: GPU node → Docker → vLLM → API)
  (diagram showing: monitoring stack → Prometheus → alerts)

## Performance Results
  (table: concurrency vs throughput vs latency vs GPU util)
  (chart: throughput curve as concurrency increases)
  (chart: GPU memory usage across configurations)
  (the bottleneck analysis: where does it break and why)

## Configuration Tuning Guide
  (what each parameter does, what you found to be optimal, why)

## Operational Runbook
  (what to do when: OOM, high latency, GPU temp warning, health check fail)

## Limitations & What I'd Do at Scale
  (honest: single GPU, single node, no redundancy)
  (what changes with 1000 GPUs: load balancing, model routing,
   node failure handling, rolling updates)

## Cost
  (exact amount you spent, hours of GPU time used)
```

### docs/runbook.md

An operational runbook. This is something most portfolio projects never include, and it's exactly what infra teams actually maintain. Write procedures for:
- Server won't start after deploy
- OOM crash during serving
- GPU temperature throttling
- Model needs to be swapped to a new version
- Performance has degraded — how to diagnose

### docs/tuning-guide.md

Your findings from Phase 3, written as a reference:
- gpu-memory-utilization: what it controls, safe range, your test results
- max-num-seqs: relationship to memory and throughput, your test results
- dtype: float16 vs float32 tradeoffs
- max-model-len: how it affects memory allocation


## Total Cost

| Item | Cost |
|---|---|
| HuggingFace account | Free |
| GPU rental (~15-20 hrs A10G) | $10-15 |
| Docker | Free |
| Prometheus | Free |
| Claude Pro (already have) | $0 |
| **Total** | **$10-15** |


## Total Timeline

| Phase | Days | What |
|---|---|---|
| Phase 0 | Day 1 | Understand GPU basics, no code |
| Phase 1 | Days 2-3 | Write all deployment/monitoring code locally, no GPU |
| Phase 2 | Days 4-6 | Rent GPU, deploy for real, verify everything works |
| Phase 3 | Days 7-11 | Load test, profile, tune, build health checks |
| Phase 4 | Days 12-14 | Document with real data, write runbook and tuning guide |


## Skills You Use vs Skills You Learn

| Already Have (Use Daily) | Learn in This Project |
|---|---|
| Docker (build, compose, networking) | GPU resource management (VRAM, utilization) |
| Python (load testing, analysis scripts) | vLLM operational parameters |
| Bash (deployment automation, monitoring) | nvidia-smi profiling |
| Linux (systemd, processes, debugging) | Inference server failure modes |
| Git (version control, CI/CD) | Prometheus for GPU metrics |


## What Midjourney's Infra Team Would See in This Project

1. You can deploy GPU workloads reliably (setup.sh works on fresh machines)
2. You understand GPU resource constraints (memory, utilization, temperature)
3. You can profile and find bottlenecks with real data (not guesses)
4. You build operational tooling (health checks, monitoring, alerting)
5. You write runbooks (you think about operations, not just deployment)
6. You're honest about limitations and articulate what changes at scale
7. Your code uses Docker, Python, Bash, Linux — the real infra stack
