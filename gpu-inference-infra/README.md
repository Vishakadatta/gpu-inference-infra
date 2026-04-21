# GPU Inference Infrastructure

One-command deployment, monitoring, and operational tooling for GPU-based model serving using vLLM.

## What This Is

Infrastructure tooling for deploying and operating vLLM inference servers on GPU hardware. This project covers:

- **Automated deployment** — bootstrap a fresh GPU machine to serving requests in one command
- **GPU monitoring** — real-time tracking of VRAM, utilization, temperature via nvidia-smi
- **Health checking** — multi-layer checks (process, GPU memory, inference validation, thermal)
- **Load testing** — configurable concurrent request testing with latency/throughput analysis
- **Configuration tuning** — tested parameter combinations with documented results
- **Operational runbook** — procedures for common failure scenarios

## Quick Start

```bash
# On a fresh GPU machine (Ubuntu 22.04 + CUDA)
git clone https://github.com/Vishakadatta/gpu-inference-infra.git
cd gpu-inference-infra

make setup-node  # One-time: install Docker + NVIDIA Container Toolkit
make setup       # Interactive wizard — picks a model, writes .env, launches
make test        # Run load test suite
make monitor     # Start GPU monitoring daemon
```

## Setup Wizard

`make setup` runs an interactive wizard (`setup/setup.py`) so you never have
to hand-edit `.env`. It:

1. Checks Docker, the NVIDIA Container Toolkit, and GPU visibility from Docker.
2. Detects available VRAM (`nvidia-smi` → `rocm-smi` → Apple `sysctl` →
   `/proc/meminfo` → manual entry).
3. Asks how you want to pick a model:
   - **Suggest one based on VRAM** — chooses from a curated list of models
     from Meta, Mistral AI, Google, or Microsoft that fit your budget.
   - **HuggingFace name** — you type `org/model`. The wizard enforces a
     vendor allowlist: Qwen, DeepSeek, Yi, THUDM, 01-ai, InternLM, Baichuan,
     and ChatGLM are rejected.
   - **Local model file** — point at a `.safetensors` directory or `.gguf`
     file. The path is mounted read-only into the container at `/model`,
     and vLLM is launched with `--model /model`. If you mark the model
     proprietary, the path is added to `.gitignore`.
4. Configures vLLM parameters (GPU mem util, max concurrent, max length, dtype).
5. Writes `deploy/.env` and runs `make deploy`.

Pass `--dry-run` to walk through the wizard without pulling models or
starting containers:

```bash
python3 setup/setup.py --dry-run
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   GPU Node                       │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │            Docker Compose                 │    │
│  │                                           │    │
│  │  ┌─────────────┐    ┌──────────────┐     │    │
│  │  │    vLLM      │    │  Prometheus   │     │    │
│  │  │              │    │              │     │    │
│  │  │  Port 8000   │◄──│  Port 9090   │     │    │
│  │  │  /v1/complete│    │  scrapes     │     │    │
│  │  │  /health     │    │  /metrics    │     │    │
│  │  │  /metrics    │    │              │     │    │
│  │  └──────┬───────┘    └──────────────┘     │    │
│  │         │                                  │    │
│  └─────────┼──────────────────────────────────┘    │
│            │                                       │
│     ┌──────▼───────┐                               │
│     │   NVIDIA GPU  │                               │
│     │   (A10, 24GB) │                               │
│     └──────────────┘                               │
│                                                    │
│  ┌──────────────────────────────────────────┐      │
│  │         Monitoring & Tooling              │      │
│  │                                           │      │
│  │  gpu-monitor.sh  → GPU metrics CSV        │      │
│  │  health-check.sh → Multi-layer validation │      │
│  │  loadtest.py     → Concurrent load test   │      │
│  │  analyze.py      → Results analysis       │      │
│  └──────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
gpu-inference-infra/
├── deploy/
│   ├── Dockerfile              # vLLM container with health check
│   ├── docker-compose.yml      # Full stack: vLLM + Prometheus
│   ├── vllm-config.sh          # vLLM launch parameters
│   └── .env.example            # Environment variables template
├── monitoring/
│   ├── gpu-monitor.sh          # nvidia-smi logging daemon
│   ├── health-check.sh         # Multi-layer health validation
│   ├── prometheus.yml          # Prometheus scrape config
│   └── alert-rules.yml         # GPU/inference alert rules
├── loadtest/
│   ├── loadtest.py             # Async concurrent load tester
│   └── analyze.py              # Results parser and summary
├── scripts/
│   ├── setup-gpu-node.sh       # Bootstrap fresh GPU machine
│   ├── deploy.sh               # One-command deploy
│   └── teardown.sh             # Clean shutdown
├── results/                    # Performance data (generated)
├── docs/
│   ├── architecture.md         # Design decisions
│   ├── runbook.md              # Operational procedures
│   └── tuning-guide.md         # Configuration tuning results
├── Makefile
└── README.md
```

## Performance Results

_Results will be populated after GPU testing in Phase 3._

## Limitations

- Single GPU, single node — no multi-node orchestration
- No redundancy or failover
- No model version management or A/B serving
- Load testing is synthetic, not production traffic patterns

## What This Would Need at Scale

- Load balancer distributing requests across GPU fleet
- Model routing: different model versions on different GPUs
- Node failure detection and automatic request rerouting
- Rolling updates: swap models without dropping requests
- GPU memory management across multiple models per node
- Fleet-wide monitoring and alerting aggregation
- Automated scaling based on request queue depth

## Cost to Reproduce

| Item | Cost |
|------|------|
| GPU rental (~15-20 hrs A10G) | $10-15 |
| Everything else | Free |
