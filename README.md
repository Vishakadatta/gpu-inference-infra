# GPU Inference Infrastructure

One-command deployment, monitoring, and operational tooling for GPU-based model serving using vLLM.

## What This Is

Infrastructure tooling for deploying and operating [vLLM](https://github.com/vllm-project/vllm) inference servers on GPU hardware. vLLM handles the ML inference internally — this project handles everything else:

- **Automated deployment** — bootstrap a fresh GPU machine to serving requests in one command
- **GPU monitoring** — real-time tracking of VRAM, utilization, temperature via nvidia-smi
- **Health checking** — multi-layer checks (process, GPU memory, inference validation, thermal)
- **Load testing** — async concurrent request testing with latency/throughput analysis
- **Alerting** — Prometheus alert rules for GPU metrics, latency spikes, and failures
- **Operational runbook** — documented procedures for common failure scenarios

## Quick Start

```bash
# On a fresh GPU machine (Ubuntu 22.04 + CUDA)
git clone https://github.com/Vishakadatta/gpu-inference-infra.git
cd gpu-inference-infra

make setup    # Install Docker, NVIDIA Container Toolkit
make deploy   # Pull model, start vLLM + Prometheus
make test     # Run load test suite
make monitor  # Start GPU monitoring daemon
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    GPU Node                          │
│                                                      │
│  ┌────────────────────────────────────────────┐      │
│  │             Docker Compose                  │      │
│  │                                             │      │
│  │  ┌──────────────┐    ┌───────────────┐     │      │
│  │  │     vLLM      │    │  Prometheus    │     │      │
│  │  │               │    │               │     │      │
│  │  │  Port 8000    │◄───│  Port 9090    │     │      │
│  │  │  /v1/complete │    │  scrapes      │     │      │
│  │  │  /health      │    │  /metrics     │     │      │
│  │  │  /metrics     │    │               │     │      │
│  │  └───────┬───────┘    └───────────────┘     │      │
│  │          │                                   │      │
│  └──────────┼───────────────────────────────────┘      │
│             │                                          │
│      ┌──────▼────────┐                                 │
│      │   NVIDIA GPU   │                                 │
│      │   (A10, 24GB)  │                                 │
│      └───────────────┘                                 │
│                                                        │
│  ┌────────────────────────────────────────────┐        │
│  │          Monitoring & Tooling               │        │
│  │                                             │        │
│  │  gpu-monitor.sh  → GPU metrics CSV          │        │
│  │  health-check.sh → Multi-layer validation   │        │
│  │  loadtest.py     → Concurrent load test     │        │
│  │  analyze.py      → Results analysis         │        │
│  └────────────────────────────────────────────┘        │
└────────────────────────────────────────────────────────┘
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
└── Makefile
```

## Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| vLLM as inference engine | Most widely adopted open-source option; PagedAttention + continuous batching |
| Docker, not bare metal | Reproducibility — `make setup && make deploy` works on any GPU machine |
| Prometheus for monitoring | Industry standard; vLLM natively exposes `/metrics` in Prometheus format |
| Bash scripts over Ansible | Single-node deployment — Ansible adds complexity without proportional value here |

See [docs/architecture.md](gpu-inference-infra/docs/architecture.md) for the full rationale.

## Documentation

- **[Architecture & Design](gpu-inference-infra/docs/architecture.md)** — component breakdown, design decisions, and scale considerations
- **[Operational Runbook](gpu-inference-infra/docs/runbook.md)** — what to do when things break (OOM, thermal throttling, degraded latency)
- **[Tuning Guide](gpu-inference-infra/docs/tuning-guide.md)** — configuration parameters, tested values, and results

## Performance Results

_Results will be populated after GPU load testing._

## Limitations

- Single GPU, single node — no multi-node orchestration
- No redundancy or failover
- No model version management or A/B serving
- Load testing is synthetic, not production traffic patterns

## What This Would Need at Scale

| Current | At scale (1000+ GPUs) |
|---------|----------------------|
| Single node | Fleet behind a load balancer |
| One model | Multiple models with routing and memory management |
| Manual deploy | CI/CD pipeline with rolling updates |
| Prometheus on same node | Centralized monitoring with aggregated fleet metrics |
| Bash scripts | Configuration management (Ansible/Puppet/Salt) |
| Docker Compose | Kubernetes with GPU operator and device plugins |
| No redundancy | Automatic failover and request rerouting |
| Static config | Auto-scaling based on request queue depth |
