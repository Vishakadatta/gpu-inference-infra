# GPU Inference Observatory

Deployment, monitoring, and operational tooling for GPU-based LLM inference — powered by NVIDIA NIM.

**[Live Demo →](https://gpu-inference-observatory.onrender.com)**

---

## What This Is

This is an **infrastructure project**, not an AI chatbot.

The project demonstrates everything that surrounds an LLM inference service in production: deployment automation, health checking, load testing, metrics collection, alerting, and operational runbooks. The AI model is a black box being operated — the engineering is in what wraps it.

| Layer | What It Demonstrates |
|---|---|
| **Deployment** | One-command bootstrap from zero to serving requests |
| **Health checking** | Multi-layer validation: liveness, readiness, GPU VRAM, temperature |
| **Load testing** | Concurrency sweeps measuring TTFT, latency, throughput, error rate |
| **Metrics** | Prometheus scraping `vllm:time_to_first_token_seconds`, `vllm:kv_cache_usage_perc`, GPU hardware metrics via DCGM Exporter |
| **Alerting** | Alert rules for latency spikes, KV cache saturation, OOM risk, GPU thermal events |
| **Runbooks** | Documented procedures for OOM crashes, thermal throttling, degraded latency |

---

## Two Backends, One Toolchain

```
make setup → "Do you have a GPU?"
              │
     ┌────────┴──────────┐
     │                   │
NIM Hosted API       NIM Container
(free, no GPU)       (your GPU, full metrics)
     │                   │
     └────────┬───────────┘
              │
   Same tools work against both:
   ├── loadtest.py      (TTFT, latency, throughput)
   ├── health-check.sh  (liveness + readiness + GPU checks)
   ├── prometheus.yml   (scrapes /v1/metrics)
   └── alert-rules.yml  (Prometheus alerting rules)
```

### NIM Hosted API (default — no GPU required)
Calls NVIDIA's hosted inference endpoint at `integrate.api.nvidia.com`.
Free tier: 1,000 API credits on signup.
Client-side metrics: TTFT, total latency, tokens/sec, error rate.
Use this to run the live demo and load tests without hardware.

### NIM Container (self-hosted — full GPU metrics)
Runs `nvcr.io/nim/meta/llama-3.1-8b-instruct:latest` on your GPU.
Exposes `/v1/health/live`, `/v1/health/ready`, `/v1/metrics` (Prometheus-native).
Add DCGM Exporter as a sidecar → GPU VRAM, utilisation %, temperature, power draw.
This is the path that shows the full infrastructure story.

---

## Quick Start

### Option A — No GPU (NIM Hosted API)
```bash
git clone https://github.com/Vishakadatta/gpu-inference-infra.git
cd gpu-inference-infra

pip install -r requirements.txt
make setup          # choose option 1 (NIM Hosted API), paste your free nvapi- key
make health         # verify NIM connectivity
make test           # run load test — short prompts, concurrency sweep 1→16
make web            # start the observatory UI at http://localhost:8080
```

Get a free NVIDIA API key at [build.nvidia.com](https://build.nvidia.com).

### Option B — GPU Available (NIM Container, full metrics)
```bash
# On Ubuntu 22.04 + CUDA machine
make setup-node     # install Docker + NVIDIA Container Toolkit
make setup          # choose option 2 (NIM Container)
make deploy         # pull NIM container + start Prometheus + DCGM Exporter
make health         # wait for model to load (~3 min), then validate all checks
make test           # full load test sweep — results saved to results/
make monitor        # start GPU metrics daemon
```

---

## Architecture

### NIM Container path (full metrics)

```
┌──────────────────────────────────────────────────────────┐
│                        GPU Node                           │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                   Docker Compose                     │  │
│  │                                                      │  │
│  │  ┌────────────────┐  ┌──────────────┐  ┌─────────┐  │  │
│  │  │   NIM Container │  │  Prometheus  │  │  DCGM   │  │  │
│  │  │                │  │              │  │Exporter │  │  │
│  │  │  Port 8000     │◄─│  Port 9090   │  │Port 9400│  │  │
│  │  │  /v1/chat/...  │  │  /v1/metrics │  │GPU HW   │  │  │
│  │  │  /v1/health/*  │  │  alert-rules │  │metrics  │  │  │
│  │  │  /v1/metrics   │  │              │  │         │  │  │
│  │  └───────┬────────┘  └──────────────┘  └─────────┘  │  │
│  │          │                                            │  │
│  └──────────┼────────────────────────────────────────────┘  │
│             │                                               │
│      ┌──────▼────────┐                                      │
│      │  NVIDIA GPU    │  VRAM, utilisation, temp via DCGM   │
│      │  (A10G, 24GB)  │  KV cache, TTFT, queue via /metrics │
│      └───────────────┘                                      │
└──────────────────────────────────────────────────────────────┘
```

### Key metrics exposed by NIM

| Metric | What it shows |
|---|---|
| `vllm:time_to_first_token_seconds` | TTFT histogram — p95/p99 SLO tracking |
| `vllm:e2e_request_latency_seconds` | Full round-trip latency |
| `vllm:num_requests_running` | Active requests in flight |
| `vllm:num_requests_waiting` | Queue backlog — fires alert at >20 |
| `vllm:kv_cache_usage_perc` | Memory pressure (0–1) — fires alert at >0.9 |
| `vllm:generation_tokens_total` | Throughput (tokens/sec as rate) |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature — fires alert at >85°C |
| `DCGM_FI_DEV_FB_USED` | GPU VRAM used — fires alert at >95% |

---

## Project Structure

```
gpu-inference-infra/
├── api/
│   └── server.py               # FastAPI backend (TTFT measurement, load test API)
├── frontend/
│   └── index.html              # Observatory dashboard UI
├── setup/
│   ├── setup.py                # Interactive wizard (NIM hosted vs container)
│   ├── nim_discover.py         # Live model discovery from NIM /v1/models
│   └── models.py               # NIM model registry + publisher allowlist
├── deploy/
│   ├── docker-compose.yml      # NIM container + Prometheus + DCGM Exporter
│   └── .env.example            # Environment variables template
├── monitoring/
│   ├── health-check.sh         # Multi-layer health validation (both backends)
│   ├── gpu-monitor.sh          # nvidia-smi logging daemon
│   ├── prometheus.yml          # Scrape config (NIM /v1/metrics + DCGM)
│   └── alert-rules.yml         # Prometheus alert rules
├── loadtest/
│   ├── loadtest.py             # Async concurrent load tester (NIM + vLLM)
│   └── analyze.py              # Results parser and summary
├── scripts/
│   ├── setup-gpu-node.sh       # Bootstrap fresh GPU machine
│   ├── deploy.sh               # One-command deploy
│   └── teardown.sh             # Clean shutdown
├── results/                    # Load test results (generated)
├── docs/
│   ├── architecture.md         # Design decisions
│   ├── runbook.md              # Operational procedures
│   └── tuning-guide.md         # Configuration parameters + results
├── requirements.txt
└── Makefile
```

---

## Key Design Decisions

| Decision | Reasoning |
|---|---|
| NVIDIA NIM over raw vLLM | NIM exposes `/v1/health/live`, `/v1/health/ready`, and `/v1/metrics` natively — no instrumentation code needed. Stronger ops story. |
| Two backends (hosted + container) | Hosted API makes the project runnable by anyone for free; container path shows the full GPU metrics story |
| DCGM Exporter as a sidecar | Standard NVIDIA-provided container for GPU hardware metrics — the right way to get VRAM/temp/utilisation into Prometheus |
| Docker Compose over Kubernetes | Single-node deployment — Kubernetes adds orchestration complexity not warranted for one GPU |
| Prometheus alert rules | `kv_cache_usage_perc > 0.9` and `num_requests_waiting > 20` are the real operational triggers — not generic HTTP alerts |
| FastAPI for the web backend | Async, minimal, same process handles streaming NIM calls + static file serving |

---

## Performance Results

_To be populated after GPU load testing (one A10G rental, ~$12)._

Run `make test-sweep` on a GPU node to generate real results in `results/`.

---

## What This Would Need at Scale

| Current | At scale (1000+ GPUs) |
|---|---|
| Single node | Fleet behind a load balancer |
| One model | Multiple models with routing and memory management |
| Manual deploy | CI/CD pipeline with rolling NIM container updates |
| Prometheus on same node | Centralised monitoring with aggregated fleet metrics |
| Docker Compose | Kubernetes with GPU operator, device plugins, KEDA autoscaler |
| No redundancy | Automatic failover, request rerouting on node failure |
| Static config | Autoscaling on `vllm:num_requests_waiting` queue depth |
| DCGM Exporter per node | Fleet-wide GPU telemetry with Grafana dashboards |
