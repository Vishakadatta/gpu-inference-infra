# Architecture & Design Decisions

## Overview

This project provides infrastructure tooling for deploying and operating a vLLM inference server on GPU hardware. vLLM handles the ML inference (model loading, tokenization, batching, generation). This project handles everything else: deployment, monitoring, health checking, load testing, and operational procedures.

## Design Decisions

### Why vLLM?

vLLM is the most widely adopted open-source inference engine. It implements PagedAttention for efficient GPU memory management and continuous batching for high throughput. Using it as a black box lets this project focus on the infrastructure layer — the same separation of concerns you'd see in production (ML team builds models, infra team deploys and operates them).

### Why Docker, not bare metal?

Reproducibility. `make setup && make deploy` should work on any GPU machine with Ubuntu and NVIDIA drivers. Docker isolates the vLLM runtime, its CUDA dependencies, and the Python environment from the host. This also mirrors real production setups where inference servers run in containers orchestrated by K8s or similar.

### Why Prometheus for monitoring?

Industry standard for infrastructure monitoring. vLLM natively exposes Prometheus-format metrics at `/metrics`. Adding Prometheus to the stack is one line in docker-compose and gives us time-series data, alerting, and (optionally) Grafana dashboards — all free and widely understood.

### Why bash scripts instead of Ansible/Terraform?

Simplicity. This is a single-node deployment. Ansible adds complexity without proportional value at this scale. The bash scripts are readable, self-contained, and easy to debug. If this were a multi-node fleet, Ansible or Terraform would be the right choice.

## Component Breakdown

### deploy/
- **Dockerfile** — Extends official vLLM image with health check and monitoring scripts
- **docker-compose.yml** — Defines the full stack (vLLM + Prometheus) with GPU resource reservations
- **vllm-config.sh** — Documented configuration parameters with explanations of each tuning knob

### monitoring/
- **gpu-monitor.sh** — nvidia-smi polling daemon that logs GPU metrics to CSV for post-analysis
- **health-check.sh** — Four-layer health validation (process alive, GPU memory sane, inference functional, temperature safe)
- **alert-rules.yml** — Prometheus alerts for latency spikes, queue backlog, and server down

### loadtest/
- **loadtest.py** — Async Python load tester with configurable concurrency sweeps and prompt lengths
- **analyze.py** — Results parser that computes p50/p95/p99 latency, throughput, and error rates per concurrency level

### scripts/
- **setup-gpu-node.sh** — Bootstrap: verifies GPU, installs Docker and NVIDIA Container Toolkit
- **deploy.sh** — Orchestrates the full deployment with pre-flight checks, progress monitoring, and verification
- **teardown.sh** — Clean shutdown with post-teardown GPU state verification

## What This Doesn't Cover (and What Would Change at Scale)

| This project | At scale (e.g. 1000+ GPUs) |
|---|---|
| Single node | Fleet of GPU nodes behind a load balancer |
| One model | Multiple models with routing and memory management |
| Manual deploy | CI/CD pipeline with rolling updates |
| Prometheus on same node | Centralized monitoring with aggregated fleet metrics |
| Bash scripts | Configuration management (Ansible/Puppet/Salt) |
| Docker Compose | Kubernetes with GPU operator and device plugins |
| No redundancy | Automatic failover and request rerouting |
| Static config | Auto-scaling based on request queue depth |
