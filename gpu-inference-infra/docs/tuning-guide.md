# Configuration Tuning Guide

Results from load testing different vLLM configurations on [GPU MODEL]. Fill this in after running Phase 3 load tests.

## Parameters Tested

### gpu-memory-utilization

Controls what percentage of GPU VRAM vLLM is allowed to use.

| Value | VRAM Used | Behavior |
|-------|-----------|----------|
| 0.70  | _TBD_     | _TBD_    |
| 0.85  | _TBD_     | _TBD_    |
| 0.90  | _TBD_     | _TBD_    |
| 0.95  | _TBD_     | _TBD_    |

**Finding:** _TBD after testing_

### max-num-seqs

Controls maximum number of requests batched together.

| Value | Throughput (req/s) | p95 Latency (ms) | GPU Util % | Notes |
|-------|--------------------|-------------------|------------|-------|
| 4     | _TBD_              | _TBD_             | _TBD_      |       |
| 8     | _TBD_              | _TBD_             | _TBD_      |       |
| 16    | _TBD_              | _TBD_             | _TBD_      |       |
| 32    | _TBD_              | _TBD_             | _TBD_      |       |

**Finding:** _TBD after testing_

### max-model-len

Controls maximum sequence length (prompt + output tokens).

| Value | Memory Impact | Use Case |
|-------|---------------|----------|
| 2048  | _TBD_         | Short responses |
| 4096  | _TBD_         | Standard |
| 8192  | _TBD_         | Long-form generation |

**Finding:** _TBD after testing_

## Recommended Configuration

_To be filled after completing load test matrix._

## Bottleneck Analysis

_For each configuration, document where the bottleneck was:_

- GPU compute bound (utilization at 100%)
- GPU memory bound (VRAM at max, requests queuing)
- Network bound (unlikely at single node)
- CPU preprocessing bound (check CPU usage)
