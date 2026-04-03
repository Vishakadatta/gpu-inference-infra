# Operational Runbook

Procedures for common failure scenarios when operating the GPU inference server.

## Server Won't Start After Deploy

**Symptoms:** `make deploy` hangs at "Waiting for vLLM to load model..."

**Steps:**
1. Check vLLM logs: `make logs`
2. Common causes:
   - **HuggingFace token missing or invalid:** Check `deploy/.env` has a valid `HF_TOKEN`
   - **Model requires access approval:** Visit the model page on HuggingFace and accept the license
   - **Insufficient VRAM:** Run `nvidia-smi` — if total VRAM < 16GB, use a quantized model
   - **Docker can't access GPU:** Run `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
3. Fix the issue, then `make teardown && make deploy`

## OOM Crash During Serving

**Symptoms:** vLLM container exits, GPU memory was at 95%+ before crash

**Steps:**
1. Check logs for OOM message: `make logs | grep -i "out of memory"`
2. Reduce memory pressure — edit `deploy/.env`:
   - Lower `GPU_MEM_UTIL` (try 0.80)
   - Lower `MAX_NUM_SEQS` (try 4)
   - Lower `MAX_MODEL_LEN` (try 2048)
3. Redeploy: `make teardown && make deploy`
4. Re-run load test at lower concurrency to verify stability

## GPU Temperature Throttling

**Symptoms:** Health check reports temp > 85C, inference latency increases

**Steps:**
1. Check current temp: `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader`
2. Reduce load temporarily — lower concurrency in load test
3. Check if the GPU node has adequate cooling (datacenter/cloud provider issue)
4. If persistent, reduce `MAX_NUM_SEQS` to lower GPU utilization

## Performance Has Degraded

**Symptoms:** Latency increased, throughput dropped compared to baseline

**Steps:**
1. Run health check: `make health`
2. Check GPU state: `nvidia-smi` — look for:
   - GPU utilization stuck at 100% (compute bound)
   - Memory near max (memory bound — requests queuing)
   - Temperature > 80C (thermal throttling)
3. Check if another process is using the GPU: `nvidia-smi` shows all processes
4. Compare current config against baseline results in `docs/tuning-guide.md`
5. If all looks normal, restart: `make teardown && make deploy`

## Model Needs to be Swapped

**Steps:**
1. Edit `deploy/.env` — change `MODEL_NAME` to the new model
2. Teardown: `make teardown`
3. Deploy: `make deploy` (will download new model on first run)
4. Run health check: `make health`
5. Run quick load test to baseline the new model: `make test`
