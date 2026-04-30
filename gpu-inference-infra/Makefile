.PHONY: setup setup-node deploy web test test-sweep monitor health teardown logs analyze clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────

setup: ## Interactive wizard — pick NIM hosted API or NIM container, writes deploy/.env
	@python3 setup/setup.py

setup-node: ## Bootstrap a fresh GPU machine (Docker + NVIDIA Container Toolkit)
	@bash scripts/setup-gpu-node.sh

# ── Deployment ────────────────────────────────────────────────────────────────

deploy: ## Pull NIM container + start Prometheus + DCGM Exporter (nim-container path)
	@bash scripts/deploy.sh

web: ## Start the Observatory web UI (runs locally on http://localhost:8080)
	@pip install -q -r requirements.txt
	@echo "Starting Observatory at http://localhost:8080"
	@uvicorn api.server:app --host 0.0.0.0 --port 8080 --reload

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run load test (reads BACKEND + NVIDIA_API_KEY from deploy/.env)
	@set -a && . deploy/.env && set +a && python3 loadtest/loadtest.py

test-sweep: ## Full sweep — short / medium / long prompts at all concurrency levels
	@set -a && . deploy/.env && set +a && \
		python3 loadtest/loadtest.py --prompt-length short  && \
		python3 loadtest/loadtest.py --prompt-length medium && \
		python3 loadtest/loadtest.py --prompt-length long

analyze: ## Analyze the most recent load test results
	@python3 loadtest/analyze.py $$(ls -t results/loadtest_*.json 2>/dev/null | head -1)

# ── Monitoring ────────────────────────────────────────────────────────────────

health: ## Multi-layer health check (nim-hosted: API ping; nim-container: liveness + GPU)
	@set -a && . deploy/.env && set +a && bash monitoring/health-check.sh

monitor: ## Start GPU monitoring daemon — logs VRAM + temp to CSV (Ctrl+C to stop)
	@bash monitoring/gpu-monitor.sh

# ── Operations ────────────────────────────────────────────────────────────────

teardown: ## Shut down all containers cleanly
	@bash scripts/teardown.sh

logs: ## Tail NIM container logs
	@cd deploy && docker-compose logs -f nim

clean: ## Stop containers, remove deploy/.env and results
	@cd deploy && docker-compose down 2>/dev/null || true
	@rm -f deploy/.env
	@rm -f results/*.csv results/*.json
	@echo "Cleaned."
