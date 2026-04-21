.PHONY: setup setup-node deploy test monitor health teardown logs clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Interactive setup wizard — detects GPU, picks model, writes .env, launches
	@python3 setup/setup.py

setup-node: ## Bootstrap a fresh GPU machine (Docker, NVIDIA toolkit)
	@bash scripts/setup-gpu-node.sh

deploy: ## Deploy vLLM inference server + Prometheus
	@bash scripts/deploy.sh

test: ## Run load test against the server
	@python3 loadtest/loadtest.py

test-sweep: ## Run load test across all prompt lengths
	@python3 loadtest/loadtest.py --prompt-length short
	@python3 loadtest/loadtest.py --prompt-length medium
	@python3 loadtest/loadtest.py --prompt-length long

monitor: ## Start GPU monitoring daemon (Ctrl+C to stop)
	@bash monitoring/gpu-monitor.sh

health: ## Run health check
	@bash monitoring/health-check.sh

teardown: ## Shut down all services
	@bash scripts/teardown.sh

logs: ## Tail vLLM server logs
	@cd deploy && docker-compose logs -f vllm

analyze: ## Analyze most recent load test results
	@python3 loadtest/analyze.py $$(ls -t results/loadtest_*.json 2>/dev/null | head -1)

clean: ## Stop containers, remove .env and results
	@cd deploy && docker-compose down 2>/dev/null || true
	@rm -f deploy/.env
	@rm -f results/*.csv results/*.json
	@echo "Cleaned: containers stopped, deploy/.env removed, results cleared."
