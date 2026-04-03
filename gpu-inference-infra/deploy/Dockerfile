FROM vllm/vllm-openai:latest

# ── Infra layer on top of vLLM base ──

# Install monitoring dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy health check script
COPY monitoring/health-check.sh /opt/health-check.sh
RUN chmod +x /opt/health-check.sh

# Copy GPU monitor script
COPY monitoring/gpu-monitor.sh /opt/gpu-monitor.sh
RUN chmod +x /opt/gpu-monitor.sh

# Docker health check — runs every 30s, fails after 3 retries
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD /opt/health-check.sh || exit 1

EXPOSE 8000
