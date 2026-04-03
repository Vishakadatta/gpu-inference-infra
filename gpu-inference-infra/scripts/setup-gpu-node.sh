#!/bin/bash
set -euo pipefail

# ────────────────────────────────────────────────────────
# setup-gpu-node.sh
# Bootstrap a fresh GPU machine for inference serving.
# Run this once on a new rented GPU node.
# ────────────────────────────────────────────────────────

echo "============================================"
echo "  GPU Node Setup"
echo "============================================"
echo ""

# ── Step 1: Verify GPU is present ──
echo "[1/5] Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "FATAL: nvidia-smi not found."
    echo "This machine either has no GPU or no NVIDIA drivers installed."
    echo "Most GPU rental providers (Lambda, RunPod, Vast.ai) pre-install drivers."
    echo "If you're on a bare machine, install CUDA toolkit first."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

echo "  GPU:    $GPU_NAME"
echo "  VRAM:   $GPU_MEM"
echo "  Driver: $DRIVER_VER"
echo ""

# ── Step 2: Install Docker ──
echo "[2/5] Checking Docker..."
if command -v docker &> /dev/null; then
    DOCKER_VER=$(docker --version | awk '{print $3}' | tr -d ',')
    echo "  Docker already installed: $DOCKER_VER"
else
    echo "  Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "  Docker installed. You may need to log out and back in for group changes."
fi
echo ""

# ── Step 3: Install NVIDIA Container Toolkit ──
echo "[3/5] Checking NVIDIA Container Toolkit..."
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "  NVIDIA Container Toolkit already configured."
else
    echo "  Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")

    # Add NVIDIA GPG key
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null

    # Add repository
    curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "  NVIDIA Container Toolkit installed and configured."
fi
echo ""

# ── Step 4: Install docker-compose ──
echo "[4/5] Checking docker-compose..."
if command -v docker-compose &> /dev/null; then
    COMPOSE_VER=$(docker-compose --version | awk '{print $NF}')
    echo "  docker-compose already installed: $COMPOSE_VER"
elif docker compose version &> /dev/null; then
    echo "  docker compose (plugin) available."
else
    echo "  Installing docker-compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "  docker-compose installed."
fi
echo ""

# ── Step 5: Verify GPU access from Docker ──
echo "[5/5] Verifying Docker can access GPU..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "  Docker GPU access verified."
else
    echo "WARN: Docker GPU test failed. You may need to:"
    echo "  1. Log out and back in (for docker group)"
    echo "  2. Run: sudo systemctl restart docker"
    echo "  3. Check that nvidia-container-toolkit is properly installed"
fi

echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Copy deploy/.env.example to deploy/.env"
echo "  2. Add your HuggingFace token to deploy/.env"
echo "  3. Run: make deploy"
echo ""
