#!/usr/bin/env python3
"""Interactive setup wizard for the vLLM GPU inference server.

Flow:
  1. Check prerequisites (Docker, NVIDIA Container Toolkit, GPU visible to Docker)
  2. Detect VRAM (nvidia-smi -> rocm-smi -> sysctl -> /proc/meminfo -> manual)
  3. Pick a model (suggestion / HuggingFace name / local file)
  4. Configure vLLM parameters
  5. Write deploy/.env and launch the server

Run with --dry-run to walk through the wizard without pulling models or
starting containers — useful for verifying the flow in CI or on a machine
without a GPU.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Make `setup.models` / `setup.detect` importable both when invoked as a
# module (`python -m setup.setup`) and as a script (`python setup/setup.py`).
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from setup.detect import detect_vram, MANUAL_HINTS
from setup.models import (
    BLOCKED_ORIGINS,
    POLICY_MSG,
    is_blocked,
    suggestions_for,
)

PROJECT_ROOT = _HERE.parent
ENV_PATH = PROJECT_ROOT / "deploy" / ".env"
COMPOSE_PATH = PROJECT_ROOT / "deploy" / "docker-compose.yml"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"


# ─────────────────────────── IO helpers ───────────────────────────


def banner(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        val = input(f"{prompt}{suffix}: ").strip()
        if val:
            return val
        if default is not None:
            return default


def ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{prompt} {hint}: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def fatal(msg: str) -> None:
    print(f"\nFATAL: {msg}\n", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────── Step 1: prereqs ───────────────────────────


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.returncode, (r.stdout + r.stderr)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


def check_prereqs(dry_run: bool) -> None:
    banner("Step 1: Checking prerequisites")

    if not shutil.which("docker"):
        print("Docker is not installed.")
        print("  macOS:   https://docs.docker.com/desktop/install/mac-install/")
        print("  Linux:   curl -fsSL https://get.docker.com | sh")
        print("  Windows: https://docs.docker.com/desktop/install/windows-install/")
        if dry_run:
            print("  [dry-run] Continuing past missing-Docker error.")
        else:
            fatal("Install Docker, then rerun this wizard.")
    else:
        code, _ = _run(["docker", "info"])
        if code != 0:
            if dry_run:
                print("  [dry-run] Docker daemon is not running — would fatal in a real run.")
            else:
                fatal("Docker is installed but the daemon is not running. Start Docker and rerun.")
        else:
            print("  [OK] Docker is installed and running.")

    # NVIDIA Container Toolkit — only strictly required when an NVIDIA GPU is present.
    has_nvidia_smi = shutil.which("nvidia-smi") is not None
    if has_nvidia_smi:
        code, out = _run(["docker", "info"])
        if "nvidia" not in out.lower():
            print("  [WARN] NVIDIA Container Toolkit may not be installed.")
            print("         Install guide: https://docs.nvidia.com/datacenter/cloud-native/"
                  "container-toolkit/latest/install-guide.html")
            if not dry_run and not ask_yn("  Continue anyway?", default=False):
                fatal("Install NVIDIA Container Toolkit and rerun.")
        else:
            print("  [OK] NVIDIA Container Toolkit detected.")

        if dry_run:
            print("  [dry-run] Skipping 'docker run --gpus all nvidia-smi' check.")
        else:
            print("  Testing GPU visibility from Docker (this pulls a small CUDA image)...")
            code, out = _run([
                "docker", "run", "--rm", "--gpus", "all",
                "nvidia/cuda:12.1.0-base-ubuntu22.04", "nvidia-smi",
            ])
            if code != 0:
                print(out.strip()[-500:])
                print("\nTroubleshooting:")
                print("  - Confirm 'nvidia-smi' works on the host.")
                print("  - Confirm nvidia-container-toolkit is installed and Docker was restarted.")
                print("  - On Linux: sudo systemctl restart docker")
                fatal("Docker cannot see the GPU.")
            print("  [OK] Docker can see the GPU.")
    else:
        print("  [WARN] No nvidia-smi found — skipping NVIDIA toolkit + GPU visibility checks.")
        print("         This machine will not be able to run GPU inference.")


# ─────────────────────────── Step 2: VRAM ───────────────────────────


def detect_or_ask_vram() -> tuple[int, str]:
    banner("Step 2: Detecting VRAM")
    gb, source = detect_vram()
    if gb:
        print(f"  Detected: {source} — {gb} GB")
        return gb, source
    print("  Auto-detection failed.")
    print(MANUAL_HINTS)
    while True:
        raw = ask("Enter available VRAM in GB (integer)")
        try:
            val = int(raw)
            if val > 0:
                return val, "manual entry"
        except ValueError:
            pass
        print("  Please enter a positive integer.")


# ─────────────────────────── Step 3: model selection ───────────────────


def pick_model(vram_gb: int) -> dict:
    banner("Step 3: Select a model")
    print("How do you want to select a model?")
    print("  1. Suggest one based on my VRAM (recommended)")
    print("  2. I know which HuggingFace model I want")
    print("  3. I have a local model file on this machine")
    choice = ask("Choice [1-3]", default="1")
    if choice == "1":
        return _option_suggest(vram_gb)
    if choice == "2":
        return _option_huggingface(vram_gb)
    if choice == "3":
        return _option_local(vram_gb)
    print("  Invalid choice.")
    return pick_model(vram_gb)


def _option_suggest(vram_gb: int) -> dict:
    choices = suggestions_for(vram_gb)
    if not choices:
        print(f"  No suggested model fits in {vram_gb} GB. Falling back to manual entry.")
        return _option_huggingface(vram_gb)

    print(f"\n  Suggested Models ({vram_gb} GB VRAM detected)")
    print("  " + "-" * 58)
    print(f"  {'#':>2}  {'Model':<28} {'VRAM':<6} {'Origin'}")
    print("  " + "-" * 58)
    for i, (_, display, v, origin) in enumerate(choices, 1):
        print(f"  {i:>2}  {display:<28} {v:>3} GB  {origin}")
    print("  " + "-" * 58)
    while True:
        raw = ask(f"Which model? [1-{len(choices)}]", default="1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                break
        except ValueError:
            pass
        print("  Invalid selection.")
    hf_name, display, v, origin = choices[idx]
    print(f"  Selected: {display} ({hf_name}) — {v} GB")
    token = _ask_hf_token()
    return {
        "model_name": hf_name,
        "hf_token": token,
        "local_path": None,
        "model_vram_gb": v,
    }


def _option_huggingface(vram_gb: int) -> dict:
    while True:
        hf_name = ask("Enter the HuggingFace model name "
                      "(e.g. mistralai/Mistral-7B-Instruct-v0.3)")
        if is_blocked(hf_name):
            print(f"  {POLICY_MSG}")
            print(f"  Blocked origins: {', '.join(BLOCKED_ORIGINS)}")
            continue
        break
    token = _ask_hf_token()
    print(f"\n  WARN: We cannot verify VRAM compatibility for custom models.")
    print(f"        Ensure your model fits within {vram_gb} GB.")
    if not ask_yn("  Proceed?", default=True):
        return pick_model(vram_gb)
    return {
        "model_name": hf_name,
        "hf_token": token,
        "local_path": None,
        "model_vram_gb": None,
    }


def _option_local(vram_gb: int) -> dict:
    while True:
        raw = ask("Enter the full path to your model file or directory "
                  "(e.g. /home/user/models/my-model or .../my-model.gguf)")
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            print(f"  Path not found: {p}")
            continue
        if not (p.is_file() or p.is_dir()):
            print(f"  Not a regular file or directory: {p}")
            continue
        break
    print(f"\n  vLLM supports safetensors and GGUF formats.")
    print(f"  Ensure your model fits within {vram_gb} GB VRAM.")
    print(f"\n  The path will be mounted read-only into the container at /model,")
    print(f"  and vLLM will be launched with --model /model.")

    if ask_yn("Is this model proprietary/confidential? (adds path to .gitignore)",
              default=False):
        _add_gitignore(str(p))
    return {
        "model_name": "/model",
        "hf_token": "",
        "local_path": str(p),
        "model_vram_gb": None,
    }


def _ask_hf_token() -> str:
    print("\n  HuggingFace token is needed to download gated models (e.g. Llama 3).")
    print("  Get yours at: https://huggingface.co/settings/tokens")
    return ask("HuggingFace token (press Enter to skip if model is not gated)",
               default="")


def _add_gitignore(path: str) -> None:
    GITIGNORE_PATH.touch(exist_ok=True)
    current = GITIGNORE_PATH.read_text()
    entry = f"\n# Proprietary model path (added by setup wizard)\n{path}\n"
    if path in current:
        print(f"  {path} already in .gitignore")
        return
    GITIGNORE_PATH.write_text(current + entry)
    print(f"  Added to .gitignore: {path}")


# ─────────────────────────── Step 4: vLLM params ───────────────────────


def configure_vllm() -> dict:
    banner("Step 4: vLLM configuration")
    defaults = {
        "GPU_MEM_UTIL": "0.85",
        "MAX_NUM_SEQS": "8",
        "MAX_MODEL_LEN": "4096",
        "DTYPE": "float16",
    }
    print("  GPU Memory Utilization: 0.85 (85% of VRAM)")
    print("  Max Concurrent Requests: 8")
    print("  Max Sequence Length:     4096")
    print("  Data Type:               float16")
    if ask_yn("\n  Use these defaults?", default=True):
        return defaults
    return {
        "GPU_MEM_UTIL": ask(
            "  GPU memory utilization (0.0-1.0): fraction of VRAM vLLM may use",
            default=defaults["GPU_MEM_UTIL"]),
        "MAX_NUM_SEQS": ask(
            "  Max concurrent requests: higher = more throughput, more VRAM",
            default=defaults["MAX_NUM_SEQS"]),
        "MAX_MODEL_LEN": ask(
            "  Max sequence length (prompt + output tokens)",
            default=defaults["MAX_MODEL_LEN"]),
        "DTYPE": ask(
            "  Data type (float16 / bfloat16 / float32)",
            default=defaults["DTYPE"]),
    }


# ─────────────────────────── Step 5: write + launch ────────────────────


def write_env(model: dict, vllm: dict, dry_run: bool) -> None:
    banner("Step 5: Writing configuration")
    lines = [
        "# Generated by setup/setup.py — do not edit by hand.",
        f"MODEL_NAME={model['model_name']}",
        f"HF_TOKEN={model['hf_token']}",
        f"GPU_MEM_UTIL={vllm['GPU_MEM_UTIL']}",
        f"MAX_NUM_SEQS={vllm['MAX_NUM_SEQS']}",
        f"MAX_MODEL_LEN={vllm['MAX_MODEL_LEN']}",
        f"DTYPE={vllm['DTYPE']}",
    ]
    if model["local_path"]:
        lines.append(f"LOCAL_MODEL_PATH={model['local_path']}")
    content = "\n".join(lines) + "\n"

    if dry_run:
        print(f"  [dry-run] Would write {ENV_PATH}:")
        print("  " + content.replace("\n", "\n  "))
        return
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text(content)
    os.chmod(ENV_PATH, 0o600)
    print(f"  Wrote {ENV_PATH}")


def launch(dry_run: bool) -> None:
    banner("Launching server")
    if dry_run:
        print("  [dry-run] Would run: make deploy")
        return
    print("  Running: make deploy")
    subprocess.run(["make", "deploy"], cwd=PROJECT_ROOT, check=False)


# ─────────────────────────── main ───────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive setup wizard.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Walk through the wizard without pulling models or "
                         "starting containers.")
    args = ap.parse_args()

    print("\nGPU Inference Infrastructure — Setup Wizard")
    if args.dry_run:
        print("(DRY RUN — no side effects will be performed)")

    check_prereqs(args.dry_run)
    vram_gb, vram_source = detect_or_ask_vram()
    model = pick_model(vram_gb)
    vllm = configure_vllm()
    write_env(model, vllm, args.dry_run)

    banner("Setup summary")
    print(f"  Model:       {model['model_name']}")
    if model["local_path"]:
        print(f"  Local path:  {model['local_path']} -> /model (ro)")
    print(f"  VRAM budget: {vram_gb} GB ({vram_source})")
    print(f"  vLLM:        mem={vllm['GPU_MEM_UTIL']}  "
          f"seqs={vllm['MAX_NUM_SEQS']}  len={vllm['MAX_MODEL_LEN']}  "
          f"dtype={vllm['DTYPE']}")

    launch(args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
