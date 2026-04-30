#!/usr/bin/env python3
"""Interactive setup wizard for the GPU Inference Observatory.

First question is always backend selection:
  1. NIM Hosted API  — NVIDIA runs the GPU, you call their endpoint (free, no hardware)
  2. NIM Container   — you run the NIM Docker container on your own GPU

Both paths write deploy/.env and are ready to serve immediately after setup.

Run with --dry-run to walk through without making API calls or starting containers.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

import httpx
from rich.console import Console
from rich.prompt import Confirm, Prompt

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from setup.models import (
    NIM_BASE,
    NIM_DEFAULT_MODEL,
    container_image,
)
from setup.nim_discover import discover_and_pick

console = Console()

PROJECT_ROOT   = _HERE.parent
ENV_PATH       = PROJECT_ROOT / "deploy" / ".env"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="GPU Inference Observatory — setup wizard")
    ap.add_argument("--dry-run", action="store_true",
                    help="Walk through without making API calls or starting containers.")
    args = ap.parse_args()

    console.rule("[bold cyan]GPU Inference Observatory — Setup[/bold cyan]")
    if args.dry_run:
        console.print("[yellow](DRY RUN — no side effects)[/yellow]\n")

    console.print(
        "\n[bold]Which backend do you want to use?[/bold]\n\n"
        "  [cyan]1.[/cyan]  NIM Hosted API   — NVIDIA runs the GPU for you  "
        "(free, no hardware needed)\n"
        "  [cyan]2.[/cyan]  NIM Container    — run the NIM container on your own GPU  "
        "(full GPU metrics)\n"
    )
    choice = Prompt.ask("Choice", choices=["1", "2"], default="1")

    if choice == "1":
        return _run_hosted(args.dry_run)
    return _run_container(args.dry_run)


# ---------------------------------------------------------------------------
# Path 1: NIM Hosted API
# ---------------------------------------------------------------------------

def _run_hosted(dry_run: bool) -> int:
    console.print(
        "\n[bold]NIM Hosted API[/bold]\n"
        "NVIDIA runs powerful models on their DGX Cloud — you call their endpoint.\n"
        "Free tier: 1,000 credits on signup (up to 5,000 on request). No GPU needed.\n\n"
        "To get your free API key:\n"
        "  1. Go to [cyan]https://build.nvidia.com[/cyan]\n"
        "  2. Sign in or create a free account\n"
        "  3. Click your profile icon → [bold]API Keys → Generate API Key[/bold]\n"
        "  4. Copy the key  (starts with [bold]nvapi-[/bold])\n"
    )

    api_key = _prompt_and_validate_key(dry_run)
    if not api_key:
        return 1

    if dry_run:
        model = NIM_DEFAULT_MODEL
        console.print(f"  [dim][dry-run] Would discover models. Using default: {model}[/dim]")
    else:
        model = discover_and_pick(api_key)
        if not model:
            return 1

    _write_env_hosted(api_key, model, dry_run)

    console.rule("[green]Setup complete[/green]")
    console.print(f"  Backend:  NIM Hosted API")
    console.print(f"  Model:    {model}")
    console.print(f"  Endpoint: {NIM_BASE}")
    console.print(
        "\n  Next steps:\n"
        "    [bold]make test[/bold]    — load test against NIM hosted API\n"
        "    [bold]make health[/bold]  — verify NIM connectivity\n"
        "    [bold]make web[/bold]     — start the observatory web UI\n"
        "\n  [dim]deploy/.env holds your API key — never commit it.[/dim]"
    )
    return 0


# ---------------------------------------------------------------------------
# Path 2: NIM Container (self-hosted GPU)
# ---------------------------------------------------------------------------

def _run_container(dry_run: bool) -> int:
    console.print(
        "\n[bold]NIM Container  (self-hosted)[/bold]\n"
        "Run the NVIDIA NIM Docker container on your own GPU.\n"
        "Gives you [bold]full GPU metrics[/bold]: "
        "VRAM, temperature, KV cache, queue depth.\n\n"
        "Requirements:\n"
        "  · NVIDIA GPU  (A10G 24 GB minimum for 8B models)\n"
        "  · Docker + NVIDIA Container Toolkit installed\n"
        "  · Free NGC API key from [cyan]https://build.nvidia.com[/cyan]\n"
    )

    api_key = _prompt_and_validate_key(dry_run)
    if not api_key:
        return 1

    # VRAM detection
    from setup.detect import detect_vram
    vram_gb, vram_source = detect_vram()
    if vram_gb:
        console.print(f"\n  Detected: {vram_source} — [bold]{vram_gb} GB[/bold]")
    else:
        console.print(f"\n  [yellow]Could not auto-detect VRAM.[/yellow]")
        vram_gb = int(Prompt.ask("  Enter available VRAM in GB", default="24"))

    # Model discovery — same live catalogue as hosted path
    if dry_run:
        model = NIM_DEFAULT_MODEL
        console.print(f"  [dim][dry-run] Using default model: {model}[/dim]")
    else:
        model = discover_and_pick(api_key)
        if not model:
            return 1

    image = container_image(model)
    console.print(f"\n  Container image: [bold]{image}[/bold]")

    if not dry_run:
        console.print(
            "\n  [yellow]IMPORTANT:[/yellow] You must accept the NIM container license "
            "on NGC before the docker pull will succeed.\n"
            "  Visit [cyan]https://catalog.ngc.nvidia.com/orgs/nim/[/cyan], "
            "find your model, and click 'Agree to License'.\n"
        )
        if not Confirm.ask("  Have you accepted the license?", default=False):
            console.print(
                "\n  Accept the license first, then re-run setup.\n"
            )
            return 1

    _write_env_container(api_key, model, vram_gb, dry_run)

    console.rule("[green]Setup complete[/green]")
    console.print(f"  Backend:  NIM Container (self-hosted)")
    console.print(f"  Model:    {model}")
    console.print(f"  Image:    {image}")
    console.print(f"  VRAM:     {vram_gb} GB")
    console.print(
        "\n  Next steps:\n"
        "    [bold]make deploy[/bold]   — pull NIM container + start Prometheus + DCGM\n"
        "    [bold]make health[/bold]   — wait for model load, validate all checks\n"
        "    [bold]make test[/bold]     — full load test sweep  (saves to results/)\n"
        "    [bold]make monitor[/bold]  — start GPU metrics daemon\n"
        "\n  [dim]deploy/.env holds your NGC key — never commit it.[/dim]"
    )
    return 0


# ---------------------------------------------------------------------------
# Shared: API key prompt + validation
# ---------------------------------------------------------------------------

def _prompt_and_validate_key(dry_run: bool) -> str | None:
    if dry_run:
        console.print("  [dim][dry-run] Skipping key prompt + validation.[/dim]")
        return "nvapi-dry-run-placeholder"

    while True:
        key = Prompt.ask(
            "  Paste your API key  (or press Enter to open browser)"
        ).strip()

        if not key:
            console.print("  Opening [cyan]https://build.nvidia.com[/cyan] in your browser…")
            webbrowser.open("https://build.nvidia.com")
            continue

        if not key.startswith("nvapi-"):
            console.print(
                "  [yellow]Key should start with 'nvapi-'. "
                "Double-check and try again.[/yellow]"
            )
            continue

        console.print("  Validating key…")
        try:
            r = httpx.get(
                f"{NIM_BASE}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=15.0,
            )
            if r.status_code == 200:
                console.print("  [green]Key valid ✓[/green]")
                return key
            if r.status_code in (401, 403):
                console.print(
                    "  [red]Invalid key (401). Check the key and try again.[/red]"
                )
                continue
            console.print(f"  [red]Unexpected HTTP {r.status_code}.[/red]")
        except httpx.HTTPError as e:
            console.print(f"  [red]Network error: {e}[/red]")

        if not Confirm.ask("  Retry?", default=True):
            return None


# ---------------------------------------------------------------------------
# .env writers
# ---------------------------------------------------------------------------

def _write_env_hosted(api_key: str, model: str, dry_run: bool) -> None:
    _write_env([
        "# Generated by setup/setup.py — do not commit this file.",
        "BACKEND=nim-hosted",
        f"NVIDIA_API_KEY={api_key}",
        f"NIM_BASE={NIM_BASE}",
        f"NIM_MODEL={model}",
    ], dry_run)


def _write_env_container(
    api_key: str, model: str, vram_gb: int, dry_run: bool
) -> None:
    _write_env([
        "# Generated by setup/setup.py — do not commit this file.",
        "BACKEND=nim-container",
        f"NGC_API_KEY={api_key}",
        f"NVIDIA_API_KEY={api_key}",
        f"NIM_MODEL={model}",
        f"NIM_IMAGE={container_image(model)}",
        f"GPU_VRAM_GB={vram_gb}",
        "NIM_HOST=localhost",
        "NIM_PORT=8000",
    ], dry_run)


def _write_env(lines: list[str], dry_run: bool) -> None:
    content = "\n".join(lines) + "\n"
    if dry_run:
        console.print(f"\n  [dim][dry-run] Would write {ENV_PATH}:[/dim]")
        console.print("  " + content.replace("\n", "\n  "))
        return
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text(content)
    os.chmod(ENV_PATH, 0o600)
    console.print(f"\n  [green]Wrote {ENV_PATH}[/green]")


if __name__ == "__main__":
    sys.exit(main())
