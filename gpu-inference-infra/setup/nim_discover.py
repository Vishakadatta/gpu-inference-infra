"""Dynamic model discovery from the NVIDIA NIM API.

Queries https://integrate.api.nvidia.com/v1/models live, applies policy rules,
and returns a validated model ID for inference.

Policy rules (in order):
  1. Reject blocked publishers  (Chinese-origin — see NIM_BLOCKED_PUBLISHERS)
  2. Reject unknown publishers   (unknown origin = not trusted)
  3. Extract parameter count     (reject if unparseable)
  4. Speed classification        (fast <15B, medium 15–100B, large >100B)

Adapted from the ModelCouncil project's setup/nim_discover.py.
"""

from __future__ import annotations

import re
import webbrowser
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from setup.models import NIM_BLOCKED_PUBLISHERS, NIM_PUBLISHER_MAP, NIM_BASE

console = Console()

SPEED_TIERS = {
    "fast":   "< 15B   — fastest TTFT, best for load testing",
    "medium": "15–100B — balanced quality / latency",
    "large":  "> 100B  — highest quality, highest latency",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NIMModel:
    model_id: str
    publisher: str
    company: str
    country: str
    param_b: float
    speed: str   # "fast" | "medium" | "large"

    @property
    def origin_str(self) -> str:
        return f"{self.company}, {self.country}"

    @property
    def param_str(self) -> str:
        if self.param_b >= 1:
            return f"{self.param_b:.0f}B"
        return f"{self.param_b * 1000:.0f}M"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _publisher_from_id(model_id: str) -> str:
    return model_id.split("/", 1)[0].lower() if "/" in model_id else model_id.lower()


def _company_country(publisher: str) -> tuple[str, str] | None:
    if publisher in NIM_BLOCKED_PUBLISHERS:
        return None
    entry = NIM_PUBLISHER_MAP.get(publisher)
    return entry if entry else None


def _extract_param_b(model_id: str) -> float | None:
    """Parse parameter count (billions) from a NIM model ID string."""
    name = model_id.lower()
    # Mixture-of-Experts: 8x7b → 56B
    moe = re.search(r"(\d+)x(\d+)b", name)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    # Standard: -8b-, _3.8b_
    sep = re.search(r"[\-_\.](\d+(?:\.\d+)?)b(?:[\-_\.]|$)", name)
    if sep:
        return float(sep.group(1))
    # Bare token: "8b"
    bare = re.search(r"(?<!\d)(\d+(?:\.\d+)?)b(?!\w)", name)
    if bare:
        return float(bare.group(1))
    return None


def _speed_tier(param_b: float) -> str:
    if param_b < 15:
        return "fast"
    if param_b <= 100:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Core: fetch + filter
# ---------------------------------------------------------------------------

def fetch_nim_models(api_key: str) -> list[NIMModel]:
    """
    GET /v1/models, apply policy rules, return usable models sorted by size.
    Raises httpx.HTTPError on network/auth failure.
    """
    r = httpx.get(
        f"{NIM_BASE}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15.0,
    )
    r.raise_for_status()

    raw: list[dict] = r.json().get("data", [])
    usable: list[NIMModel] = []
    n_filtered = 0

    for m in raw:
        model_id: str = m.get("id", "").strip()
        if not model_id:
            continue
        publisher = _publisher_from_id(model_id)

        cc = _company_country(publisher)
        if cc is None:
            n_filtered += 1
            continue
        company, country = cc

        param_b = _extract_param_b(model_id)
        if param_b is None:
            n_filtered += 1
            continue

        usable.append(NIMModel(
            model_id=model_id,
            publisher=publisher,
            company=company,
            country=country,
            param_b=param_b,
            speed=_speed_tier(param_b),
        ))

    usable.sort(key=lambda m: m.param_b)

    if n_filtered:
        console.print(
            f"[dim]  Filtered {n_filtered} model(s): "
            f"blocked/unknown origin or unparseable size.[/dim]"
        )
    return usable


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _show_table(models: list[NIMModel], default_idx: int) -> None:
    table = Table(
        title="Available NVIDIA NIM Models  (live discovery — free tier)",
        show_lines=True,
    )
    table.add_column("#",        style="bold cyan", min_width=3)
    table.add_column("Model ID", min_width=44)
    table.add_column("Params",   min_width=7)
    table.add_column("Speed",    min_width=8)
    table.add_column("Origin",   min_width=20)

    for i, m in enumerate(models, 1):
        marker = " ★" if i - 1 == default_idx else ""
        table.add_row(
            str(i),
            m.model_id + marker,
            m.param_str,
            m.speed,
            m.origin_str,
        )
    console.print(table)
    console.print(
        f"[dim]  ★ = recommended default  "
        f"· fast models show the clearest latency/queue behaviour under load[/dim]"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def discover_and_pick(api_key: str) -> str | None:
    """
    Full discovery flow: fetch → filter → show table → user picks.

    Returns the chosen NIM model ID, or None if cancelled.
    Raises httpx.HTTPError on unrecoverable network failure.
    """
    console.print("\n  Fetching available models from NVIDIA NIM…")

    try:
        models = fetch_nim_models(api_key)
    except httpx.HTTPStatusError as e:
        console.print(
            f"[red]  NIM API returned {e.response.status_code}. "
            f"Check your API key and try again.[/red]"
        )
        return None
    except httpx.HTTPError as e:
        console.print(f"[red]  Network error reaching NIM: {e}[/red]")
        return None

    if not models:
        console.print(
            "[red]  No eligible models found after filtering.\n"
            "  Visit https://build.nvidia.com to check the current catalogue.[/red]"
        )
        return None

    # Default: first fast model (smallest, most stable under load tests)
    fast_models = [m for m in models if m.speed == "fast"]
    default_idx = models.index(fast_models[0]) if fast_models else 0

    console.print()
    _show_table(models, default_idx)

    while True:
        raw = Prompt.ask(
            f"\n  Which model? [1–{len(models)}]",
            default=str(default_idx + 1),
        )
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                chosen = models[idx]
                console.print(
                    f"\n  [green]Selected:[/green] {chosen.model_id}  "
                    f"({chosen.param_str} · {chosen.origin_str} · {chosen.speed})"
                )
                return chosen.model_id
        except ValueError:
            pass
        console.print(f"  [yellow]Enter a number between 1 and {len(models)}.[/yellow]")
