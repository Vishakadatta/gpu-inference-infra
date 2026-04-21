"""Suggested model list and vendor-allowlist enforcement.

Policy: only models from Meta, Mistral AI, Google, or Microsoft are permitted.
Chinese-origin models are explicitly blocked regardless of license.
"""

from __future__ import annotations

# (hf_name, display_name, vram_gb, origin)
SUGGESTED_MODELS: list[tuple[str, str, int, str]] = [
    ("microsoft/Phi-3-mini-4k-instruct",       "Phi-3 Mini",           4,  "Microsoft, USA"),
    ("mistralai/Mistral-7B-Instruct-v0.3",     "Mistral 7B",           6,  "Mistral AI, France"),
    ("meta-llama/Meta-Llama-3-8B-Instruct",    "Llama 3 8B",           6,  "Meta, USA"),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct",  "Llama 3.1 8B",         6,  "Meta, USA"),
    ("google/gemma-2-9b-it",                   "Gemma 2 9B",           8,  "Google, USA"),
    ("google/gemma-2-27b-it",                  "Gemma 2 27B",         18,  "Google, USA"),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct", "Llama 3.1 70B",       40,  "Meta, USA"),
]

# Case-insensitive substrings; matched against the org prefix or full name.
BLOCKED_ORIGINS = (
    "qwen", "deepseek", "yi-", "01-ai", "thudm", "internlm",
    "baichuan", "chatglm",
)

POLICY_MSG = (
    "This model origin is not permitted. This project enforces a vendor "
    "allowlist requiring models from Meta, Mistral AI, Google, or Microsoft."
)


def fits(model_vram_gb: int, available_gb: int) -> bool:
    return model_vram_gb <= available_gb


def suggestions_for(vram_gb: int) -> list[tuple[str, str, int, str]]:
    """Return suggested models whose VRAM footprint fits in the given budget.

    De-duplicates by HF name while preserving order.
    """
    seen: set[str] = set()
    out: list[tuple[str, str, int, str]] = []
    for entry in SUGGESTED_MODELS:
        hf_name = entry[0]
        if hf_name in seen:
            continue
        if fits(entry[2], vram_gb):
            seen.add(hf_name)
            out.append(entry)
    return out


def is_blocked(hf_name: str) -> bool:
    lowered = hf_name.lower()
    org = lowered.split("/", 1)[0] if "/" in lowered else lowered
    for needle in BLOCKED_ORIGINS:
        if needle in org or needle in lowered:
            return True
    return False
