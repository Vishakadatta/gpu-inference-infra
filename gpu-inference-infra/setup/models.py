"""NIM model registry, publisher allowlist, and policy enforcement.

NIM_MODELS             — curated list of inference models on the NIM free tier.
NIM_PUBLISHER_MAP      — maps lowercase publisher prefix → (company, country).
NIM_BLOCKED_PUBLISHERS — publishers rejected by policy (Chinese-origin).

Ported and adapted from the ModelCouncil project.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Curated NIM model list (free tier, verified live)
# (nim_id, display_name, param_b, publisher, speed_tier)
# ---------------------------------------------------------------------------

NIM_MODELS: list[tuple[str, str, float, str, str]] = [
    # ── Fast / small — best for load testing (shows queue + latency behaviour) ──
    ("meta/llama-3.2-3b-instruct",               "Llama 3.2 3B",        3.0,   "meta",      "fastest"),
    ("microsoft/phi-3-mini-128k-instruct",        "Phi-3 Mini 128K",     3.8,   "microsoft", "fastest"),
    ("google/gemma-3-4b-it",                     "Gemma 3 4B",          4.0,   "google",    "fastest"),
    ("meta/llama-3.1-8b-instruct",                "Llama 3.1 8B",        8.0,   "meta",      "fast"),
    ("mistralai/mistral-7b-instruct-v0.3",        "Mistral 7B",          7.0,   "mistralai", "fast"),
    ("google/gemma-3-12b-it",                    "Gemma 3 12B",        12.0,   "google",    "fast"),
    # ── Medium — good quality, moderate latency ──
    ("nvidia/llama-3.3-nemotron-super-49b-v1.5",  "Nemotron Super 49B", 49.0,   "nvidia",    "medium"),
    ("meta/llama-3.3-70b-instruct",               "Llama 3.3 70B",      70.0,   "meta",      "medium"),
    ("meta/llama-3.1-70b-instruct",               "Llama 3.1 70B",      70.0,   "meta",      "medium"),
    ("mistralai/mixtral-8x7b-instruct",           "Mixtral 8x7B",       56.0,   "mistralai", "medium"),
    # ── Large — highest quality, highest latency (shows infra under pressure) ──
    ("nvidia/llama-3.1-nemotron-ultra-253b-v1",   "Nemotron Ultra 253B",253.0,  "nvidia",    "slow"),
    ("meta/llama-3.1-405b-instruct",              "Llama 3.1 405B",    405.0,   "meta",      "slow"),
]

# Default model — fast, free, reliable, good for demonstrating infra behaviour
NIM_DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"

# NVIDIA NIM hosted API base URL
NIM_BASE = "https://integrate.api.nvidia.com/v1"

# NIM self-hosted container registry prefix
NIM_REGISTRY = "nvcr.io/nim"


# ---------------------------------------------------------------------------
# Publisher registry
# Keys are lowercase publisher prefixes from NIM model IDs ("publisher/name").
# ---------------------------------------------------------------------------

NIM_PUBLISHER_MAP: dict[str, tuple[str, str]] = {
    # North America
    "meta":             ("Meta",          "USA"),
    "nvidia":           ("NVIDIA",        "USA"),
    "microsoft":        ("Microsoft",     "USA"),
    "google":           ("Google",        "USA"),
    "cohere":           ("Cohere",        "Canada"),
    "writer":           ("Writer",        "USA"),
    "snowflake":        ("Snowflake",     "USA"),
    "databricks":       ("Databricks",    "USA"),
    "togethercomputer": ("Together AI",   "USA"),
    "nomic-ai":         ("Nomic AI",      "USA"),
    "allenai":          ("Allen AI",      "USA"),
    "salesforce":       ("Salesforce",    "USA"),
    "ibm":              ("IBM",           "USA"),
    "openai":           ("OpenAI",        "USA"),
    "abacusai":         ("Abacus.AI",     "USA"),
    "zyphra":           ("Zyphra",        "USA"),
    # Europe
    "mistralai":        ("Mistral AI",    "France"),
    "nv-mistralai":     ("Mistral AI",    "France"),
    "bigscience":       ("BigScience",    "France"),
    "stabilityai":      ("Stability AI",  "UK"),
    # Middle East / Asia-Pacific (non-China)
    "tiiuae":           ("TII",           "UAE"),
    "upstage":          ("Upstage",       "South Korea"),
    "ai21labs":         ("AI21 Labs",     "Israel"),
    "aisingapore":      ("AI Singapore",  "Singapore"),
    "sarvamai":         ("Sarvam AI",     "India"),
}

# Publishers whose models are rejected — no Chinese-origin models.
NIM_BLOCKED_PUBLISHERS: set[str] = {
    "qwen", "alibaba", "alibabacloud",
    "baidu",
    "bytedance",
    "tencent",
    "deepseek-ai", "deepseek",
    "01-ai", "01ai",
    "thudm", "zhipu-ai", "zhipuai",
    "minimax-ai", "minimaxai", "minimax",
    "moonshot-ai", "moonshotai", "moonshot", "kimi",
    "baichuan-inc", "baichuan",
    "internlm", "shanghaiailab",
    "senseauto", "sensetime", "megvii",
    "modelbest", "idea-ccnl", "idea-research",
    "pkuslam", "thu-coai", "fudan-nlp",
    "z-ai", "glm",
    "stepfun-ai", "stepfun",
    "baai",
    "infini-ai", "openbmb",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def origin_for(nim_id: str) -> str:
    """Return 'Company, Country' for a NIM model ID."""
    publisher = nim_id.split("/", 1)[0].lower() if "/" in nim_id else nim_id.lower()
    entry = NIM_PUBLISHER_MAP.get(publisher)
    return f"{entry[0]}, {entry[1]}" if entry else "Unknown"


def is_blocked(nim_id: str) -> bool:
    """Return True if this model's publisher is on the blocklist."""
    publisher = nim_id.split("/", 1)[0].lower() if "/" in nim_id else nim_id.lower()
    return publisher in NIM_BLOCKED_PUBLISHERS


def container_image(nim_id: str) -> str:
    """Return the full nvcr.io container image path for a NIM model ID."""
    return f"{NIM_REGISTRY}/{nim_id}:latest"
