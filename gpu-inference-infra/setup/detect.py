"""Cross-platform VRAM detection.

Priority order: nvidia-smi -> rocm-smi -> Apple sysctl -> /proc/meminfo.
Always returns (gb_or_none, source_label) — never raises. Caller decides
whether to fall back to manual user entry.

Adapted from the ModelCouncil project's setup/detect.py.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess


def _run(cmd: list[str], timeout: int = 5) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _nvidia_smi() -> tuple[int, str] | None:
    if not shutil.which("nvidia-smi"):
        return None
    mem = _run([
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ])
    name = _run([
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader",
    ])
    if not mem:
        return None
    try:
        mb = max(int(line.strip()) for line in mem.splitlines() if line.strip())
    except ValueError:
        return None
    gpu_name = (name.splitlines()[0].strip() if name else "NVIDIA GPU")
    return round(mb / 1024), f"NVIDIA {gpu_name}"


def _rocm_smi() -> tuple[int, str] | None:
    if not shutil.which("rocm-smi"):
        return None
    out = _run(["rocm-smi", "--showmeminfo", "vram"])
    if not out:
        return None
    m = re.search(r"Total\s*Memory.*?:\s*(\d+)", out)
    if not m:
        return None
    return round(int(m.group(1)) / (1024 ** 3)), "AMD ROCm GPU"


def _macos_unified_memory() -> tuple[int, str] | None:
    if platform.system() != "Darwin":
        return None
    out = _run(["sysctl", "-n", "hw.memsize"])
    if not out:
        return None
    try:
        gb = round(int(out.strip()) / (1024 ** 3))
    except ValueError:
        return None
    return gb, "Apple Silicon (unified memory)"


def _proc_meminfo() -> tuple[int, str] | None:
    if platform.system() != "Linux":
        return None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2)), "system RAM (/proc/meminfo)"
    except OSError:
        return None
    return None


def detect_vram() -> tuple[int | None, str]:
    """Return (gb, source_label). gb is None if all detectors failed."""
    for fn in (_nvidia_smi, _rocm_smi, _macos_unified_memory, _proc_meminfo):
        result = fn()
        if result:
            return result
    return None, "auto-detection failed"


MANUAL_HINTS = """\
Manual VRAM check commands:
  macOS:    system_profiler SPDisplaysDataType | grep VRAM
  Linux:    nvidia-smi --query-gpu=memory.total --format=csv
  Windows:  nvidia-smi  (in CMD, or Task Manager -> Performance -> GPU)
"""
