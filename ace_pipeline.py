"""ACE-Step pipeline lifecycle: device autodetect, lazy load, cache mirror.

Mirrors z-image-studio's `models.py` pattern. M0 only implements device
detection — the pipeline class itself is filled in at M1.
"""

from __future__ import annotations


def detect_device() -> str:
    """Returns 'cuda', 'mps', or 'cpu' in priority order."""
    try:
        import torch  # local import: keep module import cheap for CI
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    # macOS: torch.backends.mps appeared in 2.0; guard for the rare absence
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def vram_limit_for(device: str) -> int | None:
    """Returns a VRAM cap in bytes for CUDA, None otherwise.

    `torch.mps` has no `mem_get_info` — calling DiffSynth-style free-VRAM
    gates with a numeric limit would crash on MPS. Returning None lets the
    pipeline short-circuit those checks.
    """
    if device != "cuda":
        return None
    try:
        import torch

        free, _total = torch.cuda.mem_get_info()
        # Leave 2 GiB headroom for activations
        return max(0, free - 2 * 1024**3)
    except Exception:
        return None
