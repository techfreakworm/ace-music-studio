"""ACE-Step pipeline lifecycle: device autodetect, lazy load, cache mirror.

Mirrors z-image-studio's `models.py` pattern. M0 only implements device
detection — the pipeline class itself is filled in at M1.
"""

from __future__ import annotations

import os


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


_PIPELINE = None  # module-level lazy singleton
_DEFAULT_MODEL_ID = "ACE-Step/acestep-v15-xl-sft"


def _load_pipeline(device: str, model_path: str):
    """Construct the ACE-Step pipeline. Heavy import is local so unit tests can mock."""
    from ace_step import ACEStepPipeline  # type: ignore[import-not-found]

    # On Mac, the apple-silicon fork sets dtype + backend automatically.
    # On CUDA we pass bf16 explicitly.
    if device == "cuda":
        pipe = ACEStepPipeline.from_pretrained(model_path, torch_dtype="bf16")
    else:
        pipe = ACEStepPipeline.from_pretrained(model_path)

    pipe.to(device)
    return pipe


def get_pipeline():
    """Lazy-load the ACE-Step pipeline once per process."""
    global _PIPELINE
    if _PIPELINE is None:
        device = detect_device()
        model_path = os.environ.get("ACE_MODEL_PATH", _DEFAULT_MODEL_ID)
        _PIPELINE = _load_pipeline(device, model_path)
    return _PIPELINE
