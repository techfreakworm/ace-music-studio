"""ACEStepStudioBackend — dispatch + ZeroGPU lifetime + duration estimator.

Off Spaces, @spaces.GPU is a no-op identity decorator (`spaces` may not be
installed locally). On Spaces, the HF runtime injects it at startup and
the decorator applies for real.
"""

from __future__ import annotations

import random
import time
from typing import Any

try:
    import spaces  # type: ignore[import-not-found]

    _HAS_SPACES = True
except ImportError:  # pragma: no cover - covered by manual local testing
    spaces = None
    _HAS_SPACES = False

import ace_pipeline as ap


def _maybe_seed(seed: int | None) -> int:
    if seed and int(seed) > 0:
        return int(seed)
    return random.randint(1, 2_147_483_647)


def _duration_estimate(mode: str, params: dict[str, Any]) -> int:
    """ZeroGPU per-call duration cap, clamped [60, 180] s."""
    base = 60
    duration_s = int(params.get("duration_s", 30) or 30)
    if duration_s > 60:
        base = 90
    if duration_s > 120:
        base = 120
    if mode == "edit":
        base = max(base, 90)
    if mode == "extend":
        base = max(base, 120)
    return min(180, max(60, base))


class ACEStepStudioBackend:
    """Lazy backend singleton. Owns @spaces.GPU and pipeline lifecycle."""

    def dispatch(self, mode: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        params = dict(params)
        params["seed"] = _maybe_seed(params.get("seed"))
        t0 = time.time()
        pipe = ap.get_pipeline()
        out_path = self._call_pipe_for_mode(pipe, mode, params)
        meta = {
            "mode": mode,
            "seed": params["seed"],
            "duration_s": params.get("duration_s"),
            "wall_seconds": round(time.time() - t0, 2),
            "estimated_duration_s": _duration_estimate(mode, params),
            "loras": [
                {"name": lora.get("name"), "scale": lora.get("scale"), "sha256": lora.get("sha256")}
                for lora in params.get("loras", [])
            ],
            "lm": params.get("lm", {}),
            "dcw": params.get("dcw", {}),
        }
        return out_path, meta

    def _call_pipe_for_mode(self, pipe, mode: str, params: dict[str, Any]) -> str:
        """Dispatch to the pipeline wrapper.

        ``pipe`` is the ``ACEStepStudio`` wrapper returned by
        ``ace_pipeline.get_pipeline()``. It exposes a single
        ``generate(params)`` method that handles the underlying
        AceStepHandler + LLMHandler + generate_music plumbing.

        Cover / Extend / Edit / Lyrics task_types are mapped here at
        M3 / M4 by switching ``params["task_type"]`` before calling.
        """
        if mode == "generate":
            return pipe.generate(params)
        # cover / extend / edit / lyrics get filled in at M3 / M4
        raise NotImplementedError(f"Mode {mode!r} is not wired yet")
