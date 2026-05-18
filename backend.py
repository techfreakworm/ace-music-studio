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
import lora_stack


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
        lora_stack.apply_stack(pipe, params.get("loras", []))
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

        All four song modes (``generate``, ``cover``, ``extend``, ``edit``)
        flow through ``pipe.generate(params)``. The pipeline wrapper
        switches its ``GenerationParams.task_type`` based on ``params["mode"]``
        — see ``ace_pipeline.ACEStepStudio.generate`` for the mapping. The
        ``lyrics`` mode is wired separately at M4.
        """
        if mode in ("generate", "cover", "extend", "edit"):
            params_with_mode = {**params, "mode": mode}
            return pipe.generate(params_with_mode)
        raise NotImplementedError(f"Mode {mode!r} is not wired yet")
