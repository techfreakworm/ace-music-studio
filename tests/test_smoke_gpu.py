"""GPU smoke tests — skipped by default. Opt in with: pytest -m gpu

Generates the minimum-viable songs end-to-end through the real ACE-Step
pipeline. Run before each release tag.

Skipped automatically in CI by the pyproject ``addopts = -m 'not gpu'``
default. Requires:

- ``acestep`` package installed (Apple Silicon fork on Mac, upstream on CUDA)
- DiT checkpoint at ``./checkpoints/acestep-v15-xl-sft/`` (~16 GB) — download via
  ``hf download ACE-Step/acestep-v15-xl-sft --local-dir checkpoints/acestep-v15-xl-sft``
- LM checkpoint at ``./checkpoints/acestep-5Hz-lm-0.6B/`` (~1.4 GB) — download via
  ``hf download ACE-Step/acestep-5Hz-lm-0.6B --local-dir checkpoints/acestep-5Hz-lm-0.6B``
- A real MPS / CUDA device — CPU inference is functionally untested
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


def test_generate_minimum_song():
    """Smallest end-to-end: 10 s instrumental drone, seed=1, 16 diffusion steps.

    Asserts the pipeline produces a non-empty audio file. Wall time on
    cold start (handlers + weight loading) should be < 5 min on M5 Max
    with checkpoints pre-downloaded; subsequent calls in the same process
    are bounded by the diffusion compute itself (~10-30 s for these settings).
    """
    from backend import ACEStepStudioBackend

    b = ACEStepStudioBackend()
    out_path, meta = b.dispatch(
        mode="generate",
        params={
            "prompt": "ambient drone, sine pad, slow swell",
            "lyrics": "",
            "duration_s": 10,
            "instrumental": True,
            "seed": 1,
            "loras": [],
            # Tune for smoke speed: fewer steps, lower CFG, skip LM CoT
            "advanced": {"steps": 16, "cfg": 3.0, "audio_format": "wav"},
            "lm": {"thinking": False},
            "dcw": {},
        },
    )

    p = Path(out_path)
    assert p.exists(), f"generated file missing: {out_path}"
    assert p.stat().st_size > 0, "generated file is empty"
    assert meta["mode"] == "generate"
    assert meta["seed"] == 1
    assert meta["wall_seconds"] > 0
