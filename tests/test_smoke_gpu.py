"""GPU smoke tests — skipped by default. Opt in with: pytest -m gpu

Generates the minimum-viable songs end-to-end through the real ACE-Step
pipeline. Run before each release tag.

Skipped automatically in CI by the pyproject ``addopts = -m 'not gpu'``
default. Requires:

- ``ace-step`` installed (Apple Silicon fork on Mac, upstream on CUDA)
- First run downloads ACE-Step 1.5 XL SFT weights (~16 GB) into the HF cache
- A real MPS / CUDA device — CPU inference is functionally untested
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


def test_generate_minimum_song(tmp_path):
    """Smallest end-to-end: 5 s instrumental drone, seed=1."""
    os.environ.setdefault("ACE_MODEL_PATH", "ACE-Step/ACE-Step-v1.5-XL-SFT")

    from backend import ACEStepStudioBackend

    b = ACEStepStudioBackend()
    out_path, meta = b.dispatch(
        mode="generate",
        params={
            "prompt": "test tone, simple drone",
            "lyrics": "[intro] tone",
            "duration_s": 5,
            "instrumental": True,
            "seed": 1,
            "loras": [],
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 0
    assert meta["mode"] == "generate"
    assert meta["seed"] == 1
    # Wall time should be < 5 min even on first cold run + 16 GB weight download.
    # Subsequent runs should be < 30 s on M5 Max.
    assert meta["wall_seconds"] > 0
