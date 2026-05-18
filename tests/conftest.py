"""Shared pytest fixtures.

The default pytest config (pyproject.toml) skips tests marked `gpu`. Opt in
to GPU smoke tests with `pytest -m gpu`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is importable as flat modules (matches z-image-studio convention).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _silence_mps_fallback_env(monkeypatch):
    """L1+L2 tests don't touch torch/mps; clear the env so test logs are clean."""
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
