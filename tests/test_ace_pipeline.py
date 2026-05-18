"""L1 tests for device autodetect — no torch needed if we mock importlib."""

from __future__ import annotations

import ace_pipeline as ap


def test_detect_device_returns_one_of_cuda_mps_cpu():
    device = ap.detect_device()
    assert device in {"cuda", "mps", "cpu"}


def test_vram_limit_for_mps_is_none():
    """MPS has no torch.mps.mem_get_info; return None so DiffSynth-style gates
    short-circuit instead of crashing (z-image-studio paid this debug cycle)."""
    assert ap.vram_limit_for("mps") is None


def test_vram_limit_for_cpu_is_none():
    assert ap.vram_limit_for("cpu") is None
