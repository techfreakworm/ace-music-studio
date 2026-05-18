"""L2 tests for pipeline lazy load — mock the heavy ACE-Step import."""

from __future__ import annotations

from unittest.mock import MagicMock

import ace_pipeline as ap


def test_get_pipeline_loads_lazily_first_call_only(monkeypatch):
    fake_pipe = MagicMock(name="fake_ace_pipeline")
    loader = MagicMock(return_value=fake_pipe)
    monkeypatch.setattr(ap, "_load_pipeline", loader)
    monkeypatch.setattr(ap, "_PIPELINE", None, raising=False)

    p1 = ap.get_pipeline()
    p2 = ap.get_pipeline()

    assert p1 is fake_pipe
    assert p2 is fake_pipe
    assert loader.call_count == 1, "pipeline should load exactly once"


def test_get_pipeline_uses_detected_device(monkeypatch):
    monkeypatch.setattr(ap, "_PIPELINE", None, raising=False)
    monkeypatch.setattr(ap, "detect_device", lambda: "mps")
    captured = {}

    def fake_load(device, model_path):
        captured["device"] = device
        captured["model_path"] = model_path
        return MagicMock()

    monkeypatch.setattr(ap, "_load_pipeline", fake_load)

    ap.get_pipeline()

    assert captured["device"] == "mps"
    assert captured["model_path"] is not None
