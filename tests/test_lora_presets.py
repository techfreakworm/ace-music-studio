"""L1 tests for the LoRA preset registry."""

from __future__ import annotations

import lora_stack as ls


def test_load_presets_returns_four():
    presets = ls.load_presets()
    assert len(presets) == 4
    names = [p["name"] for p in presets]
    assert "RapMachine" in names
    assert "Lyric2Vocal" in names


def test_preset_has_required_fields():
    presets = ls.load_presets()
    for p in presets:
        assert "name" in p
        assert "hf_id" in p
        assert "default_scale" in p
        assert 0 <= p["default_scale"] <= 1.5
