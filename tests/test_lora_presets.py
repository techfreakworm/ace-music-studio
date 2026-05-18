"""L1 tests for the LoRA preset registry.

The preset list itself can grow / shrink as ACE-Step publishes (or
removes) official LoRAs on HuggingFace. These tests verify the
manifest schema is correct, not the exact preset count.
"""

from __future__ import annotations

import lora_stack as ls


def test_load_presets_returns_at_least_one():
    presets = ls.load_presets()
    assert len(presets) >= 1
    # As of 2026-05, ACE-Step's chinese-rap is the only LoRA the org
    # has actually published. If they later add RapMachine etc., the
    # manifest is updated and this test still passes.
    names = [p["name"] for p in presets]
    assert "Chinese Rap" in names


def test_preset_has_required_fields():
    presets = ls.load_presets()
    for p in presets:
        assert "name" in p
        assert "hf_id" in p
        assert "filename" in p
        assert "default_scale" in p
        assert 0 <= p["default_scale"] <= 1.5


def test_preset_hf_id_is_under_ace_step_org():
    """Every shipped preset must live under the official ACE-Step org so
    we can trust the upstream quality. Community LoRAs go via the
    custom-upload path, not the preset chip row."""
    presets = ls.load_presets()
    for p in presets:
        assert p["hf_id"].startswith("ACE-Step/"), (
            f"Preset {p['name']!r} hf_id {p['hf_id']!r} is not under the official ACE-Step org"
        )
