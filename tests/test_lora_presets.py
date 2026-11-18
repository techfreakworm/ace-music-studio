"""L1 tests for the LoRA preset registry.

The preset list itself can grow / shrink as ACE-Step publishes (or
removes) official LoRAs on HuggingFace. These tests verify the
manifest schema is correct, not the exact preset count.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import lora_stack as ls
import model_store


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


# --- pinned revisions and where presets come from ---------------------------------


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Preset weights are 88-524 MB: no test here may reach the Hub."""
    import socket

    def _refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


PRESET_REVISIONS = {
    "ACE-Step/ACE-Step-v1-chinese-rap-LoRA": "e40a223cb69a60ba8a8c46752987c152b8a5f828",
    "ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA": "cb829a12775740c830a6d49795f16913065dc492",
}


def test_presets_pin_their_commits():
    presets = ls.load_presets()
    assert {p["hf_id"]: p["revision"] for p in presets} == PRESET_REVISIONS


def test_presets_are_not_in_the_startup_manifest():
    # Presets are fetched on click; the startup bootstrap never downloads them.
    repo_ids = {r["repo_id"] for r in model_store.load_manifest()["repos"]}
    assert not repo_ids & set(PRESET_REVISIONS)


@pytest.mark.parametrize("space_id", [None, "test/test"])
def test_download_preset_passes_the_pinned_revision(monkeypatch, tmp_path, space_id):
    import huggingface_hub

    calls = []

    def _hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    if space_id is None:
        monkeypatch.delenv("SPACE_ID", raising=False)
    else:
        monkeypatch.setenv("SPACE_ID", space_id)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _hf_hub_download)
    monkeypatch.setattr(model_store, "ensure_repo", MagicMock(side_effect=AssertionError("store used")))

    for preset in ls.load_presets():
        calls.clear()
        path = ls.download_preset(preset["name"])
        assert path == tmp_path / preset["filename"]
        assert calls == [
            {"repo_id": preset["hf_id"], "filename": preset["filename"], "revision": preset["revision"]}
        ]


def test_download_preset_unknown_name():
    with pytest.raises(ls.LoRAValidationError, match="Unknown preset"):
        ls.download_preset("No Such Preset")
