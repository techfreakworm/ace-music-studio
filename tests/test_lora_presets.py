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


def test_presets_pin_the_model_manifest_commits():
    import re

    manifest = model_store.load_manifest()
    for p in ls.load_presets():
        assert re.fullmatch(r"[0-9a-f]{40}", p["revision"]), p
        entry = model_store.get_repo(manifest, repo_id=p["hf_id"])
        assert p["revision"] == entry["revision"], p["name"]
        assert entry["startup"] == "lazy", p["name"]
        assert p["filename"] in [f["path"] for f in entry["files"]], p["name"]


def test_download_preset_off_spaces_passes_pinned_revision(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []

    def _hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _hf_hub_download)
    monkeypatch.setattr(model_store, "ensure_repo", MagicMock(side_effect=AssertionError("store used")))
    preset = ls.load_presets()[0]

    path = ls.download_preset(preset["name"])

    assert path == tmp_path / preset["filename"]
    assert calls == [
        {"repo_id": preset["hf_id"], "filename": preset["filename"], "revision": preset["revision"]}
    ]


def test_download_preset_on_spaces_goes_through_the_model_store(monkeypatch, tmp_path):
    import huggingface_hub

    roots = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
    ensured = []

    def _ensure_repo(entry, roots_arg, **kwargs):
        ensured.append((entry, list(roots_arg)))
        return tmp_path / "snap"

    monkeypatch.setenv("SPACE_ID", "test/test")
    monkeypatch.setattr(model_store, "default_roots", lambda: roots)
    monkeypatch.setattr(model_store, "ensure_repo", _ensure_repo)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", MagicMock(side_effect=AssertionError("hub used")))
    preset = next(p for p in ls.load_presets() if p["name"] == "Chinese New Year")

    path = ls.download_preset(preset["name"])

    assert path == tmp_path / "snap" / preset["filename"]
    assert len(ensured) == 1
    entry, used_roots = ensured[0]
    assert entry == model_store.get_repo(model_store.load_manifest(), repo_id=preset["hf_id"])
    assert used_roots == roots


def test_download_preset_on_spaces_reports_store_failure(monkeypatch):
    import huggingface_hub

    monkeypatch.setenv("SPACE_ID", "test/test")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", MagicMock(side_effect=AssertionError("hub used")))
    monkeypatch.setattr(
        model_store, "ensure_repo", MagicMock(side_effect=model_store.ModelStoreError("no root worked"))
    )
    with pytest.raises(ls.LoRAValidationError, match="no root worked"):
        ls.download_preset(ls.load_presets()[0]["name"])


def test_download_preset_unknown_name():
    with pytest.raises(ls.LoRAValidationError, match="Unknown preset"):
        ls.download_preset("No Such Preset")
