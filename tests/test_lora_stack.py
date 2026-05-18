"""L1 tests for LoRA header sniffing + apply_stack — no torch, no pipeline."""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import lora_stack as ls


def _write_safetensors(path: Path, key_dict: dict[str, dict]) -> None:
    """Minimal safetensors writer: header JSON + dummy tensor bytes."""
    header_json = json.dumps(key_dict).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    path.write_bytes(header_len + header_json + b"\0" * 8)


def test_sniff_accepts_ace_step_lora(tmp_path):
    p = tmp_path / "psytrance.safetensors"
    _write_safetensors(
        p,
        {
            "transformer.blocks.0.attn.to_q.lora_A.weight": {
                "dtype": "BF16",
                "shape": [64, 768],
                "data_offsets": [0, 8],
            },
            "transformer.blocks.0.attn.to_q.lora_B.weight": {
                "dtype": "BF16",
                "shape": [768, 64],
                "data_offsets": [0, 8],
            },
        },
    )
    info = ls.sniff(p)
    assert info.compatible is True
    assert info.rank == 64
    assert "to_q" in info.target_modules


def test_sniff_rejects_sdxl_lora(tmp_path):
    p = tmp_path / "sdxl.safetensors"
    _write_safetensors(
        p,
        {
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight": {
                "dtype": "F16",
                "shape": [16, 320],
                "data_offsets": [0, 8],
            },
        },
    )
    info = ls.sniff(p)
    assert info.compatible is False
    assert "expected" in info.diagnostic.lower()


def test_sniff_rejects_oversize(tmp_path):
    p = tmp_path / "huge.safetensors"
    p.write_bytes(b"\0" * (600 * 1024 * 1024))
    with pytest.raises(ls.LoRAValidationError, match="too large"):
        ls.sniff(p)


def test_apply_stack_empty_disables_lora():
    pipe = MagicMock()
    pipe._dit = MagicMock()
    ls.apply_stack(pipe, [])
    pipe._dit.unload_lora.assert_called_once()
    pipe._dit.set_use_lora.assert_called_with(False)


def test_apply_stack_single_lora_loads_and_enables(tmp_path):
    pipe = MagicMock()
    pipe._dit = MagicMock()
    fake_path = tmp_path / "psy.safetensors"
    fake_path.write_bytes(b"\0")
    stack = [{"name": "psytrance_v2", "scale": 0.95, "path": str(fake_path), "sha256": "a" * 64}]
    ls.apply_stack(pipe, stack)
    pipe._dit.load_lora.assert_called_once_with(str(fake_path))
    pipe._dit.set_lora_scale.assert_called_once_with(0.95)
    pipe._dit.set_use_lora.assert_called_with(True)


def test_apply_stack_multi_lora_uses_first_and_warns(tmp_path, caplog):
    pipe = MagicMock()
    pipe._dit = MagicMock()
    a = tmp_path / "a.safetensors"
    a.write_bytes(b"\0")
    b = tmp_path / "b.safetensors"
    b.write_bytes(b"\0")
    stack = [
        {"name": "a", "scale": 0.85, "path": str(a), "sha256": "1" * 64},
        {"name": "b", "scale": 0.95, "path": str(b), "sha256": "2" * 64},
    ]
    with caplog.at_level(logging.WARNING):
        ls.apply_stack(pipe, stack)
    pipe._dit.load_lora.assert_called_once_with(str(a))
    pipe._dit.set_lora_scale.assert_called_once_with(0.85)
    assert any("only one" in r.message.lower() or "single" in r.message.lower() for r in caplog.records)
