"""L1 tests for LoRA header sniffing — no torch, no pipeline."""

from __future__ import annotations

import json
import struct
from pathlib import Path

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
