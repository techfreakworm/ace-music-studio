"""LoRA stack: sniff/validate user-uploaded .safetensors files and
manage which one is active on the ACE-Step DiT handler.

Single-LoRA semantics
---------------------
The Apple-Silicon ACE-Step fork's AceStepHandler exposes a one-LoRA-
at-a-time API (load_lora / unload_lora / set_use_lora / set_lora_scale),
not the multi-adapter PEFT pattern the plan's Task D3 originally
described. ``apply_stack(pipe, stack)`` therefore supports:

- empty stack -> ``unload_lora`` + ``set_use_lora(False)``
- single-entry stack -> ``load_lora(path)`` + ``set_lora_scale(scale)``
  + ``set_use_lora(True)``
- multi-entry stack -> use only the first, log a warning

If the upstream pipeline ever exposes multi-adapter support, this
function can be extended without changing the wrapper's call sites.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

# Expected DiT module suffixes for ACE-Step 1.5 XL SFT.
# Match against `*.to_q.lora_A.weight`, etc.
_EXPECTED_MODULES = {"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"}
_MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB cap
_MAX_RANK = 256


class LoRAValidationError(ValueError):
    """Raised when a LoRA file fails validation."""


@dataclass
class LoRAInfo:
    path: Path
    compatible: bool
    rank: int
    alpha: int | None
    target_modules: set[str]
    diagnostic: str
    file_size: int


def sniff(path: Path | str) -> LoRAInfo:
    """Read the safetensors header; do not materialise tensors."""
    path = Path(path)
    if not path.exists():
        raise LoRAValidationError(f"File not found: {path}")

    file_size = path.stat().st_size
    if file_size > _MAX_FILE_BYTES:
        raise LoRAValidationError(
            f"File too large ({file_size / 1e6:.0f} MB > {_MAX_FILE_BYTES / 1e6:.0f} MB cap)."
        )

    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise LoRAValidationError("Not a valid .safetensors file (truncated)")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if header_len <= 0 or header_len > 10 * 1024 * 1024:
            raise LoRAValidationError(f"Unreasonable header length: {header_len}")
        header_bytes = f.read(header_len)

    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as e:
        raise LoRAValidationError(f"Invalid header JSON: {e}") from e

    target_modules: set[str] = set()
    rank = 0
    alpha = None
    has_ace_prefix = False

    for k, v in header.items():
        if k == "__metadata__":
            if isinstance(v, dict):
                if "lora_alpha" in v:
                    try:
                        alpha = int(v["lora_alpha"])
                    except (TypeError, ValueError):
                        pass
            continue
        if not isinstance(v, dict) or "shape" not in v:
            continue
        # ACE-Step DiT keys start with "transformer." (the diffusers DiT prefix).
        # SDXL UNet LoRAs start with "unet." — reject those even though the
        # inner attention layer names overlap (`.to_q.lora_A.weight`).
        if k.startswith("transformer.") or k.startswith("transformer_blocks."):
            has_ace_prefix = True
        # Extract module suffix from things like "transformer.blocks.0.attn.to_q.lora_A.weight"
        for suffix in _EXPECTED_MODULES:
            if f".{suffix}.lora_A.weight" in k or f".{suffix}.lora_B.weight" in k:
                target_modules.add(suffix)
                if "lora_A.weight" in k:
                    rank = max(rank, int(v["shape"][0]))
                break

    compatible = has_ace_prefix and bool(target_modules) and (rank > 0) and (rank <= _MAX_RANK)
    diagnostic = (
        "OK"
        if compatible
        else (
            f"Expected ACE-Step DiT modules ({sorted(_EXPECTED_MODULES)}), got modules in: "
            f"{sorted(set(header.keys()) - {'__metadata__'})[:3]}…"
        )
    )

    return LoRAInfo(
        path=path,
        compatible=compatible,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        diagnostic=diagnostic,
        file_size=file_size,
    )


_PRESETS_PATH = Path(__file__).resolve().parent / "presets" / "manifest.json"


def load_presets() -> list[dict]:
    """Load the bundled LoRA preset manifest."""
    return json.loads(_PRESETS_PATH.read_text())


def download_preset(name: str) -> Path:
    """Download a preset LoRA from HF if not already cached.

    Returns the local path on success. Raises LoRAValidationError if the
    preset name is unknown OR the HF download fails (network, 404, etc.).
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    for p in load_presets():
        if p["name"] == name:
            try:
                local = hf_hub_download(repo_id=p["hf_id"], filename=p["filename"])
                return Path(local)
            except HfHubHTTPError as e:
                raise LoRAValidationError(
                    f"Could not download preset {name!r} from {p['hf_id']!r}: {e}"
                ) from e
    raise LoRAValidationError(f"Unknown preset: {name}")
