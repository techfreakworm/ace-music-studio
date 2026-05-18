"""Pure mode handlers — one function per generation mode.

Each handler validates inputs, builds the ACE-Step kwargs for its mode, and
hands off to `backend.dispatch(...)`. Backend ownership of @spaces.GPU and
pipeline lifecycle keeps these handlers cheap to test.
"""

from __future__ import annotations

from typing import Any


def _require(params: dict[str, Any], field: str) -> Any:
    v = params.get(field)
    if v is None or (isinstance(v, str) and not v.strip()):
        raise ValueError(f"Missing required field: {field}")
    return v


def generate(backend, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Text → song. Vocals + instruments in one stream."""
    prompt = _require(params, "prompt")
    lyrics = params.get("lyrics", "")
    duration_s = int(params.get("duration_s", 30))
    instrumental = bool(params.get("instrumental", False))

    return backend.dispatch(
        mode="generate",
        params={
            "prompt": prompt,
            "lyrics": lyrics,
            "duration_s": duration_s,
            "instrumental": instrumental,
            "seed": params.get("seed"),
            "loras": params.get("loras", []),
            "advanced": params.get("advanced", {}),
            "lm": params.get("lm", {}),
            "dcw": params.get("dcw", {}),
        },
    )
