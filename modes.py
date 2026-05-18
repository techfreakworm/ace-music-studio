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


def cover(backend, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Audio-reference cover — reference audio + new prompt -> song in that style.

    Maps to ACE-Step's ``GenerationParams(task_type="cover")`` with
    ``reference_audio`` set to the uploaded clip and ``audio_cover_strength``
    controlling how tightly the new song hugs the reference timbre/structure.
    """
    ref_audio = _require(params, "ref_audio")
    prompt = params.get("prompt", "")
    lyrics = params.get("lyrics", "")
    duration_s = int(params.get("duration_s", 30))

    return backend.dispatch(
        mode="cover",
        params={
            "prompt": prompt,
            "ref_audio": ref_audio,
            "lyrics": lyrics,
            "duration_s": duration_s,
            "audio_cover_strength": float(params.get("audio_cover_strength", 0.93)),
            "cover_noise_strength": float(params.get("cover_noise_strength", 0.0)),
            "seed": params.get("seed"),
            "loras": params.get("loras", []),
            "advanced": params.get("advanced", {}),
            "lm": params.get("lm", {}),
            "dcw": params.get("dcw", {}),
        },
    )


def extend(backend, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Song continuation — seed audio + extension prompt -> extended song.

    Maps to ACE-Step's ``GenerationParams(task_type="repaint")`` with
    ``src_audio`` set to the seed and ``repainting_start``/``repainting_end``
    pointing past the end of the seed so the model paints new audio after it.
    """
    seed_audio = _require(params, "seed_audio")
    extra_prompt = params.get("extra_prompt", "")
    extra_duration_s = int(params.get("extra_duration_s", 60))

    return backend.dispatch(
        mode="extend",
        params={
            "seed_audio": seed_audio,
            "extra_prompt": extra_prompt,
            "extension_lyrics": params.get("extension_lyrics", ""),
            "extra_duration_s": extra_duration_s,
            "repaint_mode": params.get("repaint_mode", "balanced"),
            "repaint_strength": float(params.get("repaint_strength", 0.5)),
            "wav_crossfade_s": float(params.get("wav_crossfade_s", 2.0)),
            "latent_crossfade_frames": int(params.get("latent_crossfade_frames", 10)),
            "chunk_mask_mode": params.get("chunk_mask_mode", "auto"),
            "seed": params.get("seed"),
            "loras": params.get("loras", []),
            "advanced": params.get("advanced", {}),
            "lm": params.get("lm", {}),
            "dcw": params.get("dcw", {}),
        },
    )


def edit(backend, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Segment-level edit — repaint a region OR morph caption-to-caption.

    Two sub-modes:

    - ``"repaint"`` (default): paint over ``[segment_start_s, segment_end_s]``
      using ACE-Step's ``task_type="repaint"`` with the segment bounds wired
      into ``repainting_start`` / ``repainting_end``.
    - ``"flow_edit"``: caption-to-caption morph. The installed ACE-Step
      ``GenerationParams`` dataclass has no native ``flow_edit_*`` fields, so
      flow-edit is implemented downstream as a ``task_type="repaint"`` pass
      with a lower ``audio_cover_strength`` to allow more style drift. The
      ``flow_source_caption`` / ``flow_n_*`` knobs are carried through the
      internal params dict so the pipeline wrapper can use them if/when the
      upstream dataclass grows native support.
    """
    source_audio = _require(params, "source_audio")
    sub_mode = params.get("sub_mode", "repaint")

    out_params: dict[str, Any] = {
        "source_audio": source_audio,
        "source_lyrics": params.get("source_lyrics", ""),
        "target_lyrics": params.get("target_lyrics", ""),
        "segment_start_s": float(params.get("segment_start_s", 0.0)),
        "segment_end_s": float(params.get("segment_end_s", 30.0)),
        "sub_mode": sub_mode,
        "seed": params.get("seed"),
        "loras": params.get("loras", []),
        "advanced": params.get("advanced", {}),
        "lm": params.get("lm", {}),
        "dcw": params.get("dcw", {}),
    }
    if sub_mode == "repaint":
        out_params.update(
            {
                "repaint_mode": params.get("repaint_mode", "balanced"),
                "repaint_strength": float(params.get("repaint_strength", 0.5)),
                "chunk_mask_mode": params.get("chunk_mask_mode", "auto"),
                "latent_crossfade_frames": int(params.get("latent_crossfade_frames", 10)),
                "wav_crossfade_s": float(params.get("wav_crossfade_s", 0.0)),
            }
        )
    elif sub_mode == "flow_edit":
        out_params.update(
            {
                "flow_source_caption": params.get("flow_source_caption", ""),
                "flow_n_min": float(params.get("flow_n_min", 0.0)),
                "flow_n_max": float(params.get("flow_n_max", 1.0)),
                "flow_n_avg": int(params.get("flow_n_avg", 1)),
            }
        )

    return backend.dispatch(mode="edit", params=out_params)
