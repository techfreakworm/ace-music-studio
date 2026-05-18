"""L2 tests for cover / extend / edit mode handlers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import modes


def test_cover_requires_ref_audio():
    with pytest.raises(ValueError, match="ref_audio"):
        modes.cover(
            MagicMock(),
            params={"prompt": "p", "lyrics": "[v]", "duration_s": 30, "ref_audio": None},
        )


def test_cover_passes_audio_cover_strength():
    backend = MagicMock()
    backend.dispatch.return_value = ("/tmp/x.wav", {})
    modes.cover(
        backend,
        params={
            "prompt": "p",
            "lyrics": "[v]",
            "duration_s": 30,
            "ref_audio": "/tmp/ref.wav",
            "audio_cover_strength": 0.9,
            "loras": [],
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )
    args = backend.dispatch.call_args.kwargs
    assert args["mode"] == "cover"
    assert args["params"]["audio_cover_strength"] == 0.9
    assert args["params"]["ref_audio"] == "/tmp/ref.wav"


def test_extend_requires_seed_audio():
    with pytest.raises(ValueError, match="seed_audio"):
        modes.extend(
            MagicMock(),
            params={"extra_prompt": "p", "extra_duration_s": 60, "seed_audio": None},
        )


def test_extend_passes_repaint_params():
    backend = MagicMock()
    backend.dispatch.return_value = ("/tmp/x.wav", {})
    modes.extend(
        backend,
        params={
            "seed_audio": "/tmp/seed.wav",
            "extra_prompt": "more",
            "extra_duration_s": 60,
            "extension_lyrics": "[v]",
            "repaint_strength": 0.5,
            "wav_crossfade_s": 2.0,
            "loras": [],
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )
    args = backend.dispatch.call_args.kwargs
    assert args["mode"] == "extend"
    assert args["params"]["repaint_strength"] == 0.5
    assert args["params"]["wav_crossfade_s"] == 2.0


def test_edit_repaint_passes_segment_bounds():
    backend = MagicMock()
    backend.dispatch.return_value = ("/tmp/x.wav", {})
    modes.edit(
        backend,
        params={
            "source_audio": "/tmp/src.wav",
            "source_lyrics": "[v]",
            "target_lyrics": "[c] new",
            "segment_start_s": 50.0,
            "segment_end_s": 90.0,
            "sub_mode": "repaint",
            "repaint_strength": 0.5,
            "loras": [],
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )
    args = backend.dispatch.call_args.kwargs
    assert args["mode"] == "edit"
    assert args["params"]["segment_start_s"] == 50.0
    assert args["params"]["segment_end_s"] == 90.0
    assert args["params"]["sub_mode"] == "repaint"


def test_edit_flow_morph_sub_mode_passes_through():
    backend = MagicMock()
    backend.dispatch.return_value = ("/tmp/x.wav", {})
    modes.edit(
        backend,
        params={
            "source_audio": "/tmp/src.wav",
            "source_lyrics": "[v]",
            "target_lyrics": "[c]",
            "segment_start_s": 0.0,
            "segment_end_s": 30.0,
            "sub_mode": "flow_edit",
            "flow_source_caption": "acoustic ballad",
            "flow_n_min": 0.0,
            "flow_n_max": 1.0,
            "flow_n_avg": 1,
            "loras": [],
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )
    args = backend.dispatch.call_args.kwargs
    assert args["params"]["sub_mode"] == "flow_edit"
    assert args["params"]["flow_source_caption"] == "acoustic ballad"
