"""L2 tests for the generate mode handler — backend is mocked at the pipeline boundary."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import modes


def test_generate_validates_prompt_required():
    backend = MagicMock()
    with pytest.raises(ValueError, match="prompt"):
        modes.generate(backend, params={"prompt": "", "lyrics": "[verse] x", "duration_s": 10})


def test_generate_passes_params_to_backend(monkeypatch):
    backend = MagicMock()
    backend.dispatch.return_value = ("/tmp/audio.wav", {"seed": 42})
    out_path, meta = modes.generate(
        backend,
        params={
            "prompt": "psytrance",
            "lyrics": "[verse] x",
            "duration_s": 30,
            "instrumental": False,
            "seed": 42,
        },
    )

    assert out_path == "/tmp/audio.wav"
    assert meta["seed"] == 42
    backend.dispatch.assert_called_once()
    call_kwargs = backend.dispatch.call_args.kwargs
    assert call_kwargs["mode"] == "generate"
    # Cover-style params must be absent for the generate mode
    assert "audio_cover_strength" not in call_kwargs["params"]
