"""L2 tests for backend.dispatch — pipeline is mocked at the wrapper boundary."""

from __future__ import annotations

from unittest.mock import MagicMock

import backend as be


def test_dispatch_generate_calls_pipeline_generate(monkeypatch, tmp_path):
    """Backend should call ``pipe.generate(params)`` and return its path."""
    fake_out = tmp_path / "out.wav"
    fake_out.write_bytes(b"RIFF" + b"\0" * 1000)

    fake_pipe = MagicMock()
    fake_pipe.generate.return_value = str(fake_out)
    monkeypatch.setattr("ace_pipeline.get_pipeline", lambda: fake_pipe)

    b = be.ACEStepStudioBackend()
    out_path, meta = b.dispatch(
        mode="generate",
        params={
            "prompt": "psytrance",
            "lyrics": "[verse]",
            "duration_s": 10,
            "instrumental": False,
            "seed": 42,
            "loras": [],
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )

    assert out_path == str(fake_out)
    assert meta["mode"] == "generate"
    assert meta["seed"] == 42
    fake_pipe.generate.assert_called_once()
    # The full params dict is forwarded to pipe.generate
    sent_params = fake_pipe.generate.call_args.args[0]
    assert sent_params["prompt"] == "psytrance"
    assert sent_params["seed"] == 42


def test_dispatch_random_seed_if_zero(monkeypatch, tmp_path):
    out = tmp_path / "x.wav"
    out.write_bytes(b"RIFF")
    fake_pipe = MagicMock()
    fake_pipe.generate.return_value = str(out)
    monkeypatch.setattr("ace_pipeline.get_pipeline", lambda: fake_pipe)

    b = be.ACEStepStudioBackend()
    _, meta = b.dispatch(
        mode="generate",
        params={
            "prompt": "p",
            "lyrics": "",
            "duration_s": 5,
            "instrumental": False,
            "seed": 0,
            "loras": [],
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )

    assert 1 <= meta["seed"] <= 2_147_483_647
    # The seed-resolved value is the one forwarded to the wrapper
    sent_params = fake_pipe.generate.call_args.args[0]
    assert sent_params["seed"] == meta["seed"]
