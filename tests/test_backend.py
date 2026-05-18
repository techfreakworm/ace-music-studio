"""L2 tests for backend.dispatch — pipeline is mocked at the boundary."""

from __future__ import annotations

from unittest.mock import MagicMock

import backend as be


def test_dispatch_generate_calls_pipeline_with_expected_kwargs(monkeypatch, tmp_path):
    fake_pipe = MagicMock()
    fake_out = tmp_path / "out.wav"
    fake_out.write_bytes(b"RIFF" + b"\0" * 1000)
    fake_pipe.return_value = str(fake_out)

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
    fake_pipe.assert_called_once()


def test_dispatch_random_seed_if_zero(monkeypatch, tmp_path):
    fake_pipe = MagicMock(return_value=str(tmp_path / "x.wav"))
    monkeypatch.setattr("ace_pipeline.get_pipeline", lambda: fake_pipe)
    (tmp_path / "x.wav").write_bytes(b"RIFF")

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
