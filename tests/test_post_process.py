"""L2 tests for post-processing — Demucs and ffmpeg are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import post_process as pp


def test_separate_stems_returns_four_paths(tmp_path, monkeypatch):
    """Mocks the lower-level demucs path used by post_process.separate_stems:
    torchaudio.load → apply_model → sf.write. The wrapper convenience API
    (Separator.separate_audio_file) is intentionally not in this code path
    because it only ships with demucs >= 4.1."""
    import sys
    import types

    import numpy as np
    import torch

    src = tmp_path / "song.wav"
    src.write_bytes(b"RIFF" + b"\0" * 100)

    fake_model = MagicMock()
    fake_model.samplerate = 44100
    fake_model.sources = ["drums", "bass", "other", "vocals"]
    fake_model.audio_channels = 2
    monkeypatch.setattr(pp, "_get_demucs", lambda: fake_model)

    fake_torchaudio = types.ModuleType("torchaudio")
    fake_torchaudio.load = lambda _path: (torch.zeros((2, 44100)), 44100)
    fake_torchaudio.functional = types.SimpleNamespace(resample=lambda wav, _sr_in, _sr_out: wav)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    fake_demucs_apply = types.ModuleType("demucs.apply")
    fake_demucs_apply.apply_model = lambda _m, batch, **_kw: torch.zeros((batch.shape[0], 4, 2, 44100))
    monkeypatch.setitem(sys.modules, "demucs.apply", fake_demucs_apply)

    written: list[str] = []

    def fake_sf_write(path, _data, _sr):
        written.append(path)
        Path(path).write_bytes(b"RIFF" + b"\0" * 100)

    fake_sf = types.ModuleType("soundfile")
    fake_sf.write = fake_sf_write
    fake_sf.read = lambda _path: (np.zeros((44100, 2)), 44100)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    stems = pp.separate_stems(src)
    assert set(stems.keys()) == {"vocals", "drums", "bass", "other"}
    for p in stems.values():
        assert Path(p).exists()


def test_normalise_lufs_invokes_pyloudnorm(monkeypatch, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFF" + b"\0" * 100)
    captured = {}

    def fake_norm(in_path, out_path, target_lufs):
        captured.update({"in": in_path, "out": out_path, "target": target_lufs})
        Path(out_path).write_bytes(b"RIFF")

    monkeypatch.setattr(pp, "_pyloudnorm_normalise", fake_norm)

    out = pp.normalise_lufs(src, target_lufs=-14.0)
    assert captured["target"] == -14.0
    assert Path(out).exists()


def test_to_mp3_invokes_ffmpeg(monkeypatch, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFF")
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"ID3")
        return MagicMock(returncode=0)

    monkeypatch.setattr(pp.subprocess, "run", fake_run)
    out = pp.to_mp3(src, bitrate_kbps=320)
    assert Path(out).exists()
    assert "320k" in " ".join(captured["cmd"])
