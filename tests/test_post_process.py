"""L2 tests for post-processing — Demucs and ffmpeg are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import post_process as pp


def test_separate_stems_returns_four_paths(tmp_path, monkeypatch):
    src = tmp_path / "song.wav"
    src.write_bytes(b"RIFF" + b"\0" * 100)

    fake_sep = MagicMock()
    fake_sep.separate_audio_file.return_value = {
        "vocals": tmp_path / "vocals.wav",
        "drums": tmp_path / "drums.wav",
        "bass": tmp_path / "bass.wav",
        "other": tmp_path / "other.wav",
    }
    for k in ("vocals", "drums", "bass", "other"):
        (tmp_path / f"{k}.wav").write_bytes(b"RIFF" + b"\0" * 100)
    monkeypatch.setattr(pp, "_get_demucs", lambda: fake_sep)

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
