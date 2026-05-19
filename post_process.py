"""Post-generation: stem separation (Demucs), loudness normalisation
(pyloudnorm), and MP3 export (ffmpeg)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

_DEMUCS = None


def _get_demucs() -> Any:
    """Lazy-load the htdemucs model.

    Demucs 4.0.x exposes ``demucs.pretrained.get_model`` and
    ``demucs.apply.apply_model`` — the higher-level
    ``demucs.api.Separator`` convenience wrapper only appears in 4.1+.
    We pin to the lower-level API so this works across both pip-installable
    lines without forcing an upgrade on the apple-silicon torch stack.
    """
    global _DEMUCS
    if _DEMUCS is None:
        from demucs.pretrained import get_model

        _DEMUCS = get_model("htdemucs")
    return _DEMUCS


def separate_stems(audio_path: Path | str) -> dict[str, Path]:
    """Split into vocals/drums/bass/other via htdemucs.

    Uses the lower-level ``demucs.apply.apply_model`` so we don't depend
    on the ``demucs.api.Separator`` wrapper (which only ships with
    demucs >= 4.1). Returns a dict mapping stem name to written file path.
    """
    import soundfile as sf
    import torch
    import torchaudio
    from demucs.apply import apply_model

    model = _get_demucs()
    target_sr = int(getattr(model, "samplerate", 44100))
    sources = list(getattr(model, "sources", ["drums", "bass", "other", "vocals"]))
    audio_channels = int(getattr(model, "audio_channels", 2))

    waveform, sr = torchaudio.load(str(audio_path))  # (channels, frames)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Match the model's expected channel count (htdemucs is stereo).
    if waveform.shape[0] == 1 and audio_channels == 2:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > audio_channels:
        waveform = waveform[:audio_channels]

    # apply_model expects shape (batch, channels, frames).
    batch = waveform.unsqueeze(0)
    with torch.no_grad():
        # apply_model returns (batch, sources, channels, frames).
        out = apply_model(model, batch, device="cpu", progress=False)
    out = out[0]  # drop batch dim -> (sources, channels, frames)

    base = Path(audio_path).with_suffix("")
    stems: dict[str, Path] = {}
    for idx, name in enumerate(sources):
        out_path = base.with_name(f"{base.name}.{name}.wav")
        data = out[idx].cpu().numpy()
        # soundfile expects (frames, channels); demucs gives (channels, frames)
        if data.ndim == 2 and data.shape[0] in (1, 2):
            data = data.T
        sf.write(str(out_path), data, target_sr)
        stems[name] = out_path
    return stems


def _pyloudnorm_normalise(in_path: str, out_path: str, target_lufs: float) -> None:
    """Real pyloudnorm path; isolated for easy mocking in tests."""
    import pyloudnorm as pyln
    import soundfile as sf

    data, rate = sf.read(in_path)
    meter = pyln.Meter(rate)
    current = meter.integrated_loudness(data)
    normalised = pyln.normalize.loudness(data, current, target_lufs)
    sf.write(out_path, normalised, rate)


def normalise_lufs(audio_path: Path | str, target_lufs: float = -14.0) -> Path:
    """Normalise to streaming-spec LUFS. Writes a new file alongside the input."""
    audio_path = Path(audio_path)
    out_path = audio_path.with_name(f"{audio_path.stem}.lufs{int(target_lufs)}.wav")
    _pyloudnorm_normalise(str(audio_path), str(out_path), target_lufs)
    return out_path


def to_mp3(wav_path: Path | str, bitrate_kbps: int = 320) -> Path:
    """Encode WAV to MP3 via system ffmpeg."""
    wav_path = Path(wav_path)
    out_path = wav_path.with_suffix(".mp3")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-b:a",
        f"{bitrate_kbps}k",
        "-ar",
        "44100",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
