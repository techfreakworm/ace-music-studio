"""Post-generation: stem separation (Demucs), loudness normalisation
(pyloudnorm), and MP3 export (ffmpeg)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

_DEMUCS = None


def _get_demucs() -> Any:
    global _DEMUCS
    if _DEMUCS is None:
        from demucs.api import Separator

        _DEMUCS = Separator(model="htdemucs_ft")
    return _DEMUCS


def separate_stems(audio_path: Path | str) -> dict[str, Path]:
    """Split into vocals/drums/bass/other via htdemucs_ft.

    Returns a dict mapping stem name to written file path.
    """
    sep = _get_demucs()
    result = sep.separate_audio_file(str(audio_path))
    # `result` may be either {name: path} OR (origin, separated) tuple
    # depending on demucs version. Normalise to dict[str, Path].
    if isinstance(result, dict):
        return {name: Path(p) for name, p in result.items()}
    # Newer demucs returns (origin_tensor, separated_dict_of_tensors)
    # We persist tensors next to the input file with stem suffixes.
    import soundfile as sf

    _origin, sep_tensors = result
    base = Path(audio_path).with_suffix("")
    stems: dict[str, Path] = {}
    for name, tensor in sep_tensors.items():
        out = base.with_name(f"{base.name}.{name}.wav")
        data = tensor.detach().cpu().numpy()
        if data.ndim == 2 and data.shape[0] in (1, 2):
            data = data.T
        sf.write(str(out), data, sep.samplerate)
        stems[name] = out
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
