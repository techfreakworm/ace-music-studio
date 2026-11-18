"""L2 tests: the ``GenerationParams`` each song mode hands to the fork.

The heavy ``acestep`` modules are replaced by fakes at the call boundary
(same pattern as ``test_ace_pipeline_lazy.py``); every keyword argument
passed to ``GenerationParams`` is pinned, so a change to any mode's mapping
shows up here.
"""

from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import ace_pipeline as ap

REPAINT_INSTRUCTION = "Repaint the mask area based on the given conditions:"
_FORK_CONSTANTS = Path(ap.__file__).resolve().parent / "vendor" / "ace-step" / "acestep" / "constants.py"

# The dicts app._build_advanced_params produces with the UI defaults.
UI_ADVANCED = {
    "inference_steps": 27,
    "guidance_scale": 7.0,
    "infer_method": "ode",
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "shift": 1.0,
    "use_adg": False,
    "bpm": None,
    "keyscale": "",
    "timesignature": "",
    "vocal_language": "unknown",
}
UI_LM = {
    "thinking": True,
    "use_cot_caption": True,
    "use_cot_metas": True,
    "use_cot_language": True,
    "temperature": 0.85,
    "top_p": 0.9,
    "top_k": 0,
    "cfg": 2.0,
    "negative_prompt": "NO USER INPUT",
}
# GenerationParams fields every mode derives the same way from advanced/lm.
SHARED_FIELDS = {
    "inference_steps": 27,
    "guidance_scale": 7.0,
    "infer_method": "ode",
    "use_adg": False,
    "shift": 1.0,
    "bpm": None,
    "keyscale": "",
    "timesignature": "",
    "vocal_language": "unknown",
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "thinking": True,
    "lm_temperature": 0.85,
    "lm_cfg_scale": 2.0,
    "lm_top_k": 0,
    "lm_top_p": 0.9,
    "lm_negative_prompt": "NO USER INPUT",
    "use_cot_metas": True,
    "use_cot_caption": True,
    "use_cot_language": True,
}


def _write_wav(path: Path, seconds: float, sr: int = 48000) -> Path:
    """A stereo 16-bit wav of the given length (stdlib only)."""
    n = round(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x01" * 2 * n)
    return path


@pytest.fixture
def fake_fork(monkeypatch, tmp_path):
    """Fake ``acestep.inference`` + ``acestep.constants``; loaded studio.

    Returns ``(studio, captured)``; ``captured["gp"]`` holds the kwargs of
    the last ``GenerationParams(...)`` call.
    """
    out_wav = _write_wav(tmp_path / "out.wav", 1.0)
    captured: dict = {}

    def fake_gp(**kw):
        captured["gp"] = kw
        return kw

    def fake_gc(**kw):
        captured["gc"] = kw
        return kw

    fake_inference = MagicMock()
    fake_inference.GenerationParams = MagicMock(side_effect=fake_gp)
    fake_inference.GenerationConfig = MagicMock(side_effect=fake_gc)
    fake_inference.generate_music = MagicMock(
        return_value=MagicMock(success=True, audios=[{"path": str(out_wav)}], error=None)
    )
    fake_constants = MagicMock()
    fake_constants.TASK_INSTRUCTIONS = {
        "text2music": "Fill the audio semantic mask based on the given conditions:",
        "repaint": REPAINT_INSTRUCTION,
        "cover": "Generate audio semantic tokens based on the given conditions:",
    }
    monkeypatch.setitem(sys.modules, "acestep.inference", fake_inference)
    monkeypatch.setitem(sys.modules, "acestep.constants", fake_constants)

    studio = ap.ACEStepStudio(device="cpu")
    studio._dit = MagicMock(name="dit")
    studio._llm = MagicMock(name="llm")
    return studio, captured


def _ui_params(mode: str, **mode_params) -> dict:
    """What backend.dispatch hands the pipeline: modes.<mode> output + seed + mode."""
    base = {"loras": [], "advanced": dict(UI_ADVANCED), "lm": dict(UI_LM), "dcw": {}}
    return {**base, **mode_params, "mode": mode}


# --- Generate and Cover: pinned exactly as they were before the Extend/Edit fix ---


def test_generate_params_are_pinned(fake_fork):
    studio, captured = fake_fork
    studio.generate(
        _ui_params(
            "generate",
            prompt="warm lofi piano",
            lyrics="[verse]\nla la",
            duration_s=10,
            instrumental=False,
            seed=1234,
        )
    )
    assert captured["gp"] == {
        **SHARED_FIELDS,
        "task_type": "text2music",
        "caption": "warm lofi piano",
        "lyrics": "[verse]\nla la",
        "instrumental": False,
        "duration": 10,
        "seed": 1234,
        "reference_audio": None,
        "src_audio": None,
        "audio_cover_strength": 0.0,
        "repainting_start": 0.0,
        "repainting_end": -1.0,
    }
    assert captured["gc"] == {
        "batch_size": 1,
        "audio_format": "wav",
        "use_random_seed": False,
        "seeds": [1234],
    }


def test_generate_instrumental_params_are_pinned(fake_fork):
    studio, captured = fake_fork
    studio.generate(
        _ui_params("generate", prompt="drone", lyrics="", duration_s=5, instrumental=True, seed=7)
    )
    assert captured["gp"] == {
        **SHARED_FIELDS,
        "task_type": "text2music",
        "caption": "drone",
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "duration": 5,
        "seed": 7,
        "reference_audio": None,
        "src_audio": None,
        "audio_cover_strength": 0.0,
        "repainting_start": 0.0,
        "repainting_end": -1.0,
    }


def test_cover_params_are_pinned(fake_fork):
    studio, captured = fake_fork
    studio.generate(
        _ui_params(
            "cover",
            prompt="bright synthwave",
            ref_audio="/uploads/ref.wav",
            lyrics="",
            duration_s=10,
            audio_cover_strength=0.93,
            cover_noise_strength=0.0,
            seed=333,
        )
    )
    assert captured["gp"] == {
        **SHARED_FIELDS,
        "task_type": "cover",
        "caption": "bright synthwave",
        "lyrics": "",
        "instrumental": False,
        "duration": 10,
        "seed": 333,
        "reference_audio": "/uploads/ref.wav",
        "src_audio": None,
        "audio_cover_strength": 0.93,
        "repainting_start": 0.0,
        "repainting_end": -1.0,
    }


# --- Extend: outpaint after the seed, conditioned on it ---


def _extend_params(seed_audio: str, extra: int = 10) -> dict:
    return _ui_params(
        "extend",
        seed_audio=seed_audio,
        extra_prompt="continue the same groove",
        extension_lyrics="",
        extra_duration_s=extra,
        repaint_mode="balanced",
        repaint_strength=0.5,
        wav_crossfade_s=2.0,
        latent_crossfade_frames=10,
        chunk_mask_mode="auto",
        seed=111,
    )


def test_extend_params_are_pinned(fake_fork, monkeypatch):
    studio, captured = fake_fork
    probed = []

    def fake_len(path):
        probed.append(path)
        return 10.0

    monkeypatch.setattr(ap, "_audio_duration_s", fake_len)
    studio.generate(_extend_params("/uploads/seed.wav", extra=10))

    assert probed == ["/uploads/seed.wav"]
    assert captured["gp"] == {
        **SHARED_FIELDS,
        "task_type": "repaint",
        "caption": "continue the same groove",
        "lyrics": "",
        "instrumental": False,
        "duration": 10,
        "seed": 111,
        "reference_audio": None,
        "src_audio": "/uploads/seed.wav",
        # Repaint window starts at the end of the seed and runs extra seconds
        # past it; the fork right-pads src_audio up to repainting_end.
        "repainting_start": 10.0,
        "repainting_end": 20.0,
        # Fork default: < 1.0 swaps in a silence-built condition and drops the seed.
        "audio_cover_strength": 1.0,
        "instruction": REPAINT_INSTRUCTION,
    }


def test_extend_seed_length_comes_from_the_uploaded_file(fake_fork, tmp_path):
    pytest.importorskip("soundfile")
    studio, captured = fake_fork
    seed = _write_wav(tmp_path / "seed.wav", 7.5)

    studio.generate(_extend_params(str(seed), extra=15))

    assert captured["gp"]["repainting_start"] == pytest.approx(7.5)
    assert captured["gp"]["repainting_end"] == pytest.approx(22.5)
    assert captured["gp"]["src_audio"] == str(seed)


def test_extend_unreadable_seed_is_a_value_error_before_loading(fake_fork, tmp_path, monkeypatch):
    studio, captured = fake_fork
    studio._dit = None  # cold pipeline
    load = MagicMock(name="_ensure_loaded")
    monkeypatch.setattr(studio, "_ensure_loaded", load)
    bad = tmp_path / "seed.wav"
    bad.write_bytes(b"this is not audio" * 64)

    with pytest.raises(ValueError, match=r"seed\.wav"):
        studio.generate(_extend_params(str(bad)))

    load.assert_not_called()  # no model load (and no GPU time) for a bad upload
    assert "gp" not in captured


def test_extend_rejects_non_positive_extra_duration(fake_fork, monkeypatch):
    studio, _captured = fake_fork
    monkeypatch.setattr(ap, "_audio_duration_s", lambda path: 10.0)
    with pytest.raises(ValueError, match="extra"):
        studio.generate(_extend_params("/uploads/seed.wav", extra=0))


# --- Edit ---


def _edit_params(sub_mode: str, start: float, end: float) -> dict:
    params = _ui_params(
        "edit",
        source_audio="/uploads/src.wav",
        source_lyrics="",
        target_lyrics="",
        segment_start_s=start,
        segment_end_s=end,
        sub_mode=sub_mode,
        seed=222,
    )
    if sub_mode == "repaint":
        params.update(
            repaint_mode="balanced",
            repaint_strength=0.5,
            chunk_mask_mode="auto",
            latent_crossfade_frames=10,
            wav_crossfade_s=0.0,
        )
    else:
        params.update(flow_source_caption="acoustic ballad", flow_n_min=0.0, flow_n_max=1.0, flow_n_avg=1)
    return params


def test_edit_repaint_params_are_pinned(fake_fork):
    studio, captured = fake_fork
    studio.generate(_edit_params("repaint", 6.0, 12.0))
    assert captured["gp"] == {
        **SHARED_FIELDS,
        "task_type": "repaint",
        "caption": "",
        "lyrics": "",
        "instrumental": False,
        "duration": 30,
        "seed": 222,
        "reference_audio": None,
        "src_audio": "/uploads/src.wav",
        "repainting_start": 6.0,
        "repainting_end": 12.0,
        "audio_cover_strength": 1.0,
        "instruction": REPAINT_INSTRUCTION,
    }


def test_edit_flow_edit_params_are_pinned(fake_fork):
    """flow_edit is unchanged: repaint task, cover strength 0.3, default instruction."""
    studio, captured = fake_fork
    studio.generate(_edit_params("flow_edit", 0.0, 20.0))
    assert captured["gp"] == {
        **SHARED_FIELDS,
        "task_type": "repaint",
        "caption": "acoustic ballad",
        "lyrics": "",
        "instrumental": False,
        "duration": 30,
        "seed": 222,
        "reference_audio": None,
        "src_audio": "/uploads/src.wav",
        "repainting_start": 0.0,
        "repainting_end": 20.0,
        "audio_cover_strength": 0.3,
    }


def test_repaint_instruction_matches_the_vendored_fork():
    """The fake above mirrors the fork's TASK_INSTRUCTIONS['repaint']."""
    if not _FORK_CONSTANTS.exists():
        pytest.skip("vendor/ace-step submodule not checked out")
    spec = importlib.util.spec_from_file_location("_fork_constants", _FORK_CONSTANTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TASK_INSTRUCTIONS["repaint"] == REPAINT_INSTRUCTION


# --- Seed length read from the file ---


@pytest.mark.parametrize(("fmt", "ext"), [("WAV", "wav"), ("FLAC", "flac"), ("MP3", "mp3")])
def test_audio_duration_reads_wav_flac_mp3(tmp_path, fmt, ext):
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    sr = 44100
    t = np.arange(int(sr * 3.5)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    path = tmp_path / f"clip.{ext}"
    try:
        sf.write(str(path), np.stack([tone, tone], axis=1), sr, format=fmt)
    except Exception as exc:  # libsndfile built without MPEG support
        pytest.skip(f"cannot write {fmt} here: {exc}")

    assert ap._audio_duration_s(str(path)) == pytest.approx(3.5, abs=0.05)


def test_audio_duration_unreadable_file_is_value_error(tmp_path):
    bad = tmp_path / "upload.mp3"
    bad.write_bytes(b"\x00garbage" * 100)
    with pytest.raises(ValueError, match=r"upload\.mp3"):
        ap._audio_duration_s(str(bad))


def test_audio_duration_missing_file_is_value_error(tmp_path):
    with pytest.raises(ValueError):
        ap._audio_duration_s(str(tmp_path / "gone.wav"))
    with pytest.raises(ValueError):
        ap._audio_duration_s(None)


def test_audio_duration_empty_wav_is_value_error(tmp_path):
    pytest.importorskip("soundfile")
    empty = _write_wav(tmp_path / "empty.wav", 0.0)
    with pytest.raises(ValueError, match=r"empty\.wav"):
        ap._audio_duration_s(str(empty))
