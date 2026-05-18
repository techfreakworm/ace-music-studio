"""L2 tests for the ACEStepStudio wrapper — mocks the heavy acestep imports."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import ace_pipeline as ap


def test_get_pipeline_returns_singleton(monkeypatch):
    monkeypatch.setattr(ap, "_PIPELINE", None, raising=False)
    p1 = ap.get_pipeline()
    p2 = ap.get_pipeline()
    assert p1 is p2
    assert isinstance(p1, ap.ACEStepStudio)


def test_studio_constructor_uses_detected_device(monkeypatch):
    monkeypatch.setattr(ap, "detect_device", lambda: "mps")
    studio = ap.ACEStepStudio()
    assert studio.device == "mps"
    assert studio.is_loaded is False  # handlers are lazy


def test_studio_constructor_respects_env_overrides(monkeypatch):
    monkeypatch.setenv("ACE_DIT_CONFIG", "custom-dit")
    monkeypatch.setenv("ACE_LM_MODEL", "custom-lm")
    monkeypatch.setattr(ap, "detect_device", lambda: "cpu")
    studio = ap.ACEStepStudio()
    assert studio._dit_config == "custom-dit"
    assert studio._lm_model == "custom-lm"


def test_studio_ensure_loaded_constructs_both_handlers(monkeypatch):
    fake_dit_cls = MagicMock(name="AceStepHandler")
    fake_lm_cls = MagicMock(name="LLMHandler")
    fake_dit = MagicMock()
    fake_lm = MagicMock()
    fake_dit_cls.return_value = fake_dit
    fake_lm_cls.return_value = fake_lm

    handler_mod = MagicMock()
    handler_mod.AceStepHandler = fake_dit_cls
    llm_mod = MagicMock()
    llm_mod.LLMHandler = fake_lm_cls

    monkeypatch.setitem(sys.modules, "acestep.handler", handler_mod)
    monkeypatch.setitem(sys.modules, "acestep.llm_inference", llm_mod)
    monkeypatch.setattr(ap, "detect_device", lambda: "mps")

    studio = ap.ACEStepStudio()
    studio._ensure_loaded()

    fake_dit_cls.assert_called_once()
    fake_lm_cls.assert_called_once()
    fake_dit.initialize_service.assert_called_once()
    fake_lm.initialize.assert_called_once()
    assert fake_dit.initialize_service.call_args.kwargs["device"] == "mps"
    assert fake_lm.initialize.call_args.kwargs["device"] == "mps"
    assert fake_dit.initialize_service.call_args.kwargs["config_path"] == "acestep-v15-xl-sft"
    assert fake_lm.initialize.call_args.kwargs["lm_model_path"] == "acestep-5Hz-lm-0.6B"


def _install_fake_inference(monkeypatch, success=True, audios=None, error=None):
    """Plant a fake ``acestep.inference`` module and return the spies."""
    if audios is None:
        audios = [{"path": "/tmp/x.wav"}]
    fake_result = MagicMock(success=success, audios=audios, error=error)
    fake_generate = MagicMock(return_value=fake_result)
    captured = {"gp": {}, "gc": {}}

    def fake_gp(**kw):
        captured["gp"] = kw
        return kw

    def fake_gc(**kw):
        captured["gc"] = kw
        return kw

    fake_inference = MagicMock()
    fake_inference.generate_music = fake_generate
    fake_inference.GenerationParams = MagicMock(side_effect=fake_gp)
    fake_inference.GenerationConfig = MagicMock(side_effect=fake_gc)
    monkeypatch.setitem(sys.modules, "acestep.inference", fake_inference)
    return fake_generate, captured


def test_studio_generate_builds_params_and_calls_generate_music(monkeypatch, tmp_path):
    out_wav = tmp_path / "out.wav"
    out_wav.write_bytes(b"RIFF" + b"\0" * 100)

    fake_generate, captured = _install_fake_inference(monkeypatch, audios=[{"path": str(out_wav)}])

    studio = ap.ACEStepStudio()
    studio._dit = MagicMock(name="dit")
    studio._llm = MagicMock(name="llm")

    result_path = studio.generate(
        {
            "prompt": "psytrance",
            "lyrics": "[verse]",
            "duration_s": 30,
            "instrumental": False,
            "seed": 42,
            "loras": [],
            "advanced": {"steps": 32, "cfg": 4.0, "bpm": 135},
            "lm": {"thinking": False},
            "dcw": {},
        }
    )

    assert result_path == str(out_wav)
    fake_generate.assert_called_once()
    assert captured["gp"]["caption"] == "psytrance"
    assert captured["gp"]["duration"] == 30
    assert captured["gp"]["seed"] == 42
    assert captured["gp"]["inference_steps"] == 32
    assert captured["gp"]["bpm"] == 135


def test_studio_generate_raises_on_failure(monkeypatch):
    _install_fake_inference(monkeypatch, success=False, audios=[], error="OOM")
    studio = ap.ACEStepStudio()
    studio._dit = MagicMock()
    studio._llm = MagicMock()

    with pytest.raises(RuntimeError, match="OOM"):
        studio.generate(
            {
                "prompt": "p",
                "lyrics": "",
                "duration_s": 5,
                "instrumental": True,
                "seed": 1,
                "advanced": {},
                "lm": {},
                "dcw": {},
            }
        )


def test_studio_generate_uses_instrumental_marker_when_lyrics_empty(monkeypatch):
    _fake_generate, captured = _install_fake_inference(monkeypatch)
    studio = ap.ACEStepStudio()
    studio._dit = MagicMock()
    studio._llm = MagicMock()

    studio.generate(
        {
            "prompt": "drone",
            "lyrics": "",
            "duration_s": 5,
            "instrumental": True,
            "seed": 1,
            "advanced": {},
            "lm": {},
            "dcw": {},
        }
    )

    # Instrumental + empty lyrics → ACE-Step convention is "[Instrumental]"
    assert captured["gp"]["lyrics"] == "[Instrumental]"
    assert captured["gp"]["instrumental"] is True
