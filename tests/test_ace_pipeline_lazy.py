"""L2 tests for the ACEStepStudio wrapper — mocks the heavy acestep imports."""

from __future__ import annotations

import os
import re
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


def _install_fake_handlers(
    monkeypatch,
    dit_result=("DiT ready", True),
    lm_result=("LM ready", True),
    dit_error=None,
    lm_error=None,
):
    """Plant fake ``acestep.handler`` / ``acestep.llm_inference`` modules.

    The fork's initialisers return ``(status, ok)`` tuples; ``*_error``
    makes them raise instead. Returns the two fake classes so tests can
    inspect every instance that was built.
    """
    fake_dit_cls = MagicMock(name="AceStepHandler")
    fake_lm_cls = MagicMock(name="LLMHandler")

    def _new_dit():
        dit = MagicMock(name="dit")
        if dit_error is not None:
            dit.initialize_service.side_effect = dit_error
        else:
            dit.initialize_service.return_value = dit_result
        return dit

    def _new_lm():
        lm = MagicMock(name="lm")
        if lm_error is not None:
            lm.initialize.side_effect = lm_error
        else:
            lm.initialize.return_value = lm_result
        return lm

    fake_dit_cls.side_effect = _new_dit
    fake_lm_cls.side_effect = _new_lm

    handler_mod = MagicMock()
    handler_mod.AceStepHandler = fake_dit_cls
    llm_mod = MagicMock()
    llm_mod.LLMHandler = fake_lm_cls

    monkeypatch.setitem(sys.modules, "acestep.handler", handler_mod)
    monkeypatch.setitem(sys.modules, "acestep.llm_inference", llm_mod)
    monkeypatch.setattr(ap, "detect_device", lambda: "mps")
    monkeypatch.delenv("ACE_LM_MODEL", raising=False)
    monkeypatch.delenv("ACE_DIT_CONFIG", raising=False)
    return fake_dit_cls, fake_lm_cls


def test_studio_ensure_loaded_constructs_both_handlers(monkeypatch):
    fake_dit_cls, fake_lm_cls = _install_fake_handlers(monkeypatch)

    studio = ap.ACEStepStudio()
    studio._ensure_loaded()

    fake_dit_cls.assert_called_once()
    fake_lm_cls.assert_called_once()
    fake_dit = studio._dit
    fake_lm = studio._llm
    fake_dit.initialize_service.assert_called_once()
    fake_lm.initialize.assert_called_once()
    assert fake_dit.initialize_service.call_args.kwargs["device"] == "mps"
    assert fake_lm.initialize.call_args.kwargs["device"] == "mps"
    assert fake_dit.initialize_service.call_args.kwargs["config_path"] == "acestep-v15-xl-sft"
    # Default 5Hz LM since f9a1a8e.
    assert fake_lm.initialize.call_args.kwargs["lm_model_path"] == "acestep-5Hz-lm-1.7B"
    assert studio.is_loaded is True


def test_lm_checkpoint_dir_is_the_vendored_fork_checkpoints(monkeypatch):
    """The LM must be looked up where the DiT handler resolves its checkpoints.

    The fork's AceStepHandler ignores ``project_root`` and uses
    ``<fork>/checkpoints`` (vendor/ace-step/checkpoints), which is the only
    directory app.py populates on the Space. ``<repo>/checkpoints`` exists
    only on the operator's Mac.
    """
    _install_fake_handlers(monkeypatch)

    studio = ap.ACEStepStudio()
    studio._ensure_loaded()

    expected = ap._REPO_ROOT / "vendor" / "ace-step" / "checkpoints"
    assert studio._llm.initialize.call_args.kwargs["checkpoint_dir"] == str(expected)


def test_ace_lm_model_absolute_path_is_passed_through(monkeypatch, tmp_path):
    """An absolute ACE_LM_MODEL keeps working: the fork joins it onto
    checkpoint_dir with os.path.join, which keeps the absolute path."""
    _install_fake_handlers(monkeypatch)
    lm_dir = tmp_path / "my-lm"
    monkeypatch.setenv("ACE_LM_MODEL", str(lm_dir))

    studio = ap.ACEStepStudio()
    studio._ensure_loaded()

    kwargs = studio._llm.initialize.call_args.kwargs
    assert kwargs["lm_model_path"] == str(lm_dir)
    assert os.path.join(kwargs["checkpoint_dir"], kwargs["lm_model_path"]) == str(lm_dir)


def test_lm_not_found_is_logged_and_generation_runs_without_lm(monkeypatch, capsys):
    """ACE_LM_MODEL naming a model that does not exist: the fork returns
    ("... not found ...", False) without raising. That must be loud on
    stderr and must NOT stop generation (today's effective behaviour)."""
    _fake_dit_cls, fake_lm_cls = _install_fake_handlers(
        monkeypatch,
        lm_result=("❌ 5Hz LM model not found at /x/checkpoints/no-such-lm", False),
    )
    monkeypatch.setenv("ACE_LM_MODEL", "no-such-lm")

    studio = ap.ACEStepStudio()
    studio._ensure_loaded()

    err = capsys.readouterr().err
    assert "[ams]" in err
    assert "no-such-lm" in err
    assert "not found" in err
    assert studio.is_loaded is True
    assert studio._llm is None  # generate_music then skips the LM

    # Generation goes ahead with the DiT only.
    fake_generate, _captured = _install_fake_inference(monkeypatch)
    studio.generate({"mode": "generate", "prompt": "p", "duration_s": 5, "seed": 1})
    fake_generate.assert_called_once()
    assert fake_generate.call_args.args[1] is None

    # A failed LM is not retried on every call.
    studio._ensure_loaded()
    fake_lm_cls.assert_called_once()


def test_lm_init_exception_is_logged_and_not_fatal(monkeypatch, capsys):
    _install_fake_handlers(monkeypatch, lm_error=RuntimeError("mlx exploded"))

    studio = ap.ACEStepStudio()
    studio._ensure_loaded()

    err = capsys.readouterr().err
    assert "[ams]" in err
    assert "mlx exploded" in err
    assert studio.is_loaded is True
    assert studio._llm is None


def test_dit_init_failure_raises_with_fork_status_and_is_retried(monkeypatch):
    status = "❌ Failed to download main model: no space left on device"
    _fake_dit_cls, fake_lm_cls = _install_fake_handlers(monkeypatch, dit_result=(status, False))

    studio = ap.ACEStepStudio()
    with pytest.raises(RuntimeError, match="no space left on device"):
        studio._ensure_loaded()

    # Nothing half-built is cached, and the LM is not loaded for nothing.
    assert studio.is_loaded is False
    assert studio._dit is None
    assert studio._llm is None
    fake_lm_cls.assert_not_called()

    # The next call tries again from scratch.
    _install_fake_handlers(monkeypatch)
    studio._ensure_loaded()
    assert studio.is_loaded is True


def test_dit_init_exception_raises_runtime_error_and_is_not_cached(monkeypatch):
    _install_fake_handlers(monkeypatch, dit_error=OSError("disk went away"))

    studio = ap.ACEStepStudio()
    with pytest.raises(RuntimeError, match="disk went away"):
        studio._ensure_loaded()
    assert studio.is_loaded is False
    assert studio._dit is None


def test_generate_does_not_run_when_dit_init_fails(monkeypatch):
    _install_fake_handlers(monkeypatch, dit_result=("❌ Unknown DiT model", False))
    fake_generate, _captured = _install_fake_inference(monkeypatch)

    studio = ap.ACEStepStudio()
    with pytest.raises(RuntimeError, match="Unknown DiT model"):
        studio.generate({"mode": "generate", "prompt": "p", "duration_s": 5, "seed": 1})
    fake_generate.assert_not_called()


@pytest.mark.parametrize(
    ("lm_result", "active"),
    [(("LM ready", True), "yes"), (("❌ 5Hz LM model not found at /x", False), "no")],
)
def test_one_timing_line_per_load(monkeypatch, capsys, lm_result, active):
    _install_fake_handlers(monkeypatch, lm_result=lm_result)

    studio = ap.ACEStepStudio()
    studio._ensure_loaded()
    studio._ensure_loaded()  # already loaded: no second line

    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[ams] pipeline load:")]
    assert len(lines) == 1
    line = lines[0]
    assert re.search(r"DiT/VAE/text encoder \d+\.\d s", line), line
    assert re.search(r"5Hz LM \d+\.\d s", line), line
    assert f"LM active: {active}" in line


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
            # New advanced contract: ``inference_steps`` + ``guidance_scale``
            # + ``infer_method`` replace the old ``steps`` + ``cfg`` keys.
            # See ace_pipeline.ACEStepStudio.generate for the full schema.
            "advanced": {
                "inference_steps": 32,
                "guidance_scale": 4.0,
                "infer_method": "ode",
                "bpm": 135,
            },
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
    assert captured["gp"]["guidance_scale"] == 4.0
    assert captured["gp"]["infer_method"] == "ode"
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
