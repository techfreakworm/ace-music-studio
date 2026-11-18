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


def _install_fake_constants(monkeypatch):
    fake_constants = MagicMock()
    fake_constants.TASK_INSTRUCTIONS = {"repaint": "Repaint the mask area based on the given conditions:"}
    monkeypatch.setitem(sys.modules, "acestep.constants", fake_constants)


ALL_LM_FLAGS_OFF = {
    "thinking": False,
    "use_cot_caption": False,
    "use_cot_metas": False,
    "use_cot_language": False,
}

# One call per song mode, shaped like backend.dispatch's params.
MODE_CALLS = {
    "generate": {"mode": "generate", "prompt": "p", "duration_s": 5, "seed": 1},
    "cover": {"mode": "cover", "ref_audio": "/r.wav", "prompt": "p", "duration_s": 5, "seed": 1},
    "extend": {"mode": "extend", "seed_audio": "/s.wav", "extra_duration_s": 5, "seed": 1},
    "edit_repaint": {
        "mode": "edit",
        "source_audio": "/s.wav",
        "segment_start_s": 1.0,
        "segment_end_s": 2.0,
        "sub_mode": "repaint",
        "seed": 1,
    },
    "flow_edit": {
        "mode": "edit",
        "source_audio": "/s.wav",
        "segment_start_s": 0.0,
        "segment_end_s": 5.0,
        "sub_mode": "flow_edit",
        "flow_source_caption": "c",
        "seed": 1,
    },
}


def _cold_studio(monkeypatch, **handler_kwargs):
    """A cold pipeline with fake fork handlers, inference and constants."""
    fake_dit_cls, fake_lm_cls = _install_fake_handlers(monkeypatch, **handler_kwargs)
    fake_generate, _captured = _install_fake_inference(monkeypatch)
    _install_fake_constants(monkeypatch)
    monkeypatch.setattr(ap, "_audio_duration_s", lambda path: 10.0)
    return ap.ACEStepStudio(), fake_dit_cls, fake_lm_cls, fake_generate


def _call(studio, name, lm=None):
    params = dict(MODE_CALLS[name])
    if lm is not None:
        params["lm"] = lm
    return studio.generate(params)


def test_ensure_loaded_loads_only_the_dit(monkeypatch):
    fake_dit_cls, fake_lm_cls = _install_fake_handlers(monkeypatch)

    studio = ap.ACEStepStudio()
    studio._ensure_loaded()

    fake_dit_cls.assert_called_once()
    studio._dit.initialize_service.assert_called_once()
    kwargs = studio._dit.initialize_service.call_args.kwargs
    assert kwargs["device"] == "mps"
    assert kwargs["config_path"] == "acestep-v15-xl-sft"
    assert studio.is_loaded is True
    # The LM is not part of the base load (see the lazy-LM tests below).
    fake_lm_cls.assert_not_called()
    assert studio._llm is None


def test_generate_with_defaults_loads_the_lm_once(monkeypatch):
    studio, fake_dit_cls, fake_lm_cls, fake_generate = _cold_studio(monkeypatch)

    _call(studio, "generate")
    _call(studio, "generate")

    fake_lm_cls.assert_called_once()
    lm = studio._llm
    lm.initialize.assert_called_once()
    kwargs = lm.initialize.call_args.kwargs
    assert kwargs["device"] == "mps"
    # Default 5Hz LM since f9a1a8e.
    assert kwargs["lm_model_path"] == "acestep-5Hz-lm-1.7B"
    assert [c.args[1] for c in fake_generate.call_args_list] == [lm, lm]
    fake_dit_cls.assert_called_once()


def test_lm_checkpoint_dir_is_the_vendored_fork_checkpoints(monkeypatch):
    """The LM must be looked up where the DiT handler resolves its checkpoints.

    The fork's AceStepHandler ignores ``project_root`` and uses
    ``<fork>/checkpoints`` (vendor/ace-step/checkpoints), which is the only
    directory app.py populates on the Space. ``<repo>/checkpoints`` exists
    only on the operator's Mac.
    """
    studio, *_ = _cold_studio(monkeypatch)
    _call(studio, "generate")

    expected = ap._REPO_ROOT / "vendor" / "ace-step" / "checkpoints"
    assert studio._llm.initialize.call_args.kwargs["checkpoint_dir"] == str(expected)


def test_ace_lm_model_absolute_path_is_passed_through(monkeypatch, tmp_path):
    """An absolute ACE_LM_MODEL keeps working: the fork joins it onto
    checkpoint_dir with os.path.join, which keeps the absolute path."""
    studio, *_ = _cold_studio(monkeypatch)
    lm_dir = tmp_path / "my-lm"
    monkeypatch.setenv("ACE_LM_MODEL", str(lm_dir))
    studio = ap.ACEStepStudio()  # reads ACE_LM_MODEL at construction
    _call(studio, "generate")

    kwargs = studio._llm.initialize.call_args.kwargs
    assert kwargs["lm_model_path"] == str(lm_dir)
    assert os.path.join(kwargs["checkpoint_dir"], kwargs["lm_model_path"]) == str(lm_dir)


@pytest.mark.parametrize("name", ["cover", "extend", "edit_repaint", "flow_edit"])
def test_modes_the_fork_runs_without_lm_never_load_it(monkeypatch, name):
    """The fork skips the LM for task_type cover and repaint
    (inference.py skip_lm_tasks), so a cold Cover/Extend/Edit call must not
    pay for loading it, even with thinking and CoT on (the UI defaults)."""
    studio, fake_dit_cls, fake_lm_cls, fake_generate = _cold_studio(monkeypatch)

    _call(studio, name)
    _call(studio, name)

    fake_lm_cls.assert_not_called()
    assert studio._llm is None
    assert [c.args[1] for c in fake_generate.call_args_list] == [None, None]
    fake_dit_cls.assert_called_once()


def test_generate_without_thinking_or_cot_does_not_load_the_lm(monkeypatch):
    studio, _dit_cls, fake_lm_cls, fake_generate = _cold_studio(monkeypatch)

    _call(studio, "generate", lm=dict(ALL_LM_FLAGS_OFF))

    fake_lm_cls.assert_not_called()
    assert fake_generate.call_args.args[1] is None


@pytest.mark.parametrize("flag", ["thinking", "use_cot_caption", "use_cot_metas", "use_cot_language"])
def test_generate_loads_the_lm_when_any_lm_flag_is_on(monkeypatch, flag):
    """Mirror of the fork's rule: thinking OR any use_cot_* flag."""
    studio, _dit_cls, fake_lm_cls, fake_generate = _cold_studio(monkeypatch)

    _call(studio, "generate", lm={**ALL_LM_FLAGS_OFF, flag: True})

    fake_lm_cls.assert_called_once()
    assert fake_generate.call_args.args[1] is studio._llm


def test_lm_loads_on_the_first_generate_after_other_modes(monkeypatch):
    studio, fake_dit_cls, fake_lm_cls, fake_generate = _cold_studio(monkeypatch)

    for name in ("cover", "extend", "edit_repaint", "flow_edit"):
        _call(studio, name)
    fake_lm_cls.assert_not_called()

    _call(studio, "generate")
    fake_lm_cls.assert_called_once()
    assert fake_generate.call_args.args[1] is studio._llm
    fake_dit_cls.assert_called_once()


def test_lm_not_found_is_logged_and_generation_runs_without_lm(monkeypatch, capsys):
    """ACE_LM_MODEL naming a model that does not exist: the fork returns
    ("... not found ...", False) without raising. That must be loud on
    stderr and must NOT stop generation (today's effective behaviour)."""
    studio, fake_dit_cls, fake_lm_cls, fake_generate = _cold_studio(
        monkeypatch,
        lm_result=("❌ 5Hz LM model not found at /x/checkpoints/no-such-lm", False),
    )
    monkeypatch.setenv("ACE_LM_MODEL", "no-such-lm")
    studio = ap.ACEStepStudio()  # reads ACE_LM_MODEL at construction

    _call(studio, "generate")

    err = capsys.readouterr().err
    assert "[ams]" in err
    assert "no-such-lm" in err
    assert "not found" in err
    assert studio._llm is None
    assert fake_generate.call_args.args[1] is None  # the DiT runs alone

    # A failed LM is not retried on the next Generate, and the DiT is untouched.
    _call(studio, "generate")
    fake_lm_cls.assert_called_once()
    assert fake_generate.call_count == 2
    assert fake_generate.call_args.args[1] is None
    fake_dit_cls.assert_called_once()
    studio._dit.initialize_service.assert_called_once()


def test_lm_init_exception_is_logged_and_not_fatal(monkeypatch, capsys):
    studio, fake_dit_cls, fake_lm_cls, fake_generate = _cold_studio(
        monkeypatch, lm_error=RuntimeError("mlx exploded")
    )

    _call(studio, "generate")
    _call(studio, "generate")

    err = capsys.readouterr().err
    assert "[ams]" in err
    assert "mlx exploded" in err
    assert studio.is_loaded is True
    assert studio._llm is None
    assert [c.args[1] for c in fake_generate.call_args_list] == [None, None]
    fake_lm_cls.assert_called_once()
    fake_dit_cls.assert_called_once()
    studio._dit.initialize_service.assert_called_once()


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
    studio, _dit_cls, fake_lm_cls, fake_generate = _cold_studio(
        monkeypatch, dit_result=("❌ Unknown DiT model", False)
    )
    with pytest.raises(RuntimeError, match="Unknown DiT model"):
        _call(studio, "generate")
    fake_generate.assert_not_called()
    fake_lm_cls.assert_not_called()


def test_dit_load_logs_one_timing_line(monkeypatch, capsys):
    _install_fake_handlers(monkeypatch)

    studio = ap.ACEStepStudio()
    studio._ensure_loaded()
    studio._ensure_loaded()  # already loaded: no second line

    err = capsys.readouterr().err.splitlines()
    lines = [ln for ln in err if ln.startswith("[ams] pipeline load:")]
    assert len(lines) == 1
    assert re.search(r"DiT/VAE/text encoder \d+\.\d s", lines[0]), lines[0]
    assert not [ln for ln in err if ln.startswith("[ams] 5Hz LM load:")]


@pytest.mark.parametrize(
    ("lm_result", "active"),
    [(("LM ready", True), "yes"), (("❌ 5Hz LM model not found at /x", False), "no")],
)
def test_lm_load_logs_one_timing_line_when_it_happens(monkeypatch, capsys, lm_result, active):
    studio, *_ = _cold_studio(monkeypatch, lm_result=lm_result)

    _call(studio, "cover")
    assert not [ln for ln in capsys.readouterr().err.splitlines() if "5Hz LM load:" in ln]

    _call(studio, "generate")
    _call(studio, "generate")
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[ams] 5Hz LM load:")]
    assert len(lines) == 1
    assert re.search(r"5Hz LM load: \d+\.\d s", lines[0]), lines[0]
    assert f"LM active: {active}" in lines[0]


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
