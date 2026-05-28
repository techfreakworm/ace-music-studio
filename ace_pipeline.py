"""ACE-Step pipeline lifecycle: device autodetect, lazy load, cache mirror.

The installed ``acestep`` package (apple-silicon fork on Mac, upstream on
CUDA) does NOT expose a single ``ACEStepPipeline.from_pretrained`` entry
point. The real API is a split-handler pattern:

  from acestep.handler import AceStepHandler           # DiT side
  from acestep.llm_inference import LLMHandler         # 5Hz LM planner
  from acestep.inference import (
      GenerationParams, GenerationConfig, generate_music,
  )

  dit = AceStepHandler()
  dit.initialize_service(project_root=..., config_path="acestep-v15-xl-sft",
                         device="mps")
  lm = LLMHandler()
  lm.initialize(checkpoint_dir=..., lm_model_path="acestep-5Hz-lm-0.6B",
                backend="vllm",      # auto-routes to mlx on mps
                device="mps")
  params = GenerationParams(caption=..., lyrics=..., duration=..., seed=...)
  cfg = GenerationConfig(batch_size=1, audio_format="wav")
  result = generate_music(dit, lm, params, cfg)
  # result.audios[0]["path"] is the WAV file

To keep ``backend.py`` and ``modes.py`` clean, this module exposes a
single ``ACEStepStudio`` wrapper that owns both handlers and exposes a
``generate(params: dict) -> str`` method returning the audio path.
``get_pipeline()`` returns the lazy singleton wrapper.

Checkpoints live under ``{project_root}/checkpoints/{config_path}/``.
On Mac with the apple-silicon fork, the fork auto-downloads from
HuggingFace if a checkpoint is missing, but in practice we pre-download
via ``hf download`` before the first inference call to avoid pytest
timeouts.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _REPO_ROOT / "checkpoints"
_OUTPUT_DIR = _REPO_ROOT / "output"

_DEFAULT_DIT_CONFIG = "acestep-v15-xl-sft"
_DEFAULT_LM_MODEL = "acestep-5Hz-lm-1.7B"


def detect_device() -> str:
    """Returns 'cuda', 'mps', or 'cpu' in priority order."""
    try:
        import torch  # local import: keep module import cheap for CI
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    # macOS: torch.backends.mps appeared in 2.0; guard for the rare absence
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def vram_limit_for(device: str) -> int | None:
    """Returns a VRAM cap in bytes for CUDA, None otherwise.

    ``torch.mps`` has no ``mem_get_info`` — calling DiffSynth-style
    free-VRAM gates with a numeric limit would crash on MPS. Returning
    None lets the pipeline short-circuit those checks.
    """
    if device != "cuda":
        return None
    try:
        import torch

        free, _total = torch.cuda.mem_get_info()
        # Leave 2 GiB headroom for activations
        return max(0, free - 2 * 1024**3)
    except Exception:
        return None


class ACEStepStudio:
    """Wrapper around the apple-silicon fork's split-handler API.

    Owns one ``AceStepHandler`` (DiT) and one ``LLMHandler`` (5Hz LM
    planner). Both are lazy-loaded on the first ``generate(...)`` call.
    """

    def __init__(
        self,
        dit_config: str | None = None,
        lm_model: str | None = None,
        device: str | None = None,
    ) -> None:
        self._dit = None
        self._llm = None
        self._dit_config = dit_config or os.environ.get("ACE_DIT_CONFIG", _DEFAULT_DIT_CONFIG)
        self._lm_model = lm_model or os.environ.get("ACE_LM_MODEL", _DEFAULT_LM_MODEL)
        self._device = device or detect_device()

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._dit is not None and self._llm is not None

    def _ensure_loaded(self) -> None:
        """First-call lazy load of both handlers. Heavy imports stay local."""
        if self.is_loaded:
            return

        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        dit = AceStepHandler()
        dit.initialize_service(
            project_root=str(_REPO_ROOT),
            config_path=self._dit_config,
            device=self._device,
        )

        llm = LLMHandler()
        llm.initialize(
            checkpoint_dir=str(_CHECKPOINTS_DIR),
            lm_model_path=self._lm_model,
            backend="vllm",  # fork auto-routes to mlx on mps + mlx-lm installed
            device=self._device,
        )

        self._dit = dit
        self._llm = llm

    def generate(self, params: dict) -> str:
        """Run a single song generation across all four modes.

        ``params`` is the dict produced by the mode handlers in ``modes.py``.
        The ``params["mode"]`` key (``generate`` | ``cover`` | ``extend`` |
        ``edit``) selects the ACE-Step ``task_type`` and which audio inputs
        get wired through to ``GenerationParams``:

        - ``generate``: ``task_type="text2music"``
        - ``cover``:    ``task_type="cover"`` + ``reference_audio`` +
          ``audio_cover_strength``
        - ``extend``:   ``task_type="repaint"`` + ``src_audio`` set to the
          seed, with ``repainting_start=-1`` / ``repainting_end=-1`` as a
          sentinel meaning "paint after the end of the seed". The actual
          mask shaping ultimately lives inside ACE-Step's repaint path.
        - ``edit``:     ``task_type="repaint"`` + ``src_audio`` + explicit
          ``[segment_start_s, segment_end_s]`` segment bounds.

        Flow-edit (``sub_mode="flow_edit"``) is implemented as a repaint
        pass: the installed ACE-Step ``GenerationParams`` dataclass has no
        native ``flow_edit_*`` fields, so the extra flow-edit knobs carried
        in the internal params dict are ignored at the ``GenerationParams``
        instantiation level and will need wiring once upstream grows them.

        Returns the path to the produced audio file.
        """
        self._ensure_loaded()

        from acestep.inference import (
            GenerationConfig,
            GenerationParams,
            generate_music,
        )

        advanced = params.get("advanced", {}) or {}
        lm_opts = params.get("lm", {}) or {}
        mode = params.get("mode", "generate")

        # Map our internal dict to ACE-Step's GenerationParams.
        # Lyrics "[Instrumental]" is the ACE-Step convention for instrumental.
        lyrics = params.get("lyrics", "") or params.get("extension_lyrics", "") or ""
        if mode == "edit":
            lyrics = params.get("target_lyrics", "") or lyrics
        instrumental = bool(params.get("instrumental", False))
        if instrumental and not lyrics:
            lyrics = "[Instrumental]"

        # Mode-specific task_type + audio inputs.
        # All five fields below MUST resolve before we instantiate
        # GenerationParams so that the dataclass ctor sees consistent values.
        ref_audio: str | None = None
        src_audio: str | None = None
        audio_cover_strength = 0.0
        repainting_start = 0.0
        repainting_end = -1.0

        if mode == "generate":
            task_type = "text2music"
        elif mode == "cover":
            task_type = "cover"
            ref_audio = params.get("ref_audio")
            audio_cover_strength = float(params.get("audio_cover_strength", 0.93))
        elif mode == "extend":
            task_type = "repaint"
            src_audio = params.get("seed_audio")
            # Sentinel: -1 / -1 means "append after the seed audio's end".
            # ACE-Step's repaint path interprets these bounds against the
            # src_audio duration; the actual semantics need verifying once
            # we run a full pass on real hardware (M3 GPU smoke).
            repainting_start = -1.0
            repainting_end = -1.0
        elif mode == "edit":
            task_type = "repaint"
            src_audio = params.get("source_audio")
            repainting_start = float(params.get("segment_start_s", 0.0))
            repainting_end = float(params.get("segment_end_s", 30.0))
            # flow_edit sub-mode: lower audio_cover_strength to allow style
            # drift while still using the repaint task type. The extra
            # flow_* fields in our internal params dict are kept around for
            # future use but not forwarded to GenerationParams (no native
            # support in the installed dataclass).
            if params.get("sub_mode") == "flow_edit":
                audio_cover_strength = 0.3
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        # Caption can come from the per-mode handlers under different keys.
        caption = (
            params.get("prompt") or params.get("extra_prompt") or params.get("flow_source_caption") or ""
        )
        duration_s = int(params.get("duration_s") or params.get("extra_duration_s") or 30)

        # ``advanced``/``lm`` dicts are sent by app.py's
        # ``_build_advanced_params``. Key changes from the prior contract:
        # - ``inference_steps`` (was ``steps``, defaulted to 8 which made the
        #   XL SFT model behave too turbo-ish; new default 27).
        # - ``guidance_scale`` (was ``cfg``, default 7.0 for stronger prompt
        #   adherence).
        # - ``infer_method`` (new — ``"ode"`` deterministic / ``"sde"``
        #   stochastic; the user can now flip to ``sde`` to actually get
        #   different output each click even with the same seed).
        # - ``use_adg`` (new — Adaptive Dual Guidance; experimental).
        # - ``thinking`` (5Hz LM CoT — default flips to True so the LM can
        #   reason about caption + metadata, which is the actual source of
        #   the "no matter what prompt the style barely changes" symptom).
        # - ``use_cot_metas`` / ``use_cot_caption`` / ``use_cot_language``
        #   keys renamed from ``cot_*`` for consistency with the dataclass.
        gen_params = GenerationParams(
            task_type=task_type,
            caption=caption,
            lyrics=lyrics,
            instrumental=instrumental,
            duration=duration_s,
            seed=int(params.get("seed", -1)),
            inference_steps=int(advanced.get("inference_steps", 27)),
            guidance_scale=float(advanced.get("guidance_scale", 7.0)),
            infer_method=str(advanced.get("infer_method", "ode")),
            use_adg=bool(advanced.get("use_adg", False)),
            shift=float(advanced.get("shift", 1.0)),
            bpm=advanced.get("bpm"),
            keyscale=advanced.get("keyscale", ""),
            timesignature=advanced.get("timesignature", ""),
            vocal_language=advanced.get("vocal_language", "unknown"),
            cfg_interval_start=float(advanced.get("cfg_interval_start", 0.0)),
            cfg_interval_end=float(advanced.get("cfg_interval_end", 1.0)),
            # Mode-specific audio inputs + repaint bounds
            reference_audio=ref_audio,
            src_audio=src_audio,
            audio_cover_strength=audio_cover_strength,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
            # 5Hz language model knobs — defaults flipped to True so the
            # LM actually reasons about each prompt instead of returning
            # blank captions / metadata back to the DiT.
            thinking=bool(lm_opts.get("thinking", True)),
            lm_temperature=float(lm_opts.get("temperature", 0.85)),
            lm_cfg_scale=float(lm_opts.get("cfg", 2.0)),
            lm_top_k=int(lm_opts.get("top_k", 0)),
            lm_top_p=float(lm_opts.get("top_p", 0.9)),
            lm_negative_prompt=lm_opts.get("negative_prompt", "NO USER INPUT"),
            use_cot_metas=bool(lm_opts.get("use_cot_metas", True)),
            use_cot_caption=bool(lm_opts.get("use_cot_caption", True)),
            use_cot_language=bool(lm_opts.get("use_cot_language", True)),
        )

        gen_config = GenerationConfig(
            batch_size=1,
            audio_format=advanced.get("audio_format", "wav"),
            use_random_seed=False,
            seeds=[int(params.get("seed", 1))],
        )

        # generate_music only writes a file when save_dir is provided; otherwise
        # result.audios[i]["path"] is empty and ["tensor"] holds the raw audio.
        # Pass an explicit output dir so the path is always usable.
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        result = generate_music(
            self._dit,
            self._llm,
            gen_params,
            gen_config,
            save_dir=str(_OUTPUT_DIR),
        )

        if not result.success:
            raise RuntimeError(f"ACE-Step generation failed: {result.error}")
        if not result.audios:
            raise RuntimeError("ACE-Step returned no audio outputs")

        audio = result.audios[0]
        path = audio.get("path") or ""
        if not path:
            # generate_music returned an empty path despite save_dir being passed.
            # Fall back to writing the in-memory tensor so callers always get a
            # valid file path (Gradio cannot serve an empty path).
            import soundfile as sf

            tensor = audio.get("tensor")
            if tensor is None:
                raise RuntimeError("ACE-Step returned neither an audio path nor a tensor")
            sample_rate = int(audio.get("sample_rate", 48000))
            audio_format = advanced.get("audio_format", "wav")
            fallback = _OUTPUT_DIR / f"{audio.get('key', 'fallback')}.{audio_format}"
            data = tensor.detach().cpu().numpy()
            # soundfile expects (frames, channels); acestep tensors are (channels, frames)
            if data.ndim == 2 and data.shape[0] in (1, 2):
                data = data.T
            sf.write(str(fallback), data, sample_rate)
            path = str(fallback)

        return path


_PIPELINE: ACEStepStudio | None = None  # module-level lazy singleton


def get_pipeline() -> ACEStepStudio:
    """Lazy-construct the ACE Music Studio wrapper.

    The wrapper itself is cheap to construct; both handlers (DiT, LM)
    are only loaded on the first ``generate(...)`` call.
    """
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = ACEStepStudio()
    return _PIPELINE
