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

_DEFAULT_DIT_CONFIG = "acestep-v15-xl-sft"
_DEFAULT_LM_MODEL = "acestep-5Hz-lm-0.6B"


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
        """Run a single text→song generation.

        ``params`` is the dict produced by ``modes.generate``:
        ``{"prompt", "lyrics", "duration_s", "instrumental", "seed",
        "loras", "advanced", "lm", "dcw"}``. Returns the path to the
        produced audio file.
        """
        self._ensure_loaded()

        from acestep.inference import (
            GenerationConfig,
            GenerationParams,
            generate_music,
        )

        advanced = params.get("advanced", {}) or {}
        lm_opts = params.get("lm", {}) or {}

        # Map our internal dict to ACE-Step's GenerationParams.
        # Lyrics "[Instrumental]" is the ACE-Step convention for instrumental.
        lyrics = params.get("lyrics", "") or ""
        instrumental = bool(params.get("instrumental", False))
        if instrumental and not lyrics:
            lyrics = "[Instrumental]"

        gen_params = GenerationParams(
            task_type="text2music",
            caption=params.get("prompt", ""),
            lyrics=lyrics,
            instrumental=instrumental,
            duration=int(params.get("duration_s", 30)),
            seed=int(params.get("seed", -1)),
            inference_steps=int(advanced.get("steps", 32)),
            guidance_scale=float(advanced.get("cfg", 4.0)),
            shift=float(advanced.get("shift", 1.0)),
            bpm=advanced.get("bpm"),
            keyscale=advanced.get("keyscale", ""),
            timesignature=advanced.get("timesignature", ""),
            vocal_language=advanced.get("vocal_language", "unknown"),
            cfg_interval_start=float(advanced.get("cfg_interval_start", 0.0)),
            cfg_interval_end=float(advanced.get("cfg_interval_end", 1.0)),
            thinking=bool(lm_opts.get("thinking", False)),
            lm_temperature=float(lm_opts.get("temperature", 0.85)),
            lm_cfg_scale=float(lm_opts.get("cfg", 2.0)),
            lm_top_k=int(lm_opts.get("top_k", 0)),
            lm_top_p=float(lm_opts.get("top_p", 0.9)),
            lm_negative_prompt=lm_opts.get("negative_prompt", ""),
            use_cot_metas=bool(lm_opts.get("cot_metas", False)),
            use_cot_caption=bool(lm_opts.get("cot_caption", False)),
            use_cot_language=bool(lm_opts.get("cot_language", False)),
        )

        gen_config = GenerationConfig(
            batch_size=1,
            audio_format=advanced.get("audio_format", "wav"),
            use_random_seed=False,
            seeds=[int(params.get("seed", 1))],
        )

        result = generate_music(self._dit, self._llm, gen_params, gen_config)

        if not result.success:
            raise RuntimeError(f"ACE-Step generation failed: {result.error}")
        if not result.audios:
            raise RuntimeError("ACE-Step returned no audio outputs")

        return result.audios[0]["path"]


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
