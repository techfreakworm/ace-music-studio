"""Qwen 2.5 7B Instruct as the lyrics writer.

Mac path: ``mlx-lm`` with the 4-bit MLX quantisation (``mlx-community/
Qwen2.5-7B-Instruct-4bit``) for speed and a low VRAM footprint on Apple
Silicon. The 4-bit pack is ~4 GB on disk and runs in ~8-12 s per draft
on an M5 Max after the first warm-up.

CUDA / CPU path: ``transformers`` with the full ``Qwen/Qwen2.5-7B-Instruct``
checkpoint, ``apply_chat_template`` for the prompt, and ``do_sample=True``
generation.

Loading is lazy — the module-level ``_LM`` singleton is constructed on the
first call to ``_get_lm()`` so module import stays fast for CI and so the
~4 GB MLX download is only triggered when the user actually clicks
"▶ Draft lyrics" in the Lyrics tab.

Tests in ``tests/test_lyrics_lm.py`` mock ``_get_lm`` at the module
boundary so the real Qwen weights are never loaded in CI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import logging

import ace_pipeline as ap

_DEFAULT_MAC_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
_DEFAULT_CUDA_ID = "Qwen/Qwen2.5-7B-Instruct"

_LM = None  # lazy module-level singleton
_log = logging.getLogger("ams.lyrics")


def build_system_prompt() -> str:
    """Locked songwriter system prompt for the Lyrics tab.

    Returns a single multi-line string that instructs Qwen to emit ONLY
    structurally-tagged lyrics (``[intro]`` ``[verse 1]`` ``[chorus]``
    etc.). The exact tag vocabulary is what ACE-Step's 5Hz LM planner
    expects downstream when the user pipes the draft into the Generate
    tab via the "Use these in Generate" button.
    """
    return (
        "You are a songwriter. Output ONLY structured lyrics for an AI music generator.\n"
        "Use these section tags exactly: [intro] [verse 1] [verse 2] [chorus] [bridge] [outro] (etc.)\n"
        "Each section is on its own line, followed by the lyrics for that section. "
        "Keep verses 4-8 lines, choruses 4 lines, bridges 2-4 lines. "
        "Match the requested tone and language. "
        "Do not include commentary, headers, or markdown."
    )


def _build_user_prompt(
    brief: str,
    structure: str,
    language: str,
    tone: str,
    verse_lines: int,
    chorus_lines: int,
    bridge_lines: int,
    rhyme: str,
) -> str:
    return (
        f"Write lyrics with this structure: {structure}.\n"
        f"Language: {language}. Tone: {tone or 'neutral'}. Rhyme: {rhyme}.\n"
        f"Verse: {verse_lines} lines. Chorus: {chorus_lines} lines. Bridge: {bridge_lines} lines.\n\n"
        f"Brief:\n{brief}\n"
    )


def _normalise(text: str) -> str:
    """Lowercase section tags and strip outer whitespace.

    Qwen occasionally emits ``[Verse 1]`` or ``[CHORUS]`` despite the
    system prompt asking for lowercase tags. ACE-Step's 5Hz LM expects
    lowercase, so we coerce here rather than relying on every downstream
    consumer to lowercase before parsing.
    """

    def lower_tag(match: re.Match[str]) -> str:
        return "[" + match.group(1).lower() + "]"

    return re.sub(r"\[([^\]]+)\]", lower_tag, text).strip()


def _get_lm():
    """Return the lazy module-level LM singleton.

    Tests in ``tests/test_lyrics_lm.py`` monkeypatch this function so
    ``_load_lm()`` is never invoked under pytest. In production the
    first call constructs the singleton once and caches it for the
    process lifetime.
    """
    global _LM
    if _LM is None:
        _LM = _load_lm()
    return _LM


def _load_hflm(device: str) -> "_HFLM":
    """Load Qwen 2.5 7B via transformers on the given device (mps/cuda/cpu)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(_DEFAULT_CUDA_ID)
    model = AutoModelForCausalLM.from_pretrained(_DEFAULT_CUDA_ID, torch_dtype=dtype).to(device)
    return _HFLM(model=model, tokenizer=tok)


def _load_lm():
    """Construct the per-device LM wrapper.

    On MPS, try mlx-lm first (4-bit, fast). If MLX fails to load, fall back
    to transformers on MPS. On CUDA/CPU use transformers directly.
    """
    device = ap.detect_device()
    if device == "mps":
        try:
            from mlx_lm import load  # type: ignore[import-not-found]

            model, tokenizer = load(_DEFAULT_MAC_ID)
            return _MLXLM(model=model, tokenizer=tokenizer)
        except Exception as exc:
            _log.warning("MLX load failed (%s); falling back to transformers on MPS", exc)
            return _load_hflm("mps")

    # CUDA / CPU path.
    return _load_hflm(device)


@dataclass
class _MLXLM:
    """mlx-lm wrapper. ``generate`` returns a plain string (post-decode)."""

    model: Any
    tokenizer: Any

    def generate(self, system: str, user: str, **kw: Any) -> str:
        import mlx.core as mx  # type: ignore[import-not-found]
        import mlx_lm.generate as mlx_gen_mod  # type: ignore[import-not-found]
        from mlx_lm import generate  # type: ignore[import-not-found]

        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            with mx.stream(mx.gpu):
                mlx_gen_mod.generation_stream = mx.new_stream(mx.default_device())
                return generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=int(kw.get("max_new_tokens", 600)),
                )
        except RuntimeError as exc:
            _log.warning("MLX generate failed (%s); switching to transformers on MPS", exc)
            global _LM
            _LM = _load_hflm("mps")
            return _LM.generate(system, user, **kw)


@dataclass
class _HFLM:
    """transformers wrapper. ``generate`` returns the assistant continuation."""

    model: Any
    tokenizer: Any

    def generate(self, system: str, user: str, **kw: Any) -> str:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=int(kw.get("max_new_tokens", 600)),
            temperature=float(kw.get("temperature", 0.85)),
            top_p=float(kw.get("top_p", 0.9)),
            top_k=int(kw.get("top_k", 40)),
            repetition_penalty=float(kw.get("repetition_penalty", 1.1)),
            do_sample=True,
        )
        # Slice off the prompt tokens at the *token* level. Doing it at the
        # string level (full.startswith(prompt)) is brittle because
        # ``skip_special_tokens=True`` strips the ChatML markers from
        # ``full`` but they're still present in ``prompt`` — so the prefix
        # match fails and the system + user turns leak into the output.
        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = out[0][prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_lyrics(
    brief: str,
    structure: str,
    language: str,
    tone: str,
    verse_lines: int,
    chorus_lines: int,
    bridge_lines: int,
    rhyme: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int | None = None,
) -> str:
    """Draft structurally-tagged lyrics for the Lyrics tab.

    Builds the user prompt from the form fields, asks the LM to generate,
    and runs the output through ``_normalise()`` so section tags are
    lowercase. ``seed`` is accepted for parity with the UI but is not
    threaded through the mlx-lm / transformers ``generate`` calls because
    neither backend's high-level ``generate(...)`` helper accepts a seed
    in the version we ship with — deterministic seeding would require
    dropping to the per-step token loop, which we'll add if reproducibility
    becomes a hard requirement.
    """
    lm = _get_lm()
    user = _build_user_prompt(
        brief, structure, language, tone, verse_lines, chorus_lines, bridge_lines, rhyme
    )
    raw = lm.generate(
        system=build_system_prompt(),
        user=user,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )
    return _normalise(raw)
