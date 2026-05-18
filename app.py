"""ACE Music Studio — Gradio entrypoint.

UI ARCHITECTURE (locked — read this before editing):

The five "modes" (Generate / Cover / Extend / Edit / Lyrics) are NOT
implemented via ``gr.Tabs``. The wireframes at
``docs/superpowers/specs/mockups/`` show a LEFT sidebar with mode pills +
a session History section, and a single content column on the right.

The implementation pattern is:

  gr.Row(elem_classes=["ams-body"])
  ├── gr.Column(min_width=190, elem_classes=["ams-sidebar"])
  │   ├── gr.Radio(label=None, elem_classes=["ams-side-radio"])   ← 5 mode choices
  │   └── gr.HTML(... "History · session" ...)
  └── gr.Column(elem_classes=["ams-content"])
      ├── gr.Group(visible=True)  ← pane_generate
      ├── gr.Group(visible=False) ← pane_cover
      ├── gr.Group(visible=False) ← pane_extend
      ├── gr.Group(visible=False) ← pane_edit
      └── gr.Group(visible=False) ← pane_lyrics

The Radio's ``change`` event fires ``_switch_pane(mode)`` which returns
visibility updates for the five Groups. The Radio's native ``:checked``
state gives us the sidebar "active item" highlight for free via CSS
(see ``theme.CSS`` for ``.ams-side-radio`` selectors).

DO NOT switch this back to ``gr.Tabs`` — that produces top-positioned
horizontal tabs which contradicts the wireframes.

On HF Spaces (``SPACE_ID`` env present), ``_bootstrap_spaces_cache()``
runs once on import to symlink HF hub cache snapshots into the
acestep-apple-silicon fork's ``<site-packages>/checkpoints/`` layout.
On Mac/Linux locally, it's a no-op.
"""

from __future__ import annotations

import os

# Set MPS fallback BEFORE any torch import path is taken.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Don't pin HF download source — let HF default for both Spaces and local cache.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import hashlib
import random
from pathlib import Path

import gradio as gr

import ace_pipeline
import backend as be
import lora_stack
import modes
import post_process
import theme
import ui

_BACKEND: be.ACEStepStudioBackend | None = None


def get_backend() -> be.ACEStepStudioBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = be.ACEStepStudioBackend()
    return _BACKEND


def _bootstrap_spaces_cache() -> None:
    """On HF Spaces, mirror the HF hub cache into the site-packages checkpoints/ dir.

    The acestep-apple-silicon fork resolves checkpoints relative to its own install
    location (``.venv/.../site-packages/checkpoints/``). HF Spaces puts model weights
    in the HF hub cache. This bootstrap snapshot-downloads the two required repos
    into the cache (using preload mirror if available) and then symlinks each
    snapshot child into ``checkpoints/`` so the fork's resolver finds them.

    Local Mac/CUDA: no-op (guarded by ``SPACE_ID`` env var).
    """
    if not os.getenv("SPACE_ID"):
        return

    import site

    from huggingface_hub import snapshot_download

    site_pkgs = site.getsitepackages()[0]
    target_dir = Path(site_pkgs) / "checkpoints"

    if target_dir.exists():
        return  # already bootstrapped

    hf_home = os.getenv("HF_HOME", "/home/user/.cache/huggingface")

    # Download Ace-Step1.5 umbrella (vae + encoder).
    umbrella_path = snapshot_download(
        repo_id="ACE-Step/Ace-Step1.5",
        cache_dir=hf_home,
    )

    # Download the XL SFT diffusion variant.
    xl_sft_path = snapshot_download(
        repo_id="ACE-Step/acestep-v15-xl-sft",
        cache_dir=hf_home,
    )

    # Merge both snapshots into ``checkpoints/`` via symlinks.
    target_dir.mkdir(parents=True, exist_ok=True)
    for src_path in [umbrella_path, xl_sft_path]:
        for child in Path(src_path).iterdir():
            link = target_dir / child.name
            if not link.exists():
                link.symlink_to(child)


def _maybe_spaces_gpu():
    """Return ``@spaces.GPU(duration=180)`` on HF Spaces, otherwise a no-op decorator.

    The decorator MUST be applied at module load time — ZeroGPU's startup
    analyzer doesn't see runtime decoration. Local dev is a transparent pass-through.
    """
    if os.getenv("SPACE_ID"):
        try:
            import spaces

            return spaces.GPU(duration=180)
        except ImportError:
            pass

    def _noop(fn):
        return fn

    return _noop


# Run cache bootstrap at module import so HF Spaces' startup analyzer sees
# the symlinks before the lazy backend singleton is constructed on first click.
_bootstrap_spaces_cache()


def _safe_call(fn, *args, **kwargs):
    """Wrap a mode handler so all known exceptions become friendly gr.Error toasts.

    Centralising this here lets every on_*_click handler stay a single-line
    call into modes.* without each one repeating the try/except mosaic. The
    error classes mirror what each mode handler can actually raise:

    - ``lora_stack.LoRAValidationError`` — uploaded LoRA isn't compatible
    - ``ValueError`` — mode-handler param validation (missing prompt, etc.)
    - ``FileNotFoundError`` — user-supplied ref_audio path doesn't exist
    - ``RuntimeError`` — pipeline crash, including MPS op-fallback failures
    """
    try:
        return fn(*args, **kwargs)
    except lora_stack.LoRAValidationError as e:
        raise gr.Error(str(e)) from e
    except ValueError as e:
        raise gr.Error(str(e)) from e
    except FileNotFoundError as e:
        raise gr.Error(f"File not found: {e}") from e
    except RuntimeError as e:
        msg = str(e)
        if "MPS" in msg or "mps" in msg:
            raise gr.Error(f"Apple Silicon op issue: {msg}. PYTORCH_ENABLE_MPS_FALLBACK is enabled.") from e
        raise gr.Error(f"Generation failed: {msg}") from e


def _sha256(path: str) -> str:
    """Stream a file through SHA-256 in 64 KB chunks.

    Used to fingerprint the active LoRA so the generation metadata
    includes a provenance hash (useful when the user uploads variants
    of the same psytrance fine-tune with subtly different weights).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _active_md(name: str, scale: float, kind: str) -> str:
    """Format the 'Active: …' line shown under the strength slider."""
    return f"**Active:** `{name}` &nbsp;·&nbsp; scale `{scale:.2f}` &nbsp;·&nbsp; {kind}"


def on_lora_preset_change(preset_name: str, strength: float):
    """User picked a preset (or 'None'). Downloads + validates + sets state.

    Returns (state, active_markdown, upload_clear_value) — the third
    value clears any custom-upload widget so the two inputs stay
    mutually exclusive.
    """
    if preset_name == "None" or not preset_name:
        return None, "_No LoRA active_", None

    try:
        local_path = lora_stack.download_preset(preset_name)
    except lora_stack.LoRAValidationError as e:
        raise gr.Error(str(e)) from e

    info = lora_stack.sniff(local_path)
    if not info.compatible:
        raise gr.Error(
            f"Preset {preset_name!r} is not compatible with ACE-Step 1.5 XL SFT: {info.diagnostic}"
        )

    state = {
        "name": preset_name,
        "scale": float(strength),
        "path": str(local_path),
        "sha256": _sha256(str(local_path)),
    }
    return state, _active_md(preset_name, float(strength), "preset"), None


def on_lora_upload(file_obj, strength: float):
    """User dropped a custom .safetensors. Replaces any active preset.

    Returns (state, active_markdown, preset_reset_value) — the third
    value resets the preset radio to 'None' so the two inputs stay
    mutually exclusive.
    """
    if file_obj is None:
        return None, "_No LoRA active_", "None"

    path_str = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    try:
        info = lora_stack.sniff(path_str)
    except lora_stack.LoRAValidationError as e:
        raise gr.Error(str(e)) from e

    if not info.compatible:
        raise gr.Error(f"Uploaded LoRA isn't compatible with ACE-Step 1.5 XL SFT: {info.diagnostic}")

    name = Path(path_str).stem
    state = {
        "name": name,
        "scale": float(strength),
        "path": path_str,
        "sha256": _sha256(path_str),
    }
    return state, _active_md(name, float(strength), "custom"), "None"


def on_lora_strength_change(state, strength: float):
    """User dragged the strength slider. Update scale on the active LoRA.

    No-op if no LoRA is active.
    """
    if not state:
        return state, "_No LoRA active_"
    new_state = {**state, "scale": float(strength)}
    # Preserve the "preset" vs "custom" tag — presets resolve to a path
    # under the HF cache (~/.cache/huggingface/hub/…), uploads land
    # under /tmp/gradio/… or the user's pwd. Use the same heuristic
    # the upload/preset handlers used: a path inside the HF cache or
    # snapshot tree counts as preset, otherwise custom.
    path = str(new_state.get("path", ""))
    kind = "preset" if (".cache/huggingface" in path or "snapshots" in path) else "custom"
    return new_state, _active_md(new_state["name"], float(strength), kind)


@_maybe_spaces_gpu()
def on_generate_click(
    prompt: str,
    lyrics: str,
    duration_s: float,
    instrumental_label: str,
    lora_state,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    loras = [lora_state] if lora_state else []
    out_path, meta = _safe_call(
        modes.generate,
        get_backend(),
        params={
            "prompt": prompt,
            "lyrics": lyrics,
            "duration_s": int(duration_s),
            "instrumental": instrumental_label == "Instrumental",
            "seed": random.randint(1, 2_147_483_647),
            "loras": loras,
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )
    new_history = _history_push("generate", prompt or "(no prompt)")
    return out_path, meta, new_history


@_maybe_spaces_gpu()
def on_cover_click(
    ref_audio,
    prompt: str,
    lyrics: str,
    duration_s: float,
    audio_cover_strength: float,
    lora_state,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    """Cover-mode click. ref_audio is a filepath from gr.Audio(type='filepath')."""
    loras = [lora_state] if lora_state else []
    out_path, meta = _safe_call(
        modes.cover,
        get_backend(),
        params={
            "ref_audio": ref_audio,
            "prompt": prompt,
            "lyrics": lyrics,
            "duration_s": int(duration_s),
            "audio_cover_strength": float(audio_cover_strength),
            "seed": random.randint(1, 2_147_483_647),
            "loras": loras,
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )
    new_history = _history_push("cover", prompt or "(cover)")
    return out_path, meta, new_history


@_maybe_spaces_gpu()
def on_extend_click(
    seed_audio,
    extra_prompt: str,
    extension_lyrics: str,
    extra_duration_s: float,
    wav_crossfade_s: float,
    repaint_mode: str,
    repaint_strength: float,
    latent_crossfade_frames: float,
    chunk_mask_mode: str,
    lora_state,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    """Extend-mode click. seed_audio is a filepath from gr.Audio(type='filepath')."""
    loras = [lora_state] if lora_state else []
    out_path, meta = _safe_call(
        modes.extend,
        get_backend(),
        params={
            "seed_audio": seed_audio,
            "extra_prompt": extra_prompt,
            "extension_lyrics": extension_lyrics,
            "extra_duration_s": int(extra_duration_s),
            "wav_crossfade_s": float(wav_crossfade_s),
            "repaint_mode": repaint_mode,
            "repaint_strength": float(repaint_strength),
            "latent_crossfade_frames": int(latent_crossfade_frames),
            "chunk_mask_mode": chunk_mask_mode,
            "seed": random.randint(1, 2_147_483_647),
            "loras": loras,
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )
    new_history = _history_push("extend", extra_prompt or "(extend)")
    return out_path, meta, new_history


@_maybe_spaces_gpu()
def on_draft_lyrics(
    brief: str,
    structure: str,
    language: str,
    tone: str,
    verse_lines: float,
    chorus_lines: float,
    bridge_lines: float,
    rhyme: str,
    temperature: float,
    top_p: float,
    top_k: float,
    max_new_tokens: float,
    seed,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    """Lyrics-mode click. Calls ``modes.lyrics(...)`` directly — no ACE-Step
    pipeline is touched. Qwen 2.5 7B is its own lazy singleton inside
    ``lyrics_lm``; the first click triggers a ~4 GB MLX download (cached
    afterwards) and ~30 s warm-up before the draft appears.
    """
    lyrics_text, meta = _safe_call(
        modes.lyrics,
        get_backend(),
        params={
            "brief": brief,
            "structure": structure,
            "language": language,
            "tone": tone,
            "verse_lines": int(verse_lines),
            "chorus_lines": int(chorus_lines),
            "bridge_lines": int(bridge_lines),
            "rhyme": rhyme,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_new_tokens": int(max_new_tokens),
            "seed": int(seed) if seed is not None else None,
        },
    )
    new_history = _history_push("lyrics", brief or "(brief)")
    return lyrics_text, meta, new_history


def on_separate_stems(audio_path):
    """Run Demucs on the current Output audio and surface 4 stem files."""
    if not audio_path:
        raise gr.Error("Generate a song first.")
    try:
        stems = post_process.separate_stems(audio_path)
    except Exception as e:
        raise gr.Error(f"Demucs failed: {e}") from e
    return gr.Files(value=list(stems.values()), visible=True)


def on_normalise(audio_path):
    """Run pyloudnorm at -14 LUFS and surface the normalised WAV."""
    if not audio_path:
        raise gr.Error("Generate a song first.")
    try:
        out = post_process.normalise_lufs(audio_path, target_lufs=-14.0)
    except Exception as e:
        raise gr.Error(f"Normalisation failed: {e}") from e
    return gr.Audio(value=str(out), visible=True)


def on_export_mp3(audio_path):
    """Encode the current Output to MP3 320 k via ffmpeg and surface the file."""
    if not audio_path:
        raise gr.Error("Generate a song first.")
    try:
        out = post_process.to_mp3(audio_path, bitrate_kbps=320)
    except Exception as e:
        raise gr.Error(f"MP3 export failed: {e}") from e
    return gr.File(value=str(out), visible=True)


@_maybe_spaces_gpu()
def on_edit_click(
    source_audio,
    sub_mode: str,
    source_lyrics: str,
    target_lyrics: str,
    segment_start_s: float,
    segment_end_s: float,
    repaint_strength: float,
    repaint_mode: str,
    flow_source_caption: str,
    flow_n_min: float,
    flow_n_max: float,
    flow_n_avg: float,
    lora_state,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    """Edit-mode click. source_audio is a filepath from gr.Audio(type='filepath')."""
    loras = [lora_state] if lora_state else []
    out_path, meta = _safe_call(
        modes.edit,
        get_backend(),
        params={
            "source_audio": source_audio,
            "sub_mode": sub_mode,
            "source_lyrics": source_lyrics,
            "target_lyrics": target_lyrics,
            "segment_start_s": float(segment_start_s),
            "segment_end_s": float(segment_end_s),
            "repaint_strength": float(repaint_strength),
            "repaint_mode": repaint_mode,
            "flow_source_caption": flow_source_caption,
            "flow_n_min": float(flow_n_min),
            "flow_n_max": float(flow_n_max),
            "flow_n_avg": int(flow_n_avg),
            "seed": random.randint(1, 2_147_483_647),
            "loras": loras,
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )
    new_history = _history_push("edit", target_lyrics or sub_mode or "(edit)")
    return out_path, meta, new_history


HEADER_HTML = """
<div class="ams-header">
  <div>
    <div class="ams-brand">ACE Music Studio<span class="ams-brand-period">.</span></div>
  </div>
  <div class="ams-status" id="ams-status">ready</div>
</div>
""".strip()


def _status_html(device: str) -> str:
    """Right-aligned status indicator in the header. Updated at boot only."""
    return f"""
<div class="ams-header">
  <div>
    <div class="ams-brand">ACE Music Studio<span class="ams-brand-period">.</span></div>
  </div>
  <div class="ams-status">ready · {device.upper()}</div>
</div>
""".strip()


CTA_HTML = """
<div class="ams-cta">
  Built with <span class="ams-cta-heart">♥</span>.
  <strong>Drop a like</strong> at the top
  &nbsp;·&nbsp;
  Follow <a href="https://huggingface.co/techfreakworm" target="_blank" rel="noopener noreferrer"><strong>@techfreakworm</strong></a>
  for what's next.
</div>
""".strip()


HISTORY_HTML = """
<div class="ams-history">
  <div class="ams-history-title">History · session</div>
  <div class="ams-history-empty">No generations yet</div>
</div>
""".strip()


# --- In-memory history (M6/H2) ----------------------------------------------
# Per spec §13, persistent history is out of scope for v1. The sidebar block
# is an in-process list that lives for the lifetime of the Gradio process and
# resets on reload. Newest entries first; capped at _HISTORY_MAX so the
# bordered sidebar stays compact at the desktop breakpoint.
_HISTORY: list[dict] = []
_HISTORY_MAX = 12


def _history_render() -> str:
    """Render _HISTORY into the sidebar HTML block.

    Falls back to the empty-state HTML constant when no rows are present so
    the placeholder copy stays exactly aligned with the wireframe.
    """
    if not _HISTORY:
        return HISTORY_HTML
    rows_html = "\n".join(
        f'<div class="ams-history-row" title="{h["label"]}">'
        f'<span class="ams-history-mode">{h["mode"]}</span>'
        f'<span class="ams-history-label">{h["label"]}</span>'
        f"</div>"
        for h in _HISTORY
    )
    return f'<div class="ams-history"><div class="ams-history-title">History · session</div>{rows_html}</div>'


def _history_push(mode: str, label: str) -> str:
    """Push a generation onto the history and return the new HTML."""
    _HISTORY.insert(0, {"mode": mode, "label": (label or "").strip()[:30] or "(untitled)"})
    while len(_HISTORY) > _HISTORY_MAX:
        _HISTORY.pop()
    return _history_render()


MODE_CHOICES = [
    ("🎵 Generate", "generate"),
    ("🎤 Cover", "cover"),
    ("⏩ Extend", "extend"),
    ("✏️ Edit", "edit"),
    ("✍️ Lyrics", "lyrics"),
]


def build_app() -> gr.Blocks:
    device = ace_pipeline.detect_device()

    with gr.Blocks(theme=theme.build_theme(), css=theme.CSS, title="ACE Music Studio") as demo:
        gr.HTML(_status_html(device))
        gr.HTML(CTA_HTML)

        with gr.Row(elem_classes=["ams-body"]):
            # --- Sidebar ----------------------------------------------------
            with gr.Column(scale=0, min_width=190, elem_classes=["ams-sidebar"]):
                mode = gr.Radio(
                    choices=MODE_CHOICES,
                    value="generate",
                    label=None,
                    show_label=False,
                    container=False,
                    elem_classes=["ams-side-radio"],
                )
                # Dynamic in-memory history (M6/H2). Initial value renders
                # the same "No generations yet" placeholder the static block
                # used to emit; each click handler refreshes the HTML via
                # _history_push().
                history_html = gr.HTML(HISTORY_HTML, elem_classes=["ams-history-wrapper"])

            # --- Content ----------------------------------------------------
            with gr.Column(scale=10, elem_classes=["ams-content"]):
                with gr.Group(visible=True, elem_classes=["ams-tab-pane"]) as pane_generate:
                    g = ui.build_generate_tab()
                    g["lora_preset"].change(
                        fn=on_lora_preset_change,
                        inputs=[g["lora_preset"], g["lora_strength"]],
                        outputs=[g["lora_state"], g["lora_active"], g["lora_upload"]],
                    )
                    g["lora_upload"].change(
                        fn=on_lora_upload,
                        inputs=[g["lora_upload"], g["lora_strength"]],
                        outputs=[g["lora_state"], g["lora_active"], g["lora_preset"]],
                    )
                    g["lora_strength"].change(
                        fn=on_lora_strength_change,
                        inputs=[g["lora_state"], g["lora_strength"]],
                        outputs=[g["lora_state"], g["lora_active"]],
                    )
                    g["generate_btn"].click(
                        fn=on_generate_click,
                        inputs=[
                            g["prompt"],
                            g["lyrics"],
                            g["duration_s"],
                            g["instrumental"],
                            g["lora_state"],
                        ],
                        outputs=[g["output_audio"], g["output_meta"], history_html],
                    )
                    # Post-processing actions (M5/G2)
                    g["separate_stems_btn"].click(
                        fn=on_separate_stems,
                        inputs=[g["output_audio"]],
                        outputs=[g["stem_files"]],
                    )
                    g["normalise_btn"].click(
                        fn=on_normalise,
                        inputs=[g["output_audio"]],
                        outputs=[g["normalised_audio"]],
                    )
                    g["mp3_btn"].click(
                        fn=on_export_mp3,
                        inputs=[g["output_audio"]],
                        outputs=[g["mp3_file"]],
                    )
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_cover:
                    c = ui.build_cover_tab()
                    c["lora_preset"].change(
                        fn=on_lora_preset_change,
                        inputs=[c["lora_preset"], c["lora_strength"]],
                        outputs=[c["lora_state"], c["lora_active"], c["lora_upload"]],
                    )
                    c["lora_upload"].change(
                        fn=on_lora_upload,
                        inputs=[c["lora_upload"], c["lora_strength"]],
                        outputs=[c["lora_state"], c["lora_active"], c["lora_preset"]],
                    )
                    c["lora_strength"].change(
                        fn=on_lora_strength_change,
                        inputs=[c["lora_state"], c["lora_strength"]],
                        outputs=[c["lora_state"], c["lora_active"]],
                    )
                    c["generate_btn"].click(
                        fn=on_cover_click,
                        inputs=[
                            c["ref_audio"],
                            c["prompt"],
                            c["lyrics"],
                            c["duration_s"],
                            c["audio_cover_strength"],
                            c["lora_state"],
                        ],
                        outputs=[c["output_audio"], c["output_meta"], history_html],
                    )
                    # Post-processing actions (M5/G2)
                    c["separate_stems_btn"].click(
                        fn=on_separate_stems,
                        inputs=[c["output_audio"]],
                        outputs=[c["stem_files"]],
                    )
                    c["normalise_btn"].click(
                        fn=on_normalise,
                        inputs=[c["output_audio"]],
                        outputs=[c["normalised_audio"]],
                    )
                    c["mp3_btn"].click(
                        fn=on_export_mp3,
                        inputs=[c["output_audio"]],
                        outputs=[c["mp3_file"]],
                    )
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_extend:
                    x = ui.build_extend_tab()
                    x["lora_preset"].change(
                        fn=on_lora_preset_change,
                        inputs=[x["lora_preset"], x["lora_strength"]],
                        outputs=[x["lora_state"], x["lora_active"], x["lora_upload"]],
                    )
                    x["lora_upload"].change(
                        fn=on_lora_upload,
                        inputs=[x["lora_upload"], x["lora_strength"]],
                        outputs=[x["lora_state"], x["lora_active"], x["lora_preset"]],
                    )
                    x["lora_strength"].change(
                        fn=on_lora_strength_change,
                        inputs=[x["lora_state"], x["lora_strength"]],
                        outputs=[x["lora_state"], x["lora_active"]],
                    )
                    x["generate_btn"].click(
                        fn=on_extend_click,
                        inputs=[
                            x["seed_audio"],
                            x["extra_prompt"],
                            x["extension_lyrics"],
                            x["extra_duration_s"],
                            x["wav_crossfade_s"],
                            x["repaint_mode"],
                            x["repaint_strength"],
                            x["latent_crossfade_frames"],
                            x["chunk_mask_mode"],
                            x["lora_state"],
                        ],
                        outputs=[x["output_audio"], x["output_meta"], history_html],
                    )
                    # Post-processing actions (M5/G2)
                    x["separate_stems_btn"].click(
                        fn=on_separate_stems,
                        inputs=[x["output_audio"]],
                        outputs=[x["stem_files"]],
                    )
                    x["normalise_btn"].click(
                        fn=on_normalise,
                        inputs=[x["output_audio"]],
                        outputs=[x["normalised_audio"]],
                    )
                    x["mp3_btn"].click(
                        fn=on_export_mp3,
                        inputs=[x["output_audio"]],
                        outputs=[x["mp3_file"]],
                    )
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_edit:
                    e = ui.build_edit_tab()
                    e["lora_preset"].change(
                        fn=on_lora_preset_change,
                        inputs=[e["lora_preset"], e["lora_strength"]],
                        outputs=[e["lora_state"], e["lora_active"], e["lora_upload"]],
                    )
                    e["lora_upload"].change(
                        fn=on_lora_upload,
                        inputs=[e["lora_upload"], e["lora_strength"]],
                        outputs=[e["lora_state"], e["lora_active"], e["lora_preset"]],
                    )
                    e["lora_strength"].change(
                        fn=on_lora_strength_change,
                        inputs=[e["lora_state"], e["lora_strength"]],
                        outputs=[e["lora_state"], e["lora_active"]],
                    )
                    e["generate_btn"].click(
                        fn=on_edit_click,
                        inputs=[
                            e["source_audio"],
                            e["sub_mode"],
                            e["source_lyrics"],
                            e["target_lyrics"],
                            e["segment_start_s"],
                            e["segment_end_s"],
                            e["repaint_strength"],
                            e["repaint_mode"],
                            e["flow_source_caption"],
                            e["flow_n_min"],
                            e["flow_n_max"],
                            e["flow_n_avg"],
                            e["lora_state"],
                        ],
                        outputs=[e["output_audio"], e["output_meta"], history_html],
                    )
                    # Post-processing actions (M5/G2)
                    e["separate_stems_btn"].click(
                        fn=on_separate_stems,
                        inputs=[e["output_audio"]],
                        outputs=[e["stem_files"]],
                    )
                    e["normalise_btn"].click(
                        fn=on_normalise,
                        inputs=[e["output_audio"]],
                        outputs=[e["normalised_audio"]],
                    )
                    e["mp3_btn"].click(
                        fn=on_export_mp3,
                        inputs=[e["output_audio"]],
                        outputs=[e["mp3_file"]],
                    )
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_lyrics:
                    lyr = ui.build_lyrics_tab()
                    lyr["draft_btn"].click(
                        fn=on_draft_lyrics,
                        inputs=[
                            lyr["brief"],
                            lyr["structure"],
                            lyr["language"],
                            lyr["tone"],
                            lyr["verse_lines"],
                            lyr["chorus_lines"],
                            lyr["bridge_lines"],
                            lyr["rhyme"],
                            lyr["temperature"],
                            lyr["top_p"],
                            lyr["top_k"],
                            lyr["max_new_tokens"],
                            lyr["seed"],
                        ],
                        outputs=[lyr["lyrics_output"], lyr["meta_output"], history_html],
                    )
                    # Cross-tab "Use these in Generate" — pipes the drafted
                    # text straight into the Generate tab's lyrics textbox.
                    # Both panes were declared inside the same gr.Blocks
                    # context so referencing g["lyrics"] across panes works.
                    lyr["use_in_generate_btn"].click(
                        fn=lambda txt: txt,
                        inputs=[lyr["lyrics_output"]],
                        outputs=[g["lyrics"]],
                    )

        panes = [pane_generate, pane_cover, pane_extend, pane_edit, pane_lyrics]

        def _switch_pane(selected: str):
            order = ["generate", "cover", "extend", "edit", "lyrics"]
            return tuple(gr.Group(visible=(selected == name)) for name in order)

        mode.change(fn=_switch_pane, inputs=mode, outputs=panes)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
