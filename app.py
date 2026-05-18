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

On HF Spaces, ``_bootstrap()`` runs once on import to mirror the
read-only preload cache into a writable tree. On Mac/Linux locally,
it's a no-op until M7.
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
import theme
import ui

_BACKEND: be.ACEStepStudioBackend | None = None


def get_backend() -> be.ACEStepStudioBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = be.ACEStepStudioBackend()
    return _BACKEND


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


def on_generate_click(
    prompt: str,
    lyrics: str,
    duration_s: float,
    instrumental_label: str,
    lora_state,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    loras = [lora_state] if lora_state else []
    try:
        out_path, meta = modes.generate(
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
    except ValueError as e:
        raise gr.Error(str(e)) from e
    return out_path, meta


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
    try:
        return modes.cover(
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
    except ValueError as e:
        raise gr.Error(str(e)) from e


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
    try:
        return modes.extend(
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
    except ValueError as e:
        raise gr.Error(str(e)) from e


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
    try:
        return modes.lyrics(
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
    except ValueError as e:
        raise gr.Error(str(e)) from e


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
    try:
        return modes.edit(
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
    except ValueError as e:
        raise gr.Error(str(e)) from e


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


MODE_CHOICES = [
    ("🎵 Generate", "generate"),
    ("🎤 Cover", "cover"),
    ("⏩ Extend", "extend"),
    ("✏️ Edit", "edit"),
    ("✍️ Lyrics", "lyrics"),
]


def _bootstrap() -> None:
    """HF Spaces: mirror read-only preload cache into a writable tree.

    Local Mac/CUDA: no-op. Implemented at M7 when we wire deployment.
    """
    pass


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
                gr.HTML(HISTORY_HTML)

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
                        outputs=[g["output_audio"], g["output_meta"]],
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
                        outputs=[c["output_audio"], c["output_meta"]],
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
                        outputs=[x["output_audio"], x["output_meta"]],
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
                        outputs=[e["output_audio"], e["output_meta"]],
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
                        outputs=[lyr["lyrics_output"], lyr["meta_output"]],
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
    _bootstrap()
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
