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
                    gr.Markdown("### 🎤 Cover\n\nPlaceholder — implemented in M3.")
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_extend:
                    gr.Markdown("### ⏩ Extend\n\nPlaceholder — implemented in M3.")
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_edit:
                    gr.Markdown("### ✏️ Edit\n\nPlaceholder — implemented in M3.")
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_lyrics:
                    gr.Markdown("### ✍️ Lyrics\n\nPlaceholder — implemented in M4.")

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
