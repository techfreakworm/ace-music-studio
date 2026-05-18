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

import random

import gradio as gr

import ace_pipeline
import backend as be
import modes
import theme
import ui

_BACKEND: be.ACEStepStudioBackend | None = None


def get_backend() -> be.ACEStepStudioBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = be.ACEStepStudioBackend()
    return _BACKEND


def on_generate_click(
    prompt: str,
    lyrics: str,
    duration_s: float,
    instrumental_label: str,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    try:
        out_path, meta = modes.generate(
            get_backend(),
            params={
                "prompt": prompt,
                "lyrics": lyrics,
                "duration_s": int(duration_s),
                "instrumental": instrumental_label == "Instrumental",
                "seed": random.randint(1, 2_147_483_647),
                "loras": [],
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
                    g["generate_btn"].click(
                        fn=on_generate_click,
                        inputs=[g["prompt"], g["lyrics"], g["duration_s"], g["instrumental"]],
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
