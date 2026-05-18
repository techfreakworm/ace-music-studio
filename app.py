"""ACE Music Studio — Gradio entrypoint.

On HF Spaces, `_bootstrap()` runs once on import to mirror the read-only
preload cache into a writable tree. On Mac/Linux locally, it's a no-op.
The backend singleton is lazy-loaded on first generation request.
"""
from __future__ import annotations

import os

# Set MPS fallback BEFORE any torch import path is taken.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Don't pin HF download source — let HF default for both Spaces and local cache.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import gradio as gr

import theme

HEADER_HTML = """
<div class="ams-header">
  <div>
    <div class="ams-brand">ACE Music Studio<span class="ams-brand-period">.</span></div>
  </div>
  <div class="ams-status">ready</div>
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


def _bootstrap() -> None:
    """HF Spaces: mirror read-only preload cache into a writable tree.

    Local Mac/CUDA: no-op. Implemented at M7 when we wire deployment.
    """
    pass


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=theme.build_theme(), css=theme.CSS, title="ACE Music Studio") as demo:
        gr.HTML(HEADER_HTML)
        gr.HTML(CTA_HTML)

        with gr.Tabs():
            with gr.Tab("🎵 Generate"):
                gr.Markdown("Generate tab placeholder — implemented in M1.")
            with gr.Tab("🎤 Cover"):
                gr.Markdown("Cover tab placeholder — implemented in M3.")
            with gr.Tab("⏩ Extend"):
                gr.Markdown("Extend tab placeholder — implemented in M3.")
            with gr.Tab("✏️ Edit"):
                gr.Markdown("Edit tab placeholder — implemented in M3.")
            with gr.Tab("✍️ Lyrics"):
                gr.Markdown("Lyrics tab placeholder — implemented in M4.")

    return demo


if __name__ == "__main__":
    _bootstrap()
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
