"""Per-tab Gradio component builders + shared output panel.

Each builder returns a dict of components keyed by purpose so app.py wires
events without depending on Gradio's positional return order.

NOTE: builders DO NOT instantiate the surrounding gr.Group / pane — they
ONLY build the form + output components inside it. app.py wraps the
result in pane_generate / pane_cover / etc.
"""

from __future__ import annotations

import gradio as gr

import tooltips


def build_generate_tab() -> dict[str, gr.components.Component]:
    """Generate tab body: 2-column row (form left, output right).

    LoRA / Advanced / LM-planner / DCW accordions are deferred to
    M2-M4 and will be added by extending this builder.
    """
    components: dict[str, gr.components.Component] = {}

    with gr.Row():
        # --- FORM column (left, ~60% width) ---
        with gr.Column(scale=13):
            components["prompt"] = gr.Textbox(
                label="Style prompt",
                placeholder="psytrance, rolling triplet bassline, acid squelch, metallic leads",
                lines=2,
                info=tooltips.GENERATE_PROMPT,
            )
            components["lyrics"] = gr.Textbox(
                label="Lyrics",
                placeholder="[intro] atmospheric pads\n[verse] ...",
                lines=6,
                info=tooltips.GENERATE_LYRICS,
            )
            with gr.Row():
                components["duration_s"] = gr.Slider(
                    minimum=5,
                    maximum=240,
                    step=5,
                    value=30,
                    label="Duration (s)",
                    info=tooltips.GENERATE_DURATION,
                )
                components["instrumental"] = gr.Radio(
                    choices=["With vocals", "Instrumental"],
                    value="With vocals",
                    label="Vocal mode",
                    info=tooltips.GENERATE_VOCAL,
                )
            components["generate_btn"] = gr.Button(
                "▶ Generate",
                variant="primary",
            )

        # --- OUTPUT column (right, ~40% width) ---
        with gr.Column(scale=10):
            components["output_audio"] = gr.Audio(
                label="Output",
                type="filepath",
                interactive=False,
            )
            components["output_meta"] = gr.Code(
                label="Metadata",
                language="json",
            )

    return components
