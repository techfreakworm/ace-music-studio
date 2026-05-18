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

    Includes a single-LoRA picker in a collapsed accordion between the
    duration/vocal-mode row and the Generate button. The Apple-Silicon
    ACE-Step fork's AceStepHandler only supports one active LoRA at a
    time (see ``lora_stack.apply_stack`` for the gory details), so the
    UI surfaces a single slot — a preset radio OR a custom upload — and
    a strength slider, with a Markdown "active LoRA" display.

    Advanced / LM-planner / DCW accordions are deferred to M2-M4 and
    will be added by extending this builder.
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

            # --- LoRA accordion (collapsed by default) ---
            # Single-LoRA-slot UI: the apple-silicon fork's AceStepHandler
            # can only hold one active adapter, so multi-row stacks are
            # deferred until upstream lands multi-adapter support.
            with gr.Accordion(
                label="LoRA",
                open=False,
                elem_classes=["ams-lora", "ams-lora-accordion"],
            ):
                gr.Markdown(
                    "_Only one LoRA at a time on this build. "
                    "Picking a preset or uploading a custom file "
                    "replaces the active LoRA._",
                    elem_classes=["ams-lora-note"],
                )
                components["lora_preset"] = gr.Radio(
                    choices=[
                        "None",
                        "RapMachine",
                        "Chinese Rap",
                        "Lyric2Vocal",
                        "Text2Samples",
                    ],
                    value="None",
                    label="Preset",
                    elem_classes=["ams-lora-preset"],
                    interactive=True,
                )
                components["lora_upload"] = gr.File(
                    label="Custom LoRA (.safetensors)",
                    file_types=[".safetensors"],
                    file_count="single",
                    elem_classes=["ams-lora-file"],
                )
                components["lora_strength"] = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    step=0.05,
                    value=0.95,
                    label="Strength",
                    elem_classes=["ams-lora-strength"],
                )
                components["lora_active"] = gr.Markdown(
                    "_No LoRA active_",
                    elem_classes=["ams-lora-active"],
                )
                # Hidden state holding the resolved active LoRA dict
                # ``{name, scale, path, sha256}`` so on_generate_click
                # can pass it straight to backend.dispatch.
                components["lora_state"] = gr.State(None)

            components["generate_btn"] = gr.Button(
                "▶ Generate",
                variant="primary",
            )

        # --- OUTPUT column (right, ~40% width) ---
        # elem_classes on each output component give CSS hooks for the
        # Brutalist Mono treatment (uppercase mono labels + bordered
        # empty-state panels). Without these we'd need to target
        # svelte-hashed classes which can change across Gradio versions.
        with gr.Column(scale=10):
            components["output_audio"] = gr.Audio(
                label="Output",
                type="filepath",
                interactive=False,
                elem_classes=["ams-out", "ams-out-audio"],
            )
            # gr.JSON renders a dict directly as a syntax-highlighted, expandable
            # tree. gr.Code(language="json") refuses dicts — it requires a
            # pre-stringified blob — and crashes with "'dict' has no .strip()".
            components["output_meta"] = gr.JSON(
                label="Metadata",
                elem_classes=["ams-out", "ams-out-meta"],
            )

    return components
