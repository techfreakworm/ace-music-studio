"""Per-tab Gradio component builders + shared output panel.

Each builder returns a dict of components keyed by purpose so app.py wires
events without depending on Gradio's positional return order.

NOTE: builders DO NOT instantiate the surrounding gr.Group / pane — they
ONLY build the form + output components inside it. app.py wraps the
result in pane_generate / pane_cover / etc.
"""

from __future__ import annotations

import gradio as gr

import lora_stack
import tooltips


def _build_lora_accordion(components: dict[str, gr.components.Component]) -> None:
    """LoRA accordion with single-LoRA semantics. Mutates ``components``.

    Each song mode (generate / cover / extend / edit) calls this so the
    form has a consistent LoRA picker. Apple-Silicon ACE-Step fork's
    AceStepHandler can only hold one active adapter at a time (see
    ``lora_stack.apply_stack``), so the UI surfaces a single slot — a
    preset radio OR a custom upload — and a strength slider, with a
    Markdown "active LoRA" display.
    """
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
        # Preset choices are read from presets/manifest.json so the
        # radio stays in sync with whatever official ACE-Step LoRAs
        # are actually published on HuggingFace.
        _preset_names = ["None"] + [p["name"] for p in lora_stack.load_presets()]
        components["lora_preset"] = gr.Radio(
            choices=_preset_names,
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
        # ``{name, scale, path, sha256}`` so the click handler can pass
        # it straight to backend.dispatch.
        components["lora_state"] = gr.State(None)


def _build_output_panel(components: dict[str, gr.components.Component]) -> None:
    """Shared OUTPUT (gr.Audio) + post-process actions + METADATA (gr.JSON).

    elem_classes on each output component give CSS hooks for the
    Brutalist Mono treatment (uppercase mono labels + bordered
    empty-state panels). Without these we'd need to target
    svelte-hashed classes which can change across Gradio versions.

    gr.JSON renders a dict directly as a syntax-highlighted, expandable
    tree. gr.Code(language="json") refuses dicts — it requires a
    pre-stringified blob — and crashes with "'dict' has no .strip()".

    Below the Audio we expose three secondary post-process actions
    (M5/G2): Demucs stem separation, pyloudnorm LUFS normalisation, and
    ffmpeg MP3 export. Each emits to a hidden output (stem_files /
    normalised_audio / mp3_file) that becomes visible only once the
    click handler returns a populated value.
    """
    components["output_audio"] = gr.Audio(
        label="Output",
        type="filepath",
        interactive=False,
        elem_classes=["ams-out", "ams-out-audio"],
    )
    with gr.Row(elem_classes=["ams-post-actions"]):
        components["separate_stems_btn"] = gr.Button(
            "↯ Separate stems",
            variant="secondary",
            elem_classes=["ams-post-btn"],
        )
        components["normalise_btn"] = gr.Button(
            "▮ Normalise -14 LUFS",
            variant="secondary",
            elem_classes=["ams-post-btn"],
        )
        components["mp3_btn"] = gr.Button(
            "↓ MP3 320k",
            variant="secondary",
            elem_classes=["ams-post-btn"],
        )
    components["stem_files"] = gr.Files(
        label="Stems",
        visible=False,
        elem_classes=["ams-stem-files"],
    )
    components["normalised_audio"] = gr.Audio(
        label="Normalised (-14 LUFS)",
        type="filepath",
        interactive=False,
        visible=False,
        elem_classes=["ams-out", "ams-out-normalised"],
    )
    components["mp3_file"] = gr.File(
        label="MP3 download",
        visible=False,
        elem_classes=["ams-mp3-file"],
    )
    components["output_meta"] = gr.JSON(
        label="Metadata",
        elem_classes=["ams-out", "ams-out-meta"],
    )


def build_generate_tab() -> dict[str, gr.components.Component]:
    """Generate tab body: 2-column row (form left, output right).

    Includes a single-LoRA picker in a collapsed accordion between the
    duration/vocal-mode row and the Generate button.

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

            _build_lora_accordion(components)

            components["generate_btn"] = gr.Button(
                "▶ Generate",
                variant="primary",
            )

        # --- OUTPUT column (right, ~40% width) ---
        with gr.Column(scale=10):
            _build_output_panel(components)

    return components


def build_cover_tab() -> dict[str, gr.components.Component]:
    """Cover tab body: reference audio + new lyrics -> cover in that style.

    Maps to ACE-Step's ``task_type="cover"`` with the uploaded reference
    feeding ``reference_audio`` and the strength slider controlling
    ``audio_cover_strength``. Higher strength clings to the reference;
    lower lets the new prompt/lyrics drift the timbre.
    """
    components: dict[str, gr.components.Component] = {}
    with gr.Row():
        with gr.Column(scale=13):
            components["ref_audio"] = gr.Audio(
                label="Reference audio",
                type="filepath",
                sources=["upload"],
                elem_classes=["ams-input-audio"],
            )
            components["prompt"] = gr.Textbox(
                label="New style prompt (optional)",
                placeholder="faster, more aggressive leads",
                lines=2,
            )
            components["lyrics"] = gr.Textbox(
                label="New lyrics",
                placeholder="[verse] new lyrics over the reference style",
                lines=5,
            )
            with gr.Row():
                components["duration_s"] = gr.Slider(
                    minimum=5,
                    maximum=240,
                    step=5,
                    value=30,
                    label="Duration (s)",
                )
                components["audio_cover_strength"] = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.93,
                    label="Cover strength",
                    info="Higher = closer to reference. Lower = more drift.",
                )

            _build_lora_accordion(components)

            components["generate_btn"] = gr.Button(
                "▶ Generate cover",
                variant="primary",
            )

        with gr.Column(scale=10):
            _build_output_panel(components)

    return components


def build_extend_tab() -> dict[str, gr.components.Component]:
    """Extend tab body: seed audio + extension prompt -> continued song.

    Maps to ACE-Step's ``task_type="repaint"`` with ``src_audio`` set to
    the uploaded seed and the repaint window pointing past the end of
    the seed so the model paints new audio after it.

    The repaint params (``repaint_mode``, ``repaint_strength``,
    ``latent_crossfade_frames``, ``chunk_mask_mode``, ``wav_crossfade_s``)
    are surfaced in an experimental accordion because the installed
    ACE-Step ``GenerationParams`` dataclass doesn't expose them yet — the
    UI captures them so they're ready to plumb through once upstream
    adds the fields.
    """
    components: dict[str, gr.components.Component] = {}
    with gr.Row():
        with gr.Column(scale=13):
            components["seed_audio"] = gr.Audio(
                label="Seed audio",
                type="filepath",
                sources=["upload"],
                elem_classes=["ams-input-audio"],
            )
            components["extra_prompt"] = gr.Textbox(
                label="Extension prompt",
                placeholder="build to climax, layered acid leads",
                lines=2,
            )
            components["extension_lyrics"] = gr.Textbox(
                label="Extension lyrics (optional)",
                placeholder="[bridge] the drop is coming...",
                lines=4,
            )
            with gr.Row():
                components["extra_duration_s"] = gr.Slider(
                    minimum=5,
                    maximum=120,
                    step=5,
                    value=60,
                    label="Extra duration (s)",
                )
                components["wav_crossfade_s"] = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    step=0.1,
                    value=2.0,
                    label="WAV crossfade (s)",
                    info="Experimental — not yet wired in this acestep build.",
                )

            with gr.Accordion(
                "Repaint params (experimental)",
                open=False,
                elem_classes=["ams-experimental"],
            ):
                gr.Markdown(
                    "_These knobs are captured in the request but the installed "
                    "ACE-Step dataclass doesn't expose them yet._",
                    elem_classes=["ams-lora-note"],
                )
                components["repaint_mode"] = gr.Dropdown(
                    choices=["balanced", "left", "right"],
                    value="balanced",
                    label="Repaint mode",
                )
                components["repaint_strength"] = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.5,
                    label="Repaint strength",
                )
                components["latent_crossfade_frames"] = gr.Slider(
                    minimum=0,
                    maximum=30,
                    step=1,
                    value=10,
                    label="Latent crossfade frames",
                )
                components["chunk_mask_mode"] = gr.Dropdown(
                    choices=["auto", "manual"],
                    value="auto",
                    label="Chunk mask",
                )

            _build_lora_accordion(components)

            components["generate_btn"] = gr.Button(
                "▶ Extend",
                variant="primary",
            )

        with gr.Column(scale=10):
            _build_output_panel(components)

    return components


def build_edit_tab() -> dict[str, gr.components.Component]:
    """Edit tab body: source audio + segment + target lyrics -> repaint/morph.

    Two sub-modes:

    - ``repaint`` (default): paint over [segment_start_s, segment_end_s]
      using ACE-Step's repaint task_type. ``segment_start_s`` and
      ``segment_end_s`` are wired through the params dict to
      ``repainting_start`` / ``repainting_end`` on the pipeline side.
    - ``flow_edit``: caption-to-caption morph. The installed ACE-Step
      ``GenerationParams`` has no ``flow_edit_*`` fields, so this
      sub-mode falls back to a repaint pass with lower
      ``audio_cover_strength``. The flow knobs are still captured so
      they're ready once upstream adds native support.
    """
    components: dict[str, gr.components.Component] = {}
    with gr.Row():
        with gr.Column(scale=13):
            components["source_audio"] = gr.Audio(
                label="Source audio",
                type="filepath",
                sources=["upload"],
                elem_classes=["ams-input-audio"],
            )
            components["sub_mode"] = gr.Radio(
                choices=["repaint", "flow_edit"],
                value="repaint",
                label="Edit sub-mode",
                info=(
                    "repaint: regenerate the segment from new lyrics. "
                    "flow_edit: morph caption-to-caption (experimental)."
                ),
            )
            components["source_lyrics"] = gr.Textbox(
                label="Source lyrics",
                lines=3,
            )
            components["target_lyrics"] = gr.Textbox(
                label="Target lyrics",
                placeholder="[chorus] new chorus replaces the old",
                lines=3,
            )
            with gr.Row():
                components["segment_start_s"] = gr.Number(
                    value=0.0,
                    label="Segment start (s)",
                    precision=1,
                )
                components["segment_end_s"] = gr.Number(
                    value=30.0,
                    label="Segment end (s)",
                    precision=1,
                )

            with gr.Accordion(
                "Repaint options (experimental)",
                open=False,
                elem_classes=["ams-experimental"],
            ):
                gr.Markdown(
                    "_These knobs are captured in the request but the installed "
                    "ACE-Step dataclass doesn't expose them yet._",
                    elem_classes=["ams-lora-note"],
                )
                components["repaint_strength"] = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.5,
                    label="Repaint strength",
                )
                components["repaint_mode"] = gr.Dropdown(
                    choices=["balanced", "left", "right"],
                    value="balanced",
                    label="Repaint mode",
                )

            with gr.Accordion(
                "Flow-morph options (experimental)",
                open=False,
                elem_classes=["ams-experimental"],
            ):
                gr.Markdown(
                    "_flow_edit sub-mode currently falls back to a repaint pass with "
                    "lower audio_cover_strength. flow-specific params are captured "
                    "but not yet wired._",
                    elem_classes=["ams-lora-note"],
                )
                components["flow_source_caption"] = gr.Textbox(
                    label="Source caption",
                    placeholder="acoustic ballad, gentle piano",
                )
                components["flow_n_min"] = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="n_min"
                )
                components["flow_n_max"] = gr.Slider(
                    minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="n_max"
                )
                components["flow_n_avg"] = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="n_avg")

            _build_lora_accordion(components)

            components["generate_btn"] = gr.Button(
                "▶ Apply edit",
                variant="primary",
            )

        with gr.Column(scale=10):
            _build_output_panel(components)

    return components


def build_lyrics_tab() -> dict[str, gr.components.Component]:
    """Lyrics tab body: Qwen 2.5 7B drafts structurally-tagged lyrics.

    Compact 2-column row: form on the left (brief / structure / language /
    line counts / tone / rhyme + collapsed LM-params accordion), output on
    the right (read-only multi-line textbox + ``Use these in Generate``
    cross-tab CTA + bordered JSON metadata panel).

    The output textbox carries ``elem_classes=["ams-lyrics-output"]`` so
    the Brutalist Mono treatment in ``theme.CSS`` (mono font, 12 px,
    280 px min-height) applies. The "Use in Generate" button is tagged
    ``ams-lyrics-use-btn`` so it gets a small top margin instead of
    sitting flush against the textbox.

    Does NOT include the LoRA accordion — Qwen-7B has no LoRA picker and
    the audio-mode LoRA semantics don't apply here.
    """
    c: dict[str, gr.components.Component] = {}
    with gr.Row():
        # --- FORM column (left) ---
        with gr.Column(scale=12):
            c["brief"] = gr.Textbox(
                label="Brief",
                lines=4,
                placeholder=("Describe the song. Tone, mood, references, specific images, lines to avoid…"),
            )
            with gr.Row():
                c["structure"] = gr.Textbox(
                    label="Structure",
                    value="intro, verse, chorus, verse, chorus, bridge, chorus, outro",
                )
                c["language"] = gr.Dropdown(
                    choices=["en", "zh", "ja", "ko", "es", "fr", "de"],
                    value="en",
                    label="Language",
                )
            with gr.Row():
                c["verse_lines"] = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=6,
                    step=1,
                    label="Verse lines",
                )
                c["chorus_lines"] = gr.Slider(
                    minimum=2,
                    maximum=8,
                    value=4,
                    step=1,
                    label="Chorus lines",
                )
                c["bridge_lines"] = gr.Slider(
                    minimum=1,
                    maximum=6,
                    value=2,
                    step=1,
                    label="Bridge lines",
                )
            c["tone"] = gr.Textbox(
                label="Tone / mood",
                placeholder="euphoric, hypnotic, transcendent, not cheesy",
            )
            c["rhyme"] = gr.Radio(
                choices=["strict", "loose", "none"],
                value="loose",
                label="Rhyme",
            )
            with gr.Accordion(
                "LM parameters",
                open=False,
                elem_classes=["ams-lm-accordion"],
            ):
                c["temperature"] = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.85,
                    step=0.05,
                    label="Temperature",
                )
                c["top_p"] = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p",
                )
                c["top_k"] = gr.Slider(
                    minimum=0,
                    maximum=200,
                    value=40,
                    step=1,
                    label="Top-k",
                )
                c["max_new_tokens"] = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=600,
                    step=50,
                    label="Max new tokens",
                )
                c["seed"] = gr.Number(
                    value=42,
                    precision=0,
                    label="Seed",
                )
            c["draft_btn"] = gr.Button(
                "▶ Draft lyrics",
                variant="primary",
            )

        # --- OUTPUT column (right) ---
        with gr.Column(scale=10):
            # NOTE: gr.Textbox in Gradio 6.14 doesn't accept ``show_copy_button``
            # (the kwarg landed in a later 6.x). The Brutalist Mono textbox already
            # exposes a native selection + browser copy via Cmd-A / Cmd-C; the
            # copy-button affordance is therefore a no-op miss here.
            c["lyrics_output"] = gr.Textbox(
                label="Draft",
                lines=14,
                interactive=False,
                elem_classes=["ams-lyrics-output"],
            )
            c["use_in_generate_btn"] = gr.Button(
                "↑ Use these in Generate",
                variant="primary",
                elem_classes=["ams-lyrics-use-btn"],
            )
            c["meta_output"] = gr.JSON(
                label="Metadata",
                elem_classes=["ams-out", "ams-out-meta"],
            )
    return c
