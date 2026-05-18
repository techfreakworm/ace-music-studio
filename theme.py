"""Brutalist Mono — pure black/white, no color accent.

Palette tokens are the source of truth; CSS pulls from them. The audio
waveform is the only optionally-colored element (rendered white in v1).
"""

from __future__ import annotations

import gradio as gr

# --- Palette ----------------------------------------------------------------
BG = "#0A0A0A"
SURFACE = "#141414"
SURFACE_STRONG = "#000000"
BORDER = "#1F1F1F"
BORDER_STRONG = "#2A2A2A"
INK = "#E5E5E5"
INK_MUTED = "#6B6B6B"
PRIMARY = "#FFFFFF"
ERROR_BG = "#1A1A1A"
RADIUS = "6px"
FONT_STACK = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'


def build_theme() -> gr.themes.Base:
    """Returns a Gradio theme keyed to Brutalist Mono tokens."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill=BG,
        body_text_color=INK,
        block_background_fill=SURFACE,
        block_border_color=BORDER,
        block_border_width="1px",
        block_radius=RADIUS,
        input_background_fill=SURFACE_STRONG,
        input_border_color=BORDER_STRONG,
        input_border_color_focus=PRIMARY,
        button_primary_background_fill=PRIMARY,
        button_primary_text_color=BG,
        button_primary_background_fill_hover=PRIMARY,
        button_secondary_background_fill=SURFACE_STRONG,
        button_secondary_text_color=INK,
        button_secondary_border_color=BORDER_STRONG,
    )


CSS = f"""
/* --- Sole brand bits --------------------------------------------------- */
.ams-header {{
  display:flex; justify-content:space-between; align-items:baseline;
  padding:8px 0 4px 0;
}}
.ams-brand {{
  font-size:16px; font-weight:600; letter-spacing:-0.01em; color:{INK};
}}
.ams-brand-period {{ color:{PRIMARY}; }}
.ams-status {{ font-size:11px; color:{INK_MUTED}; letter-spacing:0.02em; }}

/* --- CTA banner -------------------------------------------------------- */
.ams-cta {{
  font-size:13px; color:{INK_MUTED}; margin:2px 0 12px 0; padding-bottom:10px;
  border-bottom:1px solid {BORDER};
}}
.ams-cta strong {{ color:{INK}; }}
.ams-cta-heart {{ color:{PRIMARY}; }}
.ams-cta a {{ color:{INK}; text-decoration:underline; }}

/* --- Sidebar nav (desktop >= 1024) ------------------------------------ */
.ams-sidebar {{ background:{SURFACE_STRONG}; padding:14px 10px; border-radius:{RADIUS}; min-width:170px; }}
.ams-side-item {{
  display:block; padding:8px 10px; border-radius:4px; margin-bottom:3px;
  font-size:13px; color:{INK_MUTED}; cursor:pointer; text-decoration:none;
}}
.ams-side-item.active {{
  background:#1A1A1A; color:{PRIMARY};
  border-left:2px solid {PRIMARY}; padding-left:8px;
}}

/* --- LoRA chip pill --------------------------------------------------- */
.ams-chip {{
  display:inline-block; padding:5px 10px; border-radius:14px;
  font-size:11px; margin:0 5px 5px 0; background:{SURFACE_STRONG};
  border:1px solid {BORDER_STRONG}; color:{INK_MUTED}; cursor:pointer;
}}
.ams-chip.on {{ border-color:{PRIMARY}; color:{PRIMARY}; }}
.ams-chip.upload {{ border-style:dashed; color:{PRIMARY}; }}

/* --- LoRA file drop zone (tighten Gradio default ~400px height) ------ */
.ams-lora-file .upload-container {{ min-height:56px !important; }}

/* --- Hide Gradio footer ----------------------------------------------- */
footer {{ display:none !important; }}

/* --- Responsive: tablet 640-1024 px ----------------------------------- */
@media (max-width: 1024px) {{
  .ams-sidebar {{ min-width:34px; padding:6px 4px; }}
  .ams-side-item {{ font-size:0; padding:6px; }}
  .ams-side-item::first-letter {{ font-size:16px; }}
}}

/* --- Responsive: mobile < 640 px -------------------------------------- */
@media (max-width: 640px) {{
  .ams-sidebar {{ display:none; }}
  .ams-cta {{ font-size:11px; }}
}}
"""
