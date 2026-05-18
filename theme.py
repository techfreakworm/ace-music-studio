"""Brutalist Mono — pure black/white, no color accent.

Palette tokens are the source of truth; CSS pulls from them. The audio
waveform is the only optionally-colored element (rendered white in v1).

UI architecture (locked):
- Sidebar layout (NOT ``gr.Tabs``) per wireframes at
  ``docs/superpowers/specs/mockups/``.
- ``.ams-sidebar`` is a flex column at desktop, fixed-width 170-190 px.
- ``.ams-side-radio`` is the mode-nav: a ``gr.Radio`` re-skinned via CSS
  so each option renders as a full-width sidebar pill. The native
  ``:checked`` pseudo-class supplies the "active" highlight.
- ``.ams-content`` is the right column containing 5 ``.ams-tab-pane``
  groups; one is visible at a time.
- Media queries: at ``<= 1024 px`` the sidebar shrinks to an icon rail.
  At ``<= 640 px`` the sidebar is replaced by a horizontal scroll strip
  at the top.
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
HOVER_BG = "#1A1A1A"
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
/* === Body chrome ======================================================= */
.ams-header {{
  display:flex; justify-content:space-between; align-items:baseline;
  padding:10px 4px 6px 4px;
}}
.ams-brand {{
  font-size:18px; font-weight:600; letter-spacing:-0.01em; color:{INK};
}}
.ams-brand-period {{ color:{PRIMARY}; }}
.ams-status {{
  font-size:11px; color:{INK_MUTED};
  letter-spacing:0.06em; text-transform:uppercase;
}}

.ams-cta {{
  font-size:13px; color:{INK_MUTED};
  margin:2px 4px 14px 4px; padding-bottom:12px;
  border-bottom:1px solid {BORDER};
}}
.ams-cta strong {{ color:{INK}; }}
.ams-cta-heart {{ color:{PRIMARY}; }}
.ams-cta a {{ color:{INK}; text-decoration:underline; }}

/* === Body row (sidebar + content) ====================================== */
.ams-body {{
  gap:16px !important;
  align-items:stretch !important;
}}

/* === Sidebar (desktop >= 1024px) ======================================= */
.ams-sidebar {{
  background:{SURFACE_STRONG} !important;
  padding:14px 8px !important;
  border-radius:{RADIUS} !important;
  border:1px solid {BORDER} !important;
  min-width:190px;
  max-width:210px;
}}

/* --- Mode radio (re-skin gr.Radio as a vertical sidebar nav) ----------- */
.ams-side-radio {{
  background:transparent !important;
  border:none !important;
  padding:0 !important;
  width:100%;
}}
.ams-side-radio .wrap {{
  display:flex !important;
  flex-direction:column !important;
  gap:2px !important;
  background:transparent !important;
  border:none !important;
}}
/* Each radio option becomes a sidebar pill */
.ams-side-radio label {{
  display:flex !important;
  align-items:center !important;
  padding:9px 12px !important;
  margin:0 !important;
  border-radius:4px !important;
  border:none !important;
  border-left:2px solid transparent !important;
  background:transparent !important;
  color:{INK_MUTED} !important;
  font-size:13px !important;
  font-weight:500 !important;
  cursor:pointer !important;
  transition:background 80ms ease, color 80ms ease, border-color 80ms ease;
  min-height:0 !important;
  width:100%;
  box-sizing:border-box;
}}
.ams-side-radio label:hover {{
  background:{HOVER_BG} !important;
  color:{INK} !important;
}}
/* Hide the native radio circle */
.ams-side-radio label input[type="radio"] {{
  display:none !important;
}}
/* Active state: white text + white left border + dark bg */
.ams-side-radio label.selected,
.ams-side-radio label:has(input[type="radio"]:checked) {{
  background:{HOVER_BG} !important;
  color:{PRIMARY} !important;
  border-left-color:{PRIMARY} !important;
  font-weight:600 !important;
}}
/* Hide the (now-empty) form-element-info row that gr.Radio injects */
.ams-side-radio + div:empty {{ display:none !important; }}

/* --- Session history block (below the mode radio) --------------------- */
.ams-history {{
  margin-top:14px;
  padding-top:10px;
  border-top:1px solid {BORDER};
}}
.ams-history-title {{
  font-size:10px; color:{INK_MUTED};
  letter-spacing:0.1em; text-transform:uppercase;
  padding:0 4px 6px 4px;
}}
.ams-history-empty {{
  font-size:11px; color:#3F3F3F;
  font-style:italic;
  padding:6px 4px;
}}

/* === Content area ====================================================== */
.ams-content {{
  background:{SURFACE} !important;
  border:1px solid {BORDER} !important;
  border-radius:{RADIUS} !important;
  padding:16px !important;
  min-height:540px;
}}
.ams-tab-pane {{
  background:transparent !important;
  border:none !important;
  padding:0 !important;
}}

/* === LoRA chip pill (used in M2+) ====================================== */
.ams-chip {{
  display:inline-block; padding:5px 10px; border-radius:14px;
  font-size:11px; margin:0 5px 5px 0; background:{SURFACE_STRONG};
  border:1px solid {BORDER_STRONG}; color:{INK_MUTED}; cursor:pointer;
}}
.ams-chip.on {{ border-color:{PRIMARY}; color:{PRIMARY}; }}
.ams-chip.upload {{ border-style:dashed; color:{PRIMARY}; }}

/* === LoRA file drop zone (tighten Gradio default ~400px height) ======== */
.ams-lora-file .upload-container {{ min-height:56px !important; }}

/* === Hide Gradio footer ================================================ */
footer {{ display:none !important; }}

/* === Responsive: tablet 640-1024 px ==================================== */
@media (max-width: 1024px) {{
  .ams-sidebar {{
    min-width:48px !important;
    max-width:48px !important;
    padding:8px 4px !important;
  }}
  /* Hide labels, keep only the leading emoji */
  .ams-side-radio label {{
    font-size:0 !important;
    padding:8px 0 !important;
    justify-content:center !important;
  }}
  .ams-side-radio label::first-letter {{
    font-size:16px !important;
  }}
  /* Hide history in tablet rail mode */
  .ams-history {{ display:none !important; }}
}}

/* === Responsive: mobile < 640 px ======================================= */
@media (max-width: 640px) {{
  .ams-body {{
    flex-direction:column !important;
  }}
  .ams-sidebar {{
    min-width:100% !important;
    max-width:100% !important;
    padding:6px !important;
  }}
  /* Mobile: switch sidebar to horizontal scroll strip */
  .ams-side-radio .wrap {{
    flex-direction:row !important;
    overflow-x:auto !important;
    gap:4px !important;
  }}
  .ams-side-radio label {{
    font-size:11px !important;
    white-space:nowrap !important;
    border-left:none !important;
    border-bottom:2px solid transparent !important;
    padding:8px 10px !important;
    justify-content:flex-start !important;
  }}
  .ams-side-radio label::first-letter {{
    font-size:13px !important;
  }}
  .ams-side-radio label:has(input[type="radio"]:checked) {{
    border-left-color:transparent !important;
    border-bottom-color:{PRIMARY} !important;
  }}
  .ams-history {{ display:none !important; }}
  .ams-cta {{ font-size:11px; }}
}}
"""
