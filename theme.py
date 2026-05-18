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
/* === Override Gradio's Tailwind-slate "neutral" palette ================
   Gradio's gr.themes.colors.gray is actually Tailwind slate-* which has
   a noticeable blue tint (--neutral-900 = #111827, --neutral-950 = #0b0f19).
   On phone displays with cool color temperatures this reads as bluish-grey
   and contradicts the Brutalist Mono spec. Force true-neutral hex values. */
:root, html.dark {{
  --neutral-50: #FAFAFA !important;
  --neutral-100: #F5F5F5 !important;
  --neutral-200: #E5E5E5 !important;
  --neutral-300: #D4D4D4 !important;
  --neutral-400: #A3A3A3 !important;
  --neutral-500: #737373 !important;
  --neutral-600: #525252 !important;
  --neutral-700: #404040 !important;
  --neutral-800: #262626 !important;
  --neutral-900: #141414 !important;
  --neutral-950: #0A0A0A !important;
  --body-background-fill: {BG} !important;
  --background-fill-primary: {SURFACE} !important;
  --background-fill-secondary: {SURFACE_STRONG} !important;
  --block-background-fill: {SURFACE} !important;
  --block-label-background-fill: transparent !important;
  --block-title-background-fill: transparent !important;
  --input-background-fill: {SURFACE_STRONG} !important;
  --border-color-primary: {BORDER} !important;
  --border-color-accent: {BORDER_STRONG} !important;
  --color-accent: {PRIMARY} !important;
}}

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

/* === Tighten Gradio chrome (narrow scope to avoid breaking output) =====
   Only sharpen the INPUT control surfaces (textareas, number inputs)
   so they read as crisp Brutalist Mono panels. Do NOT touch generic
   ``.block`` padding — that collapses gr.Audio / gr.JSON which need
   their own internal spacing. */
.ams-content textarea,
.ams-content input[type="text"],
.ams-content input[type="number"] {{
  background:{SURFACE_STRONG} !important;
  border:1px solid {BORDER} !important;
  border-radius:4px !important;
  color:{INK} !important;
  padding:10px !important;
}}
.ams-content textarea:focus,
.ams-content input[type="text"]:focus,
.ams-content input[type="number"]:focus {{
  outline:none !important;
  border-color:{PRIMARY} !important;
}}
.ams-content input[type="range"] {{
  accent-color:{PRIMARY} !important;
}}
/* Component labels — uppercase, muted, no shadow.
   ``.gradio-container .block .label`` already covers this from the
   earlier component-label rule, but restating for the form blocks
   specifically gives the wireframe a consistent label scale. */
.ams-content .label-wrap,
.ams-content [data-testid="block-label"] {{
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
  color:{INK_MUTED} !important;
  background:transparent !important;
  border:none !important;
  padding:0 0 4px 0 !important;
}}

/* === Component labels — kill the white pill, make them inline muted ==== */
/* Gradio renders component labels (e.g. gr.Audio "Output", gr.Code
   "Metadata") as elevated white-pill blocks by default. The Brutalist
   Mono theme wants them as plain muted-ink inline text. */
.gradio-container .block .label,
.gradio-container [data-testid="block-label"],
.gradio-container span.label-wrap > span {{
  background:transparent !important;
  color:{INK_MUTED} !important;
  font-size:10px !important;
  font-weight:500 !important;
  letter-spacing:0.06em !important;
  text-transform:uppercase !important;
  border:none !important;
  box-shadow:none !important;
  padding:4px 0 !important;
}}
.gradio-container [data-testid="block-label"] svg,
.gradio-container .label svg {{
  display:none !important;  /* drop the music-note / code glyph next to label */
}}

/* === Responsive: keep full sidebar down to mobile threshold ============ */
/* The previous tablet "icon rail" mode at 640-1024 px relied on
   ``::first-letter`` to keep the emoji visible while hiding the label
   text, but Gradio wraps the radio option text in a <span> so the
   pseudo-class never hits the emoji. Rather than fight the DOM, we keep
   the full sidebar at all widths >= 640 px and switch to a stacked
   layout below that. */

/* === Responsive: mobile < 640 px ======================================= */
@media (max-width: 640px) {{
  /* Stack body so sidebar (now a tab strip) sits above content */
  .ams-body {{
    flex-direction:column !important;
    gap:8px !important;
  }}

  /* Sidebar = horizontal scroll strip. Strip its desktop chrome
     (border, large padding, fixed width) so it reads as a tab bar. */
  .ams-sidebar {{
    min-width:100% !important;
    max-width:100% !important;
    padding:2px !important;
    border:none !important;
    background:transparent !important;
    border-radius:0 !important;
  }}

  /* The radio's outer block (gr.Radio with container=False still gets
     padding from Gradio's base styles). Flatten it. */
  .ams-side-radio {{
    padding:0 !important;
    background:transparent !important;
  }}

  /* Real options live in the second .wrap (Gradio renders an extra
     hidden one first); both flex-row + overflow + nowrap.
     CRITICAL: override the desktop label width:100% — that's what
     makes labels stack vertically inside a flex-row container.
     flex-wrap:nowrap forces a single row + horizontal scroll instead
     of wrapping to 2 rows. */
  .ams-side-radio .wrap {{
    flex-direction:row !important;
    flex-wrap:nowrap !important;
    overflow-x:auto !important;
    overflow-y:hidden !important;
    gap:6px !important;
    padding-bottom:2px !important;
    /* Hide scrollbar but keep scrolling */
    scrollbar-width:none !important;
    -ms-overflow-style:none !important;
  }}
  .ams-side-radio .wrap::-webkit-scrollbar {{
    display:none !important;
  }}

  .ams-side-radio label {{
    /* Compact pill: just enough room for emoji + label, no flex-grow */
    width:auto !important;
    min-width:0 !important;
    max-width:max-content !important;
    flex:0 0 auto !important;
    font-size:11px !important;
    font-weight:600 !important;
    white-space:nowrap !important;
    padding:8px 12px !important;
    /* Bottom border instead of left border for the horizontal context */
    border-left:none !important;
    border-bottom:2px solid transparent !important;
    border-radius:4px !important;
    justify-content:center !important;
    background:{SURFACE_STRONG} !important;
    border-top:1px solid {BORDER} !important;
    border-right:1px solid {BORDER} !important;
    border-left:1px solid {BORDER} !important;
  }}
  .ams-side-radio label.selected,
  .ams-side-radio label:has(input[type="radio"]:checked) {{
    border-left-color:transparent !important;
    border-bottom-color:{PRIMARY} !important;
    background:{HOVER_BG} !important;
  }}

  /* History block off-screen on mobile (already display:none on tablet+;
     restate here in case the cascade gets weird) */
  .ams-history {{ display:none !important; }}

  /* Tighter chrome */
  .ams-header {{ padding:6px 2px 2px 2px !important; }}
  .ams-brand {{ font-size:15px !important; }}
  .ams-cta {{ font-size:11px !important; padding-bottom:8px !important; margin-bottom:8px !important; }}
  .ams-content {{ padding:12px !important; }}
}}
"""
