"""Brutalist Mono — pure black/white, no color accent.

The aesthetic identity of ACE Music Studio: precision, density, mono-
spaced labels, true neutral grays, and a single white accent rationed
across active states + the primary CTA. Inspired by record-sleeve
liner notes and a code editor's chrome.

Typography (selected per the frontend-design discipline):
- IBM Plex Sans  — body, brand, helper text. Mechanically refined,
  warmer than Inter, free via Google Fonts.
- IBM Plex Mono  — labels, status indicator, brand period, metadata
  JSON, all-caps small text. Distinctive character without going
  novelty.

The wireframes at ``docs/superpowers/specs/mockups/`` remain the visual
source of truth. The mobile breakpoint at 640 px replaces the sidebar
with a horizontal scroll strip and crushes the outer Gradio padding so
the full 360 px viewport is actually usable.
"""

from __future__ import annotations

import gradio as gr

# --- Palette ----------------------------------------------------------------
BG = "#0A0A0A"
SURFACE = "#141414"
SURFACE_STRONG = "#000000"
SURFACE_RAISED = "#1A1A1A"
BORDER = "#1F1F1F"
BORDER_STRONG = "#2A2A2A"
INK = "#E5E5E5"
INK_MUTED = "#6B6B6B"
INK_FAINT = "#3F3F3F"
PRIMARY = "#FFFFFF"
HOVER_BG = "#1A1A1A"
RADIUS = "4px"
FONT_SANS = '"IBM Plex Sans", -apple-system, BlinkMacSystemFont, system-ui, sans-serif'
FONT_MONO = '"IBM Plex Mono", "JetBrains Mono", ui-monospace, Menlo, monospace'


def build_theme() -> gr.themes.Base:
    """Returns a Gradio theme keyed to Brutalist Mono tokens.

    Uses IBM Plex Sans as the body font and IBM Plex Mono as the
    monospace partner. Both are pulled from Google Fonts at runtime.
    """
    return gr.themes.Base(
        primary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=[
            gr.themes.GoogleFont("IBM Plex Sans"),
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ],
    ).set(
        body_background_fill=BG,
        body_text_color=INK,
        background_fill_primary=BG,
        background_fill_secondary=SURFACE,
        block_background_fill=SURFACE,
        block_border_color=BORDER,
        block_border_width="1px",
        block_radius=RADIUS,
        block_label_text_color=INK_MUTED,
        block_label_background_fill="transparent",
        block_title_background_fill="transparent",
        block_title_text_color=INK_MUTED,
        input_background_fill=SURFACE_STRONG,
        input_border_color=BORDER,
        input_border_color_focus=PRIMARY,
        input_placeholder_color=INK_FAINT,
        button_primary_background_fill=PRIMARY,
        button_primary_text_color=BG,
        button_primary_background_fill_hover=PRIMARY,
        button_secondary_background_fill=SURFACE_STRONG,
        button_secondary_text_color=INK,
        button_secondary_border_color=BORDER_STRONG,
        border_color_primary=BORDER,
        border_color_accent=BORDER_STRONG,
        color_accent=PRIMARY,
        color_accent_soft=SURFACE_RAISED,
    )


# Note: all `!important` below is intentional — Gradio's svelte-hashed
# classes load AFTER our CSS in some browsers, and the framework's
# default rules sit at the same specificity as ours without it.
CSS = f"""
/* ============================================================
 * Brutalist Mono — global palette + Gradio variable overrides
 * Gradio's gr.themes.colors.gray maps to Tailwind slate-*, which has
 * a perceptible blue tint on cool-temperature phone displays. Pin
 * every neutral + surface variable to true monochrome hex.
 * ============================================================ */
:root,
html.dark,
.gradio-container,
.gradio-container.dark,
.gradio-container .dark {{
  --neutral-50:  #FAFAFA !important;
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
  --body-background-fill:           {BG} !important;
  --background-fill-primary:        {BG} !important;
  --background-fill-secondary:      {SURFACE} !important;
  --block-background-fill:          {SURFACE} !important;
  --block-label-background-fill:    transparent !important;
  --block-title-background-fill:    transparent !important;
  --input-background-fill:          {SURFACE_STRONG} !important;
  --border-color-primary:           {BORDER} !important;
  --border-color-accent:            {BORDER_STRONG} !important;
  --color-accent:                   {PRIMARY} !important;
  --body-text-color:                {INK} !important;
  --body-text-color-subdued:        {INK_MUTED} !important;
  --block-label-text-color:         {INK_MUTED} !important;
  --block-title-text-color:         {INK_MUTED} !important;
  --link-text-color:                {INK} !important;
  font-family: {FONT_SANS};
}}
body, .gradio-container {{
  background: {BG} !important;
  color: {INK} !important;
  font-family: {FONT_SANS} !important;
  font-feature-settings: "ss01", "ss03", "cv11";
}}

/* ============================================================
 * Crush Gradio's default ``.app`` wrapper padding.
 * Default ``.gradio-container > .app`` ships with 16px 32px which
 * eats 64 px of the 360 px mobile viewport. Replace with a sane
 * scale that respects the breakpoints.
 * ============================================================ */
.gradio-container > .app,
.gradio-container .main.fillable {{
  padding: 16px 20px !important;
  max-width: none !important;
}}
@media (max-width: 640px) {{
  .gradio-container > .app,
  .gradio-container .main.fillable {{
    padding: 8px 10px !important;
  }}
}}
main, .contain {{
  width: 100% !important;
  max-width: none !important;
}}

/* ============================================================
 * Header + CTA banner
 * ============================================================ */
.ams-header {{
  display:flex; justify-content:space-between; align-items:baseline;
  padding:10px 2px 6px 2px;
}}
.ams-brand {{
  font-family: {FONT_SANS};
  font-size:17px; font-weight:600;
  letter-spacing:-0.01em; color:{INK};
}}
.ams-brand-period {{ color:{PRIMARY}; font-family: {FONT_MONO}; }}
.ams-status {{
  font-family: {FONT_MONO};
  font-size:10px; color:{INK_MUTED};
  letter-spacing:0.08em; text-transform:uppercase;
}}

.ams-cta {{
  font-size:12px; color:{INK_MUTED};
  margin:2px 2px 14px 2px; padding-bottom:10px;
  border-bottom:1px solid {BORDER};
  line-height:1.5;
}}
.ams-cta strong {{ color:{INK}; font-weight:600; }}
.ams-cta-heart {{ color:{PRIMARY}; }}
.ams-cta a {{ color:{INK}; text-decoration:underline; text-decoration-color:{BORDER_STRONG}; }}

/* ============================================================
 * Body row (sidebar + content)
 * ============================================================ */
.ams-body {{
  gap:12px !important;
  align-items:stretch !important;
}}

/* ============================================================
 * Sidebar — desktop ≥ 1024
 * ============================================================ */
.ams-sidebar {{
  background:{SURFACE_STRONG} !important;
  padding:12px 6px !important;
  border-radius:{RADIUS} !important;
  border:1px solid {BORDER} !important;
  min-width:188px;
  max-width:210px;
}}

.ams-side-radio {{
  background:transparent !important;
  border:none !important;
  padding:0 !important;
  width:100%;
}}
.ams-side-radio > div > .wrap,
.ams-side-radio .wrap {{
  display:flex !important;
  flex-direction:column !important;
  gap:2px !important;
  background:transparent !important;
  border:none !important;
}}
.ams-side-radio label {{
  display:flex !important;
  align-items:center !important;
  padding:8px 11px !important;
  margin:0 !important;
  border-radius:3px !important;
  border:none !important;
  border-left:2px solid transparent !important;
  background:transparent !important;
  color:{INK_MUTED} !important;
  font-family: {FONT_SANS} !important;
  font-size:12px !important;
  font-weight:500 !important;
  letter-spacing:0.005em !important;
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
.ams-side-radio label input[type="radio"] {{
  display:none !important;
}}
.ams-side-radio label.selected,
.ams-side-radio label:has(input[type="radio"]:checked) {{
  background:{HOVER_BG} !important;
  color:{PRIMARY} !important;
  border-left-color:{PRIMARY} !important;
  font-weight:600 !important;
}}
.ams-side-radio + div:empty {{ display:none !important; }}

/* History block (below the mode radio) */
.ams-history {{
  margin-top:12px;
  padding-top:8px;
  border-top:1px solid {BORDER};
}}
.ams-history-title {{
  font-family: {FONT_MONO};
  font-size:9px; color:{INK_MUTED};
  letter-spacing:0.12em; text-transform:uppercase;
  padding:0 4px 6px 4px;
}}
.ams-history-empty {{
  font-family: {FONT_SANS};
  font-size:11px; color:{INK_FAINT};
  font-style:italic;
  padding:4px 4px;
}}

/* ============================================================
 * Content pane
 * ============================================================ */
.ams-content {{
  background:{SURFACE} !important;
  border:1px solid {BORDER} !important;
  border-radius:{RADIUS} !important;
  padding:14px !important;
  min-height:540px;
}}
.ams-tab-pane {{
  background:transparent !important;
  border:none !important;
  padding:0 !important;
}}
/* Force the inner 2-column row inside each pane to actually stack
   on narrow screens — Gradio gr.Row keeps row direction by default. */
@media (max-width: 768px) {{
  .ams-tab-pane .row,
  .ams-tab-pane [class*="row"][class*="svelte"] {{
    flex-direction:column !important;
  }}
}}

/* ============================================================
 * Form field chrome — labels, helper info, inputs
 * Gradio's defaults render labels and helper text at 14-16 px.
 * Brutalist Mono wants labels small + uppercase + mono.
 * ============================================================ */
/* Scope the small-uppercase-mono treatment to ONLY the label text
   spans, NOT the entire <label> wrapper. Cascading text-transform
   from a label wrapper would otherwise uppercase the helper text
   (.info-text) and the input's own text. */
.ams-content span.has-info,
.ams-content span.svelte-jdcl7l,
.ams-content .block-label > span,
.ams-content [data-testid="block-label"] > span,
.ams-content .label-wrap > span:first-child {{
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
  color:{INK_MUTED} !important;
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
  padding:0 0 4px 0 !important;
  margin:0 !important;
  font-weight:500 !important;
  display:block !important;
}}
/* Reset the <label> wrapper itself so it doesn't cascade transforms */
.ams-content label.container,
.ams-content label.svelte-1hguek3 {{
  text-transform:none !important;
  font-family:inherit !important;
  font-size:inherit !important;
  letter-spacing:normal !important;
  display:block !important;
  padding:0 !important;
}}
/* Helper text — Gradio 6.14 renders it as <div class="info-text …">.
   Force sentence case + sans + small + faint italic. */
.ams-content .info-text,
.ams-content [class*="info-text"],
.ams-content .block .info,
.ams-content [data-testid="info"] {{
  font-family: {FONT_SANS} !important;
  font-size:10px !important;
  color:{INK_FAINT} !important;
  letter-spacing:0 !important;
  text-transform:none !important;
  font-style:italic !important;
  padding:0 0 6px 0 !important;
  line-height:1.4 !important;
  font-weight:400 !important;
}}

.ams-content textarea,
.ams-content input[type="text"],
.ams-content input[type="number"] {{
  background:{SURFACE_STRONG} !important;
  border:1px solid {BORDER} !important;
  border-radius:3px !important;
  color:{INK} !important;
  font-family: {FONT_SANS} !important;
  font-size:13px !important;
  padding:10px 12px !important;
  line-height:1.5 !important;
}}
.ams-content textarea:focus,
.ams-content input[type="text"]:focus,
.ams-content input[type="number"]:focus {{
  outline:none !important;
  border-color:{PRIMARY} !important;
  box-shadow:none !important;
}}
.ams-content textarea::placeholder,
.ams-content input::placeholder {{
  color:{INK_FAINT} !important;
}}
.ams-content input[type="range"] {{
  accent-color:{PRIMARY} !important;
}}

/* ============================================================
 * Form Radio (Vocal mode) — compact pills, NOT the sidebar tabs
 * ============================================================ */
.ams-content .block:has(input[type="radio"]) .wrap {{
  display:flex !important;
  flex-direction:column !important;
  gap:6px !important;
}}
.ams-content .wrap > label {{
  font-family: {FONT_SANS} !important;
  font-size:12px !important;
  text-transform:none !important;
  letter-spacing:0 !important;
  font-weight:500 !important;
  color:{INK} !important;
  background:{SURFACE_STRONG} !important;
  border:1px solid {BORDER} !important;
  border-radius:3px !important;
  padding:7px 12px !important;
  display:flex !important;
  align-items:center !important;
  gap:8px !important;
  cursor:pointer !important;
}}
.ams-content .wrap > label:hover {{
  border-color:{BORDER_STRONG} !important;
  background:{HOVER_BG} !important;
}}
.ams-content .wrap > label:has(input[type="radio"]:checked) {{
  border-color:{PRIMARY} !important;
  background:{SURFACE_RAISED} !important;
}}
/* Custom radio dot */
.ams-content .wrap > label input[type="radio"] {{
  appearance:none !important;
  -webkit-appearance:none !important;
  width:12px !important; height:12px !important;
  border:1px solid {BORDER_STRONG} !important;
  border-radius:50% !important;
  margin:0 !important;
  flex-shrink:0 !important;
}}
.ams-content .wrap > label input[type="radio"]:checked {{
  border-color:{PRIMARY} !important;
  background: radial-gradient({PRIMARY} 0 4px, transparent 5px) !important;
}}

/* ============================================================
 * Primary button (▶ Generate)
 * ============================================================ */
.ams-content button.primary {{
  background:{PRIMARY} !important;
  color:{BG} !important;
  border:none !important;
  border-radius:3px !important;
  font-family: {FONT_SANS} !important;
  font-size:13px !important;
  font-weight:600 !important;
  letter-spacing:0.005em !important;
  padding:11px 18px !important;
  cursor:pointer !important;
  transition:transform 80ms ease, opacity 80ms ease;
  margin-top:6px !important;
}}
.ams-content button.primary:hover {{
  opacity:0.92 !important;
  transform:translateY(-1px);
}}
.ams-content button.primary:active {{
  transform:translateY(0);
}}

/* ============================================================
 * Output panel — gr.Audio (.ams-out-audio) and gr.JSON (.ams-out-meta)
 * Targeted via the elem_classes hooks defined in ui.py so we don't
 * have to chase svelte-hashed class names.
 * ============================================================ */
.ams-content .ams-out {{
  background:{SURFACE_STRONG} !important;
  border:1px solid {BORDER} !important;
  border-radius:3px !important;
  padding:12px !important;
  margin-top:10px !important;
}}
.ams-content .ams-out-audio {{
  min-height:90px !important;
}}
.ams-content .ams-out-meta {{
  min-height:80px !important;
  font-family: {FONT_MONO} !important;
  font-size:11px !important;
  line-height:1.6 !important;
}}
.ams-content .ams-out-meta span {{
  font-family: {FONT_MONO} !important;
  font-size:11px !important;
}}
/* The Output/Metadata block labels live as <label class="svelte-19djge9">.
   Force the Brutalist label treatment + uppercase regardless of the
   svelte hash, since they're inside .ams-out. */
.ams-content .ams-out label,
.ams-content .ams-out label > span {{
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
  color:{INK_MUTED} !important;
  background:transparent !important;
  font-weight:500 !important;
  padding:0 0 6px 0 !important;
  display:block !important;
}}
/* Empty-state svg glyphs (music note, JSON braces) — center + muted */
.ams-content .ams-out svg {{
  color:{INK_FAINT} !important;
  opacity:0.5 !important;
}}

/* Add explicit gap between form column and output column when stacked */
@media (max-width: 768px) {{
  .ams-tab-pane > .row,
  .ams-tab-pane > [class*="row"] {{
    gap:18px !important;
  }}
  .ams-content .block + .block {{
    margin-top:4px !important;
  }}
}}

/* ============================================================
 * LoRA chip pill — kept for M2 wiring
 * ============================================================ */
.ams-chip {{
  display:inline-block; padding:5px 10px; border-radius:14px;
  font-size:11px; margin:0 5px 5px 0; background:{SURFACE_STRONG};
  border:1px solid {BORDER_STRONG}; color:{INK_MUTED}; cursor:pointer;
}}
.ams-chip.on {{ border-color:{PRIMARY}; color:{PRIMARY}; }}
.ams-chip.upload {{ border-style:dashed; color:{PRIMARY}; }}
.ams-lora-file .upload-container {{ min-height:56px !important; }}

/* Hide Gradio footer + the floating "Use via API" / settings panel */
footer {{ display:none !important; }}
.show-api {{ display:none !important; }}
.built-with {{ display:none !important; }}

/* ============================================================
 * Responsive: tablet 640-1024 px
 * Keep the full sidebar (with labels) — the icon-rail middle state
 * fights Gradio's DOM. Just narrow it slightly.
 * ============================================================ */
@media (min-width: 641px) and (max-width: 1024px) {{
  .ams-sidebar {{ min-width:160px; max-width:180px; padding:10px 6px !important; }}
  .ams-side-radio label {{ font-size:11px !important; padding:7px 9px !important; }}
}}

/* ============================================================
 * Responsive: mobile < 640 px
 * Sidebar becomes a horizontal scroll pill strip at the top.
 * Form + Output stack with proper gap.
 * ============================================================ */
@media (max-width: 640px) {{
  .ams-body {{
    flex-direction:column !important;
    gap:8px !important;
  }}
  .ams-sidebar {{
    min-width:100% !important;
    max-width:100% !important;
    padding:2px 0 !important;
    border:none !important;
    background:transparent !important;
    border-radius:0 !important;
  }}
  .ams-side-radio .wrap {{
    flex-direction:row !important;
    flex-wrap:nowrap !important;
    overflow-x:auto !important;
    overflow-y:hidden !important;
    gap:6px !important;
    padding:2px 0 !important;
    scrollbar-width:none !important;
    -ms-overflow-style:none !important;
  }}
  .ams-side-radio .wrap::-webkit-scrollbar {{ display:none !important; }}
  .ams-side-radio label {{
    width:auto !important;
    min-width:0 !important;
    max-width:max-content !important;
    flex:0 0 auto !important;
    font-family: {FONT_SANS} !important;
    font-size:11px !important;
    font-weight:600 !important;
    white-space:nowrap !important;
    padding:7px 12px !important;
    background:{SURFACE_STRONG} !important;
    border:1px solid {BORDER} !important;
    border-radius:3px !important;
    justify-content:center !important;
  }}
  .ams-side-radio label.selected,
  .ams-side-radio label:has(input[type="radio"]:checked) {{
    border-color:{PRIMARY} !important;
    background:{SURFACE_RAISED} !important;
    color:{PRIMARY} !important;
  }}
  .ams-history {{ display:none !important; }}

  /* Header + CTA tighter */
  .ams-header {{ padding:2px 2px 2px 2px !important; }}
  .ams-brand {{ font-size:14px !important; }}
  .ams-status {{ font-size:9px !important; }}
  .ams-cta {{ font-size:11px !important; margin:0 2px 8px 2px !important; padding-bottom:8px !important; }}

  /* Content pane tighter */
  .ams-content {{ padding:12px !important; border-radius:3px !important; }}

  /* Field labels + info shrink further */
  .ams-content label,
  .ams-content .block-label,
  .ams-content [data-testid="block-label"],
  .ams-content span.svelte-jdcl7l,
  .ams-content .label-wrap {{
    font-size:9px !important;
  }}
  .ams-content .block .info,
  .ams-content [data-testid="info"] {{
    font-size:9px !important;
    padding-bottom:4px !important;
  }}
  .ams-content textarea,
  .ams-content input[type="text"],
  .ams-content input[type="number"] {{
    font-size:12px !important;
    padding:8px 10px !important;
  }}
  .ams-content .wrap > label {{
    padding:6px 10px !important;
    font-size:11px !important;
  }}
  .ams-content button.primary {{
    padding:11px 14px !important;
    font-size:12px !important;
  }}
  .ams-content [data-testid="audio"],
  .ams-content .audio-container,
  .ams-content .json-holder {{
    min-height:64px !important;
  }}
}}
"""
