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
  /* Cap the body at the viewport and clip any horizontal overflow so a
     long-clip waveform inside .ams-content can't push the row sideways. */
  max-width:100% !important;
  overflow:hidden !important;
}}
/* IMPORTANT: do NOT apply ``min-width: 0`` to ``.ams-body > *`` — that
   selector also matches ``.ams-sidebar``, overriding its 188 px
   min-width and collapsing it to almost nothing on desktop (seen as a
   vertical sliver of stacked single characters). The flex-shrink fix
   we need is on ``.ams-content`` only; sidebar keeps its hard minimum. */

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
  /* Flex children default to ``min-width: auto`` which means they CANNOT
     shrink below their content's intrinsic width. The wavesurfer.js
     waveform renders pixel-perfect to the audio duration (e.g. a 60 s
     clip wants ~600 px), which would push this column wider than the
     viewport on mobile and cause the layout to "dance" between
     pre-generation and post-generation widths. ``min-width: 0`` lets
     the column shrink, and the audio block's own ``overflow: hidden``
     clips the inner waveform. */
  min-width:0 !important;
  /* Match: the row child is also constrained so the audio waveform
     can't push it out of bounds on mobile. */
  max-width:100% !important;
  overflow-x:hidden !important;
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
 * Checkbox — native browser checkboxes render almost invisibly on
 * the Brutalist Mono dark palette (white outline on dark surface,
 * no visible fill when checked). Replace with a custom drawn box:
 *   - unchecked: dark surface, subtle 1.5 px white border
 *   - checked:   WHITE fill with a black checkmark SVG drawn inline
 *                (no external asset, no CSP issues)
 *   - hover:     border brightens
 *   - focus:     2 px white outline
 * ``accent-color`` alone isn't enough — Gradio 6.14's checkbox style
 * has tiny dimensions (12 px) and a transparent background that
 * still hides the indicator on cool-temperature phone displays.
 * ============================================================ */
.ams-content input[type="checkbox"] {{
  -webkit-appearance:none !important;
  appearance:none !important;
  width:16px !important;
  height:16px !important;
  min-width:16px !important;
  min-height:16px !important;
  margin:0 8px 0 0 !important;
  padding:0 !important;
  border:1.5px solid {INK_MUTED} !important;
  border-radius:2px !important;
  background:{SURFACE_STRONG} !important;
  cursor:pointer !important;
  transition:background 80ms ease, border-color 80ms ease;
  vertical-align:middle !important;
  flex-shrink:0;
}}
.ams-content input[type="checkbox"]:hover {{
  border-color:{INK} !important;
}}
.ams-content input[type="checkbox"]:focus-visible {{
  outline:2px solid {PRIMARY} !important;
  outline-offset:2px !important;
}}
.ams-content input[type="checkbox"]:checked {{
  background:{PRIMARY} !important;
  border-color:{PRIMARY} !important;
  /* Black checkmark drawn inline (no external assets) — uses a small
     polyline SVG sized to fit the 16 px box. */
  background-image:url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='%230A0A0A' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3,8 7,12 13,4'/%3E%3C/svg%3E") !important;
  background-repeat:no-repeat !important;
  background-position:center !important;
  background-size:12px 12px !important;
}}
/* The label that wraps the checkbox + text should align them on a
   single baseline so the new larger box doesn't push the text down. */
.ams-content label.checkbox-container {{
  display:inline-flex !important;
  align-items:center !important;
  gap:2px !important;
  cursor:pointer !important;
  padding:4px 0 !important;
}}
.ams-content label.checkbox-container .label-text {{
  font-family: {FONT_SANS} !important;
  font-size:12px !important;
  color:{INK} !important;
  letter-spacing:0;
  text-transform:none !important;
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
  /* Pure pitch-black background so cool-display tinting can't read
     into a neutral grey as bluish. */
  background:#000 !important;
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
  background:#0F0F0F !important;
}}
.ams-content .wrap > label:has(input[type="radio"]:checked) {{
  border-color:{PRIMARY} !important;
  background:#0F0F0F !important;
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
  /* Outer panel: cage everything so a long-clip wavesurfer canvas
     can't push the parent column wider than the viewport. The inner
     wavesurfer layout below is allowed to render freely — we only
     clip at THIS boundary. */
  min-width:0 !important;
  max-width:100% !important;
  overflow:hidden !important;
}}
/* Inner wavesurfer wrappers: let them lay out their grid/flex children
   freely. Earlier we set ``overflow: hidden`` on these too, which
   hid the play/skip controls + 1:00 timestamp during transient
   wavesurfer re-renders (controls would briefly compute wider than
   the wrapper and get clipped → user saw the "fluctuating" preview).
   Width / max-width caps remain so the layout still respects the
   viewport. */
.ams-content .ams-out-audio .component-wrapper {{
  width:100% !important;
  max-width:100% !important;
  min-width:0 !important;
  overflow:visible !important;
}}
.ams-content .ams-out-audio .waveform-container,
.ams-content .ams-out-audio [data-testid^="waveform"],
.ams-content .ams-out-audio #waveform {{
  width:100% !important;
  max-width:100% !important;
  min-width:0 !important;
  /* Clip the canvas only — keep the wave bars from spilling but let
     the parent containers show their controls. */
  overflow:hidden !important;
}}
/* Wavesurfer renders a <canvas> sized in CSS pixels from the audio
   duration; force it to the wrapper's width on small screens. */
.ams-content .ams-out-audio canvas,
.ams-content .ams-out-audio wave {{
  max-width:100% !important;
}}
/* Reserve vertical space for the timestamps row and the controls
   row so they NEVER collapse to zero during a wavesurfer re-render
   transition. Without this min-height, the play/skip/forward icons
   briefly vanish on mobile after a generation completes, giving the
   "preview is fluctuating" look. */
.ams-content .ams-out-audio .timestamps {{
  min-height:24px !important;
  overflow:visible !important;
}}
.ams-content .ams-out-audio .controls {{
  min-height:60px !important;
  overflow:visible !important;
}}
.ams-content .ams-out-audio .play-pause-wrapper,
.ams-content .ams-out-audio .control-wrapper {{
  min-width:0 !important;
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
/* Hide Gradio's tiny inline icon SVG that sits inside the Output /
   Metadata label (an 11x11 music-note / braces glyph). The wireframe
   doesn't show these; they overlap the label text and create a
   "broken" feel. Note that the LARGE empty-state SVG (centered in
   the panel body) lives inside ``.empty.svelte-v95lt3`` and is
   spared by this selector since it's not a direct label child. */
.ams-content .ams-out label svg,
.ams-content .ams-out > label > span:first-child:has(svg) {{
  display:none !important;
}}
/* The large empty-state SVG centered in the panel body — keep, but
   make it visually softer (muted faint ink, low opacity). */
.ams-content .ams-out .empty svg,
.ams-content .ams-out [class*="empty"] svg {{
  color:{INK_FAINT} !important;
  opacity:0.5 !important;
}}

/* ============================================================
 * Defeat Gradio's ``div.styler`` wrapper backgrounds.
 * Gradio wraps every Row / Form in a <div class="styler svelte-..."
 * which has background:var(--border-color-primary) = #1F1F1F by
 * default. That produces visible slate-blue "bands" between the
 * form rows (most obviously around the Generate button) on cool
 * displays. Force it transparent everywhere inside our content.
 * ============================================================ */
.ams-content .styler,
.ams-content [class*="styler"],
.ams-content .form > .styler {{
  background:transparent !important;
  border:none !important;
  padding:0 !important;
}}

/* Defeat any residual Gradio block bg that might pull a slate value
   despite our --neutral-* override (defensive, harmless if it's
   already correct). */
.ams-content > .row,
.ams-content > .row > .column,
.ams-content .form {{
  background:transparent !important;
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
 * LoRA chip pill — kept for legacy use (chip-style hooks elsewhere)
 * ============================================================ */
.ams-chip {{
  display:inline-block; padding:5px 10px; border-radius:14px;
  font-size:11px; margin:0 5px 5px 0; background:{SURFACE_STRONG};
  border:1px solid {BORDER_STRONG}; color:{INK_MUTED}; cursor:pointer;
}}
.ams-chip.on {{ border-color:{PRIMARY}; color:{PRIMARY}; }}
.ams-chip.upload {{ border-style:dashed; color:{PRIMARY}; }}

/* ============================================================
 * LoRA accordion (D5)
 * The collapsed accordion sits between the duration/vocal-mode row
 * and the Generate button. Inside: a note, preset radio, custom
 * upload, strength slider, and an "Active: …" Markdown line.
 * The outer chrome matches the wireframe's bordered section header.
 * ============================================================ */
.ams-content .ams-lora {{
  border:1px solid {BORDER} !important;
  border-radius:3px !important;
  background:{SURFACE_STRONG} !important;
  margin-top:10px !important;
  padding:0 !important;
}}
/* Accordion summary / label. Gradio 6.14 renders this as either
   .label-wrap (older builds) or a native <summary> element. Style
   both so the uppercase-mono header is consistent. */
.ams-content .ams-lora > .label-wrap,
.ams-content .ams-lora summary,
.ams-content .ams-lora > button {{
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
  color:{INK_MUTED} !important;
  padding:10px 12px !important;
  background:transparent !important;
  border:none !important;
}}
.ams-content .ams-lora > .label-wrap span,
.ams-content .ams-lora summary span,
.ams-content .ams-lora > button span {{
  color:{INK_MUTED} !important;
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
}}
/* Italic note under the header */
.ams-content .ams-lora-note p {{
  font-family: {FONT_SANS} !important;
  font-size:10px !important;
  font-style:italic !important;
  color:{INK_FAINT} !important;
  line-height:1.4 !important;
  margin:0 0 10px 0 !important;
  padding:0 12px !important;
}}
/* The expanded body padding — Gradio drops the children inside
   .gap (or unnamed wrapper) directly. Use a left/right padding so
   the radio + file + slider don't hug the border. */
.ams-content .ams-lora > div:not(.label-wrap):not(summary) {{
  padding:0 12px 12px 12px !important;
}}
/* Preset radio: row of compact pills. Gradio renders the radio body
   as ``fieldset.ams-lora-preset > div.wrap.svelte-e4x47i > label*``.
   The generic Vocal-mode rule
       .ams-content .block:has(input[type="radio"]) .wrap
   computes specificity (0,4,1) — three classes + the inner attribute
   selector via :has. To beat that we chain ``.ams-content .ams-lora
   .ams-lora-preset.ams-lora-preset > .wrap`` which is (0,5,0), winning
   by one class. */
.ams-content .ams-lora .ams-lora-preset.ams-lora-preset > .wrap {{
  display:flex !important;
  flex-direction:row !important;
  flex-wrap:wrap !important;
  gap:6px !important;
  background:transparent !important;
  border:none !important;
  padding:0 !important;
  width:100% !important;
}}
.ams-content .ams-lora .ams-lora-preset.ams-lora-preset > .wrap > label {{
  flex:0 0 auto !important;
  width:auto !important;
  max-width:max-content !important;
  min-width:0 !important;
  background:#000 !important;
  border:1px solid {BORDER} !important;
  border-radius:14px !important;
  padding:5px 12px !important;
  font-size:11px !important;
  color:{INK_MUTED} !important;
  font-weight:500 !important;
  display:inline-flex !important;
  align-items:center !important;
  gap:0 !important;
  cursor:pointer !important;
}}
.ams-content .ams-lora .ams-lora-preset.ams-lora-preset > .wrap > label:hover {{
  color:{INK} !important;
  border-color:{BORDER_STRONG} !important;
}}
.ams-content .ams-lora .ams-lora-preset.ams-lora-preset > .wrap > label:has(input[type="radio"]:checked) {{
  border-color:{PRIMARY} !important;
  color:{PRIMARY} !important;
  background:#0F0F0F !important;
}}
/* Hide the inner radio-dot input; the pill border + color carries
   the on/off state on its own. */
.ams-content .ams-lora .ams-lora-preset.ams-lora-preset > .wrap > label input[type="radio"] {{
  display:none !important;
  width:0 !important; height:0 !important;
  margin:0 !important; padding:0 !important;
  background:none !important;
  border:none !important;
}}
.ams-content .ams-lora .ams-lora-preset.ams-lora-preset > .wrap > label span {{
  text-transform:none !important;
  letter-spacing:0 !important;
  font-family: {FONT_SANS} !important;
  font-size:11px !important;
  color:inherit !important;
  font-weight:500 !important;
}}

/* Custom-upload file widget. Gradio 6.14 renders the drop-zone as
   ``button.svelte-8prmba`` (NOT ``.upload-container``). The label
   above it is ``label.svelte-19djge9.float`` which carries the
   uploaded-file metadata once a file is dropped. Style the actual
   drop-button and override the legacy ``.upload-container`` rule
   from above for forward compatibility. */
.ams-content .ams-lora-file > button {{
  min-height:80px !important;
  background:#000 !important;
  border:1px dashed {BORDER_STRONG} !important;
  border-radius:3px !important;
  color:{INK_MUTED} !important;
  padding:14px 12px !important;
}}
.ams-content .ams-lora-file > button:hover {{
  border-color:{PRIMARY} !important;
  color:{INK} !important;
}}
.ams-content .ams-lora-file > button .or {{
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  color:{INK_FAINT} !important;
  letter-spacing:0.04em !important;
}}
.ams-content .ams-lora-file > button .icon-wrap svg {{
  color:{INK_MUTED} !important;
  opacity:0.7 !important;
  width:18px !important;
  height:18px !important;
}}
/* The floating label that appears once a file is uploaded — give it
   the standard Brutalist mono treatment and hide its decorative SVG
   so the label text reads cleanly. */
.ams-content .ams-lora-file > label.float {{
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
  color:{INK_MUTED} !important;
  background:transparent !important;
  border:none !important;
  padding:0 0 6px 0 !important;
}}
.ams-content .ams-lora-file > label.float svg {{
  display:none !important;
}}

/* Strength slider — the .info text just below it inherits the
   generic helper rule, so no extra work needed. */
.ams-content .ams-lora-strength input[type="range"] {{
  accent-color:{PRIMARY} !important;
}}

/* Active LoRA display — high-contrast block, mono font, code in
   white so the LoRA name pops. The accordion's own background is
   SURFACE_STRONG (true black), so use a slightly raised surface
   here to make the bordered box visible. */
.ams-content .ams-lora-active .prose p {{
  font-family: {FONT_MONO} !important;
  font-size:11px !important;
  color:{INK} !important;
  background:{SURFACE_RAISED} !important;
  border:1px solid {BORDER_STRONG} !important;
  border-radius:3px !important;
  padding:8px 10px !important;
  margin:8px 0 0 0 !important;
  line-height:1.5 !important;
}}
.ams-content .ams-lora-active .prose code {{
  background:transparent !important;
  color:{PRIMARY} !important;
  font-family: {FONT_MONO} !important;
  font-size:11px !important;
  padding:0 !important;
}}
.ams-content .ams-lora-active .prose em {{
  color:{INK_FAINT} !important;
  font-style:italic !important;
  font-family: {FONT_SANS} !important;
}}
@media (max-width: 640px) {{
  .ams-content .ams-lora > .label-wrap,
  .ams-content .ams-lora summary,
  .ams-content .ams-lora > button {{
    font-size:9px !important;
    padding:8px 10px !important;
  }}
  .ams-content .ams-lora-note p {{
    font-size:9px !important;
    padding:0 10px !important;
  }}
  .ams-content .ams-lora > div:not(.label-wrap):not(summary) {{
    padding:0 10px 10px 10px !important;
  }}
  .ams-content .ams-lora-active .prose p {{
    font-size:10px !important;
    padding:6px 8px !important;
  }}
  .ams-content .ams-lora .ams-lora-preset.ams-lora-preset > .wrap > label {{
    font-size:10px !important;
    padding:5px 9px !important;
  }}
  .ams-content .ams-lora-file > button {{
    min-height:64px !important;
    padding:10px 8px !important;
  }}
}}

/* ============================================================
 * Audio upload widget (Cover / Extend / Edit reference inputs)
 * Tagged with ``ams-input-audio`` via elem_classes. Match the dark
 * input chrome so it sits next to the textboxes without contrast
 * jumps; the gr.Audio drop-button gets the same dashed outline as
 * the LoRA upload so users recognise it as a drop-zone.
 * ============================================================ */
.ams-content .ams-input-audio {{
  background:{SURFACE_STRONG} !important;
  border:1px solid {BORDER} !important;
  border-radius:3px !important;
  padding:8px !important;
  margin-bottom:4px !important;
}}
.ams-content .ams-input-audio .empty,
.ams-content .ams-input-audio [class*="empty"] {{
  min-height:90px !important;
}}
.ams-content .ams-input-audio button {{
  background:#000 !important;
  border:1px dashed {BORDER_STRONG} !important;
  border-radius:3px !important;
  color:{INK_MUTED} !important;
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.04em !important;
}}
.ams-content .ams-input-audio button:hover {{
  border-color:{PRIMARY} !important;
  color:{INK} !important;
}}
.ams-content .ams-input-audio svg {{
  color:{INK_MUTED} !important;
  opacity:0.7 !important;
}}

/* ============================================================
 * Experimental accordion (Extend / Edit)
 * Reuse the LoRA accordion's visual chrome so the bordered section
 * shape is consistent across all accordions, but visually demote
 * the summary so users can tell these knobs aren't fully wired.
 * ============================================================ */
.ams-content .ams-experimental {{
  border:1px solid {BORDER} !important;
  border-radius:3px !important;
  background:{SURFACE_STRONG} !important;
  margin-top:10px !important;
  padding:0 !important;
}}
.ams-content .ams-experimental > .label-wrap,
.ams-content .ams-experimental summary,
.ams-content .ams-experimental > button {{
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
  color:{INK_MUTED} !important;
  padding:10px 12px !important;
  background:transparent !important;
  border:none !important;
  opacity:0.7 !important;
}}
.ams-content .ams-experimental > .label-wrap span,
.ams-content .ams-experimental summary span,
.ams-content .ams-experimental > button span {{
  color:{INK_MUTED} !important;
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
}}
.ams-content .ams-experimental > div:not(.label-wrap):not(summary) {{
  padding:0 12px 12px 12px !important;
}}

/* ============================================================
 * Lyrics tab (M4)
 * Mono draft textbox + secondary "Use in Generate" CTA. The LM-params
 * accordion reuses the same chrome as the LoRA + experimental
 * accordions so the bordered section header reads consistently.
 * ============================================================ */
.ams-content .ams-lyrics-output textarea {{
  font-family: {FONT_MONO} !important;
  font-size: 12px !important;
  line-height: 1.6 !important;
  min-height: 280px !important;
  background:{SURFACE_STRONG} !important;
  border:1px solid {BORDER} !important;
  color:{INK} !important;
}}
.ams-content .ams-lyrics-output textarea::placeholder {{
  font-style: italic;
}}
.ams-content .ams-lyrics-use-btn {{
  margin-top: 6px !important;
}}
.ams-content .ams-lm-accordion {{
  border:1px solid {BORDER} !important;
  border-radius:3px !important;
  background:{SURFACE_STRONG} !important;
  margin-top:10px !important;
  padding:0 !important;
}}
.ams-content .ams-lm-accordion > .label-wrap,
.ams-content .ams-lm-accordion summary,
.ams-content .ams-lm-accordion > button {{
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
  color:{INK_MUTED} !important;
  padding:10px 12px !important;
  background:transparent !important;
  border:none !important;
}}
.ams-content .ams-lm-accordion > .label-wrap span,
.ams-content .ams-lm-accordion summary span,
.ams-content .ams-lm-accordion > button span {{
  color:{INK_MUTED} !important;
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
}}
.ams-content .ams-lm-accordion > div:not(.label-wrap):not(summary) {{
  padding:0 12px 12px 12px !important;
}}

/* ============================================================
 * Advanced controls accordion (M0-X)
 * Bordered chrome matching the LoRA + LM + experimental accordions so
 * the four song-mode panes read consistently. Inside the accordion we
 * additionally render small <h>/<p strong> section headers (Diffusion,
 * CFG schedule, 5Hz LM, Music metadata) to chunk the 21 knobs into
 * logical groups; those need their own mono-uppercase-faint treatment
 * so they don't compete with the form labels for visual weight.
 * ============================================================ */
.ams-content .ams-advanced {{
  border:1px solid {BORDER} !important;
  border-radius:3px !important;
  background:{SURFACE_STRONG} !important;
  margin-top:10px !important;
  padding:0 !important;
}}
.ams-content .ams-advanced > .label-wrap,
.ams-content .ams-advanced summary,
.ams-content .ams-advanced > button {{
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
  color:{INK_MUTED} !important;
  padding:10px 12px !important;
  background:transparent !important;
  border:none !important;
}}
.ams-content .ams-advanced > .label-wrap span,
.ams-content .ams-advanced summary span,
.ams-content .ams-advanced > button span {{
  color:{INK_MUTED} !important;
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.08em !important;
  text-transform:uppercase !important;
}}
.ams-content .ams-advanced > div:not(.label-wrap):not(summary) {{
  padding:0 12px 12px 12px !important;
}}
/* Section divider Markdown headers inside the accordion. We render them
   as **Diffusion** (etc) via gr.Markdown — Gradio wraps that in
   ``.prose strong``. Treat the strong tag as a small mono uppercase
   header with a subtle underline so the four groups have clear visual
   boundaries without competing with the actual form labels. */
.ams-content .ams-advanced .ams-adv-section .prose p {{
  margin:14px 0 4px 0 !important;
  padding:0 0 4px 0 !important;
  border-bottom:1px solid {BORDER} !important;
}}
.ams-content .ams-advanced .ams-adv-section .prose p:first-child {{
  margin-top:6px !important;
}}
.ams-content .ams-advanced .ams-adv-section .prose strong {{
  font-family: {FONT_MONO} !important;
  font-size:10px !important;
  letter-spacing:0.12em !important;
  text-transform:uppercase !important;
  color:{INK} !important;
  font-weight:600 !important;
}}
.ams-content .ams-advanced .ams-adv-section .prose {{
  background:transparent !important;
}}
@media (max-width: 640px) {{
  .ams-content .ams-advanced > .label-wrap,
  .ams-content .ams-advanced summary,
  .ams-content .ams-advanced > button {{
    font-size:9px !important;
    padding:8px 10px !important;
  }}
  .ams-content .ams-advanced > div:not(.label-wrap):not(summary) {{
    padding:0 10px 10px 10px !important;
  }}
  .ams-content .ams-advanced .ams-adv-section .prose strong {{
    font-size:9px !important;
  }}
}}

/* ============================================================
 * Post-process action row (M5/G2) — sits below the Output Audio.
 * Three compact mono pills (separate stems / normalise / mp3 export)
 * that surface hidden gr.Files / gr.Audio / gr.File widgets once a
 * post-process click handler returns. The bordered list chrome on
 * stem_files + mp3_file matches the generic .ams-out treatment so
 * the populated state reads as a continuation of the Output panel.
 * ============================================================ */
.ams-content .ams-post-actions {{
  gap: 6px !important;
  margin: 8px 0 0 0 !important;
}}
.ams-content .ams-post-btn {{
  font-family: {FONT_MONO} !important;
  font-size: 11px !important;
  letter-spacing: 0.04em !important;
  padding: 8px 10px !important;
  background: #000 !important;
  border: 1px solid {BORDER} !important;
  color: {INK} !important;
  border-radius: 3px !important;
}}
.ams-content .ams-post-btn:hover {{
  border-color: {PRIMARY} !important;
}}
/* Stem files + mp3 file widgets — compact bordered list */
.ams-content .ams-stem-files,
.ams-content .ams-mp3-file {{
  background: #000 !important;
  border: 1px solid {BORDER} !important;
  border-radius: 3px !important;
  margin-top: 6px !important;
}}
@media (max-width: 640px) {{
  .ams-content .ams-post-btn {{
    font-size: 10px !important;
    padding: 7px 8px !important;
  }}
}}

/* ============================================================
 * History rows — clickable-looking compact list (M6/H2)
 * Replaces the static "No generations yet" placeholder with a live
 * in-memory feed of mode + label tuples. The mode segment renders in
 * mono uppercase to mirror the small uppercase labels used throughout
 * the sidebar; the label segment uses the sans body face and truncates
 * with ellipsis at the sidebar's compact 188-210 px width.
 * ============================================================ */
.ams-content .ams-history-wrapper {{
  margin-top: 4px;
}}
.ams-history-row {{
  display: flex;
  gap: 6px;
  align-items: baseline;
  font-family: {FONT_MONO};
  font-size: 10px;
  color: {INK_MUTED};
  padding: 4px 6px;
  border-radius: 3px;
  cursor: default;
}}
.ams-history-row:hover {{
  background: {HOVER_BG};
  color: {INK};
}}
.ams-history-mode {{
  color: {PRIMARY};
  text-transform: lowercase;
  letter-spacing: 0;
  flex-shrink: 0;
}}
.ams-history-label {{
  color: {INK_MUTED};
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: {FONT_SANS};
  font-size: 11px;
}}
@media (max-width: 640px) {{
  .ams-history,
  .ams-history-wrapper {{
    display: none !important;
  }}
}}

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
