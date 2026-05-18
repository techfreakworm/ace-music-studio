# ACE Music Studio — UI mockups

Visual source-of-truth for the design spec at `../2026-05-18-ace-music-studio-design.md`. Open the HTML files in a browser to see the rendered Brutalist Mono interface.

| File | Tabs / screens covered | Source |
|---|---|---|
| [`01_generate_mobile_errors.html`](./01_generate_mobile_errors.html) | **Generate** tab fully expanded · 3 phone screens (Generate, Cover, Lyrics) · 6 error / edge-case states · in-progress generation banner | brainstorm session 24743 |
| [`02_cover_extend.html`](./02_cover_extend.html) | **Cover** tab fully expanded · **Extend** tab fully expanded | brainstorm session 24743 |
| [`03_edit_lyrics.html`](./03_edit_lyrics.html) | **Edit** tab fully expanded with both sub-modes (Repaint active, Flow Morph dimmed) · **Lyrics** tab fully expanded with Qwen 2.5 LM params | brainstorm session 24743 |

## What every tab shares

- Sticky header with brand "ACE Music Studio." and CTA: *Built with ♥. Drop a like · Follow @techfreakworm for what's next.*
- Sidebar with 5 mode pills + session History list (desktop ≥ 1024 px)
- 2-column body: form on left, output on right
- LoRA stack section with 4 bundled preset chips + active stack rows (per-row strength slider + ×) + custom upload zone
- Advanced accordion: BPM, key/scale, time sig, sampler, language, steps, CFG, shift, negative prompt, audio format, loudness, fade in/out, seed + lock
- LM planner accordion: thinking, constrained decoding, temp / top-k / top-p / LM CFG, CoT toggles (metas / caption / lyrics / language), LM negative prompt, CoT override fields
- DCW accordion: enabled, mode (single / double), wavelet, scaler, high scaler
- Output panel: waveform · play/scrub · retake · stems (Demucs htdemucs_ft) · export (mp3 / wav / stems zip / meta JSON / share link) · full metadata JSON

## What each tab adds

- **Generate** — duration slider, vocals/instrumental pills, CFG-interval start/end, latent shift/rescale
- **Cover** — reference-audio dropzone, cover-strength slider, cover-noise slider, compare-side-by-side toggle in output
- **Extend** — seed-audio dropzone with auto-detected BPM/key, extension prompt, extra-duration slider, repaint mode, repaint strength, latent crossfade frames, WAV crossfade seconds, chunk mask mode, seed-boundary marker on output waveform, separate "extension-only" download
- **Edit** — source audio + source/target lyrics, repaint-vs-flow-morph sub-mode pills, segment-selection bar with start/end timestamps, repaint sub-options (mode / chunk-mask / strength / crossfade), flow-morph sub-options (source caption / n_min / n_max / n_avg), A/B comparison in output
- **Lyrics** — brief, structure sequence, language, per-section line counts (verse / chorus / bridge), tone descriptors, rhyme preference pills (strict / loose / none), LM params accordion (temp / top-p / top-k / rep penalty / max tokens / seed / show system prompt / enforce-tag-format), quick-refinement chips (more cryptic, less rhyme, etc.), variants

## Mobile (phone)

- Native `gr.Tabs` horizontal scroll strip at top (icons + first label visible)
- Sidebar hidden via CSS media query at `< 640 px`
- Output stacks below form
- Sliders bounded by parent width (the desktop's pixel-art `━` characters were replaced with proper CSS slider tracks for mobile)

## Error / edge states

- **LoRAValidationError** — toast with module-mismatch diagnostics + "Remove from stack" / "View header diagnostics" actions
- **ZeroGPU timeout** — auto-retry once at 2× duration, then warning toast with "Lower steps" / "Reduce duration" hints
- **MPS op fallback** — info toast naming the op (e.g., `aten::_fft_r2c`), CPU fallback engaged via `PYTORCH_ENABLE_MPS_FALLBACK=1`
- **Audio format rejected** — clear constraints (wav/mp3/flac, ≤ 60 s for Cover, ≤ 50 MB) + "Auto-convert + trim" action
- **First-request warm-up** — informational banner ("Loading ACE-Step v1.5 XL SFT into MPS memory ~45 s")
- **In-progress generation** — `gr.Progress`-driven banner with step / total, ETA, elapsed, cancel link

## Note on the "approve / revise" cards

Each HTML file has a card-options block at the bottom — vestigial from the visual-companion brainstorm flow. It's harmless when viewed outside the companion (the `toggleSelect` call is a no-op without the companion's helper.js).

If they bother you, delete the trailing `<div class="options">…</div>` block from each file. Otherwise leave them — they document which question each mockup answered.
