# ACE Music Studio — Design Spec

**Date:** 2026-05-18
**Status:** Approved — ready for implementation plan
**Repo:** `~/Projects/llm/music-generator/` → GitHub `techfreakworm/ace-music-studio` (to be created)
**HF Space:** `huggingface.co/spaces/techfreakworm/ace-music-studio` (to be created)
**Companion docs:** `research/00_executive_summary.md` (model selection rationale)

---

## 1. Goal

A single-process Gradio app that wraps **ACE-Step 1.5 XL SFT** for full-song generation with vocals, deployable both to a free non-profit **Hugging Face ZeroGPU Space** and locally on **Apple M5 Max (MPS / MLX)** or **NVIDIA (CUDA)** workstations. Supports the full ACE-Step feature surface — text-to-song, audio-reference cover, song extension, segment-level edit/repaint, plus an in-app lyrics writer powered by a bundled small LM. Users can stack any number of LoRAs from a curated preset library or upload custom `.safetensors` files at runtime.

Non-goals (v1): commercial-tier SaaS, multi-user accounts, persistent storage across sessions, social features, payment integration.

---

## 2. Locked product decisions

| Decision | Value | Source |
|---|---|---|
| Product name | **ACE Music Studio** (slug `ace-music-studio`) | brainstorming Q1 |
| Base model | ACE-Step 1.5 XL SFT (4 B DiT + 4 B Qwen3 planner) | research bundle `03_acestep.md` |
| Backend pattern | Direct ACE-Step Python API, single Gradio process | brainstorming Q architecture |
| UI layout | Sidebar nav + form + output (3 columns on desktop) | brainstorming Q layout = B |
| Theme | Brutalist Mono (pure black/white, no accent) | brainstorming Q palette = E |
| Tab set | Generate · Cover · Extend · Edit · Lyrics | brainstorming Q scope = all |
| LoRA capability | Multi-stack via PEFT + bundled presets + custom upload | brainstorming Q scope |
| Lyrics LM | Qwen 2.5 7B Instruct (Apache-2.0, ~14 GB bf16) | brainstorming Q lyrics LLM |
| Hosting | Free ZeroGPU (community grant if needed) | brainstorming Q hosting |
| License | MIT, public GitHub | brainstorming Q license |
| Mobile | Horizontal scroll tabs at top, ≤ 640 px | brainstorming Q responsive = A |
| Authorship rule | Mayank Gupta sole author on every commit | user prior memory `feedback_git_authorship.md` |

---

## 3. Architecture

### 3.1 Top-level shape

```
              ┌──────────────────────────────────────────┐
   browser ─▶ │  app.py — Gradio Blocks                  │
              │  header · sidebar · 5 tabs · CTA footer  │
              └─────────────────┬────────────────────────┘
                                │
                                ▼
              ┌──────────────────────────────────────────┐
              │  backend.py — ACEStepStudioBackend       │
              │  @spaces.GPU(duration=callable)          │
              │  lazy singletons; one mode-dispatch fn   │
              └─────────────────┬────────────────────────┘
                                │
   ┌──────────────┬─────────────┴────────┬─────────────────┐
   ▼              ▼                      ▼                 ▼
ace_pipeline.py  lora_stack.py     lyrics_lm.py     post_process.py
ACEStepPipeline  preset registry   Qwen 2.5 7B      Demucs stems
device/cache     PEFT adapters     MLX or PyTorch    pyloudnorm
                 sniff + validate  lazy load
```

### 3.2 Backend singleton — `ACEStepStudioBackend`

One per-process instance, constructed lazily on first request. Owns three independently-lazy sub-singletons:

| Sub-singleton | Loads when | Holds |
|---|---|---|
| `ACEStepPipeline` instance | first generation request | DiT, Qwen3 planner, audio codec, VAE |
| `LyricsLM` instance | first lyrics-tab request | Qwen 2.5 7B weights, tokenizer |
| `Demucs` instance | first stem-separation request | `htdemucs_ft` weights |

Boot cost: only `_bootstrap()` (cache mirror + symlinks) — ~1–5 s. First gen request: +30–60 s warm-up. First lyrics request: +20–40 s. First stem request: +10 s. All amortised across the session.

### 3.3 Device autodetect (`ace_pipeline.py`)

Priority: **CUDA → MPS → CPU**.

Apple Silicon path:

- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before any torch import (in `app.py` module preamble, before backend imports torch).
- Use the **Apple-Silicon fork's branch of ACE-Step** (`clockworksquirrel/ace-step-apple-silicon`) on Mac — pinned via `requirements-mac.txt` extra. Hybrid MLX (LM planner) + PyTorch MPS (DiT decoder).
- Skip the CUDA-only `torch.mps.mem_get_info` gate — `vram_limit_for("mps")` returns `None` so ACE-Step's free-VRAM check short-circuits.
- bf16 throughout; `--bf16 false` only if a specific kernel falls back.

CUDA path:

- Vanilla `ace-step` from git (or PyPI when published).
- bf16; allow flash-attn if installed.
- `vram_limit_for("cuda")` returns the safe cap from `torch.cuda.mem_get_info`.

CPU path (warning only, not blocked):

- Single warning banner on app load if no GPU detected: "CPU inference: expect ~10× slower."

### 3.4 HF Spaces bootstrap (`app.py:_bootstrap()`)

Direct port of z-image-studio's pattern, with model paths swapped:

1. If `on_spaces()`, mirror the read-only `HF_HOME` (build cache) to `~/hf-cache-rw/`.
2. Repoint `HF_HOME` and `HF_HUB_CACHE` env vars at the writable copy.
3. Set `ACESTEP_MODEL_BASE_PATH` (or whatever the fork's env var is) to a project-local `./models/`.
4. Symlink each cached HF snapshot into `./models/<repo>/` so the pipeline's loader finds them locally.

This avoids re-downloads on every cold container start and works around HF's read-only build cache layer.

### 3.5 ZeroGPU integration

- `@spaces.GPU(duration=…)` decorates `backend.generate(mode, params)` at module load time. The decorator is a no-op identity off Spaces.
- `duration` is a callable that estimates per-call timeout from `(mode, params)`, clamped to `[60, 180] s`:
  - Generate / Cover at default settings → 60 s
  - Long Generate (>120 s output) or Edit → 90–120 s
  - Extend with large repaint window → 120–180 s
  - Lyrics (separate decoration) → 30 s
- On `"GPU task aborted"` exception, auto-retry once at 2× duration. After second failure, return `gr.Warning` with timing diagnostics.
- `requirements.txt` **must not pin `spaces`** (HF injects its own version).

---

## 4. The five modes

All mode handlers live in `modes.py` as pure functions over `(backend, params) → (audio_path, meta_dict)`. They share the **LoRA stack** and **advanced opts** code paths via shared helpers.

### 4.1 Generate (text → song)

**Inputs**: `prompt` (style), `lyrics`, `duration_s` (5–240), `instrumental` (bool), `lora_stack`, `advanced`.

**ACE-Step params**: `audio_cover_strength=0`, `repaint_mode=None`, `flow_edit_morph=False`, `cot_*` controlled by advanced "LM thinking" toggle.

**Output**: WAV (44.1 kHz stereo) + metadata JSON.

### 4.2 Cover (audio reference → song in that style)

**Inputs**: `prompt` (new style hint, optional), `ref_audio` file (any of mp3/wav/flac, ≤ 60 s), `lyrics` (new lyrics), `duration_s`, `lora_stack`, `advanced`.

**ACE-Step params**: `audio_cover_strength≈0.93` (configurable in advanced), `cover_noise_strength=0`, `infer_method="ode"`.

**Output**: WAV.

### 4.3 Extend (continue an existing song)

**Inputs**: `seed_audio` (≤ 240 s), `extra_prompt`, `extra_duration_s` (5–120), `lora_stack`, `advanced`.

**ACE-Step params**: `repaint_mode="balanced"`, `repaint_strength` configurable, `repainting_start` set to the seed-audio end timestamp, `repainting_end` set to seed-end + `extra_duration_s`. Exact param names + sentinels for "append-after-end" must be verified against the current ACE-Step Python API during M3 implementation — see §14 open question.

**Output**: WAV (seed + extension concatenated).

### 4.4 Edit (repaint / flow morph a segment)

**Inputs**: `source_audio`, `source_lyrics`, `target_lyrics`, `segment_start_s`, `segment_end_s`, `mode` ∈ {`repaint`, `flow_edit`}, `lora_stack`, `advanced`.

**ACE-Step params**:

- repaint sub-mode: `repaint_mode="balanced"`, `repainting_start=segment_start_s`, `repainting_end=segment_end_s`, `repaint_strength=0.5`.
- flow_edit sub-mode: `flow_edit_morph=True`, `flow_edit_source_caption`, `flow_edit_source_lyrics`, `flow_edit_n_min=0.0`, `flow_edit_n_max=1.0`, `flow_edit_n_avg=1`.

**Output**: WAV.

### 4.5 Lyrics (Qwen 2.5 → structured lyrics)

**Inputs**: `brief` (free-text prompt), `target_structure` (e.g., "intro, verse, chorus, verse, chorus, bridge, chorus, outro"), `language`, `tone` (optional).

**System prompt** (locked):

```
You are a songwriter. Output ONLY structured lyrics for an AI music generator. Use these section tags exactly:
[intro] [verse 1] [verse 2] [chorus] [bridge] [outro] (etc.)

Each section is on its own line, followed by the lyrics for that section. Keep verses 4-8 lines, choruses 4 lines, bridges 2-4 lines. Match the requested tone and language. Do not include commentary, headers, or markdown.
```

**Output**: plain text with structural tags. A "Use these in Generate" button populates the Generate tab's `lyrics` field.

### 4.6 Retake button

Every mode's output panel has a "↻ retake" button. It re-runs the same mode handler with a new random seed, all other params unchanged.

---

## 5. LoRA stack (`lora_stack.py`)

### 5.1 Preset registry

`presets/manifest.json`:

```json
[
  {"name":"RapMachine","hf_id":"ACE-Step/ACE-Step-v1-RapMachine-LoRA","kind":"genre"},
  {"name":"Chinese Rap","hf_id":"ACE-Step/ACE-Step-v1-Chinese-Rap-LoRA","kind":"genre"},
  {"name":"Lyric2Vocal","hf_id":"ACE-Step/ACE-Step-v1-Lyric2Vocal-LoRA","kind":"voice"},
  {"name":"Text2Samples","hf_id":"ACE-Step/ACE-Step-v1-Text2Samples-LoRA","kind":"instrumental"}
]
```

Presets are downloaded from HF on first preset-click, cached, and registered as PEFT adapters with the preset name. The four preset chips appear in every song-mode tab.

### 5.2 Custom upload

User drops a `.safetensors` file into the upload zone:

1. `sniff(path)` reads the safetensors header (no full load, just metadata).
2. Verifies key naming matches ACE-Step 1.5 XL DiT (`*.to_q.lora_A.weight`, etc.) and rank ≤ 256, alpha set, file ≤ 500 MB.
3. On success, registers as a new PEFT adapter under `Path(path).stem` as adapter name; appears in the active stack.
4. On failure, raises `LoRAValidationError` → `gr.Error` toast: "This LoRA isn't compatible with ACE-Step 1.5 XL SFT. Expected DiT modules: to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2."

### 5.3 Active stack management

UI shows a list of active LoRAs with per-row strength slider (0.0–1.5) and × button. State held in `gr.State` per tab. On generate:

```python
backend.apply_lora_stack(active_adapters)   # pipe.set_adapters(names, weights=scales)
audio, meta = backend.generate(mode, params)
meta["loras"] = [{"name":n, "scale":s, "sha256":h} for n,s,h in active_adapters]
```

After generation the adapters stay loaded (cheap memory cost) but are deactivated via `pipe.disable_adapters()` if the user clears the stack.

### 5.4 Sole-LoRA edge cases

- All chips off + no upload → `pipe.disable_adapters()` (vanilla SFT XL output).
- One LoRA with scale 0.0 → effectively disabled but still listed (UX: don't surprise the user by silently dropping it).
- Same LoRA loaded twice (user dragged the same file twice) → dedupe by file sha256; UI flash: "already in stack."

---

## 6. Lyrics LM (`lyrics_lm.py`)

### 6.1 Backend selection

| Device | Backend | Weights size |
|---|---|---|
| `mps` (Mac) | `mlx-lm` with quantised Qwen 2.5 7B 4-bit | ~4 GB |
| `cuda` | `transformers` with bf16 | ~14 GB |
| ZeroGPU | `transformers` bf16, sliced into the `@spaces.GPU` lifetime | ~14 GB |

Quantisation on Mac is the practical choice — 4-bit MLX-quant Qwen 2.5 7B runs ~3× faster than full-precision PyTorch MPS and barely affects lyric quality.

### 6.2 Generation

- `max_new_tokens=600`, `temperature=0.85`, `top_p=0.9`, `repetition_penalty=1.1`.
- Stop sequences: `\n\n[end]`, `</lyrics>`.
- Post-process: strip leading/trailing whitespace, normalize section tags to lowercase (e.g., `[Verse 1]` → `[verse 1]`).

### 6.3 Lazy loading

```python
class LyricsLM:
    _instance = None
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls._load()
        return cls._instance
```

First call cost: ~20–40 s on Mac, ~10 s on CUDA. Surfaced to the user via `gr.Progress` on the Lyrics tab.

---

## 7. Post-processing (`post_process.py`)

### 7.1 Stem separation

- `demucs.api.Separator(model="htdemucs_ft")` lazy singleton.
- Output: 4 WAV files (vocals, drums, bass, other).
- Runs synchronously after generation if the user expands the Stems section, or on-demand via a "Separate stems" button in the output panel.
- On ZeroGPU, counted in the same `@spaces.GPU` lifetime as the generation that produced the audio.

### 7.2 Loudness normalization

- `pyloudnorm` normalises to **-14 LUFS** (streaming spec).
- Toggled by an `Advanced` checkbox per mode (default ON).
- Applied to the final WAV before MP3 encoding.

### 7.3 MP3 export

- `ffmpeg` via `subprocess` — 320 kbps CBR, 44.1 kHz, stereo.
- Embeds metadata as ID3 tags (prompt, lora hashes, seed).

---

## 8. Frontend (`app.py` + `ui.py` + `theme.py`)

> **Reference mockups (visual source of truth):**
>
> | File | Covers |
> |---|---|
> | [`mockups/01_generate_mobile_errors.html`](./mockups/01_generate_mobile_errors.html) | Generate tab (fully expanded), mobile phone screens, error / edge-case states |
> | [`mockups/02_cover_extend.html`](./mockups/02_cover_extend.html) | Cover tab + Extend tab (both fully expanded) |
> | [`mockups/03_edit_lyrics.html`](./mockups/03_edit_lyrics.html) | Edit tab (Repaint + Flow Morph sub-modes) + Lyrics tab (Qwen LM params) |
> | [`mockups/README.md`](./mockups/README.md) | What's shared across tabs + what each tab adds |
>
> The mockups define the **layout, spacing, control surface, and disclosure hierarchy.** The prose below defines the **semantics** — what each control does, what the defaults are, what the responsive breakpoints are. If a discrepancy ever shows up, the mockups are the source for layout, and §3–§7 of this spec are the source for behaviour.

### 8.1 Page chrome

```html
HEADER (sticky):
  [brand: "ACE Music Studio." in 15px white, "." in #FFF as period]
  [status: "ready · MPS · M5 Max" in 10px muted]

CTA (below header, separator below):
  Built with ♥.  Drop a like  ·  Follow @techfreakworm  for what's next.

(Tab content)
```

### 8.2 Sidebar (desktop ≥ 1024 px)

5 mode items + History section below. Active item: white left border + brighter text. Width: 170 px.

### 8.3 Tablet (640–1024 px)

Sidebar collapses to 30 px wide **icon rail**. Hover shows tooltip with full label. Same active treatment.

### 8.4 Mobile (< 640 px)

Native `gr.Tabs` (horizontal scroll) replaces the sidebar entirely. Hidden via CSS media query swap: `display: none` on `.ms-sidebar`, `display: flex` on a `.ms-mobile-tabs`. No JS.

### 8.5 Tab body

Two-column on desktop (form 60% / output 40%), stacks vertically on tablet and mobile.

Form layer order (top to bottom, always-visible by default):

1. Style prompt (textarea, ~3 rows)
2. Lyrics (textarea, ~6 rows) — except Lyrics tab, which replaces with brief + structure inputs
3. Mode-specific: ref audio (Cover), seed audio (Extend), source + segment (Edit)
4. Duration slider + vocals/instrumental toggle (Generate only)
5. LoRA section (collapsed by default; chip row visible if any preset is "on")
6. Advanced accordion (collapsed by default)
7. LM-planner accordion (collapsed by default)
8. Generate button (primary; white-on-black; full-width on mobile)

### 8.6 Output panel

- Audio player with built-in waveform (Gradio 5 native)
- Retake button (↻)
- Stems grid (Demucs) — only visible after Demucs runs
- Action row: ↓ mp3 · ↓ wav · `{ }` meta · ↗ share (copies a permalink with prompt+seed in URL params)
- Metadata JSON viewer (collapsible, default closed)

### 8.7 Theme tokens (`theme.py`)

```python
BG = "#0A0A0A"
SURFACE = "#141414"
SURFACE_STRONG = "#000000"
BORDER = "#1F1F1F"
BORDER_STRONG = "#2A2A2A"
INK = "#E5E5E5"
INK_MUTED = "#6B6B6B"
PRIMARY = "#FFFFFF"
ERROR = "#E5E5E5"  # high-contrast white in Brutalist Mono; gradio error background still red-ish but our text is white
RADIUS = "6px"
FONT_STACK = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'
```

CSS injected via `gr.Blocks(css=…)` covers sidebar layout, responsive media queries, LoRA chip pill, waveform tightening, accordion arrow customization, hide-Gradio-footer.

---

## 9. Data flow per generation

```
1. User clicks "Generate" button on the Generate tab.
2. app.py:on_generate(...) handler reads all gr inputs, coerces types.
3. Handler validates active LoRAs (cheap header sniff) — raises gr.Error on failure.
4. Handler calls backend.generate_with_retry(mode="generate", params={...}).
5. backend.generate_with_retry is the @spaces.GPU-decorated entrypoint.
6. Inside the GPU lifetime:
   a. _ensure_pipeline()              — lazy load on first call
   b. _apply_lora_stack(params.loras) — pipe.set_adapters(names, weights)
   c. _dispatch_mode("generate", params) — calls pipe(...) with mode-specific kwargs
   d. _post_process(audio, params)     — loudness norm, optionally stems
   e. _emit_meta(params, audio)        — build metadata JSON, sha256s
7. Returns (audio_path, meta_dict).
8. Handler updates UI: audio player, metadata JSON viewer.
9. History entry appended (in-memory, last 10).
```

ZeroGPU abort handling wraps step 5 in a one-shot retry at 2× duration. Beyond that: `gr.Warning` with the suggestion to reduce duration or steps.

---

## 10. Error handling matrix

| Trigger | User-facing | Logs |
|---|---|---|
| LoRA file invalid (rank, modules, size) | `gr.Error("This LoRA isn't compatible with ACE-Step 1.5 XL SFT. …")` | full traceback to stderr |
| Audio input wrong format | `gr.Error("Audio must be wav/mp3/flac, ≤ 240 s.")` | format diagnostics |
| Cover/Extend/Edit missing required input | `gr.Error("Reference audio is required for Cover mode.")` | param dump |
| ZeroGPU abort | auto-retry once at 2× duration; if still aborts: `gr.Warning("Generation timed out. Try a shorter duration or fewer steps.")` | timing info |
| Lyrics LM cold-load fails (OOM) | `gr.Error("Couldn't load lyrics model. Free some memory and retry.")` | full traceback |
| MPS op not implemented | falls back to CPU via env var; if still crashes: `gr.Error("This ACE-Step op isn't yet supported on Apple Silicon. Generation aborted.")` | op name + diagnostics |
| Demucs separator fails on weird audio | `gr.Warning("Stem separation failed — audio still saved.")` | traceback |
| Custom-LoRA download fails (preset) | `gr.Error("Couldn't download preset 'X'. Check network.")` | network log |
| Out-of-disk on cache mirror | `gr.Error("Disk full. Free space and reload.")` | mount stats |

---

## 11. Testing

### 11.1 Layers

- **L1 — no GPU, no models**: module structure, type signatures, theme CSS asserts, LoRA-header sniff unit tests, metadata JSON shape, preset manifest schema. ~30 tests, runs in < 5 s.
- **L2 — mocked pipeline**: each mode handler calls the backend with the right kwargs; `set_adapters` invoked with correct order/weights; lyrics LM prompt template asserted. ~25 tests, runs in < 30 s.
- **GPU smoke (`@pytest.mark.gpu`, skipped by default)**: one Generate + one Cover + one Extend + one Lyrics at minimum settings, asserts output exists and is non-zero size. ~4 tests, runs in 5–10 min on M5 Max.

### 11.2 CI

- GitHub Actions: Python 3.11, run L1 + L2 with `pytest -m "not gpu"`.
- ruff format + ruff check both pass.
- No GPU testing in CI (cost). The user runs `pytest -m gpu` locally on the M5 Max before each release tag.

### 11.3 Manual verification before merge

- Each new mode handler: at least one end-to-end on M5 Max with a real prompt + the psytrance LoRA loaded.
- LoRA upload: at least one bad-file rejection (rank mismatch) + one good-file success.
- Responsive: open on phone (Safari iOS), verify horizontal tab strip, verify generate end-to-end.

---

## 12. Deployment

### 12.1 HF Spaces

`README.md` frontmatter:

```yaml
---
title: ACE Music Studio
emoji: 🎵
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
python_version: "3.11"
suggested_hardware: zero-a10g
hf_oauth: false
preload_from_hub:
  - ACE-Step/ACE-Step-v1.5-XL-SFT *.safetensors,config.json,scheduler/*,vae/*,tokenizer/*
  - Qwen/Qwen2.5-7B-Instruct *.safetensors,config.json,tokenizer*
  - facebook/htdemucs_ft *.th
  - ACE-Step/ACE-Step-v1-RapMachine-LoRA *.safetensors
  - ACE-Step/ACE-Step-v1-Chinese-Rap-LoRA *.safetensors
  - ACE-Step/ACE-Step-v1-Lyric2Vocal-LoRA *.safetensors
  - ACE-Step/ACE-Step-v1-Text2Samples-LoRA *.safetensors
---
```

Preload size estimate: ACE-Step XL SFT ~16 GB + Qwen 2.5 ~14 GB + htdemucs ~250 MB + 4 LoRAs ~400 MB = **~31 GB**, well under HF's 150 GB cap.

### 12.2 GitHub

- Repo: `techfreakworm/ace-music-studio` (public).
- License: MIT.
- HF Space mirror via dedicated git remote (`git push space main`).
- README badges: HF Space, GitHub stars, MIT license, Python 3.11, backend ACE-Step.

### 12.3 Local install

```bash
git clone https://github.com/techfreakworm/ace-music-studio
cd ace-music-studio
bash setup.sh           # creates .venv (Python 3.11), installs requirements
source .venv/bin/activate
python app.py           # http://127.0.0.1:7860
```

`setup.sh` detects Mac vs CUDA and installs the right ACE-Step branch + Qwen backend (mlx-lm on Mac).

---

## 13. Out of scope for v1

These are deferred to v2+ — do **not** implement without explicit user OK:

- Multi-prompt batch queue (generate 5 variants in a row)
- Persistent generation history across sessions (DB-backed)
- User accounts / auth
- Telemetry dashboard
- Voice cloning ("Persona" feature — RVC integration)
- LoRA training inside the app
- ControlNet-style conditioning (rhythm tracks, MIDI input)
- Spectrogram visualization (waveform is enough for v1)
- Multi-language UI strings (English only; song content can be any language)
- Watermarking output audio
- Browser-side audio editing (cut, paste, fade)
- Multi-tenant rate limiting
- Export to DAW format (stem zip is enough for v1)
- Visual regression tests for the Gradio UI

---

## 14. Open implementation questions (defer to writing-plans)

1. **ACE-Step package — git or PyPI?** As of 2026-05-18, the official `ace-step` PyPI package exists for v1.5 but the Apple-Silicon fork is git-only. Decision: `pip install ace-step` on CUDA, `pip install git+https://github.com/clockworksquirrel/ace-step-apple-silicon` on Mac (detected by `setup.sh`).
2. **Demucs model — `htdemucs` or `htdemucs_ft`?** `htdemucs_ft` is the fine-tuned variant with slightly better separation. Larger weight (~250 MB) but trivial in our budget. Default: `htdemucs_ft`.
3. **LoRA preset HF IDs** — placeholder paths above (`ACE-Step/ACE-Step-v1-*-LoRA`) may not match the exact HF org/repo naming when this is implemented; the plan should verify each preset's actual canonical HF path before the preload directive is finalised.
4. **Qwen 2.5 7B vs 3B for ZeroGPU comfort** — 7B is correct per the brainstorming answer. If ZeroGPU's 60 s budget is too tight for cold-load + generate, fall back to **Qwen 2.5 3B Instruct** (~6 GB) without UI changes.
5. **Edit-mode UX for segment selection** — start with two numeric inputs (start_s, end_s). v1.5 can add a waveform-clickable selector if user feedback demands it.
6. **History persistence** — v1 is in-memory only. The sidebar history list is `gr.State`-backed and wipes on page reload. Persistent history is v2.
7. **ACE-Step Extend / Repaint exact API surface** — the psytrance LoRA generation config shows the relevant kwargs (`repainting_start`, `repainting_end`, `repaint_mode`, `repaint_strength`, `chunk_mask_mode`, `repaint_latent_crossfade_frames`, `repaint_wav_crossfade_sec`). Verify the conventions for "append after end of seed audio" (e.g., does `repainting_end > audio_length` extend, or do we need a different sentinel?) before M3 ships.
8. **MLX-quant Qwen 2.5 7B availability** — confirm `mlx-community/Qwen2.5-7B-Instruct-4bit` exists and produces acceptable lyric quality. If not, use `mlx-community/Qwen2.5-3B-Instruct-4bit` as the Mac path (the model card under §6.1's table moves to 3B-on-Mac, 7B-on-CUDA).

---

## 15. Sole-author rule

Per the user's permanent feedback (memory `feedback_git_authorship.md`):

- Mayank Gupta is sole author on every commit.
- **NO** `Co-Authored-By: Claude…` trailer.
- **NO** `Generated with Claude Code` footer.
- **NO** `--author=…` flag.
- This applies to commits made by any AI assistant working on this repo.

Encoded in `CLAUDE.md`, `AGENTS.md`, and `SKILLS.md` at the top of the repo so every assistant sees it on first read.

---

## 16. Implementation milestones (rough)

(Detailed sequencing belongs in the implementation plan — see `docs/superpowers/plans/`.)

| Milestone | Deliverable | Validates |
|---|---|---|
| M0 — Bootstrap | `app.py:_bootstrap()` + device autodetect + Gradio Blocks skeleton + theme | App boots on M5 Max and on a Space-equivalent CPU env |
| M1 — Generate mode (no LoRA) | `modes.generate` + `ace_pipeline.py` + audio player output | End-to-end "psytrance, 30 s" generation on M5 Max |
| M2 — LoRA stack | `lora_stack.py` + preset chips + custom upload + active stack UI | Psytrance v2 + RapMachine stacked at 0.95 / 0.85 produce visibly different output |
| M3 — Cover, Extend, Edit | Three more handlers + their tab UIs | Each mode produces a non-trivial output |
| M4 — Lyrics LM | `lyrics_lm.py` + Lyrics tab + "use these" flow | Qwen 2.5 emits valid structural-tag lyrics; round-trip into Generate works |
| M5 — Post-processing | Demucs + pyloudnorm + mp3 export | Stems download, normalised output, ID3-tagged MP3 |
| M6 — Responsive + polish | Mobile media queries + tooltips + error UX + history sidebar | Phone Safari renders + generates end-to-end |
| M7 — Deploy | Preload directive + ZeroGPU decorator + retry logic + Space mirror | Public Space serves requests at parity with local |

---

## 17. References

- ACE-Step 1.5 paper: [arXiv 2506.00045](https://arxiv.org/abs/2506.00045)
- ACE-Step 1.5 repo: [github.com/ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)
- Apple Silicon fork: [github.com/clockworksquirrel/ace-step-apple-silicon](https://github.com/clockworksquirrel/ace-step-apple-silicon)
- ACE-Step LoRA family: [ace-step.github.io](https://ace-step.github.io/)
- Qwen 2.5: [huggingface.co/Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- Demucs: [github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)
- z-image-studio (architectural precedent): `~/Projects/llm/z-image-studio/`
- Research bundle: `research/00_executive_summary.md` and siblings
