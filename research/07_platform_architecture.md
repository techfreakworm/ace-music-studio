# Suno-Clone Platform Architecture — Build Plan

*Compiled 2026-05-18. Target hardware: Apple M5 Max, 128 GB unified memory. Core model decision: ACE-Step 1.5 XL.*

---

## Mental model

Suno (and Udio) are not just a song-generation model. They are a **product stack** with at least five distinct AI components and a few non-AI scaffolds. If we want to replicate the product experience, we have to plan for all of them. The song-gen model is the headline; everything else is what makes it usable.

```
                ┌─────────────────────────────────────┐
                │           Web / mobile UI           │
                │  (text prompt + style + lyrics)     │
                └─────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────┐
│                    Orchestrator API                       │
│   - prompt routing, queue, billing, history, sharing      │
└──────────────────────────────────────────────────────────┘
                  │            │            │            │
                  ▼            ▼            ▼            ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │  Lyrics LLM │ │  Style/Tag  │ │  Song-gen   │ │  Voice       │
        │  (Llama 3.3 │ │  rewriter   │ │  router     │ │  cloning     │
        │   or Qwen)  │ │  (small LM) │ │             │ │  (RVC)       │
        └─────────────┘ └─────────────┘ └──────┬──────┘ └─────────────┘
                                               │
                                               ▼
                            ┌─────────────────────────────────┐
                            │  Model pool (the actual research)│
                            │   - ACE-Step 1.5 XL (default)   │
                            │   - HeartMuLa-MLX (A/B)         │
                            │   - DiffRhythm 2 (speed tier)   │
                            │   - YuE on Replicate (intl.)    │
                            └─────────────────────────────────┘
                                               │
                                               ▼
                            ┌─────────────────────────────────┐
                            │   Post-processing pipeline      │
                            │   - Loudness normalization      │
                            │   - Demucs stem separation      │
                            │   - Watermarking (audible+meta) │
                            │   - FFmpeg encoding → m4a/mp3   │
                            └─────────────────────────────────┘
                                               │
                                               ▼
                            ┌─────────────────────────────────┐
                            │   Storage + streaming           │
                            │   - S3 / R2 origin              │
                            │   - HLS for in-browser playback │
                            │   - CDN                         │
                            └─────────────────────────────────┘
```

---

## Component-by-component plan

### 1. Song generation — primary model

- **ACE-Step 1.5 XL** via [`clockworksquirrel/ace-step-apple-silicon`](https://github.com/clockworksquirrel/ace-step-apple-silicon) on M5 Max.
- Hybrid backend: Qwen3 planner on **MLX**, DiT decoder on **PyTorch MPS**, bf16 throughout.
- Why XL over standard 2B: 128 GB unified eats the cost, and the 4 B DiT closes meaningful quality gaps for paying users.

**LoRA fine-tuning path (when needed):**
- Document the platform's target genres → curate ~50–200 song lyric/audio pairs per genre.
- Train a per-genre LoRA on the 3090-class budget (~1 hour per LoRA per [`ace-step-1.5 README`](https://github.com/ace-step/ACE-Step-1.5)).
- Serve via the same inference pipeline with LoRA hot-swap.

**Fallback / A-B candidates:**
- **HeartMuLa-MLX** ([`Acelogic/heartlib-mlx`](https://github.com/Acelogic/heartlib-mlx)) — 2.1× faster than PyTorch MPS, full numerical parity, Apache 2.0.
- **DiffRhythm 2** ([`ASLP-lab/DiffRhythm`](https://github.com/ASLP-lab/DiffRhythm)) — for the speed/instrumental tier (210 s ceiling acceptable for short-form features like background loops).
- **YuE via Replicate** ([`replicate.com/fofr/yue`](https://replicate.com/fofr/yue/api)) — only for EN+Mandarin+Cantonese+JP+KR generations that ACE-Step underperforms; pay-per-second, no local infra cost.

### 2. Lyrics generation — separate LLM

The song-gen model takes **lyrics + style** as input, not raw user prompts. Suno's "song description" flow is actually two stages: prompt → lyrics LLM → lyrics → song model.

- Use any decent open LLM running on the user's M5 Max. Candidates:
  - **Qwen 2.5 Coder 32B / Qwen 3 7B** — good multilingual chops, fast on MPS via Ollama or mlx-lm.
  - **Llama 3.3 70B 4-bit** — premium tier; fits comfortably in 128 GB unified.
  - **GPT-OSS-20B** — Apache 2.0, sturdy English.
- Prompt template should:
  1. Parse user style hint into tags (genre, tempo, mood, instruments).
  2. Output structured lyrics with `[verse]`, `[chorus]`, `[bridge]`, `[outro]` markers — these are **exactly the structural tags ACE-Step's `TextEncodeAceStepAudio` consumes**.
  3. Constrain section count and line count to roughly match the target song duration.

**This LLM is independent of the song-gen model and can be swapped freely.**

### 3. Style / tag normalization

A small classifier or 3 B LM that normalizes user free-text into the controlled-vocabulary tag set the song model was trained on (per genre, BPM bucket, vocal gender, mood). For ACE-Step this maps to its lyric-tag schema; for YuE it maps to `top_200_tags.json`.

Implementation: 1-shot prompt to the lyrics LLM with examples; cache results.

### 4. Voice cloning / personas (optional but Suno-equivalent)

To match Suno's "Personas" feature:
- **RVC v2** (Retrieval-based Voice Conversion) — open source, fast, runs on MPS, well-supported.
- Train a 5-minute reference clip → 10–15 min on M5 Max → speaker embedding.
- Apply to the generated vocal stem (Demucs-extracted) → remix.

ACE-Step's **ICL mode** (in-context learning from a reference clip) and YuE's ICL variants partly cover this too, but RVC gives explicit per-speaker control.

### 5. Stem separation

For Suno's "download stems" feature:
- **Demucs v4 / HTDemucs** — open source, Apache 2.0, runs on MPS, separates into vocals / drums / bass / other.
- Already bundled in [`fspecii/ace-step-ui`](https://github.com/fspecii/ace-step-ui).

### 6. Mastering / loudness normalization

- **pyloudnorm** for LUFS normalization to streaming spec (-14 LUFS Spotify, -16 for AirPods).
- **ffmpeg-normalize** as a CLI wrapper.
- **Optional: TBProAudio mvMeter / Voxengo Span equivalents** via web-audio for UI metering.

### 7. Watermarking + content credentials

This is a **legal must-have** for any 2026 generative-music product (training-data lawsuits against Suno/Udio set the precedent).

- **Inaudible audio watermark**: AudioSeal or SilentCipher — open-source, Meta-built, survives MP3 transcoding.
- **C2PA metadata**: sign the m4a with model name + version + prompt + timestamp via the C2PA SDK.
- **Visible "AI-generated" tag** in UI per the YuE model card's recommendation (and increasingly per platform policy).

### 8. Storage and streaming

- **S3-compatible object store** (R2, Backblaze B2, or self-hosted MinIO on the M5 Max if dev-only).
- **HLS encoding pipeline**: ffmpeg → m3u8 + 4 s segments; serve via NGINX or Cloudflare.
- For local dev, plain m4a + range requests are fine.

### 9. Orchestrator API

- **FastAPI** for the request-handling layer.
- **Redis Streams** or **Hatchet** for the generation queue (songs are 30 s–2 min jobs on M5 Max — non-trivial latency, must be async).
- **PostgreSQL** for users, songs, lyrics, LoRAs, billing.
- **Server-Sent Events** for progress streaming back to the UI ("planner stage", "DiT denoising step 14/27", "mastering...").

### 10. Frontend

- **Next.js 16** + Cache Components for the user dashboard / library.
- **Wavesurfer.js** for waveform display and scrubbing.
- **Tone.js** for any in-browser preview / mixing.
- Auth via Clerk or Auth0 — the user's portfolio revamp may already include this.

---

## Build order (incremental milestones)

| Milestone | Scope | Validates |
|---|---|---|
| **M0 — Spike** | Get ACE-Step 1.5 XL running locally via clockworksquirrel fork; generate one 30 s song end-to-end | Hardware compatibility, RTF on M5 Max |
| **M1 — CLI MVP** | Wrap in a Python CLI: `genmusic --prompt "..." --lyrics "..." --out song.m4a` | Headless generation, mastering chain, file output |
| **M2 — Local UI** | Replace UI with `fspecii/ace-step-ui` initially (fastest path); add Demucs stem download | Browser flow, multi-song library, LAN access |
| **M3 — Lyrics LLM integration** | Plug Qwen 3 / Llama 3.3 as the lyrics generator; produce structured lyrics from a one-line prompt | Suno-equivalent prompt UX |
| **M4 — Multi-model router** | Add HeartMuLa-MLX as alternate; add Replicate YuE as multilingual fallback; user can pick or auto-route | A/B capability, breadth |
| **M5 — LoRA pipeline** | First custom LoRA on a target genre (e.g., user's preferred style); hot-swap at inference | Differentiation vs Suno |
| **M6 — Production wrapper** | FastAPI + Postgres + queue + auth + watermarking + C2PA signing | Real product surface |
| **M7 — Deploy** | Move heavy inference behind a rented A100 endpoint for paid users; keep M5 Max for free tier / personal use | Paid-tier economics |

---

## Open questions for the user before M0

1. **Commercial intent.** Is this a personal portfolio project (research mode → SongGeneration 2 is fair game) or a real SaaS (must stay Apache/MIT)? The license map changes drastically.
2. **Target audience.** Western pop (where Suno still wins polish) vs world music / experimental genres (where ACE-Step / YuE compete fairly)?
3. **Latency target.** Suno generates in ~30 s; users tolerate up to 90 s. ACE-Step on M5 Max hits this; YuE local does not.
4. **Hosting plan.** Local-only for personal use? Or eventually paid tier on rented GPU?
5. **Vocal cloning.** Is Suno-style "Persona" upload a must-have v1 feature, or v2?
6. **Catalog / training data.** Any in-house licensed song catalog for LoRA fine-tuning, or strictly the public-domain model out of the box?

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MPS regression in a future PyTorch release breaks ACE-Step | medium | Pin torch version; keep CPU fallback path. |
| ACE-Step releases v2 with breaking API mid-build | medium | Wrap inference in a thin adapter; abstract model behind a single `Generator.generate()` interface. |
| Vendor PER claims (HeartMuLa, LeVo) overstated → quality disappointment | medium | Run internal blind A/B on 20+ prompts before featuring a model in the UI. |
| Output watermark stripped by transcoding | low | Use AudioSeal which survives MP3; double-stamp with C2PA metadata. |
| Lyrics LLM hallucinates copyrighted hooks | medium | Run a similarity check against an embeddings index of known songs; flag for human review. |
| Training-data IP suit (Suno-style) | low for derivative usage | Use models with documented public-data training (ACE-Step's paper is reasonably transparent); avoid Tencent's non-commercial weights. |
| MPS OOM on long sequences | low (128 GB) | `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`; chunk generation; offload non-active LoRAs. |

---

## Why ACE-Step 1.5 XL is the foundation (not just a model pick)

This is worth saying explicitly. Choosing the base model determines:

1. **Inference budget and unit economics** — ACE-Step is the only model where <2 s/song on A100 makes a paid tier economically obvious.
2. **Mac developer ergonomics** — first-class MPS means the user can iterate on the M5 Max for weeks without renting cloud GPU.
3. **License-clean output ownership** — MIT means users own their songs unambiguously.
4. **Future-proof on multilingual** — 50+ languages out of the box matters if the platform grows beyond an English audience.
5. **LoRA personalization is the differentiator** — fine-tuning support that works on MPS lets the user ship genre-specialist sub-models that Suno can't, because Suno's weights are locked.
6. **Production deployments exist** — AMD vendor-backed, `fspecii/ace-step-ui` running at scale, multiple SaaS already on the open weights. This is not betting on a research artifact.

The compound effect of those six is why ACE-Step is recommended as the platform foundation rather than just "the model to start with."
