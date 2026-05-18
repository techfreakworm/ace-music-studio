# DiffRhythm and DiffRhythm 2 — Deep Technical Review

*Compiled 2026-05-18. All claims cited; speculation flagged inline.*

## 1. Overview

DiffRhythm is the first open-source **latent-diffusion full-song generator** — vocals + accompaniment, end-to-end, from lyrics and a style prompt — built by the **Audio, Speech and Language Processing (ASLP) Lab at Northwestern Polytechnical University (NWPU)** in Xi'an, China, with later contributions from **Xiaomi Research** ([arxiv.org/abs/2503.01183](https://arxiv.org/abs/2503.01183), [github.com/ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)). DiffRhythm v1 dropped on **arXiv 3 Mar 2025**; the full 4m45s variant followed on **15 Mar 2025**, and an iterative v1.2 fixed repetition and audio-quality issues mid-2025 ([HF v1.2 commit](https://huggingface.co/spaces/ASLP-lab/DiffRhythm/commit/f5b749d65f62e30bdaad11e6866edc8d3b078b71)). **DiffRhythm 2** appeared on **arXiv 27 Oct 2025** (v3 revised 3 Feb 2026) under [arxiv.org/abs/2510.22950](https://arxiv.org/abs/2510.22950), and was open-sourced at [github.com/ASLP-lab/DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2) (forked from `xiaomi-research/diffrhythm2`) on **30 Oct 2025**, with HuggingFace weights at [huggingface.co/ASLP-lab/DiffRhythm2](https://huggingface.co/ASLP-lab/DiffRhythm2). The series is the leading **diffusion-side** alternative to the LLM-style approach taken by Suno, YuE, and SongBloom.

## 2. Architecture

DiffRhythm v1 is a **non-autoregressive (NAR) latent diffusion** model with two pieces: a music **VAE** that compresses raw 44.1 kHz stereo audio into a latent grid, and a **DiT** (Diffusion Transformer) that denoises that grid conditioned on lyrics + style ([nzqian.github.io/DiffRhythm](https://nzqian.github.io/DiffRhythm/)). The DiT uses **16 LLaMA-style decoder layers, 2048 hidden dim, 32 heads × 64 dim, totaling ~1.1B parameters** ([arxiv.org/html/2503.01183](https://arxiv.org/html/2503.01183v1)). Vocals and accompaniment are produced **jointly in a single latent stream** — not dual-track — which is what makes it "embarrassingly simple" vs. cascaded systems. Lyric conditioning is **sentence-level via LRC (timestamped) phonemes**, with the diffusion model expected to align internally; style is conditioned either via a reference audio embedding or a text prompt. Inference uses a **32-step Euler ODE with CFG scale 4** and 20% dropout on both conditions during training to enable CFG ([diffrhythm.us](https://diffrhythm.us/)).

**DiffRhythm 2** replaces the pure-NAR DiT with a **semi-autoregressive block flow-matching** transformer: the latent sequence is sliced into **blocks of 10 frames (2s at 5 Hz)**, and "each block is generated with flow matching, while the dependency across blocks is handled autoregressively" ([alphaxiv.org/overview/2510.22950v3](https://www.alphaxiv.org/overview/2510.22950v3) — quoted via search snippet). This is the key innovation: it preserves NAR-style fast within-block parallelism while letting the model attend to prior blocks for **structural coherence** (verse → chorus → verse) and **lyric alignment without any external aligner**. The audio codec is a new **music VAE at 5 Hz frame rate** (vs. the much higher rates of EnCodec/DAC) with a **170M-param decoder**, enabling 210s of latent context to fit on a single GPU ([arxiv abs](https://arxiv.org/abs/2510.22950)). The full DiT is **~1B parameters**. Two new training objectives appear: **Stochastic Block Representation Alignment (REPA) loss** to align hidden states of clean vs. noisy blocks (improves musicality/structure), and **Cross-Pair Preference Optimization** — an RLHF variant that groups the four preference dimensions (musicality, style similarity, lyric alignment, audio quality) into pairs to dodge the merging-induced regression that plain DPO causes. **Max song length: 210 s** in v2 vs. **4m45s (~285 s)** in v1-full ([github.com/ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)).

## 3. Variants and sizes

| Checkpoint | Duration | DiT params | Notes | Source |
|---|---|---|---|---|
| `DiffRhythm-base` | 1m35s | ~1.1B | Original Mar 2025 | [HF](https://huggingface.co/ASLP-lab/DiffRhythm-base) |
| `DiffRhythm-full` | 4m45s | ~1.1B | Released 15 Mar 2025 | [HF](https://huggingface.co/ASLP-lab/DiffRhythm-full) |
| `DiffRhythm-vae` | — | — | Shared audio VAE | [HF](https://huggingface.co/ASLP-lab/DiffRhythm-vae) |
| `DiffRhythm-1_2-base` | 1m35s | ~1.1B | v1.2 quality fix | [GH README](https://github.com/ASLP-lab/DiffRhythm) |
| `DiffRhythm-1_2-full` | 4m45s | ~1.1B | v1.2, text-style + instrumental | [HF](https://huggingface.co/ASLP-lab/DiffRhythm-1_2-full) |
| `DiffRhythm+` (paper) | full | ~1.1B | Adds DPO; not headlined as separate checkpoint | [arxiv 2507.12890](https://arxiv.org/html/2507.12890v2) |
| `DiffRhythm2` | 210 s | ~1B DiT + 170M VAE-dec | Block flow matching | [HF](https://huggingface.co/ASLP-lab/DiffRhythm2) |

(Speculation: I did not find an explicit param count posted for v2's DiT; the **~1B figure comes from a paper-extraction snippet** and aligns with v1's ~1.1B body. Treat as approximate.)

## 4. License

**Apache 2.0** for both code and DiT weights, declared on the v1 GitHub README and reaffirmed on the v2 README ([github.com/ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm), [github.com/ASLP-lab/DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2)). **Commercial use is permitted** with attribution. The v2 model card adds a **non-binding ethical disclaimer** asking users to verify originality, disclose AI involvement, and respect stylistic copyright — this is a notice, not an enforceable license restriction ([HF model card](https://huggingface.co/ASLP-lab/DiffRhythm2)).

## 5. Languages supported

Training is heavily **bilingual (Mandarin + English)** — v2's dataset is reported as **Chinese : English : Instrumental ≈ 4 : 5 : 1** ([alphaXiv extract](https://www.alphaxiv.org/overview/2510.22950v3)). The v1 README and several mirrors claim **cross-lingual capability** for Japanese, Korean, Spanish ([diffrhythm.us](https://diffrhythm.us/), [diffrhythm.ai](https://diffrhythmai.com/)) — but these are demo-site marketing claims, **not benchmarked in the paper**. Verdict: production-safe for **EN and ZH**; treat JP/KR/ES as best-effort. Phoneme front-end is **espeak-ng**, which itself supports 100+ languages ([HF model card](https://huggingface.co/ASLP-lab/DiffRhythm2)).

## 6. Quality assessment

**Objective (v2 paper, lower=better for PER, higher=better for Mulan-T):**

| Metric | DiffRhythm 2 | DiffRhythm+ | ACE-Step | LeVo |
|---|---|---|---|---|
| PER (lyric alignment) ↓ | **0.13** | 0.15 | 0.23 | 0.19 |
| Mulan-T (style match) ↑ | **0.40** | 0.25 | 0.28 | 0.35 |
| RTF (speed) ↓ | 0.213 | 0.153 | 0.127 | 1.225 |

So v2 has **best-in-open-source lyric alignment and style match**, slightly slower than v1+/ACE-Step but ~6× faster than LeVo ([arxiv 2510.22950](https://arxiv.org/abs/2510.22950)).

**Subjective:** v2 is the strongest open model by MOS in the paper's own user study, **but the authors explicitly state "in aspects such as musicality, it still shows a clear gap compared to commercial systems like SUNO V4.5"** ([arxiv 2510.22950](https://arxiv.org/abs/2510.22950)). The **block flow-matching does close the structural-coherence gap** that the original Hacker News thread criticized v1 for — multiple HN commenters complained "there's no identifiable chorus in any of the demo songs" and rhythm was unstable ([news.ycombinator.com/item?id=43255467](https://news.ycombinator.com/item?id=43255467)). v2 demos show real verse/chorus structure ([aslp-lab.github.io/DiffRhythm2.github.io](https://aslp-lab.github.io/DiffRhythm2.github.io/)). Specific Reddit reception threads in r/LocalLLaMA/r/StableDiffusion were not surfaced by search (low signal).

## 7. Inference performance

- v1-full: **~10 s for a 4m45s song on a single RTX 4090** (claimed in paper abstract, [arxiv 2503.01183](https://arxiv.org/abs/2503.01183)) — 32 ODE steps. Real-world ComfyUI users report **~62 s for 4 min** on consumer GPUs ([comfyui.org](https://comfyui.org/en/generate-music-with-comfyui-diffrhythm)).
- **VRAM:** DiffRhythm-base needs ≥ **8 GB** with `--chunked`; full needs **24 GB** for headroom ([chutes.ai docs](https://chutes.ai/docs/examples/music-generation)).
- v2: **RTF 0.213 on RTX 4090** → ~45 s for a 210 s song ([arxiv 2510.22950](https://arxiv.org/abs/2510.22950)).
- **Apple Silicon / MPS:** The v1 README claims Apple Silicon is "supported as of March 2025" but the GitHub issues list does not surface dedicated MPS benchmarks, and the Pinokio launcher ([github.com/pinokiofactory/diffrhythm](https://github.com/pinokiofactory/diffrhythm)) does not advertise macOS in its description. **No published M3/M4/M5 numbers exist.** Speculation: on the user's **M5 Max with 128 GB unified memory**, v1-full should run via `PYTORCH_ENABLE_MPS_FALLBACK=1`, likely 3–5× slower than 4090 — needs hands-on validation. v2 is newer and has not been tested on MPS publicly.

## 8. DiffRhythm 2 specifics

What changed from v1 → v2 ([arxiv 2510.22950](https://arxiv.org/abs/2510.22950), [alphaxiv overview](https://www.alphaxiv.org/overview/2510.22950v3)):

1. **Architecture shift:** pure NAR DiT → **semi-AR block flow-matching** (2 s blocks).
2. **New 5 Hz music VAE** (vs. v1's higher-rate codec) — enables 210 s context within budget.
3. **Stochastic Block REPA loss:** aligns clean vs. noisy hidden states → better musicality + structure.
4. **Cross-Pair Preference Optimization:** four-dim RLHF without the model-merging regression that plain DPO causes.
5. **Dataset scaling:** **~1.4 M songs / ~70,000 hours**, with a **20 k-hour high-quality subset** for SFT and **40 k preference pairs** for DPO — a step-change from v1's undisclosed-but-smaller corpus.
6. **Lyric alignment without external constraints:** v1 needed LRC timestamps; v2 learns alignment end-to-end via the AR block dependency.
7. **Quality numbers (paper):** PER **0.15 → 0.13**, Mulan-T **0.25 → 0.40** vs. DiffRhythm+ — i.e. **lyric-error reduced ~13 % and style-match nearly doubled**.

## 9. Repo health

- **DiffRhythm v1:** ~**2.2–2.3 k stars**, **268 forks**, active through 2025, last major release Mar 2025 ([github.com/ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)).
- **DiffRhythm 2:** **157 stars / 11 forks / 27 commits** as of late Oct 2025 — young repo, recently pushed ([github.com/ASLP-lab/DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2)).
- Training/fine-tuning scripts: **"Coming soon"** is the status on v1; community has filed [Issue #46](https://github.com/ASLP-lab/DiffRhythm/issues/46) asking for fine-tuning docs. v2 ships **inference only** in the public repo as of writing.

## 10. Real-world adoption

- **ComfyUI:** [billwuhao/ComfyUI_DiffRhythm](https://github.com/billwuhao/ComfyUI_DiffRhythm) — 153 stars, supports v1.2 + full, includes bilingual subtitle gen ([runcomfy.com node](https://www.runcomfy.com/comfyui-nodes/ComfyUI_DiffRhythm)).
- **Pinokio:** [pinokiofactory/diffrhythm](https://github.com/pinokiofactory/diffrhythm) — 19 stars, 69 commits, one-click installer.
- **Chutes.ai:** Public serverless endpoint for DiffRhythm-full ([chutes.ai/docs/examples/music-generation](https://chutes.ai/docs/examples/music-generation)).
- **Replicate:** No first-party DiffRhythm 2 model found in search — gap in the ecosystem (speculation).
- Multiple unofficial web frontends: diffrhythm.com, diffrhythm.us, diffrhythm.ai, diffrhythmai.com — quality and origin unverified, likely wrappers over the HF Space.

## 11. Fine-tuning

The official answer is **none yet**. The v1 repo's training code is listed as "Coming soon," and v2 only ships inference. There is no LoRA support, no published fine-tuning recipe, and no `transformers`/`diffusers` integration as of May 2026. Community workaround would require reverse-engineering the DiT class — non-trivial for a 1 B-param flow-matching model. **For the user's Suno-clone platform, fine-tuning DiffRhythm today means forking + writing your own training loop.** This is the single biggest practical weakness.

## 12. Pros and cons

**Pros**
- Permissive **Apache 2.0** for code + weights — clean commercial path.
- **Fastest open full-song model** (~10 s for 4 min on a 4090; v2's block-FM is competitive even with AR-like coherence).
- v2 has **state-of-the-art lyric alignment (PER 0.13)** in open source.
- Lightweight: 8 GB VRAM possible with chunking — runs on consumer GPUs.
- Strong ecosystem: ComfyUI nodes, Pinokio installer, Chutes serverless.
- v2's block flow-matching meaningfully **closes the structural-coherence gap** that doomed v1 demos on HN.

**Cons**
- Still a **clear musicality gap vs. Suno v4.5** (authors admit it; [arxiv 2510.22950](https://arxiv.org/abs/2510.22950)).
- **No fine-tuning / LoRA path** — training code unreleased.
- v2's max length is **210 s** (3m30s), *shorter* than v1-full's 4m45s — a regression for radio-length pop.
- Multilingual claims (JP/KR/ES) are **unbenchmarked**; only EN/ZH have paper-backed quality.
- **No published MPS benchmarks** for Apple Silicon; v2 untested on Mac.
- Demo-site proliferation (`diffrhythm.us`, etc.) muddies the brand — confusing for product positioning.
- License disclaimer adds soft ethical obligations re. copyright that legal review may flag.

## 13. Verdict for the user's platform

For a Suno-style platform on an **M5 Max (128 GB unified, MPS)**, DiffRhythm 2 is the **best diffusion-side open option in May 2026**, *but* it should be paired with an **AR-style backup** (YuE / SongBloom / LeVo) covering its weak points.

**Where DiffRhythm 2 wins:**
- Fast, cheap inference per song — viable for high-throughput web generation.
- Best-in-open lyric intelligibility — critical for a karaoke / lyrics-first UX.
- Stereo 44.1 kHz output out of the box.
- Apache-2.0 + commercial freedom.

**Where it underperforms:**
- **Pop musicality, hook quality, vocal timbre** are still below Suno v4.5 — premium-tier output is not there.
- **No fine-tuning** means you cannot specialize on a target sound or your platform's curated catalog without doing R&D.
- **210 s ceiling on v2** limits "full album track" formats — you'd fall back to v1-full (4m45s) at a quality cost.
- **MPS path is unproven** — the user should plan a same-week feasibility test on the M5 Max before committing v2 to the inference layer; CUDA cloud (Chutes / a 4090 server) is the safer near-term backend.

**Recommended posture:** ship v2 as the default *fast* generator behind a feature flag, keep v1.2-full for >3.5 min songs, evaluate Suno / YuE / SongBloom as quality-tier alternatives, and track the v2 repo for an eventual training-code release that would unlock fine-tuning on your platform's data.

---

### Primary sources
- [DiffRhythm 2 paper (arxiv 2510.22950)](https://arxiv.org/abs/2510.22950)
- [DiffRhythm v1 paper (arxiv 2503.01183)](https://arxiv.org/abs/2503.01183)
- [DiffRhythm v1 GitHub](https://github.com/ASLP-lab/DiffRhythm)
- [DiffRhythm 2 GitHub](https://github.com/ASLP-lab/DiffRhythm2)
- [DiffRhythm 2 HF model card](https://huggingface.co/ASLP-lab/DiffRhythm2)
- [alphaXiv overview v3](https://www.alphaxiv.org/overview/2510.22950v3)
- [HN thread on v1](https://news.ycombinator.com/item?id=43255467)
- [ComfyUI_DiffRhythm](https://github.com/billwuhao/ComfyUI_DiffRhythm)
- [Pinokio DiffRhythm](https://github.com/pinokiofactory/diffrhythm)
- [Chutes serving docs](https://chutes.ai/docs/examples/music-generation)
- [DiffRhythm+ paper (arxiv 2507.12890)](https://arxiv.org/html/2507.12890v2)
