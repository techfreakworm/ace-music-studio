# ACE-Step — Deep Technical Report

*Researched 2026-05-18 for a Suno-like platform build on M5 Max (128 GB unified) / MPS.*

---

## 1. Overview

ACE-Step is a foundation model for music generation jointly built by **ACE Studio** (the consumer music-tech outfit behind ACE Studio's vocal synth) and **StepFun** ("Step-AI"), a Beijing-based foundation-model lab. Core authors: Junmin Gong, Sean Zhao, Sen Wang, Shengyuan Xu, Joe Guo ([ace-step.github.io](https://ace-step.github.io/)).

Release timeline:
- **v1 (3.5B)** — open-sourced May 2025; technical report posted on arXiv on 2 Jun 2025 as 2506.00045 ([arxiv.org/abs/2506.00045](https://arxiv.org/abs/2506.00045)).
- **v1.5** — released **28 Jan 2026** as a separate repo, [`ace-step/ACE-Step-1.5`](https://github.com/ace-step/ACE-Step-1.5). Adds a hybrid Language-Model + Diffusion-Transformer planner.
- **XL series (4B DiT decoder)** — released 2 Apr 2026 as a higher-quality variant inside the v1.5 family.
- **Latest tag** — v0.1.7 on 24 Apr 2026 ([ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)).
- **v2** — **no public roadmap or announcement** as of 18 May 2026.

Current status: actively maintained, 10.4k stars on the v1.5 repo and 4.5k on the original v1 repo, with a thriving ComfyUI ecosystem and third-party UIs ([ace-step/ACE-Step](https://github.com/ace-step/ACE-Step), [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)).

---

## 2. Architecture

**v1 (3.5B):** a hybrid that fuses three pieces (per the paper, [arxiv.org/abs/2506.00045](https://arxiv.org/abs/2506.00045)):
1. **Sana Deep Compression AutoEncoder (DCAE)** — high-compression audio latent space borrowed from NVIDIA's Sana image work.
2. **Lightweight linear transformer** — the diffusion backbone, deliberately linear-attention to keep RTF low.
3. **Diffusion training** with **MERT + m-HuBERT** providing semantic-alignment supervision (REPA-style) during training so latents stay musically coherent.

This sits between LLM-token approaches (Suno/YuE, slow but lyric-tight) and pure diffusion (DiffRhythm, fast but structurally weak). The design goal stated in the paper is "a fast, general-purpose, efficient yet flexible architecture" — explicitly a *foundation model*, not just a text-to-song pipeline ([arxiv.org/abs/2506.00045](https://arxiv.org/abs/2506.00045)).

**v1.5:** a hybrid **LM-as-planner + Diffusion-Transformer (DiT)**. A small Qwen3-based LM (0.6B / 1.7B / 4B) turns the user prompt into a structured "song blueprint" (sections, key, bpm, lyrics, vocal style) which the DiT (2B standard or 4B XL) decodes into audio. This brings chain-of-thought reasoning to music structure, lifting long-range coherence — Suno's main historic advantage ([ACE-Step-1.5 README](https://github.com/ace-step/ACE-Step-1.5)).

**Parameter counts:**
| Variant | DiT | LM planner | Total |
|---|---|---|---|
| v1-3.5B | 3.5B (DiT only) | — | 3.5B |
| v1.5 standard | 2B | 0.6B / 1.7B | ~2.6 – 3.7B |
| v1.5 XL | 4B | up to 4B | up to 8B |

---

## 3. Variants and checkpoints

All on Hugging Face under the `ACE-Step/` org ([ACE-Step org on HF](https://huggingface.co/ACE-Step)):
- `ACE-Step-v1-3.5B` — the original generalist model ([HF card](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)).
- `ACE-Step-v1-chinese-rap-LoRA` ("RapMachine") — genre-specific LoRA.
- **LoRA family** shipped by the team: `RapMachine`, `Lyric2Vocal` (vocal-only stem from lyrics), `Text2Samples` (instrumental loops/samples) ([ace-step.github.io](https://ace-step.github.io/)).
- **v1.5 DiT checkpoints:** 2B standard and 4B XL.
- **v1.5 LM planners:** 0.6B, 1.7B, 4B.
- A public **Space demo** at [huggingface.co/spaces/ACE-Step/ACE-Step](https://huggingface.co/spaces/ACE-Step/ACE-Step).

No v2 checkpoint exists yet.

---

## 4. License

**Apache 2.0** for v1 ([ace-step/ACE-Step](https://github.com/ace-step/ACE-Step)) and **MIT** for v1.5 ([ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)). Both are unambiguously **commercial-use-permitted, royalty-free**. This is the single biggest licensing advantage over Suno/Udio and even over YuE (which carries non-commercial clauses in parts of its weights chain).

---

## 5. Vocal support — CRITICAL VERIFICATION

**Verdict: YES — ACE-Step generates vocals natively. The "instrumental-only" claim circulating in some reviews is wrong (likely conflating it with `Text2Samples` LoRA or with DiffRhythm).**

Evidence:
- The **v1 HF model card** describes the model as full-song (vocals + instruments) with the explicit caveat: *"Coarse vocal synthesis lacking nuance"* and *"Rare instruments may not render perfectly"* ([HF card](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)).
- The paper claims **lyric alignment across melody/harmony/rhythm metrics** — only meaningful for sung vocals ([arxiv.org/abs/2506.00045](https://arxiv.org/abs/2506.00045)).
- The ComfyUI native node `TextEncodeAceStepAudio` accepts lyrics with `[verse] [chorus] [bridge]` structural tags ([comfyui-wiki guide](https://comfyui-wiki.com/en/tutorial/advanced/audio/ace-step/ace-step-v1)).
- `Lyric2Vocal` LoRA exists *because* the base model already does vocals — the LoRA isolates the vocal stem ([ace-step.github.io](https://ace-step.github.io/)).
- Blind-listening review of 50 participants scored ACE-Step v1.5 **4.4/5 on SongEval Vocal vs Suno v4 at 4.1/5** ([fm9.ai/ace-step/vs-suno](https://fm9.ai/ace-step/vs-suno)).

**Quality reality check:** v1 vocals are admitted to be "coarse"; v1.5 markedly improves vocal clarity and now beats Suno v4 in blind tests on naturalness for folk/classical/jazz, while Suno still wins on "radio-ready polish" for pop/EDM ([fm9.ai/ace-step/vs-suno](https://fm9.ai/ace-step/vs-suno)).

---

## 6. Languages supported

- **v1:** 19 languages, with the top 10 (English, Mandarin Chinese, Russian, Spanish, Japanese, German, French, Portuguese, Italian, Korean) performing best ([ace-step/ACE-Step](https://github.com/ace-step/ACE-Step)). Less-represented languages underperform due to training-data imbalance.
- **v1.5:** Expanded to **50+ languages** with lyric control, alongside the planner LM ([ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)).

Known weakness from the team itself: Chinese rap was historically weak, motivating the `chinese-rap-LoRA` ([ace-step.github.io](https://ace-step.github.io/)).

---

## 7. Speed claims — verified

The famous claim: *"synthesizes up to 4 minutes of music in just 20 seconds on an A100 GPU — 15× faster than LLM-based baselines"* ([arxiv.org/abs/2506.00045](https://arxiv.org/abs/2506.00045), [ace-step.github.io](https://ace-step.github.io/)). Hardware: **NVIDIA A100 80GB**.

Published RTF table from the v1 HF card ([HF card](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)):

| Device | 27 steps RTF | 60 steps RTF |
|---|---|---|
| RTX 4090 | 34.48× | 15.63× |
| A100 | 27.27× | 12.27× |
| RTX 3090 | 12.76× | 6.48× |
| **M2 Max** | **2.27×** | **1.03×** |

v1.5 is faster still: *"under 2 seconds per full song on A100 and under 10 seconds on an RTX 3090"* ([ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)).

**Apple-Silicon equivalents** (from the dedicated [clockworksquirrel/ace-step-apple-silicon](https://github.com/clockworksquirrel/ace-step-apple-silicon) port):

| Task | M1 Pro 16 GB | M3 Pro 36 GB | A100 |
|---|---|---|---|
| 30 s turbo | ~45 s | ~25 s | ~2 s |
| 30 s SFT (full) | ~3 min | ~1.5 min | ~8 s |

**M5 Max projection:** The M5 Max's GPU TFLOPS lineage (MPS SGEMM scaled M1→M4: 1.36 → 2.24 → 2.47 → 2.9 TFLOPS, per [arxiv 2502.05317](https://arxiv.org/html/2502.05317v1)) plus the M5 generation's ~30 % uplift suggests roughly **3.5–4× the throughput of M2 Max**, i.e. an **estimated 8–10× RTF at 27 steps** for v1, and full-song generation in **~30–50 s for a 4-minute song**. No M5-specific public benchmark exists yet.

---

## 8. Quality assessment

From the cross-model evaluation summarised in research-aggregator coverage ([researchgate paper page](https://www.researchgate.net/publication/392334894_ACE-Step_A_Step_Towards_Music_Generation_Foundation_Model), [fm9.ai/ace-step/vs-suno](https://fm9.ai/ace-step/vs-suno)):

| Dimension | Leader | Where ACE-Step sits |
|---|---|---|
| Aesthetic quality | Hailuo > DiffRhythm | mid-upper |
| Musicality (coherence) | Suno v3 | competitive, strong on memorability/clarity |
| Style alignment | Udio v1 > Hailuo | 3rd |
| Lyric alignment | Hailuo | strong, beats Suno v3, Udio, YuE |
| **Vocal naturalness (v1.5)** | **ACE-Step 4.4/5** | beats Suno v4 (4.1/5) |
| Speed (RTF) | **ACE-Step 15.63×** | best in class; DiffRhythm 10.03×, YuE 0.083× |

User-facing reception is positive on customisability and speed; the most-cited weakness is "gacha"-style seed sensitivity — re-rolls produce noticeably different outputs ([ace-step.github.io](https://ace-step.github.io/)).

---

## 9. Inference performance & Apple Silicon

- **VRAM (v1):** minimum **8 GB with CPU offload**; comfortable on 12 GB+ ([ace-step/ACE-Step](https://github.com/ace-step/ACE-Step)).
- **VRAM (v1.5):** **<4 GB** for 2B-turbo with offload; **≥12 GB** for XL with offload; **≥20 GB** without offload; **≥24 GB optimal** ([ACE-Step-1.5 README](https://github.com/ace-step/ACE-Step-1.5)).
- **MPS support:** **first-class.** Use `--bf16 false` on M-series to avoid kernel issues ([ace-step/ACE-Step](https://github.com/ace-step/ACE-Step)). The dedicated [clockworksquirrel/ace-step-apple-silicon](https://github.com/clockworksquirrel/ace-step-apple-silicon) fork adds: bfloat16 throughout, MPS-safe pipeline with `torch.mps.empty_cache()` synchronisation, **MLX backend (567 LoC)** that auto-converts the Qwen3 planner LM to MLX with quantisation, and **LoRA training on MPS**.
- **ComfyUI:** **native nodes** ship in upstream ComfyUI (`TextEncodeAceStepAudio` etc.) plus the official [`ace-step/ACE-Step-ComfyUI`](https://github.com/ace-step/ACE-Step-ComfyUI). v1.5 has dedicated workflows (split-LLM and AIO checkpoint variants) on comfy.org ([Purz blog post](https://blog.comfy.org/p/ace-step-15-is-now-available-in-comfyui)).
- **128 GB unified on M5 Max** comfortably fits the full XL stack plus the 4B planner LM with no offload needed; user's hardware is essentially overkill for ACE-Step.

---

## 10. Repo health

| Repo | Stars | Forks | Last release |
|---|---|---|---|
| `ace-step/ACE-Step` (v1) | 4.5k | 568 | quiet since v1.5 fork |
| `ace-step/ACE-Step-1.5` | **10.4k** | 1.3k | v0.1.7 on 24 Apr 2026 |
| `fspecii/ace-step-ui` (popular community UI) | 3.8k | 561 | active |
| `clockworksquirrel/ace-step-apple-silicon` | — (smaller) | — | active |

The team also curates [`ace-step/awesome-ace-step`](https://github.com/ace-step/awesome-ace-step). Issue activity, ComfyUI integration cadence, and the LM-planner architectural jump in v1.5 all indicate a project that is healthier and growing faster than YuE or DiffRhythm.

---

## 11. Real-world adoption

- **AMD vendor-backed deployment:** AMD published a blog *"Commercial-grade AI music generation on AMD Ryzen AI processors and Radeon graphics with ACE Step 1.5"* in 2026, explicitly endorsing it for Ryzen AI / Radeon production stacks ([AMD blog](https://www.amd.com/en/blogs/2026/commercial-grade-ai-music-generation-on-amd-ryzen-ai-and-radeon-ace-step-1-5.html)).
- **Third-party SaaS:** `acestep.io` and `ace-step.app` run hosted song-generation services on the open weights ([acestep.io](https://acestep.io/), [ace-step.app](https://ace-step.app/)).
- **Production-grade UI:** `fspecii/ace-step-ui` brands itself as *"the Ultimate Open Source Suno Alternative"* with stem extraction (Demucs), batch generation, library/playlist management, LAN access ([fspecii/ace-step-ui](https://github.com/fspecii/ace-step-ui)).
- Heart-MuLa and similar music platforms cite ACE-Step 1.5 in their stack comparisons ([heart-mula.com/ace-step](https://heart-mula.com/ace-step)).

---

## 12. Fine-tuning + LoRA

- **Training code released**; documented in [`TRAIN_INSTRUCTION.md`](https://github.com/ace-step/ACE-Step) and `ZH_RAP_LORA.md` ([ace-step/ACE-Step](https://github.com/ace-step/ACE-Step)).
- **Genre / task LoRAs from the team:** `RapMachine` (general rap), `Chinese-Rap-LoRA`, `Lyric2Vocal`, `Text2Samples` ([HF org](https://huggingface.co/ACE-Step), [ace-step.github.io](https://ace-step.github.io/)).
- v1.5 quotes **"8 songs trainable in ~1 hour on a single RTX 3090"** for LoRA personalisation ([ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)).
- LoRA training is verified working on **MPS** via the Apple-Silicon fork ([clockworksquirrel/ace-step-apple-silicon](https://github.com/clockworksquirrel/ace-step-apple-silicon)).

---

## 13. Pros and cons

**Pros**
- Apache-2.0 / MIT — **fully commercial-friendly**, unique in this tier.
- **Fastest open music model**: 15.63× RTF on a 4090; sub-2 s/song on A100 (v1.5).
- Vocals **and** instruments natively; v1.5 vocal quality now beats Suno v4 in blind tests.
- 50+ languages with lyric structural tags.
- First-class **MPS + MLX** support and a dedicated Apple-Silicon fork.
- ComfyUI native + thriving UI ecosystem (`ace-step-ui`).
- LoRA training is cheap (~1 hour for 8 songs on 3090), well-documented.
- Hybrid LM-planner (v1.5) closes the long-range structure gap with Suno.

**Cons**
- v1 vocals are admitted "coarse"; even v1.5 trails Suno on pop/EDM polish.
- High **seed sensitivity** → "gacha" outputs; multiple re-rolls needed in production.
- Less-represented languages underperform.
- Memory for XL series can exceed 24 GB without offload.
- No official **v2** announced; the rapid v1 → v1.5 → XL fork hints at API/checkpoint churn.
- Smaller benchmark literature than Suno/YuE; some metrics still self-reported.

---

## 14. Verdict for the user's platform

For a **Suno-like platform on M5 Max with 128 GB unified memory**, ACE-Step is currently the **single strongest open-source choice** and should be the **default base model**:

- **Best for:** full-song generation with vocals in 50+ languages, fast iteration (sub-minute per song expected on M5 Max), genre-specific LoRA fine-tuning, and any deployment where commercial rights matter (Apache/MIT vs Suno's locked-down terms).
- **Recommended stack:** ACE-Step **v1.5 XL (4B DiT) + 1.7B Qwen3 planner**, run via the `clockworksquirrel/ace-step-apple-silicon` MPS/MLX fork, served behind the `fspecii/ace-step-ui` frontend, with ComfyUI workflows for power-user editing.
- **Weaknesses to mitigate:** budget for **n-of-k re-roll selection** in the product UX (the gacha problem); pair with a **Demucs stem-extraction post-process** (already in `ace-step-ui`) so users can mix-down; do not pitch the platform on pop/EDM polish alone — lean into folk/classical/jazz and rap, where ACE-Step now leads.
- **Where you may still need Suno-style commercial APIs:** clients demanding broadcast-radio pop polish; otherwise, ACE-Step is sufficient.

---

### Sources

- [ACE-Step paper, arXiv 2506.00045](https://arxiv.org/abs/2506.00045)
- [ace-step.github.io](https://ace-step.github.io/)
- [ace-step/ACE-Step (v1 repo)](https://github.com/ace-step/ACE-Step)
- [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)
- [ACE-Step v1-3.5B model card](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)
- [ACE-Step org on Hugging Face](https://huggingface.co/ACE-Step)
- [clockworksquirrel/ace-step-apple-silicon](https://github.com/clockworksquirrel/ace-step-apple-silicon)
- [fspecii/ace-step-ui](https://github.com/fspecii/ace-step-ui)
- [ace-step/ACE-Step-ComfyUI](https://github.com/ace-step/ACE-Step-ComfyUI)
- [ace-step/awesome-ace-step](https://github.com/ace-step/awesome-ace-step)
- [ComfyUI native ACE-Step tutorial](https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1)
- [ComfyUI Wiki ACE-Step guide](https://comfyui-wiki.com/en/tutorial/advanced/audio/ace-step/ace-step-v1)
- [Purz blog – ACE-Step 1.5 in ComfyUI](https://blog.comfy.org/p/ace-step-15-is-now-available-in-comfyui)
- [AMD blog – ACE-Step 1.5 on Ryzen AI / Radeon](https://www.amd.com/en/blogs/2026/commercial-grade-ai-music-generation-on-amd-ryzen-ai-and-radeon-ace-step-1-5.html)
- [FM9 – ACE-Step vs Suno blind test](https://fm9.ai/ace-step/vs-suno)
- [HeartMuLa – ACE-Step 1.5 review](https://heart-mula.com/ace-step)
- [ResearchGate – ACE-Step paper page](https://www.researchgate.net/publication/392334894_ACE-Step_A_Step_Towards_Music_Generation_Foundation_Model)
- [Apple Silicon HPC benchmark, arXiv 2502.05317](https://arxiv.org/html/2502.05317v1)
- [acestep.io – hosted service](https://acestep.io/)
- [ace-step.app – hosted service](https://ace-step.app/)
