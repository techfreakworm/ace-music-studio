# YuE — Open Full-Song Music Generation Foundation Model

*Research date: 2026-05-18*

---

## 1. Overview

**YuE** (乐, "yue" — Chinese for "music") is an open-source family of long-form, lyrics-to-song foundation models that produce vocals + accompaniment end-to-end, explicitly positioned as the open competitor to Suno.ai and Udio. It was built by the **M-A-P (Multimodal Art Projection) collective**, led by researchers at **HKUST (Hong Kong University of Science and Technology)** with collaborators from multiple academic and industry institutions (58 authors are credited on the paper, with hardware support from Geely and Moonshot AI) ([arXiv 2503.08638](https://arxiv.org/abs/2503.08638), [HF model card](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl)).

**Release timeline:**

- **2025-01-26** — Initial YuE-s1-7B series released ([GitHub README](https://github.com/multimodal-art-projection/YuE))
- **2025-01-30** — Apache 2.0 license adopted; dual-track ICL mode added
- **2025-02-07** — Windows / Pinokio support
- **2025-02-17** — Music continuation + Google Colab support
- **2025-03-11/12** — Anneal checkpoints + technical report on arXiv (v1)
- **2025-06-04** — LoRA fine-tuning code merged (PR #126)
- **ICLR 2026** — Paper presented

**Current status (May 2026): effectively frozen / community-maintained.** The official `multimodal-art-projection/YuE` repo's last commit is **2025-06-04** (GitHub API, retrieved 2026-05-18), nearly 12 months stale. There is no announced YuE-2 or successor from the M-A-P org. All forward development (quantization, ComfyUI, GUI, MPS attempts, exllama, mp3 extension) now happens in community forks like [YuEGP](https://github.com/deepbeepmeep/YuEGP), [YuE-exllamav2](https://github.com/sgsdxzy/YuE-exllamav2), and [YuE-extend](https://github.com/Mozer/YuE-extend). The space the team itself has moved into is **ACE-Step** (released January 2026), which the ACE-Step paper explicitly critiques YuE for "slow inference and structural artifacts" ([arXiv 2506.00045](https://arxiv.org/abs/2506.00045)).

---

## 2. Architecture

YuE is a **two-stage autoregressive LLM** pipeline built on the **LLaMA2** decoder-only transformer backbone — *not* a diffusion model ([paper](https://arxiv.org/html/2503.08638v1)).

**Stage-1 LM (the headline 7B model):**
- LLaMA2-style decoder, ~6B–7B parameters (HF metadata reports 6B for the s1 checkpoints).
- Performs **track-decoupled next-token prediction**: interleaves *vocal* and *instrumental* token streams in a single sequence, so a single AR pass produces both tracks rather than mixing them. This is YuE's central architectural innovation.
- Conditioned on (genre tags || lyrics) using **structural progressive conditioning** — lyrics are chunked per section (verse/chorus/bridge) and re-injected so attention does not lose alignment over a 5-minute generation.
- Native context: 8192 tokens (~163 s of mix-track audio, ~81 s of dual-track); extended to **16384** in the anneal phase.

**Stage-2 LM:**
- 1B-parameter LLaMA2 model (HF reports ~2B for `YuE-s2-1B-general`).
- Predicts the **residual RVQ codebooks (layers 1–7)** conditioned on Stage-1's codebook-0 output, restoring acoustic fidelity that the semantic-rich layer-0 tokens omit.
- Context length 8192.

**Audio tokenizer — X-Codec:**
- YuE uses **X-Codec** (from the same M-A-P lineage as MERT), a *semantic-acoustic fused* RVQ codec that bolts a HuBERT-based semantic stream onto an RVQ-VAE acoustic stream.
- 12 RVQ codebooks total; YuE uses the first **8** (codebook size 1024 each).
- 50 Hz frame rate over 16 kHz audio.
- A separate **YuE-upsampler** (GAN-based) converts the 16 kHz output up to higher sample rate / better fidelity for delivery ([paper §3](https://arxiv.org/html/2503.08638v1), [HF Transformers X-Codec docs](https://huggingface.co/docs/transformers/main/model_doc/xcodec)).

**Track handling:** Dual-track. Vocal and accompaniment are *separately tokenized* via X-Codec, then interleaved in the AR sequence — this is the paper's claimed advantage over single-track-mixture baselines (less information loss, cleaner vocal/inst separation).

**Max generation length:** Up to **~5 minutes** per song, generated in chunks/sessions and stitched.

**Lyrics conditioning:** Plain text lyrics with section tags ([verse], [chorus], etc.) + a genre tag prompt (a vocabulary from `top_200_tags.json` such as "pop", "female vocal", "energetic", "120 bpm"). The progressive conditioning means each new section re-references the relevant lyric chunk.

**Training scale:** Stage-1 used ~**2T tokens** across phases; data includes ~**650K hours of in-the-wild music** plus ~**70K hours of TTS** for vocal grounding ([paper](https://arxiv.org/html/2503.08638v1)).

---

## 3. Variants and Sizes

From the [M-A-P YuE collection on HuggingFace](https://huggingface.co/collections/m-a-p/yue-6797d55e22990ae89b90a3d6) (downloads accurate as of mid-2026):

| Model | Params | Stage | Language | Mode | Downloads (last month) |
|---|---|---|---|---|---|
| `YuE-s1-7B-anneal-en-cot` | 6B | 1 | English | Chain-of-Thought (default) | 8.48k |
| `YuE-s1-7B-anneal-en-icl` | 6B | 1 | English | In-Context Learning (style cloning) | 805 |
| `YuE-s1-7B-anneal-zh-cot` | 6B | 1 | Mandarin/Cantonese | CoT | 203 |
| `YuE-s1-7B-anneal-zh-icl` | 6B | 1 | Mandarin/Cantonese | ICL | 89 |
| `YuE-s1-7B-anneal-jp-kr-cot` | 6B | 1 | Japanese/Korean | CoT | 95 |
| `YuE-s1-7B-anneal-jp-kr-icl` | 6B | 1 | Japanese/Korean | ICL | 25 |
| `YuE-s2-1B-general` | 2B | 2 | language-agnostic | residual decoder | 6.01k |
| `YuE-s1-0.5B` | 0.5B | 1 | research/ablation | partial training | 94 |
| `YuE-upsampler` | – | post | n/a | GAN upsampler | – |
| `xcodec_mini_infer` | – | tokenizer | n/a | X-Codec encoder/decoder | – |

**Naming key:**
- `s1` / `s2` = Stage-1 (semantic) / Stage-2 (acoustic residual).
- `anneal` = checkpoints after the final "annealing" pretraining phase (highest quality public weights).
- `cot` = chain-of-thought prompting variant; `icl` = in-context learning variant (used for *style/voice cloning* from a reference audio).
- A community **GGUF quantization** of the Stage-2 model exists at [`multimodalart/YuE-s2-1B-general-Q8_0-GGUF`](https://huggingface.co/multimodalart/YuE-s2-1B-general-Q8_0-GGUF) — useful for Mac llama.cpp paths.

There is **no official "YuE-2" or major version bump**. The team's successor effort is the separately branded ACE-Step.

---

## 4. License

**Apache License 2.0** for code *and* weights — switched on 2025-01-30 in response to community pressure ([GitHub README news entry](https://github.com/multimodal-art-projection/YuE), [HF model card](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl)).

- **Commercial use:** *Permitted and explicitly encouraged.* The model card says: "Artists and content creators are encouraged to sample and incorporate outputs into their own works, and even monetize them, with attribution to the model's name (\"YuE by HKUST/M-A-P\")."
- **Attribution:** Required for public / commercial outputs.
- **Recommended labeling:** outputs should be marked "AI-generated", "YuE-generated", "AI-assisted", or "AI-auxiliated".
- **No training-data redistribution clause** — Apache 2.0 covers code and the released weights; training data itself was *not* released, so no redistribution permission is granted on data.
- **Liability:** users bear sole responsibility for any copyright infringement, plagiarism, or misuse. Likely — no explicit watermarking or content-credentials are baked into output (no direct confirmation in docs).

Practical takeaway for the user's Suno-like platform: **YuE is one of the very few music-generation foundation models with a clean, no-strings commercial license**, which is the single most valuable thing about it.

---

## 5. Languages Supported

Five officially: **English, Mandarin Chinese, Cantonese, Japanese, Korean** ([GitHub README](https://github.com/multimodal-art-projection/YuE), [demo page](https://map-yue.github.io/)).

- English has the deepest training and the most-downloaded checkpoint.
- `zh` covers Mandarin and Cantonese (sharing a checkpoint).
- `jp-kr` shares one checkpoint for Japanese and Korean.
- The demo site shows code-switching (English ↔ Mandarin within the same song) working.
- No official support for Spanish, French, German, Hindi, Arabic, etc. — outputs in those languages will likely be poor or accented (no direct user reports confirm, but architecturally the model has never seen them at scale).

---

## 6. Quality Assessment

**Strengths (from paper + demos):**
- Wide vocal range — the paper reports YuE "closely matching top-performing closed-source systems like Suno V4" on vocal-range metrics ([WhiteFiber summary](https://www.whitefiber.com/blog/yue-ai-music-generator)).
- Strong **musical structure** — verse/chorus/bridge transitions are coherent over 3–5 min, which most diffusion music models still struggle with.
- Demos show death-growl metal, scatting jazz, Beijing opera, rap, ballad, country, and soul — *genre breadth* is genuinely impressive ([map-yue.github.io](https://map-yue.github.io/)).
- ICL mode can clone the timbre/style of a reference clip — closest open-source analogue to Suno's "cover" or Udio's style transfer.

**Weaknesses (from paper's own discussion + community feedback):**
- **Acoustic fidelity gap.** Multiple sources, including the paper itself, note "clear deficiencies in vocal and accompaniment acoustic quality, likely due to limitations of its current audio tokenization method"; the authors propose super-resolution / better decoders as future work.
- **Mono / narrow stereo image** — third-party reviews call out that output "lacks the production quality needed for commercial music platforms" and is essentially mono ([articlex review](https://www.articlex.com/open-source-ai-music-generation-breakthrough-with-yue-software/)).
- **Slow inference + structural artifacts** — the explicit critique from the ACE-Step authors (ICLR 2026 submission): "LLM-based models like YuE excel at lyrics alignment but suffer from slow inference and structural artifacts" ([ACE-Step paper](https://arxiv.org/abs/2506.00045)).
- **Mumbling / lyric drift** appears in long sections — there is no explicit Reddit thread surfacing here, but the paper's "Section 12 Unsuccessful Attempts" and `--repetition-penalty` / decoding-temperature emphasis in the GitHub Issues suggest users hit it.

**Quality verdict vs Suno v4 / v5:**
- Suno v4 ≈ YuE on *vocal range and genre breadth.*
- Suno v4/v5 clearly ahead on *mix polish, stereo width, vocal clarity, and emotional nuance.*
- YuE ahead of Suno only on *openness, controllability via lyrics tags, and structural macro-form for niche genres*.

---

## 7. Inference Performance

From the README's official hardware table:

| GPU | Time for 30 s of audio (Stage-1 + Stage-2) |
|---|---|
| NVIDIA H800 80GB | **~150 s** |
| NVIDIA RTX 4090 24GB | **~360 s** |
| ≤24GB GPU | Max ~2 concurrent sessions; cannot generate a full song in one pass |
| ≥80GB GPU (H100/A100/H800) | Recommended for a full 4+ session song |

Extrapolating to a **3-minute song** (~6× a 30 s clip, plus some overhead for stitching):
- H800: ~15–18 minutes
- A100 80GB: ~18–22 minutes (likely — close to H800 throughput)
- RTX 4090: ~35–45 minutes
- M5 Max MPS (user's machine): **no official support, no public benchmark.**

**VRAM:** Full-precision FP16 Stage-1 needs ~16–18 GB; Stage-2 + upsampler add ~4–6 GB. Single-pass full-song generation comfortably wants 40–80 GB.

**Quantized / community paths:**
- **YuEGP** ("YuE for the GPU Poor") brings VRAM down to **<10 GB** via 8-bit quantization and sequential offload ([YuEGP repo](https://github.com/deepbeepmeep/YuEGP)).
- **YuE-exllamav2** claims up to **5× speedup** via ExLlamaV2 + FlashAttention-2 + BF16 ([YuE-exllamav2](https://github.com/sgsdxzy/YuE-exllamav2)) — NVIDIA-only.
- **GGUF Stage-2** exists ([multimodalart/YuE-s2-1B-general-Q8_0-GGUF](https://huggingface.co/multimodalart/YuE-s2-1B-general-Q8_0-GGUF)). Stage-1 7B GGUF is not officially published as of 2026-05.

**Apple Silicon / MPS:**
- **No official MPS support.** GitHub README references `--cuda_idx`, no `mps` or `mac` mentions.
- No HF Space or fork advertises working MPS inference. The architecture is plain LLaMA2 + standard transformer ops, so MPS port is *technically feasible* (likely — Stage-1 fits well within the user's 128GB unified memory), but the X-Codec encoder/decoder has Flash-Attention CUDA kernels that would need replacement. Realistic path on M5 Max today: run the Stage-2 GGUF via llama.cpp Metal backend, but Stage-1 has no public Metal/MPS port.
- A community attempt to MPS-port has *not* surfaced in any search or GitHub issue as of May 2026.

---

## 8. Repo Health

Data from the GitHub API on 2026-05-18 for `multimodal-art-projection/YuE`:

- **Stars:** 6,219
- **Forks:** 741
- **Open issues:** 86
- **License:** Apache-2.0
- **Default branch last push:** `2025-06-04T13:08:48Z` — **~11 months stale**
- **Most-recent commits:** all README edits and the finetune-merge PRs on the same day (2025-06-04).
- **Recent issue traffic (sampled 2025-Q4 through 2026-Q2):** install errors (CUDA / `codecmanipulator` missing), ComfyUI integration questions, attention-mask warnings, "how do I generate a full song" basics, a Feb-2026 PR proposing `SDPA as default attention` that received zero engagement. Maintainer responses are essentially absent in 2026.
- **Fine-tuning support:** present, merged June 2025 via PR #126 (LoRA, no QLoRA, requires CUDA 12.1+, PyTorch 2.4, Megatron-formatted JSONL data).
- **vLLM / SGLang:** listed in TODO, never implemented.
- **llama.cpp:** community Stage-2 GGUF exists but no official integration; Stage-1 not converted.
- **Tensor parallel / Stemgen mode:** TODO, never shipped.

**Verdict:** The repo is in **maintenance/abandonment limbo.** Apache 2.0 + open weights mean anyone can fork; community forks are where the energy is.

---

## 9. Real-World Adoption

- **Replicate:** Hosted at [`fofr/yue`](https://replicate.com/fofr/yue/api) with an official cog wrapper at [`replicate/cog-yue`](https://github.com/replicate/cog-yue) — production-ready pay-per-second API.
- **HuggingFace Spaces:** at least three live demos — [`fffiloni/YuE`](https://huggingface.co/spaces/fffiloni/YuE), [`innova-ai/YuE-music-generator-demo`](https://huggingface.co/spaces/innova-ai/YuE-music-generator-demo), `Harveyu/YuE-music-generator-demo`.
- **ComfyUI:** community node [`smthemex/ComfyUI_YuE`](https://github.com/smthemex/ComfyUI_YuE) exposes YuE as a node graph (issue #148 confirms active users in 2026).
- **Pinokio:** one-click Windows installer ships in the official Pinokio script directory ([pinokio.co](https://pinokio.co/)).
- **GPU-poor / consumer forks:** `deepbeepmeep/YuEGP` (sub-10 GB VRAM), `sgsdxzy/YuE-exllamav2` (5× speedup), `Mozer/YuE-extend` (mp3 extension + GUI), `Sorrymakershen/YuE-for-windows`.
- **SiliconFlow:** no public listing found as of 2026-05 (likely — search returned no SiliconFlow YuE endpoint).
- **Forks:** 741 total, dominated by consumer-VRAM optimization rather than research extension.

For a Suno-like platform, the **Replicate `fofr/yue` endpoint is the lowest-friction starting point** to test quality before self-hosting.

---

## 10. Fine-Tuning

- **LoRA fine-tuning is documented and supported** since June 2025, in the [`finetune/` directory](https://github.com/multimodal-art-projection/YuE/tree/main/finetune) with `scripts/preprocess_data.sh` and `scripts/run_finetune.sh`.
- Configurable `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`.
- **Training scripts are open** — Megatron-style data pipeline; data must be converted to JSONL containing X-Codec tokens + lyric/structure/genre metadata, then to Megatron binary.
- **QLoRA: not documented.** No 4-bit fine-tuning path is described in the official repo (likely — community forks may have hacked it together).
- Requires CUDA 12.1+, PyTorch 2.4, Python 3.10; GPU memory not explicitly stated but realistically wants ≥40 GB VRAM for the 7B Stage-1 LoRA.
- No published guide for full-parameter fine-tuning of Stage-1 — implied to need multi-node H100.

---

## 11. Pros and Cons

**Pros**
- True open weights (Apache 2.0), commercial-use-friendly, with strong attribution-only requirements.
- Genuine dual-track output (vocals + instrumentals as separable streams), not just a mix.
- Multilingual coverage of EN / ZH / Cantonese / JP / KR with code-switching demos.
- Strong macro-structure for 3–5 minute songs — verses, choruses, bridges hold together.
- Healthy ecosystem of quantized / consumer-VRAM forks and a turnkey Replicate endpoint.
- LoRA fine-tuning code is shipped and merged.
- Comparable vocal range to Suno v4 on the paper's metrics.

**Cons**
- **Repo is effectively dormant since June 2025** — no maintainer engagement on 2026 issues/PRs.
- Acoustic fidelity is noticeably below Suno v4/v5 — mono-ish, less polished mix, occasional vocal artifacts/mumbling on long passages.
- **No MPS / Apple Silicon support**, official or community — a real problem for the user's M5 Max workflow.
- Slow inference even on H800 (~150 s per 30 s clip, → 15+ minutes per full song before quantization).
- VRAM hungry: full-song single-pass wants 80 GB; consumer GPUs need session-stitching tricks.
- No QLoRA / no vLLM / no SGLang / no tensor parallel — all in TODO purgatory.
- Training data not released → fine-tuning needs you to bring your own licensed corpus.
- Tokenizer (X-Codec) is the bottleneck for fidelity, and YuE inherits this ceiling — no upgrade path planned in this codebase.
- An explicit successor effort (ACE-Step) from an adjacent team claims to fix YuE's specific weaknesses.

---

## 12. Verdict for the User's Suno-like Platform

**Best fit for the user's M5 Max / 128 GB platform if:**
- The product needs **commercial-grade licensing freedom** above all else — YuE is one of the very few open music models you can ship in a paid product without licensing carve-outs.
- You target **multilingual song generation (EN + Mandarin/Cantonese + JP/KR)** with code-switching — YuE is the strongest open option here.
- You can offload generation to a **rented H100/H800 (Replicate, Runpod, Lambda)** rather than insisting on local M5 Max inference — *MPS support is the blocker on the user's hardware.*
- You want a base to **LoRA fine-tune on a proprietary genre/voice corpus** — the official fine-tune scripts work today, and Apache 2.0 lets you keep your LoRA private and commercial.

**Where YuE will underperform competitors:**
- **Acoustic polish** — Suno v4/v5 and Udio will sound noticeably more professional out of the box. If your platform's selling point is "studio-quality vocals", YuE is not there.
- **Throughput per dollar** — diffusion-based ACE-Step and DiffRhythm-2 are dramatically faster (ACE-Step claims ~15× speedup); for a high-volume product, the AR-LLM architecture is expensive.
- **Real-time / interactive generation** — not viable; YuE is batch-only.
- **Local Mac inference** — until somebody ports Stage-1 to MPS or ships a Stage-1 GGUF, the user's M5 Max can at best play around with the Stage-2 model in llama.cpp Metal mode.

**Concrete recommendation for the user:** use YuE via Replicate's `fofr/yue` endpoint as the **commercial-license-clean fallback / multilingual specialist** in the platform's model router, and seriously evaluate ACE-Step in parallel for the throughput-sensitive default path. Plan a future LoRA fine-tune on YuE only after the platform has clear vertical (genre, language, or vocal-style) demand that the closed APIs cannot serve.

---

## References

- GitHub repo: <https://github.com/multimodal-art-projection/YuE>
- Paper (arXiv): <https://arxiv.org/abs/2503.08638>
- Paper (HTML): <https://arxiv.org/html/2503.08638v1>
- OpenReview: <https://openreview.net/forum?id=hZy6YG2Ij8>
- Project / demos: <https://map-yue.github.io/>
- HF collection: <https://huggingface.co/collections/m-a-p/yue-6797d55e22990ae89b90a3d6>
- HF s1 English ICL card: <https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl>
- Replicate: <https://replicate.com/fofr/yue/api>
- Replicate cog: <https://github.com/replicate/cog-yue>
- YuEGP fork: <https://github.com/deepbeepmeep/YuEGP>
- YuE-exllamav2 fork: <https://github.com/sgsdxzy/YuE-exllamav2>
- YuE-extend fork: <https://github.com/Mozer/YuE-extend>
- ComfyUI node: <https://github.com/smthemex/ComfyUI_YuE>
- GGUF Stage-2: <https://huggingface.co/multimodalart/YuE-s2-1B-general-Q8_0-GGUF>
- HF X-Codec docs: <https://huggingface.co/docs/transformers/main/model_doc/xcodec>
- ACE-Step paper (successor-style critique): <https://arxiv.org/abs/2506.00045>
- WhiteFiber technical summary: <https://www.whitefiber.com/blog/yue-ai-music-generator>
- HF Space demo (fffiloni): <https://huggingface.co/spaces/fffiloni/YuE>
- HF Space demo (innova-ai): <https://huggingface.co/spaces/innova-ai/YuE-music-generator-demo>
