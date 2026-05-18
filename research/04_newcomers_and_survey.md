# 2026 Open-Source Music Generation Models — Newcomers and Survey

*Date: 2026-05-18. Target hardware: M5 Max, 128 GB unified memory, MPS backend.*

This report investigates the freshest 2026 open-source song-with-vocals generators relevant to building a Suno-like platform locally. Primary focus: **SongGeneration 2 / LeVo 2** (Tencent, March 2026) and **HeartMuLa** (Jan 2026). Also covered: DiffRhythm 2, ACE-Step 1.5 XL, SongBloom, YuE, FunMusic/InspireMusic, NotaGen. Independent benchmark sources are sparse for releases this fresh; vendor claims are flagged.

---

## 1. SongGeneration 2 / LeVo 2 (Tencent AI Lab)

**Overview.** Builder: Tencent AI Lab. Release: 2026-03-01 (v2-large weights), arXiv paper "LeVo" appeared 2025-06-09 (2506.07520). Status: actively updated, v2 is the headline model on the repo ([GitHub](https://github.com/tencent-ailab/SongGeneration), [HF](https://huggingface.co/tencent/SongGeneration)).

**Architecture.** Hybrid LLM + Diffusion. The **LeLM** language model handles global structure and performance details with a hierarchical scheme that parallel-models *Mixed Tokens* (melody/structure) and *Dual-Track Tokens* (separate vocal vs. accompaniment streams). A downstream diffusion module synthesises the high-fidelity acoustic waveform from those tokens. Multi-preference DPO alignment (~200k positive/negative pairs) is applied offline ([repo README](https://github.com/tencent-ailab/SongGeneration/blob/main/README.md)).

**Variants and sizes.** Five tiers ([HF model card](https://huggingface.co/tencent/SongGeneration/blob/main/README.md)):
- `base` (2:30 max, zh) — 10/16 GB VRAM, RTF 0.67
- `base-new` (zh + en) — same VRAM
- `base-full` (4:30, zh + en) — 12/18 GB VRAM, RTF 0.69
- `large` (zh + en) — 22/28 GB VRAM, RTF 0.82
- **`v2-large` — 4 B params, multilingual (zh/en/es/ja/…), 22/28 GB VRAM, RTF 0.82, 4:30 max length**

**License.** Custom Tencent "academic, research and education purposes" license, **commercial use explicitly prohibited** ([LICENSE](https://github.com/tencent-ailab/SongGeneration/blob/main/LICENSE)). This is the headline blocker for a Suno-like SaaS product.

**Languages.** v2-large: Chinese, English, Spanish, Japanese plus others (multilingual lyrics input).

**Vocals.** Yes. Separable dual-track output (vocals + accompaniment, instrumental-only, or a cappella).

**Speed and hardware.** Reference numbers measured on Tencent's H20 (96 GB) GPU: RTF 0.82 for v2-large. No first-party MPS code path, but a community fork **[SongGen-Mac](https://github.com/Rdx-ai-art/SongGen-Mac)** runs the older base/large models via PyTorch MPS on M-series Macs — author reports **~6 min wall-clock per ~2 min song on M1 Max 64 GB (base), ~12 min for large**, and notes RAM+swap usage hits ~70 GB during inference. The fork is tiny (9 GitHub stars) and does **not** yet wrap v2-large — porting that to MPS on the M5 Max 128 GB is a real engineering task and will likely need careful attention bf16 casts (LeLM) + diffusion sampler patches.

**Benchmarks.** Vendor claims ([repo README](https://github.com/tencent-ailab/SongGeneration)): Phoneme Error Rate **8.55 %** vs. Suno v5 12.4 % and Mureka v8 9.96 %. Subjective panel: 20 industry professionals scored across Overall Quality, Melody, Arrangement, Sound Quality (instrument and vocal), Structure on 100 songs/model — Tencent reports v2-large above all open-source baselines and parity with top commercial. **All numbers vendor-reported; no independent re-run located.** The arXiv "Benchmarking Music Generation Models via Human Preference Studies" paper (2506.19085) precedes v2 and tops out at Suno v3.5 / Udio — does not cover LeVo ([arXiv](https://arxiv.org/html/2506.19085v1)).

**Repo health.** 1.6 k stars / 191 forks, last meaningful update 2026-03-01. 12 active discussion threads ([repo](https://github.com/tencent-ailab/SongGeneration)).

**Adoption.** Hugging Face Space (free demo), WaveSpeed AI hosted endpoint ([WaveSpeed](https://wavespeed.ai/models/wavespeed-ai/song-generation)), SECourses Patreon GUI wrapper, vllm-omni issue tracking integration ([HF Space](https://huggingface.co/spaces/tencent/SongGeneration)). No production SaaS adoption seen.

**Pros.** State-of-art lyric accuracy (vendor); dual-track outputs ready for mixing; multilingual; clear inference budget; 4 B params fits comfortably in 128 GB unified memory in fp16.

**Cons.** **License kills commercial use** for a Suno-clone product. No official MPS path. Community Mac fork lags v2. Inference time on Apple Silicon is multi-minute per song. No independent benchmark verification.

---

## 2. HeartMuLa (HeartMuLa team / academic group)

**Overview.** Builder: HeartMuLa research collective, paper credited to Jordi Pons-affiliated group ([Substack explainer](https://artintech.substack.com/p/heartmula-explained)). First weights: 2026-01-19 (`HeartMuLa-oss-3B`), latest: 2026-02-13 (`HeartMuLa-oss-3B-happy-new-year`). arXiv 2601.10547 ([abs](https://arxiv.org/abs/2601.10547)).

**Architecture.** Four-stage family ([landing page](https://heartmula.github.io/)): **HeartCLAP** (audio-text alignment / retrieval), **HeartTranscriptor** (Whisper-style lyric ASR), **HeartCodec** (12.5 Hz neural audio codec, low frame rate but high-fi), **HeartMuLa** (LLM-based song generator conditioned on lyrics, tags, and reference audio). Section-level fine-grained control (intro/verse/chorus) is a stated feature.

**Variants and sizes.** Six published weights on [HF](https://huggingface.co/HeartMuLa):
- `HeartMuLa-oss-3B` — 4 B text-to-audio (1.21 k downloads, 255 likes)
- `HeartMuLa-RL-oss-3B-20260123` — 4 B RL-tuned variant
- `HeartMuLa-oss-3B-happy-new-year` — 4 B latest checkpoint
- `HeartCodec-oss-20260123` — 2 B codec
- `HeartTranscriptor-oss` — 0.8 B ASR
- `HeartMuLa-7B` — internal/unreleased

(Note the naming oddity: HF model card lists "3B" name but 4 B parameter size; treat as ~4 B.)

**License.** **Apache 2.0** — confirmed via [LICENSE](https://github.com/HeartMuLa/heartlib/blob/main/LICENSE). Commercial use permitted. This is the strongest licensing position of any model in this report.

**Languages.** Multilingual; demo page covers en, zh, ja, ko, es. Paper claims "almost all languages."

**Vocals.** Yes — lyric-conditioned vocal synthesis is the core capability. The paper claims best-in-class lyric intelligibility.

**Speed and hardware.** RTF ≈ 1.0 (paper). VRAM via the ComfyUI integration ([FL-HeartMuLa](https://github.com/filliptm/ComfyUI_FL-HeartMuLa)): 3 B model needs **12 GB+ VRAM** at full precision, **6 GB with 4-bit bnb quantisation** (CUDA-only). 7 B will need 24 GB / 12 GB quantised. **MPS supported** on M1/M2/M3/M4 (M5 implied), but 4-bit quantisation does not work on MPS, so the M5 Max will run native bf16. 128 GB unified memory is plenty headroom for the 4 B model and an eventual 7 B release.

**Benchmarks.** Vendor PER claims: **0.09 (English), 0.12 (Chinese)** — flagged "lowest across every language tested," beating Suno v5 and MiniMax Music 2.0 ([blog](https://huggingface.co/blog/azhan77168/heartmula)). **Note PER unit mismatch with SongGeneration's 8.55 % — these are likely measured on different scales (HeartMuLa percentages may be normalised differently); direct comparison unreliable.** Demo page compares against Suno v4.5, Mureka v7.6, YuE, DiffRhythm 2, ACE-Step ([demos](https://heartmula.github.io/)). The single HN comment ([46691275](https://news.ycombinator.com/item?id=46691275)) said "initial results promising, more so than recent ACE-Step 1.5." Otherwise **no independent A/B tests located**; the HF promo blog is vendor-aligned content.

**Repo health.** [github.com/HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib): 3.6 k stars / 396 forks / 71 open issues. Last release Feb 2026. Larger and more active than SongGeneration's repo.

**Adoption.** WaveSpeed AI hosted endpoint ([blog](https://wavespeed.ai/blog/posts/introducing-wavespeed-ai-heartmula-generate-music-on-wavespeedai/)); ComfyUI node `FL-HeartMuLa`; HeartMuse local app integrating Ollama for lyric writing ([HN](https://news.ycombinator.com/item?id=46871828)).

**Pros.** Apache 2.0 — usable for a commercial product. Modular architecture (codec + ASR + CLAP + gen) is reusable. Strong lyric intelligibility claim. Active repo. Explicit MPS support documented downstream.

**Cons.** Heavy marketing tone in third-party coverage; benchmarks all vendor-published. 7 B not yet released. No standardised MOS or ELO numbers from a neutral evaluator. PER values reported in non-comparable units to peers.

---

## 3. DiffRhythm 2 (ASLP-Lab)

**Overview.** Successor to DiffRhythm v1.2. arXiv 2510.22950, v3 2026-02-03 ([arXiv](https://arxiv.org/abs/2510.22950)). Original repo: [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm).

**Architecture.** Music VAE at 5 Hz frame rate + Diffusion Transformer with **block flow matching** for lyric-to-vocal alignment. Adds cross-pair preference optimisation (RLHF) and a stochastic block representation alignment loss for musicality. Semi-autoregressive blockwise generation.

**License.** Apache 2.0 (inherited from v1, confirmed 2025-03-07).

**Languages, vocals, hardware.** Multilingual; full vocals + instrumental; uses 44.1 kHz stereo; up to 4:45 song length. DiffRhythm v1 can generate a full song in ~10 s on a single A100 — v2 should be in the same ballpark. MPS not officially stated but PyTorch DiT models port relatively cleanly. Parameter count not disclosed in v2 abstract.

**Benchmarks.** Vendor claims top-of-class fidelity; no independent verification specific to v2.

**Pros/cons.** Pros: very fast, permissive license, mature codebase. Cons: no public param count, no first-party MPS path, lyric clarity historically the weak spot vs LeVo/HeartMuLa.

---

## 4. ACE-Step 1.5 XL (ACE Studio × StepFun)

**Overview.** [github.com/ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5). arXiv 2602.00744. Most user-tested local-first option. 10.4 k stars / 1.3 k forks — **biggest community by far**.

**Architecture.** LM planner (0.6 B / 1.7 B / 4 B selectable) + DiT decoder (2 B or 4 B XL). XL DiT ~9 GB bf16.

**License.** **MIT**. Commercial use allowed.

**Languages.** 50+.

**Speed and hardware.** Under 2 s/song on A100, under 10 s on RTX 3090, **<4 GB VRAM** for DiT-only minimum. **Explicit Mac MPS support** with `start_gradio_ui_macos.sh`; MLX backend optimisation noted. Easiest M5 Max install of any model in this list.

**Benchmarks.** Vendor: SongEval 8.12, AudioBox 7.76, claims to beat Suno v5 and MiniMax 2.5 across 11 dimensions ([project page](https://ace-step.github.io/ace-step-v1.5.github.io/)). DEV Community write-up positions it "between Suno v4.5 and v5" — more honest framing.

**Pros.** Best Mac story, MIT licence, LoRA personalisation in days, tiny VRAM. **Cons.** Vocal naturalness still trails Suno v5 in casual user tests.

---

## 5. SongBloom (Tencent AI Lab)

[github.com/tencent-ailab/SongBloom](https://github.com/tencent-ailab/SongBloom). 778 stars. Interleaved autoregressive sketch + diffusion refinement, 2 B params, MPS supported, lengths up to 240 s in Oct 2025 update. Same Tencent academic-only LICENSE pattern (not Apache). Up to 150 s songs from lyrics + 10 s reference audio. Useful as a research baseline; **same commercial-use prohibition as SongGeneration** likely applies — verify before deploying.

---

## 6. YuE (M-A-P / HKUST)

[github.com/multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE). LLaMA-2 backbone, lyric-to-song, **Apache 2.0** since 2025-01-30, 5 min max length, dual-track ICL mode, no v2 announced. Strong vocal emotion for ballads/R&B. Llama.cpp issue 11467 still tracks GGUF support. Solid permissive fallback if HeartMuLa underperforms.

---

## 7. FunMusic / InspireMusic (Alibaba FunAudioLLM)

[github.com/FunAudioLLM/FunMusic](https://github.com/FunAudioLLM/FunMusic). Qwen2.5 backbone + flow-matching super-res. 1.3 k stars. Apache 2.0. **No MPS support, requires Flash Attention 2.6 + CUDA 11.8+** — effectively NVIDIA-only. Song-with-vocals models announced but not yet released; current ships are music-only/audio.

---

## Survey table — 2026 open-source song generators

| Model | Builder | Release | Params | License | Vocals | Repo |
|---|---|---|---|---|---|---|
| SongGeneration 2 / LeVo 2 | Tencent AI Lab | 2026-03 | 4 B | Custom non-commercial | Yes, dual-track | [link](https://github.com/tencent-ailab/SongGeneration) |
| HeartMuLa-oss-3B | HeartMuLa | 2026-01 | ~4 B + 2 B codec + 0.8 B ASR | Apache 2.0 | Yes, multilingual | [link](https://github.com/HeartMuLa/heartlib) |
| DiffRhythm 2 | ASLP-Lab | 2025-10 → 2026-02 (v3) | undisclosed | Apache 2.0 | Yes | [link](https://github.com/ASLP-lab/DiffRhythm) |
| ACE-Step 1.5 XL | ACE Studio × StepFun | 2026-01 | LM 0.6–4 B + DiT 2–4 B | MIT | Yes | [link](https://github.com/ace-step/ACE-Step-1.5) |
| SongBloom | Tencent AI Lab | 2025-06 → 2025-10 | 2 B | Custom (likely non-commercial) | Yes | [link](https://github.com/tencent-ailab/SongBloom) |
| YuE | M-A-P / HKUST | 2025-01 | up to 7 B | Apache 2.0 | Yes | [link](https://github.com/multimodal-art-projection/YuE) |
| InspireMusic (FunMusic) | Alibaba FunAudioLLM | 2025-01 | 1.5 B | Apache 2.0 | Coming (music only today) | [link](https://github.com/FunAudioLLM/FunMusic) |
| NotaGen / NotaGen-X | Central Conservatory + ElectricAlexis | 2025 | symbolic-only | MIT | n/a (ABC/XML) | [link](https://github.com/ElectricAlexis/NotaGen) |

---

## Dark horses / experimental

- **NotaGen-X** — DeepSeek-R1-style RL on symbolic music. Outputs ABC/MusicXML (not audio). Could feed a TTS-vocal model for a hybrid composer → singer pipeline ([repo](https://github.com/ElectricAlexis/NotaGen), [arXiv](https://arxiv.org/abs/2502.18008)).
- **LLaSA / LLaSA+** — Llama-3B-backbone TTS pipeline ([arXiv](https://arxiv.org/html/2508.06262v1)); not music, but emergent prosody good enough to consider as the vocal layer behind a NotaGen score.
- **DiffRhythm+** — preference-optimised DiffRhythm variant, arXiv 2507.12890; mid-stage between v1 and v2.
- **AudioX** — anything-to-audio DiT, 2503.10522; useful for sound design and SFX layering, not full-song.
- **MelodyFlow** — text-controllable DiT with flow-matching for music editing.
- **HeartMuse** — local Ollama-orchestrated lyric → HeartMuLa song app ([HN](https://news.ycombinator.com/item?id=46871828)); reference for building a thin product wrapper.

---

## Skeptic's bottom line for the M5 Max 128 GB build

1. **For a commercial Suno-clone**: **HeartMuLa** (Apache 2.0, native MPS, 4 B fits easily, Feb-2026 checkpoint, modular components reusable) is the strongest pick. Verify their PER claims yourself before fundraising-style messaging.
2. **For best raw quality, research only**: **SongGeneration 2 v2-large** — but the Tencent licence forbids commercial deployment and the v2 weights don't yet have a maintained MPS port. The community SongGen-Mac fork targets the older base/large.
3. **For fastest iteration / smallest VRAM**: **ACE-Step 1.5 XL** (MIT, native Mac script, <4 GB VRAM) — under-promises vocal naturalness vs HeartMuLa but ships today on Apple Silicon with the cleanest licence story.
4. Reliable independent benchmark for these specific 2026 releases does not yet exist; the only neutral preference study found ([arXiv 2506.19085](https://arxiv.org/html/2506.19085v1)) stops at Suno v3.5 and does not cover LeVo, HeartMuLa, or ACE-Step. **Run your own blind A/B before betting a product on any vendor PER number.**
