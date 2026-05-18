# Open-Source Song Generation Models — Side-by-Side Comparison

*Compiled 2026-05-18 for M5 Max / 128 GB unified memory target.*

---

## Headline matrix

| Property | **ACE-Step 1.5 XL** | **HeartMuLa 4B** | **DiffRhythm 2** | **YuE 7B** | SongGeneration 2 |
|---|---|---|---|---|---|
| **Builder** | ACE Studio × StepFun | HeartMuLa | NWPU ASLP-lab + Xiaomi | M-A-P / HKUST | Tencent AI Lab |
| **Release** | 2026-01-28 | 2026-01-19 | 2025-10-27 → 2026-02-03 (v3) | 2025-01-26 | 2026-03-01 |
| **License** | **MIT** | **Apache 2.0** | **Apache 2.0** | **Apache 2.0** | **Custom NON-commercial** |
| **Repo stars** | 10.4 k | 3.6 k | ~2.3 k (v1) + 0.16 k (v2) | 6.2 k | 1.6 k |
| **Last major commit** | v0.1.7 (2026-04-24) | 2026-02 | 2026-02 | 2025-06-04 (stale) | 2026-03-01 |
| **Architecture** | LM-planner (Qwen3 0.6/1.7/4 B) + DiT (2/4 B) | CLAP + ASR + 12.5 Hz Codec + 4 B LLM | 5 Hz Music VAE + DiT w/ block flow matching | LLaMA2 7B AR Stage-1 + 1B Stage-2 + X-Codec | LeLM hybrid + diffusion decoder |
| **Params (largest)** | up to 8 B (4 B DiT + 4 B LM) | ~4 B + 2 B codec + 0.8 B ASR | ~1 B DiT + 170 M VAE-dec | 7 B + 1 B + upsampler | 4 B (v2-large) |
| **Audio rate** | 44.1 kHz stereo | 24 kHz neural codec | 44.1 kHz stereo | 16 kHz then upsampled | High-fi via diffusion |
| **Max length** | 4+ min | ≥1 min, scaling | **210 s (regression from v1)** | 5 min | 4:30 |
| **Vocals + Instruments** | ✅ Native | ✅ Native | ✅ Native, single stream | ✅ Native, dual-track AR | ✅ Dual-track |
| **Languages** | 50+ | 5+ (en/zh/ja/ko/es benchmarked) | Bilingual EN/ZH + JP/KR/ES marketing-only | EN, Mandarin, Cantonese, JP, KR | zh/en/es/ja + others |
| **VRAM (minimum)** | **<4 GB** with offload (turbo) | 6 GB 4-bit / 12 GB bf16 | 8 GB v1 with `--chunked` | 24 GB consumer / 80 GB single-pass | 22–28 GB |
| **VRAM (recommended)** | 12 GB+ offload, 24 GB optimal | 24 GB for 7B (unreleased) | 24 GB | 80 GB H100/H800 | 28 GB |
| **MPS / Apple Silicon** | **First-class, MLX + MPS, dedicated fork** | **MLX port, 2.1× PyTorch MPS** | Likely OK; clean deps; untested | ❌ Mandatory flash-attn | Community fork, pre-chorus bug |
| **MPS bench M-series (30 s clip)** | M3 Pro 25 s turbo / 1.5 min SFT | M2 Max 11.6 s for 50 frames | not published | not published | M1 Max 4–6 min for 2 min |
| **MPS bench M5 Max (projected)** | turbo ~10–15 s / SFT ~45–60 s | <real-time | low-minute range | n/a | ~2–3× M1 Max | 
| **Speed (RTF on A100 / 4090)** | sub-2 s/song on A100 (v1.5) | RTF ≈ 1.0 | v2 RTF 0.213 (4090) → ~45 s for 210 s | 27 steps RTF 27.27× on A100 (v1, ~15 min/song) | RTF 0.82 (H20) |
| **Vocal naturalness vs Suno v4** | **4.4/5 vs 4.1/5** (blind 50-person test) | Vendor only, unverified | Authors admit clear gap vs v4.5 | Comparable vocal range; weaker mix | Vendor claim parity, unverified |
| **Lyric alignment (PER)** | Strong (lyric tags) | Vendor: 0.09 EN / 0.12 ZH (unit mismatch) | **0.13 (open-source SOTA)** | Strong from lyric tags | Vendor: 8.55 % |
| **Fine-tuning support** | ✅ LoRA, 8 songs/1h on 3090, **MPS-validated** | ❌ public training code | ❌ "Coming soon" since Mar 2025 | ✅ LoRA (Megatron pipeline, CUDA 12.1+) | ❌ |
| **ComfyUI integration** | ✅ Native, official workflows | ✅ FL-HeartMuLa | ✅ billwuhao/ComfyUI_DiffRhythm | ✅ smthemex/ComfyUI_YuE | ✅ |
| **Replicate hosted** | ❌ no first-party | ❌ | ❌ | ✅ fofr/yue | ❌ |
| **Style/audio reference** | LoRA + lyric tags | Reference audio supported | Reference audio supported | ICL mode (style cloning) | Limited |
| **Stem separation** | Built into `fspecii/ace-step-ui` via Demucs | Modular Codec is reusable | ❌ single stream | ✅ AR dual-track is inherently separable | ✅ Dual-track output |
| **Continuation / extension** | Supported in workflows | Limited | Supported | ✅ explicit continuation mode | Supported |
| **Production deployments** | acestep.io, ace-step.app, fspecii/ace-step-ui, AMD-blessed | WaveSpeed AI, HeartMuse local app | Chutes serverless | Replicate fofr/yue, HF Spaces | WaveSpeed AI, HF Space |
| **Watermarking / content credentials** | None baked-in | None baked-in | None baked-in | None baked-in | None baked-in |
| **License gotchas** | None (MIT) | None (Apache 2.0) | Ethical disclaimer (non-binding) | Attribution required ("YuE by HKUST/M-A-P"), label "AI-generated" | **Commercial use prohibited** |
| **Independent benchmarks** | Yes — 50-person blind test, AMD vendor-validated | None located | Internal MOS only | Paper + community | None — Tencent only |

---

## Quality dimensions (qualitative)

| Dimension | Best (open source) | Notes |
|---|---|---|
| **Pop / EDM polish** | (none — Suno v4/v5 still wins) | All open models lag commercial. |
| **Folk / classical / jazz vocal naturalness** | **ACE-Step 1.5 XL** | Wins blind test vs Suno v4 in these genres. |
| **Lyric intelligibility (PER)** | **DiffRhythm 2** (0.13) | HeartMuLa claims lower but unit-incomparable. |
| **Musical macro-structure (verse/chorus/bridge over 3-5 min)** | **YuE** or **ACE-Step 1.5** (planner) | LM-planner models lead diffusion-only here. |
| **Stereo image, mix depth** | **DiffRhythm 2** (44.1 kHz stereo native) | YuE is mono-ish; ACE-Step is stereo but variable. |
| **Genre breadth** | **YuE** | Death-growl metal to Beijing opera to rap. |
| **Multilingual breadth** | **ACE-Step 1.5** | 50+ languages w/ lyric tags; YuE deep on 5 only. |
| **Code-switching (English ↔ Mandarin in one song)** | **YuE** | Explicit demos. |
| **Speed / cost per song** | **ACE-Step 1.5** | Sub-2 s/song on A100; <minute on M5 Max. |
| **Modular reusability of components** | **HeartMuLa** | Codec/ASR/CLAP separately exportable. |

---

## Cost model (rough)

| Path | Per-song cost | Latency | Best for |
|---|---|---|---|
| Self-host ACE-Step 1.5 on M5 Max | $0 marginal (electricity) | ~30-50 s | Dev, beta, low-volume |
| Self-host ACE-Step 1.5 on rented A100 80 GB | ~$0.0001 (sub-2 s × $1.50/hr) | <2 s | Production, paid SaaS |
| Replicate `fofr/yue` | ~$0.30-1.00 per song (estimated from 4090 cog runtime) | 5-15 min | Multilingual fallback, occasional |
| Self-host DiffRhythm 2 on 4090 | $0 marginal on owned 4090 | ~45 s | Speed tier, instrumentals |
| Replicate / WaveSpeed managed endpoints | varies | varies | Cold-start / spike capacity |

---

## License risk matrix

| License | Commercial SaaS | Output ownership | Risk |
|---|---|---|---|
| MIT (ACE-Step 1.5) | ✅ | User owns | Lowest |
| Apache 2.0 (ACE-Step v1, HeartMuLa, DiffRhythm v1/v2, YuE) | ✅ with attribution | User owns | Low |
| Tencent custom (SongGeneration, SongBloom) | ❌ **prohibited** | n/a | **Blocks SaaS** |
| Suno API (closed-source baseline) | $ paid tier | platform terms | Medium |

---

## Hardware sizing on M5 Max (128 GB unified memory)

| Model | Fits? | Headroom | Notes |
|---|---|---|---|
| ACE-Step 1.5 XL (4 B DiT + 4 B planner) | ✅ huge | ~120 GB free | Overkill; LoRA training viable in-RAM |
| HeartMuLa 4B + 2 B codec + 0.8 B ASR | ✅ huge | ~120 GB free | 7 B variant when released will also fit |
| DiffRhythm 2 (~1 B + 170 M VAE-dec) | ✅ trivial | ~125 GB free | Tiny by 2026 standards |
| YuE 7B Stage-1 + 1B Stage-2 + upsampler | ✅ but blocked | n/a | Memory fine, **flash-attn dep blocks MPS** |
| SongGeneration 2-large (4 B + diffusion) | ✅ comfortable | ~100 GB free | Community fork bug aside, fits |

**Conclusion:** the user's 128 GB unified memory completely eliminates memory pressure for every model in this list. The constraint is software (MPS kernel compat, flash-attn substitution), not hardware.
