# Open-Source Song Generation for a Suno-Like Platform — Executive Summary

*Research compiled 2026-05-18. Target hardware: Apple M5 Max, 128 GB unified memory, MPS backend. Deployment target: **free non-profit Hugging Face Space.** Commercial license is NOT a constraint.*

---

## TL;DR

**Use ACE-Step 1.5 XL as the default base model.** It is the open-source full-song-with-vocals foundation model in May 2026 that combines:

1. **First-class Apple Silicon support** (hybrid MLX + PyTorch MPS, dedicated `clockworksquirrel/ace-step-apple-silicon` fork) — best local-dev experience.
2. **MIT license** — clean for forks, attribution, and weight redistribution on the HF Space.
3. **State-of-art-or-better quality** — 4.4/5 vs Suno v4's 4.1/5 vocal naturalness in a 50-person blind test (folk, classical, jazz; Suno still wins pop/EDM polish).
4. **Sub-minute generation** on M5 Max (projected ~30 – 50 s for a 4-min song). Sub-2 s/song on A100 — fits inside HF ZeroGPU's free 60 s budget.
5. **Cheap LoRA fine-tuning** — 8 songs trainable in ~1 hour on a single 3090, LoRA training works on MPS.
6. **50+ languages**, vocals + instrumentation natively, **<4 GB VRAM minimum** — runs on free ZeroGPU Spaces.
7. **Active 10.4 k-star repo**, native ComfyUI integration, AMD vendor-blessed for production.

**Now that commercial use is not a constraint** (free non-profit HF Space deployment), **SongGeneration 2 / LeVo 2** comes back into contention as a premium-quality alternative — its Tencent non-commercial license permits academic/research/education use. Vendor benchmarks (unverified) put it ahead of Suno v5 on lyric accuracy. The trade-off is **22 – 28 GB VRAM** (needs paid Space tier, not free ZeroGPU) and no first-party MPS path (only a buggy community `SongGen-Mac` fork) — meaning M5 Max local dev is painful.

Pair the primary pick with **HeartMuLa-MLX** as an alternate-quality choice (Apache 2.0, 2.1× faster than ACE-Step on M-series via Apple's MLX) and **YuE on Replicate** as the multilingual fallback.

---

## Ranking (non-profit HF Space context)

| Rank | Model | Params | bf16 weights | License | MPS | Vocal Quality vs Suno | LoRA | Verdict |
|---|---|---|---|---|---|---|---|---|
| **1** | **ACE-Step 1.5 XL** | ~8 B (4 B DiT + 4 B planner) | ~16 GB | MIT | First-class | 4.4/5 vs Suno v4 4.1 (blind test) | ✅ 1h on 3090 | **Default base.** Fits free ZeroGPU. |
| **2** | **SongGeneration 2 / LeVo 2** | 4 B | ~8 GB | Tencent non-commercial (OK for non-profit Space) | Buggy community fork only | Vendor PER 8.55 % vs Suno v5 12.4 % | ❌ | Premium quality. Needs paid Space (22 – 28 GB VRAM). |
| **3** | **HeartMuLa** | ~6.8 B (4 B MuLa + 2 B Codec + 0.8 B ASR) | ~13.6 GB | Apache 2.0 | Strong MLX port | Vendor: lowest PER per-language, unverified | ❌ public | Strong A/B alternate. |
| **4** | **DiffRhythm 2** | ~1.17 B (1 B DiT + 170 M VAE-dec) | ~2.4 GB | Apache 2.0 | Likely OK, untested | Authors admit gap vs Suno v4.5 | ❌ no training code | Speed tier. 210 s ceiling. Cheapest to host. |
| **5** | **YuE** | ~8 B (7 B + 1 B + upsampler) | ~16 GB | Apache 2.0 | ❌ broken (flash-attn hard dep) | Vocal range matches Suno v4 | ✅ LoRA, CUDA-only | Multilingual specialist; via Replicate only. |
| — | SongBloom | 2 B | ~4 GB | Custom (likely NC) | Reported OK | unknown | ❌ | Research baseline. |
| — | InspireMusic / FunMusic | 1.5 B | ~3 GB | Apache 2.0 | ❌ CUDA-only deps | No vocals yet | n/a | Skip until vocal release. |

---

## Decision tree (non-profit HF Space deployment)

```
HF Space tier?
  ├── Free ZeroGPU (60s/req on shared A100) ─┐
  │                                          ├── ACE-Step 1.5 (turbo workflow generates a song well under 60 s)
  │                                          └── DiffRhythm 2 (smallest, fastest, fits easily)
  │
  └── Paid GPU Space (A10G / A100 dedicated) ─┐
                                              ├── Default: ACE-Step 1.5 XL (best speed-quality, MPS for local dev)
                                              ├── Premium tier: SongGeneration 2 v2-large (best vendor benchmarks)
                                              ├── Multilingual breadth: YuE (50+ via Replicate; local broken)
                                              └── Alternate: HeartMuLa via heartlib-mlx
```

---

## What the research surfaced that changes the picture

1. **Non-profit HF Space deployment removes the Tencent-license blocker.** SongGeneration 2 / LeVo 2 is back in contention as a premium-quality alternative. Its custom license permits "academic, research, and education purposes" — a free non-profit Space sits comfortably inside that scope. Practical blockers remain (22 – 28 GB VRAM means paid Space tier, no working MPS) but the licence is no longer a no-go.

2. **The YuE team migrated to ACE-Step.** The ACE-Step paper (Jun 2025) explicitly critiques YuE for "slow inference and structural artifacts." YuE's repo has been dormant since 2025-06-04. Treat YuE as a frozen capability, not a developing one.

3. **Vocal-support contradiction on ACE-Step is resolved: yes, it does vocals.** Several search results said "instrumental only" — that's confused with the `Text2Samples` LoRA. The base model produces vocals + instruments natively, lyric-conditioned, with `[verse] [chorus] [bridge]` structural tags.

4. **DiffRhythm 2's biggest fix is structural coherence**, not raw quality. Its v1's brutal Hacker News thread complained "no identifiable chorus in any of the demo songs"; v2's block flow-matching (semi-autoregressive over 2 s blocks) closes that gap. Its **210 s ceiling is a regression** from v1-full's 4m45s.

5. **HeartMuLa is the dark-horse 2026 entrant.** Apache 2.0, 4 B params, modular (CLAP + Transcriptor + Codec + MuLa LM), MLX port available. Vendor PER claims are aggressive (0.09 EN / 0.12 ZH) but not in comparable units to LeVo's 8.55 % — direct comparison unreliable until somebody runs a neutral A/B.

6. **Every "beats Suno v5" claim is vendor-published.** The only neutral preference study located ([arXiv 2506.19085](https://arxiv.org/html/2506.19085v1)) stops at Suno v3.5. **Plan an in-house blind A/B before betting product positioning on any vendor number.**

7. **Apple Silicon is fine for music gen — much friendlier than LTX-Video 2.3.** No complex64, no SDPA-on-meta-tensor traps, no multimodal-Gemma gotchas. The mundane MPS issues here are: `flash-attn` substitution with SDPA, fp16 conv1d → fp32 in audio decoders, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` for OOM tuning. Three of the five candidate models already ship a working MPS or MLX path.

8. **HF Space hardware tier dictates the model choice as much as quality does.** Free ZeroGPU = 60 s budget per request, shared A100 — only ACE-Step or DiffRhythm 2 finish in time. Paid A10G/A100 Spaces unlock SongGeneration 2 v2-large but the user has to pay (or get an HF community grant).

---

## Recommended starting setup for the M5 Max (with HF Space deploy in mind)

```bash
# 1. Primary base model — ACE-Step 1.5 XL via the Apple Silicon fork
git clone https://github.com/clockworksquirrel/ace-step-apple-silicon \
  ~/Projects/llm/music-generator/ace-step
cd ~/Projects/llm/music-generator/ace-step
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Hybrid backend: Qwen3 planner → MLX, DiT decoder → PyTorch MPS, bf16 throughout
# ~16 GB bf16 weights for the XL stack; M5 Max 128 GB has massive headroom

# 2. Production UI — ace-step-ui (stem extraction, library, LAN access)
git clone https://github.com/fspecii/ace-step-ui \
  ~/Projects/llm/music-generator/ace-step-ui

# 3. Alternate model — HeartMuLa via MLX port (~13.6 GB bf16)
git clone https://github.com/Acelogic/heartlib-mlx \
  ~/Projects/llm/music-generator/heartlib-mlx

# 4. (Optional) Premium-quality experiment — SongGeneration 2 / LeVo 2
# Mac fork has a pre-chorus bug; only do this if you're OK developing on a rented
# Linux+CUDA box and the M5 Max becomes just your control plane.
git clone https://github.com/tencent-ailab/SongGeneration \
  ~/Projects/llm/music-generator/songgeneration
```

For the throughput-sensitive **multilingual fallback (YuE)**, use Replicate's `fofr/yue` endpoint — do *not* attempt local inference on M5 Max until somebody ports Stage-1 to MPS. Treat YuE as remote-only for now.

**HF Space deployment notes:**
- **Free ZeroGPU Space** → only ACE-Step or DiffRhythm 2 will finish a song inside the 60 s shared-A100 budget. Use ACE-Step's turbo workflow.
- **Paid GPU Space** → A10G (24 GB) handles ACE-Step XL comfortably; A100 (40 GB) opens the door to SongGeneration 2 v2-large.
- **Apply for a [Community GPU Grant](https://huggingface.co/docs/hub/en/spaces-gpus#community-gpu-grants)** if budget is the deciding factor — HF approves these regularly for non-profit demos.

---

## Sources

All claims are cited inline in the per-model deep-dives:

- [01_yue.md](./01_yue.md)
- [02_diffrhythm.md](./02_diffrhythm.md)
- [03_acestep.md](./03_acestep.md)
- [04_newcomers_and_survey.md](./04_newcomers_and_survey.md)
- [05_apple_silicon_mps_audit.md](./05_apple_silicon_mps_audit.md)
- [06_comparison_matrix.md](./06_comparison_matrix.md) — side-by-side spec table
- [07_platform_architecture.md](./07_platform_architecture.md) — Suno-clone system design with ACE-Step at the core
