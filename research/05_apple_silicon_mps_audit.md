# Apple Silicon / MPS Compatibility Audit — Music Generation Models

Hardware target: **M5 Max, 128 GB unified memory**. Date: 2026-05-18.

Honest read: MPS is a second-class citizen for almost every music-gen repo. CUDA is the assumed default; Mac support, when it exists, is community-driven. Below is the per-model evidence with verdicts.

---

## 1. YuE (multimodal-art-projection/YuE)

- **Official MPS support:** None. The README requires `cuda >= 11.8`, conda-installed `cudatoolkit=11.8`, and **flash-attn 2 is mandatory** to avoid OOM on long sequences ([YuE README](https://github.com/multimodal-art-projection/YuE/blob/main/README.md)).
- **Community reports:** Issue #51 ("Instructions to run on Mac") is open and **unanswered** ([#51](https://github.com/multimodal-art-projection/YuE/issues/51)). No working Mac fork.
- **Backend compatibility:** Hard CUDA dependency through flash-attn; xformers/triton flash paths are CUDA-only ([HF forum thread](https://discuss.huggingface.co/t/best-practices-to-use-models-requiring-flash-attn-on-apple-silicon-macs-or-non-cuda/97562)). Stage 1 (7B LLaMA-2-style) and Stage 2 (1B) both transformer-based; in principle portable, but no one has shipped it.
- **Memory:** 7B + 1B + upsampler. Author recommends **≥80 GB VRAM** for full song; 24 GB OK for short clips. On 128 GB unified memory this fits, *if* you can swap flash-attn for SDPA.
- **Apple-Silicon timing:** None reported.
- **Verdict:** **Doesn't work out of the box. Likely broken on MPS.** Would need a non-trivial fork: strip flash-attn, replace with `torch.nn.functional.scaled_dot_product_attention`, and audit RoPE/KV-cache for MPS dtype quirks. There is also a "GPU Poor" fork ([deepbeepmeep/YuEGP](https://github.com/deepbeepmeep/YuEGP)) but it targets CUDA/ROCm with 8-bit quant — **no Mac path**.

## 2. DiffRhythm v1 and v2 (ASLP-lab)

- **Official MPS support:** DiffRhythm v1 explicitly states *"DiffRhythm can now run on MacOS!"* with `brew install espeak-ng` ([Readme](https://github.com/ASLP-lab/DiffRhythm/blob/main/Readme.md)). No specific MPS notes, but it works.
- **DiffRhythm 2:** `requirements.txt` is **clean of CUDA-only packages** — no flash-attn, xformers, triton, mamba_ssm, deepspeed, bitsandbytes ([requirements.txt](https://github.com/ASLP-lab/DiffRhythm2/blob/main/requirements.txt)). Just `torch==2.7`, `torchaudio==2.7`, `transformers`, `safetensors`, `muq`, `librosa`. The 3.9 % "CUDA" language stat in the repo is benign — auto-detected from a small kernel file, but no compiled extensions in the pip install path.
- **Community reports:** No GitHub issues or Reddit threads surface specific MPS bugs for DiffRhythm — implying it either works quietly or no one has tried at scale. The architecture (latent diffusion + DiT with flow matching, very similar to Stable Audio Open / SD3) is the same class that *does* work on MPS via diffusers.
- **Memory:** DiffRhythm-base needs **≥8 GB VRAM**; `--chunked` decoding reduces it further. Trivial on 128 GB.
- **Apple-Silicon timing:** Not benchmarked publicly, but extrapolating from Stable Audio Open MPS (≈3× CPU speedup) the 285-second full-song run should land in the low minutes on M5 Max.
- **Verdict:** **Just works on MPS (likely) / Works with workarounds.** Highest confidence pick.

## 3. ACE-Step 1.5 (ace-step/ACE-Step)

- **Official MPS support:** **First-class.** README explicitly advertises Mac + AMD + Intel + CUDA. macOS scripts auto-set `ACESTEP_LM_BACKEND=mlx --backend mlx` — the language-model side runs on Apple's **MLX**, the DiT side on **PyTorch MPS** ([INSTALL.md](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md)). bfloat16 supported on MPS since PyTorch 2.4.
- **Community reports:** Real-world M2 Air 16 GB run: 5–10 min per song, hit MPS-OOM, fixed with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` ([bioerrorlog](https://en.bioerrorlog.work/entry/ace-step-15-local-m2-macbook)). A dedicated [clockworksquirrel/ace-step-apple-silicon](https://github.com/clockworksquirrel/ace-step-apple-silicon) fork already centralised MPS detection, swapped CUDA cache calls for `torch.mps.empty_cache()` / `torch.mps.synchronize()`, and tuned VAE conv1d tile sizes for Metal limits.
- **Backend compatibility:** Flash-attn auto-disabled on MPS. `torch.compile` disabled on MPS. nanovllm not on Mac. Otherwise clean.
- **Memory:** 4 GB DiT-only / 6 GB LLM+DiT minimum; ~10 GB total install.
- **Apple-Silicon timing (M1 Pro 16 GB vs M3 Pro 36 GB vs A100, from the AS fork's benchmarks):**

  | Task | M1 Pro | M3 Pro | A100 |
  | --- | --- | --- | --- |
  | 30 s turbo song | ~45 s | ~25 s | ~2 s |
  | 30 s SFT song | ~3 min | ~1.5 min | ~8 s |

  **Extrapolated M5 Max:** turbo ~10–15 s, SFT ~45–60 s for 30 s output. Best Mac-citizen of the bunch.

- **Verdict:** **Just works on MPS.** Already production-grade on M-series.

## 4. SongGeneration 2 / LeVo 2 (Tencent)

- **Official MPS support:** None. Official repo pins `flash-attn 2.7.4.post1` for CUDA 12 + torch 2.6, though `--not_use_flash_attn` flag exists ([Tencent SongGeneration](https://github.com/tencent-ailab/SongGeneration)).
- **Community reports:** [Rdx-ai-art/SongGen-Mac](https://github.com/Rdx-ai-art/SongGen-Mac) fork — "Runs completely on your Mac's GPU via MPS on PyTorch." Tested on M1 Max 64 GB / macOS 15.7.2. **Pre-chorus block produces gibberish vocals** — known regression vs CUDA.
- **Backend compatibility:** Hybrid LLM + diffusion architecture. Once flash-attn is stripped, the LLM side uses SDPA fine on MPS.
- **Memory (Mac fork):** Base ≥24 GB RAM, ~70 GB total app RAM including swap during inference. Large ≥32 GB, hits ~80 GB. **On 128 GB M5 Max this fits cleanly without swap.**
- **Apple-Silicon timing (M1 Max 64 GB):** Base ~4–6 min for ~2 min of audio. Large ~10–25 min for ~2:30. M5 Max should be roughly 2–3× faster (better mem bandwidth + more GPU cores).
- **Verdict:** **Works with workarounds (community fork only).** Functional but watch the pre-chorus bug.

## 5. HeartMuLa (HeartMuLa/heartlib)

- **Official MPS support:** Not in the README. CUDA-first design with `--mula_device` / `--codec_device` flags ([heartlib](https://github.com/HeartMuLa/heartlib)). RTF ≈ 1.0 on CUDA.
- **Community reports:** **Strong MLX port exists**: [Acelogic/heartlib-mlx](https://github.com/Acelogic/heartlib-mlx). Claims **2.1× faster than PyTorch MPS** on M2 Max (13.4 s vs 27.9 s end-to-end), 8.7× faster model load, 100 % numerical parity with PyTorch.
- **Backend compatibility:** No flash-attn / mamba / triton in the official deps — clean transformer + neural codec. MLX port supports bfloat16.
- **Memory (MLX port):** 3B model ~6 GB, HeartCodec ~2 GB, KV-cache ~1 GB/min of audio. **Full 1-min song ≈ 11 GB.** 32 GB minimum recommended; M5 Max 128 GB blows past this. 7B variant not yet released as of Feb 2026.
- **Apple-Silicon timing:** M2 Max ≈ 11.6 s to generate 50 frames; M5 Max should comfortably exceed real-time for the 3B model.
- **Verdict:** **Just works on MPS via MLX port.** Second-best Mac story after ACE-Step. The official PyTorch path is untested but should run on MPS once you bypass any CUDA cache calls.

## 6. MusicGen (Meta / audiocraft) — reference

- **Official MPS support:** None. AudioCraft officially supports CUDA or CPU only ([audiocraft README](https://github.com/facebookresearch/audiocraft)). Issues [#13](https://github.com/facebookresearch/audiocraft/issues/13) and [#31](https://github.com/facebookresearch/audiocraft/issues/31) are open requests, no merged PR. EnCodec decoder ops misbehave on MPS — common workaround is to **move decoder to CPU** while keeping the LM on MPS.
- **Community / MLX:** Multiple solid ports — [Andrade Olivier's port](https://medium.com/@andradeolivier/i-ported-musicgen-to-apple-silicon-generate-music-from-text-on-your-macbook-9eaf95992053), [Nat Taylor's MusicGen MLX test](https://nattaylor.com/blog/2024/musicgen-via-mlx/). M4 Max: small model 8 s audio in ~6 s (faster than realtime). M1: ~60 s for 9 s of audio at 500 steps. AudioGen (sibling model) [works on MPS](https://blog.peddals.com/en/apple-mps-to-generate-audio-with-meta-audiogen/) by moving decoder ops to CPU.
- **Memory:** 300 M small / 1.5 B medium / 3.3 B large. Trivial on 128 GB.
- **Verdict:** **Partial on raw PyTorch MPS (CPU fallback for decoder); Just works via MLX port.**

## 7. Stable Audio Open (Stability AI) — reference

- **Official MPS support:** Diffusers supports `device="mps"` for the SAO pipeline ([Stable Audio docs](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_audio)).
- **Community reports:** [phlo.info](https://phlo.info/posts/using-stable-audio-tools-on-apple-silicon/) reports 51 s CPU → 17 s MPS by swapping `cuda` → `mps` in two files. **fp16 conv1d in the decoder is pathologically slow on MPS** — fix is `model.pretransform.model_half = False; model.to(torch.float32)` ([HF discussion](https://huggingface.co/stabilityai/stable-audio-open-small/discussions/1)).
- **Memory:** 1.21 B params. Trivial.
- **Apple-Silicon timing:** ~17 s per 3-s sample on M1-class; M5 Max should be a few seconds.
- **Verdict:** **Works with workarounds** (force fp32 in decoder).

---

## Metal / MLX Apple-Native Equivalents

- **ACE-Step**: Native MLX backend in the official repo for the LM side. **Closest thing to a first-party Mac music model.**
- **HeartMuLa**: [heartlib-mlx](https://github.com/Acelogic/heartlib-mlx) — 2.1× speedup over PyTorch MPS, full numerical parity.
- **MusicGen**: Multiple MLX ports, faster than real-time on M4 Max small model.
- **Stable Audio Open**: MLX-Audio family ([Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)) covers TTS/STT; SAO has unofficial MLX ports.
- **YuE / DiffRhythm / SongGeneration**: **No MLX ports** as of May 2026.

There is no umbrella "MLX-music" framework; each project rolls its own port.

---

## Practical Recommendation

**Start with ACE-Step 1.5.** It is the only model with first-party Apple Silicon support, hybrid MLX + MPS execution, published M-series benchmarks, and no CUDA-only dependencies. The user's 128 GB unified memory completely eliminates the OOM workaround other Mac users hit on 16–36 GB machines.

**Second pick: HeartMuLa via the MLX port** ([heartlib-mlx](https://github.com/Acelogic/heartlib-mlx)). Faster than the PyTorch MPS path, bfloat16, well-benchmarked. 3B only for now; 7B unreleased.

**Third pick: DiffRhythm v2** — clean deps, README claims macOS support, similar architecture class to Stable Audio Open which is known to work on MPS with the fp32 decoder workaround.

**Avoid on MPS unless you enjoy yak-shaving:**
- **YuE** — flash-attn-mandatory, no Mac fork, no MLX port.
- **SongGeneration / LeVo** — only via [SongGen-Mac](https://github.com/Rdx-ai-art/SongGen-Mac) fork, pre-chorus bug, 70+ GB RAM pressure with swap. Workable on 128 GB but not pleasant.

**Remote-dev path:** For YuE specifically, **train/develop on a rented H100 or A100** (RunPod, Lambda, Modal, Replicate) and pull weights for inference on M5 Max **only if** you fork it to drop flash-attn. Otherwise treat YuE as a remote-only model. For everything else on this list, M5 Max is sufficient as the primary development machine.

**On the user's prior LTX-Video burns:** music models are LM/diffusion stacks without the multi-modal Gemma + complex64 + SDPA-on-meta-tensor traps that bit LTX-2.3. The main MPS gotchas here are mundane: flash-attn substitution, fp16 conv1d in audio decoders, and `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` for high-watermark allocator behaviour.
