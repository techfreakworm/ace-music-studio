---
title: ACE Music Studio
emoji: 🎵
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.2.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Song generation studio on ACE-Step 1.5 XL SFT
preload_from_hub:
- ACE-Step/Ace-Step1.5
- ACE-Step/acestep-v15-xl-sft
- ACE-Step/ACE-Step-v1-chinese-rap-LoRA
- ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA
- Qwen/Qwen2.5-7B-Instruct
---

# ACE Music Studio

A single-process Gradio app that wraps [ACE-Step 1.5 XL SFT](https://github.com/ace-step/ACE-Step-1.5) for full-song generation with vocals, with Qwen 2.5 for lyrics drafting and Demucs for stem separation. Runs locally on **Apple Silicon (MPS+MLX)** or **NVIDIA (CUDA)**, deploys to **Hugging Face Spaces (ZeroGPU)**.

→ **Live demo:** https://huggingface.co/spaces/techfreakworm/ace-music-studio

---

## What's inside

Five modes. One ACE-Step pipeline underneath. Progressive disclosure — defaults stay short and reveal advanced controls only when asked.

| Mode | Inputs | What it does |
|---|---|---|
| **Generate** | prompt + lyrics + tags | Text → full song with vocals + instruments |
| **Cover** | reference audio + new lyrics | Style transfer from a reference clip |
| **Extend** | seed audio + extension prompt | Continue an existing song forward |
| **Edit** | source audio + segment + target lyrics | Repaint a segment OR flow-morph caption-to-caption |
| **Lyrics** | brief + structure | Qwen 2.5 7B drafts structurally-tagged lyrics |

Every song mode supports a single stacked LoRA — bundled presets plus arbitrary `.safetensors` uploads. The preset registry ships with the official ACE-Step LoRAs published on HF:

- [ACE-Step/ACE-Step-v1-chinese-rap-LoRA](https://huggingface.co/ACE-Step/ACE-Step-v1-chinese-rap-LoRA)
- [ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA](https://huggingface.co/ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA)

After every song generation, three post-process actions sit beneath the player: **Demucs stem separation**, **pyloudnorm normalisation to -14 LUFS**, and **MP3 320k export via ffmpeg**.

---

## Local setup

Requires **Python 3.11**, ~32 GB free disk for weights, and **128 GB unified memory recommended on Apple Silicon** (M5 Max ideal; M3 Max+ workable). On NVIDIA, ~24 GB VRAM.

```bash
git clone https://github.com/techfreakworm/ace-music-studio
cd ace-music-studio
bash setup.sh             # creates .venv, installs requirements
source .venv/bin/activate
python app.py             # http://127.0.0.1:7860
```

`setup.sh` detects Apple Silicon and adds `requirements-mac.txt` (MLX-LM + the [`clockworksquirrel/ace-step-apple-silicon`](https://github.com/clockworksquirrel/ace-step-apple-silicon) fork). First launch downloads weights into your HF cache (`~/.cache/huggingface/hub/`).

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically by `app.py` so the few MPS-unsupported ops degrade to CPU.

## HF Spaces deploy

```bash
git remote add space https://huggingface.co/spaces/<your-handle>/ace-music-studio
git push space main
```

`preload_from_hub` (this README's frontmatter) pre-downloads the ACE-Step 1.5 XL SFT umbrella weights at build time. `app._bootstrap_spaces_cache()` runs once at module init when `SPACE_ID` is set, symlinking the HF cache into the fork's expected `<site-packages>/checkpoints/` layout so the pipeline finds them on first request. `@spaces.GPU(duration=180)` decorates the click handlers — on Spaces it gates them to a ZeroGPU worker, locally it's a no-op.

## Architecture

See [`docs/superpowers/specs/2026-05-18-ace-music-studio-design.md`](docs/superpowers/specs/2026-05-18-ace-music-studio-design.md) for the full design and [`docs/superpowers/plans/2026-05-18-ace-music-studio.md`](docs/superpowers/plans/2026-05-18-ace-music-studio.md) for the implementation plan.

## License

MIT for the app code (see `LICENSE`). ACE-Step 1.5 XL SFT, Qwen 2.5 7B Instruct, and Demucs htdemucs_ft retain their respective upstream licenses (Apache 2.0 / Apache 2.0 / MIT).

## Credits

ACE-Step by [ACE Studio × StepFun](https://ace-step.github.io/). Apple Silicon port by [clockworksquirrel](https://github.com/clockworksquirrel/ace-step-apple-silicon). Qwen 2.5 by [Alibaba](https://huggingface.co/Qwen). Demucs by [Meta AI](https://github.com/facebookresearch/demucs).

Made with ❤️ by [Mayank Gupta](https://huggingface.co/techfreakworm).
