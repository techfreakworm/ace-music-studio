---
title: ACE Music Studio
emoji: "🎵"
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
python_version: "3.11"
suggested_hardware: zero-a10g
hf_oauth: false
preload_from_hub:
  - ACE-Step/acestep-v15-xl-sft *.safetensors,config.json,scheduler/*,vae/*,tokenizer/*
  - Qwen/Qwen2.5-7B-Instruct *.safetensors,config.json,tokenizer*
  - facebook/htdemucs_ft *.th
---

# ACE Music Studio

A single-process Gradio app that wraps [ACE-Step 1.5 XL SFT](https://github.com/ace-step/ACE-Step-1.5) for full-song generation with vocals, with bundled Qwen 2.5 7B for lyrics and Demucs for stem separation. Runs locally on Apple Silicon (MPS+MLX) or NVIDIA (CUDA), deploys to Hugging Face Spaces (ZeroGPU).

[![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Live-FFFFFF?style=flat-square)](https://huggingface.co/spaces/techfreakworm/ace-music-studio)
[![GitHub stars](https://img.shields.io/github/stars/techfreakworm/ace-music-studio?style=flat-square)](https://github.com/techfreakworm/ace-music-studio/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-FFFFFF?style=flat-square)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-FFFFFF?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Backed by ACE-Step](https://img.shields.io/badge/backend-ACE--Step%201.5%20XL%20SFT-FFFFFF?style=flat-square)](https://github.com/ace-step/ACE-Step-1.5)

→ **Live demo:** https://huggingface.co/spaces/techfreakworm/ace-music-studio

---

## What's inside

Five tabs. One ACE-Step pipeline underneath. Progressive disclosure — defaults stay short and reveal advanced controls only when asked.

| Mode | Inputs | What it does |
|---|---|---|
| **Generate** | prompt + lyrics + tags | Text → full song with vocals + instruments |
| **Cover** | reference audio + new lyrics | Style transfer from a reference clip |
| **Extend** | seed audio + extension prompt | Continue an existing song forward |
| **Edit** | source audio + segment + target lyrics | Repaint a segment OR flow-morph caption-to-caption |
| **Lyrics** | brief + structure | Qwen 2.5 7B drafts structurally-tagged lyrics |

Every song tab supports stacked LoRAs — 4 bundled presets (RapMachine, Chinese Rap, Lyric2Vocal, Text2Samples) plus arbitrary `.safetensors` uploads.

---

## Quick start (local)

Requires **Python 3.11**, ~32 GB free disk for weights, and **128 GB unified memory recommended on Apple Silicon** (M5 Max ideal; M3 Max+ workable).

```bash
git clone https://github.com/techfreakworm/ace-music-studio
cd ace-music-studio
bash setup.sh
source .venv/bin/activate
python app.py    # http://127.0.0.1:7860
```

First launch downloads the ACE-Step + Qwen + Demucs weights into your HF cache (`~/.cache/huggingface/hub/`). Subsequent starts are fast.

**Apple Silicon notes:** `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically by `app.py`. The Mac path uses the [`clockworksquirrel/ace-step-apple-silicon`](https://github.com/clockworksquirrel/ace-step-apple-silicon) fork for MLX-LM + MPS-DiT hybrid execution.

## Quick start (HF Spaces)

```bash
git remote add space https://huggingface.co/spaces/techfreakworm/ace-music-studio
git push space main
```

`preload_from_hub` in this README pre-downloads ~32 GB of weights at build time. `app._bootstrap()` mirrors the read-only build cache into `~/hf-cache-rw/` then symlinks every snapshot into `./models/<repo>/` so the pipeline finds them locally on first request.

## Architecture

See [`docs/superpowers/specs/2026-05-18-ace-music-studio-design.md`](docs/superpowers/specs/2026-05-18-ace-music-studio-design.md) for the full design. UI mockups live in [`docs/superpowers/specs/mockups/`](docs/superpowers/specs/mockups/).

## License

MIT for the app code (see `LICENSE`). ACE-Step 1.5 XL SFT, Qwen 2.5 7B Instruct, and Demucs htdemucs_ft retain their respective upstream licenses (Apache 2.0 / Apache 2.0 / MIT).

## Credits

ACE-Step by [ACE Studio × StepFun](https://ace-step.github.io/). Apple Silicon port by [clockworksquirrel](https://github.com/clockworksquirrel/ace-step-apple-silicon). Qwen 2.5 by [Alibaba](https://huggingface.co/Qwen). Demucs by [Meta AI](https://github.com/facebookresearch/demucs). Built by [@techfreakworm](https://huggingface.co/techfreakworm).
