# ACE Music Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-process Gradio app at `~/Projects/llm/music-generator/` that wraps ACE-Step 1.5 XL SFT for full-song generation, deploys to free Hugging Face ZeroGPU + runs locally on M5 Max (MPS+MLX) and CUDA. Five modes (Generate / Cover / Extend / Edit / Lyrics), preset + custom LoRA stacking, bundled Qwen 2.5 7B for lyrics, Demucs stems, Brutalist Mono theme.

**Architecture:** Direct ACE-Step Python API, lazy backend singleton, PEFT adapter-stacking for LoRAs, mlx-lm on Mac / transformers on CUDA for Qwen. Single Gradio process. Mirrors z-image-studio's proven pattern (see `~/Projects/llm/z-image-studio/`).

**Tech Stack:** Python 3.11 · Gradio 5.50 · ACE-Step (`clockworksquirrel/ace-step-apple-silicon` on Mac, `ace-step` from PyPI/git on CUDA) · Qwen 2.5 7B Instruct · Demucs htdemucs_ft · pyloudnorm · PEFT · safetensors · ruff · pytest. **MIT** licensed.

**Sole-author rule (NON-NEGOTIABLE):** Every commit lists Mayank Gupta as sole author. NO `Co-Authored-By: Claude`, NO `Generated with Claude Code` footer, NO `--author=…` flag. This is baked into `CLAUDE.md`, `AGENTS.md`, `SKILLS.md`, and every commit message in this plan.

**Conventional Commits**: `feat(scope): subject` / `fix(scope): subject` / `chore` / `docs` / `test` / `refactor` / `ci` / `perf`. Lowercase subject, no trailing period.

**Source spec:** [`../specs/2026-05-18-ace-music-studio-design.md`](../specs/2026-05-18-ace-music-studio-design.md) — read this first if context is needed.

**Reference repo:** `~/Projects/llm/z-image-studio/` — read `app.py`, `backend.py`, `theme.py`, `models.py`, `lora.py` for patterns to port.

---

## File structure

| File | Responsibility |
|---|---|
| `app.py` | Gradio Blocks entrypoint, `_bootstrap()`, mode event handlers, lazy backend accessor |
| `backend.py` | `ACEStepStudioBackend` singleton; `@spaces.GPU(duration=…)` wrappers; one `generate_with_retry` dispatch |
| `ace_pipeline.py` | `ACEStepPipeline` lifecycle, device autodetect, HF-cache → `./models/` symlink mirror |
| `lora_stack.py` | `sniff()`, `LoRAValidationError`, preset registry, PEFT adapter management |
| `lyrics_lm.py` | Qwen 2.5 7B lazy singleton, mlx-lm on Mac / transformers on CUDA, lyrics system prompt |
| `post_process.py` | Demucs separator (lazy), pyloudnorm normalize, mp3 export via ffmpeg |
| `modes.py` | 5 pure mode handlers: `generate`, `cover`, `extend`, `edit`, `lyrics`. Pure functions over `(backend, params) → (audio_path, meta_dict)` |
| `ui.py` | Per-tab component builders + sidebar + output panel |
| `theme.py` | Brutalist Mono palette tokens + responsive CSS + `build_theme()` |
| `tooltips.py` | Centralised `info=` strings |
| `presets/manifest.json` | 4 bundled LoRA presets (HF IDs + metadata) |
| `tests/` | L1 (no GPU), L2 (mocked pipeline), GPU smoke (`@pytest.mark.gpu`) |
| `requirements.txt` · `requirements-mac.txt` · `pyproject.toml` · `setup.sh` · `LICENSE` · `.gitignore` · `.github/workflows/ci.yml` | Project tooling |
| `README.md` · `AGENTS.md` · `CLAUDE.md` · `SKILLS.md` | Documentation. HF Space frontmatter lives in `README.md`. |

Every file has one responsibility. Lifting boundaries straight from z-image-studio.

---

## Pre-flight

- [ ] **P.1: Confirm working directory**

Run: `pwd`
Expected: `/Users/techfreakworm/Projects/llm/music-generator`

If not, `cd /Users/techfreakworm/Projects/llm/music-generator`.

- [ ] **P.2: Verify Python 3.11 is available**

Run: `python3.11 --version`
Expected: `Python 3.11.x`

If missing, `brew install python@3.11`.

- [ ] **P.3: Verify reference repo exists for pattern porting**

Run: `ls ~/Projects/llm/z-image-studio/app.py`
Expected: file path printed (no error).

---

## Part A — Repo skeleton

### Task A1: git init + LICENSE + .gitignore

**Files:**
- Create: `LICENSE`
- Create: `.gitignore`

- [ ] **Step 1: Initialise git**

```bash
cd /Users/techfreakworm/Projects/llm/music-generator
git init -b main
```

- [ ] **Step 2: Write MIT LICENSE**

Use the standard MIT text, year `2026`, copyright holder `Mayank Gupta`. Copy from `~/Projects/llm/z-image-studio/LICENSE` and update the year if needed.

```bash
cp ~/Projects/llm/z-image-studio/LICENSE ./LICENSE
sed -i '' 's/Copyright (c) [0-9]*/Copyright (c) 2026/' LICENSE
```

- [ ] **Step 3: Write `.gitignore`**

File contents:
```
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
.DS_Store
.superpowers/brainstorm/
models/
output/
hf-cache-rw/
*.safetensors
*.bin
*.pth
*.wav
*.mp3
*.flac
!presets/*.safetensors
```

The `!presets/*.safetensors` allows the bundled LoRA-preset metadata to ship in the repo if we ever vendor a small one.

- [ ] **Step 4: Initial commit**

```bash
git add LICENSE .gitignore
git commit -m "chore: init repo with mit license and gitignore"
```

### Task A2: pyproject.toml + ruff config

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "ace-music-studio"
version = "0.1.0"
description = "Open-source full-song generation studio wrapping ACE-Step 1.5 XL"
authors = [{ name = "Mayank Gupta" }]
license = { text = "MIT" }
readme = "README.md"
requires-python = ">=3.11,<3.12"

[tool.ruff]
line-length = 110
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]
ignore = ["E501"]  # we cap via formatter, not linter

[tool.pytest.ini_options]
addopts = "-m 'not gpu' --tb=short"
markers = ["gpu: requires a real GPU/MPS device; opt in with -m gpu"]
testpaths = ["tests"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyproject with ruff and pytest config"
```

### Task A3: setup.sh

**Files:**
- Create: `setup.sh`

- [ ] **Step 1: Write `setup.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .venv ]; then
  echo "Creating .venv (Python 3.11)…"
  python3.11 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip

# Detect platform
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "Apple Silicon detected — installing Mac requirements"
  pip install -r requirements.txt -r requirements-mac.txt
else
  echo "Non-Mac platform — installing CUDA-path requirements"
  pip install -r requirements.txt
fi

echo "Setup complete. Activate with: source .venv/bin/activate"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x setup.sh
```

- [ ] **Step 3: Commit**

```bash
git add setup.sh
git commit -m "chore: add setup.sh with mac vs cuda branching"
```

### Task A4: requirements.txt + requirements-mac.txt

**Files:**
- Create: `requirements.txt`
- Create: `requirements-mac.txt`

- [ ] **Step 1: Write `requirements.txt` (cross-platform)**

```
gradio>=5.50,<6
transformers>=4.45,<5
torch>=2.4
torchaudio>=2.4
safetensors>=0.4
peft>=0.13
demucs>=4.0
pyloudnorm>=0.1.1
soundfile>=0.12
librosa>=0.10
huggingface_hub>=0.25
numpy>=1.26,<2
# ACE-Step on CUDA — replaced on Mac via requirements-mac.txt
ace-step @ git+https://github.com/ace-step/ACE-Step-1.5.git
# Do NOT pin `spaces` — HF Spaces injects its own version
```

- [ ] **Step 2: Write `requirements-mac.txt` (Mac-only adds)**

```
mlx>=0.18
mlx-lm>=0.18
# Apple Silicon ACE-Step fork (overrides the line in requirements.txt at install time
# because pip resolves the last matching entry per package).
ace-step @ git+https://github.com/clockworksquirrel/ace-step-apple-silicon.git
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt requirements-mac.txt
git commit -m "chore: pin deps with mac vs cuda ace-step branching"
```

### Task A5: CLAUDE.md / AGENTS.md / SKILLS.md (sole-author rule + project guide)

**Files:**
- Create: `CLAUDE.md`
- Create: `AGENTS.md`
- Create: `SKILLS.md`

- [ ] **Step 1: Copy z-image-studio's files as starting points and rewrite for ACE Music Studio**

```bash
cp ~/Projects/llm/z-image-studio/CLAUDE.md ./CLAUDE.md
cp ~/Projects/llm/z-image-studio/AGENTS.md ./AGENTS.md
cp ~/Projects/llm/z-image-studio/SKILLS.md ./SKILLS.md
```

- [ ] **Step 2: Edit each file**

For each of `CLAUDE.md`, `AGENTS.md`, `SKILLS.md`:
- Replace every "Z-Image Studio" → "ACE Music Studio"
- Replace every "z-image-studio" → "ace-music-studio"
- Replace the DiffSynth / Z-Image references with ACE-Step / Qwen 2.5 / Demucs equivalents
- Update the "Gotchas we already paid for" section in `CLAUDE.md` to:
  - Remove Z-Image-specific items (DiffSynth `from_pretrained` discard bug, etc.)
  - Keep MPS gotchas (`PYTORCH_ENABLE_MPS_FALLBACK=1`, no `torch.mps.mem_get_info`)
  - Add ACE-Step-specific gotchas as we discover them (placeholder for now)
- Keep the **sole-author rule** at the top of `CLAUDE.md` verbatim. Critical.
- Update `## Out of scope for v1` to match §13 of the spec

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md AGENTS.md SKILLS.md
git commit -m "docs: add claude/agents/skills guides with sole-author rule"
```

### Task A6: README.md (HF Space frontmatter + GitHub badges + intro)

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
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
  - ACE-Step/ACE-Step-v1.5-XL-SFT *.safetensors,config.json,scheduler/*,vae/*,tokenizer/*
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add readme with hf space frontmatter and quick start"
```

### Task A7: tests directory + conftest

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create empty `tests/__init__.py`**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures.

The default pytest config (pyproject.toml) skips tests marked `gpu`. Opt in
to GPU smoke tests with `pytest -m gpu`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable as flat modules (matches z-image-studio convention).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _silence_mps_fallback_env(monkeypatch):
    """L1+L2 tests don't touch torch/mps; clear the env so test logs are clean."""
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
```

- [ ] **Step 3: Verify pytest collects (zero tests, exit 0 or 5)**

```bash
source .venv/bin/activate || python3.11 -m venv .venv && source .venv/bin/activate
pip install pytest
pytest tests/ -v
```

Expected: `no tests ran in 0.XX s` (exit code 5 is fine; just confirms pytest sees the directory).

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add pytest scaffolding"
```

### Task A8: CI workflow (L1+L2, no GPU)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write CI**

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install deps (CPU-only, skip ace-step + mlx)
        run: |
          python -m pip install --upgrade pip
          pip install ruff pytest gradio safetensors peft pyloudnorm numpy
      - name: Lint
        run: |
          ruff format --check .
          ruff check .
      - name: Tests (L1+L2, no GPU)
        run: pytest -m "not gpu" --tb=short
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add github actions for ruff + pytest l1+l2"
```

---

## Part B — M0: Bootstrap (skeleton boots end-to-end)

**Goal of M0:** Empty Gradio app boots on M5 Max, shows the Brutalist Mono header + CTA + 5 empty tabs + sidebar. No model loading yet. Validates device autodetect + theme + Gradio plumbing.

### Task B1: `ace_pipeline.py` — device autodetect (no model load yet)

**Files:**
- Create: `ace_pipeline.py`
- Create: `tests/test_ace_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ace_pipeline.py`:
```python
"""L1 tests for device autodetect — no torch needed if we mock importlib."""
from __future__ import annotations

import ace_pipeline as ap


def test_detect_device_returns_one_of_cuda_mps_cpu():
    device = ap.detect_device()
    assert device in {"cuda", "mps", "cpu"}


def test_vram_limit_for_mps_is_none():
    """MPS has no torch.mps.mem_get_info; return None so DiffSynth-style gates
    short-circuit instead of crashing (z-image-studio paid this debug cycle)."""
    assert ap.vram_limit_for("mps") is None


def test_vram_limit_for_cpu_is_none():
    assert ap.vram_limit_for("cpu") is None
```

- [ ] **Step 2: Run test (expect failure: `ace_pipeline` not found)**

```bash
pytest tests/test_ace_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'ace_pipeline'`.

- [ ] **Step 3: Write `ace_pipeline.py`**

```python
"""ACE-Step pipeline lifecycle: device autodetect, lazy load, cache mirror.

Mirrors z-image-studio's `models.py` pattern. M0 only implements device
detection — the pipeline class itself is filled in at M1.
"""
from __future__ import annotations


def detect_device() -> str:
    """Returns 'cuda', 'mps', or 'cpu' in priority order."""
    try:
        import torch  # local import: keep module import cheap for CI
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    # macOS: torch.backends.mps appeared in 2.0; guard for the rare absence
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def vram_limit_for(device: str) -> int | None:
    """Returns a VRAM cap in bytes for CUDA, None otherwise.

    `torch.mps` has no `mem_get_info` — calling DiffSynth-style free-VRAM
    gates with a numeric limit would crash on MPS. Returning None lets the
    pipeline short-circuit those checks.
    """
    if device != "cuda":
        return None
    try:
        import torch
        free, _total = torch.cuda.mem_get_info()
        # Leave 2 GiB headroom for activations
        return max(0, free - 2 * 1024 ** 3)
    except Exception:
        return None
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_ace_pipeline.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add ace_pipeline.py tests/test_ace_pipeline.py
git commit -m "feat(pipeline): add device autodetect with mps-safe vram limit"
```

### Task B2: `theme.py` — Brutalist Mono tokens + base CSS

**Files:**
- Create: `theme.py`
- Create: `tests/test_theme.py`

- [ ] **Step 1: Write the failing test**

`tests/test_theme.py`:
```python
"""L1 theme assertions — palette tokens, CSS presence."""
from __future__ import annotations

import theme


def test_palette_tokens_are_brutalist_mono():
    assert theme.BG == "#0A0A0A"
    assert theme.INK == "#E5E5E5"
    assert theme.PRIMARY == "#FFFFFF"
    # No color accent — that's the whole point of Brutalist Mono
    for v in (theme.BG, theme.SURFACE, theme.SURFACE_STRONG, theme.BORDER, theme.BORDER_STRONG, theme.INK, theme.INK_MUTED, theme.PRIMARY):
        assert v.startswith("#")
        assert len(v) == 7  # all hex, no rgba


def test_css_contains_responsive_breakpoints():
    css = theme.CSS
    assert "@media" in css
    assert "1024px" in css   # tablet breakpoint
    assert "640px" in css    # mobile breakpoint


def test_build_theme_returns_gradio_theme():
    import gradio as gr
    t = theme.build_theme()
    # gr.themes.Base is the parent class
    assert isinstance(t, gr.themes.Base)
```

- [ ] **Step 2: Run test (expect failure)**

```bash
pytest tests/test_theme.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `theme.py`**

```python
"""Brutalist Mono — pure black/white, no color accent.

Palette tokens are the source of truth; CSS pulls from them. The audio
waveform is the only optionally-colored element (rendered white in v1).
"""
from __future__ import annotations

import gradio as gr

# --- Palette ----------------------------------------------------------------
BG = "#0A0A0A"
SURFACE = "#141414"
SURFACE_STRONG = "#000000"
BORDER = "#1F1F1F"
BORDER_STRONG = "#2A2A2A"
INK = "#E5E5E5"
INK_MUTED = "#6B6B6B"
PRIMARY = "#FFFFFF"
ERROR_BG = "#1A1A1A"      # toast background; ink stays white
RADIUS = "6px"
FONT_STACK = (
    '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'
)


def build_theme() -> gr.themes.Base:
    """Returns a Gradio theme keyed to Brutalist Mono tokens.

    Gradio themes accept tokens by name; we map only the ones that affect the
    body chrome. Per-component CSS lives in `CSS` below.
    """
    return gr.themes.Base(
        primary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill=BG,
        body_text_color=INK,
        block_background_fill=SURFACE,
        block_border_color=BORDER,
        block_border_width="1px",
        block_radius=RADIUS,
        input_background_fill=SURFACE_STRONG,
        input_border_color=BORDER_STRONG,
        input_border_color_focus=PRIMARY,
        button_primary_background_fill=PRIMARY,
        button_primary_text_color=BG,
        button_primary_background_fill_hover=PRIMARY,
        button_secondary_background_fill=SURFACE_STRONG,
        button_secondary_text_color=INK,
        button_secondary_border_color=BORDER_STRONG,
    )


CSS = f"""
/* --- Sole brand bits --------------------------------------------------- */
.ams-header {{
  display:flex; justify-content:space-between; align-items:baseline;
  padding:8px 0 4px 0;
}}
.ams-brand {{
  font-size:16px; font-weight:600; letter-spacing:-0.01em; color:{INK};
}}
.ams-brand-period {{ color:{PRIMARY}; }}
.ams-status {{ font-size:11px; color:{INK_MUTED}; letter-spacing:0.02em; }}

/* --- CTA banner -------------------------------------------------------- */
.ams-cta {{
  font-size:13px; color:{INK_MUTED}; margin:2px 0 12px 0; padding-bottom:10px;
  border-bottom:1px solid {BORDER};
}}
.ams-cta strong {{ color:{INK}; }}
.ams-cta-heart {{ color:{PRIMARY}; }}
.ams-cta a {{ color:{INK}; text-decoration:underline; }}

/* --- Sidebar nav (desktop >= 1024) ------------------------------------ */
.ams-sidebar {{ background:{SURFACE_STRONG}; padding:14px 10px; border-radius:{RADIUS}; min-width:170px; }}
.ams-side-item {{
  display:block; padding:8px 10px; border-radius:4px; margin-bottom:3px;
  font-size:13px; color:{INK_MUTED}; cursor:pointer; text-decoration:none;
}}
.ams-side-item.active {{
  background:#1A1A1A; color:{PRIMARY};
  border-left:2px solid {PRIMARY}; padding-left:8px;
}}

/* --- LoRA chip pill --------------------------------------------------- */
.ams-chip {{
  display:inline-block; padding:5px 10px; border-radius:14px;
  font-size:11px; margin:0 5px 5px 0; background:{SURFACE_STRONG};
  border:1px solid {BORDER_STRONG}; color:{INK_MUTED}; cursor:pointer;
}}
.ams-chip.on {{ border-color:{PRIMARY}; color:{PRIMARY}; }}
.ams-chip.upload {{ border-style:dashed; color:{PRIMARY}; }}

/* --- LoRA file drop zone (tighten Gradio default ~400px height) ------ */
.ams-lora-file .upload-container {{ min-height:56px !important; }}

/* --- Hide Gradio footer ----------------------------------------------- */
footer {{ display:none !important; }}

/* --- Responsive: tablet 640-1024 px ----------------------------------- */
@media (max-width: 1024px) {{
  .ams-sidebar {{ min-width:34px; padding:6px 4px; }}
  .ams-side-item {{ font-size:0; padding:6px; }}
  .ams-side-item::first-letter {{ font-size:16px; }}  /* show only the emoji */
}}

/* --- Responsive: mobile < 640 px -------------------------------------- */
@media (max-width: 640px) {{
  .ams-sidebar {{ display:none; }}
  .ams-cta {{ font-size:11px; }}
}}
"""
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pip install gradio safetensors peft pyloudnorm
pytest tests/test_theme.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add theme.py tests/test_theme.py
git commit -m "feat(ui): add brutalist mono theme tokens and responsive css"
```

### Task B3: `app.py` — Gradio Blocks skeleton with header / CTA / sidebar / 5 mode panes

**⚠ READ FIRST: the `WIREFRAME COMPLIANCE` section between Part G and Part H of this plan.** This task originally specified `gr.Tabs` and was corrected on 2026-05-18 (commit `59b9fee`). The correct pattern is sidebar nav + visibility-toggled `gr.Group` panes, NOT top tabs.

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write `app.py`**

```python
"""ACE Music Studio — Gradio entrypoint.

UI ARCHITECTURE (locked — read this before editing):

The five "modes" (Generate / Cover / Extend / Edit / Lyrics) are NOT
implemented via ``gr.Tabs``. The wireframes at
``docs/superpowers/specs/mockups/`` show a LEFT sidebar with mode pills +
a session History section, and a single content column on the right.

The implementation pattern is documented in the WIREFRAME COMPLIANCE
section of the plan. Summary:

  gr.Row(elem_classes=["ams-body"])
  ├── gr.Column(min_width=190, elem_classes=["ams-sidebar"])
  │   ├── gr.Radio(label=None, elem_classes=["ams-side-radio"])
  │   └── gr.HTML(HISTORY_HTML)
  └── gr.Column(elem_classes=["ams-content"])
      ├── gr.Group(visible=True)   ← pane_generate
      ├── gr.Group(visible=False)  ← pane_cover
      ├── gr.Group(visible=False)  ← pane_extend
      ├── gr.Group(visible=False)  ← pane_edit
      └── gr.Group(visible=False)  ← pane_lyrics

DO NOT switch this back to ``gr.Tabs``.
"""

from __future__ import annotations

import os

# Set MPS fallback BEFORE any torch import path is taken.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Don't pin HF download source — let HF default for both Spaces and local cache.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import gradio as gr

import ace_pipeline
import theme


def _status_html(device: str) -> str:
    """Right-aligned status indicator in the header (e.g. READY · MPS)."""
    return f"""
<div class="ams-header">
  <div>
    <div class="ams-brand">ACE Music Studio<span class="ams-brand-period">.</span></div>
  </div>
  <div class="ams-status">ready · {device.upper()}</div>
</div>
""".strip()


CTA_HTML = """
<div class="ams-cta">
  Built with <span class="ams-cta-heart">♥</span>.
  <strong>Drop a like</strong> at the top
  &nbsp;·&nbsp;
  Follow <a href="https://huggingface.co/techfreakworm" target="_blank" rel="noopener noreferrer"><strong>@techfreakworm</strong></a>
  for what's next.
</div>
""".strip()


HISTORY_HTML = """
<div class="ams-history">
  <div class="ams-history-title">History · session</div>
  <div class="ams-history-empty">No generations yet</div>
</div>
""".strip()


MODE_CHOICES = [
    ("🎵 Generate", "generate"),
    ("🎤 Cover", "cover"),
    ("⏩ Extend", "extend"),
    ("✏️ Edit", "edit"),
    ("✍️ Lyrics", "lyrics"),
]


def _bootstrap() -> None:
    """HF Spaces: mirror read-only preload cache into a writable tree.

    Local Mac/CUDA: no-op. Implemented at M7 when we wire deployment.
    """
    pass


def build_app() -> gr.Blocks:
    device = ace_pipeline.detect_device()

    with gr.Blocks(theme=theme.build_theme(), css=theme.CSS, title="ACE Music Studio") as demo:
        gr.HTML(_status_html(device))
        gr.HTML(CTA_HTML)

        with gr.Row(elem_classes=["ams-body"]):
            # --- Sidebar ----------------------------------------------------
            with gr.Column(scale=0, min_width=190, elem_classes=["ams-sidebar"]):
                mode = gr.Radio(
                    choices=MODE_CHOICES,
                    value="generate",
                    label=None,
                    show_label=False,
                    container=False,
                    elem_classes=["ams-side-radio"],
                )
                gr.HTML(HISTORY_HTML)

            # --- Content ----------------------------------------------------
            with gr.Column(scale=10, elem_classes=["ams-content"]):
                with gr.Group(visible=True, elem_classes=["ams-tab-pane"]) as pane_generate:
                    gr.Markdown("### 🎵 Generate\n\nPlaceholder — implemented in M1.")
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_cover:
                    gr.Markdown("### 🎤 Cover\n\nPlaceholder — implemented in M3.")
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_extend:
                    gr.Markdown("### ⏩ Extend\n\nPlaceholder — implemented in M3.")
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_edit:
                    gr.Markdown("### ✏️ Edit\n\nPlaceholder — implemented in M3.")
                with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_lyrics:
                    gr.Markdown("### ✍️ Lyrics\n\nPlaceholder — implemented in M4.")

        panes = [pane_generate, pane_cover, pane_extend, pane_edit, pane_lyrics]

        def _switch_pane(selected: str):
            order = ["generate", "cover", "extend", "edit", "lyrics"]
            return tuple(gr.Group(visible=(selected == name)) for name in order)

        mode.change(fn=_switch_pane, inputs=mode, outputs=panes)

    return demo


if __name__ == "__main__":
    _bootstrap()
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
```

- [ ] **Step 2: Verify app boots**

```bash
source .venv/bin/activate
# Skip ace-step install for M0 — only need gradio + theme deps
pip install gradio
python app.py
```

Open `http://127.0.0.1:7860` in a browser. Expected:
- Dark background (#0A0A0A)
- White brand text "ACE Music Studio."
- CTA banner: "Built with ♥. Drop a like at the top · Follow @techfreakworm for what's next."
- 5 tabs visible
- Each tab shows a placeholder
- No Gradio footer

Kill with `Ctrl-C` when verified.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): bootstrap gradio blocks with brutalist mono chrome"
```

### M0 verification gate

- [ ] **G.M0.1:** `python app.py` boots without errors on M5 Max
- [ ] **G.M0.2:** **Visual check** — open `http://127.0.0.1:7860` and confirm:
  - Header row: "ACE Music Studio." on the left, "READY · MPS" (or CPU if torch not yet installed) on the right
  - CTA below: "Built with ♥. **Drop a like** at the top · Follow **@techfreakworm** for what's next."
  - **LEFT sidebar** (dark, ~190 px wide) with 5 mode pills, "🎵 Generate" highlighted (white text + white left border + dark hover bg)
  - "HISTORY · SESSION" section below the mode pills, "No generations yet" italic placeholder
  - **RIGHT content pane** (dark surface, bordered) showing "🎵 Generate / Placeholder — implemented in M1."
  - Click "🎤 Cover" → content swaps to Cover pane, active highlight moves to Cover
  - If the layout shows top tabs instead of a left sidebar, the implementation is wrong — see the WIREFRAME COMPLIANCE section.
- [ ] **G.M0.3:** `pytest -m "not gpu"` shows 6 tests passing (3 theme + 3 ace_pipeline)
- [ ] **G.M0.4:** `ruff format --check .` and `ruff check .` both pass
- [ ] **G.M0.5:** Tag the milestone: `git tag m0` (no push yet)

---

## Part C — M1: Generate mode end-to-end (no LoRA)

**Goal:** Click "Generate" on the Generate tab → ACE-Step pipeline produces a real WAV file. No LoRA stacking, no Demucs, no Lyrics LM. Validates the inference path on M5 Max.

### Task C1: `ace_pipeline.py` — `ACEStepStudio` lazy wrapper

**⚠ API CORRECTION (commit `99375d0`, 2026-05-18).** The plan originally assumed `from ace_step import ACEStepPipeline` with a `from_pretrained` entry point. That API does **NOT** exist in the installed `acestep` package (both upstream and the Apple-Silicon fork). The real API is:

```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

dit = AceStepHandler()
dit.initialize_service(project_root=..., config_path="acestep-v15-xl-sft", device="mps")
lm  = LLMHandler()
lm.initialize(checkpoint_dir=..., lm_model_path="acestep-5Hz-lm-0.6B", backend="vllm", device="mps")
result = generate_music(dit, lm, GenerationParams(...), GenerationConfig(...))
# result.audios[0]["path"] is the WAV file
```

To keep `backend.py` and `modes.py` clean, `ace_pipeline.py` wraps both handlers in a single `ACEStepStudio` class exposing `generate(params: dict) -> str`. `get_pipeline()` returns the lazy singleton wrapper. Module name is `acestep` (no underscore) — not `ace_step`. Two HF model paths needed: `ACE-Step/acestep-v15-xl-sft` (DiT, ~16 GB) + `ACE-Step/acestep-5Hz-lm-0.6B` (LM planner, ~1.4 GB), placed under `./checkpoints/<config>/`. Read the committed `ace_pipeline.py` for the actual implementation; the code block below this header is the ORIGINAL plan version and is kept only for historical context.

**Files:**
- Modify: `ace_pipeline.py`
- Create: `tests/test_ace_pipeline_lazy.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ace_pipeline_lazy.py`:
```python
"""L2 tests for pipeline lazy load — mock the heavy ACE-Step import."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import ace_pipeline as ap


def test_get_pipeline_loads_lazily_first_call_only(monkeypatch):
    fake_pipe = MagicMock(name="fake_ace_pipeline")
    loader = MagicMock(return_value=fake_pipe)
    monkeypatch.setattr(ap, "_load_pipeline", loader)
    monkeypatch.setattr(ap, "_PIPELINE", None, raising=False)

    p1 = ap.get_pipeline()
    p2 = ap.get_pipeline()

    assert p1 is fake_pipe
    assert p2 is fake_pipe
    assert loader.call_count == 1, "pipeline should load exactly once"


def test_get_pipeline_uses_detected_device(monkeypatch):
    monkeypatch.setattr(ap, "_PIPELINE", None, raising=False)
    monkeypatch.setattr(ap, "detect_device", lambda: "mps")
    captured = {}
    def fake_load(device, model_path):
        captured["device"] = device
        captured["model_path"] = model_path
        return MagicMock()
    monkeypatch.setattr(ap, "_load_pipeline", fake_load)

    ap.get_pipeline()

    assert captured["device"] == "mps"
    assert captured["model_path"] is not None
```

- [ ] **Step 2: Run test (expect failure: `get_pipeline` not defined)**

```bash
pytest tests/test_ace_pipeline_lazy.py -v
```

- [ ] **Step 3: Extend `ace_pipeline.py`**

Add to the bottom of `ace_pipeline.py`:

```python
import os
from pathlib import Path

_PIPELINE = None  # module-level lazy singleton
_DEFAULT_MODEL_ID = "ACE-Step/ACE-Step-v1.5-XL-SFT"


def _load_pipeline(device: str, model_path: str):
    """Construct the ACE-Step pipeline. Heavy import is local so unit tests can mock."""
    from ace_step import ACEStepPipeline  # type: ignore[import-not-found]

    # On Mac, the apple-silicon fork sets dtype + backend automatically.
    # On CUDA we pass bf16 explicitly.
    if device == "cuda":
        pipe = ACEStepPipeline.from_pretrained(model_path, torch_dtype="bf16")
    else:
        pipe = ACEStepPipeline.from_pretrained(model_path)

    pipe.to(device)
    return pipe


def get_pipeline():
    """Lazy-load the ACE-Step pipeline once per process."""
    global _PIPELINE
    if _PIPELINE is None:
        device = detect_device()
        model_path = os.environ.get("ACE_MODEL_PATH", _DEFAULT_MODEL_ID)
        _PIPELINE = _load_pipeline(device, model_path)
    return _PIPELINE
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_ace_pipeline_lazy.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add ace_pipeline.py tests/test_ace_pipeline_lazy.py
git commit -m "feat(pipeline): lazy ace-step singleton with device-aware load"
```

### Task C2: `modes.py` — `generate()` pure handler

**Files:**
- Create: `modes.py`
- Create: `tests/test_modes_generate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_modes_generate.py`:
```python
"""L2 tests for the generate mode handler — backend is mocked at the pipeline boundary."""
from __future__ import annotations

from unittest.mock import MagicMock

import modes


def test_generate_validates_prompt_required():
    backend = MagicMock()
    with pytest_raises_value("prompt"):
        modes.generate(backend, params={"prompt": "", "lyrics": "[verse] x", "duration_s": 10})


def test_generate_passes_params_to_backend(monkeypatch):
    backend = MagicMock()
    backend.dispatch.return_value = ("/tmp/audio.wav", {"seed": 42})
    out_path, meta = modes.generate(
        backend,
        params={
            "prompt": "psytrance",
            "lyrics": "[verse] x",
            "duration_s": 30,
            "instrumental": False,
            "seed": 42,
        },
    )

    assert out_path == "/tmp/audio.wav"
    assert meta["seed"] == 42
    backend.dispatch.assert_called_once()
    call_kwargs = backend.dispatch.call_args.kwargs
    assert call_kwargs["mode"] == "generate"
    # Cover-style params must be absent for the generate mode
    assert "audio_cover_strength" not in call_kwargs["params"]


def pytest_raises_value(field_name):
    import pytest
    return pytest.raises(ValueError, match=field_name)
```

- [ ] **Step 2: Run test (expect failure)**

```bash
pytest tests/test_modes_generate.py -v
```

- [ ] **Step 3: Write `modes.py`**

```python
"""Pure mode handlers — one function per generation mode.

Each handler validates inputs, builds the ACE-Step kwargs for its mode, and
hands off to `backend.dispatch(...)`. Backend ownership of @spaces.GPU and
pipeline lifecycle keeps these handlers cheap to test.
"""
from __future__ import annotations

from typing import Any


def _require(params: dict[str, Any], field: str) -> Any:
    v = params.get(field)
    if v is None or (isinstance(v, str) and not v.strip()):
        raise ValueError(f"Missing required field: {field}")
    return v


def generate(backend, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Text → song. Vocals + instruments in one stream."""
    prompt = _require(params, "prompt")
    lyrics = params.get("lyrics", "")
    duration_s = int(params.get("duration_s", 30))
    instrumental = bool(params.get("instrumental", False))

    return backend.dispatch(
        mode="generate",
        params={
            "prompt": prompt,
            "lyrics": lyrics,
            "duration_s": duration_s,
            "instrumental": instrumental,
            "seed": params.get("seed"),
            "loras": params.get("loras", []),
            "advanced": params.get("advanced", {}),
            "lm": params.get("lm", {}),
            "dcw": params.get("dcw", {}),
        },
    )
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_modes_generate.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add modes.py tests/test_modes_generate.py
git commit -m "feat(modes): add generate mode handler with input validation"
```

### Task C3: `backend.py` — `ACEStepStudioBackend` with `dispatch()`

**Files:**
- Create: `backend.py`
- Create: `tests/test_backend.py`

- [ ] **Step 1: Write the failing test**

`tests/test_backend.py`:
```python
"""L2 tests for backend.dispatch — pipeline is mocked at the boundary."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import backend as be


def test_dispatch_generate_calls_pipeline_with_expected_kwargs(monkeypatch, tmp_path):
    fake_pipe = MagicMock()
    fake_out = tmp_path / "out.wav"
    fake_out.write_bytes(b"RIFF" + b"\0" * 1000)
    fake_pipe.return_value = str(fake_out)

    monkeypatch.setattr("ace_pipeline.get_pipeline", lambda: fake_pipe)

    b = be.ACEStepStudioBackend()
    out_path, meta = b.dispatch(
        mode="generate",
        params={
            "prompt": "psytrance",
            "lyrics": "[verse]",
            "duration_s": 10,
            "instrumental": False,
            "seed": 42,
            "loras": [],
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )

    assert out_path == str(fake_out)
    assert meta["mode"] == "generate"
    assert meta["seed"] == 42
    fake_pipe.assert_called_once()


def test_dispatch_random_seed_if_zero(monkeypatch, tmp_path):
    fake_pipe = MagicMock(return_value=str(tmp_path / "x.wav"))
    monkeypatch.setattr("ace_pipeline.get_pipeline", lambda: fake_pipe)
    (tmp_path / "x.wav").write_bytes(b"RIFF")

    b = be.ACEStepStudioBackend()
    _, meta = b.dispatch(
        mode="generate",
        params={"prompt": "p", "lyrics": "", "duration_s": 5, "instrumental": False, "seed": 0,
                "loras": [], "advanced": {}, "lm": {}, "dcw": {}},
    )

    assert 1 <= meta["seed"] <= 2_147_483_647
```

- [ ] **Step 2: Run test (expect failure)**

```bash
pytest tests/test_backend.py -v
```

- [ ] **Step 3: Write `backend.py`**

```python
"""ACEStepStudioBackend — dispatch + ZeroGPU lifetime + duration estimator.

Off Spaces, @spaces.GPU is a no-op identity decorator (`spaces` may not be
installed locally). On Spaces, the HF runtime injects it at startup and
the decorator applies for real.
"""
from __future__ import annotations

import random
import time
from typing import Any

try:
    import spaces  # type: ignore[import-not-found]
    _HAS_SPACES = True
except ImportError:  # pragma: no cover - covered by manual local testing
    spaces = None
    _HAS_SPACES = False

import ace_pipeline as ap


def _maybe_seed(seed: int | None) -> int:
    if seed and int(seed) > 0:
        return int(seed)
    return random.randint(1, 2_147_483_647)


def _duration_estimate(mode: str, params: dict[str, Any]) -> int:
    """ZeroGPU per-call duration cap, clamped [60, 180] s."""
    base = 60
    duration_s = int(params.get("duration_s", 30) or 30)
    if duration_s > 60:
        base = 90
    if duration_s > 120:
        base = 120
    if mode == "edit":
        base = max(base, 90)
    if mode == "extend":
        base = max(base, 120)
    return min(180, max(60, base))


def _gpu_decorate(fn):
    """Wrap the inference function with @spaces.GPU on Spaces; passthrough off."""
    if not _HAS_SPACES:
        return fn
    # The decorator accepts a callable duration; per-call we re-evaluate.
    return spaces.GPU(duration=180)(fn)  # outer cap; per-call estimate logged in meta


class ACEStepStudioBackend:
    """Lazy backend singleton. Owns @spaces.GPU and pipeline lifecycle."""

    def dispatch(self, mode: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        params = dict(params)
        params["seed"] = _maybe_seed(params.get("seed"))
        t0 = time.time()
        pipe = ap.get_pipeline()
        out_path = self._call_pipe_for_mode(pipe, mode, params)
        meta = {
            "mode": mode,
            "seed": params["seed"],
            "duration_s": params.get("duration_s"),
            "wall_seconds": round(time.time() - t0, 2),
            "estimated_duration_s": _duration_estimate(mode, params),
            "loras": [
                {"name": l.get("name"), "scale": l.get("scale"), "sha256": l.get("sha256")}
                for l in params.get("loras", [])
            ],
            "lm": params.get("lm", {}),
            "dcw": params.get("dcw", {}),
        }
        return out_path, meta

    def _call_pipe_for_mode(self, pipe, mode: str, params: dict[str, Any]) -> str:
        """Mode-specific kwargs translation. Filled out per milestone."""
        if mode == "generate":
            return pipe(
                prompt=params["prompt"],
                lyrics=params.get("lyrics", ""),
                duration_s=params["duration_s"],
                instrumental=params.get("instrumental", False),
                seed=params["seed"],
                # Generate mode never uses audio_cover_strength
            )
        # cover / extend / edit / lyrics get filled in at M3 / M4
        raise NotImplementedError(f"Mode {mode!r} is not wired yet")
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_backend.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend.py tests/test_backend.py
git commit -m "feat(backend): add ace-step studio backend with dispatch + zerogpu wrap"
```

### Task C4: `ui.py` — Generate tab builder (no LoRA chip wiring yet)

**Files:**
- Create: `ui.py`

- [ ] **Step 1: Write `ui.py` with `build_generate_tab()`**

```python
"""Per-tab Gradio component builders + shared output panel.

Each builder returns a dict of components keyed by purpose so app.py wires
events without depending on Gradio's positional return order.
"""
from __future__ import annotations

import gradio as gr


def build_generate_tab() -> dict[str, gr.components.Component]:
    """Generate tab: prompt + lyrics + duration + vocal mode + (LoRA/Advanced/LM/DCW added later)."""
    components: dict[str, gr.components.Component] = {}

    with gr.Row():
        with gr.Column(scale=13):
            components["prompt"] = gr.Textbox(
                label="Style prompt",
                placeholder="psytrance, rolling triplet bassline, acid squelch, metallic leads",
                lines=2,
            )
            components["lyrics"] = gr.Textbox(
                label="Lyrics",
                placeholder="[intro] atmospheric pads\n[verse] ...",
                lines=6,
                info="Use [verse] [chorus] [bridge] tags. Open the Lyrics tab to draft with Qwen 2.5.",
            )
            with gr.Row():
                components["duration_s"] = gr.Slider(
                    minimum=5, maximum=240, step=5, value=30,
                    label="Duration (s)",
                )
                components["instrumental"] = gr.Radio(
                    choices=["With vocals", "Instrumental"],
                    value="With vocals",
                    label="Vocal mode",
                )
            components["generate_btn"] = gr.Button("▶ Generate", variant="primary")

        with gr.Column(scale=10):
            components["output_audio"] = gr.Audio(
                label="Output",
                type="filepath",
                interactive=False,
            )
            components["output_meta"] = gr.Code(
                label="Metadata",
                language="json",
            )

    return components
```

- [ ] **Step 2: Smoke test by importing it**

```bash
python -c "import ui; print(dir(ui))"
```

Expected: `['build_generate_tab', ...]` printed; no errors.

- [ ] **Step 3: Commit**

```bash
git add ui.py
git commit -m "feat(ui): add generate tab builder with prompt/lyrics/duration/vocal mode"
```

### Task C5: Wire Generate tab into `app.py`

**⚠ Read the WIREFRAME COMPLIANCE section first.** Mode nav is `gr.Radio` + `gr.Group` panes — never `gr.Tabs`.

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the click handler and replace the Generate pane's Markdown placeholder with the real form**

Edit `app.py`. Add these helpers near the top (after imports):

```python
import random

import backend as be
import modes
import ui

_BACKEND: be.ACEStepStudioBackend | None = None


def get_backend() -> be.ACEStepStudioBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = be.ACEStepStudioBackend()
    return _BACKEND


def on_generate_click(
    prompt: str,
    lyrics: str,
    duration_s: float,
    instrumental_label: str,
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    try:
        out_path, meta = modes.generate(
            get_backend(),
            params={
                "prompt": prompt,
                "lyrics": lyrics,
                "duration_s": int(duration_s),
                "instrumental": instrumental_label == "Instrumental",
                "seed": random.randint(1, 2_147_483_647),
                "loras": [],
                "advanced": {},
                "lm": {},
                "dcw": {},
            },
        )
    except ValueError as e:
        raise gr.Error(str(e)) from e
    return out_path, meta
```

Inside `build_app()`, replace the existing `pane_generate` block's `gr.Markdown` placeholder with the real form. The rest of `build_app()` (HEADER / CTA / `ams-body` row / sidebar / `mode.change(_switch_pane, ...)`) stays exactly as-is. Pattern:

```python
with gr.Group(visible=True, elem_classes=["ams-tab-pane"]) as pane_generate:
    g = ui.build_generate_tab()
    g["generate_btn"].click(
        fn=on_generate_click,
        inputs=[g["prompt"], g["lyrics"], g["duration_s"], g["instrumental"]],
        outputs=[g["output_audio"], g["output_meta"]],
    )
```

Cover / Extend / Edit / Lyrics panes still keep their `gr.Markdown` placeholders for now (filled in at M3 / M4). The sidebar / pane-swap wiring is unchanged.

- [ ] **Step 2: Install ACE-Step on Mac**

```bash
pip install -r requirements-mac.txt
```

This pulls the Apple Silicon fork. First install will take ~5 min.

- [ ] **Step 3: Boot and manually generate**

```bash
python app.py
```

In the browser at `http://127.0.0.1:7860`:
1. Pick the Generate tab.
2. Prompt: `psytrance, rolling triplet bassline, acid squelch`
3. Lyrics: `[intro] atmospheric pads`
4. Duration: 10 s (keep short for first smoke)
5. Click **▶ Generate**.
6. Watch the progress bar. First request will be slow (~60-90 s cold load + ~10-20 s inference).

Expected: An audio player appears with a playable WAV. Metadata JSON shows `mode: "generate"`, `seed`, `wall_seconds`.

If the pipeline can't load weights, see Troubleshooting below.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): wire generate tab end-to-end to ace-step backend"
```

### Task C6: GPU smoke test for Generate

**Files:**
- Create: `tests/test_smoke_gpu.py`

- [ ] **Step 1: Write `tests/test_smoke_gpu.py`**

```python
"""GPU smoke tests — skipped by default. Opt in with: pytest -m gpu

Generates the minimum-viable songs end-to-end. Run before each release.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu


def test_generate_minimum_song(tmp_path):
    os.environ.setdefault("ACE_MODEL_PATH", "ACE-Step/ACE-Step-v1.5-XL-SFT")

    from backend import ACEStepStudioBackend

    b = ACEStepStudioBackend()
    out_path, meta = b.dispatch(
        mode="generate",
        params={
            "prompt": "test tone, simple drone",
            "lyrics": "[intro] tone",
            "duration_s": 5,
            "instrumental": True,
            "seed": 1,
            "loras": [],
            "advanced": {},
            "lm": {},
            "dcw": {},
        },
    )
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 0
    assert meta["mode"] == "generate"
    assert meta["seed"] == 1
```

- [ ] **Step 2: Run the GPU smoke**

```bash
pytest -m gpu tests/test_smoke_gpu.py::test_generate_minimum_song -v
```

Expected: PASS (warm-up will take ~60s + ~10s inference on M5 Max).

If it fails, debug interactively before continuing. Common failures:
- `flash-attn` import error → check the apple-silicon fork is installed, not stock ace-step
- MPS op missing → confirm `PYTORCH_ENABLE_MPS_FALLBACK=1` is set
- HF cache miss → `huggingface-cli login` if model is gated; otherwise check disk space

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_gpu.py
git commit -m "test(smoke): add gpu smoke for minimum generate path"
```

### M1 verification gate

- [ ] **G.M1.1:** `python app.py` boots + generates a 10 s clip locally
- [ ] **G.M1.2:** `pytest -m "not gpu"` → all L1+L2 pass
- [ ] **G.M1.3:** `pytest -m gpu tests/test_smoke_gpu.py` → passes on M5 Max
- [ ] **G.M1.4:** Wall-time for 10 s clip ≤ 30 s after warm-up on M5 Max (sanity check on §3.3's projections)
- [ ] **G.M1.5:** `ruff format --check . && ruff check .` pass
- [ ] **G.M1.6:** Tag: `git tag m1`

### Troubleshooting (M1 specific)

- **"Pipeline returns dict not path"** — adapt `_call_pipe_for_mode` in `backend.py` to extract the WAV path from whatever the ACE-Step fork returns; the upstream API may return a dict like `{"audio": "/tmp/…wav"}`. Update the test in `test_backend.py` to match.
- **"OOM on MPS"** — set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` in the shell before launching `app.py`.
- **Slow first generation** — normal. Pipeline warm-up costs 60-90 s; subsequent generations reuse weights.

---

## Part D — M2: LoRA stack

**Goal:** Bundled preset chips + custom `.safetensors` upload + multi-stack with per-row strength sliders. Loading the psytrance LoRA produces visibly different output.

### Task D1: `lora_stack.py` — header sniff + validation

**Files:**
- Create: `lora_stack.py`
- Create: `tests/test_lora_stack.py`

- [ ] **Step 1: Write failing tests**

`tests/test_lora_stack.py`:
```python
"""L1 tests for LoRA header sniffing — no torch, no pipeline."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

import lora_stack as ls


def _write_safetensors(path: Path, key_dict: dict[str, dict]) -> None:
    """Minimal safetensors writer: header JSON + dummy tensor bytes."""
    header_json = json.dumps(key_dict).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    path.write_bytes(header_len + header_json + b"\0" * 8)  # tiny payload


def test_sniff_accepts_ace_step_lora(tmp_path):
    p = tmp_path / "psytrance.safetensors"
    _write_safetensors(p, {
        "transformer.blocks.0.attn.to_q.lora_A.weight": {"dtype": "BF16", "shape": [64, 768], "data_offsets": [0, 8]},
        "transformer.blocks.0.attn.to_q.lora_B.weight": {"dtype": "BF16", "shape": [768, 64], "data_offsets": [0, 8]},
    })
    info = ls.sniff(p)
    assert info.compatible is True
    assert info.rank == 64
    assert "to_q" in info.target_modules


def test_sniff_rejects_sdxl_lora(tmp_path):
    p = tmp_path / "sdxl.safetensors"
    _write_safetensors(p, {
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight":
          {"dtype": "F16", "shape": [16, 320], "data_offsets": [0, 8]},
    })
    info = ls.sniff(p)
    assert info.compatible is False
    assert "expected" in info.diagnostic.lower()


def test_sniff_rejects_oversize(tmp_path):
    p = tmp_path / "huge.safetensors"
    p.write_bytes(b"\0" * (600 * 1024 * 1024))  # > 500 MB cap
    with pytest.raises(ls.LoRAValidationError, match="too large"):
        ls.sniff(p)
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest tests/test_lora_stack.py -v
```

- [ ] **Step 3: Write `lora_stack.py`**

```python
"""LoRA stack: sniff/validate user-uploaded .safetensors, manage PEFT adapters,
serve bundled presets.

`sniff()` reads only the safetensors header (no tensor materialisation) to
verify rank, target modules, and file size — fast even for big files.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

# Expected DiT module suffixes for ACE-Step 1.5 XL SFT.
# Match against `*.to_q.lora_A.weight`, etc.
_EXPECTED_MODULES = {"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"}
_MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB cap
_MAX_RANK = 256


class LoRAValidationError(ValueError):
    """Raised when a LoRA file fails validation."""


@dataclass
class LoRAInfo:
    path: Path
    compatible: bool
    rank: int
    alpha: int | None
    target_modules: set[str]
    diagnostic: str
    file_size: int


def sniff(path: Path | str) -> LoRAInfo:
    """Read the safetensors header; do not materialise tensors."""
    path = Path(path)
    if not path.exists():
        raise LoRAValidationError(f"File not found: {path}")

    file_size = path.stat().st_size
    if file_size > _MAX_FILE_BYTES:
        raise LoRAValidationError(
            f"File too large ({file_size / 1e6:.0f} MB > {_MAX_FILE_BYTES / 1e6:.0f} MB cap)."
        )

    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise LoRAValidationError("Not a valid .safetensors file (truncated)")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if header_len <= 0 or header_len > 10 * 1024 * 1024:
            raise LoRAValidationError(f"Unreasonable header length: {header_len}")
        header_bytes = f.read(header_len)

    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as e:
        raise LoRAValidationError(f"Invalid header JSON: {e}") from e

    target_modules: set[str] = set()
    rank = 0
    alpha = None

    for k, v in header.items():
        if k == "__metadata__":
            if isinstance(v, dict):
                if "lora_alpha" in v:
                    try:
                        alpha = int(v["lora_alpha"])
                    except (TypeError, ValueError):
                        pass
            continue
        if not isinstance(v, dict) or "shape" not in v:
            continue
        # Extract module suffix from things like "transformer.blocks.0.attn.to_q.lora_A.weight"
        for suffix in _EXPECTED_MODULES:
            if f".{suffix}.lora_A.weight" in k or f".{suffix}.lora_B.weight" in k:
                target_modules.add(suffix)
                if "lora_A.weight" in k:
                    rank = max(rank, int(v["shape"][0]))
                break

    compatible = bool(target_modules) and (rank > 0) and (rank <= _MAX_RANK)
    diagnostic = (
        "OK"
        if compatible
        else f"Expected ACE-Step DiT modules ({sorted(_EXPECTED_MODULES)}), got modules in: "
             f"{sorted(set(header.keys()) - {'__metadata__'})[:3]}…"
    )

    return LoRAInfo(
        path=path,
        compatible=compatible,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        diagnostic=diagnostic,
        file_size=file_size,
    )
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest tests/test_lora_stack.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add lora_stack.py tests/test_lora_stack.py
git commit -m "feat(lora): add safetensors header sniff with ace-step module check"
```

### Task D2: Preset registry + manifest

**Files:**
- Create: `presets/manifest.json`
- Modify: `lora_stack.py`
- Create: `tests/test_lora_presets.py`

- [ ] **Step 1: Write `presets/manifest.json`**

```json
[
  {
    "name": "RapMachine",
    "hf_id": "ACE-Step/ACE-Step-v1-RapMachine-LoRA",
    "filename": "RapMachine.safetensors",
    "kind": "genre",
    "default_scale": 0.85
  },
  {
    "name": "Chinese Rap",
    "hf_id": "ACE-Step/ACE-Step-v1-Chinese-Rap-LoRA",
    "filename": "ChineseRap.safetensors",
    "kind": "genre",
    "default_scale": 0.85
  },
  {
    "name": "Lyric2Vocal",
    "hf_id": "ACE-Step/ACE-Step-v1-Lyric2Vocal-LoRA",
    "filename": "Lyric2Vocal.safetensors",
    "kind": "voice",
    "default_scale": 0.70
  },
  {
    "name": "Text2Samples",
    "hf_id": "ACE-Step/ACE-Step-v1-Text2Samples-LoRA",
    "filename": "Text2Samples.safetensors",
    "kind": "instrumental",
    "default_scale": 0.80
  }
]
```

NOTE: the HF IDs are placeholders — verify against the actual `ACE-Step/...` org on HuggingFace before M2 deploy (spec §14 open question 3).

- [ ] **Step 2: Write the failing test**

`tests/test_lora_presets.py`:
```python
"""L1 tests for the LoRA preset registry."""
from __future__ import annotations

import lora_stack as ls


def test_load_presets_returns_four():
    presets = ls.load_presets()
    assert len(presets) == 4
    names = [p["name"] for p in presets]
    assert "RapMachine" in names
    assert "Lyric2Vocal" in names


def test_preset_has_required_fields():
    presets = ls.load_presets()
    for p in presets:
        assert "name" in p
        assert "hf_id" in p
        assert "default_scale" in p
        assert 0 <= p["default_scale"] <= 1.5
```

- [ ] **Step 3: Add `load_presets()` to `lora_stack.py`**

Append:
```python
import os

_PRESETS_PATH = Path(__file__).resolve().parent / "presets" / "manifest.json"


def load_presets() -> list[dict]:
    """Load the bundled LoRA preset manifest."""
    return json.loads(_PRESETS_PATH.read_text())


def download_preset(name: str) -> Path:
    """Download a preset LoRA from HF if not already cached. Returns local path."""
    from huggingface_hub import hf_hub_download

    for p in load_presets():
        if p["name"] == name:
            local = hf_hub_download(repo_id=p["hf_id"], filename=p["filename"])
            return Path(local)
    raise LoRAValidationError(f"Unknown preset: {name}")
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest tests/test_lora_presets.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add presets/manifest.json lora_stack.py tests/test_lora_presets.py
git commit -m "feat(lora): add preset registry with hf-download helper"
```

### Task D3: PEFT adapter stack management

**Files:**
- Modify: `lora_stack.py`
- Modify: `tests/test_lora_stack.py`

- [ ] **Step 1: Add adapter-stack tests**

Append to `tests/test_lora_stack.py`:
```python
from unittest.mock import MagicMock


def test_apply_stack_calls_set_adapters_in_order():
    pipe = MagicMock()
    stack = [
        {"name": "RapMachine", "scale": 0.85, "path": "/cache/RapMachine.safetensors", "sha256": "a" * 64},
        {"name": "psytrance_v2", "scale": 0.95, "path": "/uploads/psy.safetensors", "sha256": "b" * 64},
    ]
    ls.apply_stack(pipe, stack)

    # The implementation should load each adapter then set_adapters with both
    assert pipe.load_lora_weights.call_count == 2
    pipe.set_adapters.assert_called_once_with(
        ["RapMachine", "psytrance_v2"], adapter_weights=[0.85, 0.95]
    )


def test_apply_stack_empty_disables_adapters():
    pipe = MagicMock()
    ls.apply_stack(pipe, [])
    pipe.disable_adapters.assert_called_once()
    pipe.set_adapters.assert_not_called()
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest tests/test_lora_stack.py::test_apply_stack_calls_set_adapters_in_order -v
```

- [ ] **Step 3: Implement `apply_stack()` in `lora_stack.py`**

Append:
```python
def apply_stack(pipe, stack: list[dict]) -> None:
    """Load each LoRA as a PEFT adapter and activate them in order.

    `stack` is a list of dicts with keys: name, scale, path, sha256.
    Adapters loaded once stay resident across calls; re-loading is a no-op
    in PEFT.
    """
    if not stack:
        # No LoRAs requested — disable any prior stack so vanilla XL plays.
        if hasattr(pipe, "disable_adapters"):
            pipe.disable_adapters()
        return

    names: list[str] = []
    scales: list[float] = []
    for adapter in stack:
        name = adapter["name"]
        path = adapter["path"]
        pipe.load_lora_weights(path, adapter_name=name)
        names.append(name)
        scales.append(float(adapter["scale"]))

    pipe.set_adapters(names, adapter_weights=scales)
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest tests/test_lora_stack.py -v
```

Expected: 5 passed (3 sniff + 2 apply_stack).

- [ ] **Step 5: Commit**

```bash
git add lora_stack.py tests/test_lora_stack.py
git commit -m "feat(lora): add peft adapter stack apply + disable"
```

### Task D4: Wire LoRA stack into `backend.dispatch()`

**Files:**
- Modify: `backend.py`
- Modify: `tests/test_backend.py`

- [ ] **Step 1: Add a test for LoRA application in dispatch**

Append to `tests/test_backend.py`:
```python
def test_dispatch_applies_lora_stack(monkeypatch, tmp_path):
    fake_pipe = MagicMock()
    fake_pipe.return_value = str(tmp_path / "x.wav")
    (tmp_path / "x.wav").write_bytes(b"RIFF")
    monkeypatch.setattr("ace_pipeline.get_pipeline", lambda: fake_pipe)
    apply_mock = MagicMock()
    monkeypatch.setattr("lora_stack.apply_stack", apply_mock)

    b = be.ACEStepStudioBackend()
    stack = [{"name": "RapMachine", "scale": 0.85, "path": "/x.safetensors", "sha256": "a" * 64}]
    b.dispatch(
        mode="generate",
        params={
            "prompt": "p", "lyrics": "", "duration_s": 5, "instrumental": False,
            "seed": 1, "loras": stack, "advanced": {}, "lm": {}, "dcw": {},
        },
    )

    apply_mock.assert_called_once_with(fake_pipe, stack)
```

- [ ] **Step 2: Run test (expect failure)**

```bash
pytest tests/test_backend.py::test_dispatch_applies_lora_stack -v
```

- [ ] **Step 3: Edit `backend.py` to call `apply_stack`**

In `_call_pipe_for_mode` (or at the top of `dispatch`):
```python
import lora_stack

# Inside dispatch(), before calling self._call_pipe_for_mode:
lora_stack.apply_stack(pipe, params.get("loras", []))
```

- [ ] **Step 4: Run all backend tests**

```bash
pytest tests/test_backend.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend.py tests/test_backend.py
git commit -m "feat(backend): apply lora stack before mode dispatch"
```

### Task D5: LoRA UI in Generate tab — preset chips + active stack + upload

**Files:**
- Modify: `ui.py`
- Modify: `app.py`

- [ ] **Step 1: Extend `ui.py` with LoRA components**

Add to `build_generate_tab()` (before the generate button):

```python
    # ... after duration / instrumental row ...
    with gr.Accordion("LoRA stack", open=False):
        components["lora_active"] = gr.State([])  # list of dicts: {name, scale, path, sha256}
        gr.Markdown("**Bundled presets** — click to toggle")
        components["lora_preset_chips"] = gr.Radio(
            choices=["RapMachine", "Chinese Rap", "Lyric2Vocal", "Text2Samples"],
            label=None,
            interactive=True,
            value=None,
        )
        # Active stack table — read-only view, populated from state
        components["lora_active_view"] = gr.Dataframe(
            headers=["Name", "Scale"],
            datatype=["str", "number"],
            row_count=(0, "dynamic"),
            interactive=False,
            label="Active stack",
        )
        with gr.Row():
            components["lora_upload"] = gr.File(
                label="Drop a custom .safetensors",
                file_count="single",
                file_types=[".safetensors"],
                elem_classes=["ams-lora-file"],
            )
            components["lora_strength"] = gr.Slider(
                minimum=0.0, maximum=1.5, step=0.05, value=0.95,
                label="Strength for next-added LoRA",
            )
        with gr.Row():
            components["lora_add_btn"] = gr.Button("+ Add to stack", variant="secondary")
            components["lora_clear_btn"] = gr.Button("Clear stack", variant="secondary")
```

- [ ] **Step 2: Wire LoRA add/clear handlers in `app.py`**

Add to `app.py`:

```python
import lora_stack


def on_lora_add(
    active: list[dict],
    preset_pick: str | None,
    upload_path: str | None,
    strength: float,
):
    if preset_pick:
        local = lora_stack.download_preset(preset_pick)
        info = lora_stack.sniff(local)
        if not info.compatible:
            raise gr.Error(f"Preset {preset_pick}: {info.diagnostic}")
        active = active + [{
            "name": preset_pick, "scale": float(strength),
            "path": str(local), "sha256": "", "kind": "preset",
        }]
    elif upload_path:
        info = lora_stack.sniff(upload_path)
        if not info.compatible:
            raise gr.Error(info.diagnostic)
        name = Path(upload_path).stem
        active = active + [{
            "name": name, "scale": float(strength),
            "path": upload_path, "sha256": "", "kind": "custom",
        }]
    else:
        raise gr.Error("Pick a preset or upload a .safetensors first.")

    return active, [[a["name"], a["scale"]] for a in active]


def on_lora_clear(_active):
    return [], []
```

Add the imports at the top: `from pathlib import Path` (if missing).

Wire the buttons (inside the Generate tab block of `build_app`):
```python
g["lora_add_btn"].click(
    fn=on_lora_add,
    inputs=[g["lora_active"], g["lora_preset_chips"], g["lora_upload"], g["lora_strength"]],
    outputs=[g["lora_active"], g["lora_active_view"]],
)
g["lora_clear_btn"].click(
    fn=on_lora_clear,
    inputs=[g["lora_active"]],
    outputs=[g["lora_active"], g["lora_active_view"]],
)
```

Update `on_generate_click` to pass `g["lora_active"]` through:
```python
def on_generate_click(
    prompt: str,
    lyrics: str,
    duration_s: float,
    instrumental_label: str,
    loras: list[dict],
    progress=gr.Progress(track_tqdm=True),
):
    # ... pass loras=loras into params
```

And include the new input in the click signature:
```python
g["generate_btn"].click(
    fn=on_generate_click,
    inputs=[g["prompt"], g["lyrics"], g["duration_s"], g["instrumental"], g["lora_active"]],
    outputs=[g["output_audio"], g["output_meta"]],
)
```

- [ ] **Step 3: Smoke test with the psytrance LoRA**

```bash
python app.py
```

In the browser:
1. Download a psytrance LoRA from civitai (the one in the spec context) to `~/Downloads/psytrance_v2.safetensors`.
2. Open the Generate tab → expand "LoRA stack".
3. Drop `psytrance_v2.safetensors` into the upload zone, leave strength at 0.95.
4. Click **+ Add to stack** → the Active stack table shows "psytrance_v2 / 0.95".
5. Prompt: `psytrance, rolling triplet bassline, acid squelch`. Lyrics: `[verse] ...`. Duration: 10 s.
6. Click **▶ Generate**.
7. Listen to the output — it should sound noticeably more psytrance-like than the same prompt without the LoRA.

- [ ] **Step 4: Commit**

```bash
git add ui.py app.py
git commit -m "feat(ui): wire lora preset chips + custom upload + active stack"
```

### M2 verification gate

- [ ] **G.M2.1:** `pytest -m "not gpu"` → 12+ tests pass
- [ ] **G.M2.2:** Psytrance LoRA loaded → output is audibly genre-shifted
- [ ] **G.M2.3:** Bad-file rejection: upload a non-ACE-Step LoRA, see clear error toast
- [ ] **G.M2.4:** Stack of 2 LoRAs → both show in active table, generation succeeds
- [ ] **G.M2.5:** Clear stack → next generation reverts to vanilla XL output
- [ ] **G.M2.6:** Tag: `git tag m2`

---

## Part E — M3: Cover, Extend, Edit modes

**Goal:** All four song modes operational. Each has its own tab, mode-specific inputs, output panel.

### Task E1: `modes.py` — `cover()`, `extend()`, `edit()` handlers

**Files:**
- Modify: `modes.py`
- Create: `tests/test_modes_other.py`

- [ ] **Step 1: Write failing tests**

`tests/test_modes_other.py`:
```python
"""L2 tests for cover / extend / edit mode handlers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import modes


def test_cover_requires_ref_audio():
    with pytest.raises(ValueError, match="ref_audio"):
        modes.cover(MagicMock(), params={"prompt": "p", "lyrics": "[v]", "duration_s": 30, "ref_audio": None})


def test_cover_passes_audio_cover_strength(monkeypatch):
    backend = MagicMock()
    backend.dispatch.return_value = ("/tmp/x.wav", {})
    modes.cover(
        backend,
        params={
            "prompt": "p", "lyrics": "[v]", "duration_s": 30,
            "ref_audio": "/tmp/ref.wav", "audio_cover_strength": 0.9,
            "loras": [], "advanced": {}, "lm": {}, "dcw": {},
        },
    )
    args = backend.dispatch.call_args.kwargs
    assert args["mode"] == "cover"
    assert args["params"]["audio_cover_strength"] == 0.9
    assert args["params"]["ref_audio"] == "/tmp/ref.wav"


def test_extend_requires_seed_audio():
    with pytest.raises(ValueError, match="seed_audio"):
        modes.extend(MagicMock(), params={
            "extra_prompt": "p", "extra_duration_s": 60, "seed_audio": None})


def test_extend_passes_repaint_params(monkeypatch):
    backend = MagicMock()
    backend.dispatch.return_value = ("/tmp/x.wav", {})
    modes.extend(
        backend,
        params={
            "seed_audio": "/tmp/seed.wav", "extra_prompt": "more", "extra_duration_s": 60,
            "extension_lyrics": "[v]", "repaint_strength": 0.5, "wav_crossfade_s": 2.0,
            "loras": [], "advanced": {}, "lm": {}, "dcw": {},
        },
    )
    args = backend.dispatch.call_args.kwargs
    assert args["mode"] == "extend"
    assert args["params"]["repaint_strength"] == 0.5
    assert args["params"]["wav_crossfade_s"] == 2.0


def test_edit_repaint_passes_segment_bounds():
    backend = MagicMock()
    backend.dispatch.return_value = ("/tmp/x.wav", {})
    modes.edit(
        backend,
        params={
            "source_audio": "/tmp/src.wav", "source_lyrics": "[v]",
            "target_lyrics": "[c] new", "segment_start_s": 50.0, "segment_end_s": 90.0,
            "sub_mode": "repaint", "repaint_strength": 0.5,
            "loras": [], "advanced": {}, "lm": {}, "dcw": {},
        },
    )
    args = backend.dispatch.call_args.kwargs
    assert args["mode"] == "edit"
    assert args["params"]["segment_start_s"] == 50.0
    assert args["params"]["segment_end_s"] == 90.0
    assert args["params"]["sub_mode"] == "repaint"


def test_edit_flow_morph_passes_flow_params():
    backend = MagicMock()
    backend.dispatch.return_value = ("/tmp/x.wav", {})
    modes.edit(
        backend,
        params={
            "source_audio": "/tmp/src.wav", "source_lyrics": "[v]",
            "target_lyrics": "[c]", "segment_start_s": 0.0, "segment_end_s": 30.0,
            "sub_mode": "flow_edit", "flow_source_caption": "acoustic ballad",
            "flow_n_min": 0.0, "flow_n_max": 1.0, "flow_n_avg": 1,
            "loras": [], "advanced": {}, "lm": {}, "dcw": {},
        },
    )
    args = backend.dispatch.call_args.kwargs
    assert args["params"]["flow_edit_morph"] is True
    assert args["params"]["flow_edit_source_caption"] == "acoustic ballad"
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest tests/test_modes_other.py -v
```

- [ ] **Step 3: Add handlers to `modes.py`**

Append:
```python
def cover(backend, params):
    prompt = params.get("prompt", "")  # optional in Cover mode
    ref_audio = _require(params, "ref_audio")
    lyrics = params.get("lyrics", "")
    duration_s = int(params.get("duration_s", 30))

    return backend.dispatch(
        mode="cover",
        params={
            "prompt": prompt,
            "ref_audio": ref_audio,
            "lyrics": lyrics,
            "duration_s": duration_s,
            "audio_cover_strength": float(params.get("audio_cover_strength", 0.93)),
            "cover_noise_strength": float(params.get("cover_noise_strength", 0.0)),
            "seed": params.get("seed"),
            "loras": params.get("loras", []),
            "advanced": params.get("advanced", {}),
            "lm": params.get("lm", {}),
            "dcw": params.get("dcw", {}),
        },
    )


def extend(backend, params):
    seed_audio = _require(params, "seed_audio")
    extra_prompt = params.get("extra_prompt", "")
    extra_duration_s = int(params.get("extra_duration_s", 60))

    return backend.dispatch(
        mode="extend",
        params={
            "seed_audio": seed_audio,
            "extra_prompt": extra_prompt,
            "extension_lyrics": params.get("extension_lyrics", ""),
            "extra_duration_s": extra_duration_s,
            "repaint_mode": params.get("repaint_mode", "balanced"),
            "repaint_strength": float(params.get("repaint_strength", 0.5)),
            "wav_crossfade_s": float(params.get("wav_crossfade_s", 2.0)),
            "latent_crossfade_frames": int(params.get("latent_crossfade_frames", 10)),
            "chunk_mask_mode": params.get("chunk_mask_mode", "auto"),
            "seed": params.get("seed"),
            "loras": params.get("loras", []),
            "advanced": params.get("advanced", {}),
            "lm": params.get("lm", {}),
            "dcw": params.get("dcw", {}),
        },
    )


def edit(backend, params):
    source_audio = _require(params, "source_audio")
    source_lyrics = params.get("source_lyrics", "")
    target_lyrics = params.get("target_lyrics", "")
    sub_mode = params.get("sub_mode", "repaint")

    out_params = {
        "source_audio": source_audio,
        "source_lyrics": source_lyrics,
        "target_lyrics": target_lyrics,
        "segment_start_s": float(params.get("segment_start_s", 0.0)),
        "segment_end_s": float(params.get("segment_end_s", 30.0)),
        "sub_mode": sub_mode,
        "seed": params.get("seed"),
        "loras": params.get("loras", []),
        "advanced": params.get("advanced", {}),
        "lm": params.get("lm", {}),
        "dcw": params.get("dcw", {}),
    }
    if sub_mode == "repaint":
        out_params.update({
            "repaint_mode": params.get("repaint_mode", "balanced"),
            "repaint_strength": float(params.get("repaint_strength", 0.5)),
            "chunk_mask_mode": params.get("chunk_mask_mode", "auto"),
            "latent_crossfade_frames": int(params.get("latent_crossfade_frames", 10)),
            "wav_crossfade_s": float(params.get("wav_crossfade_s", 0.0)),
        })
    elif sub_mode == "flow_edit":
        out_params.update({
            "flow_edit_morph": True,
            "flow_edit_source_caption": params.get("flow_source_caption", ""),
            "flow_edit_n_min": float(params.get("flow_n_min", 0.0)),
            "flow_edit_n_max": float(params.get("flow_n_max", 1.0)),
            "flow_edit_n_avg": int(params.get("flow_n_avg", 1)),
        })

    return backend.dispatch(mode="edit", params=out_params)
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest tests/test_modes_other.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add modes.py tests/test_modes_other.py
git commit -m "feat(modes): add cover, extend, edit handlers with sub-mode dispatch"
```

### Task E2: `backend.py` — wire mode-specific pipeline kwargs

**Files:**
- Modify: `backend.py`

- [ ] **Step 1: Replace `_call_pipe_for_mode` with full dispatch**

Replace the existing method with:

```python
def _call_pipe_for_mode(self, pipe, mode: str, params: dict) -> str:
    if mode == "generate":
        return pipe(
            prompt=params["prompt"],
            lyrics=params.get("lyrics", ""),
            duration_s=params["duration_s"],
            instrumental=params.get("instrumental", False),
            seed=params["seed"],
        )
    if mode == "cover":
        return pipe(
            prompt=params.get("prompt", ""),
            ref_audio=params["ref_audio"],
            lyrics=params.get("lyrics", ""),
            duration_s=params["duration_s"],
            audio_cover_strength=params.get("audio_cover_strength", 0.93),
            cover_noise_strength=params.get("cover_noise_strength", 0.0),
            seed=params["seed"],
        )
    if mode == "extend":
        return pipe(
            seed_audio=params["seed_audio"],
            extra_prompt=params.get("extra_prompt", ""),
            extension_lyrics=params.get("extension_lyrics", ""),
            extra_duration_s=params["extra_duration_s"],
            repaint_mode=params.get("repaint_mode", "balanced"),
            repaint_strength=params.get("repaint_strength", 0.5),
            wav_crossfade_s=params.get("wav_crossfade_s", 2.0),
            latent_crossfade_frames=params.get("latent_crossfade_frames", 10),
            chunk_mask_mode=params.get("chunk_mask_mode", "auto"),
            seed=params["seed"],
        )
    if mode == "edit":
        if params["sub_mode"] == "flow_edit":
            return pipe(
                source_audio=params["source_audio"],
                source_lyrics=params.get("source_lyrics", ""),
                target_lyrics=params.get("target_lyrics", ""),
                segment_start_s=params["segment_start_s"],
                segment_end_s=params["segment_end_s"],
                flow_edit_morph=True,
                flow_edit_source_caption=params.get("flow_edit_source_caption", ""),
                flow_edit_n_min=params.get("flow_edit_n_min", 0.0),
                flow_edit_n_max=params.get("flow_edit_n_max", 1.0),
                flow_edit_n_avg=params.get("flow_edit_n_avg", 1),
                seed=params["seed"],
            )
        # repaint
        return pipe(
            source_audio=params["source_audio"],
            source_lyrics=params.get("source_lyrics", ""),
            target_lyrics=params.get("target_lyrics", ""),
            segment_start_s=params["segment_start_s"],
            segment_end_s=params["segment_end_s"],
            repaint_mode=params.get("repaint_mode", "balanced"),
            repaint_strength=params.get("repaint_strength", 0.5),
            chunk_mask_mode=params.get("chunk_mask_mode", "auto"),
            latent_crossfade_frames=params.get("latent_crossfade_frames", 10),
            wav_crossfade_s=params.get("wav_crossfade_s", 0.0),
            seed=params["seed"],
        )

    raise NotImplementedError(f"Mode {mode!r} is not implemented")
```

> **Open question alert (spec §14.7):** the exact ACE-Step keyword names for some of these — especially the extend / repaint conventions — must be verified against the installed `ace-step` (or apple-silicon-fork) version. Run a small script: `python -c "from ace_step import ACEStepPipeline; help(ACEStepPipeline.__call__)"` and adjust the kwarg names if they differ from the psytrance-LoRA config's field names.

- [ ] **Step 2: Verify all backend tests still pass**

```bash
pytest tests/test_backend.py -v
```

- [ ] **Step 3: Commit**

```bash
git add backend.py
git commit -m "feat(backend): wire pipeline kwargs for cover, extend, edit modes"
```

### Task E3: `ui.py` — Cover / Extend / Edit tab builders

**Files:**
- Modify: `ui.py`

- [ ] **Step 1: Add `build_cover_tab()`**

```python
def build_cover_tab() -> dict[str, gr.components.Component]:
    components: dict[str, gr.components.Component] = {}
    with gr.Row():
        with gr.Column(scale=13):
            components["ref_audio"] = gr.Audio(
                label="Reference audio (wav / mp3 / flac, ≤ 60 s)",
                type="filepath",
                sources=["upload"],
            )
            components["prompt"] = gr.Textbox(
                label="New style prompt (optional)",
                placeholder="faster, more aggressive leads",
            )
            components["lyrics"] = gr.Textbox(
                label="New lyrics",
                lines=4,
                placeholder="[verse] new lyrics over the reference style",
            )
            with gr.Row():
                components["duration_s"] = gr.Slider(5, 240, value=30, step=5, label="Duration (s)")
                components["audio_cover_strength"] = gr.Slider(
                    0.0, 1.0, value=0.93, step=0.01, label="Cover strength",
                )
            _add_lora_block(components)
            components["generate_btn"] = gr.Button("▶ Generate cover", variant="primary")
        with gr.Column(scale=10):
            components["output_audio"] = gr.Audio(label="Output", type="filepath", interactive=False)
            components["output_meta"] = gr.Code(label="Metadata", language="json")
    return components
```

Factor out the LoRA-stack accordion as a helper since 4 tabs need it:

```python
def _add_lora_block(components: dict) -> None:
    with gr.Accordion("LoRA stack", open=False):
        components["lora_active"] = gr.State([])
        gr.Markdown("**Bundled presets**")
        components["lora_preset_chips"] = gr.Radio(
            choices=["RapMachine", "Chinese Rap", "Lyric2Vocal", "Text2Samples"],
            label=None, interactive=True, value=None,
        )
        components["lora_active_view"] = gr.Dataframe(
            headers=["Name", "Scale"], datatype=["str", "number"],
            row_count=(0, "dynamic"), interactive=False, label="Active stack",
        )
        with gr.Row():
            components["lora_upload"] = gr.File(
                label="Drop a custom .safetensors",
                file_count="single", file_types=[".safetensors"],
                elem_classes=["ams-lora-file"],
            )
            components["lora_strength"] = gr.Slider(0.0, 1.5, value=0.95, step=0.05, label="Strength")
        with gr.Row():
            components["lora_add_btn"] = gr.Button("+ Add", variant="secondary")
            components["lora_clear_btn"] = gr.Button("Clear", variant="secondary")
```

Refactor `build_generate_tab()` to use `_add_lora_block(components)` in place of the inline LoRA section.

- [ ] **Step 2: Add `build_extend_tab()`**

```python
def build_extend_tab() -> dict[str, gr.components.Component]:
    components: dict[str, gr.components.Component] = {}
    with gr.Row():
        with gr.Column(scale=13):
            components["seed_audio"] = gr.Audio(
                label="Seed audio (≤ 240 s)", type="filepath", sources=["upload"],
            )
            components["extra_prompt"] = gr.Textbox(
                label="Extension prompt", placeholder="build to climax, layered acid leads",
            )
            components["extension_lyrics"] = gr.Textbox(
                label="Extension lyrics (optional)", lines=4,
                placeholder="[bridge] the drop is coming...",
            )
            with gr.Row():
                components["extra_duration_s"] = gr.Slider(5, 120, value=60, step=5, label="Extra duration (s)")
                components["wav_crossfade_s"] = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Crossfade (s)")
            with gr.Accordion("Repaint params", open=False):
                components["repaint_mode"] = gr.Dropdown(
                    choices=["balanced", "left", "right"], value="balanced", label="Repaint mode",
                )
                components["repaint_strength"] = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Repaint strength")
                components["latent_crossfade_frames"] = gr.Slider(0, 30, value=10, step=1, label="Latent crossfade frames")
                components["chunk_mask_mode"] = gr.Dropdown(
                    choices=["auto", "manual"], value="auto", label="Chunk mask",
                )
            _add_lora_block(components)
            components["generate_btn"] = gr.Button("▶ Extend", variant="primary")
        with gr.Column(scale=10):
            components["output_audio"] = gr.Audio(label="Output", type="filepath", interactive=False)
            components["output_meta"] = gr.Code(label="Metadata", language="json")
    return components
```

- [ ] **Step 3: Add `build_edit_tab()`**

```python
def build_edit_tab() -> dict[str, gr.components.Component]:
    components: dict[str, gr.components.Component] = {}
    with gr.Row():
        with gr.Column(scale=13):
            components["source_audio"] = gr.Audio(
                label="Source audio (≤ 240 s)", type="filepath", sources=["upload"],
            )
            components["sub_mode"] = gr.Radio(
                choices=["repaint", "flow_edit"], value="repaint", label="Edit sub-mode",
            )
            components["source_lyrics"] = gr.Textbox(label="Source lyrics", lines=3)
            components["target_lyrics"] = gr.Textbox(label="Target lyrics", lines=3)
            with gr.Row():
                components["segment_start_s"] = gr.Number(value=0.0, label="Segment start (s)")
                components["segment_end_s"] = gr.Number(value=30.0, label="Segment end (s)")
            with gr.Accordion("Repaint options", open=False):
                components["repaint_strength"] = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Repaint strength")
                components["repaint_mode"] = gr.Dropdown(
                    choices=["balanced", "left", "right"], value="balanced", label="Repaint mode",
                )
            with gr.Accordion("Flow-morph options", open=False):
                components["flow_source_caption"] = gr.Textbox(label="Source caption")
                components["flow_n_min"] = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="n_min")
                components["flow_n_max"] = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="n_max")
                components["flow_n_avg"] = gr.Slider(1, 5, value=1, step=1, label="n_avg")
            _add_lora_block(components)
            components["generate_btn"] = gr.Button("▶ Apply edit", variant="primary")
        with gr.Column(scale=10):
            components["output_audio"] = gr.Audio(label="Output", type="filepath", interactive=False)
            components["output_meta"] = gr.Code(label="Metadata", language="json")
    return components
```

- [ ] **Step 4: Commit**

```bash
git add ui.py
git commit -m "feat(ui): add cover, extend, edit tab builders"
```

### Task E4: Wire the 3 new tabs in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add per-tab click handlers**

Add to `app.py`:

```python
def on_cover_click(ref_audio, prompt, lyrics, duration_s, audio_cover_strength, loras):
    try:
        return modes.cover(
            get_backend(),
            params={
                "ref_audio": ref_audio, "prompt": prompt, "lyrics": lyrics,
                "duration_s": int(duration_s),
                "audio_cover_strength": float(audio_cover_strength),
                "seed": random.randint(1, 2_147_483_647),
                "loras": loras, "advanced": {}, "lm": {}, "dcw": {},
            },
        )
    except ValueError as e:
        raise gr.Error(str(e)) from e


def on_extend_click(seed_audio, extra_prompt, extension_lyrics, extra_duration_s,
                    wav_crossfade_s, repaint_mode, repaint_strength,
                    latent_crossfade_frames, chunk_mask_mode, loras):
    try:
        return modes.extend(
            get_backend(),
            params={
                "seed_audio": seed_audio, "extra_prompt": extra_prompt,
                "extension_lyrics": extension_lyrics,
                "extra_duration_s": int(extra_duration_s),
                "wav_crossfade_s": float(wav_crossfade_s),
                "repaint_mode": repaint_mode,
                "repaint_strength": float(repaint_strength),
                "latent_crossfade_frames": int(latent_crossfade_frames),
                "chunk_mask_mode": chunk_mask_mode,
                "seed": random.randint(1, 2_147_483_647),
                "loras": loras, "advanced": {}, "lm": {}, "dcw": {},
            },
        )
    except ValueError as e:
        raise gr.Error(str(e)) from e


def on_edit_click(source_audio, sub_mode, source_lyrics, target_lyrics,
                  segment_start_s, segment_end_s, repaint_strength, repaint_mode,
                  flow_source_caption, flow_n_min, flow_n_max, flow_n_avg, loras):
    try:
        return modes.edit(
            get_backend(),
            params={
                "source_audio": source_audio, "sub_mode": sub_mode,
                "source_lyrics": source_lyrics, "target_lyrics": target_lyrics,
                "segment_start_s": float(segment_start_s),
                "segment_end_s": float(segment_end_s),
                "repaint_strength": float(repaint_strength),
                "repaint_mode": repaint_mode,
                "flow_source_caption": flow_source_caption,
                "flow_n_min": float(flow_n_min), "flow_n_max": float(flow_n_max),
                "flow_n_avg": int(flow_n_avg),
                "seed": random.randint(1, 2_147_483_647),
                "loras": loras, "advanced": {}, "lm": {}, "dcw": {},
            },
        )
    except ValueError as e:
        raise gr.Error(str(e)) from e
```

- [ ] **Step 2: Replace tab placeholders in `build_app()`**

Replace each of the 3 placeholder Markdown tabs with a real builder. Use `ui.build_cover_tab()`, `ui.build_extend_tab()`, `ui.build_edit_tab()`. Wire the same LoRA add/clear handlers as Generate (they're tab-local components but share the function signatures).

Pattern for each tab (paraphrased — apply for Cover/Extend/Edit):
```python
with gr.Tab("🎤 Cover"):
    c = ui.build_cover_tab()
    c["lora_add_btn"].click(fn=on_lora_add,
        inputs=[c["lora_active"], c["lora_preset_chips"], c["lora_upload"], c["lora_strength"]],
        outputs=[c["lora_active"], c["lora_active_view"]])
    c["lora_clear_btn"].click(fn=on_lora_clear,
        inputs=[c["lora_active"]], outputs=[c["lora_active"], c["lora_active_view"]])
    c["generate_btn"].click(fn=on_cover_click,
        inputs=[c["ref_audio"], c["prompt"], c["lyrics"], c["duration_s"], c["audio_cover_strength"], c["lora_active"]],
        outputs=[c["output_audio"], c["output_meta"]])
```

- [ ] **Step 3: Manual smoke per mode**

```bash
python app.py
```

Test each tab with a small input:
- **Cover** — upload any 10–20 s WAV, prompt `psytrance`, generate.
- **Extend** — upload any 15 s WAV, extra-duration 10 s, generate.
- **Edit (repaint)** — upload any 30 s WAV, segment 5–15, target lyrics empty, generate.

If a mode crashes, the most common cause is the ACE-Step kwarg name mismatch — see the open-question alert in Task E2.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): wire cover, extend, edit tabs end-to-end"
```

### M3 verification gate

- [ ] **G.M3.1:** All 4 song modes generate at least once with a real input on M5 Max
- [ ] **G.M3.2:** `pytest -m "not gpu"` → 20+ tests pass
- [ ] **G.M3.3:** ruff format + check pass
- [ ] **G.M3.4:** Tag: `git tag m3`

---

## Part F — M4: Lyrics LM (Qwen 2.5 7B)

**Goal:** The ✍️ Lyrics tab drafts structurally-tagged lyrics. "Use these in Generate" button populates the Generate tab's lyrics field.

### Task F1: `lyrics_lm.py` — lazy Qwen 2.5 7B loader

**Files:**
- Create: `lyrics_lm.py`
- Create: `tests/test_lyrics_lm.py`

- [ ] **Step 1: Write failing tests**

`tests/test_lyrics_lm.py`:
```python
"""L2 tests for lyrics LM — generation is mocked at the model boundary."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import lyrics_lm as ll


def test_build_system_prompt_includes_tag_format():
    sp = ll.build_system_prompt()
    assert "[verse" in sp.lower()
    assert "[chorus" in sp.lower()


def test_generate_lyrics_calls_lm_and_returns_text(monkeypatch):
    fake_lm = MagicMock()
    fake_lm.generate.return_value = "[verse] x\n[chorus] y\n"
    monkeypatch.setattr(ll, "_get_lm", lambda: fake_lm)

    out = ll.generate_lyrics(
        brief="a song", structure="intro, verse, chorus, outro",
        language="en", tone="upbeat", verse_lines=4, chorus_lines=4, bridge_lines=2,
        rhyme="loose", temperature=0.85, top_p=0.9, top_k=40,
        max_new_tokens=200, seed=42,
    )

    assert "[verse]" in out
    fake_lm.generate.assert_called_once()


def test_normalise_lyrics_lowercases_tags():
    norm = ll._normalise(" [Verse 1]\nhello\n[Chorus]\nworld ")
    assert "[verse 1]" in norm
    assert "[chorus]" in norm
    assert "[Verse" not in norm
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest tests/test_lyrics_lm.py -v
```

- [ ] **Step 3: Write `lyrics_lm.py`**

```python
"""Qwen 2.5 7B Instruct as the lyrics writer.

Mac path: mlx-lm with the 4-bit MLX quantisation for speed.
CUDA / ZeroGPU path: transformers with bf16.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import ace_pipeline as ap

_DEFAULT_MAC_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
_DEFAULT_CUDA_ID = "Qwen/Qwen2.5-7B-Instruct"

_LM = None  # lazy module-level singleton


def build_system_prompt() -> str:
    return (
        "You are a songwriter. Output ONLY structured lyrics for an AI music generator.\n"
        "Use these section tags exactly: [intro] [verse 1] [verse 2] [chorus] [bridge] [outro] (etc.)\n"
        "Each section is on its own line, followed by the lyrics for that section. Keep verses 4-8 lines, "
        "choruses 4 lines, bridges 2-4 lines. Match the requested tone and language. Do not include commentary, "
        "headers, or markdown."
    )


def _build_user_prompt(brief, structure, language, tone, verse_lines, chorus_lines, bridge_lines, rhyme) -> str:
    return (
        f"Write lyrics with this structure: {structure}.\n"
        f"Language: {language}. Tone: {tone or 'neutral'}. Rhyme: {rhyme}.\n"
        f"Verse: {verse_lines} lines. Chorus: {chorus_lines} lines. Bridge: {bridge_lines} lines.\n\n"
        f"Brief:\n{brief}\n"
    )


def _normalise(text: str) -> str:
    """Lowercase section tags, strip whitespace."""
    def lower_tag(match: re.Match[str]) -> str:
        return "[" + match.group(1).lower() + "]"

    return re.sub(r"\[([^\]]+)\]", lower_tag, text).strip()


def _get_lm():
    global _LM
    if _LM is None:
        _LM = _load_lm()
    return _LM


def _load_lm():
    device = ap.detect_device()
    if device == "mps":
        from mlx_lm import load  # type: ignore[import-not-found]
        model, tokenizer = load(_DEFAULT_MAC_ID)
        return _MLXLM(model=model, tokenizer=tokenizer)
    # CUDA / CPU fallback path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_DEFAULT_CUDA_ID)
    model = AutoModelForCausalLM.from_pretrained(_DEFAULT_CUDA_ID, torch_dtype="bf16").to("cuda")
    return _HFLM(model=model, tokenizer=tok)


@dataclass
class _MLXLM:
    model: Any
    tokenizer: Any

    def generate(self, system: str, user: str, **kw) -> str:
        from mlx_lm import generate  # type: ignore[import-not-found]
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        return generate(
            self.model, self.tokenizer, prompt=prompt,
            max_tokens=kw.get("max_new_tokens", 600),
            temp=kw.get("temperature", 0.85),
            top_p=kw.get("top_p", 0.9),
        )


@dataclass
class _HFLM:
    model: Any
    tokenizer: Any

    def generate(self, system: str, user: str, **kw) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=kw.get("max_new_tokens", 600),
            temperature=kw.get("temperature", 0.85),
            top_p=kw.get("top_p", 0.9),
            top_k=kw.get("top_k", 40),
            repetition_penalty=kw.get("repetition_penalty", 1.1),
            do_sample=True,
        )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Strip the prompt prefix so only the generated text remains
        return full[len(prompt):] if full.startswith(prompt) else full


def generate_lyrics(
    brief: str, structure: str, language: str, tone: str,
    verse_lines: int, chorus_lines: int, bridge_lines: int,
    rhyme: str, temperature: float, top_p: float, top_k: int,
    max_new_tokens: int, seed: int | None = None,
) -> str:
    lm = _get_lm()
    user = _build_user_prompt(brief, structure, language, tone, verse_lines, chorus_lines, bridge_lines, rhyme)
    raw = lm.generate(
        system=build_system_prompt(),
        user=user,
        temperature=temperature, top_p=top_p, top_k=top_k,
        max_new_tokens=max_new_tokens,
    )
    return _normalise(raw)
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest tests/test_lyrics_lm.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add lyrics_lm.py tests/test_lyrics_lm.py
git commit -m "feat(lyrics): add qwen 2.5 7b loader with mlx and transformers backends"
```

### Task F2: `modes.py` — `lyrics()` handler

**Files:**
- Modify: `modes.py`

- [ ] **Step 1: Append handler**

```python
import lyrics_lm


def lyrics(backend, params: dict) -> tuple[str, dict]:
    """Lyrics-only mode — does not touch the ACE-Step backend at all."""
    brief = _require(params, "brief")
    structure = params.get("structure", "intro, verse, chorus, verse, chorus, bridge, chorus, outro")
    language = params.get("language", "en")
    tone = params.get("tone", "")
    verse_lines = int(params.get("verse_lines", 6))
    chorus_lines = int(params.get("chorus_lines", 4))
    bridge_lines = int(params.get("bridge_lines", 2))
    rhyme = params.get("rhyme", "loose")
    temperature = float(params.get("temperature", 0.85))
    top_p = float(params.get("top_p", 0.9))
    top_k = int(params.get("top_k", 40))
    max_new_tokens = int(params.get("max_new_tokens", 600))
    seed = params.get("seed")

    text = lyrics_lm.generate_lyrics(
        brief=brief, structure=structure, language=language, tone=tone,
        verse_lines=verse_lines, chorus_lines=chorus_lines, bridge_lines=bridge_lines,
        rhyme=rhyme, temperature=temperature, top_p=top_p, top_k=top_k,
        max_new_tokens=max_new_tokens, seed=seed,
    )
    meta = {
        "mode": "lyrics",
        "model": lyrics_lm._DEFAULT_MAC_ID,
        "brief_first_line": brief.splitlines()[0] if brief else "",
        "structure": structure, "language": language, "tone": tone,
        "verse_lines": verse_lines, "chorus_lines": chorus_lines, "bridge_lines": bridge_lines,
        "rhyme": rhyme, "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "max_new_tokens": max_new_tokens, "seed": seed,
    }
    return text, meta
```

- [ ] **Step 2: Quick test (rerun all mode tests)**

```bash
pytest tests/test_modes_generate.py tests/test_modes_other.py -v
```

Should still pass.

- [ ] **Step 3: Commit**

```bash
git add modes.py
git commit -m "feat(modes): add lyrics handler that returns structured text"
```

### Task F3: `ui.py` — Lyrics tab builder

**Files:**
- Modify: `ui.py`

- [ ] **Step 1: Add `build_lyrics_tab()`**

```python
def build_lyrics_tab() -> dict[str, gr.components.Component]:
    c: dict[str, gr.components.Component] = {}
    with gr.Row():
        with gr.Column(scale=12):
            c["brief"] = gr.Textbox(label="Brief", lines=4,
                placeholder="Describe the song. Tone, mood, references, specific images, lines to avoid…")
            with gr.Row():
                c["structure"] = gr.Textbox(label="Structure",
                    value="intro, verse, chorus, verse, chorus, bridge, chorus, outro")
                c["language"] = gr.Dropdown(
                    choices=["en", "zh", "ja", "ko", "es", "fr", "de"], value="en", label="Language")
            with gr.Row():
                c["verse_lines"] = gr.Slider(2, 10, value=6, step=1, label="Verse lines")
                c["chorus_lines"] = gr.Slider(2, 8, value=4, step=1, label="Chorus lines")
                c["bridge_lines"] = gr.Slider(1, 6, value=2, step=1, label="Bridge lines")
            c["tone"] = gr.Textbox(label="Tone / mood",
                placeholder="euphoric, hypnotic, transcendent, not cheesy")
            c["rhyme"] = gr.Radio(choices=["strict", "loose", "none"], value="loose", label="Rhyme")
            with gr.Accordion("LM parameters", open=False):
                c["temperature"] = gr.Slider(0.0, 2.0, value=0.85, step=0.05, label="Temperature")
                c["top_p"] = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                c["top_k"] = gr.Slider(0, 200, value=40, step=1, label="Top-k")
                c["max_new_tokens"] = gr.Slider(100, 2000, value=600, step=50, label="Max new tokens")
                c["seed"] = gr.Number(value=42, precision=0, label="Seed")
            c["draft_btn"] = gr.Button("▶ Draft lyrics", variant="primary")
        with gr.Column(scale=10):
            c["lyrics_output"] = gr.Textbox(label="Draft", lines=14, show_copy_button=True)
            c["use_in_generate_btn"] = gr.Button("↑ Use these in Generate", variant="primary")
            c["meta_output"] = gr.Code(label="Metadata", language="json")
    return c
```

- [ ] **Step 2: Commit**

```bash
git add ui.py
git commit -m "feat(ui): add lyrics tab builder with brief, structure, lm params"
```

### Task F4: Wire Lyrics tab + cross-tab "Use these in Generate"

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the click handler**

```python
def on_draft_lyrics(brief, structure, language, tone, verse_lines, chorus_lines,
                    bridge_lines, rhyme, temperature, top_p, top_k, max_new_tokens, seed):
    try:
        text, meta = modes.lyrics(
            get_backend(),
            params={
                "brief": brief, "structure": structure, "language": language, "tone": tone,
                "verse_lines": int(verse_lines), "chorus_lines": int(chorus_lines),
                "bridge_lines": int(bridge_lines), "rhyme": rhyme,
                "temperature": temperature, "top_p": top_p, "top_k": int(top_k),
                "max_new_tokens": int(max_new_tokens), "seed": int(seed) if seed else None,
            },
        )
    except ValueError as e:
        raise gr.Error(str(e)) from e
    return text, meta
```

- [ ] **Step 2: Wire the Lyrics tab in `build_app()`**

```python
with gr.Tab("✍️ Lyrics"):
    l = ui.build_lyrics_tab()
    l["draft_btn"].click(
        fn=on_draft_lyrics,
        inputs=[l["brief"], l["structure"], l["language"], l["tone"],
                l["verse_lines"], l["chorus_lines"], l["bridge_lines"], l["rhyme"],
                l["temperature"], l["top_p"], l["top_k"], l["max_new_tokens"], l["seed"]],
        outputs=[l["lyrics_output"], l["meta_output"]],
    )
    # Wire "Use these in Generate" → pipe to Generate tab's lyrics field.
    # Requires capturing g["lyrics"] in closure.
    l["use_in_generate_btn"].click(
        fn=lambda txt: txt,
        inputs=[l["lyrics_output"]],
        outputs=[g["lyrics"]],
    )
```

> **Cross-tab caveat:** Gradio supports cross-tab component references as long as both components were defined in the same `Blocks` context. Make sure the Generate tab is built first (which it already is in the tab order).

- [ ] **Step 3: Manual smoke**

```bash
python app.py
```

In the Lyrics tab:
1. Brief: `A driving psytrance anthem about losing yourself on the dancefloor at sunrise`.
2. Tone: `euphoric, hypnotic, not cheesy`.
3. Click **▶ Draft lyrics**. First call warms up Qwen (~30 s on M5 Max).
4. Read the output — should contain `[intro]`, `[verse 1]`, `[chorus]` tags.
5. Click **↑ Use these in Generate** → switch to Generate tab → lyrics field should now contain the draft.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): wire lyrics tab + cross-tab use-in-generate handoff"
```

### M4 verification gate

- [ ] **G.M4.1:** Lyrics tab drafts structurally-tagged lyrics on M5 Max in < 15 s after warm-up
- [ ] **G.M4.2:** "Use in Generate" button populates Generate's lyrics field
- [ ] **G.M4.3:** Generated lyrics actually feed a successful Generate run
- [ ] **G.M4.4:** `pytest -m "not gpu"` → 25+ tests pass
- [ ] **G.M4.5:** Tag: `git tag m4`

---

## Part G — M5: Post-processing (Demucs + loudness + mp3)

**Goal:** After each generation, optionally separate stems and export as 320 kbps MP3 with loudness normalisation.

### Task G1: `post_process.py` — Demucs + pyloudnorm + ffmpeg mp3

**Files:**
- Create: `post_process.py`
- Create: `tests/test_post_process.py`

- [ ] **Step 1: Write failing tests**

`tests/test_post_process.py`:
```python
"""L2 tests for post-processing — Demucs and ffmpeg are mocked."""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import post_process as pp


def test_separate_stems_returns_four_paths(tmp_path, monkeypatch):
    src = tmp_path / "song.wav"
    src.write_bytes(b"RIFF" + b"\0" * 100)

    fake_sep = MagicMock()
    fake_sep.separate_audio_file.return_value = {
        "vocals": tmp_path / "vocals.wav",
        "drums": tmp_path / "drums.wav",
        "bass": tmp_path / "bass.wav",
        "other": tmp_path / "other.wav",
    }
    for k in ("vocals", "drums", "bass", "other"):
        (tmp_path / f"{k}.wav").write_bytes(b"RIFF" + b"\0" * 100)
    monkeypatch.setattr(pp, "_get_demucs", lambda: fake_sep)

    stems = pp.separate_stems(src)

    assert set(stems.keys()) == {"vocals", "drums", "bass", "other"}
    for p in stems.values():
        assert Path(p).exists()


def test_normalise_lufs_invokes_pyloudnorm(monkeypatch, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFF" + b"\0" * 100)
    called = {}
    def fake_norm(in_path, out_path, target_lufs):
        called.update({"in": in_path, "out": out_path, "target": target_lufs})
        Path(out_path).write_bytes(b"RIFF")
    monkeypatch.setattr(pp, "_pyloudnorm_normalise", fake_norm)

    out = pp.normalise_lufs(src, target_lufs=-14.0)
    assert called["target"] == -14.0
    assert Path(out).exists()


def test_to_mp3_invokes_ffmpeg(monkeypatch, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFF")
    called = {}
    def fake_run(cmd, **kw):
        called["cmd"] = cmd
        # write expected output
        out = cmd[-1]
        Path(out).write_bytes(b"ID3")
        return MagicMock(returncode=0)
    monkeypatch.setattr(pp.subprocess, "run", fake_run)

    out = pp.to_mp3(src, bitrate_kbps=320)
    assert Path(out).exists()
    assert "320k" in " ".join(called["cmd"])
```

- [ ] **Step 2: Run tests (expect failure)**

```bash
pytest tests/test_post_process.py -v
```

- [ ] **Step 3: Write `post_process.py`**

```python
"""Post-generation: stem separation (Demucs), loudness normalisation, MP3 export."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

_DEMUCS = None


def _get_demucs() -> Any:
    global _DEMUCS
    if _DEMUCS is None:
        from demucs.api import Separator
        _DEMUCS = Separator(model="htdemucs_ft")
    return _DEMUCS


def separate_stems(audio_path: Path | str) -> dict[str, Path]:
    """Split into vocals/drums/bass/other via htdemucs_ft."""
    sep = _get_demucs()
    result = sep.separate_audio_file(str(audio_path))
    return {name: Path(p) for name, p in result.items()}


def _pyloudnorm_normalise(in_path: str, out_path: str, target_lufs: float) -> None:
    """Real pyloudnorm path; isolated for easy mocking in tests."""
    import soundfile as sf
    import pyloudnorm as pyln

    data, rate = sf.read(in_path)
    meter = pyln.Meter(rate)
    current = meter.integrated_loudness(data)
    gain = target_lufs - current
    out = pyln.normalize.loudness(data, current, target_lufs)
    sf.write(out_path, out, rate)


def normalise_lufs(audio_path: Path | str, target_lufs: float = -14.0) -> Path:
    audio_path = Path(audio_path)
    out_path = audio_path.with_name(audio_path.stem + f".lufs{int(target_lufs)}.wav")
    _pyloudnorm_normalise(str(audio_path), str(out_path), target_lufs)
    return out_path


def to_mp3(wav_path: Path | str, bitrate_kbps: int = 320) -> Path:
    wav_path = Path(wav_path)
    out_path = wav_path.with_suffix(".mp3")
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-b:a", f"{bitrate_kbps}k", "-ar", "44100",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest tests/test_post_process.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add post_process.py tests/test_post_process.py
git commit -m "feat(post): add demucs stems, lufs normalisation, mp3 export"
```

### Task G2: Wire post-processing into the output panel

**Files:**
- Modify: `ui.py`
- Modify: `app.py`

- [ ] **Step 1: Add a "Separate stems" + "Export MP3" button per output panel**

In `_add_output_panel()` (extract this helper if needed — currently embedded in each tab):

```python
def _add_output_panel(c: dict) -> None:
    c["output_audio"] = gr.Audio(label="Output", type="filepath", interactive=False)
    with gr.Row():
        c["separate_stems_btn"] = gr.Button("Separate stems", variant="secondary")
        c["normalise_btn"] = gr.Button("Normalise -14 LUFS", variant="secondary")
        c["mp3_btn"] = gr.Button("Export MP3 320k", variant="secondary")
    c["stem_files"] = gr.Files(label="Stems", visible=False)
    c["normalised_audio"] = gr.Audio(label="Normalised", visible=False, interactive=False)
    c["mp3_file"] = gr.File(label="MP3 download", visible=False)
    c["output_meta"] = gr.Code(label="Metadata", language="json")
```

Apply this to each tab builder, replacing the inline output components.

- [ ] **Step 2: Add per-button handlers in `app.py`**

```python
import post_process


def on_separate_stems(audio_path):
    if not audio_path:
        raise gr.Error("Generate a song first.")
    stems = post_process.separate_stems(audio_path)
    return gr.Files(value=list(stems.values()), visible=True)


def on_normalise(audio_path):
    if not audio_path:
        raise gr.Error("Generate a song first.")
    out = post_process.normalise_lufs(audio_path, target_lufs=-14.0)
    return gr.Audio(value=str(out), visible=True)


def on_export_mp3(audio_path):
    if not audio_path:
        raise gr.Error("Generate a song first.")
    out = post_process.to_mp3(audio_path, bitrate_kbps=320)
    return gr.File(value=str(out), visible=True)
```

Wire per tab (Generate example; copy for the others):
```python
g["separate_stems_btn"].click(fn=on_separate_stems, inputs=[g["output_audio"]], outputs=[g["stem_files"]])
g["normalise_btn"].click(fn=on_normalise, inputs=[g["output_audio"]], outputs=[g["normalised_audio"]])
g["mp3_btn"].click(fn=on_export_mp3, inputs=[g["output_audio"]], outputs=[g["mp3_file"]])
```

- [ ] **Step 3: Manual smoke**

```bash
brew install ffmpeg  # if not already installed
python app.py
```

Generate a 10 s clip → click **Separate stems** → 4 WAVs appear for download. Click **Normalise -14 LUFS** → normalised audio appears. Click **Export MP3 320k** → MP3 file appears.

- [ ] **Step 4: Commit**

```bash
git add ui.py app.py
git commit -m "feat(post): wire stems, lufs normalisation, mp3 export to ui"
```

### M5 verification gate

- [ ] **G.M5.1:** Demucs separates a 10 s generated clip into 4 stems on M5 Max
- [ ] **G.M5.2:** Normalised audio's integrated loudness measured at -14 ± 0.5 LUFS (verify with `ffmpeg -i out.wav -af loudnorm=print_format=summary -f null -`)
- [ ] **G.M5.3:** MP3 plays back identical-sounding to source WAV
- [ ] **G.M5.4:** `pytest -m "not gpu"` → 28+ tests pass
- [ ] **G.M5.5:** Tag: `git tag m5`

---

## ⚠ WIREFRAME COMPLIANCE — UI architecture (read before touching any tab)

**This section was added on 2026-05-18 after the first M0 implementation regressed by using `gr.Tabs` instead of a left sidebar. Fix commit: `59b9fee`.**

The wireframes at [`docs/superpowers/specs/mockups/`](../specs/mockups/) are the **visual source of truth**. The plan's prose describes semantics; the mockups describe layout. **If the two ever appear to conflict, the mockups win.** Read all three mockup HTML files in a browser before implementing or modifying any UI.

### Forbidden patterns (do NOT use)

- ❌ **`gr.Tabs` / `gr.Tab` for mode navigation.** These render horizontal top tabs by default. The wireframes show a LEFT sidebar. There is no CSS-only path to relocate Gradio tabs to the side while keeping their selection semantics intact — switch the primitive instead.
- ❌ Top-positioned mode navigation of any kind on desktop ≥ 1024 px.
- ❌ Inline `<style>` tags or unscoped CSS that leaks outside `theme.CSS`.
- ❌ Hardcoded color hex values in component code — always reference `theme.PRIMARY`, `theme.INK`, etc.

### Required pattern: sidebar nav + visibility-toggled panes

The five modes (Generate / Cover / Extend / Edit / Lyrics) are implemented as:

```python
with gr.Row(elem_classes=["ams-body"]):
    # LEFT — sidebar
    with gr.Column(scale=0, min_width=190, elem_classes=["ams-sidebar"]):
        mode = gr.Radio(
            choices=[
                ("🎵 Generate", "generate"),
                ("🎤 Cover",    "cover"),
                ("⏩ Extend",   "extend"),
                ("✏️ Edit",     "edit"),
                ("✍️ Lyrics",   "lyrics"),
            ],
            value="generate",
            label=None,
            show_label=False,
            container=False,
            elem_classes=["ams-side-radio"],
        )
        gr.HTML(HISTORY_HTML)   # "History · session" block

    # RIGHT — content area with 5 panes, one visible at a time
    with gr.Column(scale=10, elem_classes=["ams-content"]):
        with gr.Group(visible=True,  elem_classes=["ams-tab-pane"]) as pane_generate: ...
        with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_cover:    ...
        with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_extend:   ...
        with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_edit:     ...
        with gr.Group(visible=False, elem_classes=["ams-tab-pane"]) as pane_lyrics:   ...

# Wiring: radio change → swap which pane is visible
def _switch_pane(selected):
    order = ["generate", "cover", "extend", "edit", "lyrics"]
    return tuple(gr.Group(visible=(selected == name)) for name in order)

mode.change(fn=_switch_pane, inputs=mode,
            outputs=[pane_generate, pane_cover, pane_extend, pane_edit, pane_lyrics])
```

### Why `gr.Radio`, not `gr.Button`

- The Radio's native `:checked` pseudo-class provides the sidebar "active item" highlight for free via CSS — no JavaScript needed.
- Single-source-of-truth for which mode is active (the radio's value).
- `gr.Button` cannot have its `elem_classes` updated dynamically in Gradio 5, so an active-state highlight on buttons would require an HTML/JS bridge.

### Inside each mode pane

Per the wireframes (mockup `01_generate_mobile_errors.html` and `02_cover_extend.html`), each song-mode pane has a **two-column body**:

```python
with gr.Row():
    # Left column ~60% — form
    with gr.Column(scale=13):
        # mode-specific inputs (prompt, lyrics, mode-specific upload zones)
        # LoRA stack accordion (collapsed by default)
        # Advanced accordion (collapsed by default)
        # LM planner accordion (collapsed by default)
        # DCW accordion (collapsed by default)
        # Generate button (variant="primary", full width)

    # Right column ~40% — output
    with gr.Column(scale=10):
        # Audio player (gr.Audio with waveform)
        # Retake button
        # Stems grid (4 cells: vocals/drums/bass/other) — visible after Demucs runs
        # Action row (mp3, wav, stems zip, meta json, share link)
        # Metadata JSON viewer (gr.Code with language="json")
```

The Lyrics tab is the one exception — it has a brief/structure/lang/tone/rhyme form on the left and a draft-output + variants + quick-refinement-chips block on the right (see `03_edit_lyrics.html`).

### Accordion disclosure rules

- **LoRA stack** — collapsed by default; auto-opens when a preset chip or upload is added.
- **Advanced** — collapsed by default. BPM, key, time-sig, sampler, language, steps, CFG, shift, negative prompt, audio format, loudness, fade, seed + lock.
- **LM planner** — collapsed by default. Thinking toggle, constrained decoding, temp / top-k / top-p / LM-CFG, CoT toggles, LM negative prompt.
- **DCW** — collapsed by default. enabled, mode, wavelet, scaler, high scaler.

A "show all advanced" debug switch is OUT OF SCOPE for v1 — users discover sections incrementally per the wireframes' progressive-disclosure pattern.

### Sidebar History section

- A `gr.HTML` block titled "History · session" appears below the mode radio in `.ams-sidebar`.
- v1 implementation is in-memory only (per spec §13). The HTML is rewritten on each successful generation via `gr.HTML.update(value=…)`.
- Empty state: "No generations yet" in italic muted ink.
- Each entry: `▶ {mode} · {label[:30]}` — clicking does nothing in v1 (selection is v2).

### Responsive breakpoints (defined in `theme.CSS`)

| Width | Sidebar behaviour | Mode nav |
|---|---|---|
| ≥ 1024 px | Full sidebar (190 px) with mode labels + History | Vertical pills |
| 640 – 1024 px | Icon rail (48 px), labels hidden via `font-size: 0` and `::first-letter` showing only the emoji; History hidden | Vertical icon rail |
| < 640 px | Sidebar hidden; mode radio re-orients to horizontal scroll strip at the top | Horizontal scroll |

These media queries are already in `theme.CSS`. Do **not** add JavaScript-based responsive logic — CSS alone is enough.

### Verification gate for any UI work

Before reporting a UI task DONE, the implementer **must**:

1. Open the running app in a browser at the relevant breakpoint widths (≥1024, 700, 360).
2. Compare side-by-side against the mockup at the matching breakpoint.
3. If using Playwright MCP for headless verification, take screenshots at all three widths.
4. Flag any visible difference > 5 px in spacing, off-color, or out-of-position element as a concern.

This is non-negotiable. "It looks roughly right" is not acceptable.

---

## Part H — M6: Responsive + polish

**Goal:** Mobile horizontal-tab fallback, tooltips, error UX, history sidebar.

### Task H1: `tooltips.py` + apply to UI

**Files:**
- Create: `tooltips.py`
- Modify: `ui.py`

- [ ] **Step 1: Write `tooltips.py`**

```python
"""Centralised tooltip / `info=` strings — single source of truth."""

GENERATE_PROMPT = "Describe the song. Genre, instruments, tempo, mood."
GENERATE_LYRICS = "Use [verse] [chorus] [bridge] tags. Open the Lyrics tab to draft with Qwen 2.5."
GENERATE_DURATION = "Output length in seconds. Longer outputs cost more compute."
GENERATE_VOCAL = "With vocals: full song. Instrumental: no singing, just music."

COVER_REF_AUDIO = "Reference clip (≤ 60 s). First 12 s influences the output most."
COVER_STRENGTH = "How tightly to match the reference. 1.0 = strict imitation. 0.0 = ignore."

EXTEND_SEED = "Seed audio to continue from (≤ 240 s)."
EXTEND_EXTRA = "Extra time to generate after the seed (5–120 s)."
EXTEND_CROSSFADE = "Smooth the seam between seed and extension."

EDIT_SEGMENT_START = "Seconds into the source where the segment begins."
EDIT_SEGMENT_END = "Seconds into the source where the segment ends."

LORA_STRENGTH = "0.0 = LoRA disabled. 1.0 = full effect. >1.0 = overdrive (may degrade quality)."
LORA_UPLOAD = "Drop a .safetensors LoRA. Must target ACE-Step 1.5 XL DiT modules."

LYRICS_TONE = "Comma-separated descriptors. The LM uses these to flavour the words."
LYRICS_TEMP = "Higher = more creative / chaotic. 0.85 is a good default."
```

Apply `info=tooltips.X` to all major fields in the UI tabs.

- [ ] **Step 2: Commit**

```bash
git add tooltips.py ui.py
git commit -m "feat(ui): add centralised tooltips for all major fields"
```

### Task H2: Error UX hardening

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Wrap mode handlers in a single error-mapping helper**

```python
def _safe_call(fn, *args, **kwargs):
    """Map known exceptions to gr.Error with friendly messages."""
    try:
        return fn(*args, **kwargs)
    except lora_stack.LoRAValidationError as e:
        raise gr.Error(str(e)) from e
    except ValueError as e:
        raise gr.Error(str(e)) from e
    except FileNotFoundError as e:
        raise gr.Error(f"File not found: {e}") from e
    except RuntimeError as e:
        msg = str(e)
        if "MPS" in msg or "mps" in msg:
            raise gr.Error(f"Apple-Silicon op issue: {msg}. PYTORCH_ENABLE_MPS_FALLBACK is enabled.") from e
        raise gr.Error(f"Generation failed: {msg}") from e
```

Wrap every `on_*_click` body with `_safe_call`.

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "fix(ui): unify error handling with friendly gr.Error toasts"
```

### Task H3: History sidebar (in-memory)

**Files:**
- Modify: `ui.py`
- Modify: `app.py`

- [ ] **Step 1: Add a history sidebar to `app.py`**

```python
# Module-level history; gr.State per-session would be better for multi-user but
# v1 is single-user.
_HISTORY: list[dict] = []
_HISTORY_MAX = 10


def _push_history(mode: str, audio_path: str, prompt_or_brief: str) -> None:
    _HISTORY.insert(0, {"mode": mode, "audio_path": audio_path, "label": prompt_or_brief[:30]})
    while len(_HISTORY) > _HISTORY_MAX:
        _HISTORY.pop()


def _history_html() -> str:
    if not _HISTORY:
        return "<div class='ams-history-empty'>No generations yet</div>"
    rows = "".join(
        f"<div class='ams-history-item'>▶ {h['mode']} · {h['label']}</div>"
        for h in _HISTORY
    )
    return f"<div class='ams-history'><div class='ams-history-title'>History · session</div>{rows}</div>"
```

Add a Gradio sidebar with `gr.HTML` that re-renders after each generation by including it in mode outputs.

> Realistically: keep history minimal in v1 — display in a `gr.Markdown` that gets updated each click. Persistence is v2 (spec §13).

- [ ] **Step 2: Commit**

```bash
git add ui.py app.py
git commit -m "feat(ui): add in-memory session history list"
```

### Task H4: Mobile + tablet sanity check

- [ ] **Step 1: Open the running app on a phone**

With `python app.py` running and accessible at `http://192.168.0.2:7860` (use your local IP):
1. Open the URL on a phone on the same wifi.
2. Verify:
   - Sidebar hidden, horizontal tabs at top.
   - Form fills the width.
   - Output stacks below form.
   - Sliders are tappable without overflow.

- [ ] **Step 2: Fix any responsive bugs**

The most common: a Gradio component refuses to fit the narrow viewport. Add an override to `theme.CSS` (in a `@media (max-width: 640px)` block) with `min-width: 0 !important;` on the offending class.

Commit any fixes with `style(theme): fix mobile <thing> overflow`.

### M6 verification gate

- [ ] **G.M6.1:** Tooltip text visible on every important field
- [ ] **G.M6.2:** Friendly error toast for invalid LoRA, missing file, MPS issue
- [ ] **G.M6.3:** History list shows last N generations
- [ ] **G.M6.4:** Phone Safari renders + generates end-to-end
- [ ] **G.M6.5:** `pytest -m "not gpu"` → still passes
- [ ] **G.M6.6:** Tag: `git tag m6`

---

## Part I — M7: Deploy

**Goal:** Production HF Space serves requests at parity with local. GitHub repo public.

### Task I1: GitHub repo creation + push

- [ ] **Step 1: Create the GitHub repo**

Run (after `gh` is logged in):
```bash
gh repo create techfreakworm/ace-music-studio --public --description "Open-source full-song generation studio on ACE-Step 1.5 XL" --homepage "https://huggingface.co/spaces/techfreakworm/ace-music-studio"
```

- [ ] **Step 2: Add remote and push**

```bash
git remote add origin https://github.com/techfreakworm/ace-music-studio.git
git push -u origin main
git push --tags
```

- [ ] **Step 3: Verify CI runs and passes**

Open `https://github.com/techfreakworm/ace-music-studio/actions` — the workflow should run and pass on the latest commit.

### Task I2: HF Space creation + push

- [ ] **Step 1: Create the Space via the Hub UI**

Go to `https://huggingface.co/new-space`. Choose:
- Owner: `techfreakworm`
- Space name: `ace-music-studio`
- SDK: Gradio
- Hardware: ZeroGPU (community grant if needed)
- Visibility: Public

- [ ] **Step 2: Add the Space as a git remote**

```bash
git remote add space https://huggingface.co/spaces/techfreakworm/ace-music-studio
git push space main
```

- [ ] **Step 3: Watch the build logs**

`https://huggingface.co/spaces/techfreakworm/ace-music-studio?logs=build`

Expected build: ~5-10 min (preload_from_hub downloads ~32 GB of weights). If build fails, common causes:
- ACE-Step PyPI / git URL doesn't resolve → swap to a known-good HF mirror or pin to a specific commit
- `requirements.txt` includes `mlx-lm` (Mac-only) → confirm it's in `requirements-mac.txt` only

### Task I3: Implement `_bootstrap()` for real

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Port the bootstrap from z-image-studio**

Copy the logic from `~/Projects/llm/z-image-studio/app.py:_bootstrap` and adapt:

```python
def _bootstrap() -> None:
    """HF Spaces: mirror cache, set HF env, symlink into ./models/.
    Local: skip the mirror; symlink only.
    """
    repo_root = Path(__file__).resolve().parent
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)

    if _on_spaces():
        src = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        dst = Path.home() / "hf-cache-rw"
        _mirror_preload_hf_cache(src, dst)
        os.environ["HF_HOME"] = str(dst)
        os.environ["HF_HUB_CACHE"] = str(dst / "hub")
        cache_hub = dst / "hub"
    else:
        cache_hub = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub")))

    _symlink_snapshots(cache_hub, models_dir)


def _on_spaces() -> bool:
    return any(k in os.environ for k in ("SPACE_ID", "SPACE_HOST"))


def _mirror_preload_hf_cache(src: Path, dst: Path) -> None:
    """Read-only build cache → writable runtime tree. Hard-linked where possible."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    # Use rsync-style behaviour via shutil
    import shutil
    for item in src.iterdir():
        target = dst / item.name
        if not target.exists():
            try:
                shutil.copytree(item, target, copy_function=os.link)  # hard links to save inode space
            except OSError:
                shutil.copytree(item, target)


def _symlink_snapshots(cache_hub: Path, models_dir: Path) -> None:
    """ACE-Step's loader looks for ./models/<repo>/. Symlink HF snapshots there."""
    if not cache_hub.exists():
        return
    for repo_dir in cache_hub.iterdir():
        if not repo_dir.name.startswith("models--"):
            continue
        org, name = repo_dir.name.removeprefix("models--").split("--", 1)
        snapshots = repo_dir / "snapshots"
        if not snapshots.exists():
            continue
        latest = max(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, default=None)
        if latest is None:
            continue
        target = models_dir / org / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(latest)
```

- [ ] **Step 2: Push and watch the Space rebuild**

```bash
git add app.py
git commit -m "feat(deploy): implement bootstrap mirror + symlink for spaces"
git push space main
```

- [ ] **Step 3: Verify Space serves**

Open `https://huggingface.co/spaces/techfreakworm/ace-music-studio`. Run a small Generate (10 s).

### Task I4: Tag v0.1.0

```bash
git tag v0.1.0
git push origin v0.1.0
```

### M7 verification gate

- [ ] **G.M7.1:** GitHub repo public, CI green
- [ ] **G.M7.2:** HF Space builds without errors
- [ ] **G.M7.3:** Space generates a 10 s clip end-to-end
- [ ] **G.M7.4:** Mobile (phone) renders + generates on the Space
- [ ] **G.M7.5:** Tag `v0.1.0` pushed to both `origin` and `space`

---

## Post-launch: continuous tasks (out of v1 scope but ready hooks)

- v1.1: persistent history (sqlite + simple `gr.State`)
- v1.2: voice cloning ("Persona") via RVC
- v1.3: LoRA training inside the app
- v2.0: multi-tenant rate limit + watermarking

These are explicitly out of scope for v1 (spec §13).

---

## Plan self-review

The plan covers every section of the spec:

- §3 Architecture → Part B (M0)
- §4 Modes → Parts C, E, F (M1, M3, M4)
- §5 LoRA stack → Part D (M2)
- §6 Lyrics LM → Part F (M4)
- §7 Post-processing → Part G (M5)
- §8 Frontend → Parts B, C, D, E, F (M0–M4) + H (M6)
- §9 Data flow → embedded across handlers
- §10 Error handling → Part H (M6, Task H2)
- §11 Testing → throughout, plus GPU smoke in Task C6
- §12 Deployment → Part I (M7)

Type / signature consistency check:
- `dispatch(mode, params)` signature: consistent in Tasks C3, D4, E1, E2
- `apply_stack(pipe, stack)` consistent in Tasks D3, D4
- LoRA dict shape `{name, scale, path, sha256, kind}` consistent across all uses
- Mode-handler signature `(backend, params) → (path, meta)` consistent in C2, E1, F2
