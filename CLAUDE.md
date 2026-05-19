# Project Guidelines — ACE Music Studio

Working notes for AI assistants editing this repo. This file is the *what & why* — the locked architecture, the gotchas, the sole-author rule. Companion to `SKILLS.md` (the *how* — process, debugging, deployment workflow) and `AGENTS.md` (tool-agnostic version of this file).

---

## ⚠ Sole-author rule (non-negotiable)

**Mayank Gupta is the sole author on every commit in this repo.** No exceptions.

When committing:

- **NO** `Co-Authored-By: Claude…` (or any agent name) trailer.
- **NO** "Generated with Claude Code" / "🤖 Generated with…" footers.
- **NO** `--author=…` flag — let git use the user's configured identity.
- **NO** attribution in PR descriptions.

If asked to amend, re-commit, or rebase, strip any prior agent attribution from the commit message. Treat any tooling that suggests adding a Claude trailer as a bug to ignore.

---

## Architecture facts (locked — do not relitigate)

Spec: `docs/superpowers/specs/2026-05-18-ace-music-studio-design.md`
Plan: `docs/superpowers/plans/2026-05-18-ace-music-studio.md`

1. **Backend is ACE-Step 1.5 XL SFT** — not ComfyUI. Installed from git (the package isn't on PyPI). The upstream repo is `git+https://github.com/ace-step/ACE-Step-1.5.git`; the Apple Silicon fork is `git+https://github.com/clockworksquirrel/ace-step-apple-silicon.git`.
2. **Five tabs.** Generate, Cover, Extend, Edit, Lyrics. Progressive disclosure — defaults stay short and reveal advanced controls only when asked.
3. **One pipeline instance.** Single ACE-Step pipeline; mode handlers (generate / cover / extend / edit) call different pipeline entry points. No re-instantiation between calls.
4. **`@spaces.GPU` is applied at module load time.** Identity decorator off Spaces. The decorator's `duration=` parameter takes a callable that estimates per-call timeout from `(mode, params, multiplier)`. Estimator clamps at `[60, 300] s`.
5. **Qwen 2.5 7B handles lyrics generation.** Text-only inference; full multimodal weights are NOT required. On Mac the MLX path is used via mlx-lm.
6. **HF cache → `./models/<repo>/` symlink.** ACE-Step looks for files at `local_model_path/...`. `app._bootstrap()` symlinks every cached snapshot into `./models/<org>/<repo>/` so the preload weights are findable. On Spaces, the build-user-owned `~/.cache/huggingface/hub` is mirrored to runtime-writable `~/hf-cache-rw/` first, then symlinked.
7. **One Gradio process. Lazy backend singleton.** `get_backend()` constructs the pipeline on the first request (~30–60 s warm-up). Module import is fast.

---

## Gotchas we already paid for (don't re-discover)

Each of these cost a debug cycle. Read once.

### MPS / Apple Silicon

- `torch.mps` has no `mem_get_info`. Any VRAM-gate that calls that method raises AttributeError. **Fix:** `vram_limit_for("mps")` returns `None` so the gate short-circuits.
- Several ops aren't implemented on the MPS backend (SDPA variants, some index ops). `app.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so they degrade to CPU instead of crashing.

### ACE-Step gotchas

TBD as discovered during M1+ implementation. Record new ones here as they come up.

### Dependency footguns

- `ace-step` is NOT on PyPI. Install from git (see `requirements.txt`).
- Don't pin `spaces` in `requirements.txt`. HF Spaces' ZeroGPU build injects its own version. A pin causes pip-resolve failure.
- `transformers >= 5` may break imports. **Pin:** `transformers>=4.45,<5.0`.

### Gradio 6.14 quirks

- Running version is `gradio>=6.14,<7`. `requirements.txt` reflects this; HF Spaces `sdk_version: 6.14.0` matches.
- Don't put `<script>` tags inside `gr.HTML` blocks — they get stripped. JS goes in `gr.Blocks(head=…)`.
- `info=` is not accepted by `gr.Audio` or `gr.File` on 6.14. `tooltips.py` keeps the strings for `COVER_REF_AUDIO`, `EXTEND_SEED_AUDIO`, `EDIT_SOURCE_AUDIO`, `LORA_UPLOAD` as the single source of truth — when upstream lands `info=` on those components, they're a one-line wire-up away.
- Slate-blue band around primary CTA: defeated via `.styler { background: transparent }` in `theme.CSS`. If a future Gradio bump reintroduces it, the override needs revisiting.

### HF Spaces deployment

- `preload_from_hub` is build-time only. Runtime falls back to network if any required file isn't preloaded. Use broad globs so configs + index.json files come along.
- ZeroGPU build injects `spaces==0.50.0`. If `requirements.txt` pins `spaces==0.30.0`, pip resolution fails. **Don't pin `spaces` at all** — let HF provide it.
- The `@spaces.GPU` decorator must be applied at module load. Runtime decoration isn't detected by ZeroGPU's startup analyzer.

---

## Coding conventions

- **Python 3.11.** HF Spaces base image is 3.11; older syntax (like no `match`) is fine.
- **Flat top-level layout.** No `src/`, no nested packages. One `.py` per responsibility.
- **No conda.** `python3.11 -m venv .venv`; `brew` for system binaries.
- **No emojis** in code or commits unless explicitly requested. UI strings (CTA banner, button labels) are OK because they're user-facing copy, not code.
- **Type hints on public functions.** Internal helpers can skip them when obvious.
- **Imports at the top of the file.** Inline imports only to break circular deps OR to defer heavy modules (ace-step, torch, mlx) for fast CI startup.
- **`ruff format` + `ruff check`** both pass in CI. No exceptions.

---

## Commits

- **Conventional Commits:** `<type>(<scope>): <subject>` — types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `ci`, `perf`.
- Subject is **imperative**, lowercase, no trailing period.
- Body explains **why** when not obvious. Reference the spec / plan section if relevant.
- Frequent small commits — one logical change per commit.
- **NO Claude trailer.** See top of file.

---

## Testing

- **TDD per the plan.** Each implementation task has the failing test first.
- **L1 + L2 in CI** (no GPU): module structure, mocked pipeline call boundaries, ruff. `tests/test_smoke_gpu.py` is the GPU smoke; it's marked with `@pytest.mark.gpu` and skipped by default (pyproject `addopts = -m 'not gpu'`).
- **No mocks for ACE-Step internals.** Mock only the `pipe(...)` call boundary so the mode-handler logic is verified at the boundary.
- **Use `pytest -m gpu`** to opt into the GPU smoke (~32 GB download on a cold cache; runs full generate + cover + extend + edit).

---

## Out of scope for v1 (don't add without asking)

Per spec §13:

- Multi-prompt batch queue
- Persistent generation history
- User accounts
- Telemetry dashboard
- Voice cloning (RVC)
- LoRA training in-app
- ControlNet-style conditioning
- Spectrogram visualization
- Multi-language UI strings
- Watermarking output audio
- Browser audio editing
- Multi-tenant rate limiting
- DAW export

If a task feels like it needs one of these, stop and ask the user.

---

## When in doubt

1. Read the spec + plan. Fifteen minutes of reading vs a day of wrong implementation.
2. Read `SKILLS.md` for the process side — debugging, deployment, when to commit, when to verify.
3. `git log --oneline` — most non-obvious decisions have a fix-commit explaining the reasoning.
4. **Ask the user** before changing architectural shape or adding scope outside the v1 list.
