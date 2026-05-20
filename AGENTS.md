# AGENTS.md

Tool-agnostic agent guidance for the ACE Music Studio repo. If you're driving Claude Code, Cursor, Aider, Codex, or anything else with file-edit + shell access, **start here**.

This file is the authoritative project rulebook. `CLAUDE.md` is Claude-specific extensions; `SKILLS.md` is workflow rules. README.md is the public-facing project intro — different audience.

---

## TL;DR — the five rules

1. **Mayank Gupta is sole author on every commit.** No agent co-author trailers. No "generated with…" footers. No `--author=` flag. Strip any tool-suggested attribution.
2. **Backend = ACE-Step 1.5 XL SFT, not ComfyUI.** Don't add a ComfyUI dependency under any guise.
3. **One pipeline instance for all modes.** Generate / Cover / Extend / Edit call different entry points on the same pipeline object. Don't instantiate per-mode — it doubles memory and breaks LoRA state.
4. **Don't pin `spaces` in `requirements.txt`.** HF Spaces' ZeroGPU build injects its own version. A pin causes pip-resolve failure.
5. **Locally is the source of truth.** All changes restart `python app.py` and verify on http://127.0.0.1:7860 BEFORE pushing to HF. The Space rebuild is ~5–10 min; iterate locally.

If you can't satisfy these without changing architectural shape, **ask the user before proceeding**.

---

## Project shape

Single-process Gradio 6.14 app, flat top-level Python layout. ACE-Step is vendored as a git submodule at `vendor/ace-step/` (NOT pip-installed — see CLAUDE.md).

```
app.py            Gradio Blocks entry, sys.path injection, bootstrap, event handlers
backend.py        ACEStepStudioBackend; dispatch; meta-dict assembly
modes.py          generate / cover / extend / edit / lyrics — pure handlers
ace_pipeline.py   ACEStepStudio wrapper around AceStepHandler + LLMHandler
lora_stack.py     safetensors header sniff + preset registry + apply_stack
lyrics_lm.py      Qwen 2.5 7B inference (mlx-lm on Mac, transformers on CUDA)
post_process.py   Demucs htdemucs stems + LUFS normalisation + ffmpeg MP3 320 k
ui.py             Per-tab builders (Generate / Cover / Extend / Edit / Lyrics)
                  + _build_lora_accordion + _build_advanced_accordion +
                  _build_output_panel
theme.py          Brutalist Mono palette + Gradio CSS overrides
tooltips.py       Centralised info= strings — single source of truth
presets/          LoRA preset manifest.json (Chinese Rap + Chinese New Year)
tests/            L1+L2 tests + GPU-deselected smoke (54 tests pass on CPU)
docs/superpowers/ spec + plan + brainstorm artifacts + visual mockups
vendor/ace-step/  Git submodule of the apple-silicon ace-step fork
```

Same code path locally (MPS / CUDA) and on HF Spaces. The only branching is `_bootstrap_spaces_cache()` (skipped locally — gated on `SPACE_ID` env var; runs `_symlink_ace_step_checkpoints` on Spaces) and `_warm_demucs_on_spaces()` (also Spaces-only).

---

## Locked architecture decisions

These came out of brainstorming + spec design + the HF deploy push that followed. Do not relitigate.

| Decision | Why | Code reference |
|---|---|---|
| ACE-Step **vendored as git submodule**, NOT pip-installed | Upstream pyproject pins `nano-vllm; sys_platform != "darwin"` — not on PyPI, breaks pip-install on Linux. Vendoring sidesteps the dep declaration; nano-vllm imports inside ace-step are all lazy. | `vendor/ace-step/` + `app.py` sys.path injection |
| One `ACEStepStudioBackend` instance, lazy init | Avoids ~60 s pipeline rebuild per request; LoRA revert is cleaner | `backend.py` + `app.get_backend` |
| Mode dispatch = separate handler functions in `modes.py` | Clean boundaries; easy to test with mocked pipe | `modes.generate/cover/extend/edit/lyrics` |
| MPS `vram_limit = None` | `torch.mps` has no `mem_get_info`; any VRAM gate raises AttributeError otherwise | `ace_pipeline.vram_limit_for` |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` set at app import | A few MPS-unsupported ops crash mid-pipeline without it | `app.py` top-of-file |
| Preload symlinks → `vendor/ace-step/checkpoints/` (NOT `./models/<org>/<repo>/`) | The fork's `AceStepHandler._get_project_root()` ignores its kwarg and resolves checkpoints relative to its own install dir | `app._symlink_ace_step_checkpoints` |
| **No cache-mirror dance** | `cp -al` fails with EXDEV on ZeroGPU (different filesystems); inference workloads only READ the cache | `app._bootstrap_spaces_cache` |
| `HF_MODULES_CACHE=/tmp/hf-modules` at import | `~/.cache/huggingface/modules` is read-only at runtime; `trust_remote_code=True` writes there during model load | `app.py` env-var block |
| MLX path for Qwen on Mac, transformers on Linux | mlx-lm is 3-4x faster than transformers on Apple Silicon for text inference | `lyrics_lm._get_lm` |
| `_HFLM.generate` slices prompt at token level | `tokenizer.decode(skip_special_tokens=True)` strips ChatML markers, so string-level `startswith(prompt)` strip fails and the system + user turns leak into output | `lyrics_lm.py` |
| Single-LoRA semantics (one active at a time) | The apple-silicon fork's DiT exposes `load_lora`/`unload_lora`/`set_use_lora`, not the multi-adapter PEFT API. Multi-entry stacks warn + use the first. | `lora_stack.apply_stack` |
| Advanced controls accordion | User pain: outputs feel "samey" because ace-step `inference_steps` defaults to 8 (turbo). Accordion exposes 21 knobs across Diffusion / CFG schedule / 5Hz LM / Music metadata. Defaults tuned for XL SFT. | `ui._build_advanced_accordion` |
| Per-mode duration estimator | Cover/Extend have `duration_s` at positional index 3 (not 2); Extend uses kwarg `extra_duration_s`; Edit uses `segment_end_s − segment_start_s`; Lyrics has no audio duration | `app._GPU_DURATION_HINTS` + `_extract_duration_s` |

---

## Deploy state

- **GitHub:** [techfreakworm/ace-music-studio](https://github.com/techfreakworm/ace-music-studio) (mirror; canonical history)
- **HF Space:** [techfreakworm/ACE-Music-Studio](https://huggingface.co/spaces/techfreakworm/ACE-Music-Studio) on `zero-a10g` hardware
- **Remotes:** `origin → git@github.com:techfreakworm/ace-music-studio.git` and `space → https://huggingface.co/spaces/techfreakworm/ACE-Music-Studio`
- **HF token storage:** macOS keychain via `git credential-osxkeychain`. Set up once with:
  ```bash
  printf "protocol=https\nhost=huggingface.co\nusername=techfreakworm\npassword=$(cat ~/.cache/huggingface/stored_tokens | grep hf_token | cut -d'=' -f2 | tr -d ' ')\n\n" \
    | git credential-osxkeychain store
  ```
  Then push with `git -c credential.helper=osxkeychain push space main`.
- **GPG-signed deploy tag** per release. The user signs commits with SSH globally; override per-command for the dated deploy tag:
  ```bash
  git -c gpg.format=openpgp -c user.signingkey=8845ABB54D0176AA tag -s deploy-YYYY-MM-DD HEAD -m "..."
  ```
- Milestone tags (`m0`–`m7`) live on GitHub only — HF's pre-receive hook validates README YAML on every commit a tag points at, and older milestones fail the `short_description` ≤60-char rule.

---

## Commit rules

- **Conventional Commits:** `<type>(<scope>): <subject>`
  - types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `ci`, `perf`
- Subject is imperative, lowercase, **no trailing period**.
- Body explains **why** when not obvious. Reference plan task IDs (Task 7, Task A, etc.) when the change implements a specific plan step.
- Frequent small commits; one logical change per commit.
- **No agent attribution** in commit message or body. See rule 1.
- Don't `git push --force` to `main` unless the user explicitly says so. EXCEPTION: HF Space bootstrap force-push is fine — HF auto-creates a template README and that's what you're overwriting.

---

## Verification rules

- **Tests must pass before committing.** `python -m pytest tests/ -q` from the project root.
- **Ruff must be clean.** `ruff check . && ruff format --check .`
- **The local app must boot.** `python app.py` → http://127.0.0.1:7860 reachable, no import error in `/tmp/ace-music-studio.log`.
- **For UI changes:** open the URL in a browser (or Playwright eval) and verify the change is rendered. Don't trust a clean test run + clean ruff as proof that the UI works.
- **For deployment changes:** push to HF Space, watch the build, verify the runtime stage transitions to `RUNNING` before claiming success.

If a change requires breaking these rules, write the reason in the commit body.

---

## Testing conventions

- **TDD per the plan.** Failing test first, then implementation.
- **L1 + L2 in CI** (no GPU). The mode handlers are tested with a mocked pipeline. We do NOT mock ACE-Step internals.
- **L3 GPU smoke** is opt-in (`pytest -m gpu`). Lives in `tests/test_smoke_gpu.py`. Loads the real pipeline (~32 GB cache hit on a warm machine).
- **L4 HF Space smoke** is manual. Push, wait, click each tab, verify audio renders.

`pyproject.toml` has `addopts = -m 'not gpu'` so the default `pytest` invocation skips GPU. Add the marker before any test that touches ACE-Step weights.

---

## Out of scope (v1 cap — don't add without asking)

Per spec §13. If you find yourself "while I'm here"-ing into one of them, stop.

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

If a feature you're adding requires one of these as a sub-step, **ask the user** before proceeding.

---

## When you're not sure

1. Read `docs/superpowers/specs/2026-05-18-ace-music-studio-design.md` — that's the architectural source of truth.
2. Read `docs/superpowers/plans/2026-05-18-ace-music-studio.md` — the task-by-task breakdown.
3. Read `SKILLS.md` — process rules, debugging patterns, deployment workflow.
4. `git log --oneline` — every non-obvious decision has a fix-commit explaining the reasoning.
5. **Ask the user.** A clarifying question costs the user ten seconds. A wrong implementation costs everyone an hour.
