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

1. **Backend is ACE-Step 1.5 XL SFT** — not ComfyUI. Vendored as a **git submodule** at `vendor/ace-step/` (the apple-silicon fork: `clockworksquirrel/ace-step-apple-silicon`). Do NOT pip-install ace-step; the upstream pyproject declares `nano-vllm; sys_platform != "darwin"` which isn't on PyPI and breaks `pip install` on Linux. `app.py` injects `vendor/ace-step/` into `sys.path` at module load BEFORE any `from acestep import …`. Ace-step's transitive deps (diffusers, lightning, accelerate, etc.) are listed explicitly in `requirements.txt`. Upstream updates: `git submodule update --remote vendor/ace-step`.
2. **Five tabs.** Generate, Cover, Extend, Edit, Lyrics. Progressive disclosure — defaults stay short and reveal advanced controls only when asked.
3. **One pipeline instance.** Single ACE-Step pipeline; mode handlers (generate / cover / extend / edit) call different pipeline entry points. No re-instantiation between calls.
4. **`@spaces.GPU` is applied at module load time.** Identity decorator off Spaces. The decorator's `duration=` parameter takes a callable that estimates per-call timeout from `(mode, params, multiplier)`. Estimator clamps at `[60, 300] s`. Per-mode `_GPU_DURATION_HINTS` table in `app.py` handles the different positional index of `duration_s` across handlers (generate=2, cover=3, extend=3 with kwarg `extra_duration_s`, edit=segment_end−segment_start, lyrics=none).
5. **Qwen 2.5 7B handles lyrics generation.** Text-only inference; full multimodal weights are NOT required. On Mac the MLX path is used via mlx-lm; on Linux/CUDA (HF Spaces) the full bf16 transformers path is used. `_HFLM.generate` slices the prompt at the **token level** (`out[0][prompt_len:]`) — string-level `startswith(prompt)` strip fails because `tokenizer.decode(skip_special_tokens=True)` removes the ChatML `<|im_start|>` markers from `full` while they're still present in `prompt`.
6. **Fork's checkpoint resolver wants `vendor/ace-step/checkpoints/`.** NOT `./models/<org>/<repo>/`. `app._symlink_ace_step_checkpoints()` symlinks each top-level entry from the preloaded `ACE-Step/Ace-Step1.5` snapshot flat into `checkpoints/` (vae/, encoder/, 5Hz-lm/, …) and the `acestep-v15-xl-sft` snapshot as the matching subdir. Without this, `initialize_service()` kicks off an async auto-download, returns before it finishes, and the first generation hits "Model not fully initialized". **No cache mirror.** Earlier attempts to `cp -al` (hardlink) `~/.cache/huggingface` into `~/hf-cache-rw/` fail with EXDEV on ZeroGPU (HF cache and home live on different filesystems). Inference workloads only READ the cache, so the mirror was unnecessary.
7. **One Gradio process. Lazy backend singleton.** `get_backend()` constructs the pipeline on the first request (~30–60 s warm-up). Module import is fast.
8. **Advanced controls accordion** — `Advanced ▼` under every song mode (not Lyrics) exposes 21 knobs in four groups: Diffusion (inference_steps, guidance_scale, infer_method, seed), CFG schedule (cfg_interval_start/end, shift, ADG), 5Hz LM (thinking, use_cot_*, lm_temperature/top_p/top_k/cfg/negative_prompt), Music metadata (bpm, keyscale, timesignature, vocal_language). Defaults tuned for XL SFT, NOT turbo: `inference_steps=27` (ace-step default is 8 turbo, way too few), `thinking=True`, `use_cot_*=True`. `backend.dispatch` echoes the active `advanced` + `lm` dicts in the output meta JSON so users can lock-iterate from a seed they liked.

---

## Gotchas we already paid for (don't re-discover)

Each of these cost a debug cycle. Read once.

### MPS / Apple Silicon

- `torch.mps` has no `mem_get_info`. Any VRAM-gate that calls that method raises AttributeError. **Fix:** `vram_limit_for("mps")` returns `None` so the gate short-circuits.
- Several ops aren't implemented on the MPS backend (SDPA variants, some index ops). `app.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so they degrade to CPU instead of crashing.

### ACE-Step gotchas

- **`nano-vllm` is not on PyPI.** Both the upstream and the apple-silicon fork's `pyproject.toml` declare `"nano-vllm; sys_platform != 'darwin'"`. On Linux, `pip install ace-step` fails: `No matching distribution found for nano-vllm`. Fix: **vendor ace-step as a git submodule**, don't pip-install it; list its transitive deps directly in `requirements.txt`. nano-vllm imports inside ace-step are all lazy (function-scoped, try/except) so absence is fine.
- **The fork's `AceStepHandler._get_project_root()` ignores the `project_root` kwarg** and resolves checkpoints relative to its OWN install dir. With the submodule that's `vendor/ace-step/checkpoints/`. See locked architecture fact #6.
- **`AceStepHandler.initialize_service` is fire-and-forget for missing weights.** It kicks off an async download and returns immediately. If `generate_music` is called before the download finishes, you get `RuntimeError: ACE-Step generation failed: Model not fully initialized`. Pre-populate `vendor/ace-step/checkpoints/` with symlinks at module load time (`app._symlink_ace_step_checkpoints`).
- **Upstream `ace-step` pins `gradio==6.2.0` HARD.** Incompatible with HF Spaces' `gradio[oauth,mcp]==<sdk_version>` injection at any newer version. The apple-silicon fork loosens this to `>=6.5.1` — another reason we use the fork.
- **`inference_steps` default of 8 (ACE-Step turbo) is way too few for XL SFT.** Outputs feel "samey" because the model doesn't have enough steps to express prompt variation. Bump to 27+ for non-turbo runs.
- **`infer_method="sde"` adds stochastic noise per step** → genuinely different outputs each run, even with same seed. `"ode"` is deterministic per seed. Expose both as a radio.
- **`thinking` + `use_cot_*` flags default OFF in ace-step's class but ON in our pipeline.** Letting the 5Hz LM rewrite the caption + infer metadata + detect vocal language produces more semantic variety. Worth defaulting ON.
- **Demucs 4.0 vs 4.1 API drift.** 4.0.x exposes only `demucs.pretrained.get_model` + `demucs.apply.apply_model`. The higher-level `demucs.api.Separator` only ships with 4.1+. We pin to the lower-level API in `post_process.py` to be portable. Use `htdemucs` (single model, ~80 MB), NOT `htdemucs_ft` (4-model bag, ~320 MB) — they're hosted on `dl.fbaipublicfiles.com`, NOT HF Hub.
- **MLX worker-thread `generation_stream` bug.** `mlx_lm.generate` uses a module-level `generation_stream` created at import time on the MAIN thread. Gradio runs handlers in anyio worker threads. `wired_limit().__exit__` calls `mx.synchronize(generation_stream)` from the worker → `RuntimeError: There is no Stream(gpu, 0) in current thread`. Fix: re-assign `mlx_lm.generate.generation_stream = mx.new_stream(mx.default_device())` from inside the worker before each `generate()` call. Safe because Gradio queue runs at `default_concurrency_limit=1`.
- **`_HFLM.generate` prompt-strip MUST slice at the token level.** `out[0][prompt_len:]` decoded separately, not `full[len(prompt):]`. `tokenizer.decode(skip_special_tokens=True)` removes `<|im_start|>` markers from `full` while they're still present in the encoded `prompt` — the prefix never matches and system + user turns leak into the output.

### Dependency footguns

- `ace-step` is NOT on PyPI and NOT pip-installable due to the `nano-vllm` declaration. **Vendor as git submodule** (`vendor/ace-step/`), list its transitive deps explicitly in `requirements.txt`.
- Don't pin `spaces` in `requirements.txt`. HF Spaces' ZeroGPU build injects its own version. A pin causes pip-resolve failure.
- `transformers >= 5` may break imports. **Pin:** `transformers>=4.51.0,<4.58.0` (matches ace-step's range).
- `hf_transfer` is required if the user's env has `HF_HUB_ENABLE_HF_TRANSFER=1`. Locally users often have this set globally → install `hf_transfer>=0.1.9` in the venv to avoid `RuntimeError: Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available`.

### Gradio 6.14 quirks

- Running version is `gradio>=6.14,<7`. `requirements.txt` does NOT pin gradio (HF Spaces injects it via `sdk_version`). README's `sdk_version: 6.14.0` is the source of truth on Spaces; locally it's whatever pip resolved when `vendor/ace-step/`'s `gradio>=6.5.1` dep was processed (typically 6.14.x).
- Don't put `<script>` tags inside `gr.HTML` blocks — they get stripped. JS goes in `gr.Blocks(head=…)`.
- `info=` is not accepted by `gr.Audio` or `gr.File` on 6.14. `tooltips.py` keeps the strings for `COVER_REF_AUDIO`, `EXTEND_SEED_AUDIO`, `EDIT_SOURCE_AUDIO`, `LORA_UPLOAD` as the single source of truth — when upstream lands `info=` on those components, they're a one-line wire-up away.
- Slate-blue band around primary CTA: defeated via `.styler { background: transparent }` in `theme.CSS`. If a future Gradio bump reintroduces it, the override needs revisiting.
- **Native checkboxes are invisible on the Brutalist Mono palette.** `accent-color` alone doesn't help — the box dimensions are too small and the checkmark renders in a default system colour that washes out on dark surfaces. `theme.py` overrides with `appearance: none` + a custom 16 px box and a data-URI SVG checkmark drawn inline. Affects all `.ams-content input[type="checkbox"]`.

### Layout / flex gotchas (Brutalist Mono CSS)

- **Flex children default to `min-width: auto`** which equals their content's intrinsic min-size. The wavesurfer.js waveform renders at `pixel-per-second` (a 60 s clip wants ~600 px), so on a 412 px mobile viewport the audio block would push the parent column past the screen edge → whole layout "dances" between pre- and post-generation widths. Fix: `min-width: 0` on `.ams-content` (NOT on `.ams-body > *` — that broad selector ALSO matches `.ams-sidebar` and collapses it to a vertical sliver on desktop, see fix-commit `7dd8eb5`).
- **Cage the wavesurfer waveform AT the outer panel.** `overflow: hidden` on `.ams-out-audio` + `max-width: 100%`. Do NOT add `overflow: hidden` to the inner `.component-wrapper` / `.timestamps` / `.controls` — that clips the play/skip buttons + the right-end `1:00` duration timestamp during transient re-renders (URL bar show/hide on mobile triggers wavesurfer reflow). Reserve `min-height: 24px` on `.timestamps` and `min-height: 60px` on `.controls` so they can never collapse to zero.
- **Inner waveform canvas itself** keeps `overflow: hidden` + `max-width: 100%` so the bars stay inside the column.
- **Sidebar (`.ams-sidebar`) has hard `min-width: 188px`** with `max-width: 210px`. Hidden via `display: none` at `@media (max-width: 640px)` — replaced by a horizontal pill strip. Don't let any broad flex-shrink rule override the desktop minimum.

### HF Spaces deployment

- **Live Space:** [techfreakworm/ACE-Music-Studio](https://huggingface.co/spaces/techfreakworm/ACE-Music-Studio) (hardware: `zero-a10g`). Mirror: [github.com/techfreakworm/ace-music-studio](https://github.com/techfreakworm/ace-music-studio).
- **HF Spaces base image runs Python 3.13 by default for the Gradio SDK.** ACE-Step's pyproject pins `requires-python = "==3.11.*"`. Without `python_version: "3.11"` in README YAML frontmatter, pip resolves nothing. **Pin Python 3.11 in `README.md`.**
- **`sdk_version: 6.14.0`** matches `gradio>=6.5.1` from the apple-silicon fork. HF injects `gradio[oauth,mcp]==<sdk_version>` at build time. If you bump `sdk_version`, verify the fork's gradio pin still allows it.
- `preload_from_hub` is build-time only. Runtime falls back to network if any required file isn't preloaded. Use broad globs so configs + index.json files come along. Current preload list (~41.5 GB total): `ACE-Step/Ace-Step1.5` (umbrella, ~10 GB) + `ACE-Step/acestep-v15-xl-sft` (DiT, ~16 GB) + `ACE-Step/ACE-Step-v1-chinese-rap-LoRA` + `ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA` + `Qwen/Qwen2.5-7B-Instruct` (~15 GB).
- ZeroGPU build injects its own `spaces` version. If `requirements.txt` pins `spaces==…`, pip resolution fails. **Don't pin `spaces` at all** — let HF provide it. (We do declare it as `spaces; sys_platform == "linux"` so it doesn't try to install on Mac, where the import is wrapped in try/except.)
- The `@spaces.GPU` decorator must be applied at module load. Runtime decoration isn't detected by ZeroGPU's startup analyzer.
- **HF pre-receive hook rejects ANY commit whose README YAML metadata fails validation.** `short_description` must be ≤60 chars. Tags pushed to HF must point at commits with valid YAML — if a milestone tag (`m0`–`m7`) points at an older commit with the long description, HF rejects the entire tag push. We keep milestone tags GitHub-only and only push the dated deploy tag to HF.
- **`cp -al` mirror fails on ZeroGPU with EXDEV** ("Invalid cross-device link"). The HF cache and home directory are on different filesystems. Don't try to hardlink-mirror — inference workloads only read the cache anyway.
- **`HF_MODULES_CACHE` must be set to a writable location.** `~/.cache/huggingface/modules` is build-user-owned and read-only at runtime. `transformers.AutoModel.from_pretrained(trust_remote_code=True)` (used by the ACE-Step DiT loader) wants to write modeling shims there → `PermissionError: [Errno 13]`. `app.py` sets `os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf-modules")` before any imports.
- **Cloudflare proxy SSE idle-timeout ~80 s.** ZeroGPU queue waits SILENTLY (no progress events) → SSE drops → client shows "Error" even though the backend successfully generates and saves the file. The function completes, the file is written, but the user never sees it. There's no client-side fix — emit periodic progress events from inside the GPU function once it starts running. The queue-wait phase is harder to keep alive.
- **Force-push to fresh HF Spaces is the standard bootstrap pattern.** HF auto-creates a template `README.md` on `Space create`. `git push space main` fails fast-forward; `git push -f space main` overwrites the template. Don't waste time on rebase-and-merge — the template has no value.
- **Apple's bundled `git` 2.39.5 fails HF's protocol v2 fetch** with `fatal: expected 'acknowledgments'`. `ls-remote` works (queries are short), but `fetch` and `clone` choke on the negotiation. For fresh Spaces, force-push (no fetch needed). For ongoing dev, `brew install git`.
- **HTTPS push to HF requires credential storage.** Use `git credential-osxkeychain` on Mac: `printf "protocol=https\nhost=huggingface.co\nusername=<user>\npassword=<token>\n\n" | git credential-osxkeychain store`. The token is at `~/.cache/huggingface/stored_tokens` (`hf_token` key). Then `git -c credential.helper=osxkeychain push space main`.
- **GPG-signed deploy tags.** User signs commits with SSH by default (`user.signingkey=/Users/<u>/.ssh/id_ed25519`, `gpg.format=ssh`). For HF deploy tags that need GPG verification, override per-command: `git -c gpg.format=openpgp -c user.signingkey=<keyid> tag -s deploy-YYYY-MM-DD HEAD -m "..."`. Doesn't change the user's global signing config.
- **`hf` CLI replaces deprecated `huggingface-cli`.** Hardware request: use the Python API directly — `HfApi(token=…).request_space_hardware("<owner>/<space>", "zero-a10g")`. The undocumented `/api/spaces/<repo>/hardware` REST endpoint accepts POST but the CLI doesn't expose it.
- **Space stage transitions to watch:** `BUILDING` (build container) → `APP_STARTING` (preload + Python init) → `RUNNING` (Gradio listening). Terminal failure: `BUILD_ERROR` (pip / Dockerfile) or `RUNTIME_ERROR` (Python exception during init). Hardware swap (e.g. cpu-basic → zero-a10g) goes through `BUILDING` again.

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
