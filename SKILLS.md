# SKILLS.md — how to work in this repo

Process rules and habits for editing ACE Music Studio. Companion to `CLAUDE.md` (which is *what & why*); this file is *how* — debugging, verification, deployment, when to commit, when to ship.

> **Default rule when in doubt:** stop and ask the user. The user prefers a question over wrong work.

---

## Investigation before fix

### Reproduce the bug before patching

When the user reports a layout, color, click, or visibility issue, **first action is verify, not code**. Open the local app (http://127.0.0.1:7860) in a browser OR via Playwright (`mcp__playwright__browser_*`) and reproduce the issue. Take a screenshot. THEN diagnose.

Skipping the visual repro twice in a row will produce a patch that fixes a different symptom than the one the user is seeing.

For shape / data bugs: read the stack trace fully, identify the line, then read the function — don't trust the line number alone.

### Pull HF Space logs when something runs there

For Spaces failures, the run logs are the source of truth. **Repo name is case-sensitive: `techfreakworm/ACE-Music-Studio`** (uppercase A/M/S — matches the Pascal-cased Space name).

```bash
HF_TOKEN=$(grep hf_token ~/.cache/huggingface/stored_tokens | cut -d'=' -f2 | tr -d ' ')
curl -s -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/api/spaces/techfreakworm/ACE-Music-Studio/logs/run" \
  > /tmp/hf-runtime.log

# Decode the SSE-style `data: {...}` lines
python3 << 'PY'
import json
msgs = []
for line in open('/tmp/hf-runtime.log'):
    if line.startswith('data:'):
        try: msgs.append(json.loads(line[5:].strip()).get('data', '').rstrip())
        except Exception: pass
with open('/tmp/hf-runtime-decoded.log', 'w') as f:
    f.write('\n'.join(msgs))
print(f'Decoded {len(msgs)} lines')
PY

tail -100 /tmp/hf-runtime-decoded.log
```

`/logs/run` is runtime container output. `/logs/build` is the image-build phase (pip install, preload, etc.). Different problems, different endpoints.

**Important: the `/logs/run` endpoint streams LIVE events from subscription time onward** — older events from earlier in the container's lifetime are NOT replayed. To capture an error that happened minutes ago, restart the Space or repro the failure with the stream open.

### Stage check before action

```bash
curl -s -H "Authorization: Bearer ${HF_TOKEN}" \
  https://huggingface.co/api/spaces/techfreakworm/ACE-Music-Studio \
  | python3 -c "import json,sys; d=json.load(sys.stdin); rs=d.get('runtime',{}); print('stage:',rs.get('stage'),'sha:',d.get('sha','')[:7],'hw:',rs.get('hardware'),'err:',rs.get('errorMessage'))"
```

Terminal stages: `RUNNING`, `RUNTIME_ERROR`, `BUILD_ERROR`, `SLEEPING`, `PAUSED`, `STOPPED`. Transient: `BUILDING`, `APP_STARTING`, `RUNNING_BUILDING` (live serving while a new build runs). Always check `errorMessage` first when stage is non-RUNNING.

### Client-side "Error" with no backend trace

If the UI shows a Gradio "Error" toast/placeholder but `/logs/run` shows the function completed (and the file was saved to `/home/user/app/output/<uuid>.wav`), the culprit is the **Cloudflare proxy SSE idle-timeout at ~80 s**. ZeroGPU's queue wait is silent — no progress events emitted while waiting for GPU allocation → SSE drops → client gives up before the response reaches it. The function still runs to completion. This is NOT a code bug; it's infrastructure timing.

Tells:
- Browser console shows `The user aborted a request.` at ~80 s intervals
- `/logs/run` shows `[AudioSaver] Saved audio to /home/user/app/output/<uuid>.wav`
- Gradio's `.ams-out-audio` has a `<span class="error">Error</span>` overlay but no actual error message in any toast

There's no clean client-side fix. Mitigations: keep the GPU pre-allocated by exercising a small request on schedule, or upgrade the Space to dedicated hardware so queue waits go away.

### Sequential thinking for repeated failures

The user has called this out: if a fix doesn't work on the first try, **stop patching**. Use the `superpowers:sequential-thinking` MCP and the `superpowers:systematic-debugging` skill. Two failed fixes is the signal — go back to root-cause investigation before attempting fix #3.

Pattern that means you're guessing:
- "Just try changing X and see if it works"
- "I see another thing it could be — fix that too"
- Multiple changes in one commit chasing a symptom

Pattern that means you're investigating:
- One hypothesis per cycle
- Each hypothesis has a falsifying experiment
- Experiments produce evidence before code changes

---

## Running locally

```bash
cd /Users/techfreakworm/Projects/llm/music-generator
source .venv/bin/activate
# Restart cleanly (kill anything on 7860)
kill -9 $(lsof -ti:7860 2>/dev/null) 2>/dev/null || true
sleep 1
nohup .venv/bin/python app.py > /tmp/ace-music-studio.log 2>&1 &
disown
# Wait for ready
for i in $(seq 1 30); do curl -sf http://127.0.0.1:7860/ -o /dev/null && echo "ready ${i}s" && break; sleep 1; done
```

`/tmp/ace-music-studio.log` is the live log. Tail it during development. The Monitor tool with a `grep -E "ERROR|Traceback|Exception"` filter is the right way to watch it across many turns without blowing context.

LAN access for phone / tablet testing: `http://192.168.0.10:7860` (the LAN IP of the dev machine). Gradio binds to `0.0.0.0:7860` by default in `app.py`.

---

## Verification before committing

Before every commit:

1. **Tests pass.** `python -m pytest tests/ -q` → target 0 failures. New code adds new tests.
2. **Ruff clean.** `ruff check . && ruff format --check .` — both no-op.
3. **App boots.** Restart the local server (kill 7860, relaunch). Confirm "ready" within ~5 seconds and no traceback in `/tmp/ace-music-studio.log`.
4. **The change is visible.** For UI changes, click through the affected tab in the browser. For backend changes, click Generate and verify the output matches expectation.

Tests + ruff alone is not proof the UI works — the test suite mocks `pipe(...)` and doesn't exercise the Gradio render tree.

---

## When to commit

- **One logical change per commit.** A fix and a refactor are TWO commits, not one.
- After a test goes red → green, commit.
- After fixing a regression, commit BEFORE adding the next feature.
- Don't bundle "while I'm here" changes — they hide the actual fix in the diff.

Conventional Commits format:

```
<type>(<scope>): <subject>

<body — explains WHY, not what>
```

Types in use: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `ci`, `perf`.

NO Claude trailer. NO "Generated with…" footer. See `CLAUDE.md` rule 1.

---

## Deployment workflow

The repo has two remotes:

```
origin  → git@github.com:techfreakworm/ace-music-studio.git
space   → https://huggingface.co/spaces/techfreakworm/ACE-Music-Studio
```

To push:

```bash
git push origin main
git -c credential.helper=osxkeychain push space main
```

The `-c credential.helper=osxkeychain` is required for the HF HTTPS push — the token was stored in the macOS keychain at deploy time (see AGENTS.md "Deploy state"). The user's SSH config handles GitHub; HF needs HTTPS + token.

After the `space` push, HF starts rebuilding. Watch:

```bash
TOKEN=$(grep hf_token ~/.cache/huggingface/stored_tokens | cut -d'=' -f2 | tr -d ' ')
until curl -s -H "Authorization: Bearer $TOKEN" \
  https://huggingface.co/api/spaces/techfreakworm/ACE-Music-Studio \
  | python3 -c "import json,sys; d=json.load(sys.stdin); rs=d.get('runtime',{}); s=rs.get('stage',''); sha=d.get('sha','')[:7]; print(f'{s} {sha}', flush=True); sys.exit(0 if s in ('RUNNING','BUILD_ERROR','RUNTIME_ERROR') else 1)"; do sleep 30; done
```

Typical hot build (cached, only README change): ~30 s + ~2 min APP_STARTING.
Typical warm build (one new dep): ~3 min build + ~3 min APP_STARTING.
Cold first build with all 41.5 GB preloads: ~15 min total.

### HF Spaces build failure modes (in order of how often we hit each)

1. **`No matching distribution found for nano-vllm`** — requirements.txt is trying to pip-install ace-step. Don't; use the vendored submodule + sys.path injection.
2. **`Package 'ace-step' requires a different Python: 3.13.x not in '<3.13,>=3.11'`** — README YAML missing `python_version: "3.11"`.
3. **`gradio==6.2.0` conflict with `gradio[oauth,mcp]==<sdk_version>`** — ace-step upstream pins gradio strictly. Use the apple-silicon fork.
4. **`"short_description" length must be less than or equal to 60 characters`** — pre-receive hook validates YAML. Tighten the README description.
5. **`cp: cannot create hard link … 'Invalid cross-device link'`** — don't `cp -al` the HF cache; the EXDEV failure is unavoidable on ZeroGPU.
6. **`PermissionError: '/home/user/.cache/huggingface/modules'`** — set `HF_MODULES_CACHE=/tmp/hf-modules` before any `trust_remote_code=True` import.
7. **`Model not fully initialized`** — preload symlinks aren't in `vendor/ace-step/checkpoints/`. Run `_symlink_ace_step_checkpoints()` at module load.
8. **`Fast download using 'hf_transfer' is enabled but 'hf_transfer' package is not available`** — add `hf_transfer>=0.1.9` to requirements.txt.

### Submodule maintenance

```bash
# Pull latest upstream changes from the apple-silicon fork
git submodule update --remote vendor/ace-step
git add vendor/ace-step
git commit -m "chore(vendor): bump ace-step to <sha>"

# On a fresh clone, initialize submodules (HF Spaces does --recurse-submodules automatically)
git submodule update --init --recursive
```

When bumping the submodule, check the new fork's `pyproject.toml` diff for added/removed deps — those must be reflected in our top-level `requirements.txt` since we don't pip-install ace-step itself.

### Don't push during HF testing

When the user is actively testing on the live Space, hold local commits — don't push mid-test. They'll explicitly say "push it now" when they're ready.

### Force-push to fresh HF Space (one-time bootstrap)

HF auto-creates a template `README.md` when a Space is created. The first push from your local repo will hit `! [rejected]  main -> main (fetch first)`. Apple's bundled git 2.39.5 ALSO can't fetch from HF (`fatal: expected 'acknowledgments'`). Force-push the bootstrap:

```bash
git -c credential.helper=osxkeychain push -f space main
```

Only do this for a fresh Space. Subsequent pushes are fast-forward.

---

## Adding a new model / weight

1. Add the repo ID to `_PRELOAD_REPOS` in `app.py` so the HF Spaces build downloads it.
2. Add the file (or glob) to `preload_from_hub:` in `README.md`'s YAML frontmatter.
3. If the model needs symlinking into `vendor/ace-step/checkpoints/` (because the fork's loader expects a specific path), extend `_symlink_ace_step_checkpoints()`.
4. If `trust_remote_code=True` is used to load it, double-check `HF_MODULES_CACHE=/tmp/hf-modules` is still in `app.py`'s env-var block.
5. Run tests, restart server, verify in browser, then commit.
6. **Watch the new build closely** — preload size is now ~41.5 GB; another large repo might bump us over the ZeroGPU 70 GB disk cap.

---

## Adding a new mode / tab

1. Spec the new mode in `docs/superpowers/specs/` first. Don't skip this.
2. Add a `<mode>(backend, params)` handler to `modes.py`. Same shape as the existing handlers (generate / cover / extend / edit / lyrics).
3. Add a `build_<mode>_tab()` to `ui.py`. Use the existing tabs as template. Include `_build_lora_accordion(c)` + `_build_advanced_accordion(c)` + `_build_output_panel(c)` if it's a song mode.
4. Add `_GPU_DURATION_HINTS["<mode>"]` to `app.py` — tell the per-mode duration estimator where to find `duration_s` in the handler's args.
5. Wire `on_<mode>_click()` in `app.py` with `progress=gr.Progress(track_tqdm=True)` and `@_maybe_spaces_gpu("<mode>")`. The handler must accept all 21 advanced inputs at the end of its signature and pack them into `params["advanced"]` + `params["lm"]` dicts. Connect `c["generate_btn"].click(inputs=[...], outputs=[c["output_audio"], c["output_meta"], history_html])`.
6. Add a branch to `ace_pipeline.ACEStepStudio.generate()` for any new `task_type`.
7. Add tests in `tests/test_modes_other.py` (or similar) mocking the `pipe` boundary.
8. Update tooltips in `tooltips.py` and the Advanced accordion builder if the mode needs different knobs.
9. Update the spec + plan to reflect the new mode.

---

## When you have 2+ failed fixes

This is a process signal, not a coding signal. Stop coding.

1. Read `superpowers:systematic-debugging` (the Iron Law: no fixes without root-cause investigation).
2. Use `mcp__sequential-thinking__sequentialthinking` to walk through hypotheses one at a time.
3. Each hypothesis needs a falsifying experiment (a log line, a Playwright eval, a test). Run the experiment before writing code.
4. If 3+ fixes have failed, the architecture is wrong — escalate to the user, don't attempt fix #4.

This rule has saved several hours of thrashing in this repo. Honour it.

---

## Brainstorm + visual companion

When making material UI changes, use:

- `superpowers:brainstorming` to clarify what's actually being built
- `superpowers:frontend-design` (or `frontend-design:frontend-design`) for design quality
- The visual companion server (under `.superpowers/brainstorm/.../content/`) for mockups the user can click through

The user's `.superpowers/` directory is git-ignored and persists per project. Don't prematurely re-mockup — confirm with the user that mockups are wanted before generating them.

Default to RESTRAINT — single accent, single font, gradio-native shapes, progressive disclosure.

---

## Skills + sub-agents

When dispatching subagents (Agent tool):

- **Brief them like they walked in cold.** They see none of this conversation. Include file paths, line numbers, what to change, what NOT to change.
- **Don't make a subagent read the plan file.** Paste the relevant section into the prompt.
- **Use Opus for design + heavy refactors.** Sonnet for mechanical implementation. Haiku for trivial CSS / config changes.
- **One subagent per task.** Two parallel subagents touching the same file is a guaranteed merge conflict.
- **Subagents commit but don't push.** The user pushes when they've reviewed the diff locally.

---

## When in doubt

1. Re-read the spec at `docs/superpowers/specs/2026-05-18-ace-music-studio-design.md`.
2. `git log --oneline` — every non-obvious decision has a fix-commit explaining the reasoning.
3. Ask the user. They prefer answering a clarifying question to debugging wrong code an hour later.
