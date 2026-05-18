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

For Spaces failures, the run logs are the source of truth.

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token)
curl -s -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/api/spaces/techfreakworm/ace-music-studio/logs/run" \
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

### Stage check before action

```bash
curl -s https://huggingface.co/api/spaces/techfreakworm/ace-music-studio/runtime | python3 -m json.tool
```

Terminal stages: `RUNNING`, `RUNTIME_ERROR`, `BUILD_ERROR`. Transient: `BUILDING`, `APP_STARTING`, `RUNNING_BUILDING` (live serving while a new build runs). Always check `errorMessage` first when stage is non-RUNNING.

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
space   → https://huggingface.co/spaces/techfreakworm/ace-music-studio
```

To push:

```bash
git push origin main
git push space main
```

After the `space` push, HF starts rebuilding. Watch:

```bash
TOKEN=$(cat ~/.cache/huggingface/token)
while true; do
  STATE=$(curl -s -H "Authorization: Bearer $TOKEN" \
    https://huggingface.co/api/spaces/techfreakworm/ace-music-studio/runtime \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('stage','?'))")
  echo "$(date +%H:%M:%S) $STATE"
  case "$STATE" in
    RUNNING|BUILD_ERROR|RUNTIME_ERROR) break ;;
  esac
  sleep 30
done
```

Typical build time: ~5 min after weights are cached. First build with new preload globs: ~15–20 min.

### Don't push during HF testing

When the user is actively testing on the live Space, hold local commits — don't push mid-test. They'll explicitly say "push it now" when they're ready.

---

## Adding a new model / weight

1. Add a `ModelConfig(...)` entry to `models.MODEL_CONFIGS`.
2. Add the file (or glob) to `preload_from_hub:` in `README.md`'s YAML frontmatter.
3. Run tests, restart server, verify in browser, then commit.

---

## Adding a new mode / tab

1. Spec the new mode in `docs/superpowers/specs/` first. Don't skip this.
2. Add a `call_<mode>(pipe, params)` to `modes.py`. Same shape as the existing handlers.
3. Add a `build_<mode>_tab()` to `ui.py`. Use the existing tabs as template.
4. Wire `on_<mode>_generate()` in `app.py` with `progress=gr.Progress(track_tqdm=True)`. Connect `c["generate_btn"].click(...)`.
5. Add tests in `tests/test_modes.py` mocking the `pipe` boundary.
6. Update tooltips dict in `tooltips.py`.
7. Update the spec + plan to reflect the new mode.

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
