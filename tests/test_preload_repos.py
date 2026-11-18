"""Spaces bootstrap configuration: the env block, the model-store hand-off,
and the files HF Spaces reads at build time.

Nothing here imports ``app`` with ``SPACE_ID`` set or touches ``vendor/``:
the bootstrap tests replace ``model_store.startup`` and the vendored path
before setting ``SPACE_ID`` for the duration of one call.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

if os.getenv("SPACE_ID"):  # importing app would run the real bootstrap
    pytest.skip("SPACE_ID is set in this environment", allow_module_level=True)

import app
import model_store

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_SOURCE = (REPO_ROOT / "app.py").read_text()


def _env_setdefault_calls(tree: ast.AST) -> list[tuple[str, ast.Call]]:
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and ast.unparse(node.func.value) == "os.environ"
        ):
            out.append((node.args[0].value, node))
    return out


def test_preload_repos_constant_is_gone():
    # model_manifest.json replaced the hand-kept tuple.
    assert not hasattr(app, "_PRELOAD_REPOS")


def test_spaces_env_block_redirects_xet_and_torch_caches_before_gradio_import():
    tree = ast.parse(APP_SOURCE)
    gradio_line = next(
        n.lineno
        for n in tree.body
        if isinstance(n, ast.Import) and any(alias.name == "gradio" for alias in n.names)
    )
    spaces_ifs = [
        n
        for n in tree.body
        if isinstance(n, ast.If) and "SPACE_ID" in ast.unparse(n.test) and n.lineno < gradio_line
    ]
    assert len(spaces_ifs) == 1, "expected one `if SPACE_ID` env block before `import gradio`"
    inside = {name: ast.unparse(call.args[1]) for name, call in _env_setdefault_calls(spaces_ifs[0])}
    assert inside == {"HF_XET_CACHE": "'/tmp/ams-hf-xet'", "TORCH_HOME": "'/tmp/ams-torch'"}
    # The pre-existing lines stay, also before gradio.
    calls = dict(_env_setdefault_calls(tree))
    assert calls["HF_MODULES_CACHE"].lineno < gradio_line
    assert calls["HF_HUB_ENABLE_HF_TRANSFER"].lineno < gradio_line


def test_app_never_moves_the_hub_cache():
    # A preload lands in the default hub cache; moving it would hide it.
    for var in ("HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME"):
        assert f'"{var}"' not in APP_SOURCE and f"'{var}'" not in APP_SOURCE, var


def test_stale_bootstrap_descriptions_are_gone():
    for stale in ("hardlink", "hf-cache-rw", "_symlink_snapshots_into_models", "./models/<org>/<repo>/"):
        assert stale not in APP_SOURCE, stale


@pytest.fixture
def fake_startup(monkeypatch, tmp_path):
    """Replace model_store.startup and the vendored path so nothing real can run."""
    calls = []

    def _startup(checkpoints_dir, **kwargs):
        assert str(checkpoints_dir).startswith(str(tmp_path)), checkpoints_dir
        calls.append((Path(checkpoints_dir), kwargs))
        if _startup.error is not None:
            raise _startup.error
        return {}

    _startup.error = None
    monkeypatch.setattr(app, "_VENDORED_ACE_STEP", tmp_path / "vendor" / "ace-step")
    monkeypatch.setattr(model_store, "startup", _startup)
    assert app.model_store.startup is _startup
    _startup.calls = calls
    return _startup


def test_bootstrap_is_a_noop_off_spaces(monkeypatch, fake_startup):
    monkeypatch.delenv("SPACE_ID", raising=False)
    app._bootstrap_spaces_cache()
    assert fake_startup.calls == []


def test_bootstrap_delegates_to_model_store_on_spaces(monkeypatch, fake_startup, tmp_path):
    monkeypatch.setenv("SPACE_ID", "test/test")
    app._bootstrap_spaces_cache()
    assert [c[0] for c in fake_startup.calls] == [tmp_path / "vendor" / "ace-step" / "checkpoints"]


@pytest.mark.parametrize("kind", ["model_store", "other"])
def test_bootstrap_failure_prints_traceback_then_summary_last_and_exits_1(
    monkeypatch, fake_startup, capsys, kind
):
    if kind == "model_store":
        summary = (
            "could not get o/xl@" + "a" * 40 + " (1 files, 0.00 GB): no cache root is usable\n  /r: denied"
        )
        fake_startup.error = model_store.ModelStoreError(summary)
    else:  # a bug or an OS error outside the resolver ends the same way
        summary = "OSError: [Errno 30] Read-only file system: 'checkpoints'"
        fake_startup.error = OSError(30, "Read-only file system", "checkpoints")
    monkeypatch.setenv("SPACE_ID", "test/test")

    with pytest.raises(SystemExit) as exc:
        app._bootstrap_spaces_cache()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err
    assert err.index("Traceback") < err.rindex("[ams] model bootstrap failed:")
    # The Space API keeps the last 2000 characters: the summary must be the last thing printed.
    assert err.rstrip().endswith(summary.splitlines()[-1])
    assert err.rstrip()[-2000:].count("[ams] model bootstrap failed:") == 1
    assert summary.splitlines()[0] in err.rstrip()[-2000:]


# --- README front matter (read by HF Spaces at build time) ------------------------

# Every front-matter line that is not python_version or a preload entry, verbatim.
_UNCHANGED_FRONT_MATTER = [
    "title: ACE Music Studio",
    "emoji: 🎵",
    "colorFrom: gray",
    "colorTo: gray",
    "sdk: gradio",
    "sdk_version: 6.14.0",
    "app_file: app.py",
    "pinned: false",
    "license: mit",
    "short_description: Song generation studio on ACE-Step 1.5 XL SFT",
    "preload_from_hub:",
]


def _front_matter() -> tuple[list[str], dict]:
    import yaml

    text = (REPO_ROOT / "README.md").read_text()
    assert text.startswith("---\n")
    raw = text.split("---\n", 2)[1]
    return raw.splitlines(), yaml.safe_load(raw)


def test_readme_preload_from_hub_matches_the_manifest_and_presets():
    import lora_stack

    _, meta = _front_matter()
    manifest = model_store.load_manifest()
    expected = [
        f"{r['repo_id']} {','.join(f['path'] for f in r['files'])} {r['revision']}" for r in manifest["repos"]
    ]
    expected += [f"{p['hf_id']} {p['filename']} {p['revision']}" for p in lora_stack.load_presets()]
    assert meta["preload_from_hub"] == expected


def test_readme_python_version_is_what_zerogpu_runs():
    lines, meta = _front_matter()
    assert meta["python_version"] == "3.10"
    assert 'python_version: "3.10"' in lines


def test_readme_gives_the_bootstrap_an_hour_to_start():
    # Documented Spaces key (default 30 min); the bootstrap's own budget is 25 min.
    lines, meta = _front_matter()
    assert meta["startup_duration_timeout"] == "1h"
    assert "startup_duration_timeout: 1h" in lines
    assert model_store.BUDGET_S <= 30 * 60


def test_readme_rest_of_front_matter_is_unchanged():
    lines, meta = _front_matter()
    others = [ln for ln in lines if not ln.startswith(("python_version:", "startup_duration_timeout:", "- "))]
    assert others == _UNCHANGED_FRONT_MATTER
    assert len(meta["short_description"]) <= 60  # HF rejects pushes otherwise


# --- requirements.txt (resolved by HF Spaces on every rebuild) --------------------

# Exact versions the Python 3.10 gate ran on the model-load path (torch is a range:
# the platform preinstalls it, and ZeroGPU supports 2.11.0 / 2.12.1 / 2.13.0).
_PINNED = [
    "torch>=2.11,<2.14",
    "torchaudio==2.11.0",
    "torchcodec==0.17.0",
    "torchao==0.18.0",
    "diffusers==0.39.0",
    "peft==0.21.3",
    "accelerate==1.15.0",
    "demucs==4.0.1",
    "vector-quantize-pytorch==1.34.1",
    "torch-einops-utils==0.2.8",
    "numba==0.68.1",
    "huggingface_hub==0.36.2",
    "hf_xet==1.8.0",
]


def _requirement_lines() -> list[str]:
    return (REPO_ROOT / "requirements.txt").read_text().splitlines()


def _requirement_names() -> dict[str, str]:
    import re

    out = {}
    for line in _requirement_lines():
        spec = line.split("#", 1)[0].strip()
        if spec:
            out[re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0].lower().replace("_", "-")] = spec
    return out


def test_requirements_pin_the_torch_stack_and_model_load_path():
    lines = _requirement_lines()
    for spec in _PINNED:
        assert spec in lines, spec
        above = lines[lines.index(spec) - 1]
        assert above.startswith("# "), f"{spec} needs a one-line comment saying why"
    names = _requirement_names()
    assert names["torchaudio"] == "torchaudio==2.11.0"  # 2.11.1's binary needs torch >= 2.13
    assert names["demucs"] == "demucs==4.0.1"  # 4.1 fetches weights from an unpinned HF repo first
    assert names["torch-einops-utils"] == "torch-einops-utils==0.2.8"  # vq 1.34.1 does not bound it


def test_requirement_lines_carry_no_trailing_comments():
    # Another ZeroGPU Space reported a CONFIG_ERROR blamed on an inline comment on the torch line.
    for line in _requirement_lines():
        if line.strip() and not line.lstrip().startswith("#"):
            assert "#" not in line, line


def test_requirements_leave_gradio_and_spaces_to_the_platform():
    names = _requirement_names()
    assert "gradio" not in names  # injected by HF from README sdk_version
    assert names["spaces"] == 'spaces; sys_platform == "linux"'  # no version: ZeroGPU injects its own
