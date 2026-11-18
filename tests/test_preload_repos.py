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


def test_required_repo_failure_ends_import_with_summary_last(monkeypatch, fake_startup):
    summary = (
        "could not get o/xl@"
        + "a" * 40
        + " (1 files, 0.00 GB) into any cache root:\n  /r: PermissionError: no"
    )
    fake_startup.error = model_store.ModelStoreError(summary)
    monkeypatch.setenv("SPACE_ID", "test/test")

    with pytest.raises(SystemExit) as exc:
        app._bootstrap_spaces_cache()

    message = str(exc.value.code)
    assert message.endswith(summary)
    assert exc.value.__suppress_context__  # no chained traceback before the summary


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


def test_readme_preload_from_hub_matches_the_manifest():
    _, meta = _front_matter()
    manifest = model_store.load_manifest()
    expected = [
        f"{r['repo_id']} {','.join(f['path'] for f in r['files'])} {r['revision']}" for r in manifest["repos"]
    ]
    assert meta["preload_from_hub"] == expected


def test_readme_python_version_is_what_zerogpu_runs():
    lines, meta = _front_matter()
    assert meta["python_version"] == "3.10"
    assert 'python_version: "3.10"' in lines


def test_readme_rest_of_front_matter_is_unchanged():
    lines, meta = _front_matter()
    others = [ln for ln in lines if not ln.startswith(("python_version:", "- "))]
    assert others == _UNCHANGED_FRONT_MATTER
    assert len(meta["short_description"]) <= 60  # HF rejects pushes otherwise
