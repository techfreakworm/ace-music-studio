"""Tests for the pinned model manifest and the model store resolver.

Everything here is offline: no weights, no network, and every cache root and
checkpoints directory lives under ``tmp_path``. The tests that reach the Hub
are marked ``network`` and also need ``AMS_NETWORK_TESTS=1``.

CI runs these with an unpinned gradio, so huggingface_hub may be 1.x/2.x
there: hub internals are only touched through getattr/raising=False, and
no test needs requests, torch or hf_transfer.
"""

from __future__ import annotations

import errno
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from huggingface_hub import constants as hf_constants

import model_store as ms

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "model_manifest.json"

UMBRELLA = "ACE-Step/Ace-Step1.5"
XL_SFT = "ACE-Step/acestep-v15-xl-sft"
QWEN = "Qwen/Qwen2.5-7B-Instruct"


def _raw_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _by_repo(manifest: dict) -> dict[str, dict]:
    return {r["repo_id"]: r for r in manifest["repos"]}


# --- R1: manifest ------------------------------------------------------------


def test_manifest_schema_and_fields():
    m = _raw_manifest()
    assert m["schema"] == 1
    assert isinstance(m["repos"], list) and m["repos"]
    keys = [r["key"] for r in m["repos"]]
    assert len(keys) == len(set(keys)), "duplicate repo keys"
    repo_ids = [r["repo_id"] for r in m["repos"]]
    assert len(repo_ids) == len(set(repo_ids)), "duplicate repo ids"
    for r in m["repos"]:
        assert re.fullmatch(r"[0-9a-f]{40}", r["revision"]), r
        assert r["startup"] in {"required", "optional"}, r
        assert r["files"], r
        paths = [f["path"] for f in r["files"]]
        assert len(paths) == len(set(paths)), f"duplicate paths in {r['repo_id']}"
        for f in r["files"]:
            assert isinstance(f["size"], int) and f["size"] > 0, f
            assert not f["path"].startswith("/") and ".." not in f["path"].split("/"), f
            assert not any(ch in f["path"] for ch in "*?[]"), f"glob character in {f['path']}"


def test_manifest_pins_the_expected_repos_and_commits():
    repos = _by_repo(_raw_manifest())
    expected = {
        UMBRELLA: ("19671f406d603126926c1b7e2adc169acbcade22", "required"),
        XL_SFT: ("d06de46b4622f781cf07f4a013a67d591ca52819", "required"),
        QWEN: ("a09a35458c702b33eeacc393d103063234e8bc28", "optional"),
    }
    # LoRA presets are fetched on click by hf_hub_download (presets/manifest.json), not at startup.
    assert set(repos) == set(expected)
    for repo_id, (revision, startup) in expected.items():
        assert repos[repo_id]["revision"] == revision, repo_id
        assert repos[repo_id]["startup"] == startup, repo_id


def test_manifest_umbrella_has_components_but_no_turbo_weights():
    umbrella = _by_repo(_raw_manifest())[UMBRELLA]
    paths = {f["path"] for f in umbrella["files"]}
    turbo = {p for p in paths if p.startswith("acestep-v15-turbo/")}
    assert turbo == {"acestep-v15-turbo/config.json"}
    for component in ("vae/", "Qwen3-Embedding-0.6B/", "acestep-5Hz-lm-1.7B/"):
        assert any(p.startswith(component) for p in paths), component
    assert "vae/diffusion_pytorch_model.safetensors" in paths
    assert "Qwen3-Embedding-0.6B/model.safetensors" in paths
    assert "acestep-5Hz-lm-1.7B/model.safetensors" in paths
    # Top-level files of the umbrella are never read by the fork.
    assert not any("/" not in p for p in paths)


def test_manifest_xl_sft_and_qwen_files():
    repos = _by_repo(_raw_manifest())
    xl = {f["path"] for f in repos[XL_SFT]["files"]}
    assert xl == {
        "apg_guidance.py",
        "config.json",
        "configuration_acestep_v15.py",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "modeling_acestep_v15_xl_base.py",
        "silence_latent.pt",
    }
    qwen = {f["path"] for f in repos[QWEN]["files"]}
    assert qwen == {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }


# --- helpers for the resolver tests --------------------------------------------

REV = "a" * 40
OTHER_REV = "b" * 40
XET_IO_ERROR = "I/O error: I/O error: Permission denied (os error 13)"


def make_entry(files, *, repo_id="org/model", revision=REV, key="model", startup="required"):
    return {
        "key": key,
        "repo_id": repo_id,
        "revision": revision,
        "startup": startup,
        "files": [{"path": p, "size": s} for p, s in files.items()],
    }


FILES = {"config.json": 11, "sub/weights.bin": 64, "sub/tokenizer.json": 7}


def repo_dir(root, entry) -> Path:
    return Path(root) / ("models--" + entry["repo_id"].replace("/", "--"))


def write_snapshot(root, entry, *, revision=None, skip=(), sizes=None, dangling=()):
    """Lay out an HF-style cache entry: blobs/ plus relative symlinks in snapshots/<rev>/."""
    revision = revision or entry["revision"]
    rdir = repo_dir(root, entry)
    snap = rdir / "snapshots" / revision
    blobs = rdir / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(entry["files"]):
        dest = snap / f["path"]
        if f["path"] in skip or dest.is_symlink() or dest.exists():
            continue
        size = (sizes or {}).get(f["path"], f["size"])
        blob = blobs / f"{revision[:8]}-{i}"
        if f["path"] not in dangling:
            blob.write_bytes(b"x" * size)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(os.path.relpath(blob, dest.parent))
    return snap


def tree_state(path) -> dict:
    """Every path under ``path`` with mode, size and mtime, to prove nothing was written."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(path):
        for name in [*dirnames, *filenames]:
            p = os.path.join(dirpath, name)
            st = os.lstat(p)
            out[p] = (st.st_mode, st.st_size, st.st_mtime_ns)
    return out


def hub_error(name: str):
    """An instance of ``huggingface_hub.errors.<name>`` without calling its __init__.

    The constructor differs between hub 0.x (requests) and 1.x/2.x (httpx,
    response required), and only the type matters to the resolver.
    """
    errors = pytest.importorskip("huggingface_hub.errors")
    cls = getattr(errors, name, None)
    if cls is None:
        pytest.skip(f"huggingface_hub.errors.{name} does not exist in this version")
    err = cls.__new__(cls)
    Exception.__init__(err, f"{name}: org/model@{REV}")
    return err


class HttpStatusError(Exception):
    def __init__(self, status):
        super().__init__(f"{status} Client Error")
        self.response = type("Response", (), {"status_code": status})()


class FakeDownloader:
    """Stands in for ``huggingface_hub.snapshot_download`` and records every call.

    ``script`` maps repo_id to a list of actions consumed one per call: an
    exception instance is raised; ``"ok"`` writes the complete snapshot into
    ``cache_dir``; ``"partial"`` writes all but the last file; ``"nothing"``
    writes nothing (but still returns the snapshot path, as hub 0.36.2 does
    when repo_info fails); ``("incomplete", exc)`` leaves a ``*.incomplete``
    blob and raises ``exc``. Once a repo's list is exhausted the action is ``"ok"``.
    """

    def __init__(self, entries, script=None):
        self.entries = {e["repo_id"]: e for e in entries}
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls = []

    def __call__(self, repo_id, *, revision, allow_patterns, cache_dir, **kwargs):
        self.calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "allow_patterns": list(allow_patterns),
                "cache_dir": Path(cache_dir),
                "xet_disabled": getattr(hf_constants, "HF_HUB_DISABLE_XET", None),
                "hf_transfer": getattr(hf_constants, "HF_HUB_ENABLE_HF_TRANSFER", None),
                "kwargs": kwargs,
            }
        )
        actions = self.script.get(repo_id, [])
        action = actions.pop(0) if actions else "ok"
        entry = self.entries[repo_id]
        snap = ms.snapshot_path(cache_dir, entry)
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, tuple) and action[0] == "incomplete":
            blobs = repo_dir(cache_dir, entry) / "blobs"
            blobs.mkdir(parents=True, exist_ok=True)
            (blobs / "deadbeef.incomplete").write_bytes(b"half")
            raise action[1]
        if action == "nothing":
            snap.mkdir(parents=True, exist_ok=True)
            return str(snap)
        skip = (entry["files"][-1]["path"],) if action == "partial" else ()
        return str(write_snapshot(cache_dir, entry, skip=skip))


class FakeTime:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def _hub_transfer_globals(monkeypatch):
    """Start every test as on the Space: xet allowed, HF_HUB_ENABLE_HF_TRANSFER on.

    raising=False: hub 1.x/2.x (CI) may not define these; the resolver only
    toggles attributes that exist, and here they always do.
    """
    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_XET", False, raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_ENABLE_HF_TRANSFER", True, raising=False)


@pytest.fixture
def fake_time():
    return FakeTime()


@pytest.fixture
def ensure_kw(fake_time):
    return {"sleep": fake_time.sleep, "clock": fake_time.clock, "disk_free": lambda path: 10**15}


@pytest.fixture
def chmod_tree():
    """Make trees read-only (or a dir unreadable) and always restore them for cleanup."""
    touched = []

    def _readonly(path):
        for dirpath, _dirnames, filenames in os.walk(path):
            for name in filenames:
                p = os.path.join(dirpath, name)
                if not os.path.islink(p):
                    os.chmod(p, 0o444)
            os.chmod(dirpath, 0o555)
            touched.append(dirpath)

    def _mode(path, mode):
        os.chmod(path, mode)
        touched.append(str(path))

    yield type("Chmod", (), {"readonly": staticmethod(_readonly), "mode": staticmethod(_mode)})
    for p in reversed(touched):
        try:
            os.chmod(p, 0o755)
        except OSError:
            pass


def _permissions_enforced(path, mode) -> bool:
    return not os.access(path, mode)


@pytest.fixture
def roots(tmp_path):
    """The two download roots: the hub cache, then the app-owned root."""
    return [tmp_path / "hub", tmp_path / "app" / ".ams-cache" / "hub"]


def _cache_dirs(dl):
    return [(c["cache_dir"], c["xet_disabled"]) for c in dl.calls]


# --- A1: roots -------------------------------------------------------------------


def test_download_roots_are_the_hub_cache_then_the_app_root(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hf" / "hub"))
    assert ms.download_roots() == [tmp_path / "hf" / "hub", ms.APP_ROOT]
    assert ms.APP_ROOT == Path(ms.__file__).resolve().parent / ".ams-cache" / "hub"


def test_lookup_roots_add_the_home_caches_when_the_hub_cache_is_elsewhere(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hf" / "hub"))
    assert ms.lookup_roots() == [
        tmp_path / "hf" / "hub",
        tmp_path / "home" / ".cache" / "huggingface" / "hub",
        Path("/home/user/.cache/huggingface/hub"),
        ms.APP_ROOT,
    ]


def test_lookup_roots_do_not_repeat_the_hub_cache(monkeypatch, tmp_path):
    home_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(home_cache))
    roots = ms.lookup_roots()
    assert roots[0] == home_cache
    assert len(roots) == len(set(roots)) == 3


def test_load_manifest_reads_repo_manifest():
    m = ms.load_manifest()
    assert m == _raw_manifest()
    assert ms.get_repo(m, repo_id=QWEN)["key"] == ms.get_repo(m, key="qwen2.5-7b-instruct")["key"]
    assert ms.get_repo(m, key=ms.UMBRELLA_KEY)["repo_id"] == UMBRELLA
    assert ms.get_repo(m, key=ms.XL_SFT_KEY)["repo_id"] == XL_SFT
    with pytest.raises(KeyError):
        ms.get_repo(m, repo_id="nobody/nothing")


# --- A2 / R9.2: complete snapshot check --------------------------------------------


def test_is_complete_for_full_snapshot(roots):
    entry = make_entry(FILES)
    snap = write_snapshot(roots[0], entry)
    assert ms.is_complete(roots[0], entry)
    assert ms.snapshot_path(roots[0], entry) == snap
    assert ms.missing_files(roots[0], entry) == []


def test_is_complete_false_when_a_file_is_missing(roots):
    entry = make_entry(FILES)
    write_snapshot(roots[0], entry, skip=("sub/tokenizer.json",))
    assert not ms.is_complete(roots[0], entry)
    assert [f["path"] for f in ms.missing_files(roots[0], entry)] == ["sub/tokenizer.json"]


def test_is_complete_false_on_size_mismatch(roots):
    entry = make_entry(FILES)
    write_snapshot(roots[0], entry, sizes={"sub/weights.bin": 63})
    assert not ms.is_complete(roots[0], entry)
    assert [f["path"] for f in ms.missing_files(roots[0], entry)] == ["sub/weights.bin"]


def test_is_complete_false_on_dangling_symlink(roots):
    entry = make_entry(FILES)
    snap = write_snapshot(roots[0], entry, dangling=("config.json",))
    assert (snap / "config.json").is_symlink() and not (snap / "config.json").exists()
    assert not ms.is_complete(roots[0], entry)


def test_is_complete_false_when_a_file_is_unreadable(roots, chmod_tree):
    entry = make_entry(FILES)
    snap = write_snapshot(roots[0], entry)
    blob = os.path.realpath(snap / "sub" / "weights.bin")
    chmod_tree.mode(blob, 0o000)
    if not _permissions_enforced(blob, os.R_OK):
        pytest.skip("permissions are not enforced for this user")
    assert [f["path"] for f in ms.missing_files(roots[0], entry)] == ["sub/weights.bin"]


def test_is_complete_false_when_only_another_commit_is_cached(roots):
    entry = make_entry(FILES)
    write_snapshot(roots[0], entry, revision=OTHER_REV)
    assert not ms.is_complete(roots[0], entry)
    assert ms.find_complete(entry, roots) is None


def test_find_complete_returns_second_root_when_only_it_is_complete(roots):
    entry = make_entry(FILES)
    write_snapshot(roots[0], entry, skip=("config.json",))
    snap = write_snapshot(roots[1], entry)
    assert ms.find_complete(entry, roots) == snap


def test_find_complete_treats_unreadable_root_as_incomplete(roots, chmod_tree):
    entry = make_entry(FILES)
    write_snapshot(roots[0], entry)
    chmod_tree.mode(roots[0], 0o000)
    if not _permissions_enforced(roots[0], os.R_OK):
        pytest.skip("permissions are not enforced for this user")
    assert ms.find_complete(entry, roots) is None


# --- local lookup (GPU worker) ------------------------------------------------------


def test_local_snapshot_finds_complete_snapshot_offline(roots, monkeypatch):
    entry = make_entry(FILES, repo_id="Qwen/Tiny")
    manifest = {"schema": 1, "repos": [entry]}
    assert ms.local_snapshot("Qwen/Tiny", manifest=manifest, roots=roots) is None
    snap = write_snapshot(roots[1], entry)
    assert ms.local_snapshot("Qwen/Tiny", manifest=manifest, roots=roots) == snap

    # Defaults come from the repo manifest and lookup_roots().
    monkeypatch.setattr(ms, "load_manifest", lambda: manifest)
    monkeypatch.setattr(ms, "lookup_roots", lambda: roots)
    assert ms.local_snapshot("Qwen/Tiny") == snap


# --- A3 / R9.3: ensure ----------------------------------------------------------------


def test_ensure_returns_readonly_complete_snapshot_without_download_or_write(roots, chmod_tree, ensure_kw):
    entry = make_entry(FILES)
    snap = write_snapshot(roots[0], entry)
    chmod_tree.readonly(roots[0])
    before = tree_state(roots[0].parent)
    dl = FakeDownloader([entry])

    assert ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw) == snap

    assert dl.calls == []
    assert tree_state(roots[0].parent) == before
    assert not roots[1].exists()


def test_ensure_uses_a_lookup_only_root_without_writing_to_it(tmp_path, roots, chmod_tree, ensure_kw, capsys):
    entry = make_entry(FILES)
    preload = tmp_path / "home-cache" / "hub"
    snap = write_snapshot(preload, entry)
    chmod_tree.readonly(preload)
    dl = FakeDownloader([entry])

    found = ms.ensure_repo(entry, roots, lookup=[roots[0], preload, roots[1]], downloader=dl, **ensure_kw)

    assert found == snap
    assert dl.calls == []
    assert not roots[0].exists() and not roots[1].exists()
    assert f"found complete in {preload}" in capsys.readouterr().err


def test_ensure_never_downloads_into_a_lookup_only_root(tmp_path, roots, ensure_kw):
    entry = make_entry(FILES)
    preload = tmp_path / "home-cache" / "hub"
    write_snapshot(preload, entry, skip=("config.json",))
    dl = FakeDownloader([entry])

    snap = ms.ensure_repo(entry, roots, lookup=[roots[0], preload, roots[1]], downloader=dl, **ensure_kw)

    assert snap == ms.snapshot_path(roots[0], entry)
    assert [c["cache_dir"] for c in dl.calls] == [roots[0]]


def test_ensure_downloads_pinned_files_into_first_root_when_cache_is_empty(roots, ensure_kw, capsys):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry])

    snap = ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert snap == ms.snapshot_path(roots[0], entry)
    assert ms.is_complete(roots[0], entry)
    assert len(dl.calls) == 1
    call = dl.calls[0]
    assert call["repo_id"] == "org/model"
    assert call["revision"] == REV
    assert call["allow_patterns"] == ["config.json", "sub/weights.bin", "sub/tokenizer.json"]
    assert call["cache_dir"] == roots[0]
    assert call["kwargs"] == {"max_workers": 1}
    assert (call["xet_disabled"], call["hf_transfer"]) == (False, True)
    err = capsys.readouterr().err
    assert f"downloaded 3 files, 0.00 GB in 0.0 s to {roots[0]} with xet on" in err


def test_ensure_reports_found_complete(roots, ensure_kw, capsys):
    entry = make_entry(FILES)
    write_snapshot(roots[1], entry)
    ms.ensure_repo(entry, roots, downloader=FakeDownloader([entry]), **ensure_kw)
    assert f"found complete in {roots[1]}" in capsys.readouterr().err


@pytest.mark.parametrize("action", ["partial", "nothing"])
def test_ensure_rechecks_files_after_every_download(roots, ensure_kw, action):
    # hub 0.36.2 returns the snapshot folder without files when repo_info fails
    # while a partial snapshot exists, and skips paths that match nothing.
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [action]})

    snap = ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert snap == ms.snapshot_path(roots[0], entry)
    assert _cache_dirs(dl) == [(roots[0], False), (roots[0], True)]


def test_ensure_retries_hf_xet_io_error_with_xet_disabled(roots, ensure_kw, fake_time, capsys):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [OSError(XET_IO_ERROR)]})

    snap = ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert snap == ms.snapshot_path(roots[0], entry)
    assert _cache_dirs(dl) == [(roots[0], False), (roots[0], True)]
    assert fake_time.sleeps == []
    err = capsys.readouterr().err
    assert "OSError: " + XET_IO_ERROR in err
    assert "with xet off" in err


@pytest.mark.parametrize(
    "error",
    [
        ValueError("hf_xet configuration error"),
        RuntimeError("Data processing error: CAS service error"),
        RuntimeError("An error occurred while downloading using `hf_transfer`."),
        OSError(XET_IO_ERROR),
        ConnectionError("Max retries exceeded"),
        TimeoutError("read timed out"),
        ImportError("cannot import name 'download_files' from 'hf_xet'"),
    ],
)
def test_ensure_retry_is_type_agnostic_and_restores_hub_globals(roots, ensure_kw, error):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [error]})

    snap = ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert snap == ms.snapshot_path(roots[0], entry)
    assert [(c["cache_dir"], c["xet_disabled"], c["hf_transfer"]) for c in dl.calls] == [
        (roots[0], False, True),
        (roots[0], True, False),  # xet off: plain HTTP, hf_transfer off too
    ]
    assert hf_constants.HF_HUB_DISABLE_XET is False
    assert hf_constants.HF_HUB_ENABLE_HF_TRANSFER is True


def test_ensure_restores_hub_globals_when_the_xet_off_attempt_fails(roots, ensure_kw):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [OSError(XET_IO_ERROR), RuntimeError("still broken")]})

    ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert _cache_dirs(dl) == [(roots[0], False), (roots[0], True), (roots[1], False)]
    assert hf_constants.HF_HUB_DISABLE_XET is False
    assert hf_constants.HF_HUB_ENABLE_HF_TRANSFER is True


def test_ensure_marks_root_unusable_on_permission_error(roots, ensure_kw):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [PermissionError(13, "Permission denied", str(roots[0]))]})

    snap = ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert snap == ms.snapshot_path(roots[1], entry)
    # The xet-off strategy of the unusable root is skipped.
    assert _cache_dirs(dl) == [(roots[0], False), (roots[1], False)]


@pytest.mark.parametrize("code", ["EACCES", "EPERM", "EROFS", "ENOSPC", "EDQUOT"])
def test_ensure_marks_root_unusable_on_write_errno(roots, ensure_kw, code):
    entry = make_entry(FILES)
    num = getattr(errno, code, None)
    if num is None:
        pytest.skip(f"errno.{code} does not exist on this platform")
    dl = FakeDownloader([entry], {"org/model": [OSError(num, os.strerror(num))]})
    assert ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw) == ms.snapshot_path(roots[1], entry)
    assert _cache_dirs(dl) == [(roots[0], False), (roots[1], False)]


def test_ensure_skips_root_that_fails_the_write_probe(roots, chmod_tree, ensure_kw):
    entry = make_entry(FILES)
    roots[0].mkdir(parents=True)
    chmod_tree.mode(roots[0], 0o555)
    if not _permissions_enforced(roots[0], os.W_OK):
        pytest.skip("permissions are not enforced for this user")
    dl = FakeDownloader([entry])
    assert ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw) == ms.snapshot_path(roots[1], entry)
    assert [c["cache_dir"] for c in dl.calls] == [roots[1]]


@pytest.mark.parametrize(
    "make_error", ["RevisionNotFoundError", "RepositoryNotFoundError", "GatedRepoError", 401, 403]
)
def test_ensure_fails_fast_when_the_repo_or_commit_is_not_reachable(roots, ensure_kw, fake_time, make_error):
    error = HttpStatusError(make_error) if isinstance(make_error, int) else hub_error(make_error)
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [error] * 10})

    with pytest.raises(ms.ModelStoreError) as exc:
        ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert len(dl.calls) == 1
    assert fake_time.sleeps == []
    assert "org/model@" + REV in str(exc.value)
    assert type(error).__name__ in str(exc.value)


def test_ensure_retries_every_strategy_after_a_backoff(roots, ensure_kw, fake_time):
    entry = make_entry(FILES)
    boom = ConnectionError("Max retries exceeded")
    dl = FakeDownloader([entry], {"org/model": [boom] * 4})

    snap = ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert snap == ms.snapshot_path(roots[0], entry)
    assert _cache_dirs(dl) == [
        (roots[0], False),
        (roots[0], True),
        (roots[1], False),
        (roots[1], True),
        (roots[0], False),
    ]
    assert fake_time.sleeps == [5]


def test_ensure_backoff_doubles_and_is_capped(roots, ensure_kw, fake_time):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [TimeoutError("read timed out")] * (4 * 6)})
    ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)
    assert fake_time.sleeps == [5, 10, 20, 40, 60, 60]
    assert len(dl.calls) == 4 * 6 + 1


def test_ensure_raises_the_summary_when_the_budget_is_spent(roots, ensure_kw, fake_time):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [ConnectionError("Max retries exceeded")] * 100})

    with pytest.raises(ms.ModelStoreError) as exc:
        ms.ensure_repo(entry, roots, downloader=dl, deadline=fake_time.clock() + 30, **ensure_kw)

    # pass at t=0, sleep 5; pass at t=5, sleep 10; pass at t=15; sleeping 20 would pass t=30.
    assert fake_time.sleeps == [5, 10]
    assert len(dl.calls) == 12
    msg = str(exc.value)
    assert "budget" in msg and "ConnectionError" in msg


def test_ensure_does_not_start_downloading_after_the_budget_is_spent(roots, ensure_kw, fake_time):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry])
    with pytest.raises(ms.ModelStoreError, match="budget"):
        ms.ensure_repo(entry, roots, downloader=dl, deadline=fake_time.clock() - 1, **ensure_kw)
    assert dl.calls == []


def test_ensure_returns_a_complete_snapshot_even_after_the_budget_is_spent(roots, ensure_kw, fake_time):
    entry = make_entry(FILES)
    snap = write_snapshot(roots[1], entry)
    dl = FakeDownloader([entry])
    assert ms.ensure_repo(entry, roots, downloader=dl, deadline=fake_time.clock() - 1, **ensure_kw) == snap


def test_ensure_raises_at_once_when_every_root_is_unusable(roots, ensure_kw, fake_time):
    entry = make_entry(FILES, repo_id="ACE-Step/Some-Repo")
    denied = PermissionError(13, "Permission denied")
    dl = FakeDownloader([entry], {"ACE-Step/Some-Repo": [denied, denied]})

    with pytest.raises(ms.ModelStoreError) as exc:
        ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert len(dl.calls) == 2
    assert fake_time.sleeps == []
    msg = str(exc.value)
    assert "ACE-Step/Some-Repo@" + REV in msg
    for root in roots:
        assert str(root) in msg
    assert "PermissionError" in msg


def test_ensure_skips_root_without_enough_free_space(roots, fake_time, capsys):
    entry = make_entry(FILES)
    free = {roots[0]: 2 * 10**9, roots[1]: 10**15}
    dl = FakeDownloader([entry])

    snap = ms.ensure_repo(
        entry,
        roots,
        downloader=dl,
        sleep=fake_time.sleep,
        clock=fake_time.clock,
        disk_free=lambda path: free[Path(path)],
    )

    # 82 bytes missing + 2 GB margin does not fit in exactly 2 GB.
    assert snap == ms.snapshot_path(roots[1], entry)
    assert [c["cache_dir"] for c in dl.calls] == [roots[1]]
    assert f"skipping {roots[0]}: not enough disk" in capsys.readouterr().err


def test_ensure_measures_free_space_on_a_root_that_does_not_exist_yet(tmp_path, fake_time):
    entry = make_entry(FILES)
    roots = [tmp_path / "fresh" / "cache" / "hub", tmp_path / "fresh" / "app" / "hub"]
    assert not roots[0].parent.parent.exists()
    dl = FakeDownloader([entry])

    # The real disk check: shutil.disk_usage raises on a missing path.
    snap = ms.ensure_repo(entry, roots, downloader=dl, sleep=fake_time.sleep, clock=fake_time.clock)

    assert snap == ms.snapshot_path(roots[0], entry)


def test_default_disk_free_uses_nearest_existing_ancestor(tmp_path):
    free = ms._disk_free(tmp_path / "does" / "not" / "exist")
    assert isinstance(free, int) and free > 0


def test_failed_attempt_in_the_app_root_removes_its_incomplete_blobs(roots, ensure_kw):
    entry = make_entry(FILES)
    script = [PermissionError(13, "Permission denied"), ("incomplete", ConnectionError("reset by peer"))]
    dl = FakeDownloader([entry], {"org/model": script})

    snap = ms.ensure_repo(entry, roots, owned=[roots[1]], downloader=dl, **ensure_kw)

    assert snap == ms.snapshot_path(roots[1], entry)
    assert list((repo_dir(roots[1], entry) / "blobs").glob("*.incomplete")) == []


def test_failed_attempt_in_the_hub_cache_leaves_its_files_alone(roots, ensure_kw):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [("incomplete", ConnectionError("reset by peer"))]})

    ms.ensure_repo(entry, roots, owned=[roots[1]], downloader=dl, **ensure_kw)

    assert [p.name for p in (repo_dir(roots[0], entry) / "blobs").glob("*.incomplete")] == [
        "deadbeef.incomplete"
    ]


def test_ensure_summary_fits_the_space_log_tail_even_with_long_errors(roots, ensure_kw, fake_time):
    entry = make_entry(FILES, repo_id="ACE-Step/Some-Repo")
    huge = OSError("I/O error: " + "x" * 5000 + " (os error 13)")
    dl = FakeDownloader([entry], {"ACE-Step/Some-Repo": [huge] * 10})

    with pytest.raises(ms.ModelStoreError) as exc:
        ms.ensure_repo(entry, roots, downloader=dl, deadline=fake_time.clock() + 1, **ensure_kw)

    msg = str(exc.value)
    # app.py adds one short line; the Space API keeps the last 2000 characters.
    assert len(msg) < 1900, len(msg)
    assert msg.startswith(f"could not get ACE-Step/Some-Repo@{REV}")
    for root in roots:
        assert str(root) in msg


# --- R9.4: the 2026-09-22 state -------------------------------------------------


def _tiny_real_manifest() -> dict:
    """The real manifest's repos, paths and commits, with tiny sizes so files can be written."""
    m = _raw_manifest()
    for repo in m["repos"]:
        for i, f in enumerate(repo["files"]):
            f["size"] = 10 + i
    return m


@pytest.fixture
def no_network(monkeypatch):
    import socket

    def _refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


def test_startup_in_2026_09_22_state_needs_no_download(tmp_path, roots, chmod_tree, monkeypatch, no_network):
    manifest = _tiny_real_manifest()
    for repo in manifest["repos"]:
        if repo["startup"] in ("required", "optional"):
            write_snapshot(roots[0], repo)
    umbrella_snap = ms.snapshot_path(roots[0], ms.get_repo(manifest, repo_id=UMBRELLA))
    # Complete for the manifest, incomplete for the whole repo: no turbo weights, no README.
    assert (umbrella_snap / "acestep-v15-turbo" / "config.json").exists()
    assert not (umbrella_snap / "acestep-v15-turbo" / "model.safetensors").exists()
    assert not (umbrella_snap / "README.md").exists()
    chmod_tree.readonly(roots[0])
    xet = tmp_path / "xet"
    xet.mkdir()
    chmod_tree.mode(xet, 0o500)
    monkeypatch.setenv("HF_XET_CACHE", str(xet))
    monkeypatch.setattr(hf_constants, "HF_XET_CACHE", str(xet), raising=False)
    calls = []

    def _downloader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("downloader called")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _downloader)
    before = tree_state(roots[0])
    ckpt = tmp_path / "checkpoints"

    resolved = ms.startup(ckpt, manifest=manifest, roots=roots, owned=[roots[1]])

    assert calls == []
    assert tree_state(roots[0]) == before
    assert set(resolved) == {"ace-step", "ace-step-xl-sft", "qwen2.5-7b-instruct"}
    for name in ms.UMBRELLA_LINKS:
        assert (ckpt / name).is_symlink()
        assert os.path.samefile(ckpt / name, umbrella_snap / name)
    assert os.path.samefile(ckpt / ms.XL_SFT_LINK, resolved["ace-step-xl-sft"])
    assert not roots[1].exists()


# --- R9.5: upstream moved ---------------------------------------------------------


def test_ensure_uses_pinned_snapshot_when_refs_main_moved(roots, ensure_kw, chmod_tree):
    entry = make_entry(FILES)
    snap = write_snapshot(roots[0], entry)
    write_snapshot(roots[0], entry, revision=OTHER_REV, skip=("config.json",))
    refs = repo_dir(roots[0], entry) / "refs"
    refs.mkdir()
    (refs / "main").write_text(OTHER_REV)
    chmod_tree.readonly(roots[0])
    dl = FakeDownloader([entry])

    assert ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw) == snap
    assert dl.calls == []


# --- R2.5 / R9.6: checkpoint links ------------------------------------------------


@pytest.fixture
def snapshots(tmp_path):
    umbrella = tmp_path / "umbrella"
    for name in ms.UMBRELLA_LINKS:
        (umbrella / name).mkdir(parents=True)
    xl = tmp_path / "xl"
    xl.mkdir()
    return umbrella, xl


def _assert_links(ckpt, umbrella, xl):
    for name in ms.UMBRELLA_LINKS:
        assert (ckpt / name).is_symlink(), name
        assert os.path.samefile(ckpt / name, umbrella / name), name
    assert (ckpt / ms.XL_SFT_LINK).is_symlink()
    assert os.path.samefile(ckpt / ms.XL_SFT_LINK, xl)


def test_link_names_match_the_fork_components():
    assert ms.UMBRELLA_LINKS == ("vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B", "acestep-v15-turbo")
    assert ms.XL_SFT_LINK == "acestep-v15-xl-sft"


def test_link_checkpoints_creates_links(tmp_path, snapshots):
    umbrella, xl = snapshots
    ckpt = tmp_path / "vendor" / "checkpoints"
    ms.link_checkpoints(ckpt, umbrella, xl)
    _assert_links(ckpt, umbrella, xl)


def test_link_checkpoints_is_idempotent(tmp_path, snapshots):
    umbrella, xl = snapshots
    ckpt = tmp_path / "checkpoints"
    ms.link_checkpoints(ckpt, umbrella, xl)
    before = tree_state(ckpt)
    ms.link_checkpoints(ckpt, umbrella, xl)
    assert tree_state(ckpt) == before
    _assert_links(ckpt, umbrella, xl)


def test_link_checkpoints_repairs_dangling_link(tmp_path, snapshots):
    umbrella, xl = snapshots
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    (ckpt / "vae").symlink_to(tmp_path / "gone")
    (ckpt / ms.XL_SFT_LINK).symlink_to(tmp_path / "gone-too")
    ms.link_checkpoints(ckpt, umbrella, xl)
    _assert_links(ckpt, umbrella, xl)


def test_link_checkpoints_replaces_wrong_target(tmp_path, snapshots):
    umbrella, xl = snapshots
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    elsewhere = tmp_path / "old-snapshot" / "vae"
    elsewhere.mkdir(parents=True)
    (ckpt / "vae").symlink_to(elsewhere)
    ms.link_checkpoints(ckpt, umbrella, xl)
    _assert_links(ckpt, umbrella, xl)
    assert elsewhere.is_dir()  # the old target itself is untouched


def test_link_checkpoints_leaves_real_directory_alone_with_warning(tmp_path, snapshots, capsys):
    umbrella, xl = snapshots
    ckpt = tmp_path / "checkpoints"
    (ckpt / "vae").mkdir(parents=True)
    (ckpt / "vae" / "keep.txt").write_text("user data")
    (ckpt / "acestep-v15-turbo").write_text("a file in the way")

    ms.link_checkpoints(ckpt, umbrella, xl)

    assert not (ckpt / "vae").is_symlink()
    assert (ckpt / "vae" / "keep.txt").read_text() == "user data"
    assert (ckpt / "acestep-v15-turbo").read_text() == "a file in the way"
    err = capsys.readouterr().err
    assert "WARNING" in err and str(ckpt / "vae") in err and str(ckpt / "acestep-v15-turbo") in err
    for name in ("Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B", ms.XL_SFT_LINK):
        assert (ckpt / name).is_symlink()


# --- R2.7 / A12 / R9.7: the startup entry point -------------------------------------


def _startup_manifest():
    return {
        "schema": 1,
        "repos": [
            make_entry(
                {f"{n}/f.bin": 5 for n in ms.UMBRELLA_LINKS}, repo_id="o/umbrella", key=ms.UMBRELLA_KEY
            ),
            make_entry({"model.bin": 9}, repo_id="o/xl", key=ms.XL_SFT_KEY),
            make_entry({"model.bin": 9}, repo_id="o/qwen", key="qwen", startup="optional"),
        ],
    }


def test_startup_downloads_required_then_links_then_optional(tmp_path, roots, ensure_kw, capsys):
    manifest = _startup_manifest()
    ckpt = tmp_path / "checkpoints"
    seen_links_before_optional = []

    class Recorder(FakeDownloader):
        def __call__(self, repo_id, **kwargs):
            if repo_id == "o/qwen":
                seen_links_before_optional.append((ckpt / ms.XL_SFT_LINK).is_symlink())
            return super().__call__(repo_id, **kwargs)

    dl = Recorder(manifest["repos"])

    resolved = ms.startup(ckpt, manifest=manifest, roots=roots, downloader=dl, **ensure_kw)

    assert [c["repo_id"] for c in dl.calls] == ["o/umbrella", "o/xl", "o/qwen"]
    assert seen_links_before_optional == [True]
    assert set(resolved) == {ms.UMBRELLA_KEY, ms.XL_SFT_KEY, "qwen"}
    _assert_links(ckpt, resolved[ms.UMBRELLA_KEY], resolved[ms.XL_SFT_KEY])
    err = capsys.readouterr().err
    assert "[ams] python" in err


def test_startup_raises_when_a_required_repo_fails(tmp_path, roots, ensure_kw):
    manifest = _startup_manifest()
    denied = [PermissionError(13, "Permission denied")] * 2
    dl = FakeDownloader(manifest["repos"], {"o/xl": denied})
    ckpt = tmp_path / "checkpoints"

    with pytest.raises(ms.ModelStoreError) as exc:
        ms.startup(ckpt, manifest=manifest, roots=roots, downloader=dl, **ensure_kw)

    assert "o/xl" in str(exc.value)
    assert not (ckpt / ms.XL_SFT_LINK).exists()
    assert "o/qwen" not in [c["repo_id"] for c in dl.calls]


def test_startup_continues_when_an_optional_repo_fails(tmp_path, roots, ensure_kw, capsys):
    manifest = _startup_manifest()
    denied = [PermissionError(13, "Permission denied")] * 2
    dl = FakeDownloader(manifest["repos"], {"o/qwen": denied})
    ckpt = tmp_path / "checkpoints"

    resolved = ms.startup(ckpt, manifest=manifest, roots=roots, downloader=dl, **ensure_kw)

    assert set(resolved) == {ms.UMBRELLA_KEY, ms.XL_SFT_KEY}
    _assert_links(ckpt, resolved[ms.UMBRELLA_KEY], resolved[ms.XL_SFT_KEY])
    err = capsys.readouterr().err
    assert "o/qwen" in err and "continuing without it" in err


def test_startup_continues_when_an_optional_repo_does_not_fit_on_disk(tmp_path, roots, fake_time, capsys):
    manifest = _startup_manifest()
    manifest["repos"][2]["files"][0]["size"] = 10**13  # 10 TB
    dl = FakeDownloader(manifest["repos"])

    resolved = ms.startup(
        tmp_path / "checkpoints",
        manifest=manifest,
        roots=roots,
        downloader=dl,
        sleep=fake_time.sleep,
        clock=fake_time.clock,
        disk_free=lambda path: 10**12,
    )

    assert "qwen" not in resolved
    assert "o/qwen" not in [c["repo_id"] for c in dl.calls]
    assert "not enough disk" in capsys.readouterr().err


def test_startup_budget_spent_during_a_required_repo_raises_the_summary(
    tmp_path, roots, ensure_kw, fake_time
):
    manifest = _startup_manifest()
    dl = FakeDownloader(manifest["repos"], {"o/xl": [ConnectionError("Max retries exceeded")] * 100})

    with pytest.raises(ms.ModelStoreError) as exc:
        ms.startup(
            tmp_path / "checkpoints", manifest=manifest, roots=roots, downloader=dl, budget_s=30, **ensure_kw
        )

    assert "o/xl" in str(exc.value) and "budget" in str(exc.value)
    assert fake_time.now <= 1000 + 30


def test_startup_budget_spent_during_an_optional_repo_skips_it(tmp_path, roots, ensure_kw, fake_time, capsys):
    manifest = _startup_manifest()
    dl = FakeDownloader(manifest["repos"], {"o/qwen": [ConnectionError("Max retries exceeded")] * 100})

    resolved = ms.startup(
        tmp_path / "checkpoints", manifest=manifest, roots=roots, downloader=dl, budget_s=30, **ensure_kw
    )

    assert set(resolved) == {ms.UMBRELLA_KEY, ms.XL_SFT_KEY}
    err = capsys.readouterr().err
    assert "o/qwen" in err and "budget" in err
    assert fake_time.now <= 1000 + 30


def test_startup_budget_is_shared_by_all_repos(tmp_path, roots, ensure_kw, fake_time):
    manifest = _startup_manifest()
    # The umbrella burns the budget in backoffs before it succeeds; nothing is left for the XL repo.
    boom = ConnectionError("Max retries exceeded")
    dl = FakeDownloader(manifest["repos"], {"o/umbrella": [boom] * 8, "o/xl": [boom] * 100})

    with pytest.raises(ms.ModelStoreError, match="o/xl"):
        ms.startup(
            tmp_path / "checkpoints", manifest=manifest, roots=roots, downloader=dl, budget_s=20, **ensure_kw
        )

    # Umbrella: failed passes at t=0 and t=5, success at t=15. XL: one failed pass at t=15; a 5 s
    # backoff would end exactly at the deadline, so no second pass and no pointless sleep.
    assert fake_time.sleeps == [5, 10]
    assert [c["repo_id"] for c in dl.calls].count("o/xl") == 4


@pytest.fixture
def no_budget_env(monkeypatch):
    monkeypatch.delenv("AMS_BOOTSTRAP_BUDGET_S", raising=False)


def test_bootstrap_budget_defaults_to_25_minutes(no_budget_env, capsys):
    assert ms.bootstrap_budget_s() == ms.BUDGET_S == 25 * 60
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("value", ["30", "30.5", " 45 "])
def test_bootstrap_budget_env_override_is_honoured(monkeypatch, capsys, value):
    monkeypatch.setenv("AMS_BOOTSTRAP_BUDGET_S", value)
    assert ms.bootstrap_budget_s() == float(value)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("value", ["abc", "0", "-5", "nan", "inf", "", "1e999"])
def test_bootstrap_budget_bad_value_falls_back_to_the_default(monkeypatch, capsys, value):
    monkeypatch.setenv("AMS_BOOTSTRAP_BUDGET_S", value)
    assert ms.bootstrap_budget_s() == ms.BUDGET_S
    lines = capsys.readouterr().err.splitlines()
    if value == "":
        assert lines == []  # empty means unset
    else:
        assert len(lines) == 1 and lines[0].startswith("[ams] ") and "AMS_BOOTSTRAP_BUDGET_S" in lines[0]


def test_startup_uses_the_budget_from_the_environment(
    tmp_path, roots, ensure_kw, fake_time, monkeypatch, capsys
):
    monkeypatch.setenv("AMS_BOOTSTRAP_BUDGET_S", "30")
    manifest = _startup_manifest()
    dl = FakeDownloader(manifest["repos"], {"o/xl": [ConnectionError("Max retries exceeded")] * 100})

    with pytest.raises(ms.ModelStoreError, match="budget"):
        ms.startup(tmp_path / "checkpoints", manifest=manifest, roots=roots, downloader=dl, **ensure_kw)

    assert fake_time.sleeps == [5, 10]  # 25 minutes would have meant many more passes
    assert "bootstrap budget 30 s" in capsys.readouterr().err


def test_startup_falls_back_to_the_default_budget_on_a_bad_value(
    tmp_path, roots, ensure_kw, monkeypatch, capsys
):
    monkeypatch.setenv("AMS_BOOTSTRAP_BUDGET_S", "soon")
    manifest = _startup_manifest()

    ms.startup(
        tmp_path / "checkpoints",
        manifest=manifest,
        roots=roots,
        downloader=FakeDownloader(manifest["repos"]),
        **ensure_kw,
    )

    err = capsys.readouterr().err
    assert "AMS_BOOTSTRAP_BUDGET_S='soon'" in err
    assert f"bootstrap budget {ms.BUDGET_S:.0f} s" in err


def test_startup_disables_hub_progress_bars_and_restores_them(tmp_path, roots, ensure_kw):
    hf_utils = pytest.importorskip("huggingface_hub.utils")
    are_disabled = getattr(hf_utils, "are_progress_bars_disabled", None)
    if are_disabled is None or not hasattr(hf_utils, "enable_progress_bars"):
        pytest.skip("this huggingface_hub has no progress-bar switch")
    hf_utils.enable_progress_bars()
    if are_disabled():
        pytest.skip("progress bars are disabled by the environment")
    manifest = _startup_manifest()
    seen = []

    class Recorder(FakeDownloader):
        def __call__(self, repo_id, **kwargs):
            seen.append(are_disabled())
            return super().__call__(repo_id, **kwargs)

    ms.startup(
        tmp_path / "checkpoints",
        manifest=manifest,
        roots=roots,
        downloader=Recorder(manifest["repos"]),
        **ensure_kw,
    )

    assert seen == [True, True, True]
    assert are_disabled() is False


def test_startup_keeps_progress_bars_disabled_if_they_were(tmp_path, roots, ensure_kw):
    hf_utils = pytest.importorskip("huggingface_hub.utils")
    if not all(hasattr(hf_utils, n) for n in ("are_progress_bars_disabled", "disable_progress_bars")):
        pytest.skip("this huggingface_hub has no progress-bar switch")
    hf_utils.disable_progress_bars()
    try:
        manifest = _startup_manifest()
        ms.startup(
            tmp_path / "checkpoints",
            manifest=manifest,
            roots=roots,
            downloader=FakeDownloader(manifest["repos"]),
            **ensure_kw,
        )
        assert hf_utils.are_progress_bars_disabled() is True
    finally:
        hf_utils.enable_progress_bars()


# --- R2.6: diagnostics ----------------------------------------------------------------


def test_diagnostics_report_versions_ids_env_and_paths(tmp_path, roots, monkeypatch, capsys):
    monkeypatch.setenv("HF_XET_CACHE", "/tmp/ams-hf-xet")
    monkeypatch.delenv("HF_HOME", raising=False)
    ms.log_diagnostics(roots, app_dir=tmp_path)
    lines = capsys.readouterr().err.splitlines()
    assert lines and all(line.startswith("[ams] ") for line in lines)
    text = "\n".join(lines)
    assert f"python {sys.version_info.major}.{sys.version_info.minor}" in text
    for pkg in (
        "torch",
        "torchaudio",
        "torchcodec",
        "transformers",
        "huggingface_hub",
        "hf_xet",
        "gradio",
        "spaces",
    ):
        assert f"{pkg}=" in text, pkg
    assert f"uid={os.getuid()}" in text and f"gid={os.getgid()}" in text
    assert "HF_XET_CACHE=/tmp/ams-hf-xet" in text and "HF_HOME=<unset>" in text
    for var in ("HOME", "HF_HUB_CACHE", "TORCH_HOME", "HF_MODULES_CACHE"):
        assert f"{var}=" in text, var
    for root in roots:
        assert str(root) in text
    assert str(tmp_path) in text
    assert "writable=" in text and "free=" in text and "mode=" in text and "owner=" in text


def test_diagnostics_print_the_effective_hub_cache_and_budget(tmp_path, roots, monkeypatch, capsys):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "effective" / "hub"), raising=False)
    ms.log_diagnostics(roots, app_dir=tmp_path, budget_s=90)
    lines = capsys.readouterr().err.splitlines()
    env = next(line for line in lines if line.startswith("[ams] env "))
    assert "HF_HUB_CACHE=<unset>" in env
    assert f"hub cache: {tmp_path / 'effective' / 'hub'}" in env
    assert any(line == "[ams] bootstrap budget 90 s" for line in lines)


def test_diagnostics_never_raise_with_unreadable_path(tmp_path, monkeypatch, capsys, chmod_tree):
    home = tmp_path / "home"
    (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        hf_constants, "HF_HUB_CACHE", str(home / ".cache" / "huggingface" / "hub"), raising=False
    )
    monkeypatch.setattr(
        hf_constants, "HF_XET_CACHE", str(home / ".cache" / "huggingface" / "xet"), raising=False
    )
    chmod_tree.mode(home / ".cache", 0o000)
    roots = [home / ".cache" / "huggingface" / "hub", tmp_path / "app", tmp_path / "tmp"]

    ms.log_diagnostics(roots, app_dir=tmp_path / "app")

    text = capsys.readouterr().err
    assert str(home / ".cache" / "huggingface") in text


def test_diagnostics_never_raise_when_every_probe_fails(tmp_path, roots, monkeypatch, capsys):
    import importlib.metadata
    import shutil

    def _boom(*args, **kwargs):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(importlib.metadata, "version", _boom)
    monkeypatch.setattr(shutil, "disk_usage", _boom)
    monkeypatch.setattr(os, "getuid", _boom)
    monkeypatch.setattr(os, "stat", _boom)
    monkeypatch.setattr(os, "access", _boom)

    ms.log_diagnostics(roots, app_dir=tmp_path)

    assert "[ams]" in capsys.readouterr().err


def _diag_text(capsys, roots, app_dir, **kw) -> str:
    ms.log_diagnostics(roots, app_dir=app_dir, **kw)
    return capsys.readouterr().err


def test_diagnostics_report_the_filesystem_type_of_each_root(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "overlay / overlay rw 0 0\n"
        "tmpfs /tmp tmpfs rw,nosuid 0 0\n"
        f"/dev/sdb1 {os.path.realpath(data)} ext4 rw 0 0\n"
    )
    roots = [data / "hub", tmp_path / "elsewhere" / "hub"]

    text = _diag_text(capsys, roots, tmp_path, mounts_path=str(mounts))

    root1 = next(line for line in text.splitlines() if f"root1 {roots[0]}:" in line)
    assert f"fs=ext4({os.path.realpath(data)})" in root1


def test_diagnostics_without_proc_mounts_say_so(tmp_path, roots, capsys):
    text = _diag_text(capsys, roots, tmp_path, mounts_path=str(tmp_path / "no-such-mounts"))
    assert "fs=n/a" in text


def test_diagnostics_report_whether_the_vendored_handler_exists(tmp_path, roots, capsys):
    assert "acestep/handler.py exists=no" in _diag_text(capsys, roots, tmp_path)
    handler = tmp_path / "vendor" / "ace-step" / "acestep" / "handler.py"
    handler.parent.mkdir(parents=True)
    handler.write_text("")
    assert "acestep/handler.py exists=yes" in _diag_text(capsys, roots, tmp_path)


def test_diagnostics_report_torch_cuda_only_when_torch_is_already_imported(
    tmp_path, roots, monkeypatch, capsys
):
    import types

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert "torch.version.cuda=<torch not imported>" in _diag_text(capsys, roots, tmp_path)
    assert "torch" not in sys.modules  # the diagnostics never import torch

    fake = types.ModuleType("torch")
    fake.version = types.SimpleNamespace(cuda="13.0")
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert "torch.version.cuda=13.0" in _diag_text(capsys, roots, tmp_path)


def _fake_spaces(monkeypatch, get_config):
    import types

    config = types.ModuleType("spaces.zero.config")
    config.get_config = get_config
    zero = types.ModuleType("spaces.zero")
    zero.config = config
    spaces = types.ModuleType("spaces")
    spaces.zero = zero
    monkeypatch.setitem(sys.modules, "spaces", spaces)
    monkeypatch.setitem(sys.modules, "spaces.zero", zero)
    monkeypatch.setitem(sys.modules, "spaces.zero.config", config)


def _no_audio_test(*args, **kwargs):
    return subprocess.CompletedProcess(args, 0, stdout="AMS_AUDIO_OK fake\n", stderr="")


def test_diagnostics_on_spaces_report_the_zerogpu_duration_factor(tmp_path, roots, monkeypatch, capsys):
    _fake_spaces(monkeypatch, lambda: {"duration_factor": 1.5, "model": "RTX PRO 6000"})
    text = _diag_text(capsys, roots, tmp_path, on_spaces=True, audio_runner=_no_audio_test)
    assert "zerogpu duration_factor=1.5" in text


def test_diagnostics_survive_a_failing_zerogpu_lookup(tmp_path, roots, monkeypatch, capsys):
    def _broken():
        raise RuntimeError("no /proc/driver/nvidia")

    _fake_spaces(monkeypatch, _broken)
    text = _diag_text(capsys, roots, tmp_path, on_spaces=True, audio_runner=_no_audio_test)
    assert "zerogpu duration_factor=? (RuntimeError)" in text


def test_diagnostics_off_spaces_skip_zerogpu_and_the_audio_self_test(tmp_path, roots, capsys):
    def _runner(*args, **kwargs):
        raise AssertionError("audio self-test ran off Spaces")

    text = _diag_text(capsys, roots, tmp_path, on_spaces=False, audio_runner=_runner)
    assert "zerogpu" not in text and "audio self-test" not in text


def test_audio_self_test_passes_and_logs_versions(capsys):
    calls = []

    def _runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        out = "AMS_AUDIO_VERSIONS torch=2.13.0\nAMS_AUDIO_OK torch=2.13.0 torchaudio=2.11.0 shape=(1, 1600) sr=16000\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    assert ms.audio_selftest(runner=_runner) is True

    ((cmd, kwargs),) = calls
    assert cmd[0] == sys.executable and cmd[1] == "-c"
    for needle in ("import torch", "import torchaudio", "import torchcodec", "wave.open", "torchaudio.load"):
        assert needle in cmd[2], needle
    assert kwargs["timeout"] == 60
    assert kwargs["capture_output"] is True and kwargs["text"] is True
    err = capsys.readouterr().err
    assert "audio self-test: pass" in err and "torchaudio=2.11.0" in err


def test_audio_self_test_reports_a_failure(capsys):
    def _runner(cmd, **kwargs):
        stderr = "Traceback...\nOSError: libtorchaudio.so: undefined symbol: torch_exception_get_what\n"
        return subprocess.CompletedProcess(cmd, 1, stdout="AMS_AUDIO_VERSIONS torch=2.11.0\n", stderr=stderr)

    assert ms.audio_selftest(runner=_runner) is False
    err = capsys.readouterr().err
    assert "audio self-test: FAIL" in err and "undefined symbol" in err and "torch=2.11.0" in err


@pytest.mark.parametrize(
    "error",
    [subprocess.TimeoutExpired(cmd="python", timeout=60), FileNotFoundError("python"), RuntimeError("x")],
)
def test_audio_self_test_never_raises(capsys, error):
    def _runner(cmd, **kwargs):
        raise error

    assert ms.audio_selftest(runner=_runner) is False
    assert "audio self-test: FAIL" in capsys.readouterr().err


def test_diagnostics_on_spaces_never_raise_when_everything_fails(tmp_path, roots, monkeypatch, capsys):
    import importlib.metadata
    import shutil

    def _boom(*args, **kwargs):
        raise RuntimeError("probe exploded")

    _fake_spaces(monkeypatch, _boom)
    monkeypatch.setattr(importlib.metadata, "version", _boom)
    monkeypatch.setattr(shutil, "disk_usage", _boom)
    monkeypatch.setattr(os, "getuid", _boom)
    monkeypatch.setattr(os, "getgid", _boom)

    ms.log_diagnostics(roots, app_dir=tmp_path, on_spaces=True, audio_runner=_boom, mounts_path=str(tmp_path))

    assert "audio self-test: FAIL" in capsys.readouterr().err


# --- opt-in network tests -------------------------------------------------------
# CI runs `pytest -m "not gpu"`, which overrides addopts' "not network", so the
# marker alone does not keep these offline: they also need AMS_NETWORK_TESTS=1.

network = pytest.mark.skipif(
    os.getenv("AMS_NETWORK_TESTS") != "1", reason="reaches the Hugging Face Hub; set AMS_NETWORK_TESTS=1"
)

_NETWORK_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, sys.argv[1])
    import model_store as ms

    umbrella = ms.get_repo(ms.load_manifest(), repo_id="ACE-Step/Ace-Step1.5")
    entry = dict(
        umbrella,
        files=[
            {"path": "acestep-v15-turbo/silence_latent.pt", "size": 3841215},
            {"path": "vae/config.json", "size": 425},
        ],
    )
    print("RESULT", ms.ensure_repo(entry, [sys.argv[2]]))
    """
)


@pytest.mark.network
@network
def test_network_download_falls_back_to_plain_http_when_xet_cache_is_unwritable(tmp_path, chmod_tree):
    xet = tmp_path / "xet-readonly"
    xet.mkdir()
    chmod_tree.mode(xet, 0o500)
    root = tmp_path / "root"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "home"),
        "HF_HOME": str(tmp_path / "hf-home"),
        "HF_XET_CACHE": str(xet),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    proc = subprocess.run(
        [sys.executable, "-c", _NETWORK_SCRIPT, str(REPO_ROOT), str(root)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "os error 13" in proc.stderr, proc.stderr[-3000:]
    assert "with xet off" in proc.stderr, proc.stderr[-3000:]
    snap = root / "models--ACE-Step--Ace-Step1.5" / "snapshots" / "19671f406d603126926c1b7e2adc169acbcade22"
    assert os.stat(snap / "acestep-v15-turbo" / "silence_latent.pt").st_size == 3841215
    assert os.stat(snap / "vae" / "config.json").st_size == 425
    assert f"RESULT {snap}" in proc.stdout


def _hub_tree(repo_id: str, revision: str) -> dict[str, int]:
    """``{path: size}`` for every file of ``repo_id`` at ``revision``, from the Hub tree API."""
    import urllib.request

    url = f"https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true"
    sizes: dict[str, int] = {}
    while url:
        with urllib.request.urlopen(url, timeout=60) as resp:
            for item in json.loads(resp.read()):
                if item.get("type") == "file":
                    sizes[item["path"]] = item["size"]
            links = resp.headers.get("Link", "")
        nxt = [part for part in links.split(",") if 'rel="next"' in part]
        url = nxt[0].split(";")[0].strip().strip("<>") if nxt else ""
    return sizes


@pytest.mark.network
@network
def test_manifest_and_presets_match_the_hub_at_their_pinned_commits():
    import lora_stack

    checked = 0
    for repo in _raw_manifest()["repos"]:
        upstream = _hub_tree(repo["repo_id"], repo["revision"])
        for f in repo["files"]:
            assert f["path"] in upstream, f"{repo['repo_id']}@{repo['revision']}: no {f['path']}"
            assert upstream[f["path"]] == f["size"], (repo["repo_id"], f, upstream[f["path"]])
            checked += 1
    for preset in lora_stack.load_presets():
        assert re.fullmatch(r"[0-9a-f]{40}", preset["revision"]), preset
        upstream = _hub_tree(preset["hf_id"], preset["revision"])
        assert preset["filename"] in upstream, preset
        checked += 1
    assert checked == sum(len(r["files"]) for r in _raw_manifest()["repos"]) + len(lora_stack.load_presets())
