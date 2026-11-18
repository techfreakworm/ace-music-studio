"""Tests for the pinned model manifest and the model store resolver.

Everything here is offline: no weights, no network, and every cache root and
checkpoints directory lives under ``tmp_path``. The one test that downloads
(two small files) is marked ``network`` and is deselected by default.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import requests
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
        assert r["startup"] in {"required", "optional", "lazy"}, r
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
        "ACE-Step/ACE-Step-v1-chinese-rap-LoRA": ("e40a223cb69a60ba8a8c46752987c152b8a5f828", "lazy"),
        "ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA": ("cb829a12775740c830a6d49795f16913065dc492", "lazy"),
    }
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


class FakeDownloader:
    """Stands in for ``huggingface_hub.snapshot_download`` and records every call.

    ``script`` maps repo_id to a list of actions consumed one per call: an
    exception instance is raised, ``"ok"`` writes the complete snapshot into
    ``cache_dir``, ``"partial"`` writes all but one file. Once a repo's list is
    exhausted the action is ``"ok"``.
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
                "xet_disabled": hf_constants.HF_HUB_DISABLE_XET,
                "kwargs": kwargs,
            }
        )
        actions = self.script.get(repo_id, [])
        action = actions.pop(0) if actions else "ok"
        if isinstance(action, BaseException):
            raise action
        entry = self.entries[repo_id]
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
def _xet_enabled(monkeypatch):
    """Start every test with xet allowed, as on the Space."""
    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_XET", False)


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


@pytest.fixture
def roots(tmp_path):
    return [tmp_path / "hub", tmp_path / "app" / ".ams-cache" / "hub", tmp_path / "tmp" / "ams-cache" / "hub"]


# --- R2.1: roots ---------------------------------------------------------------


def test_default_roots_order(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hf" / "hub"))
    roots = ms.default_roots()
    assert roots[0] == tmp_path / "hf" / "hub"
    assert roots[1] == Path(ms.__file__).resolve().parent / ".ams-cache" / "hub"
    assert roots[2].parts[-2:] == ("ams-cache", "hub")
    assert len(roots) == 3


def test_load_manifest_reads_repo_manifest():
    m = ms.load_manifest()
    assert m == _raw_manifest()
    assert ms.get_repo(m, repo_id=QWEN)["key"] == ms.get_repo(m, key="qwen2.5-7b-instruct")["key"]
    assert ms.get_repo(m, key=ms.UMBRELLA_KEY)["repo_id"] == UMBRELLA
    assert ms.get_repo(m, key=ms.XL_SFT_KEY)["repo_id"] == XL_SFT
    with pytest.raises(KeyError):
        ms.get_repo(m, repo_id="nobody/nothing")


# --- R2.2 / R9.2: complete snapshot check ---------------------------------------


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
    if os.access(roots[0], os.R_OK):  # running as root: permissions are not enforced
        pytest.skip("permissions are not enforced for this user")
    assert ms.find_complete(entry, roots) is None


# --- R2.4: local lookup ---------------------------------------------------------


def test_local_snapshot_finds_complete_snapshot_offline(roots, monkeypatch):
    entry = make_entry(FILES, repo_id="Qwen/Tiny")
    manifest = {"schema": 1, "repos": [entry]}
    assert ms.local_snapshot("Qwen/Tiny", manifest=manifest, roots=roots) is None
    snap = write_snapshot(roots[2], entry)
    assert ms.local_snapshot("Qwen/Tiny", manifest=manifest, roots=roots) == snap

    # Defaults come from the repo manifest and default_roots().
    monkeypatch.setattr(ms, "load_manifest", lambda: manifest)
    monkeypatch.setattr(ms, "default_roots", lambda: roots)
    assert ms.local_snapshot("Qwen/Tiny") == snap


# --- R2.3 / R9.3: ensure --------------------------------------------------------


def test_ensure_returns_readonly_complete_snapshot_without_download_or_write(roots, chmod_tree, ensure_kw):
    entry = make_entry(FILES)
    snap = write_snapshot(roots[0], entry)
    chmod_tree.readonly(roots[0])
    before = tree_state(roots[0].parent)
    dl = FakeDownloader([entry])

    assert ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw) == snap

    assert dl.calls == []
    assert tree_state(roots[0].parent) == before
    assert not roots[1].exists() and not roots[2].exists()


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
    assert call["xet_disabled"] is False
    err = capsys.readouterr().err
    assert f"downloaded 3 files, 0.00 GB in 0.0 s to {roots[0]}" in err


def test_ensure_reports_found_complete(roots, ensure_kw, capsys):
    entry = make_entry(FILES)
    write_snapshot(roots[1], entry)
    ms.ensure_repo(entry, roots, downloader=FakeDownloader([entry]), **ensure_kw)
    assert f"found complete in {roots[1]}" in capsys.readouterr().err


def test_ensure_moves_to_next_root_on_permission_error(roots, ensure_kw):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [PermissionError(13, "Permission denied", str(roots[0]))]})

    snap = ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert snap == ms.snapshot_path(roots[1], entry)
    assert [c["cache_dir"] for c in dl.calls] == [roots[0], roots[1]]


@pytest.mark.parametrize("code", ["EACCES", "EPERM", "EROFS", "ENOSPC", "EDQUOT"])
def test_ensure_moves_to_next_root_on_unusable_root_errno(roots, ensure_kw, code):
    import errno

    entry = make_entry(FILES)
    err = OSError(getattr(errno, code), os.strerror(getattr(errno, code)))
    dl = FakeDownloader([entry], {"org/model": [err]})
    assert ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw) == ms.snapshot_path(roots[1], entry)
    assert [c["cache_dir"] for c in dl.calls] == [roots[0], roots[1]]


def test_ensure_skips_root_that_fails_the_write_probe(roots, chmod_tree, ensure_kw):
    entry = make_entry(FILES)
    roots[0].mkdir(parents=True)
    chmod_tree.mode(roots[0], 0o555)
    if os.access(roots[0], os.W_OK):
        pytest.skip("permissions are not enforced for this user")
    dl = FakeDownloader([entry])
    assert ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw) == ms.snapshot_path(roots[1], entry)
    assert [c["cache_dir"] for c in dl.calls] == [roots[1]]


def test_ensure_retries_hf_xet_io_error_with_xet_disabled(roots, ensure_kw, fake_time, capsys):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [OSError(XET_IO_ERROR)]})

    snap = ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    assert snap == ms.snapshot_path(roots[0], entry)
    assert [(c["cache_dir"], c["xet_disabled"]) for c in dl.calls] == [(roots[0], False), (roots[0], True)]
    assert hf_constants.HF_HUB_DISABLE_XET is False  # restored
    assert fake_time.sleeps == []
    assert "xet disabled" in capsys.readouterr().err


def test_ensure_restores_xet_setting_when_the_xet_off_retry_fails(roots, ensure_kw, monkeypatch):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": [OSError(XET_IO_ERROR), OSError(XET_IO_ERROR)]})

    snap = ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw)

    # xet on, xet off in root 1; then root 2 starts again from the platform default.
    assert [(c["cache_dir"], c["xet_disabled"]) for c in dl.calls] == [
        (roots[0], False),
        (roots[0], True),
        (roots[1], False),
    ]
    assert snap == ms.snapshot_path(roots[1], entry)
    assert hf_constants.HF_HUB_DISABLE_XET is False


def test_ensure_retries_transient_errors_with_backoff_last_attempt_without_xet(roots, ensure_kw, fake_time):
    entry = make_entry(FILES)
    boom = requests.exceptions.ConnectionError("Max retries exceeded")
    dl = FakeDownloader([entry], {"org/model": [boom, boom, boom]})

    snap = ms.ensure_repo(entry, roots, downloader=dl, max_attempts=4, **ensure_kw)

    assert snap == ms.snapshot_path(roots[0], entry)
    assert [c["cache_dir"] for c in dl.calls] == [roots[0]] * 4
    assert [c["xet_disabled"] for c in dl.calls] == [False, False, False, True]
    assert fake_time.sleeps == [5, 10, 20]
    assert hf_constants.HF_HUB_DISABLE_XET is False


def test_ensure_backoff_is_capped(roots, ensure_kw, fake_time):
    entry = make_entry(FILES)
    boom = TimeoutError("read timed out")
    dl = FakeDownloader([entry], {"org/model": [boom] * 6})
    ms.ensure_repo(entry, roots, downloader=dl, max_attempts=7, backoff_cap_s=30, **ensure_kw)
    assert fake_time.sleeps == [5, 10, 20, 30, 30, 30]


def test_ensure_retries_when_download_leaves_snapshot_incomplete(roots, ensure_kw, fake_time):
    entry = make_entry(FILES)
    dl = FakeDownloader([entry], {"org/model": ["partial"]})
    assert ms.ensure_repo(entry, roots, downloader=dl, **ensure_kw) == ms.snapshot_path(roots[0], entry)
    assert len(dl.calls) == 2
    assert fake_time.sleeps == [5]


def test_ensure_stops_at_the_deadline(roots, ensure_kw, fake_time):
    entry = make_entry(FILES)
    boom = requests.exceptions.ConnectionError("Max retries exceeded")
    dl = FakeDownloader([entry], {"org/model": [boom] * 50})

    with pytest.raises(ms.ModelStoreError) as exc:
        ms.ensure_repo(entry, roots, downloader=dl, deadline_s=30, max_attempts=10, **ensure_kw)

    # t=0 fail, sleep 5; t=5 fail, sleep 10; t=15 fail, next sleep (20) would pass the deadline.
    assert len(dl.calls) == 3
    assert fake_time.sleeps == [5, 10]
    assert "deadline" in str(exc.value)


def test_ensure_skips_root_without_enough_free_space(roots, fake_time):
    entry = make_entry(FILES)
    free = {roots[0]: 2 * 10**9, roots[1]: 10**15, roots[2]: 10**15}
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


def test_default_disk_free_uses_nearest_existing_ancestor(tmp_path):
    free = ms._disk_free(tmp_path / "does" / "not" / "exist")
    assert isinstance(free, int) and free > 0


def test_ensure_raises_one_summary_when_every_root_fails(roots, fake_time):
    entry = make_entry(FILES, repo_id="ACE-Step/Some-Repo")
    dl = FakeDownloader(
        [entry],
        {
            "ACE-Step/Some-Repo": [
                PermissionError(13, "Permission denied", str(roots[0] / ".locks")),
                OSError(XET_IO_ERROR),
                OSError(XET_IO_ERROR),
            ]
        },
    )
    free = {roots[0]: 10**15, roots[1]: 10**15, roots[2]: 10**6}

    with pytest.raises(ms.ModelStoreError) as exc:
        ms.ensure_repo(
            entry,
            roots,
            downloader=dl,
            sleep=fake_time.sleep,
            clock=fake_time.clock,
            disk_free=lambda path: free[Path(path)],
        )

    msg = str(exc.value)
    assert "ACE-Step/Some-Repo" in msg and REV in msg
    for root in roots:
        assert str(root) in msg
    assert "PermissionError" in msg
    assert "os error 13" in msg
    assert "not enough disk" in msg
    assert len(msg) < 2000
    assert len(dl.calls) == 3


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
    monkeypatch.setattr(hf_constants, "HF_XET_CACHE", str(xet))
    calls = []

    def _downloader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("downloader called")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _downloader)
    before = tree_state(roots[0])
    ckpt = tmp_path / "checkpoints"

    resolved = ms.startup(ckpt, manifest=manifest, roots=roots)

    assert calls == []
    assert tree_state(roots[0]) == before
    assert set(resolved) == {"ace-step", "ace-step-xl-sft", "qwen2.5-7b-instruct"}
    for name in ms.UMBRELLA_LINKS:
        assert (ckpt / name).is_symlink()
        assert os.path.samefile(ckpt / name, umbrella_snap / name)
    assert os.path.samefile(ckpt / ms.XL_SFT_LINK, resolved["ace-step-xl-sft"])
    assert not roots[1].exists() and not roots[2].exists()


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


# --- R2.6 / R2.7 / R9.7: diagnostics and the startup entry point -----------------


def _startup_manifest():
    return {
        "schema": 1,
        "repos": [
            make_entry(
                {f"{n}/f.bin": 5 for n in ms.UMBRELLA_LINKS}, repo_id="o/umbrella", key=ms.UMBRELLA_KEY
            ),
            make_entry({"model.bin": 9}, repo_id="o/xl", key=ms.XL_SFT_KEY),
            make_entry({"model.bin": 9}, repo_id="o/qwen", key="qwen", startup="optional"),
            make_entry({"lora.bin": 3}, repo_id="o/lora", key="lora", startup="lazy"),
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
    denied = [PermissionError(13, "Permission denied")] * 3
    dl = FakeDownloader(manifest["repos"], {"o/xl": denied})
    ckpt = tmp_path / "checkpoints"

    with pytest.raises(ms.ModelStoreError) as exc:
        ms.startup(ckpt, manifest=manifest, roots=roots, downloader=dl, **ensure_kw)

    assert "o/xl" in str(exc.value)
    assert not (ckpt / ms.XL_SFT_LINK).exists()
    assert "o/qwen" not in [c["repo_id"] for c in dl.calls]


def test_startup_continues_when_an_optional_repo_fails(tmp_path, roots, ensure_kw, capsys):
    manifest = _startup_manifest()
    boom = requests.exceptions.ConnectionError("Max retries exceeded")
    dl = FakeDownloader(manifest["repos"], {"o/qwen": [boom] * 20})
    ckpt = tmp_path / "checkpoints"

    resolved = ms.startup(ckpt, manifest=manifest, roots=roots, downloader=dl, max_attempts=2, **ensure_kw)

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


def test_diagnostics_never_raise_with_unreadable_path(tmp_path, monkeypatch, capsys, chmod_tree):
    home = tmp_path / "home"
    (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(home / ".cache" / "huggingface" / "hub"))
    monkeypatch.setattr(hf_constants, "HF_XET_CACHE", str(home / ".cache" / "huggingface" / "xet"))
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


# --- opt-in network test --------------------------------------------------------

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
    assert "xet disabled" in proc.stderr, proc.stderr[-3000:]
    snap = root / "models--ACE-Step--Ace-Step1.5" / "snapshots" / "19671f406d603126926c1b7e2adc169acbcade22"
    assert os.stat(snap / "acestep-v15-turbo" / "silence_latent.pt").st_size == 3841215
    assert os.stat(snap / "vae" / "config.json").st_size == 425
    assert f"RESULT {snap}" in proc.stdout
