"""Pinned model snapshots for the Space: find them on disk, or download them.

Why this exists: on 2026-09-22 the Space died at import because the bootstrap
called ``snapshot_download`` for whole repos at upstream ``main``, assuming
``preload_from_hub`` had already filled the cache. That day one never-used
4.79 GB file was missing, hf_xet could not create its cache directory, and
the uncaught ``OSError`` killed the process. What the platform leaves in
``~/.cache/huggingface`` differs from one container start to the next, so
startup must not depend on it.

What it does instead:

* ``model_manifest.json`` pins every repo to a commit and lists exactly the
  files the app reads, with their sizes. A snapshot is *complete* when every
  ``snapshots/<pinned commit>/<path>`` exists, ``os.stat`` (which follows the
  cache's symlinks) reports the listed size, and the file is readable.
  Checking this is offline and read-only.
* Lookup searches the hub cache, the other places a platform preload may sit
  (``~/.cache/huggingface/hub``, ``/home/user/.cache/huggingface/hub``) and
  ``<app>/.ams-cache/hub``. The first complete snapshot wins, with no network
  call and no write.
* Otherwise the listed files are downloaded at the pinned commit through an
  ordered list of strategies: (hub cache, xet on), (hub cache, xet off),
  (app root, xet on), (app root, xet off). The exception type cannot tell a
  broken root from a network blip (hub 0.36.2 makes every request error an
  OSError; hf_xet also raises RuntimeError and ValueError), so any failure
  just moves to the next strategy; only permission/read-only/out-of-space
  errors retire a root, and only "repo, commit or access not found" stops at
  once. A full pass without success is followed by a capped backoff and
  another pass, until one overall startup budget is spent.
* ``startup()`` is what ``app.py`` runs at import on Spaces: diagnostics, the
  ``required`` repos (fatal on failure), the checkpoint links the fork's
  resolver needs, then the ``optional`` repos (never fatal).

Every function takes the cache roots as parameters so tests can use
temporary directories. Hub internals are reached through getattr so the
module also imports with huggingface_hub 1.x/2.x (CI).
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = APP_DIR / "model_manifest.json"
# The app-owned download root: the only place where leftovers are cleaned up.
APP_ROOT = APP_DIR / ".ams-cache" / "hub"
# Where a platform preload may sit when HF_HUB_CACHE points elsewhere
# (preload_from_hub does not follow HF_HOME). Searched, never written.
_EXTRA_LOOKUP = ("~/.cache/huggingface/hub", "/home/user/.cache/huggingface/hub")

REQUIRED = "required"
OPTIONAL = "optional"

# Manifest keys of the two repos the fork's checkpoint resolver needs.
UMBRELLA_KEY = "ace-step"
XL_SFT_KEY = "ace-step-xl-sft"
# Entries of vendor/ace-step/checkpoints/. The fork's check_main_model_exists
# requires all four umbrella components to exist; only acestep-v15-turbo's
# config.json is in the manifest, because its weights are never read when the
# DiT is XL SFT.
UMBRELLA_LINKS = ("vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B", "acestep-v15-turbo")
XL_SFT_LINK = "acestep-v15-xl-sft"

# One budget for the whole bootstrap, well inside the README's
# startup_duration_timeout (1 h), so the summary is printed before the
# platform kills the container.
BUDGET_S = 25 * 60
BACKOFF_BASE_S = 5.0
BACKOFF_CAP_S = 60.0
FREE_SPACE_MARGIN_BYTES = 2 * 10**9

# OSError numbers that mean "this cache root cannot be written".
_ROOT_ERRNOS = {
    code
    for code in (
        errno.EACCES,
        errno.EPERM,
        errno.EROFS,
        errno.ENOSPC,
        getattr(errno, "EDQUOT", None),
    )
    if code is not None
}
# Errors that no retry can fix: stop at once.
_FAIL_FAST_ERRORS = ("RepositoryNotFoundError", "RevisionNotFoundError", "GatedRepoError")
_FAIL_FAST_STATUS = {401, 403}

_DIAG_PACKAGES = (
    "torch",
    "torchaudio",
    "torchcodec",
    "transformers",
    "huggingface_hub",
    "hf_xet",
    "gradio",
    "spaces",
)
_DIAG_ENV = ("HOME", "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TORCH_HOME", "HF_MODULES_CACHE")

Downloader = Callable[..., Any]


class ModelStoreError(RuntimeError):
    """A repo could not be found complete or downloaded into any cache root."""


def _log(message: str) -> None:
    for line in message.splitlines() or [""]:
        print(f"[ams] {line}", file=sys.stderr, flush=True)


def _gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.2f}"


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _describe(exc: BaseException, limit: int = 300) -> str:
    return _shorten(f"{type(exc).__name__}: {exc}", limit)


def _label(entry: dict) -> str:
    return f"{entry['repo_id']}@{entry['revision'][:12]}"


def _unique(paths: Iterable[Path | str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(os.path.abspath(os.path.expanduser(str(p))))
        if path not in out:
            out.append(path)
    return out


# --- manifest -------------------------------------------------------------------


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict:
    """Read ``model_manifest.json``."""
    return json.loads(Path(path).read_text())


def get_repo(manifest: dict, *, key: str | None = None, repo_id: str | None = None) -> dict:
    """Return the manifest entry with this key or repo id; ``KeyError`` if absent."""
    for entry in manifest["repos"]:
        if (key is not None and entry["key"] == key) or (repo_id is not None and entry["repo_id"] == repo_id):
            return entry
    raise KeyError(key if key is not None else repo_id)


# --- roots and completeness -----------------------------------------------------


def _hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def download_roots() -> list[Path]:
    """Roots that may be written, in order: the hub cache, then ``<app>/.ams-cache/hub``."""
    return _unique([_hub_cache(), APP_ROOT])


def lookup_roots() -> list[Path]:
    """Roots searched for a complete snapshot: the download roots plus preload locations."""
    return _unique([_hub_cache(), *_EXTRA_LOOKUP, APP_ROOT])


def snapshot_path(root: Path | str, entry: dict) -> Path:
    """``<root>/models--<org>--<name>/snapshots/<pinned revision>``."""
    return _repo_folder(root, entry) / "snapshots" / entry["revision"]


def _repo_folder(root: Path | str, entry: dict) -> Path:
    return Path(root) / ("models--" + entry["repo_id"].replace("/", "--"))


def missing_files(root: Path | str, entry: dict) -> list[dict]:
    """Manifest files absent from this root's pinned snapshot, of the wrong size, or unreadable."""
    snap = snapshot_path(root, entry)
    missing = []
    for f in entry["files"]:
        path = snap / f["path"]
        try:
            st = os.stat(path)  # follows the cache's symlinks into blobs/
            ok = stat.S_ISREG(st.st_mode) and st.st_size == f["size"] and os.access(path, os.R_OK)
        except OSError:  # missing, dangling, or unreadable directory
            ok = False
        if not ok:
            missing.append(f)
    return missing


def is_complete(root: Path | str, entry: dict) -> bool:
    return not missing_files(root, entry)


def _complete_root(entry: dict, roots: Iterable[Path | str]) -> Path | None:
    for root in roots:
        if is_complete(root, entry):
            return Path(root)
    return None


def find_complete(entry: dict, roots: Iterable[Path | str]) -> Path | None:
    """First root's complete pinned snapshot, or ``None``. Offline and read-only."""
    root = _complete_root(entry, roots)
    return None if root is None else snapshot_path(root, entry)


def local_snapshot(
    repo_id: str, *, manifest: dict | None = None, roots: Iterable[Path | str] | None = None
) -> Path | None:
    """Complete local snapshot of a manifest repo, or ``None``. Never touches the network.

    For code that runs inside a GPU worker, where a Hub call would spend GPU
    time and a download would not fit in the call anyway.
    """
    manifest = load_manifest() if manifest is None else manifest
    roots = lookup_roots() if roots is None else roots
    return find_complete(get_repo(manifest, repo_id=repo_id), roots)


# --- ensure (download when needed) ----------------------------------------------


def _disk_free(path: Path | str) -> int | None:
    """Free bytes on the filesystem that holds ``path``, measured on its nearest existing ancestor.

    ``shutil.disk_usage`` raises on a path that does not exist yet, which is
    every fallback root on a fresh container.
    """
    p = Path(path)
    while not os.path.exists(p) and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def _probe_writable(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".ams-write-probe-", dir=root)
    os.close(fd)
    os.unlink(name)


def _root_unusable(exc: BaseException) -> bool:
    return isinstance(exc, PermissionError) or (isinstance(exc, OSError) and exc.errno in _ROOT_ERRNOS)


def _fail_fast(exc: BaseException) -> bool:
    try:
        from huggingface_hub import errors as hub_errors
    except Exception:
        hub_errors = None
    classes = tuple(
        c for c in (getattr(hub_errors, n, None) for n in _FAIL_FAST_ERRORS) if isinstance(c, type)
    )
    if classes and isinstance(exc, classes):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in _FAIL_FAST_STATUS


@contextlib.contextmanager
def _transfer(xet_off: bool) -> Iterator[None]:
    """For one attempt with xet off: plain HTTP, no hf_xet and no hf_transfer.

    Both are module attributes read per call (the env vars are frozen at
    import); they are restored afterwards. Only the import-time bootstrap,
    which is single-threaded, goes through here.
    """
    if not xet_off:
        yield
        return
    try:
        from huggingface_hub import constants
    except Exception:
        constants = None
    saved = {}
    for name, value in (("HF_HUB_DISABLE_XET", True), ("HF_HUB_ENABLE_HF_TRANSFER", False)):
        if constants is not None and hasattr(constants, name):
            saved[name] = getattr(constants, name)
            setattr(constants, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(constants, name, value)


def _remove_incomplete_blobs(root: Path, entry: dict) -> None:
    for leftover in (_repo_folder(root, entry) / "blobs").glob("*.incomplete"):
        try:
            leftover.unlink()
            _log(f"removed leftover {leftover}")
        except OSError:
            pass


def _summary(entry: dict, reason: str, roots: list[Path], notes: dict[Path, str]) -> ModelStoreError:
    total = sum(f["size"] for f in entry["files"])
    lines = [
        f"could not get {entry['repo_id']}@{entry['revision']} "
        f"({len(entry['files'])} files, {_gb(total)} GB): {_shorten(reason, 300)}"
    ]
    # Short enough that the whole summary survives in the 2000-character log
    # tail the Space API keeps.
    lines += [f"  {root}: {_shorten(notes.get(root, 'not tried'), 400)}" for root in roots]
    return ModelStoreError("\n".join(lines))


def ensure_repo(
    entry: dict,
    roots: Iterable[Path | str],
    *,
    lookup: Iterable[Path | str] | None = None,
    owned: Iterable[Path | str] = (),
    downloader: Downloader | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
    disk_free: Callable[[Path], int | None] | None = None,
    margin_bytes: int = FREE_SPACE_MARGIN_BYTES,
    backoff_cap_s: float = BACKOFF_CAP_S,
) -> Path:
    """Return a complete pinned snapshot of ``entry``, downloading it if no root has one.

    ``roots`` are the download roots in order; ``lookup`` (default: ``roots``)
    is searched first for a complete snapshot and never written unless it is
    also a download root; leftovers of failed attempts are removed only in
    ``owned`` roots. ``deadline`` is a ``clock()`` value (default: now plus
    ``BUDGET_S``). Raises ``ModelStoreError`` with a short per-root summary.
    """
    roots = _unique(roots)
    lookup = roots if lookup is None else _unique(lookup)
    owned = set(_unique(owned))
    label = _label(entry)

    found = _complete_root(entry, lookup)
    if found is not None:
        _log(f"{label}: found complete in {found}")
        return snapshot_path(found, entry)

    if downloader is None:
        from huggingface_hub import snapshot_download as downloader
    if disk_free is None:
        disk_free = _disk_free
    start = clock()
    deadline = start + BUDGET_S if deadline is None else deadline
    notes: dict[Path, str] = {}
    unusable: set[Path] = set()
    backoff = BACKOFF_BASE_S
    _log(f"{label}: not complete in any cache root, downloading")

    while True:
        for root in roots:
            if root in unusable:
                continue
            missing = missing_files(root, entry)
            need = sum(f["size"] for f in missing)
            free = disk_free(root)
            if free is not None and free < need + margin_bytes:
                notes[root] = (
                    f"not enough disk: {_gb(need)} GB missing + {_gb(margin_bytes)} GB margin, "
                    f"{_gb(free)} GB free"
                )
                unusable.add(root)
                _log(f"{label}: skipping {root}: {notes[root]}")
                continue
            try:
                _probe_writable(root)
            except OSError as exc:
                notes[root] = f"not writable: {_describe(exc)}"
                unusable.add(root)
                _log(f"{label}: skipping {root}: {notes[root]}")
                continue
            for xet_off in (False, True):
                if clock() >= deadline:
                    raise _summary(entry, "startup budget spent", roots, notes)
                mode = "xet off" if xet_off else "xet on"
                try:
                    with _transfer(xet_off):
                        downloader(
                            entry["repo_id"],
                            revision=entry["revision"],
                            allow_patterns=[f["path"] for f in entry["files"]],
                            cache_dir=str(root),
                            max_workers=1,
                        )
                except Exception as exc:
                    notes[root] = f"{mode}: {_describe(exc)}"
                    _log(f"{label}: {root} with {mode} failed: {_describe(exc)}")
                    if root in owned:
                        _remove_incomplete_blobs(root, entry)
                    if _fail_fast(exc):
                        raise _summary(entry, f"not retryable: {_describe(exc)}", roots, notes) from None
                    if _root_unusable(exc):
                        unusable.add(root)
                        break
                    continue
                # Never trust the return value: hub 0.36.2 returns the snapshot
                # folder without files when repo_info fails, and skips paths
                # that match nothing upstream.
                still_missing = missing_files(root, entry)
                if not still_missing:
                    _log(
                        f"{label}: downloaded {len(missing)} files, {_gb(need)} GB "
                        f"in {clock() - start:.1f} s to {root} with {mode}"
                    )
                    return snapshot_path(root, entry)
                notes[root] = (
                    f"{mode}: download returned but {len(still_missing)} file(s) are missing or "
                    f"of the wrong size, e.g. {still_missing[0]['path']}"
                )
                _log(f"{label}: {root} with {notes[root]}")
                if root in owned:
                    _remove_incomplete_blobs(root, entry)
        if all(root in unusable for root in roots):
            raise _summary(entry, "no cache root is usable", roots, notes)
        if clock() + backoff >= deadline:
            raise _summary(entry, "startup budget spent", roots, notes)
        _log(f"{label}: every strategy failed, retrying in {backoff:.0f} s")
        sleep(backoff)
        backoff = min(backoff_cap_s, backoff * 2)


# --- checkpoint links -----------------------------------------------------------


def link_checkpoints(
    checkpoints_dir: Path | str, umbrella_snapshot: Path | str, xl_snapshot: Path | str
) -> None:
    """Point ``vendor/ace-step/checkpoints/<component>`` at the resolved snapshots.

    A correct link is left as is; a dangling link or one that points elsewhere
    is replaced atomically (never followed); a real file or directory in the
    way is never deleted (it is left alone with a warning).
    """
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    wanted = {name: Path(umbrella_snapshot) / name for name in UMBRELLA_LINKS}
    wanted[XL_SFT_LINK] = Path(xl_snapshot)
    for name, target in wanted.items():
        link = checkpoints_dir / name
        if os.path.islink(link):
            try:
                if os.path.samefile(link, target):
                    continue
            except OSError:  # dangling link or missing target
                pass
            tmp = checkpoints_dir / f".{name}.ams-tmp-link"
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            os.symlink(target, tmp)
            os.replace(tmp, link)  # replaces the link itself, not what it points to
            _log(f"replaced link {link} -> {target}")
        elif link.exists():
            _log(f"WARNING: {link} is a real file or directory, not a link; left alone (wanted -> {target})")
        else:
            link.symlink_to(target)


# --- diagnostics ----------------------------------------------------------------


def _safe(fn: Callable[[], Any], default: str = "?") -> Any:
    try:
        return fn()
    except Exception as exc:
        return f"{default} ({type(exc).__name__})"


def _package_version(name: str) -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _read_mounts(mounts_path: str) -> list[tuple[str, str]] | None:
    """``(mount point, fs type)`` pairs from /proc/mounts, or ``None`` when unreadable."""
    try:
        with open(mounts_path) as f:
            text = f.read()
    except OSError:
        return None
    pairs = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            pairs.append((parts[1].replace("\\040", " "), parts[2]))
    return pairs


def _fs_type(path: Path | str, mounts: list[tuple[str, str]] | None) -> str:
    if mounts is None:
        return "n/a"
    p = Path(path)
    while not os.path.exists(p) and p != p.parent:
        p = p.parent
    real = os.path.realpath(p)
    best_mount, best_type = "", "?"
    for mount, fstype in mounts:
        inside = real == mount or real.startswith(mount.rstrip("/") + "/")
        if inside and len(mount) > len(best_mount):
            best_mount, best_type = mount, fstype
    return f"{best_type}({best_mount})" if best_mount else "?"


def _path_line(label: str, path: Path | str, mounts: list[tuple[str, str]] | None = None) -> str:
    path = str(path)
    try:
        st = os.stat(path)
        exists, owner, mode = "yes", str(st.st_uid), oct(stat.S_IMODE(st.st_mode))
    except FileNotFoundError:
        exists, owner, mode = "no", "-", "-"
    except Exception as exc:
        exists, owner, mode = f"? ({type(exc).__name__})", "?", "?"
    writable = _safe(lambda: "yes" if os.access(path, os.W_OK) else "no")
    free = _safe(lambda: _disk_free(path))
    free_text = f"{free / 1e9:.1f}GB" if isinstance(free, int) else str(free)
    fs = _safe(lambda: _fs_type(path, mounts))
    return (
        f"path {label} {path}: exists={exists} owner={owner} mode={mode} writable={writable} "
        f"free={free_text} fs={fs}"
    )


def _hub_constant(name: str) -> Any:
    from huggingface_hub import constants

    return getattr(constants, name, "<undefined>")


def _torch_cuda() -> str:
    """``torch.version.cuda`` only if torch is already imported (never import it here)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return "<torch not imported>"
    return str(getattr(getattr(torch, "version", None), "cuda", None))


def _zerogpu_duration_factor() -> Any:
    """spaces' factor on requested GPU seconds (1.5 on a large Blackwell slice)."""
    import importlib

    return importlib.import_module("spaces.zero.config").get_config()["duration_factor"]


def _diagnostic_lines(roots: list[Path], app_dir: Path, on_spaces: bool, mounts_path: str) -> list[str]:
    lines = [f"python {sys.version.split()[0]} ({sys.executable})"]
    versions = " ".join(f"{p}={_safe(lambda p=p: _package_version(p))}" for p in _DIAG_PACKAGES)
    lines.append(f"versions {versions}")
    lines.append(f"torch.version.cuda={_safe(_torch_cuda)}")
    if on_spaces:
        lines.append(f"zerogpu duration_factor={_safe(_zerogpu_duration_factor)}")
    lines.append(f"ids uid={_safe(os.getuid)} gid={_safe(os.getgid)}")
    lines.append("env " + " ".join(f"{v}={os.environ.get(v, '<unset>')}" for v in _DIAG_ENV))
    mounts = _safe(lambda: _read_mounts(mounts_path), None)
    mounts = mounts if isinstance(mounts, list) else None
    home = _safe(lambda: os.path.expanduser("~"))
    probes: list[tuple[str, Any]] = [
        ("~/.cache", os.path.join(home, ".cache")),
        ("~/.cache/huggingface", os.path.join(home, ".cache", "huggingface")),
        ("hub-cache", _safe(lambda: _hub_constant("HF_HUB_CACHE"))),
        ("xet-cache", _safe(lambda: _hub_constant("HF_XET_CACHE"))),
    ]
    probes += [(f"root{i}", root) for i, root in enumerate(roots, start=1)]
    probes += [("tempdir", _safe(tempfile.gettempdir)), ("app-dir", app_dir)]
    for label, path in probes:
        line = _safe(
            lambda label=label, path=path: _path_line(label, path, mounts), f"path {label} {path}: ?"
        )
        lines.append(line)
    handler = app_dir / "vendor" / "ace-step" / "acestep" / "handler.py"
    lines.append(f"vendor acestep/handler.py exists={_safe(lambda: 'yes' if handler.is_file() else 'no')}")
    return lines


# Runs in a subprocess: decode failures are silent in Cover/Extend/Edit (the
# fork swallows them), and only the Space has the Linux torchaudio/torchcodec
# binaries and system FFmpeg this exercises.
_AUDIO_SELFTEST = """
import os, shutil, tempfile, wave
import torch
import torchaudio
import torchcodec
print("AMS_AUDIO_VERSIONS torch=%s torchaudio=%s torchcodec=%s cuda=%s" % (
    torch.__version__, torchaudio.__version__, getattr(torchcodec, "__version__", "?"), torch.version.cuda))
d = tempfile.mkdtemp(prefix="ams-audio-selftest-")
try:
    path = os.path.join(d, "probe.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\\0\\0" * 1600)
    wav, sr = torchaudio.load(path)
    assert sr == 16000 and wav.shape[-1] == 1600, (tuple(wav.shape), sr)
    print("AMS_AUDIO_OK torch=%s torchaudio=%s torchcodec=%s shape=%s sr=%d" % (
        torch.__version__, torchaudio.__version__, getattr(torchcodec, "__version__", "?"), tuple(wav.shape), sr))
finally:
    shutil.rmtree(d, ignore_errors=True)
"""
AUDIO_SELFTEST_TIMEOUT_S = 60


def audio_selftest(
    *, runner: Callable[..., Any] | None = None, timeout_s: int = AUDIO_SELFTEST_TIMEOUT_S
) -> bool:
    """Import torch/torchaudio/torchcodec and decode a 0.1 s WAV in a subprocess. Never raises."""
    try:
        runner = subprocess.run if runner is None else runner
        start = time.monotonic()
        try:
            proc = runner(
                [sys.executable, "-c", _AUDIO_SELFTEST],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tempfile.gettempdir(),  # keep the app dir's modules off sys.path
            )
        except subprocess.TimeoutExpired:
            _log(f"audio self-test: FAIL (timed out after {timeout_s} s)")
            return False
        except Exception as exc:
            _log(f"audio self-test: FAIL (could not run: {_describe(exc)})")
            return False
        out = (proc.stdout or "").splitlines()
        ok = [line for line in out if line.startswith("AMS_AUDIO_OK ")]
        if proc.returncode == 0 and ok:
            _log(
                f"audio self-test: pass in {time.monotonic() - start:.1f} s ({ok[-1][len('AMS_AUDIO_OK ') :]})"
            )
            return True
        versions = [
            line[len("AMS_AUDIO_VERSIONS ") :] for line in out if line.startswith("AMS_AUDIO_VERSIONS ")
        ]
        err = (proc.stderr or "").strip().splitlines()
        reason = err[-1] if err else "no AMS_AUDIO_OK line"
        _log(
            f"audio self-test: FAIL (exit {proc.returncode}; {versions[-1] if versions else 'imports failed'}): "
            f"{_shorten(reason, 300)}"
        )
        return False
    except Exception as exc:
        with contextlib.suppress(Exception):
            _log(f"audio self-test: FAIL ({_describe(exc)})")
        return False


def log_diagnostics(
    roots: Iterable[Path | str],
    *,
    app_dir: Path | str = APP_DIR,
    on_spaces: bool | None = None,
    audio_runner: Callable[..., Any] | None = None,
    mounts_path: str = "/proc/mounts",
) -> None:
    """One block of ``[ams]`` lines on stderr describing the runtime. Never raises.

    On Spaces (``SPACE_ID`` set, unless ``on_spaces`` says otherwise) it also
    logs the ZeroGPU duration factor and runs the audio self-test.
    """
    try:
        if on_spaces is None:
            on_spaces = bool(os.getenv("SPACE_ID"))
        lines = _diagnostic_lines([Path(r) for r in roots], Path(app_dir), on_spaces, mounts_path)
    except Exception as exc:
        lines = [f"diagnostics failed: {_describe(exc)}"]
    for line in lines:
        with contextlib.suppress(Exception):
            _log(line)
    if on_spaces:
        audio_selftest(runner=audio_runner)


# --- startup entry point ----------------------------------------------------------


@contextlib.contextmanager
def _progress_bars_off() -> Iterator[None]:
    """Hub progress bars would fill the 2000-character log tail the Space API keeps."""
    try:
        from huggingface_hub import utils as hub_utils

        were_disabled = hub_utils.are_progress_bars_disabled()
        hub_utils.disable_progress_bars()
    except Exception:
        yield
        return
    try:
        yield
    finally:
        if not were_disabled:
            with contextlib.suppress(Exception):
                hub_utils.enable_progress_bars()


def startup(
    checkpoints_dir: Path | str,
    *,
    manifest: dict | None = None,
    roots: Iterable[Path | str] | None = None,
    lookup: Iterable[Path | str] | None = None,
    owned: Iterable[Path | str] | None = None,
    app_dir: Path | str = APP_DIR,
    budget_s: float = BUDGET_S,
    clock: Callable[[], float] = time.monotonic,
    **ensure_kw: Any,
) -> dict[str, Path]:
    """Diagnostics, required repos (raise on failure), checkpoint links, optional repos.

    One budget covers the whole call: a required repo still missing when it
    is spent raises the summary; an optional one is skipped with a log line.
    Defaults: download roots ``download_roots()``, lookup ``lookup_roots()``,
    leftovers cleaned in ``APP_ROOT``. With explicit ``roots`` (tests), lookup
    defaults to those roots and nothing is cleaned unless ``owned`` says so.
    Returns ``{manifest key: snapshot dir}`` for every repo that is available.
    """
    start = clock()
    deadline = start + budget_s
    manifest = load_manifest() if manifest is None else manifest
    if roots is None:
        roots = download_roots()
        lookup = lookup_roots() if lookup is None else lookup
        owned = [APP_ROOT] if owned is None else owned
    roots = _unique(roots)
    lookup = roots if lookup is None else _unique(lookup)
    owned = [] if owned is None else _unique(owned)
    log_diagnostics(lookup, app_dir=app_dir)

    kw = dict(ensure_kw, lookup=lookup, owned=owned, clock=clock, deadline=deadline)
    resolved: dict[str, Path] = {}
    with _progress_bars_off():
        for entry in manifest["repos"]:
            if entry["startup"] == REQUIRED:
                resolved[entry["key"]] = ensure_repo(entry, roots, **kw)

        link_checkpoints(checkpoints_dir, resolved[UMBRELLA_KEY], resolved[XL_SFT_KEY])

        for entry in manifest["repos"]:
            if entry["startup"] != OPTIONAL:
                continue
            try:
                resolved[entry["key"]] = ensure_repo(entry, roots, **kw)
            except Exception as exc:
                _log(f"WARNING: optional {_label(entry)} skipped, continuing without it:\n{exc}")
    _log(f"model bootstrap done in {clock() - start:.1f} s")
    return resolved
