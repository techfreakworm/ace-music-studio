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
  files the app reads, with their sizes. A snapshot is *complete* when
  ``snapshots/<pinned commit>/<path>`` exists for every listed file and
  ``os.stat`` (which follows the cache's symlinks) reports the listed size.
  Checking this is offline and read-only.
* Three cache roots are searched in order: the default hub cache (where a
  preload lands), ``<app>/.ams-cache/hub`` and ``<tempdir>/ams-cache/hub``.
  The first complete snapshot wins, with no network call and no write.
* Otherwise the listed files are downloaded at the pinned commit into the
  first root that works. Permission, read-only and out-of-space errors move
  on to the next root; an hf_xet I/O error is retried once with xet switched
  off (plain HTTP); anything else is retried with backoff, the last attempt
  without xet, within a per-repo deadline.
* ``startup()`` is what ``app.py`` runs at import on Spaces: diagnostics, the
  ``required`` repos (fatal on failure), the checkpoint links the fork's
  resolver needs, then the ``optional`` repos (never fatal). ``lazy`` repos
  are fetched on first use through ``ensure_repo``.

Every function takes the cache roots as a parameter so tests can use
temporary directories.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = APP_DIR / "model_manifest.json"

REQUIRED = "required"
OPTIONAL = "optional"
LAZY = "lazy"

# Manifest keys of the two repos the fork's checkpoint resolver needs.
UMBRELLA_KEY = "ace-step"
XL_SFT_KEY = "ace-step-xl-sft"
# Entries of vendor/ace-step/checkpoints/. The fork's check_main_model_exists
# requires all four umbrella components to exist; only acestep-v15-turbo's
# config.json is in the manifest, because its weights are never read when the
# DiT is XL SFT.
UMBRELLA_LINKS = ("vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B", "acestep-v15-turbo")
XL_SFT_LINK = "acestep-v15-xl-sft"

DEADLINE_S = 12 * 60
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 5.0
BACKOFF_CAP_S = 60.0
FREE_SPACE_MARGIN_BYTES = 2 * 10**9

# OSError numbers that mean "this cache root cannot be used", not "try again".
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


class IncompleteDownloadError(RuntimeError):
    """The downloader returned, but the snapshot is still not complete."""


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


def default_roots() -> list[Path]:
    """Cache roots in search order: hub cache, ``<app>/.ams-cache/hub``, ``<tmp>/ams-cache/hub``."""
    from huggingface_hub import constants

    roots: list[Path] = []
    for root in (
        Path(constants.HF_HUB_CACHE),
        APP_DIR / ".ams-cache" / "hub",
        Path(tempfile.gettempdir()) / "ams-cache" / "hub",
    ):
        if root not in roots:
            roots.append(root)
    return roots


def snapshot_path(root: Path | str, entry: dict) -> Path:
    """``<root>/models--<org>--<name>/snapshots/<pinned revision>``."""
    folder = "models--" + entry["repo_id"].replace("/", "--")
    return Path(root) / folder / "snapshots" / entry["revision"]


def missing_files(root: Path | str, entry: dict) -> list[dict]:
    """Manifest files absent from this root's pinned snapshot or of the wrong size."""
    snap = snapshot_path(root, entry)
    missing = []
    for f in entry["files"]:
        try:
            st = os.stat(snap / f["path"])  # follows the cache's symlinks into blobs/
        except OSError:  # missing, dangling, or unreadable
            missing.append(f)
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_size != f["size"]:
            missing.append(f)
    return missing


def is_complete(root: Path | str, entry: dict) -> bool:
    return not missing_files(root, entry)


def find_complete(entry: dict, roots: Iterable[Path | str]) -> Path | None:
    """First root's complete pinned snapshot, or ``None``. Offline and read-only."""
    for root in roots:
        if is_complete(root, entry):
            return snapshot_path(root, entry)
    return None


def local_snapshot(
    repo_id: str, *, manifest: dict | None = None, roots: Iterable[Path | str] | None = None
) -> Path | None:
    """Complete local snapshot of a manifest repo, or ``None``. Never touches the network.

    For code that runs inside a GPU worker, where a Hub call would spend GPU
    time (and may not be allowed at all).
    """
    manifest = load_manifest() if manifest is None else manifest
    roots = default_roots() if roots is None else roots
    return find_complete(get_repo(manifest, repo_id=repo_id), roots)


# --- ensure (download when needed) ----------------------------------------------


def _disk_free(path: Path | str) -> int | None:
    """Free bytes on the filesystem that holds ``path`` (or its nearest existing parent)."""
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


def _classify(exc: BaseException) -> str:
    """``"root"`` (root unusable), ``"xet"`` (hf_xet I/O error) or ``"transient"``."""
    if isinstance(exc, PermissionError):
        return "root"
    if isinstance(exc, OSError) and exc.errno in _ROOT_ERRNOS:
        return "root"
    if isinstance(exc, OSError) and exc.errno is None:
        # hf_xet maps Rust's std::io::Error to a bare OSError whose message
        # ends in "(os error N)", with no errno attribute.
        text = str(exc)
        if "(os error" in text or text.startswith("I/O error"):
            return "xet"
    return "transient"


@contextlib.contextmanager
def _xet_disabled(disabled: bool) -> Iterator[None]:
    """Switch hf_xet off for the duration (``is_xet_available`` reads the constant per call)."""
    if not disabled:
        yield
        return
    from huggingface_hub import constants

    previous = constants.HF_HUB_DISABLE_XET
    constants.HF_HUB_DISABLE_XET = True
    try:
        yield
    finally:
        constants.HF_HUB_DISABLE_XET = previous


def _attempt(entry: dict, root: Path, downloader: Downloader, xet_off: bool) -> Path:
    with _xet_disabled(xet_off):
        downloader(
            entry["repo_id"],
            revision=entry["revision"],
            allow_patterns=[f["path"] for f in entry["files"]],
            cache_dir=str(root),
        )
    still_missing = missing_files(root, entry)
    if still_missing:
        raise IncompleteDownloadError(
            f"{len(still_missing)} file(s) missing or of the wrong size after download, "
            f"e.g. {still_missing[0]['path']}"
        )
    return snapshot_path(root, entry)


def _download_into(
    entry: dict,
    root: Path,
    *,
    downloader: Downloader,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    deadline: float,
    max_attempts: int,
    backoff_cap_s: float,
) -> tuple[Path | None, str, bool]:
    """Try one root. Returns ``(snapshot, last_error, deadline_reached)``."""
    label = _label(entry)
    attempt = 0
    while True:
        attempt += 1
        xet_off = attempt >= max_attempts
        try:
            return _attempt(entry, root, downloader, xet_off), "", False
        except Exception as exc:
            kind = _classify(exc)
            last = _describe(exc)
            _log(f"{label}: attempt {attempt} in {root}{' (xet off)' if xet_off else ''} failed: {last}")
            if kind == "root":
                return None, last, False
            if kind == "xet" and not xet_off:
                _log(f"{label}: hf_xet error, retrying once in {root} with xet disabled")
                try:
                    return _attempt(entry, root, downloader, True), "", False
                except Exception as retry_exc:
                    last = f"{_describe(retry_exc)} (retried with xet disabled after: {last})"
                    _log(f"{label}: retry with xet disabled failed: {_describe(retry_exc)}")
                    return None, last, False
            if attempt >= max_attempts:
                return None, last, False
            delay = min(backoff_cap_s, BACKOFF_BASE_S * 2 ** (attempt - 1))
            if clock() + delay > deadline:
                return None, f"{last} (stopped: per-repo deadline reached)", True
            _log(f"{label}: retrying in {delay:.0f} s")
            sleep(delay)


def ensure_repo(
    entry: dict,
    roots: Iterable[Path | str],
    *,
    downloader: Downloader | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    disk_free: Callable[[Path], int | None] | None = None,
    deadline_s: float = DEADLINE_S,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_cap_s: float = BACKOFF_CAP_S,
    margin_bytes: int = FREE_SPACE_MARGIN_BYTES,
) -> Path:
    """Return a complete pinned snapshot of ``entry``, downloading it if no root has one.

    Raises ``ModelStoreError`` with a short per-root summary when every root failed.
    """
    roots = [Path(r) for r in roots]
    label = _label(entry)
    for root in roots:
        if is_complete(root, entry):
            _log(f"{label}: found complete in {root}")
            return snapshot_path(root, entry)

    if downloader is None:
        from huggingface_hub import snapshot_download as downloader
    if disk_free is None:
        disk_free = _disk_free

    start = clock()
    deadline = start + deadline_s
    failures: list[tuple[Path, str]] = []
    for root in roots:
        if clock() >= deadline:
            failures.append((root, "not tried: per-repo deadline reached"))
            break
        missing = missing_files(root, entry)
        need = sum(f["size"] for f in missing)
        free = disk_free(root)
        if free is not None and free < need + margin_bytes:
            reason = f"not enough disk: {_gb(need)} GB missing + {_gb(margin_bytes)} GB margin, {_gb(free)} GB free"
            _log(f"{label}: skipping {root}: {reason}")
            failures.append((root, reason))
            continue
        try:
            _probe_writable(root)
        except OSError as exc:
            reason = f"not writable: {_describe(exc)}"
            _log(f"{label}: skipping {root}: {reason}")
            failures.append((root, reason))
            continue
        _log(f"{label}: downloading {len(missing)} files ({_gb(need)} GB) to {root}")
        snap, last, out_of_time = _download_into(
            entry,
            root,
            downloader=downloader,
            sleep=sleep,
            clock=clock,
            deadline=deadline,
            max_attempts=max_attempts,
            backoff_cap_s=backoff_cap_s,
        )
        if snap is not None:
            _log(
                f"{label}: downloaded {len(missing)} files, {_gb(need)} GB "
                f"in {clock() - start:.1f} s to {root}"
            )
            return snap
        failures.append((root, last))
        if out_of_time:
            break

    total = sum(f["size"] for f in entry["files"])
    lines = [
        f"could not get {entry['repo_id']}@{entry['revision']} "
        f"({len(entry['files'])} files, {_gb(total)} GB) into any cache root:"
    ]
    # Short enough that the whole summary survives in the 2000-character log
    # tail the Space API keeps.
    lines += [f"  {root}: {_shorten(reason, 400)}" for root, reason in failures]
    if not failures:
        lines.append("  (no cache roots given)")
    raise ModelStoreError("\n".join(lines))


# --- checkpoint links -----------------------------------------------------------


def link_checkpoints(
    checkpoints_dir: Path | str, umbrella_snapshot: Path | str, xl_snapshot: Path | str
) -> None:
    """Point ``vendor/ace-step/checkpoints/<component>`` at the resolved snapshots.

    A correct link is left as is; a dangling link or one that points elsewhere
    is replaced; a real file or directory in the way is never deleted (it is
    left alone with a warning).
    """
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    wanted = {name: Path(umbrella_snapshot) / name for name in UMBRELLA_LINKS}
    wanted[XL_SFT_LINK] = Path(xl_snapshot)
    for name, target in wanted.items():
        link = checkpoints_dir / name
        if link.is_symlink():
            try:
                if os.path.samefile(link, target):
                    continue
            except OSError:  # dangling link or missing target
                pass
            link.unlink()
            link.symlink_to(target)
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


def _path_line(label: str, path: Path | str) -> str:
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
    return (
        f"path {label} {path}: exists={exists} owner={owner} mode={mode} writable={writable} free={free_text}"
    )


def _diagnostic_lines(roots: list[Path], app_dir: Path) -> list[str]:
    from huggingface_hub import constants

    lines = [f"python {sys.version.split()[0]} ({sys.executable})"]
    versions = " ".join(f"{p}={_safe(lambda p=p: _package_version(p))}" for p in _DIAG_PACKAGES)
    lines.append(f"versions {versions}")
    lines.append(f"ids uid={_safe(os.getuid)} gid={_safe(os.getgid)}")
    lines.append("env " + " ".join(f"{v}={os.environ.get(v, '<unset>')}" for v in _DIAG_ENV))
    home = _safe(lambda: os.path.expanduser("~"))
    probes: list[tuple[str, Any]] = [
        ("~/.cache", os.path.join(home, ".cache")),
        ("~/.cache/huggingface", os.path.join(home, ".cache", "huggingface")),
        ("hub-cache", _safe(lambda: constants.HF_HUB_CACHE)),
        ("xet-cache", _safe(lambda: constants.HF_XET_CACHE)),
    ]
    probes += [(f"root{i}", root) for i, root in enumerate(roots, start=1)]
    probes += [("tempdir", _safe(tempfile.gettempdir)), ("app-dir", app_dir)]
    for label, path in probes:
        lines.append(_safe(lambda label=label, path=path: _path_line(label, path), f"path {label} {path}: ?"))
    return lines


def log_diagnostics(roots: Iterable[Path | str], *, app_dir: Path | str = APP_DIR) -> None:
    """One block of ``[ams]`` lines on stderr describing the runtime. Never raises."""
    try:
        lines = _diagnostic_lines([Path(r) for r in roots], Path(app_dir))
    except Exception as exc:
        lines = [f"diagnostics failed: {_describe(exc)}"]
    for line in lines:
        with contextlib.suppress(Exception):
            _log(line)


# --- startup entry point ----------------------------------------------------------


def startup(
    checkpoints_dir: Path | str,
    *,
    manifest: dict | None = None,
    roots: Iterable[Path | str] | None = None,
    app_dir: Path | str = APP_DIR,
    **ensure_kw: Any,
) -> dict[str, Path]:
    """Diagnostics, required repos (raise on failure), checkpoint links, optional repos.

    Returns ``{manifest key: snapshot dir}`` for every repo that is available.
    """
    manifest = load_manifest() if manifest is None else manifest
    roots = default_roots() if roots is None else [Path(r) for r in roots]
    log_diagnostics(roots, app_dir=app_dir)

    resolved: dict[str, Path] = {}
    for entry in manifest["repos"]:
        if entry["startup"] == REQUIRED:
            resolved[entry["key"]] = ensure_repo(entry, roots, **ensure_kw)

    link_checkpoints(checkpoints_dir, resolved[UMBRELLA_KEY], resolved[XL_SFT_KEY])

    for entry in manifest["repos"]:
        if entry["startup"] != OPTIONAL:
            continue
        try:
            resolved[entry["key"]] = ensure_repo(entry, roots, **ensure_kw)
        except Exception as exc:
            _log(f"WARNING: optional {_label(entry)} unavailable, continuing without it:\n{exc}")
    return resolved
