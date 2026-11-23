"""Real-time filesystem and harness transcript watcher using watchfiles (FSEvents on macOS)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from chittam.adapters.files import _NOISE_DIR_NAMES, _is_noise_file
from chittam.config import is_path_excluded
from chittam.indexer import sync_changed_paths
from chittam.models import AppConfig, SyncStats


def _harness_dirs(cfg: AppConfig) -> list[Path]:
    hist_path = Path(cfg.harnesses.claude_history_path).expanduser()
    candidates = [
        Path(cfg.harnesses.antigravity_brain_dir).expanduser(),
        Path(cfg.harnesses.codex_home_dir).expanduser(),
        *(Path(p).expanduser() for p in cfg.harnesses.codex_legacy_memories_dirs),
        Path(cfg.harnesses.claude_projects_dir).expanduser(),
    ]
    if hist_path.exists():
        candidates.append(hist_path.parent)
    out: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        try:
            res = p.resolve(strict=False)
        except OSError:
            res = p
        if res not in seen and res.exists():
            seen.add(res)
            out.append(res)
    return out


def _dedupe_nested_roots(paths: list[Path]) -> list[Path]:
    ordered = sorted(set(paths), key=lambda p: len(p.parts))
    kept: list[Path] = []
    for cand in ordered:
        covered = False
        for parent in kept:
            try:
                cand.relative_to(parent)
                covered = True
                break
            except ValueError:
                continue
        if not covered:
            kept.append(cand)
    return kept


def build_watch_roots(cfg: AppConfig) -> list[Path]:
    """Return deduplicated existing scan subdirectories and harness dirs to monitor recursively."""
    roots: list[Path] = []
    try:
        home_resolved = Path.home().resolve(strict=False)
    except OSError:
        home_resolved = None

    for raw in cfg.privacy.scan_roots:
        p = Path(raw).expanduser()
        try:
            res = p.resolve(strict=False)
        except OSError:
            continue
        if not res.exists() or not res.is_dir():
            continue
        # Avoid registering a recursive FSEvents stream on all of $HOME when only top-level ~ files are enabled
        if cfg.privacy.home_top_level_files_only and (
            raw.strip() == "~" or (home_resolved is not None and res == home_resolved)
        ):
            continue
        excl, _ = is_path_excluded(res, cfg.privacy)
        if not excl:
            roots.append(res)

    for hdir in _harness_dirs(cfg):
        excl, _ = is_path_excluded(hdir, cfg.privacy)
        if not excl:
            roots.append(hdir)

    return _dedupe_nested_roots(roots)


def should_watch_path(
    path: Path | str,
    cfg: AppConfig,
    harness_roots: list[Path] | None = None,
) -> bool:
    """Filter out excluded, hidden, and build-artifact noise paths before syncing."""
    p = Path(path).expanduser()
    try:
        resolved = p.resolve(strict=False)
    except OSError:
        resolved = p

    excl, _ = is_path_excluded(resolved, cfg.privacy)
    if excl:
        return False

    h_roots = harness_roots if harness_roots is not None else _harness_dirs(cfg)
    in_harness = False
    for hr in h_roots:
        try:
            resolved.relative_to(hr)
            in_harness = True
            break
        except ValueError:
            continue

    if _is_noise_file(resolved):
        return False

    exclude_dirs = set(cfg.privacy.exclude_dir_names) | _NOISE_DIR_NAMES
    for part in resolved.parts[:-1]:
        if part in exclude_dirs or part.lower() in _NOISE_DIR_NAMES:
            return False
        if part.startswith(".") and not in_harness:
            return False

    return True


def run_watcher(
    cfg: AppConfig,
    *,
    debounce_ms: int = 1500,
    skip_embeddings: bool = False,
    skip_vision: bool = False,
    stop_event: threading.Event | None = None,
    on_batch: Callable[[SyncStats], None] | None = None,
) -> None:
    """Watch scan roots and harness directories and incrementally sync changed paths."""
    from watchfiles import watch

    roots = build_watch_roots(cfg)
    if not roots:
        return
    h_roots = _harness_dirs(cfg)

    def _watch_filter(_change: Any, raw_path: str) -> bool:
        return should_watch_path(Path(raw_path), cfg, harness_roots=h_roots)

    for changes in watch(
        *(str(r) for r in roots),
        debounce=max(200, int(debounce_ms)),
        step=200,
        stop_event=stop_event,
        watch_filter=_watch_filter,
        recursive=True,
        ignore_permission_denied=True,
    ):
        changed_paths = [Path(p) for _change, p in changes]
        if not changed_paths:
            continue
        batch_stats = sync_changed_paths(
            cfg,
            changed_paths,
            skip_embeddings=skip_embeddings,
            skip_vision=skip_vision,
        )
        if on_batch is not None:
            on_batch(batch_stats)
