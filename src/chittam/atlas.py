"""Canonical project atlas discovery, auto-classification, and YAML persistence."""

from __future__ import annotations

import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from chittam.config import is_path_excluded
from chittam.models import AppConfig, CanonicalRepo, RepoTier

_ARCHIVE_WORD_RE = re.compile(
    r"(?:^|[^a-z])(archive|archived|backup|backups|old|copy|legacy|snapshot|snapshots)(?:$|[^a-z])"
)
_SNAPSHOT_MONTH_RE = re.compile(r"(?:^|[^a-z])(nov|dec)(?:\d+|[^a-z]|$)")
_LEADING_NUM_PREFIX_RE = re.compile(r"^\d+[_\-\s]+")
_FAMILY_SUFFIX_RE = re.compile(
    r"(?:[_\-\s]+(?:oct|nov|dec)\d*(?:[_\-\s]+\d{4})?(?:\s*copy)?|(?:Oct|Nov|Dec)\d+|(?:[_\-\s]+(?:copy|old|backup|backups|archive|archived|legacy|snapshot|d\d+)))+$",
    re.IGNORECASE,
)

_TIER_ORDER = {
    RepoTier.ACTIVE: 0,
    RepoTier.REFERENCE: 1,
    RepoTier.ARCHIVE: 2,
}

def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "repo"


def infer_project_family(name: str) -> str:
    """Derive normalized project family by stripping index prefixes and snapshot suffixes."""
    cleaned = _LEADING_NUM_PREFIX_RE.sub("", name.strip())
    # Iteratively strip stacked suffixes like ` _OCT17_2025copy`
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _FAMILY_SUFFIX_RE.sub("", cleaned).strip()
    return _slugify(cleaned or name)


def _sanitize_remote_url(raw: str | None) -> str | None:
    if not raw:
        return None
    url = raw.strip()
    if not url:
        return None
    if "://" in url:
        try:
            parts = urlsplit(url)
            if parts.username or parts.password:
                netloc = parts.hostname or ""
                if parts.port:
                    netloc = f"{netloc}:{parts.port}"
                return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        except ValueError:
            pass
    return url


def _run_git(repo_dir: Path, args: list[str], timeout: float = 5.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def _parse_iso_datetime(ts: str | None) -> datetime | None:
    if not ts:
        return None
    raw = ts.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _is_snapshot_or_archive_path(repo_path: Path, scan_roots: list[Path] | None = None) -> bool:
    rel_parts: tuple[str, ...] | None = None
    if scan_roots:
        best_rel: tuple[str, ...] | None = None
        for root in scan_roots:
            try:
                rel = repo_path.relative_to(root)
                if best_rel is None or len(rel.parts) < len(best_rel):
                    best_rel = rel.parts
            except ValueError:
                continue
        rel_parts = best_rel
    if not rel_parts:
        rel_parts = repo_path.parts[-3:]

    for segment in rel_parts:
        lowered = segment.lower()
        if _ARCHIVE_WORD_RE.search(lowered) or _SNAPSHOT_MONTH_RE.search(lowered):
            return True
    return False


def classify_repo_tier(
    repo_path: Path,
    last_commit_at: str | None,
    scan_roots: list[Path] | None = None,
    now: datetime | None = None,
) -> RepoTier:
    """Classify a repo as active (<=45d & non-snapshot), reference (46-180d), or archive."""
    if _is_snapshot_or_archive_path(repo_path, scan_roots=scan_roots):
        return RepoTier.ARCHIVE

    dt = _parse_iso_datetime(last_commit_at)
    if dt is None:
        return RepoTier.ARCHIVE

    ref_now = now or datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (ref_now - dt).total_seconds() / 86400.0)

    if age_days <= 45.0:
        return RepoTier.ACTIVE
    if age_days <= 180.0:
        return RepoTier.REFERENCE
    return RepoTier.ARCHIVE


def _inspect_git_repo(repo_dir: Path) -> tuple[str | None, str | None, str | None]:
    rc, commit_ts = _run_git(repo_dir, ["log", "-1", "--format=%cI"])
    last_commit_at = commit_ts if rc == 0 and commit_ts else None
    if last_commit_at is None:
        # Unborn git repo (e.g. freshly initialized): use .git/HEAD or dir mtime.
        head_file = repo_dir / ".git" / "HEAD"
        target = head_file if head_file.exists() else repo_dir
        try:
            mtime = target.stat().st_mtime
            last_commit_at = datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        except OSError:
            last_commit_at = None

    rc_b, branch = _run_git(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    if rc_b != 0 or not branch or branch == "HEAD":
        rc_sym, sym_branch = _run_git(repo_dir, ["symbolic-ref", "--short", "HEAD"])
        branch = sym_branch if rc_sym == 0 and sym_branch else (branch or None)

    rc_r, remote_url = _run_git(repo_dir, ["config", "--get", "remote.origin.url"])
    return last_commit_at, branch, _sanitize_remote_url(remote_url if rc_r == 0 else None)


def _is_home_root(raw_root: str, resolved_root: Path) -> bool:
    if raw_root.strip() == "~":
        return True
    try:
        return resolved_root == Path.home().resolve(strict=False)
    except OSError:
        return False


def _context_disambiguator(repo_path: Path, scan_roots: list[Path]) -> str:
    for root in scan_roots:
        try:
            rel = repo_path.relative_to(root)
            parts = [root.name, *rel.parts[:-1]]
            clean = [_slugify(p) for p in parts if p]
            if clean:
                return "-".join(clean[-2:])
        except ValueError:
            continue
    return _slugify(repo_path.parent.name)


def _root_rank(repo_path: Path, scan_roots: list[Path]) -> int:
    """Earlier entries in `privacy.scan_roots` win ties between same-named repos."""
    for idx, root in enumerate(scan_roots):
        try:
            repo_path.relative_to(root)
            return idx
        except ValueError:
            continue
    return len(scan_roots)


def discover_git_repos(cfg: AppConfig) -> dict[str, CanonicalRepo]:
    """Walk cfg.privacy.scan_roots (depth <= 4) and discover canonical git repositories."""
    expanded_roots: list[Path] = []
    seen_repo_paths: set[Path] = set()
    visited_dirs: set[Path] = set()
    discovered_dirs: list[Path] = []

    for raw_root in cfg.privacy.scan_roots:
        root = Path(raw_root).expanduser()
        try:
            if not root.exists() or not root.is_dir():
                continue
            resolved_root = root.resolve(strict=False)
        except OSError:
            continue

        if cfg.privacy.home_top_level_files_only and _is_home_root(raw_root, resolved_root):
            # Home root is scanned for top-level files only in files adapter; skip deep repo walk here.
            continue

        excluded, _ = is_path_excluded(resolved_root, cfg.privacy)
        if excluded:
            continue

        expanded_roots.append(resolved_root)
        stack: list[tuple[Path, int]] = [(resolved_root, 0)]

        while stack:
            current_dir, depth = stack.pop()
            try:
                real_dir = current_dir.resolve(strict=False)
            except OSError:
                continue
            if real_dir in visited_dirs:
                continue
            visited_dirs.add(real_dir)

            excl_dir, _ = is_path_excluded(real_dir, cfg.privacy)
            if excl_dir:
                continue

            git_marker = real_dir / ".git"
            try:
                has_git = git_marker.exists()
            except OSError:
                has_git = False

            if has_git and real_dir not in seen_repo_paths:
                seen_repo_paths.add(real_dir)
                discovered_dirs.append(real_dir)

            if depth >= 4:
                continue

            try:
                with os.scandir(real_dir) as it:
                    entries = sorted(it, key=lambda e: e.name)
            except OSError:
                continue

            subdirs: list[Path] = []
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=True):
                        continue
                    cand = Path(entry.path)
                    cand_excluded, _ = is_path_excluded(cand, cfg.privacy)
                    if not cand_excluded:
                        subdirs.append(cand)
                except OSError:
                    continue

            for sub in reversed(subdirs):
                stack.append((sub, depth + 1))

    def _build_record(repo_dir: Path) -> tuple[CanonicalRepo, str]:
        last_commit_at, branch, remote_url = _inspect_git_repo(repo_dir)
        tier = classify_repo_tier(repo_dir, last_commit_at, scan_roots=expanded_roots)
        family = infer_project_family(repo_dir.name)
        base_slug = _slugify(repo_dir.name)
        record = CanonicalRepo(
            project_id=base_slug,
            family=family,
            name=repo_dir.name,
            path=str(repo_dir),
            tier=tier,
            last_commit_at=last_commit_at,
            branch=branch,
            remote_url=remote_url,
            manual_override=False,
        )
        return record, base_slug

    raw_records: list[tuple[CanonicalRepo, str]] = []
    if discovered_dirs:
        with ThreadPoolExecutor(max_workers=min(8, len(discovered_dirs))) as pool:
            raw_records = list(pool.map(_build_record, discovered_dirs))

    # Group by base_slug so duplicate repo names get deterministic unique project_ids.
    by_slug: dict[str, list[CanonicalRepo]] = {}
    for record, base_slug in raw_records:
        by_slug.setdefault(base_slug, []).append(record)

    repos: dict[str, CanonicalRepo] = {}
    for base_slug, group in sorted(by_slug.items()):
        def _sort_key(r: CanonicalRepo) -> tuple[int, int, float, str]:
            dt = _parse_iso_datetime(r.last_commit_at)
            ts = dt.timestamp() if dt else 0.0
            return (
                _TIER_ORDER[r.tier],
                _root_rank(Path(r.path), expanded_roots),
                -ts,
                r.path,
            )

        group_sorted = sorted(group, key=_sort_key)
        for idx, repo in enumerate(group_sorted):
            if idx == 0 and base_slug not in repos:
                pid = base_slug
            else:
                ctx = _context_disambiguator(Path(repo.path), expanded_roots)
                pid = f"{base_slug}--{ctx}"
                suffix = 2
                while pid in repos:
                    pid = f"{base_slug}--{ctx}-{suffix}"
                    suffix += 1
            repo.project_id = pid
            repos[pid] = repo

    return repos


def save_atlas(repos: dict[str, CanonicalRepo], path: Path) -> Path:
    """Persist canonical project atlas to YAML."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        repos.values(),
        key=lambda r: (_TIER_ORDER[r.tier], r.family, r.project_id),
    )
    payload = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "projects": {repo.project_id: repo.model_dump(mode="json") for repo in ordered},
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


def _load_existing_atlas(path: Path) -> dict[str, CanonicalRepo]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not raw:
        return {}

    entries: dict[str, CanonicalRepo] = {}
    if isinstance(raw, dict):
        projects_raw: Any = raw.get("projects") if "projects" in raw else raw
        if isinstance(projects_raw, dict):
            for key, val in projects_raw.items():
                if not isinstance(val, dict):
                    continue
                data = dict(val)
                data.setdefault("project_id", str(key))
                data.setdefault("name", data["project_id"])
                data.setdefault("family", infer_project_family(data["name"]))
                repo = CanonicalRepo.model_validate(data)
                entries[repo.project_id] = repo
        elif isinstance(projects_raw, list):
            for val in projects_raw:
                if not isinstance(val, dict):
                    continue
                repo = CanonicalRepo.model_validate(val)
                entries[repo.project_id] = repo
    return entries


def build_or_load_atlas(cfg: AppConfig, refresh: bool = False) -> dict[str, CanonicalRepo]:
    """Load existing atlas YAML or discover git repos and merge preserving manual overrides."""
    atlas_file = cfg.atlas_path
    existing = _load_existing_atlas(atlas_file)
    if existing and not refresh:
        return existing

    discovered = discover_git_repos(cfg)
    by_resolved_path: dict[str, CanonicalRepo] = {}
    for repo in discovered.values():
        try:
            key = str(Path(repo.path).expanduser().resolve(strict=False))
        except OSError:
            key = repo.path
        by_resolved_path[key] = repo

    merged: dict[str, CanonicalRepo] = dict(discovered)

    for old_pid, old_repo in existing.items():
        if not old_repo.manual_override:
            continue
        try:
            old_resolved = str(Path(old_repo.path).expanduser().resolve(strict=False))
        except OSError:
            old_resolved = old_repo.path

        matched = by_resolved_path.get(old_resolved) or merged.get(old_pid)
        if matched is not None:
            # Remove auto-assigned key if it differs from manually overridden project_id.
            if matched.project_id != old_repo.project_id and matched.project_id in merged:
                merged.pop(matched.project_id, None)
            merged[old_repo.project_id] = CanonicalRepo(
                project_id=old_repo.project_id,
                family=old_repo.family,
                name=old_repo.name,
                path=matched.path,
                tier=old_repo.tier,
                last_commit_at=matched.last_commit_at or old_repo.last_commit_at,
                branch=matched.branch or old_repo.branch,
                remote_url=matched.remote_url or old_repo.remote_url,
                manual_override=True,
                notes=old_repo.notes,
            )
        else:
            merged[old_repo.project_id] = old_repo

    save_atlas(merged, atlas_file)
    return merged


class RepoPathIndex:
    """Precomputed longest-prefix matcher from filesystem path to CanonicalRepo."""

    def __init__(self, atlas: dict[str, CanonicalRepo]) -> None:
        entries: list[tuple[str, CanonicalRepo]] = []
        for repo in atlas.values():
            try:
                resolved = str(Path(repo.path).expanduser().resolve(strict=False)).rstrip("/")
            except OSError:
                resolved = repo.path.rstrip("/")
            if resolved:
                entries.append((resolved, repo))
        entries.sort(key=lambda item: len(item[0]), reverse=True)
        self._entries = entries

    def match(self, path: Path | str | None, resolve_path: bool = True) -> CanonicalRepo | None:
        if not path or not self._entries:
            return None
        if resolve_path:
            try:
                target_str = str(Path(path).expanduser().resolve(strict=False)).rstrip("/")
            except OSError:
                return None
        else:
            target_str = str(path).rstrip("/")
        for repo_str, repo in self._entries:
            if target_str == repo_str or target_str.startswith(repo_str + "/"):
                return repo
        return None


def match_repo_for_path(path: Path | str | None, atlas: dict[str, CanonicalRepo]) -> CanonicalRepo | None:
    """Return the deepest CanonicalRepo containing path, or None if unassociated."""
    if not path or not atlas:
        return None
    return RepoPathIndex(atlas).match(path, resolve_path=True)


_BOUNDARY_AFTER_PATH = frozenset("/ \t\r\n\"'`),:;]>")


def is_atlas_stale(path: Path, max_age_hours: float) -> bool:
    """Return True if the atlas YAML is missing or older than `max_age_hours`."""
    if not path.exists():
        return True
    if max_age_hours <= 0:
        return False
    try:
        age_sec = time.time() - path.stat().st_mtime
    except OSError:
        return True
    return age_sec > float(max_age_hours) * 3600.0


def resolve_project_filter(
    project_id: str | None,
    atlas: dict[str, CanonicalRepo],
) -> list[str] | None:
    """Resolve an exact project_id or atlas family/name into matching project_ids."""
    if not project_id or not project_id.strip():
        return None
    raw = project_id.strip()
    if raw in atlas:
        return [raw]
    low = raw.lower()
    matches: list[str] = []
    for pid, repo in atlas.items():
        if (
            pid.lower() == low
            or repo.family.lower() == low
            or repo.name.lower() == low
            or pid.lower().startswith(low + "-")
        ):
            matches.append(pid)
    if not matches:
        return [raw]
    if raw not in matches:
        matches.append(raw)
    return list(dict.fromkeys(matches))


def _count_path_mentions(haystack: str, needle: str) -> int:
    """Count occurrences of `needle` with a path/token boundary after it."""
    if not haystack or not needle:
        return 0
    count = 0
    start = 0
    n_len = len(needle)
    h_len = len(haystack)
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        after_idx = idx + n_len
        if after_idx >= h_len or haystack[after_idx] in _BOUNDARY_AFTER_PATH:
            count += 1
        start = idx + 1
    return count


def match_repo_from_text(
    text: str,
    atlas: dict[str, CanonicalRepo],
    *,
    early_text: str | None = None,
) -> CanonicalRepo | None:
    """Best-effort match of a git repo path mentioned in transcript or artifact text."""
    if not text or not atlas:
        return None
    early = early_text if early_text is not None else text[:4000]
    candidates: list[tuple[tuple[int, int, int], CanonicalRepo]] = []
    for repo in atlas.values():
        repo_path = repo.path.rstrip("/")
        if len(repo_path) <= 4 or repo_path not in text:
            continue
        total_hits = _count_path_mentions(text, repo_path)
        if total_hits <= 0:
            continue
        early_hits = _count_path_mentions(early, repo_path) if early else 0
        weighted = total_hits + 2 * early_hits
        active_pref = 1 if repo.tier == RepoTier.ACTIVE else 0
        candidates.append(((weighted, active_pref, len(repo_path)), repo))

    if not candidates:
        return None

    non_archive = [c for c in candidates if c[1].tier != RepoTier.ARCHIVE]
    pool = non_archive if non_archive else candidates
    pool.sort(key=lambda item: item[0], reverse=True)
    best_score, best_repo = pool[0]

    # Guard against demoting a long general/multi-project transcript to ARCHIVE on a single stray mention.
    if best_repo.tier == RepoTier.ARCHIVE and len(text) > 5000:
        total_mentions = _count_path_mentions(text, best_repo.path.rstrip("/"))
        if total_mentions < 2:
            return None
    _ = best_score
    return best_repo
