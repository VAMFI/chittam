"""Filesystem, code, markdown, PDF, and extensible multimodal adapter."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pymupdf

from chittam.atlas import RepoPathIndex
from chittam.config import compile_privacy_policy, is_path_excluded, redact_secrets
from chittam.models import AppConfig, CanonicalRepo, DocumentRecord, MediaType, RepoTier, SyncStats
from chittam.adapters import (
    build_contextual_header,
    chunk_code,
    chunk_markdown_or_text,
    estimate_tokens,
    make_chunk_id,
    make_doc_id,
    make_l0_summary,
    normalize_iso_timestamp,
    sha256_bytes,
    sha256_text,
)

_MARKDOWN_EXTS = frozenset({".md", ".mdx", ".rst"})
_TEXT_EXTS = frozenset({".txt", ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".conf", ".csv"})
_CODE_EXTS = frozenset(
    {
        ".py",
        ".pyi",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".dart",
        ".rs",
        ".go",
        ".sh",
        ".bash",
        ".zsh",
        ".java",
        ".kt",
        ".kts",
        ".swift",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".sql",
        ".rb",
        ".php",
    }
)
_PDF_EXTS = frozenset({".pdf"})
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})

_NOISE_FILENAMES = frozenset(
    {
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "uv.lock",
        "cargo.lock",
        "composer.lock",
        "gemfile.lock",
        "podfile.lock",
        ".ds_store",
        "thumbs.db",
    }
)

_NOISE_SUFFIXES = (
    ".min.js",
    ".min.css",
    ".map",
    ".g.dart",
    ".freezed.dart",
    ".pb.go",
)

_NOISE_DIR_NAMES = frozenset({"coverage", "htmlcov", "storybook-static", ".expo", ".fvm"})

_IMAGE_NOISE_RE = re.compile(
    r"(?:favicon|ic_launcher|mipmap|appiconset|xcassets|app[-_]?icon|icon[-_]?app|icon[-_]?maskable|play[-_]?icon|splash|badge|emoji|spinner|placeholder|@(?:1|2|3)x|(?:^|[/_-])(?:16|24|32|48|64)x(?:16|24|32|48|64)(?:[._/-]|$)|web/icons/)",
    re.IGNORECASE,
)

_TIER_PRIORITY = {
    RepoTier.ACTIVE: 0,
    RepoTier.REFERENCE: 1,
    RepoTier.ARCHIVE: 2,
}

_DOC_PREFIX_BY_MEDIA: dict[MediaType, str] = {
    MediaType.MARKDOWN: "file_md",
    MediaType.TEXT: "file_txt",
    MediaType.CODE: "file_code",
    MediaType.PDF: "file_pdf",
    MediaType.IMAGE: "file_img",
}

UNREADABLE_IMAGE_SENTINEL = "__CHITTAM_UNREADABLE_IMAGE__"


@dataclass
class FileContext:
    """Shared ingestion context passed to registered media handlers."""

    cfg: AppConfig
    atlas: dict[str, CanonicalRepo]
    repo: CanonicalRepo | None
    stats: SyncStats
    vision_fn: Callable[[Path, str], str] | None = None
    image_cache_lookup: Callable[[str], str | None] | None = None
    code_count_by_repo: dict[str, int] | None = None
    new_images_captioned: int = 0


@dataclass
class MediaHandler:
    """Extensible media handler descriptor for MEDIA_HANDLERS registry."""

    media_type: MediaType
    extensions: frozenset[str]
    handler: Callable[[Path, FileContext], DocumentRecord | None]


def _file_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _is_noise_file(path: Path) -> bool:
    name = path.name
    low = name.lower()
    if name.startswith("."):
        return True
    if low in _NOISE_FILENAMES:
        return True
    if low.endswith(_NOISE_SUFFIXES):
        return True
    if ".backup" in low:
        return True
    return False


def _is_noise_image_path(path: Path) -> bool:
    for part in path.parts:
        if part.startswith("."):
            return True
    return bool(_IMAGE_NOISE_RE.search(path.as_posix()))


def _read_text_limited(path: Path, max_bytes: int) -> str | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > max_bytes:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", errors="replace")


def _handle_markdown(path: Path, ctx: FileContext) -> DocumentRecord | None:
    raw = _read_text_limited(path, ctx.cfg.privacy.max_text_file_bytes)
    if not raw:
        return None
    text = redact_secrets(raw).strip()
    if not text:
        return None

    proj_id = ctx.repo.project_id if ctx.repo else None
    tier = ctx.repo.tier if ctx.repo else RepoTier.REFERENCE
    updated_at = normalize_iso_timestamp(None, fallback_mtime=_file_mtime(path))
    doc_id = make_doc_id("file_md", str(path))
    chunks = chunk_markdown_or_text(
        doc_id=doc_id,
        text=text,
        project_id=proj_id,
        canonical_tier=tier,
        path_or_uri=str(path),
        updated_at=updated_at,
    )
    if not chunks:
        return None
    return DocumentRecord(
        doc_id=doc_id,
        uri=str(path),
        source_harness="files",
        media_type=MediaType.MARKDOWN,
        project_id=proj_id,
        canonical_tier=tier,
        title=path.name,
        l0_summary=make_l0_summary(text),
        sha256=sha256_text(text),
        updated_at=updated_at,
        metadata={"extension": path.suffix.lower()},
        chunks=chunks,
    )


def _handle_text(path: Path, ctx: FileContext) -> DocumentRecord | None:
    max_bytes = ctx.cfg.privacy.max_text_file_bytes
    if path.suffix.lower() == ".json":
        max_bytes = min(max_bytes, 150_000)
    raw = _read_text_limited(path, max_bytes)
    if not raw:
        return None
    text = redact_secrets(raw).strip()
    if not text:
        return None

    proj_id = ctx.repo.project_id if ctx.repo else None
    tier = ctx.repo.tier if ctx.repo else RepoTier.REFERENCE
    updated_at = normalize_iso_timestamp(None, fallback_mtime=_file_mtime(path))
    doc_id = make_doc_id("file_txt", str(path))
    chunks = chunk_markdown_or_text(
        doc_id=doc_id,
        text=text,
        project_id=proj_id,
        canonical_tier=tier,
        path_or_uri=str(path),
        updated_at=updated_at,
    )
    if not chunks:
        return None
    return DocumentRecord(
        doc_id=doc_id,
        uri=str(path),
        source_harness="files",
        media_type=MediaType.TEXT,
        project_id=proj_id,
        canonical_tier=tier,
        title=path.name,
        l0_summary=make_l0_summary(text),
        sha256=sha256_text(text),
        updated_at=updated_at,
        metadata={"extension": path.suffix.lower()},
        chunks=chunks,
    )


def _handle_code(path: Path, ctx: FileContext) -> DocumentRecord | None:
    if ctx.code_count_by_repo is not None and ctx.repo is not None:
        count = ctx.code_count_by_repo.get(ctx.repo.project_id, 0)
        cap = ctx.cfg.privacy.max_code_files_per_repo
        if ctx.repo.tier == RepoTier.ARCHIVE:
            cap = min(cap, 15)
        if count >= cap:
            return None

    raw = _read_text_limited(path, ctx.cfg.privacy.max_text_file_bytes)
    if not raw:
        return None
    text = redact_secrets(raw).strip()
    if not text:
        return None

    if ctx.code_count_by_repo is not None and ctx.repo is not None:
        ctx.code_count_by_repo[ctx.repo.project_id] = (
            ctx.code_count_by_repo.get(ctx.repo.project_id, 0) + 1
        )

    proj_id = ctx.repo.project_id if ctx.repo else None
    tier = ctx.repo.tier if ctx.repo else RepoTier.REFERENCE
    updated_at = normalize_iso_timestamp(None, fallback_mtime=_file_mtime(path))
    doc_id = make_doc_id("file_code", str(path))
    chunks = chunk_code(
        doc_id=doc_id,
        text=text,
        suffix=path.suffix.lower(),
        project_id=proj_id,
        canonical_tier=tier,
        path_or_uri=str(path),
        updated_at=updated_at,
    )
    if not chunks:
        return None
    return DocumentRecord(
        doc_id=doc_id,
        uri=str(path),
        source_harness="files",
        media_type=MediaType.CODE,
        project_id=proj_id,
        canonical_tier=tier,
        title=path.name,
        l0_summary=make_l0_summary(text),
        sha256=sha256_text(text),
        updated_at=updated_at,
        metadata={"extension": path.suffix.lower()},
        chunks=chunks,
    )


def _handle_pdf(path: Path, ctx: FileContext) -> DocumentRecord | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > ctx.cfg.privacy.max_pdf_bytes:
        return None

    try:
        with pymupdf.open(path) as pdf_doc:
            total_pages = len(pdf_doc)
            limit_pages = min(total_pages, ctx.cfg.privacy.max_pdf_pages)
            page_sections: list[str] = []
            sparse_pages: list[int] = []
            for idx in range(limit_pages):
                page = pdf_doc[idx]
                txt = (page.get_text("text") or "").strip()
                non_ws = len(re.sub(r"\s+", "", txt))
                if non_ws < 40:
                    sparse_pages.append(idx + 1)
                    img_cnt = len(page.get_images(full=False))
                    if txt:
                        page_sections.append(
                            f"## Page {idx + 1} (sparse text, {img_cnt} images)\n{txt}"
                        )
                    else:
                        page_sections.append(
                            f"## Page {idx + 1} (sparse/scanned page, {img_cnt} images)"
                        )
                else:
                    page_sections.append(f"## Page {idx + 1}\n{txt}")
            if total_pages > limit_pages:
                page_sections.append(f"[Truncated after {limit_pages} of {total_pages} pages]")
    except Exception as exc:
        ctx.stats.errors.append(f"pdf {path}: {exc}")
        return None

    full_text = redact_secrets("\n\n".join(page_sections)).strip()
    if not full_text:
        return None

    proj_id = ctx.repo.project_id if ctx.repo else None
    tier = ctx.repo.tier if ctx.repo else RepoTier.REFERENCE
    updated_at = normalize_iso_timestamp(None, fallback_mtime=_file_mtime(path))
    doc_id = make_doc_id("file_pdf", str(path))
    chunks = chunk_markdown_or_text(
        doc_id=doc_id,
        text=full_text,
        project_id=proj_id,
        canonical_tier=tier,
        path_or_uri=str(path),
        updated_at=updated_at,
    )
    if not chunks:
        return None
    return DocumentRecord(
        doc_id=doc_id,
        uri=str(path),
        source_harness="files",
        media_type=MediaType.PDF,
        project_id=proj_id,
        canonical_tier=tier,
        title=path.name,
        l0_summary=make_l0_summary(full_text),
        sha256=sha256_text(full_text),
        updated_at=updated_at,
        metadata={
            "total_pages": total_pages,
            "sparse_pages": sparse_pages,
        },
        chunks=chunks,
    )


def _normalize_caption(raw_caption: object) -> str:
    if isinstance(raw_caption, str):
        return raw_caption.strip()
    if isinstance(raw_caption, dict):
        parts = [f"{k}: {v}" for k, v in raw_caption.items() if v]
        return "\n".join(parts).strip()
    return ""


def _handle_image(path: Path, ctx: FileContext) -> DocumentRecord | None:
    if _is_noise_image_path(path):
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size < ctx.cfg.privacy.min_image_bytes or size > ctx.cfg.privacy.max_image_bytes:
        return None

    try:
        img_sha = sha256_bytes(path.read_bytes())
    except OSError:
        return None

    caption = ""
    from_cache = False
    if ctx.image_cache_lookup is not None:
        try:
            cached = ctx.image_cache_lookup(img_sha)
            if cached == UNREADABLE_IMAGE_SENTINEL:
                return None
            if cached and cached.strip():
                caption = cached.strip()
                from_cache = True
                ctx.stats.images_cached += 1
        except Exception as exc:
            ctx.stats.errors.append(f"image_cache_lookup {path}: {exc}")

    if not caption:
        if ctx.vision_fn is None:
            return None
        if ctx.new_images_captioned >= ctx.cfg.privacy.max_images_per_sync:
            return None
        hint_parts = [f"filename={path.name}", f"parent={path.parent.name}"]
        if ctx.repo:
            hint_parts.append(f"project={ctx.repo.project_id}")
        try:
            raw_cap = ctx.vision_fn(path, ", ".join(hint_parts))
            caption = redact_secrets(_normalize_caption(raw_cap)).strip()
            if caption:
                ctx.new_images_captioned += 1
                ctx.stats.images_captioned += 1
        except Exception as exc:
            ctx.stats.errors.append(f"vision {path}: {exc}")
            return None

    if not caption:
        return None

    caption = redact_secrets(caption)
    proj_id = ctx.repo.project_id if ctx.repo else None
    tier = ctx.repo.tier if ctx.repo else RepoTier.REFERENCE
    updated_at = normalize_iso_timestamp(None, fallback_mtime=_file_mtime(path))
    doc_id = make_doc_id("file_img", str(path))
    header = build_contextual_header(
        project_id=proj_id,
        canonical_tier=tier,
        path_or_uri=str(path),
        updated_at=updated_at,
    )
    body = f"Image: {path.name}\n{caption}"
    chunk = chunk_markdown_or_text(
        doc_id=doc_id,
        text=body,
        project_id=proj_id,
        canonical_tier=tier,
        path_or_uri=str(path),
        updated_at=updated_at,
    )
    if not chunk:
        chunk = [
            # Fallback single chunk if caption was terse
            # (normally chunk_markdown_or_text produces at least 1 chunk)
        ]
    if not chunk:
        from chittam.models import ChunkRecord

        chunk = [
            ChunkRecord(
                chunk_id=make_chunk_id(doc_id, 0),
                doc_id=doc_id,
                chunk_index=0,
                contextual_header=header,
                content=body,
                token_estimate=estimate_tokens(header + "\n" + body),
            )
        ]

    return DocumentRecord(
        doc_id=doc_id,
        uri=str(path),
        source_harness="files",
        media_type=MediaType.IMAGE,
        project_id=proj_id,
        canonical_tier=tier,
        title=path.name,
        l0_summary=make_l0_summary(caption),
        sha256=img_sha,
        updated_at=updated_at,
        metadata={
            "image_sha256": img_sha,
            "vision_caption": caption,
            "from_cache": from_cache,
            "bytes": size,
        },
        chunks=chunk,
    )


MEDIA_HANDLERS: dict[MediaType, MediaHandler] = {
    MediaType.MARKDOWN: MediaHandler(
        media_type=MediaType.MARKDOWN,
        extensions=_MARKDOWN_EXTS,
        handler=_handle_markdown,
    ),
    MediaType.TEXT: MediaHandler(
        media_type=MediaType.TEXT,
        extensions=_TEXT_EXTS,
        handler=_handle_text,
    ),
    MediaType.CODE: MediaHandler(
        media_type=MediaType.CODE,
        extensions=_CODE_EXTS,
        handler=_handle_code,
    ),
    MediaType.PDF: MediaHandler(
        media_type=MediaType.PDF,
        extensions=_PDF_EXTS,
        handler=_handle_pdf,
    ),
    MediaType.IMAGE: MediaHandler(
        media_type=MediaType.IMAGE,
        extensions=_IMAGE_EXTS,
        handler=_handle_image,
    ),
}


def register_media_handler(
    media_type: MediaType,
    extensions: frozenset[str] | set[str],
    handler: Callable[[Path, FileContext], DocumentRecord | None],
) -> None:
    """Register or replace a media handler (e.g. future MediaType.AUDIO / VIDEO adapters)."""
    MEDIA_HANDLERS[media_type] = MediaHandler(
        media_type=media_type,
        extensions=frozenset(ext.lower() for ext in extensions),
        handler=handler,
    )


def _active_extension_map(enabled_media: list[MediaType]) -> dict[str, MediaHandler]:
    enabled_set = set(enabled_media)
    ext_map: dict[str, MediaHandler] = {}
    for spec in MEDIA_HANDLERS.values():
        if spec.media_type in enabled_set:
            for ext in spec.extensions:
                ext_map[ext.lower()] = spec
    return ext_map


def _is_home_root(raw_root: str, resolved_root: Path) -> bool:
    if raw_root.strip() == "~":
        return True
    try:
        return resolved_root == Path.home().resolve(strict=False)
    except OSError:
        return False


def _stamp_file_stat(doc: DocumentRecord, path: Path) -> DocumentRecord:
    try:
        st = path.stat()
        doc.metadata.setdefault("file_mtime_ns", int(st.st_mtime_ns))
        doc.metadata.setdefault("file_size", int(st.st_size))
    except OSError:
        pass
    return doc


def _check_stat_unchanged(
    real_file: Path,
    spec: MediaHandler,
    repo: CanonicalRepo | None,
    existing_stats: dict[str, tuple[int | None, int | None, str, str, str]] | None,
    code_counts: dict[str, int],
    cfg: AppConfig,
    *,
    file_stat: os.stat_result | None = None,
    existing_projects: dict[str, str | None] | None = None,
) -> str | None:
    if not existing_stats:
        return None
    prefix = _DOC_PREFIX_BY_MEDIA.get(spec.media_type)
    if prefix is None:
        return None
    candidate_doc_id = make_doc_id(prefix, str(real_file))
    prev = existing_stats.get(candidate_doc_id)
    if prev is None:
        return None
    prev_mtime_ns, prev_size, _prev_sha, prev_tier, prev_l0 = prev
    if prev_mtime_ns is None or prev_size is None:
        return None

    expected_tier = repo.tier.value if repo else RepoTier.REFERENCE.value
    if prev_tier != expected_tier:
        return None

    if existing_projects is not None and candidate_doc_id in existing_projects:
        expected_proj = repo.project_id if repo else None
        if existing_projects[candidate_doc_id] != expected_proj:
            return None

    if spec.media_type == MediaType.IMAGE:
        if not prev_l0 or prev_l0.startswith("[image uncaptioned"):
            return None

    try:
        st = file_stat if file_stat is not None else real_file.stat()
    except OSError:
        return None

    if int(st.st_mtime_ns) != int(prev_mtime_ns) or int(st.st_size) != int(prev_size):
        return None

    if spec.media_type == MediaType.CODE and repo is not None:
        count = code_counts.get(repo.project_id, 0)
        cap = (
            min(cfg.privacy.max_code_files_per_repo, 15)
            if repo.tier == RepoTier.ARCHIVE
            else cfg.privacy.max_code_files_per_repo
        )
        if count >= cap:
            return None
        code_counts[repo.project_id] = count + 1

    return candidate_doc_id


def load_single_file_doc(
    path: Path,
    cfg: AppConfig,
    atlas: dict[str, CanonicalRepo],
    *,
    vision_fn: Callable[[Path, str], str] | None = None,
    image_cache_lookup: Callable[[str], str | None] | None = None,
    stats: SyncStats | None = None,
) -> tuple[DocumentRecord | None, str | None]:
    """Load a single file into a DocumentRecord; returns (doc_or_none, deterministic_doc_id)."""
    sync_stats = stats if stats is not None else SyncStats()
    p = Path(path).expanduser()
    try:
        real_file = p.resolve(strict=False)
    except OSError:
        real_file = p

    excl, _ = is_path_excluded(p, cfg.privacy)
    if excl:
        sync_stats.skipped_excluded += 1
        return None, None
    if real_file != p:
        excl_real, _ = is_path_excluded(real_file, cfg.privacy)
        if excl_real:
            sync_stats.skipped_excluded += 1
            return None, None

    if _is_noise_file(real_file):
        return None, None
    for part in real_file.parts[:-1]:
        if part.startswith(".") or part.lower() in _NOISE_DIR_NAMES:
            return None, None

    ext_map = _active_extension_map(cfg.privacy.enabled_media)
    spec = ext_map.get(real_file.suffix.lower())
    if spec is None:
        return None, None

    prefix = _DOC_PREFIX_BY_MEDIA.get(spec.media_type)
    doc_id = make_doc_id(prefix, str(real_file)) if prefix else None
    if not real_file.exists() or not real_file.is_file():
        return None, doc_id

    if spec.media_type == MediaType.IMAGE and _is_noise_image_path(real_file):
        return None, doc_id

    repo = RepoPathIndex(atlas).match(real_file, resolve_path=False)
    ctx = FileContext(
        cfg=cfg,
        atlas=atlas,
        repo=repo,
        stats=sync_stats,
        vision_fn=vision_fn,
        image_cache_lookup=image_cache_lookup,
        code_count_by_repo=None,
    )
    doc = spec.handler(real_file, ctx)
    if doc is not None:
        _stamp_file_stat(doc, real_file)
        return doc, doc.doc_id
    return None, doc_id


def _is_noise_name(name: str) -> bool:
    if name.startswith("."):
        return True
    low = name.lower()
    if low in _NOISE_FILENAMES:
        return True
    if low.endswith(_NOISE_SUFFIXES):
        return True
    if ".backup" in low:
        return True
    return False


def load_file_docs(
    cfg: AppConfig,
    atlas: dict[str, CanonicalRepo],
    vision_fn: Callable[[Path, str], str] | None = None,
    image_cache_lookup: Callable[[str], str | None] | None = None,
    stats: SyncStats | None = None,
    existing_stats: dict[str, tuple[int | None, int | None, str, str, str]] | None = None,
    unchanged_doc_ids: set[str] | None = None,
    existing_projects: dict[str, str | None] | None = None,
    unreadable_prefixes: set[str] | None = None,
) -> list[DocumentRecord]:
    """Scan cfg.privacy.scan_roots with compiled privacy exclusion, stat() fast-path, and media dispatch."""
    sync_stats = stats if stats is not None else SyncStats()
    ext_map = _active_extension_map(cfg.privacy.enabled_media)
    if not ext_map:
        return []

    privacy_matcher = compile_privacy_policy(cfg.privacy)
    docs: list[DocumentRecord] = []
    seen_real_files: set[str] = set()
    visited_dirs: set[Path] = set()
    deferred_images: list[tuple[int, int, float, Path, CanonicalRepo | None]] = []
    code_counts: dict[str, int] = {}
    repo_index = RepoPathIndex(atlas)

    def _process_scandir_file(
        entry: os.DirEntry[str],
        depth_in_root: int,
        dir_repo: CanonicalRepo | None,
        *,
        is_symlink: bool,
    ) -> None:
        # Enforce privacy exclusions BEFORE stat() or reading!
        if not is_symlink:
            excl, _ = privacy_matcher.check(entry.path, name=entry.name, is_dir=False)
            if excl:
                sync_stats.skipped_excluded += 1
                return
            if _is_noise_name(entry.name):
                return
            ext = os.path.splitext(entry.name)[1].lower()
            spec = ext_map.get(ext)
            if spec is None:
                return
            real_path_str = entry.path
            real_file = Path(real_path_str)
            repo = dir_repo
            try:
                entry_stat = entry.stat(follow_symlinks=True)
            except OSError:
                entry_stat = None
        else:
            cand = Path(entry.path)
            excl, _ = privacy_matcher.check(entry.path, raw_path_str=entry.path, name=entry.name, is_dir=False)
            if excl:
                sync_stats.skipped_excluded += 1
                return
            if _is_noise_name(entry.name):
                return
            ext = os.path.splitext(entry.name)[1].lower()
            spec = ext_map.get(ext)
            if spec is None:
                return
            try:
                real_file = cand.resolve(strict=False)
            except OSError:
                return
            real_path_str = str(real_file)
            if real_path_str != entry.path:
                excl_real, _ = privacy_matcher.check(real_path_str, is_dir=False)
                if excl_real:
                    sync_stats.skipped_excluded += 1
                    return
            repo = repo_index.match(real_file, resolve_path=False)
            entry_stat = None

        if real_path_str in seen_real_files:
            return
        seen_real_files.add(real_path_str)

        if spec.media_type == MediaType.IMAGE and _is_noise_image_path(real_file):
            return

        unchanged_id = _check_stat_unchanged(
            real_file=real_file,
            spec=spec,
            repo=repo,
            existing_stats=existing_stats,
            code_counts=code_counts,
            cfg=cfg,
            file_stat=entry_stat,
            existing_projects=existing_projects,
        )
        if unchanged_id is not None:
            if unchanged_doc_ids is not None:
                unchanged_doc_ids.add(unchanged_id)
            sync_stats.stat_skipped += 1
            return

        if spec.media_type == MediaType.IMAGE:
            if entry_stat is not None:
                mtime = float(entry_stat.st_mtime)
            else:
                mtime = _file_mtime(real_file) or 0.0
            tier_rank = _TIER_PRIORITY[repo.tier] if repo else 0
            deferred_images.append((tier_rank, depth_in_root, -mtime, real_file, repo))
            return

        ctx = FileContext(
            cfg=cfg,
            atlas=atlas,
            repo=repo,
            stats=sync_stats,
            vision_fn=vision_fn,
            image_cache_lookup=image_cache_lookup,
            code_count_by_repo=code_counts,
        )
        doc = spec.handler(real_file, ctx)
        if doc is not None:
            _stamp_file_stat(doc, real_file)
            docs.append(doc)

    def _record_unreadable(path_obj: Path) -> None:
        if unreadable_prefixes is not None:
            for candidate in (path_obj, path_obj.resolve(strict=False)):
                unreadable_prefixes.add(str(candidate).rstrip("/"))

    for raw_root in cfg.privacy.scan_roots:
        root = Path(raw_root).expanduser()
        try:
            exists = root.exists()
            is_dir = exists and root.is_dir()
            resolved_root = root.resolve(strict=False)
        except OSError as exc:
            _record_unreadable(root)
            sync_stats.errors.append(f"scandir {root}: {exc}")
            continue
        if not is_dir:
            continue

        root_excl, _ = privacy_matcher.check(
            str(resolved_root),
            raw_path_str=str(root),
            is_dir=True,
        )
        if root_excl:
            sync_stats.skipped_excluded += 1
            continue

        if cfg.privacy.home_top_level_files_only and _is_home_root(raw_root, resolved_root):
            try:
                with os.scandir(resolved_root) as it:
                    entries = sorted(it, key=lambda e: e.name)
            except OSError as exc:
                _record_unreadable(resolved_root)
                sync_stats.errors.append(f"scandir {resolved_root}: {exc}")
                continue

            home_repo = repo_index.match(resolved_root, resolve_path=False)
            for entry in entries:
                try:
                    is_sym = entry.is_symlink()
                    if entry.is_dir(follow_symlinks=False):
                        excl_d, _ = privacy_matcher.check(entry.path, name=entry.name, is_dir=True)
                        if excl_d:
                            sync_stats.skipped_excluded += 1
                        continue
                    if entry.is_file(follow_symlinks=True):
                        _process_scandir_file(entry, depth_in_root=0, dir_repo=home_repo, is_symlink=is_sym)
                except OSError:
                    continue
            continue

        stack: list[tuple[Path, int, bool]] = [(resolved_root, 0, True)]
        while stack:
            cur_dir, depth, already_checked = stack.pop()
            try:
                real_dir = cur_dir if already_checked else cur_dir.resolve(strict=False)
            except OSError:
                continue
            if real_dir in visited_dirs:
                continue
            visited_dirs.add(real_dir)

            if not already_checked:
                excl_dir, _ = privacy_matcher.check(str(real_dir), raw_path_str=str(cur_dir), is_dir=True)
                if excl_dir:
                    sync_stats.skipped_excluded += 1
                    continue

            dir_repo = repo_index.match(real_dir, resolve_path=False)
            try:
                with os.scandir(real_dir) as it:
                    entries = sorted(it, key=lambda e: e.name)
            except OSError as exc:
                _record_unreadable(real_dir)
                sync_stats.errors.append(f"scandir {real_dir}: {exc}")
                continue

            subdirs: list[tuple[Path, bool]] = []
            for entry in entries:
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue

                if is_dir:
                    excl_sub, _ = privacy_matcher.check(entry.path, name=entry.name, is_dir=True)
                    if excl_sub:
                        sync_stats.skipped_excluded += 1
                        continue
                    if entry.name.startswith(".") or entry.name.lower() in _NOISE_DIR_NAMES:
                        continue
                    subdirs.append((Path(entry.path), True))
                else:
                    try:
                        is_sym = entry.is_symlink()
                        if entry.is_file(follow_symlinks=True):
                            _process_scandir_file(
                                entry,
                                depth_in_root=depth,
                                dir_repo=dir_repo,
                                is_symlink=is_sym,
                            )
                    except OSError:
                        continue

            for sub, sub_checked in reversed(subdirs):
                stack.append((sub, depth + 1, sub_checked))

    # Process images in priority order (active repos & shallow files first, newest mtime first).
    # Cached lookups are free; new vision_fn captions are capped at max_images_per_sync.
    img_spec = MEDIA_HANDLERS.get(MediaType.IMAGE)
    if img_spec is not None and MediaType.IMAGE in set(cfg.privacy.enabled_media):
        img_ctx = FileContext(
            cfg=cfg,
            atlas=atlas,
            repo=None,
            stats=sync_stats,
            vision_fn=vision_fn,
            image_cache_lookup=image_cache_lookup,
            code_count_by_repo=code_counts,
        )
        for _, _, _, img_path, img_repo in sorted(deferred_images, key=lambda item: item[:4]):
            img_ctx.repo = img_repo
            doc = img_spec.handler(img_path, img_ctx)
            if doc is not None:
                _stamp_file_stat(doc, img_path)
                docs.append(doc)

    return docs
