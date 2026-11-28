"""Adapter aggregation, structural chunking, and shared normalization helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from chittam.models import AppConfig, CanonicalRepo, ChunkRecord, DocumentRecord, RepoTier, SyncStats

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

_LANG_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "tsjs",
    ".tsx": "tsjs",
    ".js": "tsjs",
    ".jsx": "tsjs",
    ".mjs": "tsjs",
    ".cjs": "tsjs",
    ".dart": "dart",
    ".rs": "rust",
    ".go": "go",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".kt": "jvm_swift",
    ".kts": "jvm_swift",
    ".java": "jvm_swift",
    ".swift": "jvm_swift",
    ".rb": "ruby",
    ".php": "php",
}

_BOUNDARY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(r"^[ \t]*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\("),
        re.compile(r"^[ \t]*class\s+([A-Za-z_]\w*)(?:\s*[\(:])?"),
    ],
    "tsjs": [
        re.compile(
            r"^[ \t]*(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:async\s+)?(?:function\*?|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)"
        ),
        re.compile(
            r"^[ \t]*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"
        ),
    ],
    "dart": [
        re.compile(
            r"^[ \t]*(?:abstract\s+|sealed\s+|final\s+|base\s+|interface\s+)?(?:class|mixin|enum|extension)\s+([A-Za-z_]\w*)"
        ),
        re.compile(
            r"^[ \t]*(?:[A-Za-z_][\w<>, ?]*\s+)?([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:async\s*)?\{"
        ),
    ],
    "rust": [
        re.compile(
            r"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:fn|struct|enum|trait|impl|mod|type)\s+([A-Za-z_]\w*)"
        ),
    ],
    "go": [
        re.compile(r"^func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\("),
        re.compile(r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)\b"),
    ],
    "shell": [
        re.compile(r"^[ \t]*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\s*\)\s*\{"),
    ],
    "jvm_swift": [
        re.compile(
            r"^[ \t]*(?:public|private|internal|protected|open|final|abstract|data|sealed|suspend|override|static|\s)*(?:class|interface|object|struct|enum|protocol|extension|fun|func)\s+([A-Za-z_]\w*)"
        ),
    ],
    "ruby": [
        re.compile(r"^[ \t]*(?:def|class|module)\s+([A-Za-z_][\w.:!?=]*)"),
    ],
    "php": [
        re.compile(r"^[ \t]*(?:public|private|protected|static|abstract|final|\s)*(?:function|class|interface|trait)\s+([A-Za-z_]\w*)"),
    ],
}


def sha256_text(text: str) -> str:
    """Compute SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def make_doc_id(prefix: str, unique_key: str) -> str:
    """Create a deterministic document ID from adapter prefix and canonical key."""
    digest = hashlib.sha256(unique_key.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Create a deterministic chunk ID."""
    return f"{doc_id}#c{chunk_index}"


def estimate_tokens(text: str) -> int:
    """Lightweight token estimate (~4 chars/token)."""
    return max(1, (len(text) + 3) // 4)


def normalize_iso_timestamp(
    value: str | int | float | None,
    fallback_mtime: float | None = None,
) -> str:
    """Normalize ISO string or epoch seconds/ms into an ISO-8601 timestamp string."""
    if isinstance(value, str):
        raw = value.strip()
        if raw:
            if raw.isdigit():
                value = int(raw)
            else:
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                try:
                    dt = datetime.fromisoformat(raw)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone().isoformat(timespec="seconds")
                except ValueError:
                    pass
    if isinstance(value, (int, float)) and value > 0:
        sec = float(value) / 1000.0 if float(value) > 1e11 else float(value)
        try:
            return datetime.fromtimestamp(sec, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        except (OSError, OverflowError, ValueError):
            pass
    if fallback_mtime is not None and fallback_mtime > 0:
        try:
            return datetime.fromtimestamp(fallback_mtime, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        except (OSError, OverflowError, ValueError):
            pass
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def make_l0_summary(text: str, max_chars: int = 240) -> str:
    """Build a compact single-line L0 summary snippet."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def build_contextual_header(
    project_id: str | None,
    canonical_tier: RepoTier,
    path_or_uri: str,
    updated_at: str,
    symbol_name: str | None = None,
) -> str:
    """Format standardized contextual header stamped onto every ChunkRecord."""
    proj = project_id or "global"
    tier = canonical_tier.value if isinstance(canonical_tier, RepoTier) else str(canonical_tier)
    date_part = updated_at[:19] if updated_at else "unknown"
    if symbol_name:
        return f"[{proj} | {tier} | {path_or_uri} | {symbol_name} | {date_part}]"
    return f"[{proj} | {tier} | {path_or_uri} | {date_part}]"


def _split_oversized_lines(
    lines: list[str],
    base_line: int,
    max_chars: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    """Split oversized line range into overlapping chunks respecting line boundaries."""
    chunks: list[tuple[int, int, str]] = []
    n = len(lines)
    idx = 0
    while idx < n:
        cur_chars = 0
        end = idx
        while end < n:
            line_len = len(lines[end]) + 1
            if end > idx and cur_chars + line_len > max_chars:
                break
            cur_chars += line_len
            end += 1

        # If a single line exceeds max_chars, hard-slice it cleanly without losing line tracking.
        chunk_text = "\n".join(lines[idx:end]).strip()
        if len(chunk_text) > max_chars and end - idx == 1:
            single = lines[idx]
            step = max(200, max_chars - overlap)
            for pos in range(0, len(single), step):
                part = single[pos : pos + max_chars].strip()
                if part:
                    chunks.append((base_line + idx, base_line + idx, part))
            idx = end
            continue

        if chunk_text:
            chunks.append((base_line + idx, base_line + end - 1, chunk_text))

        if end >= n:
            break

        # Compute line-level overlap from the tail of [idx, end).
        ov_chars = 0
        next_idx = end
        while next_idx > idx + 1 and ov_chars < overlap:
            next_idx -= 1
            ov_chars += len(lines[next_idx]) + 1
        idx = next_idx if next_idx < end else end

    return chunks


def _pack_blocks(
    doc_id: str,
    blocks: list[tuple[str | None, int, int, str]],
    project_id: str | None,
    canonical_tier: RepoTier,
    path_or_uri: str,
    updated_at: str,
    max_chars: int,
    overlap: int,
) -> list[ChunkRecord]:
    """Merge small adjacent blocks and split oversized blocks into ChunkRecords."""
    merged_pieces: list[tuple[str | None, int, int, str]] = []
    cur_symbol: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None
    cur_parts: list[str] = []
    cur_len = 0

    def _flush() -> None:
        nonlocal cur_symbol, cur_start, cur_end, cur_parts, cur_len
        if not cur_parts or cur_start is None or cur_end is None:
            cur_symbol = None
            cur_start = None
            cur_end = None
            cur_parts = []
            cur_len = 0
            return
        body = "\n\n".join(cur_parts).strip()
        if body:
            merged_pieces.append((cur_symbol, cur_start, cur_end, body))
        cur_symbol = None
        cur_start = None
        cur_end = None
        cur_parts = []
        cur_len = 0

    for symbol, start_ln, end_ln, block_text in blocks:
        text = block_text.strip()
        if not text:
            continue
        if len(text) > max_chars:
            _flush()
            for sub_start, sub_end, sub_text in _split_oversized_lines(
                text.splitlines(), start_ln, max_chars=max_chars, overlap=overlap
            ):
                merged_pieces.append((symbol, sub_start, sub_end, sub_text))
            continue

        join_extra = 2 if cur_parts else 0
        if cur_parts and cur_len + join_extra + len(text) > max_chars:
            _flush()

        if not cur_parts:
            cur_symbol = symbol
            cur_start = start_ln
            cur_end = end_ln
            cur_parts = [text]
            cur_len = len(text)
        else:
            if cur_symbol is None and symbol is not None:
                cur_symbol = symbol
            cur_end = end_ln
            cur_parts.append(text)
            cur_len += 2 + len(text)

    _flush()

    records: list[ChunkRecord] = []
    for idx, (symbol, line_start, line_end, content) in enumerate(merged_pieces):
        header = build_contextual_header(
            project_id=project_id,
            canonical_tier=canonical_tier,
            path_or_uri=path_or_uri,
            updated_at=updated_at,
            symbol_name=symbol,
        )
        records.append(
            ChunkRecord(
                chunk_id=make_chunk_id(doc_id, idx),
                doc_id=doc_id,
                chunk_index=idx,
                contextual_header=header,
                content=content,
                symbol_name=symbol,
                line_start=line_start,
                line_end=line_end,
                token_estimate=estimate_tokens(header + "\n" + content),
            )
        )
    return records


def chunk_markdown_or_text(
    doc_id: str,
    text: str,
    project_id: str | None,
    canonical_tier: RepoTier,
    path_or_uri: str,
    updated_at: str,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[ChunkRecord]:
    """Chunk Markdown, transcripts, PDFs, or plain text by headings and paragraphs."""
    if not text or not text.strip():
        return []

    lines = text.splitlines()
    blocks: list[tuple[str | None, int, int, str]] = []
    current_heading: str | None = None
    para_lines: list[str] = []
    para_start = 1

    def _flush_para(end_line: int) -> None:
        nonlocal para_lines
        if para_lines:
            joined = "\n".join(para_lines).strip()
            if joined:
                blocks.append((current_heading, para_start, end_line, joined))
            para_lines = []

    for i, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if m:
            _flush_para(i - 1)
            current_heading = m.group(2).strip()[:100]
            para_start = i
            para_lines = [line]
            continue

        if not line.strip():
            _flush_para(i - 1)
            para_start = i + 1
            continue

        if not para_lines:
            para_start = i
        para_lines.append(line)

    _flush_para(len(lines))

    return _pack_blocks(
        doc_id=doc_id,
        blocks=blocks,
        project_id=project_id,
        canonical_tier=canonical_tier,
        path_or_uri=path_or_uri,
        updated_at=updated_at,
        max_chars=max_chars,
        overlap=overlap,
    )


def _match_code_symbol(line: str, lang: str | None) -> str | None:
    if not lang:
        return None
    patterns = _BOUNDARY_PATTERNS.get(lang, [])
    for pat in patterns:
        m = pat.match(line)
        if m:
            for group in m.groups():
                if group:
                    return group
    return None


def chunk_code(
    doc_id: str,
    text: str,
    suffix: str,
    project_id: str | None,
    canonical_tier: RepoTier,
    path_or_uri: str,
    updated_at: str,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[ChunkRecord]:
    """Chunk source code along structural function/class boundaries with split-merge."""
    if not text or not text.strip():
        return []

    lang = _LANG_BY_SUFFIX.get(suffix.lower())
    lines = text.splitlines()
    blocks: list[tuple[str | None, int, int, str]] = []
    cur_symbol: str | None = None
    cur_start = 1
    cur_lines: list[str] = []

    for i, line in enumerate(lines, start=1):
        # Only split on top-level or lightly-indented structural definitions (<= 4 spaces)
        # so nested closures don't fragment every single line.
        indent = len(line) - len(line.lstrip(" \t"))
        sym = _match_code_symbol(line, lang) if indent <= 4 else None
        if sym and cur_lines:
            segment = "\n".join(cur_lines).strip()
            if segment:
                blocks.append((cur_symbol, cur_start, i - 1, segment))
            cur_symbol = sym
            cur_start = i
            cur_lines = [line]
        else:
            if sym and not cur_lines:
                cur_symbol = sym
                cur_start = i
            elif not cur_lines:
                cur_start = i
            cur_lines.append(line)

    if cur_lines:
        segment = "\n".join(cur_lines).strip()
        if segment:
            blocks.append((cur_symbol, cur_start, len(lines), segment))

    return _pack_blocks(
        doc_id=doc_id,
        blocks=blocks,
        project_id=project_id,
        canonical_tier=canonical_tier,
        path_or_uri=path_or_uri,
        updated_at=updated_at,
        max_chars=max_chars,
        overlap=overlap,
    )


from chittam.adapters.antigravity import load_antigravity_docs  # noqa: E402
from chittam.adapters.claude_code import load_claude_code_docs  # noqa: E402
from chittam.adapters.codex import load_codex_docs  # noqa: E402
from chittam.adapters.files import MEDIA_HANDLERS, load_file_docs, load_single_file_doc  # noqa: E402


def collect_all_documents(
    cfg: AppConfig,
    atlas: dict[str, CanonicalRepo],
    vision_fn: Callable[[Path, str], str] | None = None,
    image_cache_lookup: Callable[[str], str | None] | None = None,
    existing_stats: dict[str, tuple[int | None, int | None, str, str, str]] | None = None,
    unchanged_doc_ids: set[str] | None = None,
    existing_projects: dict[str, str | None] | None = None,
    unreadable_prefixes: set[str] | None = None,
    failed_harnesses: set[str] | None = None,
) -> tuple[list[DocumentRecord], SyncStats]:
    """Collect normalized DocumentRecords across all harness and filesystem adapters."""
    stats = SyncStats()
    docs: list[DocumentRecord] = []
    enabled = set(cfg.privacy.enabled_media)

    def _collect(name: str, fn: Callable[[], list[DocumentRecord]]) -> None:
        try:
            loaded = fn()
        except Exception as exc:
            stats.errors.append(f"{name}: {exc}")
            if failed_harnesses is not None:
                failed_harnesses.add(name)
            return
        for doc in loaded:
            if doc.media_type in enabled:
                docs.append(doc)

    _collect("antigravity", lambda: load_antigravity_docs(cfg, atlas))
    _collect("codex", lambda: load_codex_docs(cfg, atlas))
    _collect("claude_code", lambda: load_claude_code_docs(cfg, atlas))
    _collect(
        "files",
        lambda: load_file_docs(
            cfg,
            atlas,
            vision_fn=vision_fn,
            image_cache_lookup=image_cache_lookup,
            stats=stats,
            existing_stats=existing_stats,
            unchanged_doc_ids=unchanged_doc_ids,
            existing_projects=existing_projects,
            unreadable_prefixes=unreadable_prefixes,
        ),
    )

    deduped: dict[str, DocumentRecord] = {}
    for doc in docs:
        deduped[doc.doc_id] = doc

    result = list(deduped.values())
    stats.scanned_docs = len(result)
    return result, stats


__all__ = [
    "MEDIA_HANDLERS",
    "build_contextual_header",
    "chunk_code",
    "chunk_markdown_or_text",
    "collect_all_documents",
    "estimate_tokens",
    "load_antigravity_docs",
    "load_claude_code_docs",
    "load_codex_docs",
    "load_file_docs",
    "load_single_file_doc",
    "make_chunk_id",
    "make_doc_id",
    "make_l0_summary",
    "normalize_iso_timestamp",
    "sha256_bytes",
    "sha256_text",
]
