"""Codex SQLite (read-only URI mode) and legacy markdown memories adapter."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from chittam.atlas import match_repo_for_path, match_repo_from_text
from chittam.config import is_path_excluded, redact_secrets
from chittam.models import AppConfig, CanonicalRepo, DocumentRecord, MediaType, RepoTier
from chittam.adapters import (
    chunk_markdown_or_text,
    make_doc_id,
    make_l0_summary,
    normalize_iso_timestamp,
    sha256_text,
)

_INCLUDED_ITEM_TYPES = {
    "userMessage",
    "agentMessage",
    "plan",
    "user",
    "assistant",
    "assistantMessage",
}


def _connect_ro(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists() or not db_path.is_file():
        return None
    try:
        resolved = db_path.resolve(strict=False)
        uri = f"file:{quote(str(resolved))}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(r[1]) for r in rows}
    except sqlite3.Error:
        return set()


def _load_thread_metadata(codex_home: Path, cfg: AppConfig) -> dict[str, dict[str, Any]]:
    state_db = codex_home / "state_5.sqlite"
    excl, _ = is_path_excluded(state_db, cfg.privacy)
    if excl:
        return {}
    conn = _connect_ro(state_db)
    if conn is None:
        return {}
    meta: dict[str, dict[str, Any]] = {}
    try:
        cols = _table_columns(conn, "threads")
        if not cols or "id" not in cols:
            return {}
        wanted = [c for c in ("id", "cwd", "title", "name", "git_branch", "updated_at") if c in cols]
        rows = conn.execute(f"SELECT {', '.join(wanted)} FROM threads").fetchall()
        for row in rows:
            d = dict(row)
            tid = str(d.get("id") or "")
            if tid:
                meta[tid] = d
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return meta


def _extract_codex_item_text(item_type: str, item_json: str) -> tuple[str, str] | None:
    try:
        payload = json.loads(item_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    low_type = item_type.lower()
    if low_type in ("usermessage", "user"):
        role = "User"
    elif low_type == "plan":
        role = "Plan"
    else:
        role = "Assistant"

    parts: list[str] = []
    if isinstance(payload.get("text"), str) and payload["text"].strip():
        parts.append(payload["text"].strip())

    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())
    elif isinstance(content, list):
        for elem in content:
            if isinstance(elem, dict):
                t = elem.get("text") or elem.get("content")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            elif isinstance(elem, str) and elem.strip():
                parts.append(elem.strip())

    joined = "\n".join(parts).strip()
    if not joined:
        return None
    return role, joined


def _load_stage1_memories(
    codex_home: Path,
    cfg: AppConfig,
    atlas: dict[str, CanonicalRepo],
    thread_meta: dict[str, dict[str, Any]],
) -> list[DocumentRecord]:
    mem_db = codex_home / "memories_1.sqlite"
    excl, _ = is_path_excluded(mem_db, cfg.privacy)
    if excl:
        return []
    conn = _connect_ro(mem_db)
    if conn is None:
        return []

    docs: list[DocumentRecord] = []
    try:
        cols = _table_columns(conn, "stage1_outputs")
        if not cols:
            return []
        candidate_cols = [
            "thread_id",
            "rollout_slug",
            "rollout_summary",
            "raw_memory",
            "source_updated_at",
            "generated_at",
            "cwd",
        ]
        select_cols = [c for c in candidate_cols if c in cols]
        if not select_cols:
            return []

        rows = conn.execute(f"SELECT {', '.join(select_cols)} FROM stage1_outputs").fetchall()
        for idx, row in enumerate(rows):
            d = dict(row)
            thread_id = str(d.get("thread_id") or f"row-{idx}")
            slug = str(d.get("rollout_slug") or "").strip()
            summary = str(d.get("rollout_summary") or "").strip()
            raw_mem = str(d.get("raw_memory") or "").strip()
            if not summary and not raw_mem:
                continue

            sections: list[str] = []
            if slug:
                sections.append(f"# {slug}")
            if summary:
                sections.append(f"## Rollout Summary\n{summary}")
            if raw_mem:
                sections.append(f"## Raw Memory\n{raw_mem}")

            text = redact_secrets("\n\n".join(sections))
            tmeta = thread_meta.get(thread_id, {})
            cwd = d.get("cwd") or tmeta.get("cwd")
            repo = match_repo_for_path(cwd, atlas) if cwd else None
            if repo is None:
                repo = match_repo_from_text(text, atlas)
            proj_id = repo.project_id if repo else None
            tier = repo.tier if repo else RepoTier.REFERENCE
            updated_at = normalize_iso_timestamp(
                d.get("source_updated_at") or d.get("generated_at") or tmeta.get("updated_at")
            )
            title = slug or tmeta.get("title") or tmeta.get("name") or f"Codex memory {thread_id[:8]}"
            uri = f"{mem_db}#stage1:{thread_id}"
            doc_id = make_doc_id("codex_mem", uri)
            chunks = chunk_markdown_or_text(
                doc_id=doc_id,
                text=text,
                project_id=proj_id,
                canonical_tier=tier,
                path_or_uri=uri,
                updated_at=updated_at,
            )
            if chunks:
                docs.append(
                    DocumentRecord(
                        doc_id=doc_id,
                        uri=uri,
                        source_harness="codex",
                        media_type=MediaType.TRANSCRIPT,
                        project_id=proj_id,
                        canonical_tier=tier,
                        title=str(title),
                        l0_summary=make_l0_summary(summary or raw_mem),
                        sha256=sha256_text(text),
                        updated_at=updated_at,
                        metadata={
                            "thread_id": thread_id,
                            "rollout_slug": slug,
                            "cwd": cwd,
                        },
                        chunks=chunks,
                    )
                )
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return docs


def _load_thread_history(
    codex_home: Path,
    cfg: AppConfig,
    atlas: dict[str, CanonicalRepo],
    thread_meta: dict[str, dict[str, Any]],
) -> list[DocumentRecord]:
    hist_db = codex_home / "thread_history_1.sqlite"
    excl, _ = is_path_excluded(hist_db, cfg.privacy)
    if excl:
        return []
    conn = _connect_ro(hist_db)
    if conn is None:
        return []

    docs: list[DocumentRecord] = []
    try:
        cols = _table_columns(conn, "thread_items")
        if not {"thread_id", "item_type", "item_json"}.issubset(cols):
            return []

        order_parts = []
        if "rollout_ordinal" in cols:
            order_parts.append("rollout_ordinal ASC")
        if "created_at_ms" in cols:
            order_parts.append("created_at_ms ASC")
        order_clause = f"ORDER BY thread_id ASC, {', '.join(order_parts)}" if order_parts else "ORDER BY rowid ASC"

        select_cols = ["thread_id", "item_type", "item_json"]
        if "created_at_ms" in cols:
            select_cols.append("created_at_ms")

        placeholders = ", ".join("?" for _ in _INCLUDED_ITEM_TYPES)
        query = f"SELECT {', '.join(select_cols)} FROM thread_items WHERE item_type IN ({placeholders}) {order_clause}"
        rows = conn.execute(query, tuple(_INCLUDED_ITEM_TYPES)).fetchall()

        grouped: dict[str, list[tuple[str, str, int | None]]] = {}
        for row in rows:
            d = dict(row)
            tid = str(d.get("thread_id") or "")
            if not tid:
                continue
            parsed = _extract_codex_item_text(str(d["item_type"]), str(d["item_json"]))
            if not parsed:
                continue
            role, text = parsed
            ts_ms = d.get("created_at_ms")
            grouped.setdefault(tid, []).append((role, text, int(ts_ms) if ts_ms else None))

        for thread_id, turns in grouped.items():
            blocks: list[str] = []
            max_ts: int | None = None
            first_user_text: str | None = None
            for role, text, ts_ms in turns:
                if ts_ms is not None and (max_ts is None or ts_ms > max_ts):
                    max_ts = ts_ms
                if role == "User" and first_user_text is None:
                    first_user_text = text
                ts_label = f" ({normalize_iso_timestamp(ts_ms)[:19]})" if ts_ms else ""
                blocks.append(f"## {role}{ts_label}\n{text}")

            full_text = redact_secrets("\n\n".join(blocks))
            tmeta = thread_meta.get(thread_id, {})
            cwd = tmeta.get("cwd")
            repo = match_repo_for_path(cwd, atlas) if cwd else None
            if repo is None:
                repo = match_repo_from_text(full_text, atlas)
            proj_id = repo.project_id if repo else None
            tier = repo.tier if repo else RepoTier.REFERENCE
            updated_at = normalize_iso_timestamp(max_ts or tmeta.get("updated_at"))
            title = (
                tmeta.get("name")
                or tmeta.get("title")
                or (make_l0_summary(first_user_text, 80) if first_user_text else f"Codex thread {thread_id[:8]}")
            )
            uri = f"codex://thread/{thread_id}"
            doc_id = make_doc_id("codex_tx", uri)
            chunks = chunk_markdown_or_text(
                doc_id=doc_id,
                text=full_text,
                project_id=proj_id,
                canonical_tier=tier,
                path_or_uri=uri,
                updated_at=updated_at,
            )
            if chunks:
                docs.append(
                    DocumentRecord(
                        doc_id=doc_id,
                        uri=uri,
                        source_harness="codex",
                        media_type=MediaType.TRANSCRIPT,
                        project_id=proj_id,
                        canonical_tier=tier,
                        title=str(title),
                        l0_summary=make_l0_summary(first_user_text or full_text),
                        sha256=sha256_text(full_text),
                        updated_at=updated_at,
                        metadata={
                            "thread_id": thread_id,
                            "cwd": cwd,
                            "git_branch": tmeta.get("git_branch"),
                            "turn_count": len(turns),
                        },
                        chunks=chunks,
                    )
                )
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return docs


def _load_legacy_memories(cfg: AppConfig, atlas: dict[str, CanonicalRepo]) -> list[DocumentRecord]:
    docs: list[DocumentRecord] = []
    for raw_dir in cfg.harnesses.codex_legacy_memories_dirs:
        mem_dir = Path(raw_dir).expanduser()
        if not mem_dir.exists() or not mem_dir.is_dir():
            continue
        excl_dir, _ = is_path_excluded(mem_dir, cfg.privacy)
        if excl_dir:
            continue

        try:
            md_files = sorted(mem_dir.rglob("*.md"))
        except OSError:
            continue

        for md_path in md_files:
            if not md_path.is_file():
                continue
            excl, _ = is_path_excluded(md_path, cfg.privacy)
            if excl:
                continue
            try:
                raw_text = md_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            redacted = redact_secrets(raw_text).strip()
            if not redacted:
                continue

            repo = match_repo_from_text(redacted, atlas)
            proj_id = repo.project_id if repo else None
            tier = repo.tier if repo else RepoTier.REFERENCE
            try:
                mtime = md_path.stat().st_mtime
            except OSError:
                mtime = None
            updated_at = normalize_iso_timestamp(None, fallback_mtime=mtime)
            doc_id = make_doc_id("codex_legacy", str(md_path))
            chunks = chunk_markdown_or_text(
                doc_id=doc_id,
                text=redacted,
                project_id=proj_id,
                canonical_tier=tier,
                path_or_uri=str(md_path),
                updated_at=updated_at,
            )
            if chunks:
                docs.append(
                    DocumentRecord(
                        doc_id=doc_id,
                        uri=str(md_path),
                        source_harness="codex",
                        media_type=MediaType.MARKDOWN,
                        project_id=proj_id,
                        canonical_tier=tier,
                        title=f"Codex memory: {md_path.stem}",
                        l0_summary=make_l0_summary(redacted),
                        sha256=sha256_text(redacted),
                        updated_at=updated_at,
                        metadata={"legacy_dir": str(mem_dir)},
                        chunks=chunks,
                    )
                )
    return docs


def load_codex_docs(cfg: AppConfig, atlas: dict[str, CanonicalRepo]) -> list[DocumentRecord]:
    """Read Codex SQLite databases (read-only URI mode) and legacy memory markdown files."""
    codex_home = Path(cfg.harnesses.codex_home_dir).expanduser()
    thread_meta = _load_thread_metadata(codex_home, cfg) if codex_home.exists() else {}

    docs: list[DocumentRecord] = []
    if codex_home.exists():
        docs.extend(_load_stage1_memories(codex_home, cfg, atlas, thread_meta))
        docs.extend(_load_thread_history(codex_home, cfg, atlas, thread_meta))
    docs.extend(_load_legacy_memories(cfg, atlas))
    return docs
