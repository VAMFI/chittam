"""SQLite WAL + FTS5 + USearch ANN/quantized vectors + bitemporal decision store for Chittam."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from chittam.models import (
    TIER_WEIGHTS,
    AppConfig,
    ChunkRecord,
    DecisionRecord,
    DocumentRecord,
    MediaType,
    RepoTier,
    SearchResult,
)
from chittam.vector_index import (
    VectorIndex,
    build_vector_index,
    decode_stored_array,
    dequantize_matrix,
    quantize_array,
    validate_vector_dtype,
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MAX_FTS_TOKENS = 24
_MAX_SNIPPET_CHARS = 350
_RRF_K = 20.0
_SCHEMA_VERSION = "2"

_HARNESS_ALIASES: dict[str, tuple[str, ...]] = {
    "antigravity": ("antigravity",),
    "codex": ("codex",),
    "claude": ("claude", "claude_code", "claude-code"),
    "claude_code": ("claude", "claude_code", "claude-code"),
    "claude-code": ("claude", "claude_code", "claude-code"),
}
_DEFAULT_THREAD_HARNESSES = ("antigravity", "codex", "claude", "claude_code", "claude-code")

_REQUIRED_TABLES = {
    "documents",
    "chunks",
    "chunks_fts",
    "chunk_embeddings",
    "content_embeddings",
    "vector_entries",
    "image_cache",
    "decisions",
    "index_meta",
}


def sanitize_fts5_query(query: str) -> tuple[str | None, list[str]]:
    """Convert arbitrary user text into a safe FTS5 MATCH query and token list."""
    raw_tokens = _TOKEN_RE.findall(query or "")
    seen: set[str] = set()
    tokens: list[str] = []
    for tok in raw_tokens:
        cleaned = tok.replace('"', "").strip()
        if not cleaned:
            continue
        low = cleaned.lower()
        if low in seen:
            continue
        seen.add(low)
        tokens.append(cleaned)
        if len(tokens) >= _MAX_FTS_TOKENS:
            break
    if not tokens:
        return None, []
    fts_expr = " OR ".join(f'"{t}"' for t in tokens)
    return fts_expr, tokens


def _content_sha256(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8", errors="replace")).hexdigest()


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _recency_multiplier(updated_at: str | None, now: datetime | None = None) -> float:
    """Return mild recency boost in [1.0, 1.15] (up to +15% for brand new docs)."""
    dt = _parse_iso_dt(updated_at)
    if dt is None:
        return 1.0
    ref = now or datetime.now(timezone.utc)
    age_days = max(0.0, (ref - dt).total_seconds() / 86400.0)
    return 1.0 + 0.15 * math.exp(-age_days / 30.0)


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 0.0 and math.isfinite(norm):
        return (arr / norm).astype(np.float32, copy=False)
    return arr


def _make_snippet(content: str, tokens: list[str], max_len: int = _MAX_SNIPPET_CHARS) -> str:
    """Build a compact <=350-char snippet centered near the first query token match."""
    compact = re.sub(r"\s+", " ", (content or "")).strip()
    if len(compact) <= max_len:
        return compact

    lowered = compact.lower()
    best_idx = -1
    for tok in tokens:
        idx = lowered.find(tok.lower())
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx = idx

    if best_idx == -1 or best_idx < max_len // 2:
        return compact[: max_len - 3].rstrip() + "..."

    start = max(0, best_idx - 80)
    end = start + max_len - 6
    if end >= len(compact):
        start = max(0, len(compact) - (max_len - 3))
        return "..." + compact[start:].lstrip()
    snippet = "..." + compact[start:end].strip() + "..."
    if len(snippet) > max_len:
        return snippet[: max_len - 3].rstrip() + "..."
    return snippet


def _tier_weight(tier_value: str) -> float:
    try:
        tier = RepoTier(tier_value)
    except ValueError:
        tier = RepoTier.REFERENCE
    return TIER_WEIGHTS.get(tier, 0.65)


def decision_embedding_text(dec: DecisionRecord) -> str:
    """Canonical text representation embedded for semantic decision search."""
    return f"[{dec.project_id}] {dec.topic}: {dec.statement}"


class ChittamStore:
    """SQLite WAL hybrid store with FTS5 BM25, USearch/quantized deduped vectors, image cache, and decisions."""

    def __init__(
        self,
        db_path: Path | str | AppConfig,
        *,
        vector_backend: str = "usearch",
        vector_dtype: str = "i8",
        vector_index_path: Path | str | None = None,
    ) -> None:
        cfg_obj = db_path if isinstance(db_path, AppConfig) else None
        resolved = cfg_obj.db_path if cfg_obj is not None else Path(db_path).expanduser()
        self.db_path = resolved
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if cfg_obj is not None:
            self.vector_backend = cfg_obj.vector_backend
            self.vector_dtype = validate_vector_dtype(cfg_obj.local_provider.vector_dtype)
            self.vector_index_path = cfg_obj.vector_index_path
            self._preferred_matryoshka_dim: int | None = cfg_obj.local_provider.matryoshka_dim
        else:
            self.vector_backend = vector_backend
            self.vector_dtype = validate_vector_dtype(vector_dtype)
            self.vector_index_path = (
                Path(vector_index_path).expanduser()
                if vector_index_path is not None
                else (self.db_path.parent / "vectors.usearch")
            )
            self._preferred_matryoshka_dim = None

        self._lock = threading.RLock()
        self._vec_index: VectorIndex | None = None
        self._vec_dirty = False
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self._init_db()

        if cfg_obj is not None and self._needs_v1_migration():
            self.migrate_v1_embeddings(
                target_dim=cfg_obj.local_provider.matryoshka_dim,
                target_dtype=cfg_obj.local_provider.vector_dtype,
                embed_tiers=cfg_obj.privacy.embed_tiers,
                embed_always_media=cfg_obj.privacy.embed_always_media,
            )

    def __enter__(self) -> ChittamStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._flush_vec_index()
            self.conn.close()

    def _init_db(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA busy_timeout=30000;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.execute("PRAGMA temp_store=MEMORY;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")

            jm_row = self.conn.execute("PRAGMA journal_mode;").fetchone()
            current_jm = str(jm_row[0]).lower() if jm_row else ""
            if current_jm != "wal":
                self.conn.execute("PRAGMA journal_mode=WAL;")

            existing_tables = {
                row[0]
                for row in self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            doc_cols = (
                {r[1] for r in self.conn.execute("PRAGMA table_info(documents)").fetchall()}
                if "documents" in existing_tables
                else set()
            )
            chunk_cols = (
                {r[1] for r in self.conn.execute("PRAGMA table_info(chunks)").fetchall()}
                if "chunks" in existing_tables
                else set()
            )
            if (
                _REQUIRED_TABLES.issubset(existing_tables)
                and {"file_mtime_ns", "file_size"}.issubset(doc_cols)
                and "content_sha256" in chunk_cols
            ):
                return

            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    uri TEXT NOT NULL,
                    source_harness TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    project_id TEXT,
                    canonical_tier TEXT NOT NULL,
                    title TEXT NOT NULL,
                    l0_summary TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    file_mtime_ns INTEGER,
                    file_size INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
                CREATE INDEX IF NOT EXISTS idx_documents_media ON documents(media_type);
                CREATE INDEX IF NOT EXISTS idx_documents_harness ON documents(source_harness);
                CREATE INDEX IF NOT EXISTS idx_documents_tier ON documents(canonical_tier);

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    contextual_header TEXT NOT NULL,
                    content TEXT NOT NULL,
                    symbol_name TEXT,
                    line_start INTEGER,
                    line_end INTEGER,
                    token_estimate INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    content_sha256 TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_idx ON chunks(doc_id, chunk_index);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    contextual_header,
                    content,
                    title,
                    project_id,
                    source_harness,
                    tokenize='porter unicode61'
                );

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_embeddings_model_dim ON chunk_embeddings(model, dim);

                CREATE TABLE IF NOT EXISTS content_embeddings (
                    content_sha256 TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    dtype TEXT NOT NULL DEFAULT 'i8',
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (content_sha256, model, dim)
                );

                CREATE TABLE IF NOT EXISTS vector_entries (
                    vec_key INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id TEXT NOT NULL UNIQUE,
                    target_type TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_vector_entries_model_dim
                    ON vector_entries(model, dim);
                CREATE INDEX IF NOT EXISTS idx_vector_entries_sha
                    ON vector_entries(content_sha256);
                CREATE INDEX IF NOT EXISTS idx_vector_entries_target_type
                    ON vector_entries(target_type);

                CREATE TABLE IF NOT EXISTS image_cache (
                    sha256 TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    source_harness TEXT NOT NULL DEFAULT 'manual',
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    superseded_by TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_decisions_project_topic
                    ON decisions(project_id, topic, valid_to);

                CREATE TABLE IF NOT EXISTS index_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

            doc_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(documents)").fetchall()}
            if "file_mtime_ns" not in doc_cols:
                self.conn.execute("ALTER TABLE documents ADD COLUMN file_mtime_ns INTEGER;")
            if "file_size" not in doc_cols:
                self.conn.execute("ALTER TABLE documents ADD COLUMN file_size INTEGER;")

            chunk_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "content_sha256" not in chunk_cols:
                self.conn.execute("ALTER TABLE chunks ADD COLUMN content_sha256 TEXT;")

            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_content_sha ON chunks(content_sha256);"
            )
            self.conn.commit()

    def _needs_v1_migration(self) -> bool:
        with self._lock:
            legacy_row = self.conn.execute("SELECT 1 FROM chunk_embeddings LIMIT 1").fetchone()
            if legacy_row is None:
                return False
            v2_row = self.conn.execute("SELECT 1 FROM vector_entries LIMIT 1").fetchone()
            return v2_row is None

    def get_index_meta(self) -> dict[str, str]:
        """Return index metadata key-value map (active embedding model, dim, dtype, source)."""
        with self._lock:
            rows = self.conn.execute("SELECT key, value FROM index_meta").fetchall()
            return {str(r["key"]): str(r["value"]) for r in rows}

    def _set_index_meta_unlocked(self, pairs: dict[str, str]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            [(k, str(v)) for k, v in pairs.items()],
        )

    def _flush_vec_index(self) -> None:
        if self._vec_dirty and self._vec_index is not None:
            self._vec_index.save()
            self._vec_dirty = False

    def flush_vector_index(self) -> None:
        """Persist any pending vector index updates atomically to disk."""
        with self._lock:
            self._flush_vec_index()

    def get_doc_ids_by_uri(self, uri: str) -> list[str]:
        """Return all indexed doc_ids matching the given file/document URI."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT doc_id FROM documents WHERE uri = ?",
                (str(uri),),
            ).fetchall()
            return [str(r["doc_id"]) for r in rows]

    def _ensure_vec_index_for_write(self, dim: int, dtype: str) -> VectorIndex:
        dt = validate_vector_dtype(dtype)
        with self._lock:
            if (
                self._vec_index is None
                or self._vec_index.dim != dim
                or self._vec_index.dtype != dt
            ):
                self._vec_index = build_vector_index(
                    self.vector_index_path,
                    dim=dim,
                    dtype=dt,
                    backend=self.vector_backend,
                )
            elif self.vector_backend == "usearch":
                self._vec_index.reload_if_changed()
            return self._vec_index

    def _get_vec_index_for_search(self, expected_dim: int) -> VectorIndex | None:
        with self._lock:
            meta = self.get_index_meta()
            meta_dtype = meta.get("vector_dtype", self.vector_dtype)
            try:
                dt = validate_vector_dtype(meta_dtype)
            except ValueError:
                dt = self.vector_dtype

            if self.vector_backend == "numpy":
                if (
                    self._vec_index is None
                    or self._vec_index.dim != expected_dim
                    or self._vec_index.dtype != dt
                ):
                    idx = build_vector_index(None, dim=expected_dim, dtype=dt, backend="numpy")
                    self._populate_index_from_db_unlocked(idx, dim=expected_dim, dtype=dt)
                    self._vec_index = idx
                return self._vec_index

            if not self.vector_index_path.exists():
                return None

            if (
                self._vec_index is None
                or self._vec_index.dim != expected_dim
                or self._vec_index.dtype != dt
            ):
                self._vec_index = build_vector_index(
                    self.vector_index_path,
                    dim=expected_dim,
                    dtype=dt,
                    backend=self.vector_backend,
                )
            else:
                self._vec_index.reload_if_changed()

            if self._vec_index.dim != expected_dim:
                return None
            return self._vec_index

    def _populate_index_from_db_unlocked(self, idx: VectorIndex, dim: int, dtype: str) -> int:
        rows = self.conn.execute(
            """
            SELECT ve.vec_key, ce.vector
            FROM vector_entries ve
            JOIN content_embeddings ce
              ON ce.content_sha256 = ve.content_sha256
             AND ce.model = ve.model
             AND ce.dim = ve.dim
            WHERE ve.dim = ? AND ce.dtype = ?
            """,
            (dim, dtype),
        ).fetchall()
        if not rows:
            return 0
        batch_size = 4096
        added = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            keys = np.array([int(r["vec_key"]) for r in batch], dtype=np.uint64)
            mat = decode_stored_array([bytes(r["vector"]) for r in batch], dim=dim, dtype=dtype)
            idx.add(keys, mat)
            added += len(batch)
        return added

    def get_document_manifest(self) -> dict[str, tuple[str, str]]:
        """Return {doc_id: (sha256, canonical_tier)} for Merkle incremental diffing."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT doc_id, sha256, canonical_tier FROM documents"
            ).fetchall()
            return {row["doc_id"]: (row["sha256"], row["canonical_tier"]) for row in rows}

    def get_document_fingerprints(self) -> dict[str, tuple[str, str, str]]:
        """Return {doc_id: (sha256, canonical_tier, l0_summary)} for Merkle diffing."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT doc_id, sha256, canonical_tier, l0_summary FROM documents"
            ).fetchall()
            return {
                row["doc_id"]: (row["sha256"], row["canonical_tier"], row["l0_summary"])
                for row in rows
            }

    def get_document_stat_index(self) -> dict[str, tuple[int | None, int | None, str, str, str]]:
        """Return {doc_id: (file_mtime_ns, file_size, sha256, canonical_tier, l0_summary)}."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT doc_id, file_mtime_ns, file_size, sha256, canonical_tier, l0_summary FROM documents"
            ).fetchall()
            return {
                row["doc_id"]: (
                    int(row["file_mtime_ns"]) if row["file_mtime_ns"] is not None else None,
                    int(row["file_size"]) if row["file_size"] is not None else None,
                    row["sha256"],
                    row["canonical_tier"],
                    row["l0_summary"],
                )
                for row in rows
            }

    def update_document_file_stat(
        self,
        doc_id: str,
        file_mtime_ns: int | None,
        file_size: int | None,
    ) -> None:
        """Backfill or refresh file stat metadata without rewriting document chunks."""
        if file_mtime_ns is None or file_size is None:
            return
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE documents SET file_mtime_ns = ?, file_size = ? WHERE doc_id = ?",
                (int(file_mtime_ns), int(file_size), doc_id),
            )

    def has_embeddings(self) -> bool:
        """Return True if at least one vector entry or legacy embedding is stored."""
        with self._lock:
            row = self.conn.execute("SELECT 1 FROM vector_entries LIMIT 1").fetchone()
            if row is not None:
                return True
            legacy = self.conn.execute("SELECT 1 FROM chunk_embeddings LIMIT 1").fetchone()
            return legacy is not None

    def _remove_vec_keys_unlocked(self, vec_keys: list[int], save_index: bool) -> None:
        if not vec_keys:
            return
        meta = self.get_index_meta()
        dim_raw = meta.get("active_embedding_dim")
        dt = meta.get("vector_dtype", self.vector_dtype)
        idx = self._vec_index
        if idx is None and self.vector_backend == "usearch" and self.vector_index_path.exists() and dim_raw:
            try:
                idx = build_vector_index(
                    self.vector_index_path,
                    dim=int(dim_raw),
                    dtype=dt,
                    backend="usearch",
                )
                self._vec_index = idx
            except Exception:
                idx = None
        if idx is not None:
            idx.remove(vec_keys)
            if save_index:
                idx.save()
                self._vec_dirty = False
            else:
                self._vec_dirty = True

    def upsert_document(self, doc: DocumentRecord, *, _save_vec: bool = True) -> None:
        """Atomically insert or replace a document, its chunks, FTS5 rows, and vector entries."""
        media_val = doc.media_type.value if isinstance(doc.media_type, MediaType) else str(doc.media_type)
        tier_val = doc.canonical_tier.value if isinstance(doc.canonical_tier, RepoTier) else str(doc.canonical_tier)
        doc_meta_dict = dict(doc.metadata)
        mtime_ns_raw = doc_meta_dict.get("file_mtime_ns")
        file_size_raw = doc_meta_dict.get("file_size")
        mtime_ns = int(mtime_ns_raw) if mtime_ns_raw is not None else None
        file_size = int(file_size_raw) if file_size_raw is not None else None
        doc_meta = json.dumps(doc_meta_dict, ensure_ascii=False)

        removed_keys: list[int] = []
        with self._lock, self.conn:
            exists = (
                self.conn.execute(
                    "SELECT 1 FROM documents WHERE doc_id = ? LIMIT 1",
                    (doc.doc_id,),
                ).fetchone()
                is not None
            )
            if exists:
                old_key_rows = self.conn.execute(
                    """
                    SELECT vec_key FROM vector_entries
                    WHERE target_type = 'chunk'
                      AND target_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)
                    """,
                    (doc.doc_id,),
                ).fetchall()
                removed_keys = [int(r["vec_key"]) for r in old_key_rows]
                if removed_keys:
                    self.conn.execute(
                        """
                        DELETE FROM vector_entries
                        WHERE target_type = 'chunk'
                          AND target_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)
                        """,
                        (doc.doc_id,),
                    )
                self.conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc.doc_id,))
                self.conn.execute(
                    "DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                    (doc.doc_id,),
                )
                self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.doc_id,))

            self.conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    doc_id, uri, source_harness, media_type, project_id,
                    canonical_tier, title, l0_summary, sha256, updated_at, metadata_json,
                    file_mtime_ns, file_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    doc.uri,
                    doc.source_harness,
                    media_val,
                    doc.project_id,
                    tier_val,
                    doc.title,
                    doc.l0_summary,
                    doc.sha256,
                    doc.updated_at,
                    doc_meta,
                    mtime_ns,
                    file_size,
                ),
            )

            if doc.chunks:
                chunk_rows = [
                    (
                        c.chunk_id,
                        doc.doc_id,
                        c.chunk_index,
                        c.contextual_header,
                        c.content,
                        c.symbol_name,
                        c.line_start,
                        c.line_end,
                        c.token_estimate,
                        json.dumps(c.metadata, ensure_ascii=False),
                        _content_sha256(c.content),
                    )
                    for c in doc.chunks
                ]
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, doc_id, chunk_index, contextual_header, content,
                        symbol_name, line_start, line_end, token_estimate, metadata_json, content_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    chunk_rows,
                )

                fts_rows = [
                    (
                        c.chunk_id,
                        doc.doc_id,
                        c.contextual_header,
                        c.content,
                        doc.title,
                        doc.project_id or "",
                        doc.source_harness,
                    )
                    for c in doc.chunks
                ]
                self.conn.executemany(
                    """
                    INSERT INTO chunks_fts (
                        chunk_id, doc_id, contextual_header, content, title, project_id, source_harness
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    fts_rows,
                )

            if removed_keys:
                self._remove_vec_keys_unlocked(removed_keys, save_index=_save_vec)

    def delete_document(self, doc_id: str, *, _save_vec: bool = True) -> bool:
        """Delete a document and its synced chunks, vector entries, and FTS5 rows."""
        with self._lock, self.conn:
            old_key_rows = self.conn.execute(
                """
                SELECT vec_key FROM vector_entries
                WHERE target_type = 'chunk'
                  AND target_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)
                """,
                (doc_id,),
            ).fetchall()
            removed_keys = [int(r["vec_key"]) for r in old_key_rows]
            if removed_keys:
                self.conn.execute(
                    """
                    DELETE FROM vector_entries
                    WHERE target_type = 'chunk'
                      AND target_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)
                    """,
                    (doc_id,),
                )
            self.conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            self.conn.execute(
                "DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cur = self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            if removed_keys:
                self._remove_vec_keys_unlocked(removed_keys, save_index=_save_vec)
            return cur.rowcount > 0

    def prune_documents(
        self,
        keep_doc_ids: set[str],
        preserve_media_types: set[str] | None = None,
    ) -> int:
        """Remove documents not in keep_doc_ids (optionally preserving certain media types)."""
        preserved = preserve_media_types or set()
        with self._lock:
            rows = self.conn.execute("SELECT doc_id, media_type FROM documents").fetchall()
            to_delete = [
                row["doc_id"]
                for row in rows
                if row["doc_id"] not in keep_doc_ids and row["media_type"] not in preserved
            ]
        deleted = 0
        for doc_id in to_delete:
            if self.delete_document(doc_id, _save_vec=False):
                deleted += 1
        with self._lock:
            self._flush_vec_index()
        return deleted

    def _build_eligibility_clause(
        self,
        embed_tiers: list[RepoTier] | None,
        embed_always_media: list[MediaType] | None,
    ) -> tuple[str, list[Any]]:
        if embed_tiers is None and embed_always_media is None:
            return "", []
        parts: list[str] = []
        params: list[Any] = []
        if embed_tiers:
            t_vals = [t.value if isinstance(t, RepoTier) else str(t) for t in embed_tiers]
            ph = ", ".join("?" for _ in t_vals)
            parts.append(f"d.canonical_tier IN ({ph})")
            params.extend(t_vals)
        if embed_always_media:
            m_vals = [m.value if isinstance(m, MediaType) else str(m) for m in embed_always_media]
            ph = ", ".join("?" for _ in m_vals)
            parts.append(f"d.media_type IN ({ph})")
            params.extend(m_vals)
        if not parts:
            return "AND 1 = 0", []
        return f"AND ({' OR '.join(parts)})", params

    def link_deduped_embeddings(
        self,
        model: str,
        dim: int,
        embed_tiers: list[RepoTier] | None = None,
        embed_always_media: list[MediaType] | None = None,
        dtype: str | None = None,
        source: str = "local",
    ) -> int:
        """Reuse existing content_embeddings for chunks with matching content_sha256 without calling Ollama."""
        dt = validate_vector_dtype(dtype or self.vector_dtype)
        elig_sql, elig_params = self._build_eligibility_clause(embed_tiers, embed_always_media)

        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT c.chunk_id, c.content, c.content_sha256,
                       ve.vec_key AS old_vec_key,
                       ce.vector AS vector_blob
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                JOIN content_embeddings ce
                  ON ce.content_sha256 = c.content_sha256
                 AND ce.model = ?
                 AND ce.dim = ?
                 AND ce.dtype = ?
                LEFT JOIN vector_entries ve
                  ON ve.target_id = c.chunk_id
                 AND ve.target_type = 'chunk'
                WHERE (ve.target_id IS NULL OR ve.model != ? OR ve.dim != ?)
                  {elig_sql}
                """,
                (model, int(dim), dt, model, int(dim), *elig_params),
            ).fetchall()

        if not rows:
            return 0

        keys: list[int] = []
        blobs: list[bytes] = []
        with self._lock, self.conn:
            for row in rows:
                cid = str(row["chunk_id"])
                c_sha = str(row["content_sha256"] or _content_sha256(str(row["content"])))
                old_key = row["old_vec_key"]
                if old_key is not None:
                    self.conn.execute(
                        """
                        UPDATE vector_entries
                        SET target_type = 'chunk', content_sha256 = ?, model = ?, dim = ?
                        WHERE vec_key = ?
                        """,
                        (c_sha, model, int(dim), int(old_key)),
                    )
                    vec_key = int(old_key)
                else:
                    cur = self.conn.execute(
                        """
                        INSERT INTO vector_entries (target_id, target_type, content_sha256, model, dim)
                        VALUES (?, 'chunk', ?, ?, ?)
                        """,
                        (cid, c_sha, model, int(dim)),
                    )
                    vec_key = int(cur.lastrowid)
                keys.append(vec_key)
                blobs.append(bytes(row["vector_blob"]))

            self._set_index_meta_unlocked(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "active_embedding_model": model,
                    "active_embedding_dim": str(int(dim)),
                    "active_embedding_source": source,
                    "vector_dtype": dt,
                }
            )

        idx = self._ensure_vec_index_for_write(dim=int(dim), dtype=dt)
        mat = decode_stored_array(blobs, dim=int(dim), dtype=dt)
        idx.add(keys, mat)
        idx.save()
        return len(keys)

    def get_chunks_missing_embeddings(
        self,
        model: str | None = None,
        limit: int | None = None,
        embed_tiers: list[RepoTier] | None = None,
        embed_always_media: list[MediaType] | None = None,
        dim: int | None = None,
    ) -> list[ChunkRecord]:
        """Return chunks lacking an entry for (model, dim), filtered by tier/modality and prioritized."""
        limit_sql = "LIMIT ?" if limit is not None and limit > 0 else ""
        elig_sql, elig_params = self._build_eligibility_clause(embed_tiers, embed_always_media)
        order_sql = """
            ORDER BY
                CASE d.canonical_tier WHEN 'active' THEN 0 WHEN 'reference' THEN 1 ELSE 2 END,
                CASE d.media_type
                    WHEN 'image' THEN 0
                    WHEN 'transcript' THEN 1
                    WHEN 'pdf' THEN 2
                    WHEN 'markdown' THEN 3
                    WHEN 'code' THEN 4
                    ELSE 5
                END,
                d.updated_at DESC,
                c.doc_id,
                c.chunk_index
        """
        with self._lock:
            params: list[Any] = []
            if model and dim is not None:
                where_sql = "WHERE (ve.target_id IS NULL OR ve.model != ? OR ve.dim != ?)"
                params.extend([model, int(dim)])
            elif model:
                where_sql = "WHERE (ve.target_id IS NULL OR ve.model != ?)"
                params.append(model)
            else:
                where_sql = "WHERE ve.target_id IS NULL"

            params.extend(elig_params)
            if limit_sql:
                params.append(int(limit))

            rows = self.conn.execute(
                f"""
                SELECT c.*
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                LEFT JOIN vector_entries ve
                  ON ve.target_id = c.chunk_id
                 AND ve.target_type = 'chunk'
                {where_sql}
                {elig_sql}
                {order_sql}
                {limit_sql}
                """,
                params,
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def upsert_chunk_embeddings(
        self,
        chunk_ids: list[str],
        model: str,
        vectors: np.ndarray,
        *,
        dtype: str | None = None,
        source: str = "local",
    ) -> int:
        """Store quantized deduped embeddings for chunks in SQLite and update USearch index."""
        if not chunk_ids:
            return 0
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        if len(chunk_ids) != mat.shape[0]:
            raise ValueError(
                f"chunk_ids count ({len(chunk_ids)}) != embedding rows ({mat.shape[0]})"
            )
        dim = int(mat.shape[1])
        dt = validate_vector_dtype(dtype or self.vector_dtype)
        quant_mat = quantize_array(mat, dtype=dt)
        now = datetime.now(timezone.utc).isoformat()

        ph = ", ".join("?" for _ in chunk_ids)
        with self._lock:
            chunk_rows = self.conn.execute(
                f"SELECT chunk_id, content, content_sha256 FROM chunks WHERE chunk_id IN ({ph})",
                chunk_ids,
            ).fetchall()
            chunk_map = {str(r["chunk_id"]): r for r in chunk_rows}
            existing_entry_rows = self.conn.execute(
                f"""
                SELECT target_id, vec_key FROM vector_entries
                WHERE target_type = 'chunk' AND target_id IN ({ph})
                """,
                chunk_ids,
            ).fetchall()
            existing_keys = {str(r["target_id"]): int(r["vec_key"]) for r in existing_entry_rows}

        if not chunk_map:
            return 0

        keys_to_add: list[int] = []
        vecs_to_add: list[np.ndarray] = []
        content_rows: list[tuple[str, str, int, str, bytes, str]] = []
        seen_shas: set[str] = set()
        sha_updates: list[tuple[str, str]] = []

        with self._lock, self.conn:
            for pos, cid in enumerate(chunk_ids):
                crow = chunk_map.get(cid)
                if crow is None:
                    continue
                c_sha = crow["content_sha256"]
                if not c_sha:
                    c_sha = _content_sha256(str(crow["content"]))
                    sha_updates.append((c_sha, cid))
                else:
                    c_sha = str(c_sha)

                q_row = quant_mat[pos]
                if c_sha not in seen_shas:
                    seen_shas.add(c_sha)
                    content_rows.append((c_sha, model, dim, dt, q_row.tobytes(), now))

                if cid in existing_keys:
                    vec_key = existing_keys[cid]
                    self.conn.execute(
                        """
                        UPDATE vector_entries
                        SET target_type = 'chunk', content_sha256 = ?, model = ?, dim = ?
                        WHERE vec_key = ?
                        """,
                        (c_sha, model, dim, vec_key),
                    )
                else:
                    cur = self.conn.execute(
                        """
                        INSERT INTO vector_entries (target_id, target_type, content_sha256, model, dim)
                        VALUES (?, 'chunk', ?, ?, ?)
                        """,
                        (cid, c_sha, model, dim),
                    )
                    vec_key = int(cur.lastrowid)

                keys_to_add.append(vec_key)
                vecs_to_add.append(q_row)

            if sha_updates:
                self.conn.executemany(
                    "UPDATE chunks SET content_sha256 = ? WHERE chunk_id = ?",
                    sha_updates,
                )
            if content_rows:
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO content_embeddings (
                        content_sha256, model, dim, dtype, vector, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    content_rows,
                )
            self._set_index_meta_unlocked(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "active_embedding_model": model,
                    "active_embedding_dim": str(dim),
                    "active_embedding_source": source,
                    "vector_dtype": dt,
                }
            )

        if keys_to_add:
            idx = self._ensure_vec_index_for_write(dim=dim, dtype=dt)
            idx.add(keys_to_add, np.vstack(vecs_to_add))
            idx.save()
        return len(keys_to_add)

    def upsert_decision_embedding(
        self,
        decision_id: str,
        text: str,
        model: str,
        vector: np.ndarray,
        *,
        dtype: str | None = None,
        source: str = "local",
    ) -> None:
        """Store a quantized semantic embedding for a bitemporal decision."""
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        dim = int(vec.shape[1])
        dt = validate_vector_dtype(dtype or self.vector_dtype)
        q_arr = quantize_array(vec, dtype=dt)
        c_sha = _content_sha256(text)
        now = datetime.now(timezone.utc).isoformat()

        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO content_embeddings (
                    content_sha256, model, dim, dtype, vector, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (c_sha, model, dim, dt, q_arr[0].tobytes(), now),
            )
            existing = self.conn.execute(
                """
                SELECT vec_key FROM vector_entries
                WHERE target_type = 'decision' AND target_id = ?
                """,
                (decision_id,),
            ).fetchone()
            if existing is not None:
                vec_key = int(existing["vec_key"])
                self.conn.execute(
                    """
                    UPDATE vector_entries
                    SET content_sha256 = ?, model = ?, dim = ?
                    WHERE vec_key = ?
                    """,
                    (c_sha, model, dim, vec_key),
                )
            else:
                cur = self.conn.execute(
                    """
                    INSERT INTO vector_entries (target_id, target_type, content_sha256, model, dim)
                    VALUES (?, 'decision', ?, ?, ?)
                    """,
                    (decision_id, c_sha, model, dim),
                )
                vec_key = int(cur.lastrowid)

            self._set_index_meta_unlocked(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "active_embedding_model": model,
                    "active_embedding_dim": str(dim),
                    "active_embedding_source": source,
                    "vector_dtype": dt,
                }
            )

        idx = self._ensure_vec_index_for_write(dim=dim, dtype=dt)
        idx.add([vec_key], q_arr)
        idx.save()

    def get_decisions_missing_embeddings(
        self,
        model: str,
        dim: int | None = None,
        active_only: bool = False,
    ) -> list[DecisionRecord]:
        """Return decisions lacking an embedding for (model, dim)."""
        clauses = ["(ve.target_id IS NULL OR ve.model != ?)"]
        params: list[Any] = [model]
        if dim is not None:
            clauses[0] = "(ve.target_id IS NULL OR ve.model != ? OR ve.dim != ?)"
            params.append(int(dim))
        if active_only:
            clauses.append("dec.valid_to IS NULL")
        where_sql = " AND ".join(clauses)

        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT dec.*
                FROM decisions dec
                LEFT JOIN vector_entries ve
                  ON ve.target_id = dec.decision_id
                 AND ve.target_type = 'decision'
                WHERE {where_sql}
                ORDER BY dec.created_at DESC
                """,
                params,
            ).fetchall()
        return [self._row_to_decision(r) for r in rows]

    def migrate_v1_embeddings(
        self,
        target_dim: int | None = None,
        target_dtype: str = "i8",
        embed_tiers: list[RepoTier] | None = None,
        embed_always_media: list[MediaType] | None = None,
        source: str = "local",
    ) -> dict[str, Any]:
        """Migrate legacy v0.1 768-d float32 chunk_embeddings in-place to deduped Matryoshka/quantized v2."""
        dt = validate_vector_dtype(target_dtype)
        with self._lock:
            legacy_count = int(self.conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0])
            if legacy_count == 0:
                return {"migrated": 0, "deduped": 0, "skipped_tier": 0, "dim": target_dim, "dtype": dt}

            rows = self.conn.execute(
                """
                SELECT ce.chunk_id, ce.model, ce.dim, ce.vector,
                       c.content, c.content_sha256,
                       d.canonical_tier, d.media_type
                FROM chunk_embeddings ce
                JOIN chunks c ON c.chunk_id = ce.chunk_id
                JOIN documents d ON d.doc_id = c.doc_id
                """
            ).fetchall()

        if not rows:
            return {"migrated": 0, "deduped": 0, "skipped_tier": 0, "dim": target_dim, "dtype": dt}

        allowed_tiers = (
            {t.value if isinstance(t, RepoTier) else str(t) for t in embed_tiers}
            if embed_tiers is not None
            else None
        )
        allowed_media = (
            {m.value if isinstance(m, MediaType) else str(m) for m in embed_always_media}
            if embed_always_media is not None
            else None
        )

        now = datetime.now(timezone.utc).isoformat()
        migrated_entries: list[tuple[str, str, str, int, np.ndarray]] = []
        content_rows_map: dict[tuple[str, str, int], tuple[str, str, int, str, bytes, str]] = {}
        sha_updates: list[tuple[str, str]] = []
        skipped_tier = 0
        final_model = str(rows[0]["model"])
        final_dim: int | None = None

        for row in rows:
            src_dim = int(row["dim"])
            raw_bytes = bytes(row["vector"])
            if len(raw_bytes) != src_dim * 4:
                continue
            tier_val = str(row["canonical_tier"])
            media_val = str(row["media_type"])
            if allowed_tiers is not None or allowed_media is not None:
                in_tier = allowed_tiers is not None and tier_val in allowed_tiers
                in_media = allowed_media is not None and media_val in allowed_media
                if not (in_tier or in_media):
                    skipped_tier += 1
                    continue

            vec = np.frombuffer(raw_bytes, dtype=np.float32)
            out_dim = int(target_dim) if (target_dim is not None and 0 < int(target_dim) <= src_dim) else src_dim
            sliced = vec[:out_dim]
            q_vec = quantize_array(sliced, dtype=dt).reshape(-1)
            model_name = str(row["model"])
            final_model = model_name
            final_dim = out_dim

            c_sha = row["content_sha256"]
            if not c_sha:
                c_sha = _content_sha256(str(row["content"]))
                sha_updates.append((c_sha, str(row["chunk_id"])))
            else:
                c_sha = str(c_sha)

            ce_key = (c_sha, model_name, out_dim)
            if ce_key not in content_rows_map:
                content_rows_map[ce_key] = (c_sha, model_name, out_dim, dt, q_vec.tobytes(), now)
            migrated_entries.append((str(row["chunk_id"]), c_sha, model_name, out_dim, q_vec))

        if final_dim is None or not migrated_entries:
            with self._lock, self.conn:
                self.conn.execute("DELETE FROM chunk_embeddings")
            return {
                "migrated": 0,
                "deduped": 0,
                "skipped_tier": skipped_tier,
                "dim": target_dim,
                "dtype": dt,
            }

        keys_to_add: list[int] = []
        vecs_to_add: list[np.ndarray] = []
        with self._lock, self.conn:
            if sha_updates:
                self.conn.executemany(
                    "UPDATE chunks SET content_sha256 = ? WHERE chunk_id = ?",
                    sha_updates,
                )
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO content_embeddings (
                    content_sha256, model, dim, dtype, vector, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                list(content_rows_map.values()),
            )
            for cid, c_sha, model_name, out_dim, q_vec in migrated_entries:
                existing = self.conn.execute(
                    "SELECT vec_key FROM vector_entries WHERE target_id = ?",
                    (cid,),
                ).fetchone()
                if existing is not None:
                    vec_key = int(existing["vec_key"])
                    self.conn.execute(
                        """
                        UPDATE vector_entries
                        SET target_type = 'chunk', content_sha256 = ?, model = ?, dim = ?
                        WHERE vec_key = ?
                        """,
                        (c_sha, model_name, out_dim, vec_key),
                    )
                else:
                    cur = self.conn.execute(
                        """
                        INSERT INTO vector_entries (target_id, target_type, content_sha256, model, dim)
                        VALUES (?, 'chunk', ?, ?, ?)
                        """,
                        (cid, c_sha, model_name, out_dim),
                    )
                    vec_key = int(cur.lastrowid)
                keys_to_add.append(vec_key)
                vecs_to_add.append(q_vec)

            self._set_index_meta_unlocked(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "active_embedding_model": final_model,
                    "active_embedding_dim": str(final_dim),
                    "active_embedding_source": source,
                    "vector_dtype": dt,
                }
            )

        idx = build_vector_index(
            self.vector_index_path,
            dim=final_dim,
            dtype=dt,
            backend=self.vector_backend,
        )
        idx.clear()
        batch_size = 4096
        for start in range(0, len(keys_to_add), batch_size):
            idx.add(
                keys_to_add[start : start + batch_size],
                np.vstack(vecs_to_add[start : start + batch_size]),
            )
        idx.save()
        with self._lock, self.conn:
            self._vec_index = idx
            self._vec_dirty = False
            self.conn.execute("DELETE FROM chunk_embeddings")

        deduped = max(0, len(migrated_entries) - len(content_rows_map))
        return {
            "migrated": len(migrated_entries),
            "unique_embeddings": len(content_rows_map),
            "deduped": deduped,
            "skipped_tier": skipped_tier,
            "dim": final_dim,
            "dtype": dt,
        }

    def rebuild_vector_index(self) -> dict[str, Any]:
        """Rebuild vectors.usearch completely offline from SQLite vector_entries and content_embeddings."""
        with self._lock:
            meta = self.get_index_meta()
            dim_raw = meta.get("active_embedding_dim")
            dt = meta.get("vector_dtype", self.vector_dtype)
            if not dim_raw:
                row = self.conn.execute("SELECT dim FROM vector_entries LIMIT 1").fetchone()
                if row is None:
                    if self.vector_index_path.exists():
                        self.vector_index_path.unlink()
                    self._vec_index = None
                    return {"rebuilt": 0, "dim": None, "dtype": dt, "vector_index_bytes": 0}
                dim_raw = str(row["dim"])
            dim = int(dim_raw)

            idx = build_vector_index(
                self.vector_index_path,
                dim=dim,
                dtype=dt,
                backend=self.vector_backend,
            )
            idx.clear()
            rebuilt = self._populate_index_from_db_unlocked(idx, dim=dim, dtype=dt)
            idx.save()
            self._vec_index = idx
            self._vec_dirty = False

        vec_bytes = self.vector_index_path.stat().st_size if self.vector_index_path.exists() else 0
        return {
            "rebuilt": rebuilt,
            "dim": dim,
            "dtype": dt,
            "vector_backend": idx.backend_name,
            "vector_index_bytes": vec_bytes,
        }

    def compact(self) -> dict[str, Any]:
        """Prune orphaned content_embeddings, rebuild/compact USearch, checkpoint WAL, and VACUUM."""
        db_before = self.db_path.stat().st_size if self.db_path.exists() else 0
        vec_before = self.vector_index_path.stat().st_size if self.vector_index_path.exists() else 0

        with self._lock:
            with self.conn:
                cur = self.conn.execute(
                    """
                    DELETE FROM content_embeddings
                    WHERE NOT EXISTS (
                        SELECT 1 FROM vector_entries ve
                        WHERE ve.content_sha256 = content_embeddings.content_sha256
                          AND ve.model = content_embeddings.model
                          AND ve.dim = content_embeddings.dim
                    )
                    """
                )
                orphans_deleted = int(cur.rowcount)

            rebuild_info = self.rebuild_vector_index()
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            self.conn.execute("VACUUM;")
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

        db_after = self.db_path.stat().st_size if self.db_path.exists() else 0
        vec_after = self.vector_index_path.stat().st_size if self.vector_index_path.exists() else 0
        return {
            "orphans_deleted": orphans_deleted,
            "rebuilt_vectors": rebuild_info.get("rebuilt", 0),
            "db_size_before": db_before,
            "db_size_after": db_after,
            "vector_index_bytes_before": vec_before,
            "vector_index_bytes_after": vec_after,
        }

    def get_cached_image_caption(self, sha256: str, model: str | None = None) -> str | None:
        """Return cached caption for an image sha256 if present."""
        with self._lock:
            if model:
                row = self.conn.execute(
                    "SELECT caption FROM image_cache WHERE sha256 = ? AND model = ?",
                    (sha256, model),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT caption FROM image_cache WHERE sha256 = ?",
                    (sha256,),
                ).fetchone()
            return str(row["caption"]) if row else None

    def cache_image_caption(
        self,
        sha256: str,
        model: str,
        caption: str,
        updated_at: str | None = None,
    ) -> None:
        """Persist an image vision caption keyed by content sha256."""
        ts = updated_at or datetime.now().astimezone().isoformat()
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO image_cache (sha256, model, caption, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (sha256, model, caption, ts),
            )

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """Fetch a document and all of its ordered chunks."""
        with self._lock:
            doc_row = self.conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if doc_row is None:
                return None
            chunk_rows = self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC",
                (doc_id,),
            ).fetchall()
        return DocumentRecord(
            doc_id=doc_row["doc_id"],
            uri=doc_row["uri"],
            source_harness=doc_row["source_harness"],
            media_type=MediaType(doc_row["media_type"]),
            project_id=doc_row["project_id"],
            canonical_tier=RepoTier(doc_row["canonical_tier"]),
            title=doc_row["title"],
            l0_summary=doc_row["l0_summary"],
            sha256=doc_row["sha256"],
            updated_at=doc_row["updated_at"],
            metadata=json.loads(doc_row["metadata_json"] or "{}"),
            chunks=[self._row_to_chunk(r) for r in chunk_rows],
        )

    def get_chunk(self, chunk_id: str, include_neighbors: bool = False) -> dict[str, Any] | None:
        """Fetch full chunk details with document context and optional neighbor chunks."""
        if chunk_id.startswith("decision:"):
            dec_id = chunk_id.removeprefix("decision:")
            with self._lock:
                row = self.conn.execute(
                    "SELECT * FROM decisions WHERE decision_id = ?",
                    (dec_id,),
                ).fetchone()
            if row is None:
                return None
            tier = RepoTier.ACTIVE.value if row["valid_to"] is None else RepoTier.ARCHIVE.value
            return {
                "chunk_id": chunk_id,
                "doc_id": chunk_id,
                "chunk_index": 0,
                "uri": f"chittam://decision/{dec_id}",
                "title": f"[{row['project_id']}] {row['topic']}",
                "source_harness": row["source_harness"],
                "media_type": MediaType.DECISION.value,
                "project_id": row["project_id"],
                "canonical_tier": tier,
                "contextual_header": f"[{row['project_id']} | decision | {row['topic']}]",
                "content": row["statement"],
                "symbol_name": row["topic"],
                "line_start": None,
                "line_end": None,
                "token_estimate": max(1, len(row["statement"]) // 4),
                "updated_at": row["created_at"],
                "metadata": {
                    "decision_id": row["decision_id"],
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "superseded_by": row["superseded_by"],
                },
                "neighbors": {"prev": None, "next": None} if include_neighbors else None,
            }

        with self._lock:
            row = self.conn.execute(
                """
                SELECT c.*, d.uri, d.source_harness, d.media_type, d.project_id,
                       d.canonical_tier, d.title, d.l0_summary, d.updated_at
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
            if row is None:
                return None

            neighbors: dict[str, dict[str, Any] | None] | None = None
            if include_neighbors:
                prev_row = self.conn.execute(
                    """
                    SELECT chunk_id, chunk_index, contextual_header, content, symbol_name, line_start, line_end
                    FROM chunks
                    WHERE doc_id = ? AND chunk_index = ?
                    """,
                    (row["doc_id"], row["chunk_index"] - 1),
                ).fetchone()
                next_row = self.conn.execute(
                    """
                    SELECT chunk_id, chunk_index, contextual_header, content, symbol_name, line_start, line_end
                    FROM chunks
                    WHERE doc_id = ? AND chunk_index = ?
                    """,
                    (row["doc_id"], row["chunk_index"] + 1),
                ).fetchone()
                neighbors = {
                    "prev": dict(prev_row) if prev_row else None,
                    "next": dict(next_row) if next_row else None,
                }

        result: dict[str, Any] = {
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "chunk_index": row["chunk_index"],
            "uri": row["uri"],
            "title": row["title"],
            "l0_summary": row["l0_summary"],
            "source_harness": row["source_harness"],
            "media_type": row["media_type"],
            "project_id": row["project_id"],
            "canonical_tier": row["canonical_tier"],
            "contextual_header": row["contextual_header"],
            "content": row["content"],
            "symbol_name": row["symbol_name"],
            "line_start": row["line_start"],
            "line_end": row["line_end"],
            "token_estimate": row["token_estimate"],
            "updated_at": row["updated_at"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }
        if include_neighbors:
            result["neighbors"] = neighbors
        return result

    def _apply_reranker(
        self,
        query: str,
        candidates: list[SearchResult],
        *,
        reranker: Any | None,
        rerank_top_k: int,
        now: datetime,
    ) -> list[SearchResult]:
        if reranker is None or len(candidates) <= 1 or not query.strip():
            return candidates

        k = max(2, int(rerank_top_k))
        head = list(candidates[:k])
        tail = list(candidates[k:])
        passages = [
            (f"[{r.project_id}] {r.title}\n{r.snippet}" if r.project_id else f"{r.title}\n{r.snippet}").strip()
            for r in head
        ]

        try:
            ranked = reranker.rerank(query, passages, top_k=len(head))
        except Exception:
            return candidates
        if not ranked:
            return candidates

        raw_scores: dict[int, float] = {int(idx): float(sc) for idx, sc in ranked if 0 <= int(idx) < len(head)}
        if not raw_scores:
            return candidates

        min_ce = min(raw_scores.values())
        max_ce = max(raw_scores.values())
        span_ce = max_ce - min_ce
        max_rrf = max((r.score for r in head), default=1.0) or 1.0

        rescored: list[SearchResult] = []
        for pos, item in enumerate(head):
            raw_ce = raw_scores.get(pos, min_ce)
            ce_norm = ((raw_ce - min_ce) / span_ce) if span_ce > 1e-9 else 1.0
            rrf_norm = item.score / max_rrf
            tier_w = _tier_weight(item.canonical_tier.value)
            rec_w = _recency_multiplier(item.updated_at, now)
            blended = (0.70 * ce_norm + 0.30 * rrf_norm) * tier_w * rec_w
            new_meta = dict(item.metadata)
            new_meta["rerank_score"] = round(raw_ce, 6)
            rescored.append(
                item.model_copy(
                    update={
                        "score": round(blended, 8),
                        "metadata": new_meta,
                    }
                )
            )

        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored + tail

    def search(
        self,
        query: str,
        query_vec: np.ndarray | list[float] | None = None,
        limit: int = 8,
        media_type: str | MediaType | None = None,
        project_id: str | None = None,
        source_harness: str | None = None,
        include_archived: bool = True,
        reranker: Any | None = None,
        rerank_top_k: int = 20,
    ) -> list[SearchResult]:
        """Hybrid search combining FTS5 BM25 + USearch cosine similarity via RRF(k=60) * tier * recency (+ optional rerank)."""
        if limit <= 0:
            return []

        fts_expr, tokens = sanitize_fts5_query(query)
        q_vec = _normalize_vector(np.asarray(query_vec, dtype=np.float32)) if query_vec is not None else None
        if q_vec is not None and (q_vec.size == 0 or float(np.linalg.norm(q_vec)) == 0.0):
            q_vec = None

        if fts_expr is None and q_vec is None:
            return []

        media_val = media_type.value if isinstance(media_type, MediaType) else media_type
        where_clauses, params = self._build_filter_sql(
            media_type=media_val,
            project_id=project_id,
            source_harnesses=[source_harness] if source_harness else None,
            include_archived=include_archived,
        )
        candidate_k = max(limit * 8, rerank_top_k if reranker is not None else limit, 64)
        pool_limit = max(limit, rerank_top_k) if reranker is not None else limit
        candidates = self._run_hybrid_search(
            fts_expr=fts_expr,
            tokens=tokens,
            q_vec=q_vec,
            limit=pool_limit,
            candidate_k=candidate_k,
            where_clauses=where_clauses,
            params=params,
            include_decisions=(media_val is None or media_val == MediaType.DECISION.value)
            and source_harness is None,
            explicit_decision_filter=(media_val == MediaType.DECISION.value),
            project_id=project_id,
            include_archived=include_archived,
        )
        reranked = self._apply_reranker(
            query=query,
            candidates=candidates,
            reranker=reranker,
            rerank_top_k=rerank_top_k,
            now=datetime.now(timezone.utc),
        )
        return reranked[:limit]

    def recall_threads(
        self,
        query: str,
        query_vec: np.ndarray | list[float] | None = None,
        harness: str | None = None,
        limit: int = 8,
        reranker: Any | None = None,
        rerank_top_k: int = 20,
    ) -> list[SearchResult]:
        """Search cross-harness agent threads/memories deduplicated per thread document."""
        if limit <= 0:
            return []

        harnesses: list[str]
        if harness:
            harnesses = list(_HARNESS_ALIASES.get(harness.lower(), (harness,)))
        else:
            harnesses = list(_DEFAULT_THREAD_HARNESSES)

        fts_expr, tokens = sanitize_fts5_query(query)
        q_vec = _normalize_vector(np.asarray(query_vec, dtype=np.float32)) if query_vec is not None else None
        if q_vec is not None and (q_vec.size == 0 or float(np.linalg.norm(q_vec)) == 0.0):
            q_vec = None

        placeholders = ", ".join("?" for _ in harnesses)
        if harness:
            scope_sql = f"d.source_harness IN ({placeholders})"
            scope_params: list[Any] = list(harnesses)
        else:
            scope_sql = f"(d.media_type = ? OR d.source_harness IN ({placeholders}))"
            scope_params = [MediaType.TRANSCRIPT.value, *harnesses]

        if fts_expr is None and q_vec is None:
            with self._lock:
                rows = self.conn.execute(
                    f"""
                    SELECT c.chunk_id, c.doc_id, c.content, d.uri, d.title, d.source_harness,
                           d.media_type, d.project_id, d.canonical_tier, d.l0_summary,
                           d.updated_at, d.metadata_json
                    FROM documents d
                    JOIN chunks c ON c.doc_id = d.doc_id
                    WHERE {scope_sql}
                    ORDER BY d.updated_at DESC, c.chunk_index ASC
                    LIMIT ?
                    """,
                    (*scope_params, limit * 4),
                ).fetchall()
            seen_docs: set[str] = set()
            recent_results: list[SearchResult] = []
            now = datetime.now(timezone.utc)
            for row in rows:
                if row["doc_id"] in seen_docs:
                    continue
                seen_docs.add(row["doc_id"])
                score = _tier_weight(row["canonical_tier"]) * _recency_multiplier(row["updated_at"], now)
                recent_results.append(
                    SearchResult(
                        chunk_id=row["chunk_id"],
                        doc_id=row["doc_id"],
                        uri=row["uri"],
                        title=row["title"],
                        source_harness=row["source_harness"],
                        media_type=MediaType(row["media_type"]),
                        project_id=row["project_id"],
                        canonical_tier=RepoTier(row["canonical_tier"]),
                        score=round(score, 6),
                        bm25_rank=None,
                        vector_rank=None,
                        snippet=_make_snippet(row["content"], []),
                        l0_summary=row["l0_summary"],
                        updated_at=row["updated_at"],
                        metadata=json.loads(row["metadata_json"] or "{}"),
                    )
                )
                if len(recent_results) >= limit:
                    break
            return recent_results

        where_clauses = [scope_sql]
        params: list[Any] = list(scope_params)
        pool_limit = max(limit * 4, rerank_top_k * 2 if reranker is not None else limit * 4)
        raw_results = self._run_hybrid_search(
            fts_expr=fts_expr,
            tokens=tokens,
            q_vec=q_vec,
            limit=pool_limit,
            candidate_k=max(limit * 10, rerank_top_k * 2 if reranker is not None else limit * 10, 64),
            where_clauses=where_clauses,
            params=params,
            include_decisions=False,
            project_id=None,
            include_archived=True,
        )
        deduped: list[SearchResult] = []
        seen_doc_ids: set[str] = set()
        target_pool = max(limit, rerank_top_k) if reranker is not None else limit
        for item in raw_results:
            if item.doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(item.doc_id)
            deduped.append(item)
            if len(deduped) >= target_pool:
                break
        reranked = self._apply_reranker(
            query=query,
            candidates=deduped,
            reranker=reranker,
            rerank_top_k=rerank_top_k,
            now=datetime.now(timezone.utc),
        )
        return reranked[:limit]

    def record_decision(
        self,
        project_id: str,
        topic: str,
        statement: str,
        source_harness: str = "manual",
        supersedes_topic: bool = True,
    ) -> DecisionRecord:
        """Record a bitemporal decision, closing any prior active decision for (project_id, topic)."""
        now = datetime.now().astimezone().isoformat()
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        record = DecisionRecord(
            decision_id=decision_id,
            project_id=project_id,
            topic=topic,
            statement=statement,
            source_harness=source_harness,
            valid_from=now,
            valid_to=None,
            superseded_by=None,
            created_at=now,
        )
        with self._lock, self.conn:
            if supersedes_topic:
                self.conn.execute(
                    """
                    UPDATE decisions
                    SET valid_to = ?, superseded_by = ?
                    WHERE project_id = ? AND topic = ? AND valid_to IS NULL
                    """,
                    (now, decision_id, project_id, topic),
                )
            self.conn.execute(
                """
                INSERT INTO decisions (
                    decision_id, project_id, topic, statement, source_harness,
                    valid_from, valid_to, superseded_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.decision_id,
                    record.project_id,
                    record.topic,
                    record.statement,
                    record.source_harness,
                    record.valid_from,
                    record.valid_to,
                    record.superseded_by,
                    record.created_at,
                ),
            )
        return record

    def list_decisions(
        self,
        project_id: str | None = None,
        active_only: bool = True,
    ) -> list[DecisionRecord]:
        """List recorded decisions, optionally scoped to project_id and active-only."""
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if active_only:
            clauses.append("valid_to IS NULL")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM decisions {where_sql} ORDER BY created_at DESC, rowid DESC",
                params,
            ).fetchall()
        return [self._row_to_decision(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        """Return summary counts, storage footprint, and vector backend breakdown."""
        with self._lock:
            doc_count = int(self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunk_count = int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            emb_count = int(self.conn.execute("SELECT COUNT(*) FROM vector_entries").fetchone()[0])
            unique_emb_count = int(self.conn.execute("SELECT COUNT(*) FROM content_embeddings").fetchone()[0])
            legacy_emb_count = int(self.conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0])
            dec_count = int(self.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])
            active_dec_count = int(
                self.conn.execute("SELECT COUNT(*) FROM decisions WHERE valid_to IS NULL").fetchone()[0]
            )
            img_cache_count = int(self.conn.execute("SELECT COUNT(*) FROM image_cache").fetchone()[0])
            last_updated = self.conn.execute("SELECT MAX(updated_at) FROM documents").fetchone()[0]

            media_rows = self.conn.execute(
                "SELECT media_type, COUNT(*) AS cnt FROM documents GROUP BY media_type"
            ).fetchall()
            harness_rows = self.conn.execute(
                "SELECT source_harness, COUNT(*) AS cnt FROM documents GROUP BY source_harness"
            ).fetchall()
            tier_rows = self.conn.execute(
                "SELECT canonical_tier, COUNT(*) AS cnt FROM documents GROUP BY canonical_tier"
            ).fetchall()
            meta = self.get_index_meta()

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        vec_size = self.vector_index_path.stat().st_size if self.vector_index_path.exists() else 0
        dim_raw = meta.get("active_embedding_dim")
        return {
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
            "vector_index_path": str(self.vector_index_path),
            "vector_index_bytes": vec_size,
            "vector_backend": self.vector_backend,
            "vector_dim": int(dim_raw) if dim_raw else None,
            "vector_dtype": meta.get("vector_dtype", self.vector_dtype),
            "embedding_model": meta.get("active_embedding_model"),
            "embedding_source": meta.get("active_embedding_source"),
            "schema_version": meta.get("schema_version", _SCHEMA_VERSION),
            "documents": doc_count,
            "chunks": chunk_count,
            "embeddings": emb_count if emb_count > 0 or legacy_emb_count == 0 else legacy_emb_count,
            "unique_embeddings": unique_emb_count,
            "legacy_embeddings": legacy_emb_count,
            "decisions": dec_count,
            "active_decisions": active_dec_count,
            "cached_images": img_cache_count,
            "last_updated_at": last_updated,
            "by_media_type": {r["media_type"]: int(r["cnt"]) for r in media_rows},
            "by_harness": {r["source_harness"]: int(r["cnt"]) for r in harness_rows},
            "by_tier": {r["canonical_tier"]: int(r["cnt"]) for r in tier_rows},
        }

    def iter_chunks_for_export(self) -> list[dict[str, Any]]:
        """Return lightweight chunk records for optional LEANN corpus export."""
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.contextual_header, c.content,
                       d.uri, d.title, d.project_id, d.canonical_tier, d.source_harness, d.media_type
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                ORDER BY c.doc_id, c.chunk_index
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def _build_filter_sql(
        self,
        media_type: str | None,
        project_id: str | None,
        source_harnesses: list[str] | None,
        include_archived: bool,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if media_type:
            clauses.append("d.media_type = ?")
            params.append(media_type)
        if project_id:
            clauses.append("d.project_id = ?")
            params.append(project_id)
        if source_harnesses:
            if len(source_harnesses) == 1:
                clauses.append("d.source_harness = ?")
                params.append(source_harnesses[0])
            else:
                ph = ", ".join("?" for _ in source_harnesses)
                clauses.append(f"d.source_harness IN ({ph})")
                params.extend(source_harnesses)
        if not include_archived:
            clauses.append("d.canonical_tier != ?")
            params.append(RepoTier.ARCHIVE.value)
        return clauses, params

    def _run_vector_fallback_filtered_chunks_unlocked(
        self,
        q_vec: np.ndarray,
        dim: int,
        candidate_k: int,
        filter_sql: str,
        params: list[Any],
    ) -> dict[str, int]:
        rows = self.conn.execute(
            f"""
            SELECT ve.target_id, ce.vector, ce.dtype
            FROM vector_entries ve
            JOIN content_embeddings ce
              ON ce.content_sha256 = ve.content_sha256
             AND ce.model = ve.model
             AND ce.dim = ve.dim
            JOIN chunks c ON c.chunk_id = ve.target_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE ve.target_type = 'chunk' AND ve.dim = ? {filter_sql}
            """,
            (dim, *params),
        ).fetchall()
        if not rows:
            return {}

        grouped: dict[str, list[tuple[str, bytes]]] = {}
        for r in rows:
            grouped.setdefault(str(r["dtype"]), []).append((str(r["target_id"]), bytes(r["vector"])))

        scored: list[tuple[str, float]] = []
        for dt, items in grouped.items():
            cids = [cid for cid, _ in items]
            mat = dequantize_matrix([b for _, b in items], dim=dim, dtype=dt)
            sims = mat @ q_vec
            for cid, sim_val in zip(cids, sims):
                if float(sim_val) > 0.0:
                    scored.append((cid, float(sim_val)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return {cid: idx for idx, (cid, _) in enumerate(scored[:candidate_k], start=1)}

    def _run_hybrid_search(
        self,
        fts_expr: str | None,
        tokens: list[str],
        q_vec: np.ndarray | None,
        limit: int,
        candidate_k: int,
        where_clauses: list[str],
        params: list[Any],
        include_decisions: bool,
        project_id: str | None,
        include_archived: bool,
        explicit_decision_filter: bool = False,
    ) -> list[SearchResult]:
        bm25_ranks: dict[str, int] = {}
        vector_ranks: dict[str, int] = {}
        decision_vector_ranks: dict[str, int] = {}

        filter_sql = f"AND {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._lock:
            if fts_expr:
                try:
                    fts_rows = self.conn.execute(
                        f"""
                        SELECT chunks_fts.chunk_id AS chunk_id,
                               bm25(chunks_fts, 0.0, 0.0, 2.0, 1.0, 3.0, 2.0, 0.5) AS bm25_score
                        FROM chunks_fts
                        JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                        JOIN documents d ON d.doc_id = c.doc_id
                        WHERE chunks_fts MATCH ? {filter_sql}
                        ORDER BY bm25_score ASC
                        LIMIT ?
                        """,
                        (fts_expr, *params, candidate_k),
                    ).fetchall()
                    for idx, row in enumerate(fts_rows, start=1):
                        bm25_ranks[row["chunk_id"]] = idx
                except sqlite3.OperationalError:
                    pass

            if q_vec is not None:
                dim = int(q_vec.shape[0])
                vec_idx = self._get_vec_index_for_search(expected_dim=dim)
                if vec_idx is not None and vec_idx.count() > 0:
                    oversample_k = min(max(candidate_k * 6, 256), vec_idx.count())
                    vec_hits = vec_idx.search(q_vec, oversample_k)
                    if vec_hits:
                        hit_keys = [k for k, _ in vec_hits]
                        key_ph = ", ".join("?" for _ in hit_keys)
                        entry_rows = self.conn.execute(
                            f"""
                            SELECT vec_key, target_id, target_type
                            FROM vector_entries
                            WHERE vec_key IN ({key_ph})
                            """,
                            hit_keys,
                        ).fetchall()
                        key_map = {
                            int(r["vec_key"]): (str(r["target_id"]), str(r["target_type"]))
                            for r in entry_rows
                        }

                        candidate_chunk_ids: list[str] = []
                        for global_rank, (key, sim) in enumerate(vec_hits, start=1):
                            pair = key_map.get(key)
                            if pair is None:
                                continue
                            tid, ttype = pair
                            if ttype == "chunk":
                                candidate_chunk_ids.append(tid)
                            elif ttype == "decision" and sim >= 0.35 and tid not in decision_vector_ranks:
                                decision_vector_ranks[tid] = global_rank

                        if candidate_chunk_ids:
                            if where_clauses:
                                cid_ph = ", ".join("?" for _ in candidate_chunk_ids)
                                allowed_rows = self.conn.execute(
                                    f"""
                                    SELECT c.chunk_id
                                    FROM chunks c
                                    JOIN documents d ON d.doc_id = c.doc_id
                                    WHERE c.chunk_id IN ({cid_ph}) {filter_sql}
                                    """,
                                    (*candidate_chunk_ids, *params),
                                ).fetchall()
                                allowed_set = {str(r["chunk_id"]) for r in allowed_rows}
                                filtered_cids = [c for c in candidate_chunk_ids if c in allowed_set]
                            else:
                                filtered_cids = candidate_chunk_ids

                            for rank_idx, cid in enumerate(filtered_cids[:candidate_k], start=1):
                                vector_ranks[cid] = rank_idx

                    # Narrow-filter fallback when a tight filter (e.g. image/project) gets < limit hits from HNSW
                    if where_clauses and len(vector_ranks) < limit:
                        vector_ranks = self._run_vector_fallback_filtered_chunks_unlocked(
                            q_vec=q_vec,
                            dim=dim,
                            candidate_k=candidate_k,
                            filter_sql=filter_sql,
                            params=params,
                        )
                else:
                    # Fallback to direct SQLite content_embeddings scan when USearch file isn't present
                    vector_ranks = self._run_vector_fallback_filtered_chunks_unlocked(
                        q_vec=q_vec,
                        dim=dim,
                        candidate_k=candidate_k,
                        filter_sql=filter_sql,
                        params=params,
                    )

            all_chunk_ids = list(dict.fromkeys([*bm25_ranks.keys(), *vector_ranks.keys()]))
            fetched_rows: dict[str, sqlite3.Row] = {}
            if all_chunk_ids:
                ph = ", ".join("?" for _ in all_chunk_ids)
                detail_rows = self.conn.execute(
                    f"""
                    SELECT c.chunk_id, c.doc_id, c.contextual_header, c.content,
                           c.symbol_name, c.line_start, c.line_end, c.metadata_json AS chunk_meta_json,
                           d.uri, d.source_harness, d.media_type, d.project_id,
                           d.canonical_tier, d.title, d.l0_summary, d.updated_at,
                           d.metadata_json AS doc_meta_json
                    FROM chunks c
                    JOIN documents d ON d.doc_id = c.doc_id
                    WHERE c.chunk_id IN ({ph})
                    """,
                    all_chunk_ids,
                ).fetchall()
                fetched_rows = {r["chunk_id"]: r for r in detail_rows}

        now = datetime.now(timezone.utc)
        results: list[SearchResult] = []
        for cid in all_chunk_ids:
            row = fetched_rows.get(cid)
            if row is None:
                continue
            b_rank = bm25_ranks.get(cid)
            v_rank = vector_ranks.get(cid)
            rrf = 0.0
            if b_rank is not None:
                rrf += 1.0 / (_RRF_K + b_rank)
            if v_rank is not None:
                rrf += 1.0 / (_RRF_K + v_rank)

            tier_w = _tier_weight(row["canonical_tier"])
            rec_w = _recency_multiplier(row["updated_at"], now)
            final_score = rrf * tier_w * rec_w

            doc_meta = json.loads(row["doc_meta_json"] or "{}")
            chunk_meta = json.loads(row["chunk_meta_json"] or "{}")
            merged_meta = {**doc_meta, **chunk_meta}
            if row["symbol_name"]:
                merged_meta.setdefault("symbol_name", row["symbol_name"])
            if row["line_start"] is not None:
                merged_meta.setdefault("line_start", row["line_start"])
            if row["line_end"] is not None:
                merged_meta.setdefault("line_end", row["line_end"])

            results.append(
                SearchResult(
                    chunk_id=cid,
                    doc_id=row["doc_id"],
                    uri=row["uri"],
                    title=row["title"],
                    source_harness=row["source_harness"],
                    media_type=MediaType(row["media_type"]),
                    project_id=row["project_id"],
                    canonical_tier=RepoTier(row["canonical_tier"]),
                    score=round(final_score, 8),
                    bm25_rank=b_rank,
                    vector_rank=v_rank,
                    snippet=_make_snippet(row["content"], tokens),
                    l0_summary=row["l0_summary"][:_MAX_SNIPPET_CHARS],
                    updated_at=row["updated_at"],
                    metadata=merged_meta,
                )
            )

        # Keep only the highest-scoring chunk per document so multi-chunk files don't crowd out top hits
        best_by_doc: dict[str, SearchResult] = {}
        for item in results:
            prev = best_by_doc.get(item.doc_id)
            if prev is None or item.score > prev.score:
                best_by_doc[item.doc_id] = item
        results = list(best_by_doc.values())
        has_chunk_results = bool(results)

        if include_decisions and (tokens or q_vec is not None or decision_vector_ranks):
            results.extend(
                self._search_decisions(
                    tokens=tokens,
                    q_vec=q_vec,
                    decision_vector_ranks=decision_vector_ranks,
                    project_id=project_id,
                    include_archived=include_archived,
                    now=now,
                    explicit_decision_filter=explicit_decision_filter,
                    has_chunk_results=has_chunk_results,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _search_decisions(
        self,
        tokens: list[str],
        q_vec: np.ndarray | None,
        decision_vector_ranks: dict[str, int],
        project_id: str | None,
        include_archived: bool,
        now: datetime,
        *,
        explicit_decision_filter: bool = False,
        has_chunk_results: bool = True,
    ) -> list[SearchResult]:
        decisions = self.list_decisions(project_id=project_id, active_only=not include_archived)
        if not decisions:
            return []

        by_id = {d.decision_id: d for d in decisions}

        # Lexical ranks
        lex_ranks: dict[str, int] = {}
        if tokens:
            lowered_tokens = [t.lower() for t in tokens]
            scored_lex: list[tuple[int, DecisionRecord]] = []
            for dec in decisions:
                hay = f"{dec.project_id} {dec.topic} {dec.statement}".lower()
                hits = sum(1 for tok in lowered_tokens if tok in hay)
                if hits > 0:
                    scored_lex.append((hits, dec))
            scored_lex.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
            for rank, (_, dec) in enumerate(scored_lex, start=1):
                lex_ranks[dec.decision_id] = rank

        # Semantic ranks: use global USearch ranks, reserving direct decision-table scan for explicit/empty searches
        vec_ranks: dict[str, int] = {
            did: rank for did, rank in decision_vector_ranks.items() if did in by_id
        }
        if q_vec is not None and (explicit_decision_filter or not has_chunk_results):
            dim = int(q_vec.shape[0])
            with self._lock:
                rows = self.conn.execute(
                    """
                    SELECT ve.target_id, ce.vector, ce.dtype
                    FROM vector_entries ve
                    JOIN content_embeddings ce
                      ON ce.content_sha256 = ve.content_sha256
                     AND ce.model = ve.model
                     AND ce.dim = ve.dim
                    WHERE ve.target_type = 'decision' AND ve.dim = ?
                    """,
                    (dim,),
                ).fetchall()
            filtered_rows = [r for r in rows if str(r["target_id"]) in by_id]
            if filtered_rows:
                grouped: dict[str, list[tuple[str, bytes]]] = {}
                for r in filtered_rows:
                    grouped.setdefault(str(r["dtype"]), []).append((str(r["target_id"]), bytes(r["vector"])))
                scored_vec: list[tuple[str, float]] = []
                for dt, items in grouped.items():
                    dids = [did for did, _ in items]
                    mat = dequantize_matrix([b for _, b in items], dim=dim, dtype=dt)
                    sims = mat @ q_vec
                    for did, sim_val in zip(dids, sims):
                        if float(sim_val) >= 0.25:
                            scored_vec.append((did, float(sim_val)))
                scored_vec.sort(key=lambda pair: pair[1], reverse=True)
                vec_ranks = {did: idx for idx, (did, _) in enumerate(scored_vec, start=1)}

        matched_ids = list(dict.fromkeys([*lex_ranks.keys(), *vec_ranks.keys()]))
        out: list[SearchResult] = []
        for did in matched_ids:
            dec = by_id.get(did)
            if dec is None:
                continue
            l_rank = lex_ranks.get(did)
            v_rank = vec_ranks.get(did)
            rrf = 0.0
            if l_rank is not None:
                rrf += 1.0 / (_RRF_K + l_rank)
            if v_rank is not None:
                rrf += 1.0 / (_RRF_K + v_rank)
            if rrf <= 0.0:
                continue

            tier = RepoTier.ACTIVE if dec.valid_to is None else RepoTier.ARCHIVE
            score = rrf * TIER_WEIGHTS[tier] * _recency_multiplier(dec.created_at, now)
            cid = f"decision:{dec.decision_id}"
            out.append(
                SearchResult(
                    chunk_id=cid,
                    doc_id=cid,
                    uri=f"chittam://decision/{dec.decision_id}",
                    title=f"[{dec.project_id}] {dec.topic}",
                    source_harness=dec.source_harness,
                    media_type=MediaType.DECISION,
                    project_id=dec.project_id,
                    canonical_tier=tier,
                    score=round(score, 8),
                    bm25_rank=l_rank,
                    vector_rank=v_rank,
                    snippet=_make_snippet(f"{dec.topic}: {dec.statement}", tokens),
                    l0_summary=_make_snippet(dec.statement, tokens),
                    updated_at=dec.created_at,
                    metadata={
                        "decision_id": dec.decision_id,
                        "valid_from": dec.valid_from,
                        "valid_to": dec.valid_to,
                        "superseded_by": dec.superseded_by,
                    },
                )
            )
        out.sort(key=lambda r: r.score, reverse=True)
        return out

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            chunk_index=int(row["chunk_index"]),
            contextual_header=row["contextual_header"],
            content=row["content"],
            symbol_name=row["symbol_name"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            token_estimate=int(row["token_estimate"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    @staticmethod
    def _row_to_decision(row: sqlite3.Row) -> DecisionRecord:
        return DecisionRecord(
            decision_id=row["decision_id"],
            project_id=row["project_id"],
            topic=row["topic"],
            statement=row["statement"],
            source_harness=row["source_harness"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            superseded_by=row["superseded_by"],
            created_at=row["created_at"],
        )
