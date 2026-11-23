"""SQLite WAL + FTS5 + vectorized cosine + bitemporal decision store for Chittam."""

from __future__ import annotations

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

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MAX_FTS_TOKENS = 24
_MAX_SNIPPET_CHARS = 350
_RRF_K = 60.0

_HARNESS_ALIASES: dict[str, tuple[str, ...]] = {
    "antigravity": ("antigravity",),
    "codex": ("codex",),
    "claude": ("claude", "claude_code", "claude-code"),
    "claude_code": ("claude", "claude_code", "claude-code"),
    "claude-code": ("claude", "claude_code", "claude-code"),
}
_DEFAULT_THREAD_HARNESSES = ("antigravity", "codex", "claude", "claude_code", "claude-code")


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


class ChittamStore:
    """SQLite WAL hybrid store with FTS5 BM25, numpy cosine vectors, image cache, and bitemporal decisions."""

    def __init__(self, db_path: Path | str | AppConfig) -> None:
        resolved = db_path.db_path if isinstance(db_path, AppConfig) else Path(db_path).expanduser()
        self.db_path = resolved
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def __enter__(self) -> ChittamStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _init_db(self) -> None:
        required_tables = {
            "documents",
            "chunks",
            "chunks_fts",
            "chunk_embeddings",
            "image_cache",
            "decisions",
        }
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
            if required_tables.issubset(existing_tables):
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
                    metadata_json TEXT NOT NULL DEFAULT '{}'
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
                    metadata_json TEXT NOT NULL DEFAULT '{}'
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
                """
            )
            self.conn.commit()

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

    def has_embeddings(self) -> bool:
        """Return True if at least one chunk embedding is stored."""
        with self._lock:
            row = self.conn.execute("SELECT 1 FROM chunk_embeddings LIMIT 1").fetchone()
            return row is not None

    def upsert_document(self, doc: DocumentRecord) -> None:
        """Atomically insert or replace a document, its chunks, and its FTS5 entries."""
        media_val = doc.media_type.value if isinstance(doc.media_type, MediaType) else str(doc.media_type)
        tier_val = doc.canonical_tier.value if isinstance(doc.canonical_tier, RepoTier) else str(doc.canonical_tier)
        doc_meta = json.dumps(doc.metadata, ensure_ascii=False)

        with self._lock, self.conn:
            exists = (
                self.conn.execute(
                    "SELECT 1 FROM documents WHERE doc_id = ? LIMIT 1",
                    (doc.doc_id,),
                ).fetchone()
                is not None
            )
            if exists:
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
                    canonical_tier, title, l0_summary, sha256, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    )
                    for c in doc.chunks
                ]
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, doc_id, chunk_index, contextual_header, content,
                        symbol_name, line_start, line_end, token_estimate, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its synced chunks, embeddings, and FTS5 rows."""
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            self.conn.execute(
                "DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cur = self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
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
            if self.delete_document(doc_id):
                deleted += 1
        return deleted

    def get_chunks_missing_embeddings(
        self,
        model: str | None = None,
        limit: int | None = None,
    ) -> list[ChunkRecord]:
        """Return chunks lacking an embedding (or lacking an embedding for `model`), prioritized by tier and modality."""
        limit_sql = "LIMIT ?" if limit is not None and limit > 0 else ""
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
            if model:
                where_sql = "WHERE ce.chunk_id IS NULL OR ce.model != ?"
                params.append(model)
            else:
                where_sql = "WHERE ce.chunk_id IS NULL"
            if limit_sql:
                params.append(int(limit))
            rows = self.conn.execute(
                f"""
                SELECT c.*
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.chunk_id
                {where_sql}
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
    ) -> int:
        """Store L2-normalized float32 embeddings for the provided chunk IDs."""
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
        rows: list[tuple[str, str, int, bytes]] = []
        for idx, cid in enumerate(chunk_ids):
            normed = _normalize_vector(mat[idx])
            rows.append((cid, model, dim, normed.tobytes()))
        with self._lock, self.conn:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_embeddings (chunk_id, model, dim, vector)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

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

    def search(
        self,
        query: str,
        query_vec: np.ndarray | list[float] | None = None,
        limit: int = 8,
        media_type: str | MediaType | None = None,
        project_id: str | None = None,
        source_harness: str | None = None,
        include_archived: bool = True,
    ) -> list[SearchResult]:
        """Hybrid search combining FTS5 BM25 + numpy cosine similarity via RRF(k=60) * tier * recency."""
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
        candidate_k = max(limit * 8, 64)
        return self._run_hybrid_search(
            fts_expr=fts_expr,
            tokens=tokens,
            q_vec=q_vec,
            limit=limit,
            candidate_k=candidate_k,
            where_clauses=where_clauses,
            params=params,
            include_decisions=(media_val is None or media_val == MediaType.DECISION.value) and source_harness is None,
            project_id=project_id,
            include_archived=include_archived,
        )

    def recall_threads(
        self,
        query: str,
        query_vec: np.ndarray | list[float] | None = None,
        harness: str | None = None,
        limit: int = 8,
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
        raw_results = self._run_hybrid_search(
            fts_expr=fts_expr,
            tokens=tokens,
            q_vec=q_vec,
            limit=limit * 4,
            candidate_k=max(limit * 10, 64),
            where_clauses=where_clauses,
            params=params,
            include_decisions=False,
            project_id=None,
            include_archived=True,
        )
        deduped: list[SearchResult] = []
        seen_doc_ids: set[str] = set()
        for item in raw_results:
            if item.doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(item.doc_id)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

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
        return [
            DecisionRecord(
                decision_id=r["decision_id"],
                project_id=r["project_id"],
                topic=r["topic"],
                statement=r["statement"],
                source_harness=r["source_harness"],
                valid_from=r["valid_from"],
                valid_to=r["valid_to"],
                superseded_by=r["superseded_by"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_stats(self) -> dict[str, Any]:
        """Return summary counts and index breakdown."""
        with self._lock:
            doc_count = int(self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunk_count = int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            emb_count = int(self.conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0])
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

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
            "documents": doc_count,
            "chunks": chunk_count,
            "embeddings": emb_count,
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
    ) -> list[SearchResult]:
        bm25_ranks: dict[str, int] = {}
        vector_ranks: dict[str, int] = {}

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
                vec_rows = self.conn.execute(
                    f"""
                    SELECT ce.chunk_id, ce.vector
                    FROM chunk_embeddings ce
                    JOIN chunks c ON c.chunk_id = ce.chunk_id
                    JOIN documents d ON d.doc_id = c.doc_id
                    WHERE ce.dim = ? {filter_sql}
                    """,
                    (dim, *params),
                ).fetchall()
                if vec_rows:
                    expected_bytes = dim * 4
                    valid_rows = [
                        (r["chunk_id"], bytes(r["vector"]))
                        for r in vec_rows
                        if len(r["vector"]) == expected_bytes
                    ]
                    if valid_rows:
                        chunk_ids = [cid for cid, _ in valid_rows]
                        blob = b"".join(vec_bytes for _, vec_bytes in valid_rows)
                        mat = np.frombuffer(blob, dtype=np.float32).reshape(len(chunk_ids), dim)
                        norms = np.linalg.norm(mat, axis=1, keepdims=True)
                        safe_norms = np.where(norms > 0.0, norms, 1.0)
                        sims = (mat / safe_norms) @ q_vec
                        order = np.argsort(-sims)
                        rank = 1
                        for pos in order[:candidate_k]:
                            sim_val = float(sims[int(pos)])
                            if sim_val <= 0.0:
                                break
                            vector_ranks[chunk_ids[int(pos)]] = rank
                            rank += 1

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

        if include_decisions and tokens:
            results.extend(
                self._search_decisions(
                    tokens=tokens,
                    project_id=project_id,
                    include_archived=include_archived,
                    now=now,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _search_decisions(
        self,
        tokens: list[str],
        project_id: str | None,
        include_archived: bool,
        now: datetime,
    ) -> list[SearchResult]:
        decisions = self.list_decisions(project_id=project_id, active_only=not include_archived)
        if not decisions:
            return []

        lowered_tokens = [t.lower() for t in tokens]
        scored: list[tuple[int, DecisionRecord]] = []
        for dec in decisions:
            hay = f"{dec.project_id} {dec.topic} {dec.statement}".lower()
            hits = sum(1 for tok in lowered_tokens if tok in hay)
            if hits > 0:
                scored.append((hits, dec))

        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        out: list[SearchResult] = []
        for rank, (_, dec) in enumerate(scored, start=1):
            tier = RepoTier.ACTIVE if dec.valid_to is None else RepoTier.ARCHIVE
            score = (1.0 / (_RRF_K + rank)) * TIER_WEIGHTS[tier] * _recency_multiplier(dec.created_at, now)
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
                    bm25_rank=rank,
                    vector_rank=None,
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
