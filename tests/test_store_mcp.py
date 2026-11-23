"""Self-contained unit tests for ChittamStore, Merkle indexer, 6 MCP tools, and CLI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from chittam import indexer, mcp_server
from chittam.atlas import save_atlas
from chittam.cli import app as cli_app
from chittam.models import (
    AppConfig,
    CanonicalRepo,
    ChunkRecord,
    DocumentRecord,
    MediaType,
    RepoTier,
    SyncStats,
)
from chittam.store import ChittamStore


class FakeDeterministicEmbedder:
    """Deterministic hashed bag-of-words embedder returning unit-norm float32 vectors."""

    model_name = "fake-embed-32d"

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.embed_calls = 0
        self.query_calls = 0
        self.embedded_texts: list[str] = []

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        if not tokens:
            vec[0] = 1.0
            return vec
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.embed_calls += 1
        self.embedded_texts.extend(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self._encode(t) for t in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        return self._encode(query)


class FakeVisionProvider:
    """Deterministic vision captioner counting calls."""

    model_name = "fake-vision-vl"

    def __init__(self) -> None:
        self.calls = 0

    def describe_image(self, image_path: Path, context_hint: str = "") -> str:
        self.calls += 1
        return f"Diagram screenshot from {image_path.name}: system architecture ({context_hint})"


def _make_temp_config(tmp_path: Path) -> tuple[AppConfig, Path]:
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig(
        data_dir=str(data_dir),
        config_dir=str(config_dir),
        enable_leann_bridge=False,
    )
    cfg_path = config_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    return cfg, cfg_path


def _make_doc(
    doc_id: str,
    title: str,
    content: str,
    *,
    uri: str | None = None,
    project_id: str = "chittam",
    tier: RepoTier = RepoTier.ACTIVE,
    media_type: MediaType = MediaType.CODE,
    source_harness: str = "files",
    updated_at: str | None = None,
    extra_chunks: list[str] | None = None,
) -> DocumentRecord:
    ts = updated_at or datetime.now(timezone.utc).isoformat()
    all_texts = [content, *(extra_chunks or [])]
    sha = hashlib.sha256("\n---\n".join(all_texts).encode("utf-8")).hexdigest()
    chunks = [
        ChunkRecord(
            chunk_id=f"{doc_id}:chunk:{idx}",
            doc_id=doc_id,
            chunk_index=idx,
            contextual_header=f"[{project_id} | {tier.value} | {title} | {ts}]",
            content=chunk_text,
            symbol_name=f"sym_{idx}",
            line_start=idx * 10 + 1,
            line_end=idx * 10 + 10,
            token_estimate=max(1, len(chunk_text) // 4),
        )
        for idx, chunk_text in enumerate(all_texts)
    ]
    return DocumentRecord(
        doc_id=doc_id,
        uri=uri or f"/tmp/{project_id}/{title}",
        source_harness=source_harness,
        media_type=media_type,
        project_id=project_id,
        canonical_tier=tier,
        title=title,
        l0_summary=content[:180],
        sha256=sha,
        updated_at=ts,
        chunks=chunks,
    )


def test_merkle_zero_recompute_on_second_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _ = _make_temp_config(tmp_path)
    embedder = FakeDeterministicEmbedder()
    vision = FakeVisionProvider()

    img_file = tmp_path / "system_diagram.png"
    img_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 128
    img_file.write_bytes(img_bytes)
    img_sha = hashlib.sha256(img_bytes).hexdigest()

    atlas = {
        "chittam": CanonicalRepo(
            project_id="chittam",
            family="chittam",
            name="chittam",
            path=str(tmp_path / "chittam"),
            tier=RepoTier.ACTIVE,
            last_commit_at="2026-11-22T12:00:00+00:00",
        )
    }

    doc_code = _make_doc("doc:code1", "store.py", "def hybrid_search(): return rrf(60)")
    doc_md = _make_doc(
        "doc:md1",
        "ARCHITECTURE.md",
        "Chittam uses SQLite WAL with FTS5 and numpy vectors.",
        media_type=MediaType.MARKDOWN,
    )

    state: dict[str, Any] = {"docs": [doc_code, doc_md], "include_image": True}

    def fake_collect(
        _cfg: AppConfig,
        _atlas: dict[str, CanonicalRepo],
        vision_fn=None,
        image_cache_lookup=None,
    ) -> tuple[list[DocumentRecord], SyncStats]:
        s = SyncStats()
        out = list(state["docs"])
        if state["include_image"]:
            cached = image_cache_lookup(img_sha) if image_cache_lookup else None
            if cached is not None:
                caption = cached
                s.images_cached += 1
            elif vision_fn is not None:
                caption = vision_fn(img_file, "architecture")
                s.images_captioned += 1
            else:
                caption = ""
            out.append(
                DocumentRecord(
                    doc_id="doc:img1",
                    uri=str(img_file),
                    source_harness="files",
                    media_type=MediaType.IMAGE,
                    project_id="chittam",
                    canonical_tier=RepoTier.ACTIVE,
                    title=img_file.name,
                    l0_summary=caption[:180],
                    sha256=img_sha,
                    updated_at="2026-11-22T12:00:00+00:00",
                    chunks=[
                        ChunkRecord(
                            chunk_id="doc:img1:chunk:0",
                            doc_id="doc:img1",
                            chunk_index=0,
                            contextual_header="[chittam | active | system_diagram.png]",
                            content=caption,
                            token_estimate=16,
                        )
                    ],
                )
            )
        s.scanned_docs = len(out)
        return out, s

    monkeypatch.setattr(indexer, "build_or_load_atlas", lambda c, refresh=False: atlas)
    monkeypatch.setattr(indexer, "collect_all_documents", fake_collect)
    monkeypatch.setattr(indexer, "build_embedding_provider", lambda c, use_external=False: embedder)
    monkeypatch.setattr(indexer, "build_vision_provider", lambda c, use_external=False: vision)

    # First sync: indexes all 3 docs, computes 3 embeddings, captions 1 image
    stats1 = indexer.sync_memory(cfg)
    assert stats1.scanned_docs == 3
    assert stats1.indexed_docs == 3
    assert stats1.skipped_unchanged == 0
    assert stats1.chunks_written == 3
    assert stats1.embeddings_computed == 3
    assert stats1.images_captioned == 1
    assert vision.calls == 1
    assert embedder.embed_calls == 1

    # Second sync: zero changes -> Merkle skips all 3, zero embeddings, zero vision calls
    embedder.embed_calls = 0
    embedder.embedded_texts.clear()
    stats2 = indexer.sync_memory(cfg)
    assert stats2.scanned_docs == 3
    assert stats2.indexed_docs == 0
    assert stats2.skipped_unchanged == 3
    assert stats2.chunks_written == 0
    assert stats2.embeddings_computed == 0
    assert stats2.images_captioned == 0
    assert stats2.images_cached == 1
    assert embedder.embed_calls == 0
    assert vision.calls == 1  # Still 1 (served from image_cache)

    # Third sync: tier and content change on doc_md -> only doc_md re-indexes and re-embeds
    state["docs"][1] = _make_doc(
        "doc:md1",
        "ARCHITECTURE.md",
        "Chittam uses SQLite WAL with FTS5 and USearch HNSW vectors.",
        media_type=MediaType.MARKDOWN,
        tier=RepoTier.REFERENCE,
    )
    stats3 = indexer.sync_memory(cfg)
    assert stats3.indexed_docs == 1
    assert stats3.skipped_unchanged == 2
    assert stats3.embeddings_computed == 1
    assert embedder.embed_calls == 1

    # Fourth sync: image removed -> stale doc pruned cleanly
    state["include_image"] = False
    stats4 = indexer.sync_memory(cfg)
    assert stats4.scanned_docs == 2
    with ChittamStore(cfg.db_path) as store:
        assert store.get_document("doc:img1") is None
        assert store.get_stats()["documents"] == 2


def test_active_repo_beats_archive_duplicate_and_compact_snippets(tmp_path: Path) -> None:
    embedder = FakeDeterministicEmbedder()
    long_body = (
        "Geoatlas geospatial routing engine handles canonical location indexing and spatial queries. "
        + ("Extra implementation padding line for snippet length clamping check. " * 20)
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    active_doc = _make_doc(
        "doc:geoatlas_active",
        "routing.py",
        long_body,
        uri=str(tmp_path / "Projects/geoatlas/routing.py"),
        project_id="geoatlas",
        tier=RepoTier.ACTIVE,
        updated_at=now_iso,
    )
    archive_doc = _make_doc(
        "doc:geoatlas_archive",
        "routing.py",
        long_body,
        uri=str(tmp_path / "Archive/geoatlas-old/routing.py"),
        project_id="geoatlas-old",
        tier=RepoTier.ARCHIVE,
        updated_at=now_iso,
    )

    with ChittamStore(tmp_path / "memory.db") as store:
        store.upsert_document(archive_doc)
        store.upsert_document(active_doc)
        vecs = embedder.embed_texts([
            f"{active_doc.chunks[0].contextual_header}\n{active_doc.chunks[0].content}",
            f"{archive_doc.chunks[0].contextual_header}\n{archive_doc.chunks[0].content}",
        ])
        store.upsert_chunk_embeddings(
            [active_doc.chunks[0].chunk_id, archive_doc.chunks[0].chunk_id],
            embedder.model_name,
            vecs,
        )

        q = "geoatlas geospatial routing engine"
        q_vec = embedder.embed_query(q)
        results = store.search(q, query_vec=q_vec, limit=5, include_archived=True)
        assert len(results) == 2
        assert results[0].doc_id == "doc:geoatlas_active"
        assert results[0].canonical_tier == RepoTier.ACTIVE
        assert results[1].doc_id == "doc:geoatlas_archive"
        assert results[0].score > results[1].score * 3.0  # 1.0 vs 0.25 tier weight
        assert all(len(r.snippet) <= 350 for r in results)

        active_only = store.search(q, query_vec=q_vec, limit=5, include_archived=False)
        assert [r.doc_id for r in active_only] == ["doc:geoatlas_active"]


def test_fts5_special_character_query_safety(tmp_path: Path) -> None:
    doc = _make_doc(
        "doc:notes",
        "desktop-uninstall-notes.md",
        "Notes for uninstalling Docker Desktop on macOS /home/dev/projects/chittam/store.py and C++ tools.",
        media_type=MediaType.MARKDOWN,
    )
    adversarial_queries = [
        "/home/dev/projects/chittam/store.py",
        "desktop-uninstall-notes.md",
        "foo.bar:baz",
        '"unclosed quote',
        'AND OR NOT NEAR/3 * ^ (parens) [brackets] {braces}',
        "--- /// ::: *** \"\"\" '''",
        "",
        "   ",
        "C++ macOS:v14.2/beta?foo=1&bar=2",
    ]
    with ChittamStore(tmp_path / "memory.db") as store:
        store.upsert_document(doc)
        for q in adversarial_queries:
            res = store.search(q, limit=5)
            assert isinstance(res, list)

        # Meaningful special-char path/hyphen queries still retrieve the doc cleanly
        hits = store.search("desktop-uninstall-notes.md /home/dev/projects/chittam/store.py")
        assert len(hits) == 1
        assert hits[0].doc_id == "doc:notes"


def test_bitemporal_record_decision_supersession(tmp_path: Path) -> None:
    with ChittamStore(tmp_path / "memory.db") as store:
        d1 = store.record_decision(
            project_id="chittam",
            topic="vector-storage",
            statement="Store embeddings in flat JSON files.",
            source_harness="codex",
        )
        d2 = store.record_decision(
            project_id="chittam",
            topic="vector-storage",
            statement="Store normalized float32 blobs in SQLite chunk_embeddings.",
            source_harness="antigravity",
        )
        d_other = store.record_decision(
            project_id="chittam",
            topic="vision-model",
            statement="Use local qwen3-vl:2b via Ollama.",
            source_harness="claude_code",
        )

        active = store.list_decisions(project_id="chittam", active_only=True)
        active_ids = {d.decision_id for d in active}
        assert active_ids == {d2.decision_id, d_other.decision_id}
        assert d1.decision_id not in active_ids

        history = store.list_decisions(project_id="chittam", active_only=False)
        by_id = {d.decision_id: d for d in history}
        assert len(by_id) == 3
        assert by_id[d1.decision_id].valid_to is not None
        assert by_id[d1.decision_id].superseded_by == d2.decision_id
        assert by_id[d2.decision_id].valid_to is None
        assert by_id[d2.decision_id].superseded_by is None
        assert by_id[d_other.decision_id].valid_to is None


def test_all_six_mcp_tools_and_cli_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, cfg_path = _make_temp_config(tmp_path)
    monkeypatch.setenv("CHITTAM_CONFIG", str(cfg_path))

    embedder = FakeDeterministicEmbedder()
    monkeypatch.setattr(mcp_server, "build_embedding_provider", lambda c, use_external=False: embedder)

    atlas = {
        "chittam": CanonicalRepo(
            project_id="chittam",
            family="chittam",
            name="chittam",
            path=str(tmp_path / "chittam"),
            tier=RepoTier.ACTIVE,
            last_commit_at="2026-11-22T20:00:00+00:00",
        ),
        "recipely-legacy": CanonicalRepo(
            project_id="recipely-legacy",
            family="recipely",
            name="recipely-legacy",
            path=str(tmp_path / "Archive/recipely"),
            tier=RepoTier.ARCHIVE,
            last_commit_at="2024-01-01T00:00:00+00:00",
        ),
    }
    save_atlas(atlas, cfg.atlas_path)

    doc_code = _make_doc(
        "doc:chittam_store",
        "store.py",
        "ChittamStore implements hybrid BM25 + cosine RRF search.",
        project_id="chittam",
        tier=RepoTier.ACTIVE,
        extra_chunks=[
            "Neighbor chunk one: bitemporal decision supersession.",
            "Neighbor chunk two: image_cache table deduplication.",
        ],
    )
    doc_thread_codex = _make_doc(
        "doc:codex_thread",
        "Recipely architecture planning session",
        "Discussed Recipely recipe scaling and offline sync architecture.",
        project_id="recipely",
        tier=RepoTier.ACTIVE,
        media_type=MediaType.TRANSCRIPT,
        source_harness="codex",
    )
    doc_thread_claude = _make_doc(
        "doc:claude_thread",
        "Claude Code memory debugging",
        "Debugged Claude Code transcript JSONL ingestion.",
        project_id="chittam",
        tier=RepoTier.ACTIVE,
        media_type=MediaType.TRANSCRIPT,
        source_harness="claude_code",
    )

    with ChittamStore(cfg.db_path) as store:
        for d in (doc_code, doc_thread_codex, doc_thread_claude):
            store.upsert_document(d)
            vecs = embedder.embed_texts([f"{c.contextual_header}\n{c.content}" for c in d.chunks])
            store.upsert_chunk_embeddings([c.chunk_id for c in d.chunks], embedder.model_name, vecs)

    # Verify all 6 tools are registered on MCPServer
    tools = asyncio.run(mcp_server.mcp.list_tools())
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "memory_search",
        "memory_get_chunk",
        "project_atlas",
        "recall_threads",
        "record_decision",
        "index_status",
    }

    # 1. memory_search
    search_out = mcp_server.memory_search("hybrid BM25 RRF", limit=5)
    assert search_out["count"] >= 1
    assert search_out["vector_used"] is True
    assert search_out["results"][0]["doc_id"] == "doc:chittam_store"
    assert len(search_out["results"][0]["snippet"]) <= 350

    # 2. memory_get_chunk (middle chunk with prev & next neighbors)
    chunk_out = mcp_server.memory_get_chunk("doc:chittam_store:chunk:1", include_neighbors=True)
    assert chunk_out["found"] is True
    assert "bitemporal decision supersession" in chunk_out["content"]
    assert chunk_out["neighbors"]["prev"]["chunk_index"] == 0
    assert chunk_out["neighbors"]["next"]["chunk_index"] == 2

    # 3. project_atlas
    atlas_active = mcp_server.project_atlas(tier="active")
    assert atlas_active["count"] == 1
    assert atlas_active["projects"][0]["project_id"] == "chittam"
    atlas_recipely = mcp_server.project_atlas(family="recipely")
    assert atlas_recipely["count"] == 1
    assert atlas_recipely["projects"][0]["tier"] == "archive"

    # 4. recall_threads
    codex_threads = mcp_server.recall_threads("Recipely recipe scaling", harness="codex")
    assert codex_threads["count"] == 1
    assert codex_threads["threads"][0]["source_harness"] == "codex"
    claude_threads = mcp_server.recall_threads("transcript JSONL", harness="claude")
    assert claude_threads["count"] == 1
    assert claude_threads["threads"][0]["source_harness"] == "claude_code"
    all_threads = mcp_server.recall_threads("Recipely transcript", harness=None)
    assert all_threads["count"] == 2
    assert {t["source_harness"] for t in all_threads["threads"]} == {"codex", "claude_code"}

    # 5. record_decision via call_tool to exercise MCP envelope end-to-end
    dec_res = asyncio.run(
        mcp_server.mcp.call_tool(
            "record_decision",
            {
                "project_id": "chittam",
                "topic": "mcp-transport",
                "statement": "Serve pull-only tools over stdio.",
                "source_harness": "mcp",
            },
        )
    )
    assert dec_res.is_error is False
    assert dec_res.structured_content["status"] == "recorded"
    assert dec_res.structured_content["decision"]["topic"] == "mcp-transport"

    # 6. index_status
    status_out = mcp_server.index_status()
    assert status_out["status"] == "ok"
    assert status_out["db_exists"] is True
    assert status_out["atlas_exists"] is True
    assert status_out["atlas_projects"] == 2
    assert status_out["stats"]["documents"] == 3
    assert status_out["stats"]["active_decisions"] == 1

    # CLI smoke checks on isolated config
    runner = CliRunner()
    res_status = runner.invoke(cli_app, ["status", "--config", str(cfg_path), "--json"])
    assert res_status.exit_code == 0, res_status.output
    status_json = json.loads(res_status.stdout)
    assert status_json["stats"]["documents"] == 3

    res_search = runner.invoke(
        cli_app, ["search", "hybrid BM25", "--config", str(cfg_path), "--no-vector", "--json"]
    )
    assert res_search.exit_code == 0, res_search.output
    assert json.loads(res_search.stdout)[0]["doc_id"] == "doc:chittam_store"

    res_decide = runner.invoke(
        cli_app,
        ["decide", "chittam", "mcp-transport", "Use MCPServer v2 stdio.", "--config", str(cfg_path)],
    )
    assert res_decide.exit_code == 0, res_decide.output

    res_decs = runner.invoke(
        cli_app, ["decisions", "--project", "chittam", "--all", "--config", str(cfg_path), "--json"]
    )
    assert res_decs.exit_code == 0, res_decs.output
    decs_list = json.loads(res_decs.stdout)
    assert len(decs_list) == 2


def test_chunk_shrink_cleanup_decision_search_and_leann_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _ = _make_temp_config(tmp_path)
    cfg.enable_leann_bridge = True
    embedder = FakeDeterministicEmbedder()

    with ChittamStore(cfg.db_path) as store:
        d_three = _make_doc(
            "doc:shrink",
            "shrink.py",
            "chunk zero alpha",
            extra_chunks=["chunk one beta", "chunk two gamma"],
        )
        store.upsert_document(d_three)
        vecs = embedder.embed_texts([f"{c.contextual_header}\n{c.content}" for c in d_three.chunks])
        store.upsert_chunk_embeddings([c.chunk_id for c in d_three.chunks], embedder.model_name, vecs)
        assert store.get_stats()["chunks"] == 3
        assert store.get_stats()["embeddings"] == 3

        # Re-upsert same doc_id with only 1 chunk: stale chunks 1 and 2 + embeddings + FTS must be removed
        d_one = _make_doc("doc:shrink", "shrink.py", "chunk zero replaced")
        store.upsert_document(d_one)
        assert store.get_stats()["chunks"] == 1
        assert store.get_stats()["embeddings"] == 0
        assert store.search("gamma") == []
        assert store.get_chunk("doc:shrink:chunk:2") is None

        # Decisions should be searchable via search(media_type="decision") and inspectable via get_chunk
        dec = store.record_decision("chittam", "cache-policy", "Keep search snippets under 350 chars.")
        dec_hits = store.search("snippets 350", media_type="decision")
        assert len(dec_hits) == 1
        assert dec_hits[0].media_type == MediaType.DECISION
        dec_chunk = store.get_chunk(dec_hits[0].chunk_id, include_neighbors=True)
        assert dec_chunk is not None
        assert dec_chunk["metadata"]["decision_id"] == dec.decision_id

        # LEANN export helper exports corpus JSONL when leann CLI exists and threshold is met
        fake_leann = tmp_path / "leann"
        fake_leann.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        monkeypatch.setattr(indexer, "_find_leann_cli", lambda: str(fake_leann))
        export_res = indexer.build_leann_export(cfg, store, threshold=1, run_build=False)
        assert export_res["exported"] is True
        corpus_path = Path(export_res["corpus_path"])
        assert corpus_path.exists()
        lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == "doc:shrink:chunk:0"


def test_store_busy_timeout_wal_reopen_and_upsert_replace(tmp_path: Path) -> None:
    db_path = tmp_path / "lock_check.db"
    with ChittamStore(db_path) as store1:
        busy_ms = int(store1.conn.execute("PRAGMA busy_timeout;").fetchone()[0])
        jm = str(store1.conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()
        assert busy_ms == 30000
        assert jm == "wal"

        doc_v1 = _make_doc("doc:reopen", "alpha.py", "alpha_token_v1 old_body")
        store1.upsert_document(doc_v1)

    with ChittamStore(db_path) as store2:
        busy_ms2 = int(store2.conn.execute("PRAGMA busy_timeout;").fetchone()[0])
        jm2 = str(store2.conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()
        assert busy_ms2 == 30000
        assert jm2 == "wal"

        doc_v2 = _make_doc("doc:reopen", "alpha.py", "beta_token_v2 replacement_body")
        store2.upsert_document(doc_v2)

        assert store2.search("alpha_token_v1") == []
        hits = store2.search("beta_token_v2")
        assert len(hits) == 1
        assert hits[0].doc_id == "doc:reopen"
        assert store2.get_stats()["chunks"] == 1


