"""Offline tests for Chittam v0.2 Core Engine features."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from chittam import indexer, mcp_server
from chittam.adapters import files as files_adapter
from chittam.atlas import build_or_load_atlas, save_atlas
from chittam.cli import app as cli_app
from chittam.doctor import run_doctor
from chittam.models import AppConfig, CanonicalRepo, HarnessesConfig, MediaType, RepoTier
from chittam.providers import (
    ConsentRequiredError,
    ExternalOpenAICompatibleProvider,
    FlashRankRerankProvider,
    OllamaEmbeddingProvider,
    build_query_embedding_provider,
)
from chittam.store import ChittamStore
from chittam.vector_index import (
    NumpyVectorIndex,
    USearchVectorIndex,
    quantize_vector,
    dequantize_vector,
)
from chittam.watcher import build_watch_roots, should_watch_path


def _cfg_in(tmp_path: Path, **overrides: Any) -> tuple[AppConfig, Path]:
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    harnesses = HarnessesConfig(
        antigravity_mcp_path=str(tmp_path / "no-ag-mcp.json"),
        codex_config_path=str(tmp_path / "no-codex.toml"),
        claude_config_path=str(tmp_path / "no-claude.json"),
        antigravity_brain_dir=str(tmp_path / "no-ag-brain"),
        codex_home_dir=str(tmp_path / "no-codex-home"),
        claude_projects_dir=str(tmp_path / "no-claude-proj"),
        claude_history_path=str(tmp_path / "no-claude" / "history.jsonl"),
    )
    cfg = AppConfig(
        data_dir=str(data_dir),
        config_dir=str(config_dir),
        harnesses=harnesses,
        enable_leann_bridge=False,
        **overrides,
    )
    cfg_path = config_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    return cfg, cfg_path


class CountingConceptEmbedder:
    """Offline deterministic semantic embedder with synonym projection for zero-overlap concept tests."""

    model_name = "counting-embed-256d"
    dim = 256

    def __init__(self) -> None:
        self.embed_calls = 0
        self.embedded_texts: list[str] = []

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        low = text.lower()
        # Semantic axes shared between conceptual terms and lexical docs
        if any(w in low for w in ("sqlite", "database", "storage", "persistence", "wal")):
            vec[1] += 2.0
        if any(w in low for w in ("apple", "silicon", "mac", "local", "privacy", "offline")):
            vec[2] += 2.0
        if any(w in low for w in ("architecture", "graph", "diagram", "visual")):
            vec[3] += 2.0
        if any(w in low for w in ("decision", "policy", "protocol")):
            vec[4] += 1.5
        for tok in low.split():
            bucket = 8 + (int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16) % (self.dim - 8))
            vec[bucket] += 1.0
        norm = float(np.linalg.norm(vec))
        return (vec / norm) if norm > 0 else vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.embed_calls += 1
        self.embedded_texts.extend(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self._encode(t) for t in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode(query)


def test_usearch_and_numpy_quantized_accuracy_and_reload(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(150, 256)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    q = raw[0] + 0.04 * rng.normal(size=(256,)).astype(np.float32)
    q /= np.linalg.norm(q)

    # int8 roundtrip & similarity correlation
    blob = quantize_vector(raw[0], dtype="i8")
    assert len(blob) == 256
    rec = dequantize_vector(blob, dim=256, dtype="i8")
    assert float(rec @ raw[0]) > 0.995

    idx_path = tmp_path / "vectors.usearch"
    writer = USearchVectorIndex(idx_path, dim=256, dtype="i8")
    keys = np.arange(1, 151, dtype=np.uint64)
    writer.add(keys, raw)
    writer.save()
    assert writer.count() == 150

    reader = USearchVectorIndex(idx_path, dim=256, dtype="i8")
    hits = reader.search(q, k=20)
    assert hits[0][0] == 1
    exact_sims = raw @ q
    approx_sims = [sim for _, sim in hits]
    true_subset = [float(exact_sims[k - 1]) for k, _ in hits]
    assert float(np.corrcoef(approx_sims, true_subset)[0, 1]) > 0.99

    # External writer adds another key -> reader reload_if_changed picks it up
    extra_vec = -raw[0]  # negative direction; pick query near extra_vec2 instead
    extra_vec2 = rng.normal(size=(256,)).astype(np.float32)
    extra_vec2 /= np.linalg.norm(extra_vec2)
    writer.add([999], extra_vec2)
    writer.save()
    assert reader.reload_if_changed() is True
    assert reader.count() == 151
    assert reader.search(extra_vec2, k=3)[0][0] == 999

    # Remove + compact
    writer.remove([999])
    writer.compact()
    assert reader.reload_if_changed() is True
    assert reader.count() == 150

    # NumpyVectorIndex parity check
    np_idx = NumpyVectorIndex(dim=256, dtype="i8")
    np_idx.add(keys, raw)
    assert np_idx.search(q, k=5)[0][0] == 1


def test_content_hash_dedup_and_tiered_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_a = tmp_path / "projects" / "repo-alpha"
    root_b = tmp_path / "projects" / "repo-beta"
    root_old = tmp_path / "archive" / "repo-legacy"
    for d in (root_a, root_b, root_old):
        d.mkdir(parents=True, exist_ok=True)

    shared_code = (
        "def canonical_router(query: str) -> str:\n"
        "    \"\"\"Unique shared router across cloned repositories.\"\"\"\n"
        "    return query.strip().lower()\n"
    )
    archive_only_code = (
        "def deprecated_legacy_helper() -> int:\n"
        "    \"\"\"Legacy archive-only symbol unique_legacy_marker.\"\"\"\n"
        "    return 42\n"
    )
    (root_a / "router.py").write_text(shared_code, encoding="utf-8")
    (root_b / "router.py").write_text(shared_code, encoding="utf-8")
    (root_old / "legacy.py").write_text(archive_only_code, encoding="utf-8")

    cfg, _ = _cfg_in(tmp_path)
    cfg.privacy.scan_roots = [str(tmp_path / "projects"), str(tmp_path / "archive")]
    atlas = {
        "repo-alpha": CanonicalRepo(
            project_id="repo-alpha",
            family="alpha",
            name="repo-alpha",
            path=str(root_a),
            tier=RepoTier.ACTIVE,
        ),
        "repo-beta": CanonicalRepo(
            project_id="repo-beta",
            family="beta",
            name="repo-beta",
            path=str(root_b),
            tier=RepoTier.REFERENCE,
        ),
        "repo-legacy": CanonicalRepo(
            project_id="repo-legacy",
            family="legacy",
            name="repo-legacy",
            path=str(root_old),
            tier=RepoTier.ARCHIVE,
        ),
    }
    save_atlas(atlas, cfg.atlas_path)

    embedder = CountingConceptEmbedder()
    monkeypatch.setattr(indexer, "build_or_load_atlas", lambda c, refresh=False: atlas)
    monkeypatch.setattr(indexer, "build_embedding_provider", lambda c, use_external=False: embedder)

    stats = indexer.sync_memory(cfg, skip_vision=True)
    assert stats.scanned_docs == 3
    # shared_code in alpha + beta embeds ONLY ONCE via content_sha256 dedup; archive legacy.py is skipped!
    assert embedder.embed_calls == 1
    assert stats.embeddings_computed == 1
    assert stats.embeddings_deduped == 1

    with ChittamStore(cfg) as store:
        st = store.get_stats()
        assert st["embeddings"] == 2
        assert st["unique_embeddings"] == 1
        assert cfg.vector_index_path.exists()

        # Both active and reference clones are semantically searchable via USearch
        hits = store.search(
            "canonical_router",
            query_vec=embedder.embed_query("canonical_router"),
            limit=5,
        )
        assert {h.project_id for h in hits} == {"repo-alpha", "repo-beta"}
        assert hits[0].project_id == "repo-alpha"  # ACTIVE beats REFERENCE

        # Archive repo still works via BM25 even though skipped from vector embedding
        arch_hits = store.search("unique_legacy_marker", query_vec=None, limit=5, include_archived=True)
        assert len(arch_hits) == 1
        assert arch_hits[0].project_id == "repo-legacy"
        assert arch_hits[0].vector_rank is None

    # Opting ARCHIVE into embed_tiers embeds legacy.py on next sync (and shares identical existing hashes)
    cfg.privacy.embed_tiers = [RepoTier.ACTIVE, RepoTier.REFERENCE, RepoTier.ARCHIVE]
    stats2 = indexer.sync_memory(cfg, skip_vision=True)
    assert stats2.embeddings_computed == 1
    with ChittamStore(cfg) as store:
        assert store.get_stats()["embeddings"] == 3


def test_v1_in_place_auto_migration_rebuild_and_compact(tmp_path: Path) -> None:
    cfg, _ = _cfg_in(tmp_path)
    cfg.local_provider.matryoshka_dim = 256
    cfg.local_provider.vector_dtype = "i8"

    # Build a synthetic v0.1 database using raw sqlite3 + legacy 768-d float32 chunk_embeddings
    rng = np.random.default_rng(19)
    vec768_a = rng.normal(size=(768,)).astype(np.float32)
    vec768_a /= np.linalg.norm(vec768_a)
    vec768_b = rng.normal(size=(768,)).astype(np.float32)
    vec768_b /= np.linalg.norm(vec768_b)

    with ChittamStore(cfg.db_path) as init_store:
        init_store.cache_image_caption("imgsha1", "qwen3-vl:2b", "Architecture screenshot caption")
        init_store.record_decision("chittam", "engine", "Keep SQLite WAL + USearch")

    conn = sqlite3.connect(str(cfg.db_path))
    with conn:
        for idx, (tier, vec) in enumerate((("active", vec768_a), ("active", vec768_a), ("archive", vec768_b))):
            doc_id = f"doc:v1:{idx}"
            cid = f"{doc_id}#c0"
            text = "identical content across clones" if idx < 2 else "archived only text"
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, uri, source_harness, media_type, project_id, canonical_tier,
                    title, l0_summary, sha256, updated_at, metadata_json
                ) VALUES (?, ?, 'files', 'markdown', ?, ?, ?, ?, ?, '2026-11-22T00:00:00Z', '{}')
                """,
                (doc_id, f"/tmp/{doc_id}.md", f"proj{idx}", tier, f"{doc_id}.md", text, f"sha{idx}"),
            )
            conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, chunk_index, contextual_header, content, token_estimate, metadata_json
                ) VALUES (?, ?, 0, '[ctx]', ?, 10, '{}')
                """,
                (cid, doc_id, text),
            )
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, doc_id, contextual_header, content, title, project_id, source_harness) VALUES (?, ?, '[ctx]', ?, ?, ?, 'files')",
                (cid, doc_id, text, f"{doc_id}.md", f"proj{idx}"),
            )
            conn.execute(
                "INSERT INTO chunk_embeddings (chunk_id, model, dim, vector) VALUES (?, 'embeddinggemma:latest', 768, ?)",
                (cid, vec.tobytes()),
            )
    conn.close()

    # Opening ChittamStore(cfg) auto-migrates 768-d float32 -> 256-d int8 USearch, dedupes, preserves captions/decisions
    with ChittamStore(cfg) as store:
        st = store.get_stats()
        assert st["legacy_embeddings"] == 0
        assert st["vector_dim"] == 256
        assert st["vector_dtype"] == "i8"
        assert st["embeddings"] == 2  # 2 active rows (archive skipped per policy)
        assert st["unique_embeddings"] == 1  # deduplicated identical content!
        assert st["cached_images"] == 1
        assert st["active_decisions"] == 1
        assert store.get_cached_image_caption("imgsha1") == "Architecture screenshot caption"
        assert cfg.vector_index_path.exists()

        # Verify sliced 256-d query hits migrated docs cleanly
        q256 = vec768_a[:256] / np.linalg.norm(vec768_a[:256])
        hits = store.search("nonexistent_keyword", query_vec=q256, limit=5)
        assert len(hits) == 2

        # Rebuild from SQLite after deleting vectors.usearch
        cfg.vector_index_path.unlink()
        assert not cfg.vector_index_path.exists()
        rebuilt = store.rebuild_vector_index()
        assert rebuilt["rebuilt"] == 2
        assert cfg.vector_index_path.exists()

        # Deleting one doc & compacting shrinks cleanly
        store.delete_document("doc:v1:0")
        store.delete_document("doc:v1:1")
        comp = store.compact()
        assert comp["orphans_deleted"] == 1
        assert comp["rebuilt_vectors"] == 0


def test_stat_prefilter_skips_unchanged_and_handles_narrow_filter_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = tmp_path / "docs_root"
    proj.mkdir(parents=True, exist_ok=True)
    note = proj / "design.md"
    note.write_text("# Initial Design\nOriginal version.", encoding="utf-8")

    cfg, _ = _cfg_in(tmp_path)
    cfg.privacy.scan_roots = [str(proj)]

    read_calls = 0
    orig_md = files_adapter.MEDIA_HANDLERS[MediaType.MARKDOWN].handler

    def counting_md(path: Path, ctx: files_adapter.FileContext):
        nonlocal read_calls
        read_calls += 1
        return orig_md(path, ctx)

    monkeypatch.setattr(files_adapter.MEDIA_HANDLERS[MediaType.MARKDOWN], "handler", counting_md)

    stats1 = indexer.sync_memory(cfg, skip_embeddings=True, skip_vision=True)
    assert stats1.indexed_docs == 1
    assert read_calls == 1

    # Second sync with untouched file: stat() fast-path skips reading completely!
    read_calls = 0
    stats2 = indexer.sync_memory(cfg, skip_embeddings=True, skip_vision=True)
    assert stats2.indexed_docs == 0
    assert stats2.stat_skipped == 1
    assert read_calls == 0

    # Edit file with different content/size -> stat() sees change and re-indexes
    note.write_text("# Updated Design\nSecond version with more words.", encoding="utf-8")
    stats3 = indexer.sync_memory(cfg, skip_embeddings=True, skip_vision=True)
    assert stats3.indexed_docs == 1
    assert stats3.stat_skipped == 0
    assert read_calls == 1

    # Verify narrow filter fallback: add 280 unrelated code chunks + 1 image chunk with USearch oversample edge
    embedder = CountingConceptEmbedder()
    with ChittamStore(cfg) as store:
        docs_batch = []
        vecs_list = []
        chunk_ids = []
        for i in range(280):
            d_id = f"doc:filler:{i}"
            c_id = f"{d_id}#c0"
            from chittam.models import ChunkRecord, DocumentRecord

            content = f"def filler_{i}(): return 'architecture graph diagram' * {i % 3 + 1}"
            docs_batch.append(
                DocumentRecord(
                    doc_id=d_id,
                    uri=f"/tmp/filler_{i}.py",
                    source_harness="files",
                    media_type=MediaType.CODE,
                    project_id="filler",
                    canonical_tier=RepoTier.ACTIVE,
                    title=f"filler_{i}.py",
                    l0_summary=content,
                    sha256=f"fsha{i}",
                    updated_at="2026-11-22T00:00:00Z",
                    chunks=[
                        ChunkRecord(
                            chunk_id=c_id,
                            doc_id=d_id,
                            chunk_index=0,
                            contextual_header="[filler | active]",
                            content=content,
                            token_estimate=10,
                        )
                    ],
                )
            )
            store.upsert_document(docs_batch[-1], _save_vec=False)
            chunk_ids.append(c_id)
            vecs_list.append(embedder.embed_query("unrelated code filler item"))
        # Add 1 image matching "architecture graph diagram" closely
        from chittam.models import ChunkRecord, DocumentRecord

        img_doc = DocumentRecord(
            doc_id="doc:img_arch",
            uri="/tmp/arch_shot.png",
            source_harness="files",
            media_type=MediaType.IMAGE,
            project_id="chittam",
            canonical_tier=RepoTier.ACTIVE,
            title="arch_shot.png",
            l0_summary="visual architecture diagram",
            sha256="img_arch_sha",
            updated_at="2026-11-22T00:00:00Z",
            chunks=[
                ChunkRecord(
                    chunk_id="doc:img_arch#c0",
                    doc_id="doc:img_arch",
                    chunk_index=0,
                    contextual_header="[chittam | active | arch_shot.png]",
                    content="Image: arch_shot.png\nvisual architecture diagram",
                    token_estimate=10,
                )
            ],
        )
        store.upsert_document(img_doc, _save_vec=False)
        store.flush_vector_index()
        chunk_ids.append("doc:img_arch#c0")
        vecs_list.append(embedder.embed_query("architecture graph diagram visual"))
        store.upsert_chunk_embeddings(chunk_ids, embedder.model_name, np.vstack(vecs_list))

        img_hits = store.search(
            "only_semantic_concept_query",
            query_vec=embedder.embed_query("architecture graph diagram visual"),
            limit=3,
            media_type=MediaType.IMAGE,
        )
        assert len(img_hits) == 1
        assert img_hits[0].doc_id == "doc:img_arch"


def test_semantic_decision_and_reranker_preserves_tier_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _ = _cfg_in(tmp_path)
    embedder = CountingConceptEmbedder()

    class FakeRanker:
        def rerank(self, req: Any) -> list[dict[str, Any]]:
            # Intentionally prefer passage mentioning "archive-bait" slightly over plain active doc,
            # so we verify Chittam's tier weight still keeps ACTIVE ahead of identical ARCHIVE copies!
            scored = []
            for p in req.passages:
                txt = p["text"].lower()
                sc = 0.99 if "preferred" in txt else 0.25
                scored.append({"id": p["id"], "text": p["text"], "score": sc})
            return scored

    reranker = FlashRankRerankProvider(ranker=FakeRanker())

    with ChittamStore(cfg) as store:
        dec = store.record_decision(
            project_id="chittam",
            topic="persistence-engine",
            statement="Store metadata in SQLite WAL on Apple Silicon macOS for privacy.",
            source_harness="antigravity",
        )
        dec_vec = embedder.embed_query(f"[{dec.project_id}] {dec.topic}: {dec.statement}")
        store.upsert_decision_embedding(
            dec.decision_id,
            f"[{dec.project_id}] {dec.topic}: {dec.statement}",
            embedder.model_name,
            dec_vec,
        )

        # Query has ZERO overlapping words with statement, only conceptual words ("database local offline")
        q = "database local offline"
        hits = store.search(
            q,
            query_vec=embedder.embed_query(q),
            limit=5,
            media_type=MediaType.DECISION,
        )
        assert len(hits) == 1
        assert hits[0].doc_id == f"decision:{dec.decision_id}"
        assert hits[0].vector_rank == 1

        # Add Active vs Archive duplicate and run reranker
        from chittam.models import ChunkRecord, DocumentRecord

        for tier, prefix in ((RepoTier.ARCHIVE, "arch"), (RepoTier.ACTIVE, "act")):
            text = "Preferred clone content discussing database local offline design."
            d = DocumentRecord(
                doc_id=f"doc:{prefix}",
                uri=f"/tmp/{prefix}/note.md",
                source_harness="files",
                media_type=MediaType.MARKDOWN,
                project_id=f"proj-{prefix}",
                canonical_tier=tier,
                title="note.md",
                l0_summary=text,
                sha256=f"sha_{prefix}",
                updated_at="2026-11-22T00:00:00Z",
                chunks=[
                    ChunkRecord(
                        chunk_id=f"doc:{prefix}#c0",
                        doc_id=f"doc:{prefix}",
                        chunk_index=0,
                        contextual_header=f"[{prefix}]",
                        content=text,
                        token_estimate=10,
                    )
                ],
            )
            store.upsert_document(d)
            store.upsert_chunk_embeddings([f"doc:{prefix}#c0"], embedder.model_name, embedder.embed_query(text))

        # Unrelated cross-corpus search must NOT let an orthogonal active decision hijack rank #1
        unrelated_q = "visual architecture graph diagram"
        img_or_doc_hits = store.search(
            unrelated_q,
            query_vec=embedder.embed_query(unrelated_q),
            limit=5,
        )
        assert not img_or_doc_hits or img_or_doc_hits[0].media_type != MediaType.DECISION

        reranked = store.search(
            "database local offline preferred",
            query_vec=embedder.embed_query("database local offline preferred"),
            limit=5,
            media_type=MediaType.MARKDOWN,
            reranker=reranker,
        )
        assert [r.project_id for r in reranked] == ["proj-act", "proj-arch"]
        assert "rerank_score" in reranked[0].metadata

        # Failing reranker falls back cleanly without breaking search
        class BrokenRanker:
            def rerank(self, req: Any) -> list[dict[str, Any]]:
                raise RuntimeError("boom")

        fallback = store.search(
            "database",
            query_vec=embedder.embed_query("database"),
            limit=5,
            reranker=FlashRankRerankProvider(ranker=BrokenRanker()),
        )
        assert len(fallback) >= 1


def test_query_model_consistency_and_watcher_sync_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = tmp_path / "workspace" / "app"
    proj.mkdir(parents=True, exist_ok=True)
    ag_dir = tmp_path / ".gemini" / "antigravity" / "brain"
    ag_dir.mkdir(parents=True, exist_ok=True)

    cfg, cfg_path = _cfg_in(tmp_path)
    cfg.privacy.scan_roots = [str(tmp_path / "workspace")]
    cfg.harnesses.antigravity_brain_dir = str(ag_dir)
    cfg.harnesses.codex_home_dir = str(tmp_path / "no-codex")
    cfg.harnesses.claude_projects_dir = str(tmp_path / "no-claude")
    cfg.harnesses.claude_history_path = str(tmp_path / "no-claude-hist.jsonl")

    # Watch filter & roots checks
    roots = build_watch_roots(cfg)
    assert (tmp_path / "workspace").resolve() in roots
    assert ag_dir.resolve() in roots
    assert should_watch_path(proj / "main.py", cfg) is True
    assert should_watch_path(ag_dir / "conv1" / "notes.md", cfg) is True
    assert should_watch_path(proj / ".env", cfg) is False
    assert should_watch_path(proj / ".git" / "config", cfg) is False
    assert should_watch_path(proj / "node_modules" / "x.js", cfg) is False
    assert should_watch_path(tmp_path / "workspace" / "tax" / "2025.pdf", cfg) is False

    # Incremental watch delta sync: add file -> edit -> delete
    f = proj / "widget.py"
    f.write_text("def render_widget(): return 'hello_watch_v1'\n", encoding="utf-8")
    embedder = CountingConceptEmbedder()
    monkeypatch.setattr(indexer, "build_embedding_provider", lambda c, use_external=False: embedder)

    st_add = indexer.sync_changed_paths(cfg, [f], skip_vision=True)
    assert st_add.indexed_docs == 1
    with ChittamStore(cfg) as store:
        assert len(store.search("hello_watch_v1")) == 1

    f.write_text("def render_widget(): return 'hello_watch_v2'\n", encoding="utf-8")
    st_edit = indexer.sync_changed_paths(cfg, [f], skip_vision=True)
    assert st_edit.indexed_docs == 1
    with ChittamStore(cfg) as store:
        assert store.search("hello_watch_v1") == []
        assert len(store.search("hello_watch_v2")) == 1

    f.unlink()
    indexer.sync_changed_paths(cfg, [f], skip_vision=True)
    with ChittamStore(cfg) as store:
        assert store.search("hello_watch_v2") == []

    # Query-model consistency tests (respect active_embedding_source and active_embedding_dim)
    cfg.external_provider.enabled = True
    cfg.external_provider.user_consented = True
    cfg.external_provider.provider_type = "openai_compatible"
    cfg.external_provider.api_key_env = "CHITTAM_TEST_API_KEY"
    monkeypatch.setenv("CHITTAM_TEST_API_KEY", "sk-test-token")

    ext_q = build_query_embedding_provider(
        cfg,
        index_meta={"active_embedding_source": "external", "active_embedding_dim": "128"},
    )
    assert isinstance(ext_q, ExternalOpenAICompatibleProvider)
    assert ext_q.matryoshka_dim == 128

    cfg.external_provider.user_consented = False
    loc_q = build_query_embedding_provider(
        cfg,
        index_meta={"active_embedding_source": "external", "active_embedding_dim": "256"},
    )
    assert isinstance(loc_q, OllamaEmbeddingProvider)
    assert loc_q.matryoshka_dim == 256

    # Doctor & CLI checks (compact, reindex-vectors, doctor)
    monkeypatch.setenv("CHITTAM_CONFIG", str(cfg_path))
    doc_rep = run_doctor(cfg, check_ollama=False)
    assert doc_rep["healthy"] is True
    assert {c["name"] for c in doc_rep["checks"]} >= {
        "config",
        "database",
        "vector_index",
        "reranker",
        "scan_roots",
        "harnesses",
    }

    runner = CliRunner()
    res_compact = runner.invoke(cli_app, ["compact", "--config", str(cfg_path), "--json"])
    assert res_compact.exit_code == 0, res_compact.output
    assert "db_size_after" in json.loads(res_compact.stdout)

    res_reindex = runner.invoke(cli_app, ["reindex-vectors", "--config", str(cfg_path), "--json"])
    assert res_reindex.exit_code == 0, res_reindex.output
    assert "rebuilt" in json.loads(res_reindex.stdout)
