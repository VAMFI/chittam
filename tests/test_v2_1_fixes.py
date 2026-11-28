"""Regression tests for Chittam v0.2.1 reliability, ranking, Antigravity adapter, and MCP fixes."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import yaml

from chittam import indexer, mcp_server
from chittam.adapters import load_antigravity_docs
from chittam.adapters.files import UNREADABLE_IMAGE_SENTINEL, load_file_docs
from chittam.atlas import is_atlas_stale, match_repo_from_text, resolve_project_filter, save_atlas
from chittam.models import (
    AppConfig,
    CanonicalRepo,
    ChunkRecord,
    DocumentRecord,
    HarnessesConfig,
    MediaType,
    PrivacyPolicy,
    RepoTier,
)
from chittam.providers import OllamaEmbeddingProvider
from chittam.rules import install_rules, uninstall_rules
from chittam.store import ChittamStore, file_lock
from chittam.vector_index import USearchVectorIndex


def _isolated_cfg(tmp_path: Path, scan_roots: list[str] | None = None) -> tuple[AppConfig, Path]:
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig(
        data_dir=str(data_dir),
        config_dir=str(config_dir),
        enable_leann_bridge=False,
        privacy=PrivacyPolicy(scan_roots=scan_roots or []),
        harnesses=HarnessesConfig(
            antigravity_mcp_path=str(tmp_path / "no-ag.json"),
            codex_config_path=str(tmp_path / "no-codex.toml"),
            claude_config_path=str(tmp_path / "no-claude.json"),
            antigravity_rules_path=str(tmp_path / "no-GEMINI.md"),
            antigravity_rules_dir=str(tmp_path / "no-ag-rules"),
            claude_rules_path=str(tmp_path / "no-CLAUDE.md"),
            codex_rules_path=str(tmp_path / "no-AGENTS.md"),
            antigravity_brain_dir=str(tmp_path / "no-ag-brain"),
            codex_home_dir=str(tmp_path / "no-codex-home"),
            claude_projects_dir=str(tmp_path / "no-claude-proj"),
            claude_history_path=str(tmp_path / "no-claude-hist.jsonl"),
        ),
    )
    cfg_path = config_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    return cfg, cfg_path


def _init_git_repo(repo_dir: Path, days_ago: int = 1) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo_dir)], check=True, capture_output=True)
    (repo_dir / "README.md").write_text(f"# {repo_dir.name}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_dir), "add", "README.md"], check=True, capture_output=True)
    iso = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_AUTHOR_DATE": iso,
        "GIT_COMMITTER_DATE": iso,
    }
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"], check=True, capture_output=True, env=env)


def test_tcc_unreadable_root_preserves_existing_docs_while_pruning_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_a = tmp_path / "Downloads"
    root_b = tmp_path / "Projects"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "saved_spec.md").write_text("# Saved Spec\nKeep me even if TCC blocks root_a.", encoding="utf-8")
    keep_b = root_b / "keep.md"
    del_b = root_b / "delete_me.md"
    keep_b.write_text("# Keep\nStill here.", encoding="utf-8")
    del_b.write_text("# Delete\nWill be removed.", encoding="utf-8")

    cfg, _ = _isolated_cfg(tmp_path, scan_roots=[str(root_a), str(root_b)])
    save_atlas({}, cfg.atlas_path)

    st1 = indexer.sync_memory(cfg, skip_embeddings=True, skip_vision=True)
    assert st1.indexed_docs == 3

    # Remove del_b and simulate macOS TCC PermissionError when scanning root_a
    del_b.unlink()
    real_scandir = os.scandir

    def fake_scandir(path: Any = "."):
        if Path(path).resolve(strict=False) == root_a.resolve(strict=False):
            raise PermissionError(1, "Operation not permitted", str(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", fake_scandir)
    st2 = indexer.sync_memory(cfg, skip_embeddings=True, skip_vision=True)
    assert any("Operation not permitted" in err for err in st2.errors)

    with ChittamStore(cfg) as store:
        uris = {r.uri for r in store.search("Keep Saved Spec Delete", limit=10)}
        assert str(root_a / "saved_spec.md") in uris
        assert str(keep_b) in uris
        assert str(del_b) not in uris


def test_non_compacting_remove_and_metadata_only_tier_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sig = inspect.signature(USearchVectorIndex.remove)
    assert sig.parameters["compact"].default is False

    proj = tmp_path / "workspace" / "demo-app"
    _init_git_repo(proj, days_ago=1)
    readme = proj / "README.md"
    readme.write_text("# Demo App\nInitial guide for metadata-only tier update.", encoding="utf-8")

    cfg, _ = _isolated_cfg(tmp_path, scan_roots=[str(tmp_path / "workspace")])
    atlas = {
        "demo-app": CanonicalRepo(
            project_id="demo-app",
            family="demo",
            name="demo-app",
            path=str(proj),
            tier=RepoTier.REFERENCE,
        )
    }
    save_atlas(atlas, cfg.atlas_path)

    embed_calls = 0

    class TinyEmbedder:
        model_name = "tiny-256"
        dim = 256

        def embed_texts(self, texts: list[str]) -> np.ndarray:
            nonlocal embed_calls
            embed_calls += 1
            out = np.zeros((len(texts), self.dim), dtype=np.float32)
            out[:, 0] = 1.0
            return out

        def embed_query(self, query: str) -> np.ndarray:
            v = np.zeros(self.dim, dtype=np.float32)
            v[0] = 1.0
            return v

    monkeypatch.setattr(indexer, "build_embedding_provider", lambda c, use_external=False: TinyEmbedder())
    st1 = indexer.sync_memory(cfg, skip_vision=True)
    assert st1.embeddings_computed >= 1
    calls_after_first = embed_calls

    with ChittamStore(cfg) as store:
        before_entries = {
            r["target_id"]: int(r["vec_key"])
            for r in store.conn.execute("SELECT target_id, vec_key FROM vector_entries").fetchall()
        }

    # Change atlas tier from REFERENCE -> ACTIVE without editing README.md
    atlas["demo-app"].tier = RepoTier.ACTIVE
    atlas["demo-app"].manual_override = True
    save_atlas(atlas, cfg.atlas_path)

    st2 = indexer.sync_memory(cfg, skip_vision=True)
    assert st2.indexed_docs >= 1
    assert embed_calls == calls_after_first  # zero recomputed embeddings!

    with ChittamStore(cfg) as store:
        after_entries = {
            r["target_id"]: int(r["vec_key"])
            for r in store.conn.execute("SELECT target_id, vec_key FROM vector_entries").fetchall()
        }
        assert after_entries == before_entries
        hits = store.search("Demo App metadata-only", limit=5)
        assert hits and hits[0].canonical_tier == RepoTier.ACTIVE


def test_concurrent_vector_lock_preserves_decision_and_chunk_vectors_and_reentrant_compact(
    tmp_path: Path,
) -> None:
    cfg, _ = _isolated_cfg(tmp_path)
    vec_a = np.zeros(32, dtype=np.float32)
    vec_a[0] = 1.0
    vec_b = np.zeros(32, dtype=np.float32)
    vec_b[1] = 1.0

    doc = DocumentRecord(
        doc_id="doc:lock1",
        uri="/tmp/lock1.md",
        source_harness="files",
        media_type=MediaType.MARKDOWN,
        project_id="demo",
        canonical_tier=RepoTier.ACTIVE,
        title="lock1.md",
        l0_summary="Concurrent vector test",
        sha256="sha_lock1",
        updated_at="2026-11-28T00:00:00Z",
        chunks=[
            ChunkRecord(
                chunk_id="doc:lock1#c0",
                doc_id="doc:lock1",
                chunk_index=0,
                contextual_header="[demo | active | lock1.md]",
                content="Concurrent vector lock preservation test",
                token_estimate=8,
            )
        ],
    )

    with ChittamStore(cfg) as s1, ChittamStore(cfg) as s2:
        s1.upsert_document(doc, _save_vec=False)
        s1.upsert_chunk_embeddings(["doc:lock1#c0"], "test-model", vec_a.reshape(1, -1))
        # Simulate buffered delete/re-flush in s1 while s2 writes a decision embedding
        s1._vec_dirty = True
        dec = s2.record_decision("demo", "locking", "Keep USearch locked across writers.")
        s2.upsert_decision_embedding(dec.decision_id, dec.statement, "test-model", vec_b)
        s1.flush_vector_index()

    with ChittamStore(cfg) as verify_store:
        st = verify_store.get_stats()
        assert st["embeddings"] == 2
        idx = USearchVectorIndex(cfg.vector_index_path, dim=32, dtype="i8")
        assert idx.count() == 2

        # Re-entrant file_lock check: compact() called while sync.lock is already held in-process
        with file_lock(verify_store._sync_lock_path, blocking=False):
            rep = verify_store.compact()
            assert rep["rebuilt_vectors"] == 2


def test_stale_atlas_auto_refreshes_during_sync(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg, _ = _isolated_cfg(tmp_path, scan_roots=[str(workspace)])
    cfg.atlas_refresh_hours = 1.0
    save_atlas({}, cfg.atlas_path)
    assert is_atlas_stale(cfg.atlas_path, 1.0) is False

    # Backdate atlas mtime by 3 hours and add a brand-new git repo
    old_ts = time.time() - 3 * 3600
    os.utime(cfg.atlas_path, (old_ts, old_ts))
    assert is_atlas_stale(cfg.atlas_path, 1.0) is True

    new_repo = workspace / "fresh-service"
    _init_git_repo(new_repo, days_ago=1)
    indexer.sync_memory(cfg, skip_embeddings=True, skip_vision=True)

    reloaded = yaml.safe_load(cfg.atlas_path.read_text(encoding="utf-8"))
    assert "fresh-service" in reloaded.get("projects", {})
    assert reloaded["projects"]["fresh-service"]["tier"] == "active"


def test_stopword_heavy_query_does_not_let_unrelated_decision_beat_doc(tmp_path: Path) -> None:
    cfg, _ = _isolated_cfg(tmp_path)
    doc = DocumentRecord(
        doc_id="doc:admob",
        uri="/tmp/admob_mrr.md",
        source_harness="files",
        media_type=MediaType.MARKDOWN,
        project_id=None,
        canonical_tier=RepoTier.REFERENCE,
        title="admob_7k_mrr_analysis.md",
        l0_summary="How to reach 7k monthly recurring revenue with mobile ads",
        sha256="sha_admob",
        updated_at="2026-11-25T00:00:00Z",
        chunks=[
            ChunkRecord(
                chunk_id="doc:admob#c0",
                doc_id="doc:admob",
                chunk_index=0,
                contextual_header="[global | reference | admob_7k_mrr_analysis.md]",
                content="Analysis of how to reach 7k monthly recurring revenue with mobile ads eCPM and DAU.",
                token_estimate=16,
            )
        ],
    )
    with ChittamStore(cfg) as store:
        store.upsert_document(doc)
        # Active decision shares only stopwords ("to", "with", "how") or substring "ad" in "readme"
        store.record_decision(
            project_id="app-ui",
            topic="clean-layout",
            statement="How to build minimal UI with generous whitespace and readme guidelines.",
        )
        hits = store.search("how to reach 7k monthly recurring revenue with mobile ads", limit=5)
        assert hits
        assert hits[0].doc_id == "doc:admob"


def test_hyphenated_id_and_quoted_or_query_precision(tmp_path: Path) -> None:
    cfg, _ = _isolated_cfg(tmp_path)
    img_doc = DocumentRecord(
        doc_id="doc:frame005",
        uri="/tmp/frame-005.webp",
        source_harness="files",
        media_type=MediaType.IMAGE,
        project_id="design-system",
        canonical_tier=RepoTier.ACTIVE,
        title="frame-005.webp",
        l0_summary="UI animation frame 005",
        sha256="sha_frame005",
        updated_at="2026-11-28T00:00:00Z",
        chunks=[
            ChunkRecord(
                chunk_id="doc:frame005#c0",
                doc_id="doc:frame005",
                chunk_index=0,
                contextual_header="[design-system | active | frame-005.webp]",
                content="Image: frame-005.webp\nAnimation frame 005 preview.",
                token_estimate=10,
            )
        ],
    )
    ticket_doc = DocumentRecord(
        doc_id="doc:ticket005",
        uri="/tmp/tickets/VAM-005.md",
        source_harness="files",
        media_type=MediaType.MARKDOWN,
        project_id="backend-v2",
        canonical_tier=RepoTier.ACTIVE,
        title="VAM-005.md",
        l0_summary="VAM-005 signed session capability auth",
        sha256="sha_ticket005",
        updated_at="2026-11-26T00:00:00Z",
        chunks=[
            ChunkRecord(
                chunk_id="doc:ticket005#c0",
                doc_id="doc:ticket005",
                chunk_index=0,
                contextual_header="[backend-v2 | active | VAM-005.md]",
                content="Ticket VAM-005: Implement signed capability/session auth and fix IDOR.",
                token_estimate=14,
            )
        ],
    )
    with ChittamStore(cfg) as store:
        store.upsert_document(img_doc)
        store.upsert_document(ticket_doc)
        store.record_decision("org-core", "status", "Core status updated.")

        hits1 = store.search('"VAM-005"', limit=5)
        assert [h.doc_id for h in hits1] == ["doc:ticket005"]

        hits2 = store.search('"VAM-002" OR "VAM-005"', limit=5)
        assert [h.doc_id for h in hits2] == ["doc:ticket005"]


def test_match_repo_from_text_and_antigravity_adapter_upgrades(tmp_path: Path) -> None:
    active_repo = tmp_path / "Projects" / "alpha-app"
    archive_repo = tmp_path / "Archive" / "alpha-app-legacy-snapshot-long-path"
    active_repo.mkdir(parents=True)
    archive_repo.mkdir(parents=True)
    atlas = {
        "alpha-app": CanonicalRepo(
            project_id="alpha-app",
            family="alpha",
            name="alpha-app",
            path=str(active_repo),
            tier=RepoTier.ACTIVE,
        ),
        "alpha-old": CanonicalRepo(
            project_id="alpha-old",
            family="alpha",
            name="alpha-old",
            path=str(archive_repo),
            tier=RepoTier.ARCHIVE,
        ),
    }
    text_mix = f"Working in {active_repo}/src and {active_repo}/tests.\n" + ("padding " * 100) + f"\nOld note: {archive_repo}"
    assert match_repo_from_text(text_mix, atlas) == atlas["alpha-app"]

    long_general = ("Career notes and interview prep.\n" * 300) + f"Once saw {archive_repo}."
    assert match_repo_from_text(long_general, atlas) is None

    # Antigravity transcript adapter tests: voice tag strip, transcript_full recovery, tool_calls voting, subagents
    brain = tmp_path / "brain"
    # 1. Human thread with voice_instructions + truncated assistant response + tool_call repo path
    c_human = brain / "conv-human-1111"
    logs_h = c_human / ".system_generated" / "logs"
    logs_h.mkdir(parents=True)
    tx_h = [
        {"step_index": 0, "type": "SYSTEM_MESSAGE", "source": "SYSTEM", "content": f"Active Workspaces:\n- {active_repo}"},
        {
            "step_index": 1,
            "type": "USER_INPUT",
            "source": "USER_EXPLICIT",
            "created_at": "2026-11-28T01:00:00Z",
            "content": "<voice_instructions>\nInternal prompt\n</voice_instructions>\n<talker_agent_context>noise</talker_agent_context>\nReview alpha-app reliability fixes",
        },
        {
            "step_index": 2,
            "type": "PLANNER_RESPONSE",
            "source": "MODEL",
            "created_at": "2026-11-28T01:01:00Z",
            "tool_calls": [{"name": "read_file", "args": {"Path": json.dumps(str(active_repo / "main.py"))}}],
            "truncated_fields": ["content"],
            "content": "Short truncated preview...",
        },
    ]
    full_h = [
        tx_h[0],
        tx_h[1],
        {
            "step_index": 2,
            "type": "PLANNER_RESPONSE",
            "source": "MODEL",
            "created_at": "2026-11-28T01:01:00Z",
            "content": "Recovered full detailed analysis of alpha-app reliability fixes and token_unique_recovered_xyz.",
        },
    ]
    (logs_h / "transcript.jsonl").write_text("\n".join(json.dumps(x) for x in tx_h), encoding="utf-8")
    (logs_h / "transcript_full.jsonl").write_text("\n".join(json.dumps(x) for x in full_h), encoding="utf-8")

    # 2. Boilerplate tiny subagent transcript -> skipped
    c_tiny = brain / "conv-sub-tiny"
    (c_tiny / ".system_generated" / "logs").mkdir(parents=True)
    (c_tiny / ".system_generated" / "logs" / "transcript.jsonl").write_text(
        json.dumps({"step_index": 1, "type": "PLANNER_RESPONSE", "source": "MODEL", "content": "Report sent successfully."}),
        encoding="utf-8",
    )

    # 3. Substantive subagent transcript -> indexed with is_subagent=True, capped at REFERENCE, readable title
    c_sub = brain / "conv-sub-substantive"
    (c_sub / ".system_generated" / "logs").mkdir(parents=True)
    sub_body = "# Alpha App Deep Audit\n" + (
        f"Audited {active_repo} for reliability fixes and concurrency edge cases across storage modules. " * 4
    )
    (c_sub / ".system_generated" / "logs" / "transcript.jsonl").write_text(
        json.dumps({"step_index": 1, "type": "PLANNER_RESPONSE", "source": "MODEL", "content": sub_body}),
        encoding="utf-8",
    )

    cfg, _ = _isolated_cfg(tmp_path)
    cfg.harnesses.antigravity_brain_dir = str(brain)
    docs = load_antigravity_docs(cfg, atlas)
    by_conv = {d.metadata["conversation_id"]: d for d in docs}

    assert "conv-sub-tiny" not in by_conv
    human_doc = by_conv["conv-human-1111"]
    assert human_doc.project_id == "alpha-app"
    assert human_doc.canonical_tier == RepoTier.ACTIVE
    assert "<voice_instructions>" not in human_doc.title
    assert "Review alpha-app reliability fixes" in human_doc.title
    assert "token_unique_recovered_xyz" in "\n".join(c.content for c in human_doc.chunks)

    sub_doc = by_conv["conv-sub-substantive"]
    assert sub_doc.metadata.get("is_subagent") is True
    assert sub_doc.canonical_tier == RepoTier.REFERENCE
    assert sub_doc.title == "Alpha App Deep Audit"

    # Verify human thread outranks substantive subagent in recall_threads
    with ChittamStore(cfg) as store:
        store.upsert_document(sub_doc)
        store.upsert_document(human_doc)
        threads = store.recall_threads("alpha-app reliability fixes", limit=5)
        assert [t.doc_id for t in threads[:2]] == [human_doc.doc_id, sub_doc.doc_id]


def test_rules_antigravity_2_dir_and_family_project_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp_server._reset_caches()
    ag_rules_dir = tmp_path / "gemini_config" / "rules"
    ag_rules_dir.mkdir(parents=True)
    cfg, cfg_path = _isolated_cfg(tmp_path)
    cfg.harnesses.antigravity_rules_path = "~/.gemini/GEMINI.md"
    cfg.harnesses.antigravity_rules_dir = str(ag_rules_dir)

    # Even if user has a real ~/.gemini/GEMINI.md, simulate absent legacy GEMINI.md via monkeypatched target
    monkeypatch.setattr(Path, "expanduser", lambda self: Path(str(self).replace("~", str(tmp_path))))
    res = install_rules(cfg)
    expected_rule = ag_rules_dir / "chittam.md"
    assert res["targets"]["antigravity"]["path"] == str(expected_rule)
    assert expected_rule.exists()

    unres = uninstall_rules(cfg)
    assert unres["targets"]["antigravity"]["status"] == "removed"
    assert not expected_rule.exists()
    monkeypatch.undo()

    # Family-aware project_id filtering in memory_search
    atlas = {
        "auditov-v2": CanonicalRepo(
            project_id="auditov-v2",
            family="auditov",
            name="auditov-v2",
            path=str(tmp_path / "auditov-v2"),
            tier=RepoTier.ACTIVE,
        )
    }
    save_atlas(atlas, cfg.atlas_path)
    assert resolve_project_filter("auditov", atlas) == ["auditov-v2", "auditov"]

    with ChittamStore(cfg) as store:
        store.upsert_document(
            DocumentRecord(
                doc_id="doc:aud",
                uri="/tmp/auditov-v2/auth.py",
                source_harness="files",
                media_type=MediaType.CODE,
                project_id="auditov-v2",
                canonical_tier=RepoTier.ACTIVE,
                title="auth.py",
                l0_summary="Signed session capability auth",
                sha256="sha_aud",
                updated_at="2026-11-28T00:00:00Z",
                chunks=[
                    ChunkRecord(
                        chunk_id="doc:aud#c0",
                        doc_id="doc:aud",
                        chunk_index=0,
                        contextual_header="[auditov-v2 | active | auth.py]",
                        content="Signed capability session auth handler.",
                        token_estimate=8,
                    )
                ],
            )
        )

    monkeypatch.setenv("CHITTAM_CONFIG", str(cfg_path))
    mcp_server._reset_caches()
    try:
        out = mcp_server.memory_search("capability session auth", project_id="auditov", limit=5)
        assert out["count"] == 1
        assert out["results"][0]["project_id"] == "auditov-v2"
    finally:
        mcp_server._reset_caches()


def test_keep_alive_warm_mcp_store_compact_payloads_and_atlas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. keep_alive sent to Ollama
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content.decode("utf-8"))
        captured.append(payload)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.0, 0.0]]})

    provider = OllamaEmbeddingProvider(
        matryoshka_dim=4,
        keep_alive="30m",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider.embed_query("ping")
    assert captured[0]["keep_alive"] == "30m"

    # 2. Warm store reuse + compact payloads + project_atlas defaults
    cfg1, cfg1_path = _isolated_cfg(tmp_path / "env1")
    cfg2, cfg2_path = _isolated_cfg(tmp_path / "env2")

    atlas1 = {
        "act-1": CanonicalRepo(project_id="act-1", family="f1", name="act-1", path="/tmp/a1", tier=RepoTier.ACTIVE),
        "ref-1": CanonicalRepo(project_id="ref-1", family="f2", name="ref-1", path="/tmp/r1", tier=RepoTier.REFERENCE),
        "arch-1": CanonicalRepo(project_id="arch-1", family="f3", name="arch-1", path="/tmp/z1", tier=RepoTier.ARCHIVE),
    }
    save_atlas(atlas1, cfg1.atlas_path)
    save_atlas({"act-2": CanonicalRepo(project_id="act-2", family="f4", name="act-2", path="/tmp/a2", tier=RepoTier.ACTIVE)}, cfg2.atlas_path)

    with ChittamStore(cfg1) as store:
        store.upsert_document(
            DocumentRecord(
                doc_id="doc:img_compact",
                uri="/tmp/shot.png",
                source_harness="files",
                media_type=MediaType.IMAGE,
                project_id="act-1",
                canonical_tier=RepoTier.ACTIVE,
                title="shot.png",
                l0_summary="Architecture diagram " * 15,
                sha256="sha_img",
                updated_at="2026-11-28T00:00:00Z",
                metadata={"vision_caption": "Huge duplicated caption", "file_mtime_ns": 123, "bytes": 999},
                chunks=[
                    ChunkRecord(
                        chunk_id="doc:img_compact#c0",
                        doc_id="doc:img_compact",
                        chunk_index=0,
                        contextual_header="[act-1 | active | shot.png]",
                        content="Image: shot.png\nArchitecture diagram overview.",
                        token_estimate=10,
                    )
                ],
            )
        )

    monkeypatch.setenv("CHITTAM_CONFIG", str(cfg1_path))
    mcp_server._reset_caches()
    try:
        s_a = mcp_server._get_store(mcp_server._get_config())
        s_b = mcp_server._get_store(mcp_server._get_config())
        assert s_a is s_b

        atlas_out = mcp_server.project_atlas()
        assert atlas_out["tier_counts"] == {"active": 1, "reference": 1, "archive": 1}
        assert [p["project_id"] for p in atlas_out["projects"]] == ["act-1", "ref-1"]

        atlas_all = mcp_server.project_atlas(tier="all")
        assert atlas_all["count"] == 3

        search_out = mcp_server.memory_search("Architecture diagram", limit=5)
        assert search_out["count"] == 1
        hit = search_out["results"][0]
        assert len(hit["l0_summary"]) <= 160
        assert "vision_caption" not in hit["metadata"]
        assert "file_mtime_ns" not in hit["metadata"]

        monkeypatch.setenv("CHITTAM_CONFIG", str(cfg2_path))
        s_c = mcp_server._get_store(mcp_server._get_config())
        assert s_c is not s_a
        assert mcp_server.project_atlas()["projects"][0]["project_id"] == "act-2"
    finally:
        mcp_server._reset_caches()


def test_corrupt_image_negative_cache_and_needs_compaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    bad_img = img_dir / "corrupt.png"
    bad_img.write_bytes(b"not_a_real_png" + b"0" * 20000)

    cfg, _ = _isolated_cfg(tmp_path, scan_roots=[str(img_dir)])
    cfg.privacy.min_image_bytes = 1024
    save_atlas({}, cfg.atlas_path)

    vision_calls = 0

    class EmptyVision:
        model = "fake-vl"

        def describe_image(self, image_path: Path, context_hint: str = "") -> str:
            nonlocal vision_calls
            vision_calls += 1
            return ""

    monkeypatch.setattr(indexer, "build_vision_provider", lambda c, use_external=False: EmptyVision())

    st1 = indexer.sync_memory(cfg, skip_embeddings=True)
    assert vision_calls == 1
    assert st1.images_captioned == 0

    # Second sync must hit UNREADABLE_IMAGE_SENTINEL and NOT call describe_image again
    st2 = indexer.sync_memory(cfg, skip_embeddings=True)
    assert vision_calls == 1
    assert st2.images_cached == 0

    with ChittamStore(cfg) as store:
        assert store.needs_compaction(orphan_threshold=1, min_db_bytes=0) is False
        store.conn.execute(
            "INSERT INTO content_embeddings (content_sha256, model, dim, dtype, vector, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("orphan_sha", "m", 4, "i8", b"\x00\x00\x00\x00", "2026-11-28T00:00:00Z"),
        )
        store.conn.commit()
        assert store.needs_compaction(orphan_threshold=1, min_db_bytes=0) is True

        # Verify SQLite uses hash/target_id indexes rather than scanning model_dim or target_type
        orphan_plan = " ".join(
            str(r[3])
            for r in store.conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT COUNT(*) FROM content_embeddings ce
                WHERE NOT EXISTS (
                    SELECT 1 FROM vector_entries ve
                    WHERE ve.content_sha256 = ce.content_sha256
                      AND +ve.model = ce.model
                      AND +ve.dim = ce.dim
                )
                """
            ).fetchall()
        )
        assert "idx_vector_entries_sha" in orphan_plan

        ve_plan = " ".join(
            str(r[3])
            for r in store.conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT vec_key FROM vector_entries
                WHERE +target_type = 'chunk'
                  AND target_id IN (SELECT chunk_id FROM chunks WHERE doc_id = 'd1')
                """
            ).fetchall()
        )
        assert "sqlite_autoindex_vector_entries_1" in ve_plan
