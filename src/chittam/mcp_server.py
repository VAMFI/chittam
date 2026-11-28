"""Pull-only MCP stdio server exposing Chittam hybrid search, atlas, threads, and decisions."""

from __future__ import annotations

import atexit
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - fallback for mcp<2
    from mcp.server.fastmcp import FastMCP as MCPServer

from chittam.atlas import build_or_load_atlas, resolve_project_filter
from chittam.config import USER_CONFIG_FILE, load_config
from chittam.models import AppConfig, CanonicalRepo, RepoTier, SearchResult
from chittam.providers import (
    RerankProvider,
    build_embedding_provider,
    build_query_embedding_provider,
    build_rerank_provider,
)
from chittam.rules import MCP_SERVER_INSTRUCTIONS
from chittam.store import ChittamStore, decision_embedding_text

mcp = MCPServer("chittam", instructions=MCP_SERVER_INSTRUCTIONS)

_CACHE_LOCK = threading.RLock()
_CACHED_CFG_KEY: tuple[str, float | None] | None = None
_CACHED_CFG: AppConfig | None = None
_CACHED_ATLAS_KEY: tuple[str, float | None] | None = None
_CACHED_ATLAS: dict[str, CanonicalRepo] | None = None
_CACHED_STORE_KEY: tuple[str, str, str, tuple[int, int] | None] | None = None
_CACHED_STORE: ChittamStore | None = None

_KEEP_METADATA_KEYS = frozenset(
    {
        "symbol_name",
        "line_start",
        "line_end",
        "conversation_id",
        "turn_count",
        "decision_id",
        "valid_from",
        "valid_to",
        "superseded_by",
        "rerank_score",
        "is_subagent",
        "artifact",
        "git_branch",
        "thread_id",
        "rollout_slug",
        "total_pages",
    }
)


def _file_sig(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return (int(st.st_dev), int(st.st_ino))
    except OSError:
        return None


def _file_mtime(path: Path) -> float | None:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return None


def _reset_caches() -> None:
    """Close cached ChittamStore and clear config/atlas/store caches."""
    global _CACHED_CFG_KEY, _CACHED_CFG, _CACHED_ATLAS_KEY, _CACHED_ATLAS, _CACHED_STORE_KEY, _CACHED_STORE
    with _CACHE_LOCK:
        if _CACHED_STORE is not None:
            try:
                _CACHED_STORE.close()
            except Exception:
                pass
        _CACHED_STORE = None
        _CACHED_STORE_KEY = None
        _CACHED_ATLAS = None
        _CACHED_ATLAS_KEY = None
        _CACHED_CFG = None
        _CACHED_CFG_KEY = None


atexit.register(_reset_caches)


def _get_config() -> AppConfig:
    global _CACHED_CFG_KEY, _CACHED_CFG
    cfg_path = Path(os.environ.get("CHITTAM_CONFIG", USER_CONFIG_FILE)).expanduser()
    key = (str(cfg_path), _file_mtime(cfg_path))
    with _CACHE_LOCK:
        if _CACHED_CFG is not None and _CACHED_CFG_KEY == key:
            return _CACHED_CFG
        cfg = load_config()
        _CACHED_CFG = cfg
        _CACHED_CFG_KEY = key
        return cfg


def _get_atlas(cfg: AppConfig) -> dict[str, CanonicalRepo]:
    global _CACHED_ATLAS_KEY, _CACHED_ATLAS
    atlas_path = cfg.atlas_path
    key = (str(atlas_path), _file_mtime(atlas_path))
    with _CACHE_LOCK:
        if _CACHED_ATLAS is not None and _CACHED_ATLAS_KEY == key:
            return _CACHED_ATLAS
        atlas = build_or_load_atlas(cfg, refresh=False)
        _CACHED_ATLAS = atlas
        _CACHED_ATLAS_KEY = key
        return atlas


def _get_store(cfg: AppConfig) -> ChittamStore:
    global _CACHED_STORE_KEY, _CACHED_STORE
    db_path = cfg.db_path
    key = (
        str(db_path.resolve(strict=False)),
        cfg.vector_backend,
        cfg.local_provider.vector_dtype,
        _file_sig(db_path),
    )
    with _CACHE_LOCK:
        if _CACHED_STORE is not None and _CACHED_STORE_KEY == key:
            return _CACHED_STORE
        if _CACHED_STORE is not None:
            try:
                _CACHED_STORE.close()
            except Exception:
                pass
        store = ChittamStore(cfg)
        _CACHED_STORE = store
        _CACHED_STORE_KEY = (
            str(db_path.resolve(strict=False)),
            cfg.vector_backend,
            cfg.local_provider.vector_dtype,
            _file_sig(db_path),
        )
        return store


def _compact_result(r: SearchResult) -> dict[str, Any]:
    data = r.model_dump(mode="json")
    l0 = str(data.get("l0_summary") or "")
    if len(l0) > 160:
        data["l0_summary"] = l0[:157].rstrip() + "..."
    raw_meta = data.get("metadata") or {}
    if isinstance(raw_meta, dict):
        data["metadata"] = {k: v for k, v in raw_meta.items() if k in _KEEP_METADATA_KEYS}
    return data


def _try_embed_query(
    cfg: AppConfig,
    store: ChittamStore,
    query: str,
    *,
    require_existing_vectors: bool = True,
) -> tuple[np.ndarray | None, str | None, str]:
    if not query.strip():
        return None, None, "local"
    if require_existing_vectors and not store.has_embeddings():
        return None, None, "local"
    try:
        meta = store.get_index_meta()
        provider = build_query_embedding_provider(
            cfg,
            index_meta=meta,
            provider_builder=build_embedding_provider,
        )
        vec = provider.embed_query(query)
        model_name = (
            getattr(provider, "model_name", None)
            or getattr(provider, "model", None)
            or cfg.local_provider.embedding_model
        )
        src = meta.get("active_embedding_source", "local")
        return np.asarray(vec, dtype=np.float32), str(model_name), src
    except Exception:
        return None, None, "local"


def _try_get_reranker(cfg: AppConfig) -> RerankProvider | None:
    try:
        return build_rerank_provider(cfg)
    except Exception:
        return None


@mcp.tool()
def memory_search(
    query: str,
    limit: int = 8,
    media_type: str | None = None,
    project_id: str | None = None,
    include_archived: bool = True,
) -> dict[str, Any]:
    """Hybrid search across code, docs, PDFs, captioned images, threads, and decisions."""
    cfg = _get_config()
    clamped_limit = max(1, min(limit, 50))
    reranker = _try_get_reranker(cfg)
    resolved_project: str | list[str] | None = project_id
    if project_id:
        atlas = _get_atlas(cfg)
        if atlas:
            resolved_project = resolve_project_filter(project_id, atlas)

    store = _get_store(cfg)
    q_vec, _, _ = _try_embed_query(cfg, store, query)
    results = store.search(
        query=query,
        query_vec=q_vec,
        limit=clamped_limit,
        media_type=media_type,
        project_id=resolved_project,
        include_archived=include_archived,
        reranker=reranker,
        rerank_top_k=cfg.local_provider.rerank_top_k,
    )
    return {
        "query": query,
        "count": len(results),
        "vector_used": q_vec is not None,
        "rerank_used": reranker is not None,
        "results": [_compact_result(r) for r in results],
    }


@mcp.tool()
def memory_get_chunk(chunk_id: str, include_neighbors: bool = True) -> dict[str, Any]:
    """Fetch full L2 chunk content, provenance metadata, and adjacent neighbor chunks."""
    cfg = _get_config()
    store = _get_store(cfg)
    chunk = store.get_chunk(chunk_id, include_neighbors=include_neighbors)
    if chunk is None:
        return {"found": False, "chunk_id": chunk_id, "error": "Chunk not found"}
    return {"found": True, **chunk}


@mcp.tool()
def project_atlas(
    tier: str | None = None,
    family: str | None = None,
    include_archived: bool = False,
    limit: int = 50,
) -> dict[str, Any]:
    """Return canonical repositories from the Chittam project atlas, ranked active-first."""
    cfg = _get_config()
    atlas = _get_atlas(cfg)
    all_repos = list(atlas.values())
    tier_counts = {
        RepoTier.ACTIVE.value: sum(1 for r in all_repos if r.tier == RepoTier.ACTIVE),
        RepoTier.REFERENCE.value: sum(1 for r in all_repos if r.tier == RepoTier.REFERENCE),
        RepoTier.ARCHIVE.value: sum(1 for r in all_repos if r.tier == RepoTier.ARCHIVE),
    }

    repos = all_repos
    if tier:
        tier_norm = tier.strip().lower()
        if tier_norm != "all":
            repos = [r for r in repos if r.tier.value.lower() == tier_norm]
        if family:
            fam_norm = family.strip().lower()
            repos = [r for r in repos if r.family.lower() == fam_norm]
    elif family:
        fam_norm = family.strip().lower()
        repos = [r for r in repos if r.family.lower() == fam_norm]
    elif not include_archived:
        repos = [r for r in repos if r.tier != RepoTier.ARCHIVE]

    tier_order = {RepoTier.ACTIVE: 0, RepoTier.REFERENCE: 1, RepoTier.ARCHIVE: 2}
    repos.sort(key=lambda r: r.last_commit_at or "", reverse=True)
    repos.sort(key=lambda r: tier_order.get(r.tier, 1))

    clamped_limit = max(1, min(int(limit), 250))
    total_matching = len(repos)
    sliced = repos[:clamped_limit]

    return {
        "count": len(sliced),
        "total_matching": total_matching,
        "truncated": total_matching > len(sliced),
        "tier_counts": tier_counts,
        "atlas_path": str(cfg.atlas_path),
        "projects": [r.model_dump(mode="json") for r in sliced],
    }


@mcp.tool()
def recall_threads(
    query: str,
    harness: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Search prior Antigravity, Codex, and Claude Code threads and agent memories."""
    cfg = _get_config()
    clamped_limit = max(1, min(limit, 50))
    reranker = _try_get_reranker(cfg)
    store = _get_store(cfg)
    q_vec, _, _ = _try_embed_query(cfg, store, query)
    results = store.recall_threads(
        query=query,
        query_vec=q_vec,
        harness=harness,
        limit=clamped_limit,
        reranker=reranker,
        rerank_top_k=cfg.local_provider.rerank_top_k,
    )
    return {
        "query": query,
        "harness": harness,
        "count": len(results),
        "vector_used": q_vec is not None,
        "rerank_used": reranker is not None,
        "threads": [_compact_result(r) for r in results],
    }


@mcp.tool()
def record_decision(
    project_id: str,
    topic: str,
    statement: str,
    source_harness: str = "mcp",
) -> dict[str, Any]:
    """Record a bitemporal architectural decision, superseding any prior active decision on that topic."""
    cfg = _get_config()
    store = _get_store(cfg)
    dec = store.record_decision(
        project_id=project_id,
        topic=topic,
        statement=statement,
        source_harness=source_harness,
        supersedes_topic=True,
    )
    dec_text = decision_embedding_text(dec)
    vec, model_name, src = _try_embed_query(
        cfg,
        store,
        dec_text,
        require_existing_vectors=False,
    )
    if vec is not None and model_name:
        try:
            store.upsert_decision_embedding(
                dec.decision_id,
                dec_text,
                model_name,
                vec,
                source=src,
            )
        except Exception:
            pass
    return {
        "status": "recorded",
        "decision": dec.model_dump(mode="json"),
    }


@mcp.tool()
def index_status() -> dict[str, Any]:
    """Return index counts, database size, vector index breakdown, and canonical atlas presence."""
    cfg = _get_config()
    store = _get_store(cfg)
    stats = store.get_stats()

    atlas_count = 0
    if cfg.atlas_path.exists():
        atlas = _get_atlas(cfg)
        atlas_count = len(atlas)

    return {
        "status": "ok",
        "db_exists": cfg.db_path.exists(),
        "vector_index_exists": cfg.vector_index_path.exists(),
        "atlas_exists": cfg.atlas_path.exists(),
        "atlas_projects": atlas_count,
        "enable_leann_bridge": cfg.enable_leann_bridge,
        "enable_codegraph": cfg.enable_codegraph,
        "stats": stats,
    }


def main() -> None:
    """Run the Chittam MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
