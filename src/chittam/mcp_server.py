"""Pull-only MCP stdio server exposing Chittam hybrid search, atlas, threads, and decisions."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - fallback for mcp<2
    from mcp.server.fastmcp import FastMCP as MCPServer

from chittam.atlas import build_or_load_atlas
from chittam.config import load_config
from chittam.models import AppConfig, RepoTier
from chittam.providers import (
    RerankProvider,
    build_embedding_provider,
    build_query_embedding_provider,
    build_rerank_provider,
)
from chittam.rules import MCP_SERVER_INSTRUCTIONS
from chittam.store import ChittamStore, decision_embedding_text

mcp = MCPServer("chittam", instructions=MCP_SERVER_INSTRUCTIONS)


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
    cfg = load_config()
    clamped_limit = max(1, min(limit, 50))
    reranker = _try_get_reranker(cfg)
    with ChittamStore(cfg) as store:
        q_vec, _, _ = _try_embed_query(cfg, store, query)
        results = store.search(
            query=query,
            query_vec=q_vec,
            limit=clamped_limit,
            media_type=media_type,
            project_id=project_id,
            include_archived=include_archived,
            reranker=reranker,
            rerank_top_k=cfg.local_provider.rerank_top_k,
        )
    return {
        "query": query,
        "count": len(results),
        "vector_used": q_vec is not None,
        "rerank_used": reranker is not None,
        "results": [r.model_dump(mode="json") for r in results],
    }


@mcp.tool()
def memory_get_chunk(chunk_id: str, include_neighbors: bool = True) -> dict[str, Any]:
    """Fetch full L2 chunk content, provenance metadata, and adjacent neighbor chunks."""
    cfg = load_config()
    with ChittamStore(cfg) as store:
        chunk = store.get_chunk(chunk_id, include_neighbors=include_neighbors)
    if chunk is None:
        return {"found": False, "chunk_id": chunk_id, "error": "Chunk not found"}
    return {"found": True, **chunk}


@mcp.tool()
def project_atlas(tier: str | None = None, family: str | None = None) -> dict[str, Any]:
    """Return canonical repositories from the Chittam project atlas, ranked active-first."""
    cfg = load_config()
    atlas = build_or_load_atlas(cfg, refresh=False)
    repos = list(atlas.values())

    if tier:
        tier_norm = tier.strip().lower()
        repos = [r for r in repos if r.tier.value.lower() == tier_norm]
    if family:
        fam_norm = family.strip().lower()
        repos = [r for r in repos if r.family.lower() == fam_norm]

    tier_order = {RepoTier.ACTIVE: 0, RepoTier.REFERENCE: 1, RepoTier.ARCHIVE: 2}
    repos.sort(key=lambda r: r.last_commit_at or "", reverse=True)
    repos.sort(key=lambda r: tier_order.get(r.tier, 1))

    return {
        "count": len(repos),
        "atlas_path": str(cfg.atlas_path),
        "projects": [r.model_dump(mode="json") for r in repos],
    }


@mcp.tool()
def recall_threads(
    query: str,
    harness: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Search prior Antigravity, Codex, and Claude Code threads and agent memories."""
    cfg = load_config()
    clamped_limit = max(1, min(limit, 50))
    reranker = _try_get_reranker(cfg)
    with ChittamStore(cfg) as store:
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
        "threads": [r.model_dump(mode="json") for r in results],
    }


@mcp.tool()
def record_decision(
    project_id: str,
    topic: str,
    statement: str,
    source_harness: str = "mcp",
) -> dict[str, Any]:
    """Record a bitemporal architectural decision, superseding any prior active decision on that topic."""
    cfg = load_config()
    with ChittamStore(cfg) as store:
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
    cfg = load_config()
    with ChittamStore(cfg) as store:
        stats = store.get_stats()

    atlas_count = 0
    if cfg.atlas_path.exists():
        atlas = build_or_load_atlas(cfg, refresh=False)
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
