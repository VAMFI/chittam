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
from chittam.providers import build_embedding_provider
from chittam.store import ChittamStore

mcp = MCPServer("chittam")


def _try_embed_query(cfg: AppConfig, store: ChittamStore, query: str) -> np.ndarray | None:
    if not query.strip() or not store.has_embeddings():
        return None
    try:
        provider = build_embedding_provider(cfg, use_external=False)
        vec = provider.embed_query(query)
        return np.asarray(vec, dtype=np.float32)
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
    with ChittamStore(cfg.db_path) as store:
        q_vec = _try_embed_query(cfg, store, query)
        results = store.search(
            query=query,
            query_vec=q_vec,
            limit=clamped_limit,
            media_type=media_type,
            project_id=project_id,
            include_archived=include_archived,
        )
    return {
        "query": query,
        "count": len(results),
        "vector_used": q_vec is not None,
        "results": [r.model_dump(mode="json") for r in results],
    }


@mcp.tool()
def memory_get_chunk(chunk_id: str, include_neighbors: bool = True) -> dict[str, Any]:
    """Fetch full L2 chunk content, provenance metadata, and adjacent neighbor chunks."""
    cfg = load_config()
    with ChittamStore(cfg.db_path) as store:
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
    with ChittamStore(cfg.db_path) as store:
        q_vec = _try_embed_query(cfg, store, query)
        results = store.recall_threads(
            query=query,
            query_vec=q_vec,
            harness=harness,
            limit=clamped_limit,
        )
    return {
        "query": query,
        "harness": harness,
        "count": len(results),
        "vector_used": q_vec is not None,
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
    with ChittamStore(cfg.db_path) as store:
        dec = store.record_decision(
            project_id=project_id,
            topic=topic,
            statement=statement,
            source_harness=source_harness,
            supersedes_topic=True,
        )
    return {
        "status": "recorded",
        "decision": dec.model_dump(mode="json"),
    }


@mcp.tool()
def index_status() -> dict[str, Any]:
    """Return index counts, database size, and canonical atlas presence."""
    cfg = load_config()
    with ChittamStore(cfg.db_path) as store:
        stats = store.get_stats()

    atlas_count = 0
    if cfg.atlas_path.exists():
        atlas = build_or_load_atlas(cfg, refresh=False)
        atlas_count = len(atlas)

    return {
        "status": "ok",
        "db_exists": cfg.db_path.exists(),
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
