"""Typer CLI for Chittam initialization, sync, search, decisions, harnesses, launchd, and benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import typer

from chittam.atlas import build_or_load_atlas, save_atlas
from chittam.config import load_config, save_default_user_config
from chittam.indexer import build_leann_export, sync_memory
from chittam.models import AppConfig, RepoTier
from chittam.providers import build_embedding_provider
from chittam.store import ChittamStore

app = typer.Typer(
    name="chittam",
    help="Chittam — local-first multimodal memory, temporal decision graph, and canonical project atlas.",
    no_args_is_help=True,
)


def _load_cfg(config_path: Path | None) -> AppConfig:
    return load_config(config_path=config_path)


@app.command("init")
def init_cmd(
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
    refresh_atlas: bool = typer.Option(False, "--refresh-atlas", help="Force rescan of git repositories."),
) -> None:
    """Initialize Chittam config, SQLite WAL store, and canonical project atlas."""
    cfg = _load_cfg(config)
    Path(cfg.data_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(cfg.config_dir).expanduser().mkdir(parents=True, exist_ok=True)
    cfg.backup_dir.mkdir(parents=True, exist_ok=True)

    cfg_file = save_default_user_config(cfg, config_path=config)
    with ChittamStore(cfg.db_path) as store:
        stats = store.get_stats()

    atlas = build_or_load_atlas(cfg, refresh=refresh_atlas)
    if atlas and not cfg.atlas_path.exists():
        save_atlas(atlas, cfg.atlas_path)

    typer.echo(f"Config: {cfg_file}")
    typer.echo(f"Database: {cfg.db_path} ({stats['documents']} docs)")
    typer.echo(f"Atlas: {cfg.atlas_path} ({len(atlas)} repos)")


@app.command("atlas")
def atlas_cmd(
    refresh: bool = typer.Option(False, "--refresh", help="Rescan git repositories on disk."),
    tier: str | None = typer.Option(None, "--tier", help="Filter by canonical tier (active|reference|archive)."),
    family: str | None = typer.Option(None, "--family", help="Filter by repository family."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
) -> None:
    """Inspect or refresh the canonical repository atlas."""
    cfg = _load_cfg(config)
    atlas = build_or_load_atlas(cfg, refresh=refresh)
    repos = list(atlas.values())

    if tier:
        tier_low = tier.strip().lower()
        repos = [r for r in repos if r.tier.value.lower() == tier_low]
    if family:
        fam_low = family.strip().lower()
        repos = [r for r in repos if r.family.lower() == fam_low]

    tier_order = {RepoTier.ACTIVE: 0, RepoTier.REFERENCE: 1, RepoTier.ARCHIVE: 2}
    repos.sort(key=lambda r: r.last_commit_at or "", reverse=True)
    repos.sort(key=lambda r: tier_order.get(r.tier, 1))

    if json_out:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in repos], indent=2))
        return

    if not repos:
        typer.echo("No repositories matched.")
        return

    for repo in repos:
        commit = (repo.last_commit_at or "unknown")[:10]
        typer.echo(f"[{repo.tier.value:9s}] {repo.project_id:24s} ({repo.family}) {commit} -> {repo.path}")


@app.command("sync")
def sync_cmd(
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
    refresh_atlas: bool = typer.Option(False, "--refresh-atlas", help="Refresh git repository atlas first."),
    no_vision: bool = typer.Option(False, "--no-vision", help="Skip vision captioning for uncached images."),
    no_embeddings: bool = typer.Option(False, "--no-embeddings", help="Skip computing vector embeddings."),
    external: bool = typer.Option(False, "--external", help="Use opt-in external providers (requires consent)."),
    leann_export: bool = typer.Option(False, "--leann-export", help="Also build LEANN corpus/index if installed."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
) -> None:
    """Run incremental Merkle sync across harnesses and scan roots."""
    cfg = _load_cfg(config)

    def _progress(msg: str) -> None:
        if not quiet:
            typer.echo(msg)

    stats = sync_memory(
        cfg,
        refresh_atlas=refresh_atlas,
        use_external=external,
        skip_embeddings=no_embeddings,
        skip_vision=no_vision,
        progress_cb=None if quiet else _progress,
    )
    if leann_export:
        with ChittamStore(cfg.db_path) as store:
            leann_res = build_leann_export(cfg, store, threshold=1, run_build=True)
            if not quiet:
                typer.echo(f"LEANN export: {json.dumps(leann_res)}")

    if not quiet:
        typer.echo(
            f"Sync complete in {stats.duration_ms:.1f}ms | "
            f"scanned={stats.scanned_docs} indexed={stats.indexed_docs} "
            f"unchanged={stats.skipped_unchanged} excluded={stats.skipped_excluded} "
            f"chunks={stats.chunks_written} embeddings={stats.embeddings_computed} "
            f"images_captioned={stats.images_captioned} images_cached={stats.images_cached}"
        )
        if stats.errors:
            typer.echo(f"Warnings/errors ({len(stats.errors)}):", err=True)
            for err in stats.errors[:20]:
                typer.echo(f"  - {err}", err=True)


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(8, "--limit", "-n", help="Maximum number of results."),
    media_type: str | None = typer.Option(None, "--media-type", "-m", help="Filter by media type."),
    project_id: str | None = typer.Option(None, "--project", "-p", help="Filter by project_id."),
    harness: str | None = typer.Option(None, "--harness", help="Filter by source_harness."),
    include_archived: bool = typer.Option(True, "--include-archived/--no-archived", help="Include archived repos."),
    no_vector: bool = typer.Option(False, "--no-vector", help="Use FTS5 BM25 only without embedding query."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
) -> None:
    """Run hybrid BM25 + vector search against the local Chittam store."""
    cfg = _load_cfg(config)
    with ChittamStore(cfg.db_path) as store:
        q_vec: np.ndarray | None = None
        if not no_vector and store.has_embeddings():
            try:
                provider = build_embedding_provider(cfg, use_external=False)
                q_vec = np.asarray(provider.embed_query(query), dtype=np.float32)
            except Exception:
                q_vec = None

        results = store.search(
            query=query,
            query_vec=q_vec,
            limit=limit,
            media_type=media_type,
            project_id=project_id,
            source_harness=harness,
            include_archived=include_archived,
        )

    if json_out:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
        return

    if not results:
        typer.echo("No results found.")
        return

    for idx, res in enumerate(results, start=1):
        typer.echo(
            f"{idx}. [{res.canonical_tier.value}] {res.title} "
            f"(score={res.score:.4f}, media={res.media_type.value}, project={res.project_id or '-'})"
        )
        typer.echo(f"   uri: {res.uri}")
        typer.echo(f"   {res.snippet}")


@app.command("decide")
def decide_cmd(
    project_id: str = typer.Argument(..., help="Project identifier."),
    topic: str = typer.Argument(..., help="Decision topic key."),
    statement: str = typer.Argument(..., help="Decision statement."),
    harness: str = typer.Option("cli", "--harness", help="Source harness recording the decision."),
    no_supersede: bool = typer.Option(False, "--no-supersede", help="Do not supersede prior active decisions on topic."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
) -> None:
    """Record a bitemporal architectural decision."""
    cfg = _load_cfg(config)
    with ChittamStore(cfg.db_path) as store:
        dec = store.record_decision(
            project_id=project_id,
            topic=topic,
            statement=statement,
            source_harness=harness,
            supersedes_topic=not no_supersede,
        )
    typer.echo(json.dumps(dec.model_dump(mode="json"), indent=2))


@app.command("decisions")
def decisions_cmd(
    project_id: str | None = typer.Option(None, "--project", "-p", help="Filter by project_id."),
    all_history: bool = typer.Option(False, "--all", help="Include superseded decisions."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
) -> None:
    """List recorded architectural decisions."""
    cfg = _load_cfg(config)
    with ChittamStore(cfg.db_path) as store:
        items = store.list_decisions(project_id=project_id, active_only=not all_history)

    if json_out:
        typer.echo(json.dumps([d.model_dump(mode="json") for d in items], indent=2))
        return

    if not items:
        typer.echo("No decisions recorded.")
        return

    for d in items:
        state = "ACTIVE" if d.valid_to is None else f"SUPERSEDED->{d.superseded_by}"
        typer.echo(f"[{state}] {d.project_id}/{d.topic} ({d.decision_id}): {d.statement}")


@app.command("status")
def status_cmd(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
) -> None:
    """Show index stats, atlas status, and launchd background sync status."""
    cfg = _load_cfg(config)
    with ChittamStore(cfg.db_path) as store:
        stats = store.get_stats()

    atlas_count = 0
    if cfg.atlas_path.exists():
        atlas_count = len(build_or_load_atlas(cfg, refresh=False))

    launchd_info: dict[str, Any]
    try:
        from chittam.launchd import launchd_status

        launchd_info = launchd_status()
    except Exception as exc:
        launchd_info = {"error": str(exc)}

    payload = {
        "db_exists": cfg.db_path.exists(),
        "atlas_exists": cfg.atlas_path.exists(),
        "atlas_projects": atlas_count,
        "stats": stats,
        "launchd": launchd_info,
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"DB: {stats['db_path']} ({stats['db_size_bytes']} bytes)")
    typer.echo(
        f"Docs={stats['documents']} Chunks={stats['chunks']} Embeddings={stats['embeddings']} "
        f"Decisions={stats['active_decisions']}/{stats['decisions']} CachedImages={stats['cached_images']}"
    )
    typer.echo(f"Atlas: {cfg.atlas_path} ({atlas_count} repos)")
    typer.echo(f"Launchd: {json.dumps(launchd_info)}")


@app.command("install-harnesses")
def install_harnesses_cmd(
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
) -> None:
    """Register Chittam and CodeGraph MCP servers in Antigravity, Codex, and Claude Code."""
    from chittam.harnesses import install_harnesses

    cfg = _load_cfg(config)
    result = install_harnesses(cfg)
    typer.echo(json.dumps(result, indent=2))


@app.command("uninstall-harnesses")
def uninstall_harnesses_cmd(
    restore_backup: bool = typer.Option(
        False,
        "--restore-backup",
        help="Restore pre-install backups instead of selectively removing Chittam/CodeGraph.",
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
) -> None:
    """Restore or cleanly remove Chittam/CodeGraph MCP entries from harness configs."""
    from chittam.harnesses import uninstall_harnesses

    cfg = _load_cfg(config)
    result = uninstall_harnesses(cfg, restore_backup=restore_backup)
    typer.echo(json.dumps(result, indent=2))


@app.command("install-launchd")
def install_launchd_cmd(
    no_load: bool = typer.Option(False, "--no-load", help="Write plist without loading into launchctl."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
) -> None:
    """Install periodic 30-minute launchd background sync agent."""
    from chittam.launchd import install_launchd

    cfg = _load_cfg(config)
    plist_path = install_launchd(cfg, load=not no_load)
    typer.echo(f"Installed launchd plist: {plist_path}")


@app.command("uninstall-launchd")
def uninstall_launchd_cmd() -> None:
    """Unload and remove Chittam launchd background sync agent."""
    from chittam.launchd import uninstall_launchd

    removed = uninstall_launchd()
    typer.echo(f"Uninstalled launchd agent: {removed}")


@app.command("benchmark")
def benchmark_cmd(
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Run against a throwaway fixture corpus (no Ollama or local index needed)."
    ),
    cases: Path | None = typer.Option(
        None, "--cases", help="YAML file of custom cases (default: ~/.config/chittam/benchmark_cases.yaml)."
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Override config YAML path."),
) -> None:
    """Run the retrieval-quality and privacy-leakage benchmark."""
    from chittam.benchmark import load_cases_from_yaml, run_benchmark, run_synthetic_benchmark

    if synthetic:
        report = run_synthetic_benchmark()
    else:
        cfg = _load_cfg(config)
        custom = load_cases_from_yaml(cases) if cases else None
        report = run_benchmark(cfg, cases=custom)
    typer.echo(json.dumps(report, indent=2))


@app.command("mcp")
def mcp_cmd() -> None:
    """Run the Chittam MCP server over stdio."""
    from chittam.mcp_server import main

    main()


if __name__ == "__main__":  # pragma: no cover
    app()
