"""Incremental Merkle & stat() pre-filter indexer, multimodal cache, watcher delta sync, and LEANN export."""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from chittam.adapters import collect_all_documents, load_antigravity_docs, load_claude_code_docs, load_codex_docs, load_single_file_doc
from chittam.adapters.files import UNREADABLE_IMAGE_SENTINEL
from chittam.atlas import build_or_load_atlas, is_atlas_stale, save_atlas
from chittam.config import redact_secrets
from chittam.models import AppConfig, ChunkRecord, MediaType, RepoTier, SyncStats
from chittam.providers import build_embedding_provider, build_vision_provider
from chittam.store import ChittamStore, decision_embedding_text, file_lock

_EMBED_BATCH_SIZE = 32
_LEANN_DEFAULT_THRESHOLD = 500


def _content_sha256(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8", errors="replace")).hexdigest()


def _emit_progress(
    progress_cb: Callable[..., Any] | None,
    stage: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    if progress_cb is None:
        return
    try:
        sig = inspect.signature(progress_cb)
        params = list(sig.parameters.values())
        positional = [
            p
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        if has_varargs or len(positional) >= 2:
            progress_cb(stage, {"message": message, **(payload or {})})
        elif len(positional) == 1:
            progress_cb(f"[{stage}] {message}")
        else:
            progress_cb()
    except Exception:
        pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _find_leann_cli() -> str | None:
    found = shutil.which("leann")
    if found:
        return found
    candidate = Path("~/.local/bin/leann").expanduser()
    if candidate.is_file():
        return str(candidate)
    return None


def build_leann_export(
    cfg: AppConfig,
    store: ChittamStore,
    threshold: int = _LEANN_DEFAULT_THRESHOLD,
    run_build: bool = False,
) -> dict[str, Any]:
    """Export chunk corpus JSONL (and optionally build LEANN index) when leann CLI is available."""
    if not cfg.enable_leann_bridge:
        return {"exported": False, "reason": "disabled_in_config"}

    leann_bin = _find_leann_cli()
    if not leann_bin:
        return {"exported": False, "reason": "leann_cli_not_installed"}

    stats = store.get_stats()
    chunk_count = int(stats.get("chunks", 0))
    if chunk_count < threshold:
        return {
            "exported": False,
            "reason": "below_threshold",
            "chunks": chunk_count,
            "threshold": threshold,
        }

    data_dir = Path(cfg.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = data_dir / "leann_corpus.jsonl"
    chunks = store.iter_chunks_for_export()
    with corpus_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            record = {
                "id": c["chunk_id"],
                "text": f"{c['contextual_header']}\n{c['content']}".strip(),
                "metadata": {
                    "doc_id": c["doc_id"],
                    "uri": c["uri"],
                    "title": c["title"],
                    "project_id": c["project_id"],
                    "canonical_tier": c["canonical_tier"],
                    "source_harness": c["source_harness"],
                    "media_type": c["media_type"],
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    result: dict[str, Any] = {
        "exported": True,
        "leann_bin": leann_bin,
        "corpus_path": str(corpus_path),
        "chunks": len(chunks),
        "built_index": False,
    }
    if run_build:
        index_dir = data_dir / "leann_index"
        proc = subprocess.run(
            [leann_bin, "build", str(index_dir), "--docs", str(corpus_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        result["built_index"] = proc.returncode == 0
        result["returncode"] = proc.returncode
        if proc.returncode != 0:
            result["stderr"] = proc.stderr.strip()[:500]
    return result


def _build_vision_helper(
    cfg: AppConfig,
    store: ChittamStore,
    use_external: bool,
    stats: SyncStats,
) -> tuple[Callable[[Path, str], str] | None, str]:
    vision_model_name = (
        (cfg.external_provider.vision_model if use_external else cfg.local_provider.vision_model)
        or "vision"
    )
    try:
        vp = build_vision_provider(cfg, use_external=use_external)
        vision_model_name = (
            getattr(vp, "model", None) or getattr(vp, "model_name", None) or vision_model_name
        )

        def _cached_vision(image_path: Path, context_hint: str = "") -> str:
            sha = _sha256_file(image_path)
            cached = store.get_cached_image_caption(sha)
            if cached == UNREADABLE_IMAGE_SENTINEL:
                return ""
            if cached is not None:
                return cached
            raw = vp.describe_image(image_path, context_hint=context_hint)
            if isinstance(raw, dict):
                caption = str(
                    raw.get("caption")
                    or raw.get("description")
                    or json.dumps(raw, ensure_ascii=False)
                ).strip()
            else:
                caption = str(raw or "").strip()
            if caption:
                caption = redact_secrets(caption)
                store.cache_image_caption(sha, vision_model_name, caption)
            else:
                store.cache_image_caption(sha, vision_model_name, UNREADABLE_IMAGE_SENTINEL)
            return caption

        return _cached_vision, vision_model_name
    except Exception as exc:
        stats.errors.append(f"vision_provider_unavailable: {exc}")
        return None, vision_model_name


def _run_embedding_stage(
    cfg: AppConfig,
    store: ChittamStore,
    stats: SyncStats,
    *,
    use_external: bool = False,
    chunk_limit: int | None = None,
    progress_cb: Callable[..., Any] | None = None,
) -> None:
    try:
        ep = build_embedding_provider(cfg, use_external=use_external)
    except Exception as exc:
        stats.errors.append(f"embedding_provider_unavailable: {exc}")
        return

    embed_model = (
        getattr(ep, "model", None)
        or getattr(ep, "model_name", None)
        or (
            cfg.external_provider.embedding_model
            if use_external and cfg.external_provider.embedding_model
            else cfg.local_provider.embedding_model
        )
    )
    target_dim = int(
        getattr(ep, "dim", None)
        or cfg.local_provider.matryoshka_dim
        or cfg.local_provider.embedding_dim
    )
    source_label = "external" if use_external else "local"

    stats.embeddings_deduped += store.link_deduped_embeddings(
        model=embed_model,
        dim=target_dim,
        embed_tiers=cfg.privacy.embed_tiers,
        embed_always_media=cfg.privacy.embed_always_media,
        source=source_label,
    )

    effective_limit = chunk_limit if chunk_limit is not None else cfg.privacy.max_embed_chunks_per_sync
    missing_chunks: list[ChunkRecord] = store.get_chunks_missing_embeddings(
        model=embed_model,
        limit=effective_limit,
        embed_tiers=cfg.privacy.embed_tiers,
        embed_always_media=cfg.privacy.embed_always_media,
        dim=target_dim,
    )

    if missing_chunks:
        rep_by_sha: dict[str, ChunkRecord] = {}
        dup_count_in_batch = 0
        for chunk in missing_chunks:
            c_sha = _content_sha256(chunk.content)
            if c_sha not in rep_by_sha:
                rep_by_sha[c_sha] = chunk
            else:
                dup_count_in_batch += 1
        unique_reps = list(rep_by_sha.values())

        _emit_progress(
            progress_cb,
            "embed",
            f"Computing embeddings for {len(unique_reps)} unique chunks ({dup_count_in_batch} deduped)",
        )
        for start in range(0, len(unique_reps), _EMBED_BATCH_SIZE):
            batch = unique_reps[start : start + _EMBED_BATCH_SIZE]
            texts = [
                (f"{c.contextual_header}\n{c.content}".strip() or c.contextual_header or "chunk")
                for c in batch
            ]
            try:
                vecs = np.asarray(ep.embed_texts(texts), dtype=np.float32)
                if vecs.ndim == 1:
                    vecs = vecs.reshape(1, -1)
                if vecs.shape[0] != len(batch):
                    raise ValueError(
                        f"embed_texts returned {vecs.shape[0]} rows for {len(batch)} texts"
                    )
                written = store.upsert_chunk_embeddings(
                    [c.chunk_id for c in batch],
                    embed_model,
                    vecs,
                    source=source_label,
                )
                stats.embeddings_computed += written
            except Exception as exc:
                stats.errors.append(f"embedding_batch_error@{start}: {exc}")

        if dup_count_in_batch > 0:
            stats.embeddings_deduped += store.link_deduped_embeddings(
                model=embed_model,
                dim=target_dim,
                embed_tiers=cfg.privacy.embed_tiers,
                embed_always_media=cfg.privacy.embed_always_media,
                source=source_label,
            )

    # Also compute semantic vectors for any un-embedded decisions
    missing_decisions = store.get_decisions_missing_embeddings(
        model=embed_model,
        dim=target_dim,
        active_only=False,
    )
    if missing_decisions:
        for start in range(0, len(missing_decisions), _EMBED_BATCH_SIZE):
            dec_batch = missing_decisions[start : start + _EMBED_BATCH_SIZE]
            dec_texts = [decision_embedding_text(d) for d in dec_batch]
            try:
                dec_vecs = np.asarray(ep.embed_texts(dec_texts), dtype=np.float32)
                if dec_vecs.ndim == 1:
                    dec_vecs = dec_vecs.reshape(1, -1)
                for dec, dec_txt, vec in zip(dec_batch, dec_texts, dec_vecs):
                    store.upsert_decision_embedding(
                        dec.decision_id,
                        dec_txt,
                        embed_model,
                        vec,
                        source=source_label,
                    )
                    stats.embeddings_computed += 1
            except Exception as exc:
                stats.errors.append(f"decision_embedding_error@{start}: {exc}")


def sync_memory(
    cfg: AppConfig,
    refresh_atlas: bool = False,
    use_external: bool = False,
    skip_embeddings: bool = False,
    skip_vision: bool = False,
    progress_cb: Callable[..., Any] | None = None,
) -> SyncStats:
    """Run canonical atlas discovery, stat()/Merkle diffing, deduped embeddings, and LEANN export."""
    t0 = time.perf_counter()
    stats = SyncStats()

    Path(cfg.data_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(cfg.config_dir).expanduser().mkdir(parents=True, exist_ok=True)

    lock_path = Path(cfg.data_dir).expanduser() / "sync.lock"
    try:
        with file_lock(lock_path, blocking=False):
            return _sync_memory_locked(
                cfg=cfg,
                stats=stats,
                t0=t0,
                refresh_atlas=refresh_atlas,
                use_external=use_external,
                skip_embeddings=skip_embeddings,
                skip_vision=skip_vision,
                progress_cb=progress_cb,
            )
    except BlockingIOError:
        stats.errors.append("sync_already_in_progress")
        stats.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return stats


def _sync_memory_locked(
    *,
    cfg: AppConfig,
    stats: SyncStats,
    t0: float,
    refresh_atlas: bool,
    use_external: bool,
    skip_embeddings: bool,
    skip_vision: bool,
    progress_cb: Callable[..., Any] | None,
) -> SyncStats:
    _emit_progress(progress_cb, "atlas", "Loading canonical project atlas")
    effective_refresh = refresh_atlas or (
        cfg.atlas_refresh_hours > 0 and is_atlas_stale(cfg.atlas_path, cfg.atlas_refresh_hours)
    )
    atlas = build_or_load_atlas(cfg, refresh=effective_refresh)
    if not cfg.atlas_path.exists() and atlas:
        save_atlas(atlas, cfg.atlas_path)

    with ChittamStore(cfg) as store:
        vision_fn: Callable[[Path, str], str] | None = None
        vision_model_name = (
            (cfg.external_provider.vision_model if use_external else cfg.local_provider.vision_model)
            or "vision"
        )
        if not skip_vision:
            vision_fn, vision_model_name = _build_vision_helper(cfg, store, use_external, stats)

        _emit_progress(progress_cb, "collect", "Collecting documents across harnesses and roots")
        existing_stats = store.get_document_stat_index()
        existing_projects = store.get_document_projects()
        unchanged_doc_ids: set[str] = set()
        unreadable_prefixes: set[str] = set()
        failed_harnesses: set[str] = set()

        sig = inspect.signature(collect_all_documents)
        collect_kwargs: dict[str, Any] = {
            "vision_fn": vision_fn,
            "image_cache_lookup": store.get_cached_image_caption,
        }
        if "existing_stats" in sig.parameters:
            collect_kwargs["existing_stats"] = existing_stats
        if "unchanged_doc_ids" in sig.parameters:
            collect_kwargs["unchanged_doc_ids"] = unchanged_doc_ids
        if "existing_projects" in sig.parameters:
            collect_kwargs["existing_projects"] = existing_projects
        if "unreadable_prefixes" in sig.parameters:
            collect_kwargs["unreadable_prefixes"] = unreadable_prefixes
        if "failed_harnesses" in sig.parameters:
            collect_kwargs["failed_harnesses"] = failed_harnesses

        docs, collect_stats = collect_all_documents(cfg, atlas, **collect_kwargs)

        stats.stat_skipped += collect_stats.stat_skipped
        stats.skipped_excluded += collect_stats.skipped_excluded
        stats.images_captioned += collect_stats.images_captioned
        stats.images_cached += collect_stats.images_cached
        stats.errors.extend(collect_stats.errors)

        if not skip_vision:
            for doc in docs:
                if doc.media_type == MediaType.IMAGE and doc.sha256:
                    if store.get_cached_image_caption(doc.sha256) is None:
                        caption_text = str(
                            doc.metadata.get("vision_caption")
                            or doc.l0_summary
                            or (doc.chunks[0].content if doc.chunks else "")
                        ).strip()
                        if caption_text and not caption_text.startswith("[image uncaptioned"):
                            store.cache_image_caption(
                                doc.sha256,
                                vision_model_name,
                                redact_secrets(caption_text),
                            )

        _emit_progress(progress_cb, "merkle", f"Diffing {len(docs)} documents against store")
        seen_doc_ids: set[str] = set(unchanged_doc_ids)
        stats.skipped_unchanged += len(unchanged_doc_ids)

        for doc in docs:
            seen_doc_ids.add(doc.doc_id)
            doc_tier = (
                doc.canonical_tier.value
                if isinstance(doc.canonical_tier, RepoTier)
                else str(doc.canonical_tier)
            )
            existing = existing_stats.get(doc.doc_id)
            if existing is not None:
                ex_mtime, ex_size, ex_sha, ex_tier, ex_l0 = existing
                if ex_sha == doc.sha256:
                    if doc.media_type != MediaType.IMAGE or ex_l0 == doc.l0_summary:
                        ex_proj = existing_projects.get(doc.doc_id)
                        if ex_tier == doc_tier and ex_proj == doc.project_id:
                            if (ex_mtime is None or ex_size is None) and doc.metadata.get("file_mtime_ns") is not None:
                                store.update_document_file_stat(
                                    doc.doc_id,
                                    doc.metadata.get("file_mtime_ns"),
                                    doc.metadata.get("file_size"),
                                )
                            stats.skipped_unchanged += 1
                            continue
                        store.update_document_metadata(
                            doc.doc_id,
                            project_id=doc.project_id,
                            canonical_tier=doc_tier,
                            file_mtime_ns=doc.metadata.get("file_mtime_ns"),
                            file_size=doc.metadata.get("file_size"),
                        )
                        stats.indexed_docs += 1
                        continue

            store.upsert_document(doc, _save_vec=False)
            stats.indexed_docs += 1
            stats.chunks_written += len(doc.chunks)

        store.flush_vector_index()
        stats.scanned_docs = len(docs) + len(unchanged_doc_ids)

        # Prune docs no longer present, protecting unreadable TCC roots and failed harness adapters
        if docs or unchanged_doc_ids or not collect_stats.errors:
            store.prune_documents(
                seen_doc_ids,
                preserve_media_types={MediaType.DECISION.value},
                preserve_uri_prefixes=unreadable_prefixes,
                preserve_harnesses=failed_harnesses,
            )

        if not skip_embeddings:
            _run_embedding_stage(
                cfg,
                store,
                stats,
                use_external=use_external,
                chunk_limit=cfg.privacy.max_embed_chunks_per_sync,
                progress_cb=progress_cb,
            )

        if store.needs_compaction(
            freelist_ratio=cfg.auto_compact_freelist_ratio,
            orphan_threshold=cfg.auto_compact_orphan_threshold,
        ):
            _emit_progress(progress_cb, "compact", "Compacting SQLite and USearch index")
            try:
                store.compact()
            except Exception as exc:
                stats.errors.append(f"auto_compact_error: {exc}")

        if cfg.enable_leann_bridge and stats.indexed_docs > 0:
            try:
                build_leann_export(cfg, store, threshold=_LEANN_DEFAULT_THRESHOLD, run_build=False)
            except Exception as exc:
                stats.errors.append(f"leann_export_error: {exc}")

    stats.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _emit_progress(
        progress_cb,
        "done",
        (
            f"Indexed {stats.indexed_docs}, skipped {stats.skipped_unchanged} "
            f"(stat={stats.stat_skipped}), embedded {stats.embeddings_computed}, deduped {stats.embeddings_deduped}"
        ),
    )
    return stats


def _is_under_dir(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def sync_changed_paths(
    cfg: AppConfig,
    paths: list[Path | str],
    *,
    skip_embeddings: bool = False,
    skip_vision: bool = False,
) -> SyncStats:
    """Incrementally sync only a small set of changed filesystem or harness paths (used by watcher)."""
    t0 = time.perf_counter()
    stats = SyncStats()
    if not paths:
        return stats

    Path(cfg.data_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(cfg.config_dir).expanduser().mkdir(parents=True, exist_ok=True)
    lock_path = Path(cfg.data_dir).expanduser() / "sync.lock"
    try:
        with file_lock(lock_path, blocking=False):
            return _sync_changed_paths_locked(
                cfg=cfg,
                paths=paths,
                stats=stats,
                t0=t0,
                skip_embeddings=skip_embeddings,
                skip_vision=skip_vision,
            )
    except BlockingIOError:
        stats.errors.append("sync_already_in_progress")
        stats.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return stats


def _sync_changed_paths_locked(
    *,
    cfg: AppConfig,
    paths: list[Path | str],
    stats: SyncStats,
    t0: float,
    skip_embeddings: bool,
    skip_vision: bool,
) -> SyncStats:
    atlas = build_or_load_atlas(cfg, refresh=False)
    ag_dir = Path(cfg.harnesses.antigravity_brain_dir).expanduser()
    codex_dir = Path(cfg.harnesses.codex_home_dir).expanduser()
    claude_proj_dir = Path(cfg.harnesses.claude_projects_dir).expanduser()
    claude_hist_path = Path(cfg.harnesses.claude_history_path).expanduser()

    refresh_ag = False
    refresh_codex = False
    refresh_claude = False
    file_candidates: list[Path] = []

    for raw in paths:
        p = Path(raw).expanduser()
        if _is_under_dir(p, ag_dir):
            refresh_ag = True
        elif _is_under_dir(p, codex_dir):
            refresh_codex = True
        elif _is_under_dir(p, claude_proj_dir) or p.resolve(strict=False) == claude_hist_path.resolve(strict=False):
            refresh_claude = True
        else:
            file_candidates.append(p)

    enabled_media = set(cfg.privacy.enabled_media)
    with ChittamStore(cfg) as store:
        existing_stats = store.get_document_stat_index()
        existing_projects = store.get_document_projects()
        vision_fn: Callable[[Path, str], str] | None = None
        if not skip_vision:
            vision_fn, _ = _build_vision_helper(cfg, store, False, stats)

        harness_docs: list[Any] = []
        if refresh_ag:
            harness_docs.extend(load_antigravity_docs(cfg, atlas))
        if refresh_codex:
            harness_docs.extend(load_codex_docs(cfg, atlas))
        if refresh_claude:
            harness_docs.extend(load_claude_code_docs(cfg, atlas))

        for doc in harness_docs:
            if doc.media_type not in enabled_media:
                continue
            stats.scanned_docs += 1
            doc_tier = (
                doc.canonical_tier.value
                if isinstance(doc.canonical_tier, RepoTier)
                else str(doc.canonical_tier)
            )
            prev = existing_stats.get(doc.doc_id)
            if prev is not None and prev[2] == doc.sha256:
                if prev[3] == doc_tier and existing_projects.get(doc.doc_id) == doc.project_id:
                    stats.skipped_unchanged += 1
                    continue
                store.update_document_metadata(
                    doc.doc_id,
                    project_id=doc.project_id,
                    canonical_tier=doc_tier,
                )
                stats.indexed_docs += 1
                continue
            store.upsert_document(doc, _save_vec=False)
            stats.indexed_docs += 1
            stats.chunks_written += len(doc.chunks)

        seen_files: set[Path] = set()
        for p in file_candidates:
            if p in seen_files:
                continue
            seen_files.add(p)
            stats.scanned_docs += 1
            doc, det_id = load_single_file_doc(
                p,
                cfg,
                atlas,
                vision_fn=vision_fn,
                image_cache_lookup=store.get_cached_image_caption,
                stats=stats,
            )
            if doc is None:
                deleted = False
                if det_id:
                    deleted = store.delete_document(det_id, _save_vec=False) or deleted
                for old_id in store.get_doc_ids_by_uri(str(p)):
                    deleted = store.delete_document(old_id, _save_vec=False) or deleted
                continue

            if doc.media_type not in enabled_media:
                continue

            doc_tier = (
                doc.canonical_tier.value
                if isinstance(doc.canonical_tier, RepoTier)
                else str(doc.canonical_tier)
            )
            prev = existing_stats.get(doc.doc_id)
            if prev is not None and prev[2] == doc.sha256:
                if doc.media_type != MediaType.IMAGE or prev[4] == doc.l0_summary:
                    if prev[3] == doc_tier and existing_projects.get(doc.doc_id) == doc.project_id:
                        store.update_document_file_stat(
                            doc.doc_id,
                            doc.metadata.get("file_mtime_ns"),
                            doc.metadata.get("file_size"),
                        )
                        stats.skipped_unchanged += 1
                        continue
                    store.update_document_metadata(
                        doc.doc_id,
                        project_id=doc.project_id,
                        canonical_tier=doc_tier,
                        file_mtime_ns=doc.metadata.get("file_mtime_ns"),
                        file_size=doc.metadata.get("file_size"),
                    )
                    stats.indexed_docs += 1
                    continue

            store.upsert_document(doc, _save_vec=False)
            stats.indexed_docs += 1
            stats.chunks_written += len(doc.chunks)

        store.flush_vector_index()

        if not skip_embeddings and stats.indexed_docs > 0:
            _run_embedding_stage(cfg, store, stats, use_external=False, chunk_limit=256)

    stats.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return stats
