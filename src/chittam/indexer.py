"""Incremental Merkle indexer, multimodal caption cache integration, and LEANN export."""

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

from chittam.adapters import collect_all_documents
from chittam.atlas import build_or_load_atlas, save_atlas
from chittam.config import redact_secrets
from chittam.models import AppConfig, ChunkRecord, MediaType, RepoTier, SyncStats
from chittam.providers import build_embedding_provider, build_vision_provider
from chittam.store import ChittamStore

_EMBED_BATCH_SIZE = 32
_LEANN_DEFAULT_THRESHOLD = 500


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


def sync_memory(
    cfg: AppConfig,
    refresh_atlas: bool = False,
    use_external: bool = False,
    skip_embeddings: bool = False,
    skip_vision: bool = False,
    progress_cb: Callable[..., Any] | None = None,
) -> SyncStats:
    """Run canonical atlas discovery, adapter ingestion, Merkle diffing, and batch embeddings."""
    t0 = time.perf_counter()
    stats = SyncStats()

    Path(cfg.data_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(cfg.config_dir).expanduser().mkdir(parents=True, exist_ok=True)

    lock_path = Path(cfg.data_dir).expanduser() / "sync.lock"
    lock_fh = lock_path.open("w", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stats.errors.append("sync_already_in_progress")
            stats.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return stats

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
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_fh.close()


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
    atlas = build_or_load_atlas(cfg, refresh=refresh_atlas)
    if not cfg.atlas_path.exists() and atlas:
        save_atlas(atlas, cfg.atlas_path)

    with ChittamStore(cfg.db_path) as store:
        vision_fn: Callable[[Path, str], str] | None = None
        vision_model_name = (
            (cfg.external_provider.vision_model if use_external else cfg.local_provider.vision_model)
            or "vision"
        )

        if not skip_vision:
            try:
                vp = build_vision_provider(cfg, use_external=use_external)
                vision_model_name = getattr(vp, "model", None) or getattr(vp, "model_name", None) or vision_model_name

                def _cached_vision(image_path: Path, context_hint: str = "") -> str:
                    sha = _sha256_file(image_path)
                    cached = store.get_cached_image_caption(sha)
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
                    return caption

                vision_fn = _cached_vision
            except Exception as exc:
                stats.errors.append(f"vision_provider_unavailable: {exc}")
                vision_fn = None

        _emit_progress(progress_cb, "collect", "Collecting documents across harnesses and roots")
        docs, collect_stats = collect_all_documents(
            cfg,
            atlas,
            vision_fn=vision_fn,
            image_cache_lookup=store.get_cached_image_caption,
        )

        stats.scanned_docs = len(docs)
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
        fingerprints = store.get_document_fingerprints()
        seen_doc_ids: set[str] = set()

        for doc in docs:
            seen_doc_ids.add(doc.doc_id)
            doc_tier = (
                doc.canonical_tier.value
                if isinstance(doc.canonical_tier, RepoTier)
                else str(doc.canonical_tier)
            )
            existing = fingerprints.get(doc.doc_id)
            if existing is not None:
                ex_sha, ex_tier, ex_l0 = existing
                if ex_sha == doc.sha256 and ex_tier == doc_tier:
                    # Also skip images only when caption hasn't changed (handles --no-vision -> vision upgrade)
                    if doc.media_type != MediaType.IMAGE or ex_l0 == doc.l0_summary:
                        stats.skipped_unchanged += 1
                        continue

            store.upsert_document(doc)
            stats.indexed_docs += 1
            stats.chunks_written += len(doc.chunks)

        # Prune docs no longer present unless collection hit a catastrophic empty-output error
        if docs or not collect_stats.errors:
            store.prune_documents(
                seen_doc_ids,
                preserve_media_types={MediaType.DECISION.value},
            )

        if not skip_embeddings:
            try:
                ep = build_embedding_provider(cfg, use_external=use_external)
                embed_model = (
                    getattr(ep, "model", None)
                    or getattr(ep, "model_name", None)
                    or (
                        cfg.external_provider.embedding_model
                        if use_external and cfg.external_provider.embedding_model
                        else cfg.local_provider.embedding_model
                    )
                )
                missing_chunks: list[ChunkRecord] = store.get_chunks_missing_embeddings(
                    model=embed_model,
                    limit=cfg.privacy.max_embed_chunks_per_sync,
                )
                if missing_chunks:
                    _emit_progress(
                        progress_cb,
                        "embed",
                        f"Computing embeddings for {len(missing_chunks)} chunks",
                    )
                    for start in range(0, len(missing_chunks), _EMBED_BATCH_SIZE):
                        batch = missing_chunks[start : start + _EMBED_BATCH_SIZE]
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
                            )
                            stats.embeddings_computed += written
                        except Exception as exc:
                            stats.errors.append(f"embedding_batch_error@{start}: {exc}")
            except Exception as exc:
                stats.errors.append(f"embedding_provider_unavailable: {exc}")

        if cfg.enable_leann_bridge and stats.indexed_docs > 0:
            try:
                build_leann_export(cfg, store, threshold=_LEANN_DEFAULT_THRESHOLD, run_build=False)
            except Exception as exc:
                stats.errors.append(f"leann_export_error: {exc}")

    stats.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _emit_progress(
        progress_cb,
        "done",
        f"Indexed {stats.indexed_docs}, skipped {stats.skipped_unchanged}, embedded {stats.embeddings_computed}",
    )
    return stats
