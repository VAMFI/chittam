"""Health diagnostics for Chittam storage, USearch index, Ollama, reranker, TCC, and harnesses."""

from __future__ import annotations

import fcntl
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from chittam.config import USER_CONFIG_FILE, is_path_excluded
from chittam.harnesses import resolve_codegraph_command
from chittam.models import AppConfig
from chittam.store import ChittamStore
from chittam.vector_index import USEARCH_AVAILABLE, USearchVectorIndex


def _check_config(cfg: AppConfig) -> dict[str, Any]:
    cfg_env = os.environ.get("CHITTAM_CONFIG")
    cfg_path = Path(cfg_env).expanduser() if cfg_env else USER_CONFIG_FILE
    exists = cfg_path.exists()
    return {
        "name": "config",
        "status": "ok" if exists else "warn",
        "detail": (
            f"{cfg_path} (dim={cfg.local_provider.matryoshka_dim or cfg.local_provider.embedding_dim}, "
            f"dtype={cfg.local_provider.vector_dtype}, backend={cfg.vector_backend})"
        ),
        "hint": None if exists else "Run `chittam init` to create ~/.config/chittam/config.yaml",
    }


def _check_database_and_vectors(cfg: AppConfig) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not cfg.db_path.exists():
        checks.append(
            {
                "name": "database",
                "status": "warn",
                "detail": f"Database not found at {cfg.db_path}",
                "hint": "Run `chittam init && chittam sync`",
            }
        )
        checks.append(
            {
                "name": "vector_index",
                "status": "ok" if USEARCH_AVAILABLE else "warn",
                "detail": f"backend={cfg.vector_backend} usearch_installed={USEARCH_AVAILABLE}",
                "hint": None if USEARCH_AVAILABLE else "Install `usearch` or set vector_backend=numpy",
            }
        )
        return checks

    with ChittamStore(cfg) as store:
        stats = store.get_stats()

    legacy = int(stats.get("legacy_embeddings", 0))
    db_mb = round(int(stats["db_size_bytes"]) / (1024 * 1024), 2)
    db_status = "warn" if legacy > 0 else "ok"
    checks.append(
        {
            "name": "database",
            "status": db_status,
            "detail": (
                f"{cfg.db_path} ({db_mb} MiB, docs={stats['documents']}, chunks={stats['chunks']}, "
                f"embeddings={stats['embeddings']}, unique={stats['unique_embeddings']}, "
                f"images={stats['cached_images']}, decisions={stats['active_decisions']}/{stats['decisions']})"
            ),
            "hint": "Run `chittam sync && chittam compact` to migrate legacy vectors" if legacy > 0 else None,
        }
    )

    emb_count = int(stats.get("embeddings", 0))
    vec_path = cfg.vector_index_path
    vec_mb = round(int(stats.get("vector_index_bytes", 0)) / (1024 * 1024), 2)
    dim = stats.get("vector_dim")
    dtype = stats.get("vector_dtype")

    if cfg.vector_backend == "numpy":
        checks.append(
            {
                "name": "vector_index",
                "status": "ok",
                "detail": f"backend=numpy (entries={emb_count}, dim={dim}, dtype={dtype})",
                "hint": None,
            }
        )
    elif not USEARCH_AVAILABLE:
        checks.append(
            {
                "name": "vector_index",
                "status": "warn",
                "detail": "usearch package not available; falling back to numpy",
                "hint": "Run `uv sync` inside the Chittam repo",
            }
        )
    elif emb_count == 0:
        checks.append(
            {
                "name": "vector_index",
                "status": "ok",
                "detail": f"{vec_path} (empty, 0 vector entries)",
                "hint": None,
            }
        )
    elif not vec_path.exists() or not dim:
        checks.append(
            {
                "name": "vector_index",
                "status": "warn",
                "detail": f"{vec_path} missing while SQLite has {emb_count} vector entries",
                "hint": "Run `chittam reindex-vectors` to rebuild vectors.usearch offline",
            }
        )
    else:
        idx = USearchVectorIndex(vec_path, dim=int(dim), dtype=str(dtype or "i8"))
        idx_count = idx.count()
        matched = idx_count == emb_count
        checks.append(
            {
                "name": "vector_index",
                "status": "ok" if matched else "warn",
                "detail": (
                    f"{vec_path} ({vec_mb} MiB, usearch={idx_count}, sqlite={emb_count}, "
                    f"dim={dim}, dtype={dtype})"
                ),
                "hint": None if matched else "Run `chittam reindex-vectors` to sync USearch with SQLite",
            }
        )
    return checks


def _check_ollama(cfg: AppConfig) -> dict[str, Any]:
    base_url = cfg.local_provider.ollama_base_url.rstrip("/")
    want_emb = cfg.local_provider.embedding_model
    want_vis = cfg.local_provider.vision_model
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        resp.raise_for_status()
        models = {str(m.get("name", "")) for m in resp.json().get("models", [])}
        emb_ok = any(m == want_emb or m.startswith(want_emb.split(":")[0]) for m in models)
        vis_ok = any(m == want_vis or m.startswith(want_vis.split(":")[0]) for m in models)
        missing = [
            label
            for label, ok in ((want_emb, emb_ok), (want_vis, vis_ok))
            if label and not ok
        ]
        if missing:
            return {
                "name": "ollama",
                "status": "warn",
                "detail": f"{base_url} reachable, missing models: {', '.join(missing)}",
                "hint": f"Run `ollama pull {missing[0]}`",
            }
        return {
            "name": "ollama",
            "status": "ok",
            "detail": f"{base_url} reachable ({want_emb}, {want_vis} available)",
            "hint": None,
        }
    except Exception as exc:
        return {
            "name": "ollama",
            "status": "warn",
            "detail": f"Cannot reach Ollama at {base_url}: {exc}",
            "hint": "Start Ollama (`ollama serve`); BM25 keyword search continues working",
        }


def _check_reranker(cfg: AppConfig) -> dict[str, Any]:
    installed = importlib.util.find_spec("flashrank") is not None
    configured = (cfg.local_provider.rerank_model or "").strip()
    if configured and not installed:
        return {
            "name": "reranker",
            "status": "warn",
            "detail": f"rerank_model={configured} set, but flashrank is not installed",
            "hint": "Install optional extra: `uv sync --extra rerank`",
        }
    if installed:
        return {
            "name": "reranker",
            "status": "ok",
            "detail": (
                f"flashrank installed (configured={configured or 'off; pass --rerank to enable'})"
            ),
            "hint": None,
        }
    return {
        "name": "reranker",
        "status": "ok",
        "detail": "optional chittam[rerank] not installed (disabled)",
        "hint": None,
    }


def _check_scan_roots(cfg: AppConfig) -> dict[str, Any]:
    unreadable: list[str] = []
    missing: list[str] = []
    readable = 0
    for raw in cfg.privacy.scan_roots:
        p = Path(raw).expanduser()
        excl, _ = is_path_excluded(p, cfg.privacy)
        if excl:
            continue
        if not p.exists():
            missing.append(raw)
            continue
        try:
            with os.scandir(p) as it:
                for _ in zip(range(1), it):
                    break
            readable += 1
        except PermissionError:
            unreadable.append(raw)
        except OSError as exc:
            unreadable.append(f"{raw} ({exc})")

    if unreadable:
        return {
            "name": "scan_roots",
            "status": "warn",
            "detail": f"{readable} readable; TCC/permission blocked: {', '.join(unreadable)}",
            "hint": "Grant Full Disk Access to your terminal/Python or remove restricted paths from scan_roots",
        }
    detail = f"{readable} configured roots readable"
    if missing:
        detail += f" (not present: {', '.join(missing)})"
    return {
        "name": "scan_roots",
        "status": "ok",
        "detail": detail,
        "hint": None,
    }


def _check_harnesses(cfg: AppConfig) -> dict[str, Any]:
    wired: list[str] = []
    ag = Path(cfg.harnesses.antigravity_mcp_path).expanduser()
    if ag.exists() and '"chittam"' in ag.read_text(encoding="utf-8", errors="replace"):
        wired.append("antigravity")
    codex = Path(cfg.harnesses.codex_config_path).expanduser()
    if codex.exists() and "[mcp_servers.chittam]" in codex.read_text(encoding="utf-8", errors="replace"):
        wired.append("codex")
    claude = Path(cfg.harnesses.claude_config_path).expanduser()
    if claude.exists() and '"chittam"' in claude.read_text(encoding="utf-8", errors="replace"):
        wired.append("claude_code")

    cg = resolve_codegraph_command(cfg)
    status = "ok" if wired else "warn"
    return {
        "name": "harnesses",
        "status": status,
        "detail": f"chittam wired in: {', '.join(wired) or 'none'} | codegraph={cg or 'not found'}",
        "hint": None if wired else "Run `chittam install-harnesses`",
    }


def _check_background_sync(cfg: AppConfig) -> dict[str, Any]:
    lock_path = Path(cfg.data_dir).expanduser() / "sync.lock"
    sync_active = False
    if lock_path.exists():
        try:
            with lock_path.open("r", encoding="utf-8") as fh:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except BlockingIOError:
                    sync_active = True
        except OSError:
            pass

    if sys.platform == "darwin":
        from chittam.launchd import launchd_status

        st = launchd_status()
        installed = bool(st.get("installed"))
        loaded = bool(st.get("loaded"))
        detail = (
            f"launchd installed={installed} loaded={loaded} "
            f"interval={st.get('start_interval')}s active_sync={sync_active}"
        )
        return {
            "name": "background_sync",
            "status": "ok" if loaded else "warn",
            "detail": detail,
            "hint": None if loaded else "Run `chittam install-launchd` or `chittam watch`",
        }
    return {
        "name": "background_sync",
        "status": "ok",
        "detail": f"platform={sys.platform} active_sync={sync_active} (use `chittam watch` or cron/systemd)",
        "hint": None,
    }


def run_doctor(cfg: AppConfig, *, check_ollama: bool = True) -> dict[str, Any]:
    """Run non-destructive health diagnostics across all Chittam components."""
    checks: list[dict[str, Any]] = [_check_config(cfg)]
    checks.extend(_check_database_and_vectors(cfg))
    if check_ollama:
        checks.append(_check_ollama(cfg))
    checks.append(_check_reranker(cfg))
    checks.append(_check_scan_roots(cfg))
    checks.append(_check_harnesses(cfg))
    checks.append(_check_background_sync(cfg))

    errors = sum(1 for c in checks if c["status"] == "error")
    warnings = sum(1 for c in checks if c["status"] == "warn")
    return {
        "healthy": errors == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
