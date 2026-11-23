"""Retrieval quality, latency, and privacy-leakage benchmark for Chittam.

Two ways to run it:

* ``chittam benchmark --synthetic`` builds a throwaway fixture corpus (pushed through the
  real privacy gate) with a deterministic embedder. It needs no Ollama, no index, and no
  personal data, so it is reproducible on any machine and in CI.
* ``chittam benchmark`` runs against your local index. Because every machine holds
  different files, put queries that match *your* data in
  ``~/.config/chittam/benchmark_cases.yaml`` (see ``docs/HOW_TO.md``). Without that file
  the generic cases are used, which only make sense against the synthetic corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from chittam.config import is_path_excluded, load_config
from chittam.models import (
    AppConfig,
    ChunkRecord,
    DocumentRecord,
    MediaType,
    RepoTier,
    SearchResult,
)
from chittam.providers import (
    DEFAULT_RERANK_MODEL,
    EmbeddingProvider,
    FlashRankRerankProvider,
    RerankProvider,
    build_query_embedding_provider,
    build_rerank_provider,
)
from chittam.store import ChittamStore

DEFAULT_CASES_PATH = Path("~/.config/chittam/benchmark_cases.yaml").expanduser()

_PRIVATE_LEAK_RE = re.compile(
    r"(?:/tax/|/taxes/|oauth_creds\.json|\.credentials\.json|"
    r"/\.codex/auth\.json|/\.ssh/|/\.aws/|/\.gnupg/|(?:/|^)\.env(?:\.|$))",
    re.IGNORECASE,
)

Matcher = Callable[[list[SearchResult]], "int | None"]


@dataclass(frozen=True)
class BenchmarkCase:
    """Single benchmark query specification."""

    query_id: str
    category: str
    query: str
    mode: str = "search"  # "search" | "threads" | "image" | "privacy"
    harness: str | None = None
    matcher: Matcher | None = None
    notes: str = ""


# --------------------------------------------------------------------------- matchers


def _haystack(item: SearchResult) -> str:
    return " ".join(
        [item.uri or "", item.title or "", item.project_id or "", item.l0_summary or "", item.snippet or ""]
    ).lower()


def _first_match(
    results: list[SearchResult],
    predicate: Callable[[SearchResult], bool],
    top_k: int = 5,
) -> int | None:
    for idx, item in enumerate(results[:top_k], start=1):
        if predicate(item):
            return idx
    return None


def match_any(*needles: str, media: MediaType | None = None) -> Matcher:
    """Pass when any top-5 result contains one of ``needles`` (optionally of ``media`` type)."""
    lowered = tuple(n.lower() for n in needles)

    def _fn(results: list[SearchResult]) -> int | None:
        return _first_match(
            results,
            lambda r: (media is None or r.media_type == media) and any(n in _haystack(r) for n in lowered),
        )

    return _fn


def match_non_archive(*needles: str) -> Matcher:
    """Pass only when the *first* result mentioning ``needles`` is not an archived copy.

    This is what "active work outranks stale duplicates" means in practice.
    """
    lowered = tuple(n.lower() for n in needles)

    def _fn(results: list[SearchResult]) -> int | None:
        for idx, item in enumerate(results[:5], start=1):
            if any(n in _haystack(item) for n in lowered):
                return idx if item.canonical_tier != RepoTier.ARCHIVE else None
        return None

    return _fn


# ------------------------------------------------------------------------------ cases


def build_benchmark_cases() -> list[BenchmarkCase]:
    """Return the 15 generic evaluation queries that pair with the synthetic corpus."""
    return [
        # Cross-repo / project discovery
        BenchmarkCase("Q01_repo_recipes", "cross_repo", "offline recipe app SwiftUI Kotlin SQLite",
                      matcher=match_any("acme-recipes")),
        BenchmarkCase("Q02_repo_gitscope", "cross_repo", "local-first Git repository explorer",
                      matcher=match_any("gitscope")),
        BenchmarkCase("Q03_repo_substrate", "cross_repo", "portable agent installation guide",
                      matcher=match_any("agent-substrate")),
        BenchmarkCase("Q04_repo_chittam", "cross_repo", "multimodal semantic memory canonical atlas MCP",
                      matcher=match_any("chittam")),
        # Active-vs-archived disambiguation
        BenchmarkCase("Q05_tier_geo", "tier_disambiguation", "geo-atlas location frontend backend docker-compose",
                      matcher=match_non_archive("geo-atlas")),
        BenchmarkCase("Q06_tier_recipes", "tier_disambiguation", "recipe scaling offline iOS Android",
                      matcher=match_non_archive("acme-recipes")),
        # Cross-harness thread recall
        BenchmarkCase("Q07_thread_antigravity", "thread_recall", "editorial market research roadmap",
                      mode="threads", harness="antigravity", matcher=match_any("editorial")),
        BenchmarkCase("Q08_thread_codex", "thread_recall", "agent substrate recipe workflow memory",
                      mode="threads", harness="codex", matcher=match_any("agent-substrate", "workflow")),
        BenchmarkCase("Q09_thread_claude", "thread_recall", "web crawler citation audit",
                      mode="threads", harness="claude", matcher=match_any("citation")),
        # Docs & setup notes
        BenchmarkCase("Q10_docs_server", "docs_setup", "home server GPU Whisper speech diagnostic",
                      matcher=match_any("whisper")),
        BenchmarkCase("Q11_docs_overview", "docs_setup", "orchestrator agents notes-engine voice assistant",
                      matcher=match_any("notes-engine")),
        # Multimodal images / screenshots
        BenchmarkCase("Q12_image_logo", "multimodal_image", "company logo architecture diagram",
                      mode="image", matcher=match_any("logo", media=MediaType.IMAGE)),
        BenchmarkCase("Q13_image_ui", "multimodal_image", "mobile login screen UI screenshot",
                      mode="image", matcher=match_any("login", media=MediaType.IMAGE)),
        # Negative privacy assertions
        BenchmarkCase("Q14_privacy_tax", "negative_privacy", "tax return filing identity document",
                      mode="privacy"),
        BenchmarkCase("Q15_privacy_secrets", "negative_privacy", ".env auth credentials oauth_creds api key",
                      mode="privacy"),
    ]


def load_cases_from_yaml(path: Path) -> list[BenchmarkCase]:
    """Load custom cases. Each entry: ``id, category, query, mode?, harness?, expect_any?,
    expect_media?, require_non_archive?``."""
    raw = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or []
    items = raw.get("cases", []) if isinstance(raw, dict) else raw
    cases: list[BenchmarkCase] = []
    for i, item in enumerate(items, start=1):
        mode = str(item.get("mode", "search"))
        needles = [str(n) for n in item.get("expect_any", [])]
        matcher: Matcher | None = None
        if mode != "privacy" and needles:
            if item.get("require_non_archive"):
                matcher = match_non_archive(*needles)
            else:
                media = MediaType(item["expect_media"]) if item.get("expect_media") else (
                    MediaType.IMAGE if mode == "image" else None
                )
                matcher = match_any(*needles, media=media)
        cases.append(
            BenchmarkCase(
                query_id=str(item.get("id", f"C{i:02d}")),
                category=str(item.get("category", "custom")),
                query=str(item["query"]),
                mode=mode,
                harness=item.get("harness"),
                matcher=matcher,
            )
        )
    return cases


# --------------------------------------------------------------------- privacy check


def _is_leaked_result(item: SearchResult, cfg: AppConfig) -> bool:
    uri = item.uri or ""
    if uri and is_path_excluded(uri, cfg.privacy)[0]:
        return True
    return bool(_PRIVATE_LEAK_RE.search(uri))


# ------------------------------------------------------------------ synthetic corpus


class HashingEmbedder:
    """Deterministic bag-of-words embedder (stable across processes, no model needed)."""

    model_name = "hashing-embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed_query(self, query: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w+", query.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed_query(t) for t in texts])


_FIXTURES: list[tuple[str, str, MediaType, str | None, RepoTier, str, str, int]] = [
    # (uri, harness, media, project_id, tier, title, body, age_days)
    ("projects/acme-recipes/README.md", "files", MediaType.MARKDOWN, "acme-recipes", RepoTier.ACTIVE,
     "Acme Recipes", "Offline-first recipe app with SwiftUI iOS, Kotlin Android, SQLite and recipe scaling.", 2),
    ("archive/acme-recipes-old/README.md", "files", MediaType.MARKDOWN, "acme-recipes-old", RepoTier.ARCHIVE,
     "Acme Recipes (old)", "Old prototype of the offline recipe app with recipe scaling for iOS and Android.", 400),
    ("projects/gitscope/docs/overview.md", "files", MediaType.MARKDOWN, "gitscope", RepoTier.ACTIVE,
     "GitScope", "GitScope is a local-first Git repository explorer and atlas.", 5),
    ("projects/agent-substrate/INSTALL.md", "files", MediaType.MARKDOWN, "agent-substrate", RepoTier.ACTIVE,
     "Agent Substrate install", "Portable agent installation guide with reversible configuration.", 3),
    ("projects/chittam/README.md", "files", MediaType.MARKDOWN, "chittam", RepoTier.ACTIVE,
     "Chittam", "Chittam multimodal semantic memory with a canonical atlas, exposed over MCP.", 1),
    ("projects/geo-atlas/docker-compose.yml", "files", MediaType.CODE, "geo-atlas", RepoTier.ACTIVE,
     "geo-atlas stack", "geo-atlas location service: frontend, backend, docker-compose.", 4),
    ("archive/geo-atlas-old/docker-compose.yml", "files", MediaType.CODE, "geo-atlas-old", RepoTier.ARCHIVE,
     "geo-atlas old stack", "geo-atlas location service: frontend, backend, docker-compose.", 500),
    ("agent-logs/antigravity/conv-1/market_research.md", "antigravity", MediaType.TRANSCRIPT, "acme-recipes",
     RepoTier.ACTIVE, "Editorial roadmap", "Editorial cookbook market research and product roadmap discussion.", 6),
    ("agent-logs/codex/thread-1", "codex", MediaType.TRANSCRIPT, "agent-substrate", RepoTier.ACTIVE,
     "Codex memory", "Agent substrate recipe workflow memory and lessons learned.", 7),
    ("agent-logs/claude/session-1.jsonl", "claude_code", MediaType.TRANSCRIPT, "citation-audit", RepoTier.ACTIVE,
     "Citation audit session", "Web crawler citation audit session with retrieval latency checks.", 8),
    ("notes/home-server-setup.md", "files", MediaType.MARKDOWN, None, RepoTier.REFERENCE,
     "Home server notes", "Home server GPU Whisper speech-to-text diagnostic notes.", 30),
    ("notes/projects-overview.md", "files", MediaType.MARKDOWN, None, RepoTier.REFERENCE,
     "Projects overview", "Orchestrator agents, notes-engine and voice assistant overview.", 60),
    ("media/company-logo.png", "files", MediaType.IMAGE, "brand", RepoTier.ACTIVE,
     "company-logo.png", "Company logo next to a system architecture diagram.", 10),
    ("media/login-screen.png", "files", MediaType.IMAGE, "acme-recipes", RepoTier.ACTIVE,
     "login-screen.png", "Mobile login screen UI screenshot with email and password fields.", 9),
    # The following must be blocked by the privacy gate and never reach the index.
    ("tax/2025_return.txt", "files", MediaType.TEXT, None, RepoTier.REFERENCE,
     "2025 tax return", "Tax return filing with identity document numbers.", 20),
    ("projects/acme-recipes/.env", "files", MediaType.TEXT, "acme-recipes", RepoTier.ACTIVE,
     ".env", "API key and auth credentials for the recipe backend.", 2),
    (".codex/auth.json", "files", MediaType.TEXT, None, RepoTier.REFERENCE,
     "auth.json", "oauth_creds auth credentials token", 2),
]


def build_synthetic_corpus(
    store: ChittamStore,
    cfg: AppConfig,
    embedder: EmbeddingProvider,
    home: Path,
) -> dict[str, int]:
    """Insert fixture documents through the privacy gate; return inserted/blocked counts."""
    now = datetime.now(timezone.utc)
    inserted = blocked = 0
    for idx, (rel, harness, media, project, tier, title, body, age) in enumerate(_FIXTURES):
        uri = str(home / rel)
        if is_path_excluded(uri, cfg.privacy)[0]:
            blocked += 1
            continue
        updated_at = (now - timedelta(days=age)).isoformat()
        doc_id = f"synthetic:{idx}"
        header = f"[{project or 'misc'} | {tier.value} | {uri} | {updated_at[:10]}]"
        doc = DocumentRecord(
            doc_id=doc_id,
            uri=uri,
            source_harness=harness,
            media_type=media,
            project_id=project,
            canonical_tier=tier,
            title=title,
            l0_summary=body,
            sha256=hashlib.sha256(body.encode()).hexdigest(),
            updated_at=updated_at,
            chunks=[
                ChunkRecord(
                    chunk_id=f"{doc_id}#c0",
                    doc_id=doc_id,
                    chunk_index=0,
                    contextual_header=header,
                    content=f"{title}\n{body}",
                    token_estimate=max(1, len(body) // 4),
                )
            ],
        )
        store.upsert_document(doc)
        vec = embedder.embed_texts([f"{header}\n{title}\n{body}"])
        store.upsert_chunk_embeddings([f"{doc_id}#c0"], getattr(embedder, "model_name", "synthetic"), vec)
        inserted += 1
    return {"inserted": inserted, "blocked_by_privacy_gate": blocked}


# ---------------------------------------------------------------------------- runner


def run_benchmark(
    cfg: AppConfig | None = None,
    *,
    store: ChittamStore | None = None,
    embedder: EmbeddingProvider | None = None,
    use_embeddings: bool = True,
    use_rerank: bool | None = None,
    reranker: RerankProvider | None = None,
    top_k: int = 5,
    cases: list[BenchmarkCase] | None = None,
) -> dict[str, Any]:
    """Run benchmark cases and return Recall@5, MRR, latency, and privacy-leakage metrics."""
    resolved_cfg = cfg or load_config()
    owns_store = store is None
    active_store = store or ChittamStore(resolved_cfg)

    if cases is None:
        cases = load_cases_from_yaml(DEFAULT_CASES_PATH) if DEFAULT_CASES_PATH.exists() else build_benchmark_cases()

    active_embedder = embedder
    embedder_status = "provided" if embedder is not None else "disabled"
    if active_embedder is None and use_embeddings:
        try:
            active_embedder = build_query_embedding_provider(
                resolved_cfg,
                index_meta=active_store.get_index_meta(),
            )
            embedder_status = "active"
        except Exception as exc:  # pragma: no cover - defensive fallback
            active_embedder = None
            embedder_status = f"unavailable:{exc}"

    active_reranker = reranker
    rerank_status = "provided" if reranker is not None else "disabled"
    if active_reranker is None and use_rerank is not False:
        active_reranker = build_rerank_provider(resolved_cfg)
        if active_reranker is None and use_rerank is True:
            try:
                import flashrank  # noqa: F401

                active_reranker = FlashRankRerankProvider(
                    model_name=DEFAULT_RERANK_MODEL,
                    cache_dir=Path(resolved_cfg.data_dir).expanduser() / "models" / "flashrank",
                )
            except Exception as exc:
                rerank_status = f"unavailable:{exc}"
        if active_reranker is not None:
            try:
                active_reranker.rerank("warmup", ["warmup passage one", "warmup passage two"], top_k=1)
                rerank_status = f"active:{getattr(active_reranker, 'model_name', DEFAULT_RERANK_MODEL)}"
            except Exception as exc:
                rerank_status = f"unavailable:{exc}"
                active_reranker = None

    per_query: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    positive_hits = 0
    reciprocal_ranks: list[float] = []
    privacy_leakage = 0

    try:
        for case in cases:
            t0 = time.perf_counter()
            q_vec: np.ndarray | None = None
            embed_error: str | None = None
            if active_embedder is not None:
                try:
                    q_vec = active_embedder.embed_query(case.query)
                except Exception as exc:
                    embed_error = str(exc)

            rerank_k = resolved_cfg.local_provider.rerank_top_k
            if case.mode == "threads":
                results = active_store.recall_threads(
                    case.query,
                    query_vec=q_vec,
                    harness=case.harness,
                    limit=top_k,
                    reranker=active_reranker,
                    rerank_top_k=rerank_k,
                )
            elif case.mode == "image":
                results = active_store.search(
                    case.query,
                    query_vec=q_vec,
                    limit=top_k,
                    media_type=MediaType.IMAGE,
                    reranker=active_reranker,
                    rerank_top_k=rerank_k,
                )
            else:
                results = active_store.search(
                    case.query,
                    query_vec=q_vec,
                    limit=top_k,
                    reranker=active_reranker,
                    rerank_top_k=rerank_k,
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            leaked = [r.uri for r in results if _is_leaked_result(r, resolved_cfg)]
            privacy_leakage += len(leaked)

            rank: int | None = None
            if case.mode == "privacy":
                passed = not leaked
            else:
                rank = case.matcher(results) if case.matcher else (1 if results else None)
                passed = rank is not None and not leaked
                if rank is not None:
                    positive_hits += 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

            per_query.append(
                {
                    "query_id": case.query_id,
                    "category": case.category,
                    "mode": case.mode,
                    "query": case.query,
                    "passed": passed,
                    "rank": rank,
                    "latency_ms": round(elapsed_ms, 2),
                    "result_count": len(results),
                    "top_uris": [r.uri for r in results[:3]],
                    "leaked_uris": leaked,
                    "embed_error": embed_error,
                }
            )
        store_stats = active_store.get_stats()
    finally:
        if owns_store:
            active_store.close()

    pos_count = sum(1 for c in cases if c.mode != "privacy")
    return {
        "total_queries": len(cases),
        "positive_queries": pos_count,
        "privacy_queries": len(cases) - pos_count,
        "Recall@5": round(positive_hits / pos_count, 4) if pos_count else 0.0,
        "MRR": round(sum(reciprocal_ranks) / pos_count, 4) if pos_count else 0.0,
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0,
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
        "privacy_leakage": privacy_leakage,
        "embedder_status": embedder_status,
        "rerank_used": active_reranker is not None,
        "rerank_status": rerank_status,
        "vector_backend": store_stats.get("vector_backend"),
        "vector_dim": store_stats.get("vector_dim"),
        "vector_dtype": store_stats.get("vector_dtype"),
        "db_size_bytes": store_stats.get("db_size_bytes"),
        "vector_index_bytes": store_stats.get("vector_index_bytes"),
        "results": per_query,
    }


def run_synthetic_benchmark() -> dict[str, Any]:
    """Build a throwaway fixture index and run the generic 15 cases against it."""
    with tempfile.TemporaryDirectory(prefix="chittam-bench-") as td:
        root = Path(td)
        cfg = AppConfig(data_dir=str(root / "data"), config_dir=str(root / "config"))
        embedder = HashingEmbedder()
        with ChittamStore(cfg) as store:
            corpus = build_synthetic_corpus(store, cfg, embedder, home=root / "u")
            report = run_benchmark(cfg, store=store, embedder=embedder, cases=build_benchmark_cases())
        report["mode"] = "synthetic"
        report["corpus"] = corpus
        return report


def main() -> None:
    """Run the synthetic benchmark and print JSON metrics."""
    print(json.dumps(run_synthetic_benchmark(), indent=2))


if __name__ == "__main__":
    main()
