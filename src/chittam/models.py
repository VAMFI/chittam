"""Shared Pydantic contracts for Chittam."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class MediaType(str, Enum):
    """Supported and future-extensible media modalities."""

    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    TRANSCRIPT = "transcript"
    DECISION = "decision"
    AUDIO = "audio"  # Future adapter extension
    VIDEO = "video"  # Future adapter extension


class RepoTier(str, Enum):
    """Canonical project lifecycle tier for ranking and disambiguation."""

    ACTIVE = "active"
    REFERENCE = "reference"
    ARCHIVE = "archive"


TIER_WEIGHTS: dict[RepoTier, float] = {
    RepoTier.ACTIVE: 1.0,
    RepoTier.REFERENCE: 0.65,
    RepoTier.ARCHIVE: 0.25,
}


class PrivacyPolicy(BaseModel):
    """Configurable privacy and inclusion rules for arbitrary user/dev folders."""

    scan_roots: list[str] = Field(
        default_factory=lambda: [
            "~/Projects",
            "~/Code",
            "~/Documents",
            "~",
        ]
    )
    home_top_level_files_only: bool = True
    enabled_media: list[MediaType] = Field(
        default_factory=lambda: [
            MediaType.CODE,
            MediaType.MARKDOWN,
            MediaType.TEXT,
            MediaType.PDF,
            MediaType.IMAGE,
            MediaType.TRANSCRIPT,
            MediaType.DECISION,
        ]
    )
    exclude_path_globs: list[str] = Field(
        default_factory=lambda: [
            "~/tax",
            "~/tax/**",
            "~/taxes",
            "~/taxes/**",
            "~/Library/**",
            "~/.Trash/**",
            "~/.ssh/**",
            "~/.aws/**",
            "~/.gnupg/**",
            "~/.kube/**",
            "~/.gemini/oauth_creds.json",
            "~/.gemini/mcp-oauth-tokens*.json",
            "~/.gemini/google_accounts.json",
            "~/.claude/.credentials.json",
            "~/.codex/auth.json",
            "~/.codex/auth.lock",
            "~/.codex/auth.json.lock",
        ]
    )
    exclude_dir_names: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".dart_tool",
            ".gradle",
            "build",
            "dist",
            "target",
            ".next",
            ".nuxt",
            "Pods",
            "DerivedData",
            ".cache",
            "tax",
            "taxes",
        ]
    )
    exclude_file_patterns: list[str] = Field(
        default_factory=lambda: [
            ".env",
            ".env.*",
            "*.env",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "*.kdbx",
            "id_rsa*",
            "id_ed25519*",
            "*credentials*",
            "*secret*",
            "auth.json",
            "oauth_creds.json",
            "mcp-oauth-tokens*",
            "*.keystore",
            "*.jks",
            "GoogleService-Info.plist",
            "google-services.json",
        ]
    )
    max_text_file_bytes: int = 512_000
    max_pdf_bytes: int = 25_000_000
    max_pdf_pages: int = 40
    min_image_bytes: int = 10_240
    max_image_bytes: int = 20_000_000
    max_images_per_sync: int = 40
    max_code_files_per_repo: int = 250
    max_embed_chunks_per_sync: int | None = None
    embed_tiers: list[RepoTier] = Field(
        default_factory=lambda: [RepoTier.ACTIVE, RepoTier.REFERENCE]
    )
    embed_always_media: list[MediaType] = Field(
        default_factory=lambda: [
            MediaType.IMAGE,
            MediaType.TRANSCRIPT,
            MediaType.PDF,
            MediaType.DECISION,
        ]
    )


class LocalProviderConfig(BaseModel):
    """Local Ollama runtime configuration."""

    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "embeddinggemma:latest"
    embedding_dim: int = 768
    matryoshka_dim: int | None = 256
    vector_dtype: Literal["i8", "f16", "f32"] = "i8"
    vision_model: str = "qwen3-vl:2b"
    rerank_model: str | None = None
    rerank_top_k: int = 20
    timeout_seconds: float = 90.0


class ExternalProviderConfig(BaseModel):
    """Opt-in external cloud / remote API configuration (disabled by default)."""

    enabled: bool = False
    user_consented: bool = False
    provider_type: Literal["openai_compatible", "gemini", "remote_ollama"] = "openai_compatible"
    base_url: str = ""
    api_key_env: str = ""
    embedding_model: str = ""
    vision_model: str = ""
    rerank_model: str = ""


class HarnessesConfig(BaseModel):
    """Harness configuration paths for Antigravity, Codex, and Claude Code."""

    antigravity_mcp_path: str = "~/.gemini/config/mcp_config.json"
    codex_config_path: str = "~/.codex/config.toml"
    claude_config_path: str = "~/.claude.json"
    antigravity_rules_path: str = "~/.gemini/GEMINI.md"
    claude_rules_path: str = "~/.claude/CLAUDE.md"
    codex_rules_path: str = "~/.codex/AGENTS.md"
    antigravity_brain_dir: str = "~/.gemini/antigravity/brain"
    codex_home_dir: str = "~/.codex"
    codex_legacy_memories_dirs: list[str] = Field(default_factory=list)
    claude_projects_dir: str = "~/.claude/projects"
    claude_history_path: str = "~/.claude/history.jsonl"


class AppConfig(BaseModel):
    """Top-level Chittam application configuration."""

    data_dir: str = "~/.local/share/chittam"
    config_dir: str = "~/.config/chittam"
    privacy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
    local_provider: LocalProviderConfig = Field(default_factory=LocalProviderConfig)
    external_provider: ExternalProviderConfig = Field(default_factory=ExternalProviderConfig)
    harnesses: HarnessesConfig = Field(default_factory=HarnessesConfig)
    enable_leann_bridge: bool = True
    enable_codegraph: bool = True
    vector_backend: Literal["usearch", "numpy"] = "usearch"

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir).expanduser() / "memory.db"

    @property
    def vector_index_path(self) -> Path:
        return Path(self.data_dir).expanduser() / "vectors.usearch"

    @property
    def atlas_path(self) -> Path:
        return Path(self.config_dir).expanduser() / "canonical-projects.yaml"

    @property
    def chittam_md_path(self) -> Path:
        return Path(self.config_dir).expanduser() / "CHITTAM.md"

    @property
    def backup_dir(self) -> Path:
        return Path(self.data_dir).expanduser() / "backups"


class CanonicalRepo(BaseModel):
    """Canonical classification metadata for a discovered git repository."""

    project_id: str
    family: str
    name: str
    path: str
    tier: RepoTier = RepoTier.REFERENCE
    last_commit_at: str | None = None
    branch: str | None = None
    remote_url: str | None = None
    manual_override: bool = False
    notes: str = ""


class DocumentRecord(BaseModel):
    """Normalized source document ingested by an adapter."""

    doc_id: str
    uri: str
    source_harness: str
    media_type: MediaType
    project_id: str | None = None
    canonical_tier: RepoTier = RepoTier.REFERENCE
    title: str
    l0_summary: str
    sha256: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: list[ChunkRecord] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """Chunk unit stored in FTS5 + vector index."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    contextual_header: str
    content: str
    symbol_name: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


DocumentRecord.model_rebuild()


class DecisionRecord(BaseModel):
    """Bitemporal architectural or workflow decision."""

    decision_id: str
    project_id: str
    topic: str
    statement: str
    source_harness: str = "manual"
    valid_from: str = Field(default_factory=lambda: datetime.now().astimezone().isoformat())
    valid_to: str | None = None
    superseded_by: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now().astimezone().isoformat())


class SearchResult(BaseModel):
    """Compact ranked retrieval item returned by search and MCP tools."""

    chunk_id: str
    doc_id: str
    uri: str
    title: str
    source_harness: str
    media_type: MediaType
    project_id: str | None = None
    canonical_tier: RepoTier = RepoTier.REFERENCE
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    snippet: str
    l0_summary: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyncStats(BaseModel):
    """Summary counters produced by an incremental sync run."""

    scanned_docs: int = 0
    indexed_docs: int = 0
    skipped_unchanged: int = 0
    skipped_excluded: int = 0
    chunks_written: int = 0
    embeddings_computed: int = 0
    embeddings_deduped: int = 0
    stat_skipped: int = 0
    images_captioned: int = 0
    images_cached: int = 0
    errors: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0
