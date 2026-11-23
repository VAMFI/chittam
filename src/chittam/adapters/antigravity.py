"""Antigravity brain artifact and transcript adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path

from chittam.atlas import match_repo_for_path, match_repo_from_text
from chittam.config import is_path_excluded, redact_secrets
from chittam.models import AppConfig, CanonicalRepo, DocumentRecord, MediaType, RepoTier
from chittam.adapters import (
    chunk_markdown_or_text,
    make_doc_id,
    make_l0_summary,
    normalize_iso_timestamp,
    sha256_text,
)

_USER_REQUEST_RE = re.compile(r"<USER_REQUEST>\s*([\s\S]*?)\s*</USER_REQUEST>", re.IGNORECASE)
_TAG_STRIP_RE = re.compile(
    r"<(?:ADDITIONAL_METADATA|SYSTEM_MESSAGE|CRITICAL_CONSTRAINT)[^>]*>[\s\S]*?</(?:ADDITIONAL_METADATA|SYSTEM_MESSAGE|CRITICAL_CONSTRAINT)>",
    re.IGNORECASE,
)
_MD_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _clean_user_text(raw: str) -> str:
    m = _USER_REQUEST_RE.search(raw)
    if m:
        return m.group(1).strip()
    cleaned = _TAG_STRIP_RE.sub("", raw)
    return cleaned.strip()


def _extract_turn(obj: dict) -> tuple[str, str, str, str | None] | None:
    """Return (role, text, created_at, cwd) or None if turn has no user/assistant text."""
    typ = str(obj.get("type") or "").upper()
    src = str(obj.get("source") or "").upper()
    role_field = str(obj.get("role") or "").lower()
    cwd = obj.get("cwd") if isinstance(obj.get("cwd"), str) else None
    created_at = str(obj.get("created_at") or obj.get("timestamp") or "")

    content = obj.get("content")
    if content is None:
        content = obj.get("text") or obj.get("response")

    if not isinstance(content, str) or not content.strip():
        return None

    if typ == "USER_INPUT" or src == "USER_EXPLICIT" or role_field == "user":
        cleaned = _clean_user_text(content)
        if not cleaned:
            return None
        return ("User", cleaned, created_at, cwd)

    if typ == "PLANNER_RESPONSE" or (src == "MODEL" and typ not in ("GENERIC",)) or role_field == "assistant":
        cleaned = content.strip()
        if not cleaned:
            return None
        return ("Assistant", cleaned, created_at, cwd)

    return None


def _load_artifact_metadata(md_path: Path) -> tuple[str | None, str | None]:
    meta_path = Path(f"{md_path}.metadata.json")
    if not meta_path.exists():
        return None, None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            updated = data.get("updatedAt") or data.get("updated_at")
            summary = data.get("summary")
            return (str(updated) if updated else None, str(summary) if summary else None)
    except (OSError, json.JSONDecodeError):
        pass
    return None, None


def load_antigravity_docs(cfg: AppConfig, atlas: dict[str, CanonicalRepo]) -> list[DocumentRecord]:
    """Read Antigravity brain markdown artifacts and user/assistant transcript turns."""
    brain_dir = Path(cfg.harnesses.antigravity_brain_dir).expanduser()
    if not brain_dir.exists() or not brain_dir.is_dir():
        return []

    docs: list[DocumentRecord] = []
    try:
        conv_dirs = sorted(brain_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return []

    for conv_dir in conv_dirs:
        if not conv_dir.is_dir() or conv_dir.name.startswith(".") or conv_dir.name == "tempmediaStorage":
            continue
        excluded, _ = is_path_excluded(conv_dir, cfg.privacy)
        if excluded:
            continue

        conv_id = conv_dir.name
        conv_repo: CanonicalRepo | None = None

        # Check transcript first so its repo path context can also inform artifacts.
        tlog_path = conv_dir / ".system_generated" / "logs" / "transcript.jsonl"
        if tlog_path.exists() and tlog_path.is_file():
            t_excl, _ = is_path_excluded(tlog_path, cfg.privacy)
            if not t_excl:
                turns: list[str] = []
                first_user_preview: str | None = None
                latest_ts: str | None = None
                detected_cwd: str | None = None
                try:
                    with tlog_path.open("r", encoding="utf-8", errors="replace") as fh:
                        for raw_line in fh:
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(obj, dict):
                                continue
                            parsed = _extract_turn(obj)
                            if not parsed:
                                continue
                            role, text, ts, cwd = parsed
                            if cwd and not detected_cwd:
                                detected_cwd = cwd
                            if ts:
                                latest_ts = ts
                            if role == "User" and first_user_preview is None:
                                first_user_preview = make_l0_summary(text, max_chars=80)
                            step_idx = obj.get("step_index")
                            step_tag = f" [step {step_idx}]" if step_idx is not None else ""
                            ts_tag = f" ({ts[:19]})" if ts else ""
                            turns.append(f"## {role}{step_tag}{ts_tag}\n{text}")
                except OSError:
                    turns = []

                if turns:
                    full_text = redact_secrets("\n\n".join(turns))
                    if detected_cwd:
                        conv_repo = match_repo_for_path(detected_cwd, atlas)
                    if conv_repo is None:
                        conv_repo = match_repo_from_text(full_text, atlas)
                    proj_id = conv_repo.project_id if conv_repo else None
                    tier = conv_repo.tier if conv_repo else RepoTier.REFERENCE
                    try:
                        mtime = tlog_path.stat().st_mtime
                    except OSError:
                        mtime = None
                    updated_at = normalize_iso_timestamp(latest_ts, fallback_mtime=mtime)
                    title = first_user_preview or f"Antigravity session {conv_id[:8]}"
                    doc_id = make_doc_id("ag_tx", str(tlog_path))
                    chunks = chunk_markdown_or_text(
                        doc_id=doc_id,
                        text=full_text,
                        project_id=proj_id,
                        canonical_tier=tier,
                        path_or_uri=str(tlog_path),
                        updated_at=updated_at,
                    )
                    if chunks:
                        docs.append(
                            DocumentRecord(
                                doc_id=doc_id,
                                uri=str(tlog_path),
                                source_harness="antigravity",
                                media_type=MediaType.TRANSCRIPT,
                                project_id=proj_id,
                                canonical_tier=tier,
                                title=title,
                                l0_summary=make_l0_summary(full_text),
                                sha256=sha256_text(full_text),
                                updated_at=updated_at,
                                metadata={
                                    "conversation_id": conv_id,
                                    "turn_count": len(turns),
                                },
                                chunks=chunks,
                            )
                        )

        # Read top-level .md artifacts in conversation brain folder (skip *.md.resolved* snapshots).
        try:
            md_files = sorted(conv_dir.glob("*.md"), key=lambda p: p.name)
        except OSError:
            md_files = []

        for md_path in md_files:
            if not md_path.is_file():
                continue
            m_excl, _ = is_path_excluded(md_path, cfg.privacy)
            if m_excl:
                continue
            try:
                raw_md = md_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            redacted = redact_secrets(raw_md).strip()
            if not redacted:
                continue

            meta_ts, meta_summary = _load_artifact_metadata(md_path)
            repo = match_repo_from_text(redacted, atlas) or conv_repo
            proj_id = repo.project_id if repo else None
            tier = repo.tier if repo else RepoTier.REFERENCE
            try:
                mtime = md_path.stat().st_mtime
            except OSError:
                mtime = None
            updated_at = normalize_iso_timestamp(meta_ts, fallback_mtime=mtime)
            heading_match = _MD_TITLE_RE.search(redacted)
            title = (
                heading_match.group(1).strip()
                if heading_match
                else f"Antigravity {md_path.stem} ({conv_id[:8]})"
            )
            doc_id = make_doc_id("ag_md", str(md_path))
            chunks = chunk_markdown_or_text(
                doc_id=doc_id,
                text=redacted,
                project_id=proj_id,
                canonical_tier=tier,
                path_or_uri=str(md_path),
                updated_at=updated_at,
            )
            if chunks:
                docs.append(
                    DocumentRecord(
                        doc_id=doc_id,
                        uri=str(md_path),
                        source_harness="antigravity",
                        media_type=MediaType.MARKDOWN,
                        project_id=proj_id,
                        canonical_tier=tier,
                        title=title,
                        l0_summary=make_l0_summary(meta_summary or redacted),
                        sha256=sha256_text(redacted),
                        updated_at=updated_at,
                        metadata={
                            "conversation_id": conv_id,
                            "artifact": md_path.name,
                        },
                        chunks=chunks,
                    )
                )

    return docs
