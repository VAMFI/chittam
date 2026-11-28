"""Antigravity brain artifact and transcript adapter."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from chittam.atlas import RepoPathIndex, match_repo_for_path, match_repo_from_text
from chittam.config import is_path_excluded, redact_secrets
from chittam.models import AppConfig, CanonicalRepo, DocumentRecord, MediaType, RepoTier
from chittam.adapters import (
    chunk_markdown_or_text,
    make_doc_id,
    make_l0_summary,
    normalize_iso_timestamp,
    sha256_text,
)

_MAX_RECOVERED_TURN_CHARS = 16_000
_USER_REQUEST_RE = re.compile(r"<USER_REQUEST>\s*([\s\S]*?)\s*</USER_REQUEST>", re.IGNORECASE)
_TAG_STRIP_RE = re.compile(
    r"<(?:ADDITIONAL_METADATA|SYSTEM_MESSAGE|CRITICAL_CONSTRAINT|USER_SETTINGS_CHANGE|EPHEMERAL_MESSAGE|voice_instructions|talker_agent_context)[^>]*>[\s\S]*?</(?:ADDITIONAL_METADATA|SYSTEM_MESSAGE|CRITICAL_CONSTRAINT|USER_SETTINGS_CHANGE|EPHEMERAL_MESSAGE|voice_instructions|talker_agent_context)>",
    re.IGNORECASE,
)
_STRAY_VOICE_TAG_RE = re.compile(r"</?(?:voice_instructions|talker_agent_context)[^>]*>", re.IGNORECASE)
_MD_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_ACTIVE_WORKSPACE_LINE_RE = re.compile(r"^\s*-\s+(/[^ \n\r]+)\s*$", re.MULTILINE)
_ABS_PATH_IN_CMD_RE = re.compile(r"(/[^\s\"'`<>|;]+)")


def _clean_user_text(raw: str) -> str:
    m = _USER_REQUEST_RE.search(raw)
    text = m.group(1) if m else _TAG_STRIP_RE.sub("", raw)
    text = _TAG_STRIP_RE.sub("", text)
    text = _STRAY_VOICE_TAG_RE.sub("", text)
    return text.strip()


def _load_full_content_map(full_log_path: Path, needed_steps: set[Any]) -> dict[Any, str]:
    """Load full `content` strings from transcript_full.jsonl only for truncated step_index rows."""
    if not needed_steps or not full_log_path.exists() or not full_log_path.is_file():
        return {}
    out: dict[Any, str] = {}
    try:
        with full_log_path.open("r", encoding="utf-8", errors="replace") as fh:
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
                step = obj.get("step_index")
                if step in needed_steps:
                    content = obj.get("content") or obj.get("text") or obj.get("response")
                    if isinstance(content, str) and content.strip():
                        out[step] = content.strip()[:_MAX_RECOVERED_TURN_CHARS]
    except OSError:
        return {}
    return out


def _collect_repo_votes_from_obj(
    obj: dict,
    repo_index: RepoPathIndex,
    votes: Counter[str],
    atlas_by_id: dict[str, CanonicalRepo],
) -> None:
    """Record canonical repo touches from SYSTEM_MESSAGE workspaces and tool_calls arguments."""
    typ = str(obj.get("type") or "").upper()
    if typ == "SYSTEM_MESSAGE":
        content = obj.get("content")
        if isinstance(content, str) and "Active Workspaces:" in content:
            for m in _ACTIVE_WORKSPACE_LINE_RE.finditer(content):
                repo = repo_index.match(m.group(1), resolve_path=False)
                if repo is not None:
                    votes[repo.project_id] += 3
        return

    tool_calls = obj.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        raw_args = tc.get("args") or tc.get("arguments")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {"raw": raw_args}
        if not isinstance(raw_args, dict):
            continue
        for key in ("Path", "TargetFile", "Cwd", "CommandLine", "raw"):
            val = raw_args.get(key)
            if isinstance(val, str) and val.startswith('"'):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    pass
            if not isinstance(val, str) or "/" not in val:
                continue
            for m in _ABS_PATH_IN_CMD_RE.finditer(val):
                repo = repo_index.match(m.group(1), resolve_path=False)
                if repo is not None:
                    votes[repo.project_id] += 1
                    atlas_by_id.setdefault(repo.project_id, repo)


def _pick_voted_repo(votes: Counter[str], atlas: dict[str, CanonicalRepo]) -> CanonicalRepo | None:
    if not votes:
        return None
    non_archive = [
        (cnt, pid) for pid, cnt in votes.items() if pid in atlas and atlas[pid].tier != RepoTier.ARCHIVE
    ]
    if non_archive:
        non_archive.sort(reverse=True)
        return atlas[non_archive[0][1]]
    best_pid, _ = votes.most_common(1)[0]
    return atlas.get(best_pid)


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

    repo_index = RepoPathIndex(atlas)
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
                raw_objs: list[dict] = []
                needed_full_steps: set[Any] = set()
                repo_votes: Counter[str] = Counter()
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
                            raw_objs.append(obj)
                            _collect_repo_votes_from_obj(obj, repo_index, repo_votes, atlas)
                            tf = obj.get("truncated_fields")
                            if isinstance(tf, list) and "content" in tf and obj.get("step_index") is not None:
                                needed_full_steps.add(obj.get("step_index"))
                except OSError:
                    raw_objs = []

                full_content_by_step = (
                    _load_full_content_map(
                        conv_dir / ".system_generated" / "logs" / "transcript_full.jsonl",
                        needed_full_steps,
                    )
                    if needed_full_steps
                    else {}
                )

                turns: list[str] = []
                first_user_preview: str | None = None
                first_assistant_text: str | None = None
                user_turn_count = 0
                latest_ts: str | None = None
                detected_cwd: str | None = None
                for obj in raw_objs:
                    if obj.get("step_index") in full_content_by_step:
                        obj = {**obj, "content": full_content_by_step[obj["step_index"]]}
                    parsed = _extract_turn(obj)
                    if not parsed:
                        continue
                    role, text, ts, cwd = parsed
                    if cwd and not detected_cwd:
                        detected_cwd = cwd
                    if ts:
                        latest_ts = ts
                    if role == "User":
                        user_turn_count += 1
                        if first_user_preview is None:
                            first_user_preview = make_l0_summary(text, max_chars=80)
                    elif role == "Assistant" and first_assistant_text is None:
                        first_assistant_text = text
                    step_idx = obj.get("step_index")
                    step_tag = f" [step {step_idx}]" if step_idx is not None else ""
                    ts_tag = f" ({ts[:19]})" if ts else ""
                    turns.append(f"## {role}{step_tag}{ts_tag}\n{text}")

                if turns:
                    full_text = redact_secrets("\n\n".join(turns))
                    is_subagent = user_turn_count == 0
                    if is_subagent and len(full_text) < 200:
                        turns = []

                if turns:
                    early_text = "\n\n".join(turns[:3])
                    if detected_cwd:
                        conv_repo = match_repo_for_path(detected_cwd, atlas)
                    if conv_repo is None:
                        conv_repo = _pick_voted_repo(repo_votes, atlas)
                    if conv_repo is None:
                        conv_repo = match_repo_from_text(full_text, atlas, early_text=early_text)
                    proj_id = conv_repo.project_id if conv_repo else None
                    tier = conv_repo.tier if conv_repo else RepoTier.REFERENCE
                    if is_subagent and tier == RepoTier.ACTIVE:
                        tier = RepoTier.REFERENCE
                    try:
                        mtime = tlog_path.stat().st_mtime
                    except OSError:
                        mtime = None
                    updated_at = normalize_iso_timestamp(latest_ts, fallback_mtime=mtime)
                    if first_user_preview:
                        title = first_user_preview
                    elif first_assistant_text:
                        h_match = _MD_TITLE_RE.search(first_assistant_text)
                        title = (
                            h_match.group(1).strip()[:80]
                            if h_match
                            else make_l0_summary(first_assistant_text, max_chars=80)
                        )
                    else:
                        title = f"Antigravity session {conv_id[:8]}"
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
                        meta: dict[str, Any] = {
                            "conversation_id": conv_id,
                            "turn_count": len(turns),
                        }
                        if is_subagent:
                            meta["is_subagent"] = True
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
                                metadata=meta,
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
