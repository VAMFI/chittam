"""Claude Code session transcript and history adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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

_CMD_NAME_RE = re.compile(r"<command-name>\s*(.*?)\s*</command-name>", re.DOTALL)
_CMD_ARGS_RE = re.compile(r"<command-args>\s*(.*?)\s*</command-args>", re.DOTALL)
_LOCAL_CAVEAT_RE = re.compile(r"<local-command-caveat>[\s\S]*?</local-command-caveat>", re.IGNORECASE)


def _extract_claude_message_text(obj: dict[str, Any]) -> str:
    msg = obj.get("message")
    raw_content: Any
    if isinstance(msg, dict):
        raw_content = msg.get("content")
    else:
        raw_content = obj.get("content")

    text = ""
    if isinstance(raw_content, str):
        text = raw_content
    elif isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, dict) and item.get("type") == "text":
                val = item.get("text")
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        text = "\n".join(parts)

    if not text:
        return ""

    text = _LOCAL_CAVEAT_RE.sub("", text).strip()
    cmd_name = _CMD_NAME_RE.search(text)
    cmd_args = _CMD_ARGS_RE.search(text)
    if cmd_name or cmd_args:
        cmd = cmd_name.group(1).strip() if cmd_name else ""
        args = cmd_args.group(1).strip() if cmd_args else ""
        combined = f"{cmd} {args}".strip()
        if combined:
            return combined
    return text.strip()


def _load_project_sessions(cfg: AppConfig, atlas: dict[str, CanonicalRepo]) -> list[DocumentRecord]:
    projects_dir = Path(cfg.harnesses.claude_projects_dir).expanduser()
    if not projects_dir.exists() or not projects_dir.is_dir():
        return []
    excl_root, _ = is_path_excluded(projects_dir, cfg.privacy)
    if excl_root:
        return []

    try:
        jsonl_files = sorted(set(projects_dir.glob("*/*.jsonl")) | set(projects_dir.glob("*.jsonl")))
    except OSError:
        return []

    docs: list[DocumentRecord] = []
    for jsonl_path in jsonl_files:
        if not jsonl_path.is_file():
            continue
        excl, _ = is_path_excluded(jsonl_path, cfg.privacy)
        if excl:
            continue

        turns: list[str] = []
        cwd: str | None = None
        session_id: str | None = None
        git_branch: str | None = None
        latest_ts: str | None = None
        first_user_preview: str | None = None

        try:
            with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
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

                    if not cwd and isinstance(obj.get("cwd"), str) and obj["cwd"].strip():
                        cwd = obj["cwd"].strip()
                    if not session_id and isinstance(obj.get("sessionId"), str) and obj["sessionId"].strip():
                        session_id = obj["sessionId"].strip()
                    if not git_branch and isinstance(obj.get("gitBranch"), str) and obj["gitBranch"].strip():
                        git_branch = obj["gitBranch"].strip()
                    if isinstance(obj.get("timestamp"), str) and obj["timestamp"].strip():
                        latest_ts = obj["timestamp"].strip()

                    ev_type = str(obj.get("type") or "").lower()
                    if ev_type not in ("user", "assistant"):
                        continue

                    text = _extract_claude_message_text(obj)
                    if not text:
                        continue

                    role = "User" if ev_type == "user" else "Assistant"
                    if ev_type == "user" and first_user_preview is None:
                        first_user_preview = make_l0_summary(text, max_chars=80)
                    ts_str = str(obj.get("timestamp") or "")
                    ts_tag = f" ({ts_str[:19]})" if ts_str else ""
                    turns.append(f"## {role}{ts_tag}\n{text}")
        except OSError:
            continue

        if not turns:
            continue

        full_text = redact_secrets("\n\n".join(turns))
        repo = match_repo_for_path(cwd, atlas) if cwd else None
        if repo is None:
            repo = match_repo_from_text(full_text, atlas)
        proj_id = repo.project_id if repo else None
        tier = repo.tier if repo else RepoTier.REFERENCE
        try:
            mtime = jsonl_path.stat().st_mtime
        except OSError:
            mtime = None
        updated_at = normalize_iso_timestamp(latest_ts, fallback_mtime=mtime)
        sid = session_id or jsonl_path.stem
        branch_suffix = f" [{git_branch}]" if git_branch and git_branch != "HEAD" else ""
        title = (first_user_preview or f"Claude session {sid[:8]}") + branch_suffix
        doc_id = make_doc_id("claude_sess", str(jsonl_path))
        chunks = chunk_markdown_or_text(
            doc_id=doc_id,
            text=full_text,
            project_id=proj_id,
            canonical_tier=tier,
            path_or_uri=str(jsonl_path),
            updated_at=updated_at,
        )
        if chunks:
            docs.append(
                DocumentRecord(
                    doc_id=doc_id,
                    uri=str(jsonl_path),
                    source_harness="claude_code",
                    media_type=MediaType.TRANSCRIPT,
                    project_id=proj_id,
                    canonical_tier=tier,
                    title=title,
                    l0_summary=make_l0_summary(first_user_preview or full_text),
                    sha256=sha256_text(full_text),
                    updated_at=updated_at,
                    metadata={
                        "session_id": sid,
                        "cwd": cwd,
                        "git_branch": git_branch,
                        "turn_count": len(turns),
                    },
                    chunks=chunks,
                )
            )
    return docs


def _load_history_log(cfg: AppConfig, atlas: dict[str, CanonicalRepo]) -> list[DocumentRecord]:
    history_path = Path(cfg.harnesses.claude_history_path).expanduser()
    if not history_path.exists() or not history_path.is_file():
        return []
    excl, _ = is_path_excluded(history_path, cfg.privacy)
    if excl:
        return []

    grouped: dict[str, dict[str, Any]] = {}
    try:
        with history_path.open("r", encoding="utf-8", errors="replace") as fh:
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
                display = obj.get("display")
                if not isinstance(display, str) or not display.strip():
                    continue
                sid = str(obj.get("sessionId") or obj.get("project") or "default")
                entry = grouped.setdefault(
                    sid,
                    {
                        "project": obj.get("project"),
                        "max_ts": None,
                        "prompts": [],
                    },
                )
                if not entry["project"] and obj.get("project"):
                    entry["project"] = obj.get("project")
                ts = obj.get("timestamp")
                if isinstance(ts, (int, float)):
                    if entry["max_ts"] is None or ts > entry["max_ts"]:
                        entry["max_ts"] = ts
                ts_label = f"[{normalize_iso_timestamp(ts)[:19]}] " if ts else ""
                entry["prompts"].append(f"- {ts_label}{display.strip()}")
    except OSError:
        return []

    docs: list[DocumentRecord] = []
    for sid, info in grouped.items():
        prompts: list[str] = info["prompts"]
        if not prompts:
            continue
        body = redact_secrets("\n".join(prompts))
        project_path = info.get("project")
        repo = match_repo_for_path(project_path, atlas) if project_path else None
        if repo is None:
            repo = match_repo_from_text(body, atlas)
        proj_id = repo.project_id if repo else None
        tier = repo.tier if repo else RepoTier.REFERENCE
        updated_at = normalize_iso_timestamp(info.get("max_ts"))
        uri = f"{history_path}#session={sid}"
        doc_id = make_doc_id("claude_hist", uri)
        first_prompt = prompts[0].split("] ", 1)[-1].lstrip("- ")
        title = f"Claude history: {make_l0_summary(first_prompt, 70)}"
        chunks = chunk_markdown_or_text(
            doc_id=doc_id,
            text=body,
            project_id=proj_id,
            canonical_tier=tier,
            path_or_uri=uri,
            updated_at=updated_at,
        )
        if chunks:
            docs.append(
                DocumentRecord(
                    doc_id=doc_id,
                    uri=uri,
                    source_harness="claude_code",
                    media_type=MediaType.TRANSCRIPT,
                    project_id=proj_id,
                    canonical_tier=tier,
                    title=title,
                    l0_summary=make_l0_summary(body),
                    sha256=sha256_text(body),
                    updated_at=updated_at,
                    metadata={
                        "session_id": sid,
                        "project": project_path,
                        "prompt_count": len(prompts),
                    },
                    chunks=chunks,
                )
            )
    return docs


def load_claude_code_docs(cfg: AppConfig, atlas: dict[str, CanonicalRepo]) -> list[DocumentRecord]:
    """Parse Claude Code project session transcripts and prompt history."""
    docs: list[DocumentRecord] = []
    docs.extend(_load_project_sessions(cfg, atlas))
    docs.extend(_load_history_log(cfg, atlas))
    return docs
