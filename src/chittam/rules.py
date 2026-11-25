"""Managed CHITTAM.md rule installer and MCP server routing instructions."""

from __future__ import annotations

import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chittam.harnesses import _atomic_write_text, _create_backup
from chittam.models import AppConfig

RULE_START_MARKER = "<!-- chittam:start -->"
RULE_END_MARKER = "<!-- chittam:end -->"

MCP_SERVER_INSTRUCTIONS = (
    "Chittam provides pull-only local memory across repositories, docs, PDFs, captioned "
    "screenshots, agent threads, and architectural decisions. Never preload memory dumps; "
    "keep using native repo search (ripgrep, grep, direct reads) and CodeGraph for exact "
    "symbols inside the currently open repository. Call Chittam on demand when you need:\n"
    "1. Cross-repo or off-repo context (notes, PRDs, PDFs, screenshots, conceptual "
    "queries): call `memory_search`, then `memory_get_chunk` only on the 1-3 chunks needed.\n"
    "2. Canonical repository disambiguation (active vs archived clones): call `project_atlas`.\n"
    "3. Prior Antigravity, Codex, or Claude Code conversations: call `recall_threads`.\n"
    "4. Durable architectural or product decisions shared across harnesses: call `record_decision`.\n"
    "5. Index readiness or freshness checks: call `index_status`."
)

CHITTAM_MD_CONTENT = """# Chittam Local Memory Routing (`CHITTAM.md`)

Chittam is a pull-only local memory server wired over MCP. Keep prompts static to preserve prompt caching and pull tools on demand:

1. **Current repo exact symbols & edits**: Use `ripgrep` / `grep`, direct file reads, and `CodeGraph` first for live line-accurate code inside the active workspace.
2. **Off-repo & conceptual search (`memory_search`)**: Use before asking the user for file paths or hunting blindly across folders for PRDs, PDFs, notes, captioned screenshots/diagrams, or conceptual questions. Follow up with `memory_get_chunk` only for the 1–3 chunks you actually need.
3. **Canonical repo lookup (`project_atlas`)**: Check active vs. reference vs. archive copies before editing or referencing another repository.
4. **Cross-agent thread recall (`recall_threads`)**: Search prior Antigravity, Codex, and Claude Code sessions whenever previous work or context may live in another agent harness.
5. **Persist durable choices (`record_decision`)**: Save architectural or product decisions so every connected harness shares them.
"""

_BLOCK_PATTERN = re.compile(
    rf"{re.escape(RULE_START_MARKER)}.*?{re.escape(RULE_END_MARKER)}\n?",
    flags=re.DOTALL,
)


def render_managed_block(chittam_md_path: Path | str = "~/.config/chittam/CHITTAM.md") -> str:
    """Render the compact managed Markdown block injected into agent instruction files."""
    ref = str(chittam_md_path)
    return (
        f"{RULE_START_MARKER}\n"
        "## Chittam Local Memory (Pull-Only Routing)\n"
        "Keep prompts static for prompt caching; use `ripgrep`/direct reads (and `CodeGraph`) for "
        "exact symbols inside the active repo, and call Chittam MCP tools on demand:\n"
        "- `memory_search` -> cross-repo context, PDFs, notes, captioned screenshots, or conceptual lookups "
        "(fetch full text via `memory_get_chunk` only for the 1-3 hits you need).\n"
        "- `project_atlas` -> identify the canonical `active` repo vs. archived clones.\n"
        "- `recall_threads` -> search prior Antigravity, Codex, and Claude Code sessions.\n"
        "- `record_decision` -> persist durable decisions across all harnesses.\n"
        f"Full reference: `{ref}`\n"
        f"{RULE_END_MARKER}\n"
    )


def upsert_managed_block(existing_text: str, block: str) -> str:
    """Insert or replace the managed Chittam block in `existing_text` idempotently."""
    normalized_block = block if block.endswith("\n") else block + "\n"
    if _BLOCK_PATTERN.search(existing_text):
        updated = _BLOCK_PATTERN.sub(lambda _: normalized_block, existing_text, count=1)
        if updated and not updated.endswith("\n"):
            updated += "\n"
        return updated

    if not existing_text.strip():
        return normalized_block

    base = existing_text.rstrip() + "\n\n"
    return base + normalized_block


def remove_managed_block(existing_text: str) -> tuple[str, bool]:
    """Remove the managed Chittam block from `existing_text`, preserving user rules."""
    if not _BLOCK_PATTERN.search(existing_text):
        return existing_text, False

    stripped = _BLOCK_PATTERN.sub("", existing_text)
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in stripped.splitlines():
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank

    result = "\n".join(cleaned_lines).strip()
    if result:
        result += "\n"
    return result, True


def _resolve_rule_targets(cfg: AppConfig, extra_files: list[Path] | None) -> list[tuple[str, Path]]:
    targets: list[tuple[str, Path]] = [
        ("antigravity", Path(cfg.harnesses.antigravity_rules_path).expanduser()),
        ("claude_code", Path(cfg.harnesses.claude_rules_path).expanduser()),
        ("codex", Path(cfg.harnesses.codex_rules_path).expanduser()),
    ]
    seen = {path.resolve(strict=False) for _, path in targets}
    for idx, extra in enumerate(extra_files or [], start=1):
        expanded = extra.expanduser()
        resolved = expanded.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        targets.append((f"extra_{idx}", expanded))
    return targets


def install_rules(
    cfg: AppConfig,
    *,
    extra_files: list[Path] | None = None,
    create_missing: bool = True,
) -> dict[str, Any]:
    """Write `CHITTAM.md` and upsert the managed routing block into harness rule files."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    chittam_md = cfg.chittam_md_path
    _atomic_write_text(chittam_md, CHITTAM_MD_CONTENT, mode=0o644)

    block = render_managed_block(chittam_md)
    targets_summary: dict[str, dict[str, Any]] = {}

    for label, target in _resolve_rule_targets(cfg, extra_files):
        if target.exists():
            original_mode = stat.S_IMODE(target.stat().st_mode)
            before = target.read_text(encoding="utf-8")
            after = upsert_managed_block(before, block)
            if after == before:
                targets_summary[label] = {
                    "status": "unchanged",
                    "path": str(target),
                }
                continue
            backup = _create_backup(target, cfg.backup_dir, ts)
            _atomic_write_text(target, after, mode=original_mode)
            targets_summary[label] = {
                "status": "updated",
                "path": str(target),
                "backup": str(backup),
            }
        else:
            if not create_missing:
                targets_summary[label] = {
                    "status": "skipped",
                    "reason": "file_not_found",
                    "path": str(target),
                }
                continue
            _atomic_write_text(target, block, mode=0o644)
            targets_summary[label] = {
                "status": "created",
                "path": str(target),
            }

    return {
        "chittam_md": str(chittam_md),
        "targets": targets_summary,
    }


def uninstall_rules(
    cfg: AppConfig,
    *,
    extra_files: list[Path] | None = None,
    remove_chittam_md: bool = False,
) -> dict[str, Any]:
    """Remove the managed Chittam routing block from rule files while preserving user text."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    targets_summary: dict[str, dict[str, Any]] = {}

    for label, target in _resolve_rule_targets(cfg, extra_files):
        if not target.exists():
            targets_summary[label] = {
                "status": "skipped",
                "reason": "file_not_found",
                "path": str(target),
            }
            continue

        original_mode = stat.S_IMODE(target.stat().st_mode)
        before = target.read_text(encoding="utf-8")
        after, removed = remove_managed_block(before)
        if not removed:
            targets_summary[label] = {
                "status": "unchanged",
                "path": str(target),
            }
            continue

        backup = _create_backup(target, cfg.backup_dir, f"rules_uninstall_{ts}")
        _atomic_write_text(target, after, mode=original_mode)
        targets_summary[label] = {
            "status": "removed",
            "path": str(target),
            "backup": str(backup),
        }

    removed_chittam_md = False
    if remove_chittam_md and cfg.chittam_md_path.exists():
        os.unlink(cfg.chittam_md_path)
        removed_chittam_md = True

    return {
        "chittam_md_removed": removed_chittam_md,
        "targets": targets_summary,
    }
