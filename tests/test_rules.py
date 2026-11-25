"""Tests for CHITTAM.md rule installer, managed routing blocks, and MCP server instructions."""

from __future__ import annotations

import stat
from pathlib import Path

from chittam import mcp_server
from chittam.models import AppConfig, HarnessesConfig
from chittam.rules import (
    CHITTAM_MD_CONTENT,
    MCP_SERVER_INSTRUCTIONS,
    RULE_END_MARKER,
    RULE_START_MARKER,
    install_rules,
    remove_managed_block,
    render_managed_block,
    uninstall_rules,
    upsert_managed_block,
)


def _make_cfg(tmp_path: Path) -> tuple[AppConfig, Path, Path, Path]:
    ag_rules = tmp_path / "gemini" / "GEMINI.md"
    claude_rules = tmp_path / "claude" / "CLAUDE.md"
    codex_rules = tmp_path / "codex" / "AGENTS.md"
    ag_rules.parent.mkdir(parents=True, exist_ok=True)
    ag_rules.write_text("# My Existing Guardrails\n- Keep diffs small.\n", encoding="utf-8")

    cfg = AppConfig(
        data_dir=str(tmp_path / "data"),
        config_dir=str(tmp_path / "config"),
        harnesses=HarnessesConfig(
            antigravity_rules_path=str(ag_rules),
            claude_rules_path=str(claude_rules),
            codex_rules_path=str(codex_rules),
        ),
    )
    return cfg, ag_rules, claude_rules, codex_rules


def test_managed_block_upsert_idempotent_and_remove() -> None:
    block_v1 = render_managed_block("~/.config/chittam/CHITTAM.md")
    initial = "# User Rules\n- Always run tests.\n"

    first = upsert_managed_block(initial, block_v1)
    assert "# User Rules\n- Always run tests." in first
    assert RULE_START_MARKER in first
    assert RULE_END_MARKER in first

    # Idempotent re-apply
    second = upsert_managed_block(first, block_v1)
    assert second == first

    # In-place update replaces previous block cleanly without duplicating markers
    block_v2 = render_managed_block("/custom/CHITTAM.md")
    updated = upsert_managed_block(second, block_v2)
    assert updated.count(RULE_START_MARKER) == 1
    assert "/custom/CHITTAM.md" in updated
    assert "~/.config/chittam/CHITTAM.md" not in updated

    # Clean removal restores user rules untouched
    cleaned, removed = remove_managed_block(updated)
    assert removed is True
    assert cleaned == "# User Rules\n- Always run tests.\n"
    assert RULE_START_MARKER not in cleaned

    # No-op when absent
    cleaned2, removed2 = remove_managed_block(cleaned)
    assert removed2 is False
    assert cleaned2 == cleaned


def test_install_and_uninstall_rules_with_backups_and_extra_file(tmp_path: Path) -> None:
    cfg, ag_rules, claude_rules, codex_rules = _make_cfg(tmp_path)
    repo_agents = tmp_path / "my-repo" / "AGENTS.md"

    res1 = install_rules(cfg, extra_files=[repo_agents])
    assert Path(res1["chittam_md"]).exists()
    assert res1["targets"]["antigravity"]["status"] == "updated"
    assert res1["targets"]["claude_code"]["status"] == "created"
    assert res1["targets"]["codex"]["status"] == "created"
    assert res1["targets"]["extra_1"]["status"] == "created"

    # Existing user file was backed up with 0600 permissions and content preserved
    bak = Path(res1["targets"]["antigravity"]["backup"])
    assert bak.exists()
    assert stat.S_IMODE(bak.stat().st_mode) == 0o600
    assert "Keep diffs small." in bak.read_text(encoding="utf-8")

    ag_text = ag_rules.read_text(encoding="utf-8")
    assert "# My Existing Guardrails" in ag_text
    assert "- Keep diffs small." in ag_text
    assert "memory_search" in ag_text
    assert claude_rules.exists()
    assert codex_rules.exists()
    assert repo_agents.exists()

    # Second run is a no-op (`unchanged`)
    res2 = install_rules(cfg, extra_files=[repo_agents])
    assert res2["targets"]["antigravity"]["status"] == "unchanged"
    assert res2["targets"]["claude_code"]["status"] == "unchanged"

    # Uninstall removes managed block while preserving original Antigravity text
    unres = uninstall_rules(cfg, extra_files=[repo_agents], remove_chittam_md=True)
    assert unres["chittam_md_removed"] is True
    assert not cfg.chittam_md_path.exists()
    assert unres["targets"]["antigravity"]["status"] == "removed"
    assert ag_rules.read_text(encoding="utf-8") == "# My Existing Guardrails\n- Keep diffs small.\n"
    assert codex_rules.read_text(encoding="utf-8") == ""


def test_mcp_server_instructions_and_no_personal_data() -> None:
    instructions = mcp_server.mcp.instructions
    assert instructions == MCP_SERVER_INSTRUCTIONS
    for token in (
        "memory_search",
        "memory_get_chunk",
        "project_atlas",
        "recall_threads",
        "record_decision",
        "index_status",
        "ripgrep",
    ):
        assert token in instructions

    block = render_managed_block()
    assert len(block.split()) < 150
    for text in (instructions, CHITTAM_MD_CONTENT, block):
        low = text.lower()
        assert "/users/" not in low
        assert "/home/" not in low
        assert "c:\\users\\" not in low
