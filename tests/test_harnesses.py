"""Tests for harnesses registration, ROUTING_RULE.md, launchd plist, and benchmark."""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from chittam import harnesses
from chittam.harnesses import install_harnesses, uninstall_harnesses
from chittam.launchd import PLIST_LABEL, install_launchd, launchd_status, uninstall_launchd
from chittam.models import (
    AppConfig,
    ChunkRecord,
    DocumentRecord,
    HarnessesConfig,
    MediaType,
    RepoTier,
)
from chittam.store import ChittamStore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chittam.benchmark import _is_leaked_result, load_cases_from_yaml, run_synthetic_benchmark


def _make_cfg(tmp_path: Path) -> tuple[AppConfig, Path, Path, Path]:
    ag_path = tmp_path / "gemini" / "config" / "mcp_config.json"
    codex_path = tmp_path / "codex" / "config.toml"
    claude_path = tmp_path / ".claude.json"
    ag_path.parent.mkdir(parents=True, exist_ok=True)
    codex_path.parent.mkdir(parents=True, exist_ok=True)

    ag_payload = {
        "mcpServers": {
            "deepwiki": {"serverUrl": "https://mcp.deepwiki.com/sse"},
            "xcode": {
                "command": "/Applications/Xcode.app/Contents/Developer/usr/bin/mcpbridge",
                "args": [],
                "timeoutSeconds": 600,
            },
        }
    }
    ag_path.write_text(json.dumps(ag_payload, indent=2) + "\n", encoding="utf-8")

    codex_toml = (
        "# User-authored top comment\n"
        'model = "gpt-6.2-astra"\n'
        'service_tier = "default"\n\n'
        '[plugins."codex-app-tools@openai-bundled"]\n'
        "enabled = true\n\n"
        "[mcp_servers.node_repl]\n"
        "args = []\n"
        'command = "/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node_repl"\n\n'
        "[mcp_servers.node_repl.env]\n"
        'CODEX_HOME = "/Users/example/.codex"\n\n'
        "[desktop]\n"
        'conversationDetailMode = "STEPS_COMMANDS"\n\n'
        '[projects."/Users/example/projects/recipes"]\n'
        'trust_level = "trusted"\n'
    )
    codex_path.write_text(codex_toml, encoding="utf-8")

    claude_payload = {
        "numStartups": 42,
        "installMethod": "native",
        "projects": {"/Users/example/VAMFI/chittam": {"allowedTools": ["Bash"]}},
        "oauthAccount": {"emailAddress": "dev@example.com"},
        "mcpServers": {
            "deepwiki": {"type": "http", "url": "https://mcp.deepwiki.com/mcp"},
            "pencil": {
                "type": "stdio",
                "command": "/Applications/Pencil.app/Contents/MacOS/mcp",
                "args": ["--app", "desktop"],
            },
        },
    }
    claude_path.write_text(json.dumps(claude_payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(claude_path, 0o600)

    cfg = AppConfig(
        data_dir=str(tmp_path / "data"),
        config_dir=str(tmp_path / "config"),
        harnesses=HarnessesConfig(
            antigravity_mcp_path=str(ag_path),
            codex_config_path=str(codex_path),
            claude_config_path=str(claude_path),
        ),
    )
    return cfg, ag_path, codex_path, claude_path


def test_harnesses_backup_install_idempotent_uninstall(tmp_path: Path) -> None:
    cfg, ag_path, codex_path, claude_path = _make_cfg(tmp_path)
    fake_chittam = str(tmp_path / "bin" / "chittam-mcp")
    fake_codegraph = str(tmp_path / "bin" / "codegraph")

    orig_ag = ag_path.read_text(encoding="utf-8")
    orig_codex = codex_path.read_text(encoding="utf-8")
    orig_claude = claude_path.read_text(encoding="utf-8")

    with (
        patch.object(harnesses, "resolve_chittam_mcp_command", return_value=fake_chittam),
        patch.object(harnesses, "resolve_codegraph_command", return_value=fake_codegraph),
    ):
        res1 = install_harnesses(cfg)
        assert res1["harnesses"]["antigravity"]["status"] == "installed"
        assert res1["harnesses"]["codex"]["status"] == "installed"
        assert res1["harnesses"]["claude_code"]["status"] == "installed"

        # Verify backups and manifest were created
        manifest_file = cfg.backup_dir / "manifest.json"
        assert manifest_file.exists()
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert set(manifest["harnesses"].keys()) == {"antigravity", "codex", "claude_code"}
        for h_name in ("antigravity", "codex", "claude_code"):
            bak = Path(manifest["harnesses"][h_name]["initial_backup"])
            assert bak.exists()
            assert stat.S_IMODE(bak.stat().st_mode) == 0o600

        # Verify Antigravity JSON (no `type: stdio`)
        ag_after = json.loads(ag_path.read_text(encoding="utf-8"))
        assert ag_after["mcpServers"]["deepwiki"]["serverUrl"] == "https://mcp.deepwiki.com/sse"
        assert ag_after["mcpServers"]["chittam"] == {"command": fake_chittam, "args": []}
        assert ag_after["mcpServers"]["codegraph"] == {
            "command": fake_codegraph,
            "args": ["serve", "--mcp"],
        }

        # Verify Codex TOML preserved comments & unrelated sections
        codex_text = codex_path.read_text(encoding="utf-8")
        assert "# User-authored top comment" in codex_text
        codex_parsed = tomllib.loads(codex_text)
        assert codex_parsed["model"] == "gpt-6.2-astra"
        assert codex_parsed["plugins"]["codex-app-tools@openai-bundled"]["enabled"] is True
        assert codex_parsed["mcp_servers"]["node_repl"]["env"]["CODEX_HOME"] == "/Users/example/.codex"
        assert codex_parsed["projects"]["/Users/example/projects/recipes"]["trust_level"] == "trusted"
        assert codex_parsed["mcp_servers"]["chittam"] == {"command": fake_chittam, "args": []}
        assert codex_parsed["mcp_servers"]["codegraph"] == {
            "command": fake_codegraph,
            "args": ["serve", "--mcp"],
        }

        # Verify Claude Code JSON preserved all top-level keys, projects, and 0600 mode
        assert stat.S_IMODE(claude_path.stat().st_mode) == 0o600
        claude_after = json.loads(claude_path.read_text(encoding="utf-8"))
        assert claude_after["numStartups"] == 42
        assert claude_after["projects"]["/Users/example/VAMFI/chittam"]["allowedTools"] == ["Bash"]
        assert claude_after["oauthAccount"]["emailAddress"] == "dev@example.com"
        assert claude_after["mcpServers"]["chittam"] == {
            "type": "stdio",
            "command": fake_chittam,
            "args": [],
        }
        assert claude_after["mcpServers"]["codegraph"] == {
            "type": "stdio",
            "command": fake_codegraph,
            "args": ["serve", "--mcp"],
        }

        # Idempotent re-install
        codex_before_reinstall = codex_path.read_text(encoding="utf-8")
        ag_before_reinstall = ag_path.read_text(encoding="utf-8")
        claude_before_reinstall = claude_path.read_text(encoding="utf-8")

        install_harnesses(cfg)
        assert codex_path.read_text(encoding="utf-8") == codex_before_reinstall
        assert ag_path.read_text(encoding="utf-8") == ag_before_reinstall
        assert claude_path.read_text(encoding="utf-8") == claude_before_reinstall

        # Clean uninstall (removes chittam & codegraph, preserves everything else)
        unres = uninstall_harnesses(cfg)
        assert set(unres["harnesses"]["antigravity"]["removed"]) == {"chittam", "codegraph"}
        assert set(unres["harnesses"]["codex"]["removed"]) == {"chittam", "codegraph"}
        assert set(unres["harnesses"]["claude_code"]["removed"]) == {"chittam", "codegraph"}

        ag_clean = json.loads(ag_path.read_text(encoding="utf-8"))
        assert set(ag_clean["mcpServers"].keys()) == {"deepwiki", "xcode"}

        codex_clean_text = codex_path.read_text(encoding="utf-8")
        assert "# User-authored top comment" in codex_clean_text
        codex_clean = tomllib.loads(codex_clean_text)
        assert set(codex_clean["mcp_servers"].keys()) == {"node_repl"}

        claude_clean = json.loads(claude_path.read_text(encoding="utf-8"))
        assert set(claude_clean["mcpServers"].keys()) == {"deepwiki", "pencil"}
        assert claude_clean["projects"] == {"/Users/example/VAMFI/chittam": {"allowedTools": ["Bash"]}}

        # Re-install and test full backup restore mode
        install_harnesses(cfg)
        restore_res = uninstall_harnesses(cfg, restore_backup=True)
        assert restore_res["harnesses"]["antigravity"]["status"] == "restored"
        assert restore_res["harnesses"]["codex"]["status"] == "restored"
        assert restore_res["harnesses"]["claude_code"]["status"] == "restored"
        assert ag_path.read_text(encoding="utf-8") == orig_ag
        assert codex_path.read_text(encoding="utf-8") == orig_codex
        assert claude_path.read_text(encoding="utf-8") == orig_claude


def test_harnesses_skips_codegraph_when_absent_and_rolls_back_on_error(tmp_path: Path) -> None:
    cfg, ag_path, codex_path, claude_path = _make_cfg(tmp_path)
    orig_ag = ag_path.read_text(encoding="utf-8")

    with (
        patch.object(harnesses, "resolve_chittam_mcp_command", return_value="/tmp/chittam-mcp"),
        patch.object(harnesses, "resolve_codegraph_command", return_value=None),
    ):
        res = install_harnesses(cfg)
        assert list(res["servers"].keys()) == ["chittam"]
        assert "codegraph" not in json.loads(ag_path.read_text(encoding="utf-8"))["mcpServers"]
        assert "codegraph" not in tomllib.loads(codex_path.read_text(encoding="utf-8"))["mcp_servers"]
        assert "codegraph" not in json.loads(claude_path.read_text(encoding="utf-8"))["mcpServers"]

    # Rollback test if post-write validation raises
    pre_rollback_ag = ag_path.read_text(encoding="utf-8")
    with (
        patch.object(harnesses, "resolve_chittam_mcp_command", return_value="/tmp/new-chittam-mcp"),
        patch.object(harnesses, "resolve_codegraph_command", return_value=None),
        patch.object(harnesses, "_validate_json_servers", side_effect=ValueError("boom")),
    ):
        with pytest.raises(ValueError, match="boom"):
            install_harnesses(cfg)
        assert ag_path.read_text(encoding="utf-8") == pre_rollback_ag
        assert json.loads(ag_path.read_text(encoding="utf-8"))["mcpServers"]["chittam"]["command"] == "/tmp/chittam-mcp"


def test_harnesses_toml_multiline_and_codegraph_resolution(tmp_path: Path) -> None:
    toml_input = (
        '# Keep header comment\n'
        'prompt = """\n'
        '[mcp_servers.chittam]\n'
        'command = "inside-multiline-string"\n'
        '"""\n'
        '[desktop]\n'
        'theme = "dark"\n'
    )
    updated = harnesses._upsert_toml_mcp_server(toml_input, "chittam", "/tmp/chittam-mcp", [])
    parsed = tomllib.loads(updated)
    assert "[mcp_servers.chittam]" in parsed["prompt"]
    assert parsed["mcp_servers"]["chittam"] == {"command": "/tmp/chittam-mcp", "args": []}
    removed = harnesses._remove_toml_mcp_server(updated, "chittam")
    parsed_removed = tomllib.loads(removed)
    assert "mcp_servers" not in parsed_removed
    assert "[mcp_servers.chittam]" in parsed_removed["prompt"]

    # Check real codegraph resolution on this host without mutating real harness files
    resolved = harnesses.resolve_codegraph_command()
    if resolved is not None:
        assert Path(resolved).exists()
        assert Path(resolved).name == "codegraph"


def test_routing_rule_word_count_under_150() -> None:
    rule_path = REPO_ROOT / "docs" / "ROUTING_RULE.md"
    assert rule_path.exists()
    text = rule_path.read_text(encoding="utf-8")
    words = text.split()
    assert 30 < len(words) < 150, f"Expected < 150 words, got {len(words)}"
    for expected in (
        "ripgrep",
        "CodeGraph",
        "Chittam",
        "memory_search",
        "project_atlas",
        "recall_threads",
        "record_decision",
        "prompt caching",
    ):
        assert expected in text


def test_launchd_plist_generation_and_validation(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=str(tmp_path / "data"), config_dir=str(tmp_path / "config"))
    plist_target = tmp_path / "LaunchAgents" / f"{PLIST_LABEL}.plist"

    written = install_launchd(cfg, plist_path=plist_target, load=False)
    assert written == plist_target
    assert plist_target.exists()

    parsed = plistlib.loads(plist_target.read_bytes())
    assert parsed["Label"] == PLIST_LABEL
    assert parsed["StartInterval"] == 1800
    assert parsed["ProgramArguments"][1:] == ["sync", "--quiet"]
    assert parsed["StandardOutPath"] == str(tmp_path / "data" / "sync.log")

    plutil = shutil.which("plutil") or "/usr/bin/plutil"
    if Path(plutil).exists():
        lint = subprocess.run([plutil, "-lint", str(plist_target)], capture_output=True, text=True)
        assert lint.returncode == 0, lint.stderr

    status = launchd_status(plist_path=plist_target)
    assert status["installed"] is True
    assert status["start_interval"] == 1800

    assert uninstall_launchd(plist_path=plist_target, unload=False) is True
    assert not plist_target.exists()
    assert uninstall_launchd(plist_path=plist_target, unload=False) is False


def test_synthetic_benchmark_15_queries_and_zero_privacy_leakage() -> None:
    report = run_synthetic_benchmark()
    assert report["total_queries"] == 15
    assert report["positive_queries"] == 13
    assert report["privacy_queries"] == 2
    assert report["privacy_leakage"] == 0
    assert report["Recall@5"] == 1.0
    assert report["MRR"] >= 0.9
    # Three fixtures (tax return, .env, auth.json) must be stopped by the privacy gate.
    assert report["corpus"]["blocked_by_privacy_gate"] == 3
    assert report["p95_ms"] >= report["p50_ms"] >= 0.0


def test_custom_cases_yaml_round_trip(tmp_path: Path) -> None:
    cases_file = tmp_path / "cases.yaml"
    cases_file.write_text(
        "cases:\n"
        "  - {id: C1, query: offline recipe app, expect_any: [acme-recipes]}\n"
        "  - {id: C2, query: geo-atlas docker-compose, expect_any: [geo-atlas], require_non_archive: true}\n"
        "  - {id: C3, mode: privacy, query: tax return}\n",
        encoding="utf-8",
    )
    cases = load_cases_from_yaml(cases_file)
    assert [c.query_id for c in cases] == ["C1", "C2", "C3"]
    assert cases[2].mode == "privacy" and cases[2].matcher is None


def test_benchmark_privacy_leak_detector() -> None:
    from chittam.models import SearchResult

    cfg = AppConfig()
    for bad_uri in (
        "/home/demo/tax/2025_return.pdf",
        "/home/demo/taxes/w2.txt",
        "/home/demo/.codex/auth.json",
        "/home/demo/.claude/.credentials.json",
        "/home/demo/projects/app/.env",
    ):
        fake = SearchResult(
            chunk_id="c1",
            doc_id="d1",
            uri=bad_uri,
            title="secret",
            source_harness="files",
            media_type=MediaType.TEXT,
            canonical_tier=RepoTier.REFERENCE,
            score=1.0,
            snippet="secret",
            l0_summary="secret",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        assert _is_leaked_result(fake, cfg) is True
