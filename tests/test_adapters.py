"""Tests for canonical project atlas discovery, harness/file adapters, chunking, and strict privacy."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pymupdf
import pytest

from chittam.adapters import (
    MEDIA_HANDLERS,
    chunk_code,
    collect_all_documents,
    load_antigravity_docs,
    load_claude_code_docs,
    load_codex_docs,
    load_file_docs,
)
from chittam.adapters.files import FileContext, register_media_handler
from chittam.atlas import build_or_load_atlas, discover_git_repos, save_atlas
from chittam.config import load_config
from chittam.models import AppConfig, CanonicalRepo, DocumentRecord, MediaType, PrivacyPolicy, RepoTier, SyncStats


def _init_git_repo(repo_dir: Path, days_ago: int | None = 1, commit_msg: str = "init") -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo_dir)], check=True, capture_output=True)
    if days_ago is None:
        return
    (repo_dir / "README.md").write_text(f"# {repo_dir.name}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_dir), "add", "README.md"], check=True, capture_output=True)
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_AUTHOR_DATE": iso,
        "GIT_COMMITTER_DATE": iso,
    }
    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "-m", commit_msg],
        check=True,
        capture_output=True,
        env=env,
    )


def test_atlas_discovery_tiers_depth_and_manual_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    # Active, unborn, reference, archive-by-age
    _init_git_repo(tmp_path / "Projects" / "myapp", days_ago=3)
    _init_git_repo(tmp_path / "Projects" / "fresh_unborn", days_ago=None)
    _init_git_repo(tmp_path / "Code" / "ref_proj", days_ago=90)
    _init_git_repo(tmp_path / "Code" / "old_proj", days_ago=220)

    # Snapshot / archive heuristics even when commit is recent (1 day ago)
    _init_git_repo(tmp_path / "Projects" / "Nov27" / "snap", days_ago=1)
    _init_git_repo(tmp_path / "Projects" / "Dec1", days_ago=1)
    _init_git_repo(tmp_path / "Code" / "claude-user-memory _OCT17_2025copy", days_ago=1)
    _init_git_repo(tmp_path / "Projects" / "backup" / "mirror", days_ago=1)

    # Non-snapshot words containing "dec"/"nov"/"copy" must remain ACTIVE
    _init_git_repo(tmp_path / "Projects" / "decision-engine", days_ago=2)
    _init_git_repo(tmp_path / "Projects" / "novel-reader", days_ago=2)
    _init_git_repo(tmp_path / "Projects" / "copyright-tool", days_ago=2)

    # Depth boundary: depth 4 included, depth 5 skipped
    _init_git_repo(tmp_path / "Projects" / "a" / "b" / "c" / "d_ok", days_ago=2)
    _init_git_repo(tmp_path / "Projects" / "a" / "b" / "c" / "d" / "too_deep", days_ago=2)

    # Unlisted home subdir skipped when home_top_level_files_only=True
    _init_git_repo(tmp_path / "unlisted_home_dir" / "ignored_repo", days_ago=1)

    # Excluded tax dir must never be discovered
    _init_git_repo(tmp_path / "tax" / "tax_repo", days_ago=1)

    # Duplicate repo names across roots get unique project_ids sharing a family
    _init_git_repo(tmp_path / "Projects" / "geoatlas", days_ago=2)
    _init_git_repo(tmp_path / "Documents" / "geoatlas", days_ago=200)

    cfg = AppConfig(
        data_dir=str(tmp_path / ".local/share/chittam"),
        config_dir=str(tmp_path / ".config/chittam"),
        privacy=PrivacyPolicy(
            scan_roots=[
                str(tmp_path / "Projects"),
                str(tmp_path / "Code"),
                str(tmp_path / "Documents"),
                "~",
            ],
            home_top_level_files_only=True,
        ),
    )

    repos = discover_git_repos(cfg)
    assert repos["myapp"].tier == RepoTier.ACTIVE
    assert repos["fresh-unborn"].tier == RepoTier.ACTIVE
    assert repos["ref-proj"].tier == RepoTier.REFERENCE
    assert repos["old-proj"].tier == RepoTier.ARCHIVE

    assert repos["snap"].tier == RepoTier.ARCHIVE
    assert repos["dec1"].tier == RepoTier.ARCHIVE
    assert repos["claude-user-memory-oct17-2025copy"].tier == RepoTier.ARCHIVE
    assert repos["mirror"].tier == RepoTier.ARCHIVE

    assert repos["decision-engine"].tier == RepoTier.ACTIVE
    assert repos["novel-reader"].tier == RepoTier.ACTIVE
    assert repos["copyright-tool"].tier == RepoTier.ACTIVE

    assert "d-ok" in repos
    assert "too-deep" not in repos
    assert "ignored-repo" not in repos
    assert "tax-repo" not in repos

    geoatlas_family = [r for r in repos.values() if r.family == "geoatlas"]
    assert len(geoatlas_family) == 2
    assert len({r.project_id for r in geoatlas_family}) == 2
    assert repos["geoatlas"].tier == RepoTier.ACTIVE

    # Save and test manual_override preservation on refresh
    repos["old-proj"].tier = RepoTier.ACTIVE
    repos["old-proj"].manual_override = True
    repos["old-proj"].notes = "Pinned by user"
    save_atlas(repos, cfg.atlas_path)

    reloaded = build_or_load_atlas(cfg, refresh=True)
    assert reloaded["old-proj"].manual_override is True
    assert reloaded["old-proj"].tier == RepoTier.ACTIVE
    assert reloaded["old-proj"].notes == "Pinned by user"


def test_antigravity_adapter_artifacts_transcripts_and_redaction(tmp_path: Path) -> None:
    repo_path = tmp_path / "Projects" / "chittam"
    repo_path.mkdir(parents=True)
    atlas = {
        "chittam": CanonicalRepo(
            project_id="chittam",
            family="chittam",
            name="chittam",
            path=str(repo_path),
            tier=RepoTier.ACTIVE,
        )
    }

    brain = tmp_path / ".gemini" / "antigravity" / "brain"
    conv = brain / "abc12345-dead-beef"
    logs = conv / ".system_generated" / "logs"
    logs.mkdir(parents=True)

    (conv / "implementation_plan.md").write_text(
        f"# Chittam Plan\n\nWorking in {repo_path} with key sk-testsecret1234567890abcd.\n",
        encoding="utf-8",
    )
    (conv / "implementation_plan.md.resolved.0").write_text("duplicate snapshot", encoding="utf-8")
    (conv / "implementation_plan.md.metadata.json").write_text(
        json.dumps({"summary": "Chittam implementation plan", "updatedAt": "2026-11-22T20:00:00Z"}),
        encoding="utf-8",
    )

    turns = [
        {"step_index": 0, "source": "SYSTEM", "type": "SYSTEM_MESSAGE", "content": "ignore system"},
        {
            "step_index": 1,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "created_at": "2026-11-22T20:01:00Z",
            "content": f"<USER_REQUEST>\nBuild memory server in {repo_path}\n</USER_REQUEST>\n<ADDITIONAL_METADATA>noise</ADDITIONAL_METADATA>",
        },
        {
            "step_index": 2,
            "source": "MODEL",
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-11-22T20:02:00Z",
            "tool_calls": [{"name": "read_file"}],
        },
        {
            "step_index": 3,
            "source": "MODEL",
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-11-22T20:03:00Z",
            "content": "Built SQLite hybrid store. Never store ghp_abcdefghijklmnopqrstuvwxyz123456.",
        },
    ]
    (logs / "transcript.jsonl").write_text(
        "\n".join(json.dumps(t) for t in turns),
        encoding="utf-8",
    )

    cfg = AppConfig()
    cfg.harnesses.antigravity_brain_dir = str(brain)

    docs = load_antigravity_docs(cfg, atlas)
    assert len(docs) == 2  # implementation_plan.md + transcript.jsonl (NOT .resolved.0)
    by_type = {d.media_type: d for d in docs}
    assert MediaType.MARKDOWN in by_type
    assert MediaType.TRANSCRIPT in by_type

    md_doc = by_type[MediaType.MARKDOWN]
    assert md_doc.project_id == "chittam"
    assert md_doc.canonical_tier == RepoTier.ACTIVE
    assert "sk-testsecret" not in md_doc.chunks[0].content
    assert "[REDACTED_API_KEY]" in md_doc.chunks[0].content
    assert md_doc.chunks[0].contextual_header.startswith("[chittam | active |")

    tx_doc = by_type[MediaType.TRANSCRIPT]
    assert tx_doc.project_id == "chittam"
    joined = "\n".join(c.content for c in tx_doc.chunks)
    assert "Build memory server" in joined
    assert "ignore system" not in joined
    assert "ADDITIONAL_METADATA" not in joined
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in joined
    assert "[REDACTED_GITHUB_TOKEN]" in joined


def test_codex_adapter_readonly_sqlite_and_legacy_memories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_path = tmp_path / "Projects" / "CookBook"
    repo_path.mkdir(parents=True)
    atlas = {
        "cookbook": CanonicalRepo(
            project_id="cookbook",
            family="cookbook",
            name="CookBook",
            path=str(repo_path),
            tier=RepoTier.ACTIVE,
        )
    }

    codex_home = tmp_path / ".codex"
    codex_home.mkdir(parents=True)

    # state_5.sqlite
    s_conn = sqlite3.connect(codex_home / "state_5.sqlite")
    s_conn.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, cwd TEXT, title TEXT, name TEXT, git_branch TEXT, updated_at INTEGER)")
    s_conn.execute(
        "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?)",
        ("th-1", str(repo_path), "CookBook Recipe Scaling", "CookBook Scaling", "main", 1795324859),
    )
    s_conn.commit()
    s_conn.close()

    # memories_1.sqlite
    m_conn = sqlite3.connect(codex_home / "memories_1.sqlite")
    m_conn.execute(
        "CREATE TABLE stage1_outputs (thread_id TEXT, source_updated_at INTEGER, raw_memory TEXT, rollout_summary TEXT, rollout_slug TEXT, generated_at INTEGER)"
    )
    m_conn.execute(
        "INSERT INTO stage1_outputs VALUES (?, ?, ?, ?, ?, ?)",
        (
            "th-1",
            1795324859,
            "Recipely recipe scaling factor uses fractional yield. Key AIzaSyDummyGoogleKey1234567890abcdef",
            "Implemented recipe scaling engine for CookBook.",
            "cookbook-scaling",
            1795324860,
        ),
    )
    m_conn.commit()
    m_conn.close()

    # thread_history_1.sqlite in WAL mode, held open during read!
    h_conn = sqlite3.connect(codex_home / "thread_history_1.sqlite")
    h_conn.execute("PRAGMA journal_mode=WAL")
    h_conn.execute(
        "CREATE TABLE thread_items (thread_id TEXT, turn_id TEXT, item_id TEXT, rollout_ordinal INTEGER, created_at_ms INTEGER, item_json TEXT, item_type TEXT)"
    )
    h_conn.execute(
        "INSERT INTO thread_items VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "th-1",
            "t1",
            "i1",
            1,
            1795324800000,
            json.dumps({"type": "userMessage", "content": [{"type": "text", "text": "Scale recipe servings"}]}),
            "userMessage",
        ),
    )
    h_conn.execute(
        "INSERT INTO thread_items VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "th-1",
            "t1",
            "i2",
            2,
            1795324810000,
            json.dumps({"type": "agentMessage", "text": "Updated CookBook scaling multiplier."}),
            "agentMessage",
        ),
    )
    h_conn.execute(
        "INSERT INTO thread_items VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "th-1",
            "t1",
            "i3",
            3,
            1795324820000,
            json.dumps({"type": "commandExecution", "command": "pytest"}),
            "commandExecution",
        ),
    )
    h_conn.commit()

    # Legacy memories markdown
    legacy_dir = tmp_path / "codex_legacy" / "memories"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "MEMORY.md").write_text("# Legacy Codex Memory\nRemember CookBook constraints.\n", encoding="utf-8")

    # Spy on sqlite3.connect to ensure URI read-only mode (?mode=ro) is always passed
    seen_uris: list[tuple[str, bool]] = []
    real_connect = sqlite3.connect

    def _spy_connect(database, *args, **kwargs):
        seen_uris.append((str(database), bool(kwargs.get("uri", False))))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _spy_connect)

    cfg = AppConfig()
    cfg.harnesses.codex_home_dir = str(codex_home)
    cfg.harnesses.codex_legacy_memories_dirs = [str(legacy_dir)]

    try:
        docs = load_codex_docs(cfg, atlas)
    finally:
        h_conn.close()

    assert len(docs) == 3  # stage1_outputs, thread_history, legacy MEMORY.md
    assert seen_uris
    assert all(uri_flag and "?mode=ro" in db_arg for db_arg, uri_flag in seen_uris)

    mem_doc = next(d for d in docs if "#stage1:" in d.uri)
    assert mem_doc.project_id == "cookbook"
    assert "[REDACTED_GOOGLE_KEY]" in "\n".join(c.content for c in mem_doc.chunks)
    assert "AIzaSyDummy" not in "\n".join(c.content for c in mem_doc.chunks)

    tx_doc = next(d for d in docs if d.uri == "codex://thread/th-1")
    assert tx_doc.project_id == "cookbook"
    tx_body = "\n".join(c.content for c in tx_doc.chunks)
    assert "Scale recipe servings" in tx_body
    assert "Updated CookBook scaling multiplier." in tx_body
    assert "commandExecution" not in tx_body


def test_claude_code_adapter_sessions_and_history(tmp_path: Path) -> None:
    repo_path = tmp_path / "Research" / "citecheck"
    repo_path.mkdir(parents=True)
    atlas = {
        "citecheck": CanonicalRepo(
            project_id="citecheck",
            family="citecheck",
            name="citecheck",
            path=str(repo_path),
            tier=RepoTier.REFERENCE,
        )
    }

    projects_dir = tmp_path / ".claude" / "projects"
    proj_sub = projects_dir / "-Users-tmp-citecheck"
    proj_sub.mkdir(parents=True)

    events = [
        {"type": "file-history-snapshot", "messageId": "m0"},
        {
            "type": "user",
            "cwd": str(repo_path),
            "sessionId": "sess-123",
            "gitBranch": "feat/citation",
            "timestamp": "2026-11-21T10:00:00Z",
            "message": {"role": "user", "content": "<command-name>/goal</command-name><command-args>Verify citation graph</command-args>"},
        },
        {
            "type": "assistant",
            "cwd": str(repo_path),
            "sessionId": "sess-123",
            "gitBranch": "feat/citation",
            "timestamp": "2026-11-21T10:01:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Internal hidden chain of thought"},
                    {"type": "tool_use", "name": "Read", "input": {}},
                    {"type": "text", "text": "Checked citation graph; avoid AKIAIOSFODNN7EXAMPLE in logs."},
                ],
            },
        },
    ]
    (proj_sub / "sess-123.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    hist_path = tmp_path / ".claude" / "history.jsonl"
    hist_path.write_text(
        json.dumps(
            {
                "display": "Audit CiteCheck citation retrieval latency",
                "timestamp": 1775680546478,
                "project": str(repo_path),
                "sessionId": "sess-123",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = AppConfig()
    cfg.harnesses.claude_projects_dir = str(projects_dir)
    cfg.harnesses.claude_history_path = str(hist_path)

    docs = load_claude_code_docs(cfg, atlas)
    assert len(docs) == 2

    sess_doc = next(d for d in docs if d.uri.endswith("sess-123.jsonl"))
    assert sess_doc.source_harness == "claude_code"
    assert sess_doc.project_id == "citecheck"
    assert sess_doc.canonical_tier == RepoTier.REFERENCE
    assert sess_doc.metadata["git_branch"] == "feat/citation"
    body = "\n".join(c.content for c in sess_doc.chunks)
    assert "Verify citation graph" in body
    assert "Internal hidden chain of thought" not in body
    assert "[REDACTED_AWS_KEY]" in body
    assert "AKIAIOSFODNN7EXAMPLE" not in body


def test_strict_privacy_exclusions_never_ingested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    # Forbidden files & directories
    (tmp_path / "tax").mkdir()
    (tmp_path / "tax" / "2025_return.txt").write_text("SECRET TAX RETURN 2025", encoding="utf-8")
    (tmp_path / "tax" / "w2.md").write_text("# W2 confidential", encoding="utf-8")

    (tmp_path / "Documents" / "taxes").mkdir(parents=True)
    (tmp_path / "Documents" / "taxes" / "filing.txt").write_text("TAXES PRIVATE DATA", encoding="utf-8")

    (tmp_path / "id_document.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 15000)

    app_dir = tmp_path / "Projects" / "chittam"
    _init_git_repo(app_dir, days_ago=1)
    (app_dir / "main.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (app_dir / ".env").write_text("OPENAI_API_KEY=sk-secret1234567890abcdef\n", encoding="utf-8")
    (app_dir / "auth.json").write_text('{"token": "secret"}', encoding="utf-8")
    (app_dir / ".credentials.json").write_text('{"secret": "123"}', encoding="utf-8")
    (app_dir / "oauth_creds.json").write_text('{"refresh_token": "abc"}', encoding="utf-8")

    (tmp_path / ".gemini").mkdir()
    (tmp_path / ".gemini" / "oauth_creds.json").write_text('{"token": "bad"}', encoding="utf-8")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / ".credentials.json").write_text('{"token": "bad"}', encoding="utf-8")
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "auth.json").write_text('{"token": "bad"}', encoding="utf-8")

    # Legitimate allowed files
    (tmp_path / "top_note.md").write_text("# Top Level Note\nAllowed in ~", encoding="utf-8")
    (tmp_path / "Documents" / "notes.md").write_text("# Doc Note\nAllowed in Documents", encoding="utf-8")
    (tmp_path / "unlisted_folder").mkdir()
    (tmp_path / "unlisted_folder" / "deep.md").write_text("Should not be scanned from ~", encoding="utf-8")

    cfg = AppConfig(
        data_dir=str(tmp_path / ".local/share/chittam"),
        config_dir=str(tmp_path / ".config/chittam"),
        privacy=PrivacyPolicy(
            scan_roots=[str(tmp_path / "Projects"), str(tmp_path / "Documents"), "~"],
            home_top_level_files_only=True,
            # Users can add their own sensitive files on top of the defaults.
            exclude_path_globs=[*PrivacyPolicy().exclude_path_globs, "~/id_document.jpg"],
        ),
        harnesses={
            "antigravity_brain_dir": str(tmp_path / ".gemini/antigravity/brain"),
            "codex_home_dir": str(tmp_path / ".codex"),
            "codex_legacy_memories_dirs": [],
            "claude_projects_dir": str(tmp_path / ".claude/projects"),
            "claude_history_path": str(tmp_path / ".claude/history.jsonl"),
        },
    )

    atlas = discover_git_repos(cfg)
    vision_calls: list[Path] = []

    def fake_vision(path: Path, hint: str) -> str:
        vision_calls.append(path)
        return "caption"

    docs, stats = collect_all_documents(cfg, atlas, vision_fn=fake_vision)
    uris = {d.uri for d in docs}
    all_content = "\n".join(c.content for d in docs for c in d.chunks)

    assert stats.skipped_excluded >= 8
    assert not vision_calls

    forbidden_markers = [
        "/tax/",
        "/taxes/",
        "id_document.jpg",
        ".env",
        "auth.json",
        ".credentials.json",
        "oauth_creds.json",
    ]
    for uri in uris:
        for marker in forbidden_markers:
            assert marker not in uri, f"Forbidden path leaked: {uri}"
    assert "SECRET TAX RETURN" not in all_content
    assert "TAXES PRIVATE DATA" not in all_content

    assert str(tmp_path / "top_note.md") in uris
    assert str(tmp_path / "Documents" / "notes.md") in uris
    assert str(app_dir / "main.py") in uris
    assert str(tmp_path / "unlisted_folder" / "deep.md") not in uris


def test_code_chunking_pdf_images_cap_and_extensible_handlers(tmp_path: Path) -> None:
    # 1. Structural code chunking across languages
    py_code = "\n".join(
        [
            "import os",
            "",
            "def first_helper(x: int) -> int:",
            "    return x + 1",
            "",
            "class LargeService:",
            *[f"    line_{i} = {i} * 'x'" for i in range(80)],
        ]
    )
    py_chunks = chunk_code("doc1", py_code, ".py", "proj", RepoTier.ACTIVE, "/repo/a.py", "2026-11-22T10:00:00Z")
    assert len(py_chunks) >= 2
    assert all(c.contextual_header.startswith("[proj | active | /repo/a.py") for c in py_chunks)
    assert all(len(c.content) <= 1350 for c in py_chunks)
    assert any(c.symbol_name == "LargeService" for c in py_chunks)

    for suffix, snippet, expected_sym in [
        (".ts", "export async function fetchDocs(q: string) {\n  return q;\n}\n", "fetchDocs"),
        (".dart", "class CookbookScreen {\n  void build() {}\n}\n", "CookbookScreen"),
        (".rs", "pub fn search_hybrid(limit: usize) -> usize {\n    limit\n}\n", "search_hybrid"),
        (".go", "func BuildAtlas(root string) error {\n    return nil\n}\n", "BuildAtlas"),
        (".sh", "run_sync() {\n  echo ok\n}\n", "run_sync"),
    ]:
        chunks = chunk_code("d", snippet, suffix, "proj", RepoTier.ACTIVE, f"/repo/f{suffix}", "2026-11-22T10:00:00Z")
        assert chunks and chunks[0].symbol_name == expected_sym

    # 2. Code file cap, PDF extraction + sparse page note, image cache + vision cap
    repo = tmp_path / "Projects" / "demo"
    _init_git_repo(repo, days_ago=1)
    for i in range(5):
        (repo / f"mod_{i}.py").write_text(f"def fn_{i}():\n    return {i}\n", encoding="utf-8")

    # PDF with 1 text page + 1 blank sparse page + 1 extra page beyond cap
    pdf_path = tmp_path / "Projects" / "arch.pdf"
    pdf = pymupdf.open()
    p1 = pdf.new_page()
    p1.insert_text((72, 72), "Chittam local semantic memory architecture specification with hybrid RRF ranking.")
    pdf.new_page()  # blank sparse page
    p3 = pdf.new_page()
    p3.insert_text((72, 72), "Page 3 should be truncated by max_pdf_pages=2")
    pdf.save(pdf_path)
    pdf.close()

    # Images: 1 tiny (<min_image_bytes), 1 icon noise, 1 cached, 3 uncached (cap=2)
    img_dir = tmp_path / "Projects" / "imgs"
    img_dir.mkdir(parents=True)
    (img_dir / "tiny.png").write_bytes(b"123")
    icon_dir = tmp_path / "Projects" / "ios" / "Assets.xcassets" / "AppIcon.appiconset"
    icon_dir.mkdir(parents=True)
    (icon_dir / "Icon-App-1024x1024@1x.png").write_bytes(b"\x89PNG" + b"0" * 15000)

    cached_bytes = b"\x89PNG_CACHED_" + b"1" * 12000
    (img_dir / "cached.png").write_bytes(cached_bytes)
    import hashlib

    cached_sha = hashlib.sha256(cached_bytes).hexdigest()

    for i in range(3):
        (img_dir / f"screen_{i}.png").write_bytes(f"PNG_UNCA_{i}_".encode() + b"2" * 12000)

    cfg = AppConfig(
        privacy=PrivacyPolicy(
            scan_roots=[str(tmp_path / "Projects")],
            max_code_files_per_repo=3,
            max_pdf_pages=2,
            min_image_bytes=1024,
            max_images_per_sync=2,
        )
    )
    atlas = discover_git_repos(cfg)
    vision_calls: list[str] = []

    def fake_vision(path: Path, hint: str) -> str:
        vision_calls.append(path.name)
        return f"Diagram showing {path.name} ({hint})"

    def fake_cache(sha: str) -> str | None:
        return "Cached system architecture diagram" if sha == cached_sha else None

    stats = SyncStats()
    docs = load_file_docs(cfg, atlas, vision_fn=fake_vision, image_cache_lookup=fake_cache, stats=stats)

    code_docs = [d for d in docs if d.media_type == MediaType.CODE]
    assert len(code_docs) == 3

    pdf_docs = [d for d in docs if d.media_type == MediaType.PDF]
    assert len(pdf_docs) == 1
    pdf_text = "\n".join(c.content for c in pdf_docs[0].chunks)
    assert "hybrid RRF ranking" in pdf_text
    assert "sparse/scanned page" in pdf_text
    assert "Page 3 should be truncated" not in pdf_text

    img_docs = [d for d in docs if d.media_type == MediaType.IMAGE]
    assert len(img_docs) == 3  # 1 cached + 2 newly captioned
    assert stats.images_cached == 1
    assert stats.images_captioned == 2
    assert len(vision_calls) == 2
    assert "cached.png" not in vision_calls
    assert all("tiny.png" not in d.uri and "AppIcon" not in d.uri for d in img_docs)

    # 3. Extensible MEDIA_HANDLERS registry (future audio adapter)
    audio_path = tmp_path / "Projects" / "meeting.mp3"
    audio_path.write_bytes(b"ID3" + b"0" * 512)

    def fake_audio_handler(path: Path, ctx: FileContext) -> DocumentRecord:
        return DocumentRecord(
            doc_id=f"audio:{path.name}",
            uri=str(path),
            source_harness="files",
            media_type=MediaType.AUDIO,
            project_id=ctx.repo.project_id if ctx.repo else None,
            canonical_tier=ctx.repo.tier if ctx.repo else RepoTier.REFERENCE,
            title=path.name,
            l0_summary="Audio transcript",
            sha256="abc",
            updated_at="2026-11-22T10:00:00Z",
        )

    old_audio = MEDIA_HANDLERS.get(MediaType.AUDIO)
    try:
        register_media_handler(MediaType.AUDIO, {".mp3"}, fake_audio_handler)
        cfg_audio = AppConfig(
            privacy=PrivacyPolicy(
                scan_roots=[str(tmp_path / "Projects")],
                enabled_media=[MediaType.AUDIO],
            )
        )
        audio_docs = load_file_docs(cfg_audio, atlas)
        assert len(audio_docs) == 1
        assert audio_docs[0].media_type == MediaType.AUDIO
    finally:
        if old_audio is None:
            MEDIA_HANDLERS.pop(MediaType.AUDIO, None)
        else:
            MEDIA_HANDLERS[MediaType.AUDIO] = old_audio


@pytest.mark.skipif(
    os.environ.get("CHITTAM_LIVE_TESTS") != "1",
    reason="Reads real local agent histories; set CHITTAM_LIVE_TESTS=1 to run.",
)
def test_real_local_harnesses_dry_run_readonly() -> None:
    """Verify read-only ingestion on real local Antigravity, Codex, and Claude Code data if present."""
    cfg = load_config()
    atlas: dict[str, CanonicalRepo] = {}
    ag_docs = load_antigravity_docs(cfg, atlas)
    cx_docs = load_codex_docs(cfg, atlas)
    cc_docs = load_claude_code_docs(cfg, atlas)

    if Path(cfg.harnesses.antigravity_brain_dir).expanduser().exists():
        assert len(ag_docs) > 0
    if (Path(cfg.harnesses.codex_home_dir).expanduser() / "thread_history_1.sqlite").exists():
        assert len(cx_docs) > 0
    if Path(cfg.harnesses.claude_projects_dir).expanduser().exists():
        assert len(cc_docs) > 0

    for doc in [*ag_docs, *cx_docs, *cc_docs]:
        assert doc.chunks
        assert all(c.contextual_header for c in doc.chunks)
