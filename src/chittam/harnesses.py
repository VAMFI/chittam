"""Reversible MCP registration for Antigravity, Codex, and Claude Code."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chittam.models import AppConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CODEGRAPH_ARGS = ["serve", "--mcp"]
_CHITTAM_ARGS: list[str] = []


def resolve_chittam_mcp_command() -> str:
    """Resolve the absolute path to `chittam-mcp`."""
    candidates = [
        _REPO_ROOT / ".venv" / "bin" / "chittam-mcp",
        Path(sys.executable).resolve().parent / "chittam-mcp",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    which_bin = shutil.which("chittam-mcp")
    if which_bin:
        return str(Path(which_bin).expanduser())
    return str(_REPO_ROOT / ".venv" / "bin" / "chittam-mcp")


def _semver_key(path: Path) -> tuple[int, ...]:
    """Extract numeric version segments from an nvm path for descending sort."""
    for part in path.parts:
        if part.startswith("v") and any(ch.isdigit() for ch in part):
            nums = [int(seg) for seg in re.findall(r"\d+", part)]
            if nums:
                return tuple(nums)
    return (0,)


def resolve_codegraph_command(cfg: AppConfig | None = None) -> str | None:
    """Resolve the full path to `codegraph` if installed on this machine."""
    if cfg is not None and not cfg.enable_codegraph:
        return None

    which_bin = shutil.which("codegraph")
    if which_bin:
        return str(Path(which_bin).expanduser())

    nvm_matches = sorted(
        Path("~/.nvm/versions/node").expanduser().glob("*/bin/codegraph"),
        key=_semver_key,
        reverse=True,
    )
    candidates = [
        *nvm_matches,
        Path("/opt/homebrew/bin/codegraph"),
        Path("/usr/local/bin/codegraph"),
        Path("~/.npm-global/bin/codegraph").expanduser(),
        Path("~/.local/bin/codegraph").expanduser(),
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _desired_servers(cfg: AppConfig) -> dict[str, tuple[str, list[str]]]:
    """Return server definitions to register keyed by MCP server name."""
    servers: dict[str, tuple[str, list[str]]] = {
        "chittam": (resolve_chittam_mcp_command(), list(_CHITTAM_ARGS)),
    }
    codegraph_cmd = resolve_codegraph_command(cfg)
    if codegraph_cmd:
        servers["codegraph"] = (codegraph_cmd, list(_CODEGRAPH_ARGS))
    return servers


def _toml_quote(value: str) -> str:
    """Serialize a Python string as a valid TOML basic string."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\b", "\\b")
        .replace("\f", "\\f")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _toml_array(values: list[str]) -> str:
    """Serialize a list of strings as a TOML inline array."""
    return "[" + ", ".join(_toml_quote(item) for item in values) + "]"


def _update_multiline_state(line: str, active_delim: str | None) -> str | None:
    """Track TOML multiline string delimiters (`\"\"\"` / `'''`) across lines."""
    i = 0
    n = len(line)
    in_single: str | None = None
    while i < n:
        ch = line[i]
        if active_delim is not None:
            if active_delim == '"""' and ch == "\\":
                i += 2
                continue
            if line.startswith(active_delim, i):
                active_delim = None
                i += 3
                continue
            i += 1
            continue

        if in_single is not None:
            if in_single == '"' and ch == "\\":
                i += 2
                continue
            if ch == in_single:
                in_single = None
            i += 1
            continue

        if ch == "#":
            break
        if line.startswith('"""', i):
            active_delim = '"""'
            i += 3
            continue
        if line.startswith("'''", i):
            active_delim = "'''"
            i += 3
            continue
        if ch in ('"', "'"):
            in_single = ch
        i += 1
    return active_delim


def _extract_table_header(line: str) -> tuple[bool, str | None]:
    """Return (is_header, header_text) for a non-multiline-string line."""
    stripped = line.lstrip()
    if not stripped.startswith("["):
        return False, None
    if stripped.startswith("[["):
        return True, None

    in_quote: str | None = None
    i = 1
    while i < len(stripped):
        ch = stripped[i]
        if in_quote is not None:
            if in_quote == '"' and ch == "\\":
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
        else:
            if ch in ('"', "'"):
                in_quote = ch
            elif ch == "]":
                rest = stripped[i + 1 :].lstrip()
                if not rest or rest.startswith("#"):
                    return True, stripped[: i + 1]
                return False, None
        i += 1
    return False, None


def _parse_toml_table_path(header_text: str) -> tuple[str, ...] | None:
    """Resolve exact TOML table key path using stdlib `tomllib`."""
    try:
        parsed = tomllib.loads(f"{header_text}\n_chittam_probe = true\n")
    except tomllib.TOMLDecodeError:
        return None
    path: list[str] = []
    cur: Any = parsed
    while isinstance(cur, dict):
        if list(cur.keys()) == ["_chittam_probe"]:
            return tuple(path)
        if len(cur) != 1:
            return None
        key, nxt = next(iter(cur.items()))
        path.append(key)
        cur = nxt
    return None


def _find_toml_sections(toml_text: str) -> list[tuple[int, int, tuple[str, ...] | None]]:
    """Return (start_line, end_line_exclusive, table_path) for every table section."""
    lines = toml_text.splitlines(keepends=True)
    headers: list[tuple[int, tuple[str, ...] | None]] = []
    ml_state: str | None = None
    for idx, line in enumerate(lines):
        if ml_state is None:
            is_hdr, hdr_text = _extract_table_header(line)
            if is_hdr:
                path = _parse_toml_table_path(hdr_text) if hdr_text else None
                headers.append((idx, path))
        ml_state = _update_multiline_state(line, ml_state)

    sections: list[tuple[int, int, tuple[str, ...] | None]] = []
    for i, (start_idx, path) in enumerate(headers):
        end_idx = headers[i + 1][0] if i + 1 < len(headers) else len(lines)
        sections.append((start_idx, end_idx, path))
    return sections


def _remove_inline_server_key(section_lines: list[str], server_name: str) -> list[str]:
    """Remove an inline `<server_name> = ...` entry inside a `[mcp_servers]` table."""
    pat = re.compile(rf"^\s*(?:{re.escape(server_name)}|\"{re.escape(server_name)}\"|'{re.escape(server_name)}')\s*=")
    out: list[str] = []
    ml_state: str | None = None
    for line in section_lines:
        if ml_state is None and pat.match(line):
            ml_state = _update_multiline_state(line, ml_state)
            continue
        ml_state = _update_multiline_state(line, ml_state)
        out.append(line)
    return out


def _remove_toml_mcp_server(toml_text: str, server_name: str) -> str:
    """Remove `[mcp_servers.<server_name>]` (and subtables) preserving comments and formatting."""
    lines = toml_text.splitlines(keepends=True)
    sections = _find_toml_sections(toml_text)
    if not sections:
        return toml_text

    drop_ranges: list[tuple[int, int]] = []
    inline_parent_ranges: list[tuple[int, int]] = []
    for start_idx, end_idx, path in sections:
        if path and len(path) >= 2 and path[0] == "mcp_servers" and path[1] == server_name:
            drop_ranges.append((start_idx, end_idx))
        elif path == ("mcp_servers",):
            inline_parent_ranges.append((start_idx, end_idx))

    if not drop_ranges and not inline_parent_ranges:
        return toml_text

    drop_set: set[int] = set()
    for start_idx, end_idx in drop_ranges:
        for idx in range(start_idx, end_idx):
            drop_set.add(idx)

    kept: list[str] = []
    idx = 0
    while idx < len(lines):
        if idx in drop_set:
            idx += 1
            continue
        matched_parent = next((r for r in inline_parent_ranges if r[0] == idx), None)
        if matched_parent:
            p_start, p_end = matched_parent
            kept.extend(_remove_inline_server_key(lines[p_start:p_end], server_name))
            idx = p_end
            continue
        kept.append(lines[idx])
        idx += 1

    # Collapse consecutive blank lines introduced by section removal, keeping max one blank line.
    cleaned: list[str] = []
    prev_blank = False
    for line in kept:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    result = "".join(cleaned)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def _upsert_toml_mcp_server(
    toml_text: str,
    server_name: str,
    command: str,
    args: list[str],
) -> str:
    """Insert or update `[mcp_servers.<server_name>]` cleanly without clobbering existing TOML."""
    base = _remove_toml_mcp_server(toml_text, server_name)
    block = (
        f"[mcp_servers.{server_name}]\n"
        f"command = {_toml_quote(command)}\n"
        f"args = {_toml_array(args)}\n"
    )

    lines = base.splitlines(keepends=True)
    sections = _find_toml_sections(base)
    mcp_sections = [
        (start_idx, end_idx, path)
        for start_idx, end_idx, path in sections
        if path and path[0] == "mcp_servers"
    ]

    if mcp_sections:
        insert_idx = mcp_sections[-1][1]
        prefix = "".join(lines[:insert_idx])
        suffix = "".join(lines[insert_idx:])
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        if prefix and not prefix.endswith("\n\n"):
            prefix += "\n"
        if suffix and not suffix.startswith("\n"):
            block += "\n"
        updated = prefix + block + suffix
    else:
        updated = base
        if updated and not updated.endswith("\n"):
            updated += "\n"
        if updated.strip() and not updated.endswith("\n\n"):
            updated += "\n"
        updated += block

    tomllib.loads(updated)
    return updated


def _backup_name_for(target: Path, ts: str) -> str:
    """Build `<basename>.<ts>.bak`, stripping any leading dot (e.g. `.claude.json`)."""
    clean_name = target.name.lstrip(".")
    return f"{clean_name}.{ts}.bak"


def _create_backup(target: Path, backup_dir: Path, ts: str) -> Path:
    """Create a timestamped `0600` backup inside `backup_dir` before editing."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(backup_dir, 0o700)
    except OSError:
        pass

    candidate = backup_dir / _backup_name_for(target, ts)
    counter = 1
    while candidate.exists():
        candidate = backup_dir / f"{target.name.lstrip('.')}.{ts}.{counter}.bak"
        counter += 1

    candidate.write_bytes(target.read_bytes())
    try:
        os.chmod(candidate, 0o600)
    except OSError:
        pass
    return candidate


def _atomic_write_text(target: Path, content: str, mode: int | None = None) -> None:
    """Atomically replace `target` while preserving file permissions."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.chittam.tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                loaded.setdefault("version", 1)
                loaded.setdefault("harnesses", {})
                loaded.setdefault("history", [])
                return loaded
        except json.JSONDecodeError:
            pass
    return {"version": 1, "harnesses": {}, "history": []}


def _save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        mode=0o600,
    )


def _validate_json_servers(
    target: Path,
    expected_servers: dict[str, tuple[str, list[str]]],
) -> None:
    parsed = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("mcpServers"), dict):
        raise ValueError(f"Missing mcpServers object in {target}")
    mcp_servers = parsed["mcpServers"]
    for name, (cmd, args) in expected_servers.items():
        entry = mcp_servers.get(name)
        if not isinstance(entry, dict) or entry.get("command") != cmd or entry.get("args") != args:
            raise ValueError(f"Validation failed for server '{name}' in {target}")


def _validate_toml_servers(
    target: Path,
    expected_servers: dict[str, tuple[str, list[str]]],
) -> None:
    parsed = tomllib.loads(target.read_text(encoding="utf-8"))
    mcp_servers = parsed.get("mcp_servers")
    if not isinstance(mcp_servers, dict):
        raise ValueError(f"Missing [mcp_servers] table in {target}")
    for name, (cmd, args) in expected_servers.items():
        entry = mcp_servers.get(name)
        if not isinstance(entry, dict) or entry.get("command") != cmd or entry.get("args") != args:
            raise ValueError(f"Validation failed for [mcp_servers.{name}] in {target}")


def _install_json_harness(
    *,
    harness: str,
    target: Path,
    servers: dict[str, tuple[str, list[str]]],
    include_stdio_type: bool,
    backup_dir: Path,
    manifest: dict[str, Any],
    ts: str,
    create_missing: bool,
) -> dict[str, Any]:
    if not target.exists():
        if not create_missing:
            return {"status": "skipped", "reason": "config_not_found", "path": str(target)}
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, "{}\n", mode=0o600)

    original_mode = stat.S_IMODE(target.stat().st_mode)
    raw_before = target.read_text(encoding="utf-8")
    data = json.loads(raw_before) if raw_before.strip() else {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {target}")

    backup_path = _create_backup(target, backup_dir, ts)

    existing_servers = data.get("mcpServers")
    if existing_servers is None:
        mcp_servers: dict[str, Any] = {}
        data["mcpServers"] = mcp_servers
    elif isinstance(existing_servers, dict):
        mcp_servers = dict(existing_servers)
        data["mcpServers"] = mcp_servers
    else:
        raise ValueError(f"Invalid mcpServers field in {target}")

    prior_entry = manifest["harnesses"].get(harness, {})
    prior_managed = set(prior_entry.get("managed_servers") or [])
    newly_managed: list[str] = []
    for name, (cmd, args) in servers.items():
        if name not in mcp_servers or name in prior_managed or name == "chittam":
            newly_managed.append(name)
            entry: dict[str, Any] = {}
            if include_stdio_type:
                entry["type"] = "stdio"
            entry["command"] = cmd
            entry["args"] = list(args)
            mcp_servers[name] = entry

    expected_present = {name: servers[name] for name in newly_managed}
    try:
        rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        _atomic_write_text(target, rendered, mode=original_mode)
        _validate_json_servers(target, expected_present)
    except Exception:
        _atomic_write_text(target, backup_path.read_text(encoding="utf-8"), mode=original_mode)
        raise

    managed = sorted(prior_managed | set(newly_managed))
    initial_backup = prior_entry.get("initial_backup") or str(backup_path)
    manifest["harnesses"][harness] = {
        "original_path": str(target),
        "initial_backup": initial_backup,
        "latest_backup": str(backup_path),
        "managed_servers": managed,
        "updated_at": ts,
    }
    manifest["history"].append(
        {
            "action": "install",
            "harness": harness,
            "original_path": str(target),
            "backup_path": str(backup_path),
            "servers": list(servers.keys()),
            "timestamp": ts,
        }
    )

    return {
        "status": "installed",
        "path": str(target),
        "backup": str(backup_path),
        "servers": sorted(servers.keys()),
    }


def _install_codex_harness(
    *,
    target: Path,
    servers: dict[str, tuple[str, list[str]]],
    backup_dir: Path,
    manifest: dict[str, Any],
    ts: str,
    create_missing: bool,
) -> dict[str, Any]:
    if not target.exists():
        if not create_missing:
            return {"status": "skipped", "reason": "config_not_found", "path": str(target)}
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, "", mode=0o600)

    original_mode = stat.S_IMODE(target.stat().st_mode)
    raw_before = target.read_text(encoding="utf-8")
    parsed_before = tomllib.loads(raw_before) if raw_before.strip() else {}
    existing_mcp = parsed_before.get("mcp_servers") or {}
    if not isinstance(existing_mcp, dict):
        raise ValueError(f"Invalid [mcp_servers] table in {target}")

    backup_path = _create_backup(target, backup_dir, ts)

    prior_entry = manifest["harnesses"].get("codex", {})
    prior_managed = set(prior_entry.get("managed_servers") or [])
    updated_text = raw_before
    newly_managed: list[str] = []
    for name, (cmd, args) in servers.items():
        if name not in existing_mcp or name in prior_managed or name == "chittam":
            newly_managed.append(name)
            updated_text = _upsert_toml_mcp_server(updated_text, name, cmd, args)

    expected_present = {name: servers[name] for name in newly_managed}
    try:
        _atomic_write_text(target, updated_text, mode=original_mode)
        _validate_toml_servers(target, expected_present)
    except Exception:
        _atomic_write_text(target, backup_path.read_text(encoding="utf-8"), mode=original_mode)
        raise

    managed = sorted(prior_managed | set(newly_managed))
    initial_backup = prior_entry.get("initial_backup") or str(backup_path)
    manifest["harnesses"]["codex"] = {
        "original_path": str(target),
        "initial_backup": initial_backup,
        "latest_backup": str(backup_path),
        "managed_servers": managed,
        "updated_at": ts,
    }
    manifest["history"].append(
        {
            "action": "install",
            "harness": "codex",
            "original_path": str(target),
            "backup_path": str(backup_path),
            "servers": list(servers.keys()),
            "timestamp": ts,
        }
    )

    return {
        "status": "installed",
        "path": str(target),
        "backup": str(backup_path),
        "servers": sorted(servers.keys()),
    }


def install_harnesses(
    cfg: AppConfig,
    *,
    create_missing: bool = False,
) -> dict[str, Any]:
    """Safely register Chittam (and CodeGraph if present) across all 3 harness configs."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = cfg.backup_dir
    manifest_path = backup_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    servers = _desired_servers(cfg)

    ag_path = Path(cfg.harnesses.antigravity_mcp_path).expanduser()
    codex_path = Path(cfg.harnesses.codex_config_path).expanduser()
    claude_path = Path(cfg.harnesses.claude_config_path).expanduser()

    results = {
        "antigravity": _install_json_harness(
            harness="antigravity",
            target=ag_path,
            servers=servers,
            include_stdio_type=False,
            backup_dir=backup_dir,
            manifest=manifest,
            ts=ts,
            create_missing=create_missing,
        ),
        "codex": _install_codex_harness(
            target=codex_path,
            servers=servers,
            backup_dir=backup_dir,
            manifest=manifest,
            ts=ts,
            create_missing=create_missing,
        ),
        "claude_code": _install_json_harness(
            harness="claude_code",
            target=claude_path,
            servers=servers,
            include_stdio_type=True,
            backup_dir=backup_dir,
            manifest=manifest,
            ts=ts,
            create_missing=create_missing,
        ),
    }

    _save_manifest(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "servers": {k: {"command": v[0], "args": v[1]} for k, v in servers.items()},
        "harnesses": results,
    }


def _uninstall_json_harness(
    *,
    harness: str,
    target: Path,
    manifest: dict[str, Any],
    backup_dir: Path,
    ts: str,
    restore_backup: bool,
) -> dict[str, Any]:
    if not target.exists():
        return {"status": "skipped", "reason": "config_not_found", "path": str(target)}

    original_mode = stat.S_IMODE(target.stat().st_mode)
    entry = manifest.get("harnesses", {}).get(harness, {})
    backup_file = Path(entry["initial_backup"]) if entry.get("initial_backup") else None

    if restore_backup and backup_file and backup_file.exists():
        pre_restore_bak = _create_backup(target, backup_dir, f"pre_restore_{ts}")
        restored_text = backup_file.read_text(encoding="utf-8")
        try:
            json.loads(restored_text)
            _atomic_write_text(target, restored_text, mode=original_mode)
            json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            _atomic_write_text(target, pre_restore_bak.read_text(encoding="utf-8"), mode=original_mode)
            raise
        return {
            "status": "restored",
            "path": str(target),
            "restored_from": str(backup_file),
        }

    raw_before = target.read_text(encoding="utf-8")
    data = json.loads(raw_before) if raw_before.strip() else {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {target}")

    managed_servers = list(entry.get("managed_servers") or ["chittam"])
    mcp_servers = data.get("mcpServers")
    removed: list[str] = []
    if isinstance(mcp_servers, dict):
        for name in managed_servers:
            if name in mcp_servers:
                mcp_servers.pop(name, None)
                removed.append(name)

    if removed:
        bak = _create_backup(target, backup_dir, f"uninstall_{ts}")
        try:
            rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            _atomic_write_text(target, rendered, mode=original_mode)
            json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            _atomic_write_text(target, bak.read_text(encoding="utf-8"), mode=original_mode)
            raise

    manifest.get("harnesses", {}).pop(harness, None)
    manifest.setdefault("history", []).append(
        {
            "action": "uninstall",
            "harness": harness,
            "original_path": str(target),
            "removed": removed,
            "timestamp": ts,
        }
    )
    return {
        "status": "uninstalled",
        "path": str(target),
        "removed": removed,
    }


def _uninstall_codex_harness(
    *,
    target: Path,
    manifest: dict[str, Any],
    backup_dir: Path,
    ts: str,
    restore_backup: bool,
) -> dict[str, Any]:
    if not target.exists():
        return {"status": "skipped", "reason": "config_not_found", "path": str(target)}

    original_mode = stat.S_IMODE(target.stat().st_mode)
    entry = manifest.get("harnesses", {}).get("codex", {})
    backup_file = Path(entry["initial_backup"]) if entry.get("initial_backup") else None

    if restore_backup and backup_file and backup_file.exists():
        pre_restore_bak = _create_backup(target, backup_dir, f"pre_restore_{ts}")
        restored_text = backup_file.read_text(encoding="utf-8")
        try:
            tomllib.loads(restored_text)
            _atomic_write_text(target, restored_text, mode=original_mode)
            tomllib.loads(target.read_text(encoding="utf-8"))
        except Exception:
            _atomic_write_text(target, pre_restore_bak.read_text(encoding="utf-8"), mode=original_mode)
            raise
        return {
            "status": "restored",
            "path": str(target),
            "restored_from": str(backup_file),
        }

    raw_before = target.read_text(encoding="utf-8")
    parsed_before = tomllib.loads(raw_before) if raw_before.strip() else {}
    existing_mcp = parsed_before.get("mcp_servers") or {}
    managed_servers = list(entry.get("managed_servers") or ["chittam"])

    updated_text = raw_before
    removed: list[str] = []
    for name in managed_servers:
        if isinstance(existing_mcp, dict) and name in existing_mcp:
            updated_text = _remove_toml_mcp_server(updated_text, name)
            removed.append(name)

    if removed:
        bak = _create_backup(target, backup_dir, f"uninstall_{ts}")
        try:
            _atomic_write_text(target, updated_text, mode=original_mode)
            tomllib.loads(target.read_text(encoding="utf-8"))
        except Exception:
            _atomic_write_text(target, bak.read_text(encoding="utf-8"), mode=original_mode)
            raise

    manifest.get("harnesses", {}).pop("codex", None)
    manifest.setdefault("history", []).append(
        {
            "action": "uninstall",
            "harness": "codex",
            "original_path": str(target),
            "removed": removed,
            "timestamp": ts,
        }
    )
    return {
        "status": "uninstalled",
        "path": str(target),
        "removed": removed,
    }


def uninstall_harnesses(
    cfg: AppConfig,
    *,
    restore_backup: bool = False,
) -> dict[str, Any]:
    """Remove Chittam/CodeGraph registrations or restore original backups."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = cfg.backup_dir
    manifest_path = backup_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)

    ag_path = Path(cfg.harnesses.antigravity_mcp_path).expanduser()
    codex_path = Path(cfg.harnesses.codex_config_path).expanduser()
    claude_path = Path(cfg.harnesses.claude_config_path).expanduser()

    results = {
        "antigravity": _uninstall_json_harness(
            harness="antigravity",
            target=ag_path,
            manifest=manifest,
            backup_dir=backup_dir,
            ts=ts,
            restore_backup=restore_backup,
        ),
        "codex": _uninstall_codex_harness(
            target=codex_path,
            manifest=manifest,
            backup_dir=backup_dir,
            ts=ts,
            restore_backup=restore_backup,
        ),
        "claude_code": _uninstall_json_harness(
            harness="claude_code",
            target=claude_path,
            manifest=manifest,
            backup_dir=backup_dir,
            ts=ts,
            restore_backup=restore_backup,
        ),
    }

    if manifest_path.exists() or backup_dir.exists():
        _save_manifest(manifest_path, manifest)

    return {
        "manifest_path": str(manifest_path),
        "harnesses": results,
    }
