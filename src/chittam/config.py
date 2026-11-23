"""Configuration loading, strict privacy exclusions, and secret redaction for Chittam."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

import yaml

from chittam.models import AppConfig, PrivacyPolicy

DEFAULT_POLICY_FILE = Path(__file__).resolve().parents[2] / "config" / "policy.default.yaml"
USER_CONFIG_FILE = Path("~/.config/chittam/config.yaml").expanduser()

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(gh[pousr]_[A-Za-z0-9_]{16,})"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"(sk-[A-Za-z0-9\-_]{16,})"), "[REDACTED_API_KEY]"),
    (re.compile(r"(AIza[0-9A-Za-z\-_]{24,})"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"(AKIA[0-9A-Z]{16})"), "[REDACTED_AWS_KEY]"),
    (
        re.compile(
            r"(-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----)"
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(
            r'(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password)\b\s*[:=]\s*["\']?([A-Za-z0-9_\-\.=+/]{12,})["\']?'
        ),
        r"\1=[REDACTED]",
    ),
]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load Chittam configuration from user config merged over defaults."""
    raw: dict[str, Any] = {}
    if DEFAULT_POLICY_FILE.exists():
        raw = yaml.safe_load(DEFAULT_POLICY_FILE.read_text(encoding="utf-8")) or {}

    target = config_path or Path(os.environ.get("CHITTAM_CONFIG", USER_CONFIG_FILE)).expanduser()
    if target.exists():
        user_raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, user_raw)

    return AppConfig.model_validate(raw)


def save_default_user_config(config: AppConfig | None = None, config_path: Path | None = None) -> Path:
    """Write user config if missing, returning the target path."""
    cfg = config or load_config()
    target = config_path or USER_CONFIG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        data = cfg.model_dump(mode="json")
        target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def is_path_excluded(path: Path | str, policy: PrivacyPolicy) -> tuple[bool, str]:
    """Return (True, reason) if path violates privacy, credential, or noise rules."""
    p = Path(path).expanduser()
    try:
        resolved = p.resolve(strict=False)
    except OSError:
        resolved = p

    path_str = str(resolved)
    home = str(Path.home().resolve())
    parts = resolved.parts

    # 1. Directory component exclusions
    excluded_dirs = set(policy.exclude_dir_names)
    for part in parts[:-1] if not resolved.is_dir() else parts:
        if part in excluded_dirs:
            return True, f"excluded_dir:{part}"

    # 2. Filename / secret pattern exclusions
    name = resolved.name
    for pat in policy.exclude_file_patterns:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(name.lower(), pat.lower()):
            return True, f"excluded_file_pattern:{pat}"

    # 3. Full path glob exclusions (normalized relative to ~ as well as absolute)
    for raw_glob in policy.exclude_path_globs:
        exp_glob = str(Path(raw_glob).expanduser())
        # Normalize without resolving symlink if non-existent, and with resolve
        if fnmatch.fnmatch(path_str, exp_glob) or fnmatch.fnmatch(str(p), exp_glob):
            return True, f"excluded_glob:{raw_glob}"
        # Handle ~ vs /Users/<user> consistently
        if raw_glob.startswith("~/"):
            alt_glob = os.path.join(home, raw_glob[2:])
            if fnmatch.fnmatch(path_str, alt_glob) or fnmatch.fnmatch(str(p), alt_glob):
                return True, f"excluded_glob:{raw_glob}"
            if alt_glob.endswith("/**"):
                prefix = alt_glob[:-3]
                if path_str == prefix or path_str.startswith(prefix + os.sep):
                    return True, f"excluded_prefix:{raw_glob}"
            if path_str == alt_glob:
                return True, f"excluded_exact:{raw_glob}"

    # 4. Extra hard-coded safety guardrails for sensitive paths even if user edits config
    lowered = path_str.lower()
    hard_blocks = ("/.ssh/", "/.aws/", "/.gnupg/", "/tax/", "/taxes/")
    for block in hard_blocks:
        if block in lowered or lowered.endswith(block.rstrip("/")):
            return True, f"hard_privacy_guard:{block}"

    return False, ""


def redact_secrets(text: str) -> str:
    """Scrub API keys, bearer/GitHub tokens, and private keys before indexing."""
    out = text
    for pattern, repl in _SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out
