"""Configuration loading, compiled privacy exclusions, and secret redaction for Chittam."""

from __future__ import annotations

import fnmatch
import functools
import os
import re
from pathlib import Path
from typing import Any

import yaml

from chittam.models import AppConfig, PrivacyPolicy

DEFAULT_POLICY_FILE = Path(__file__).resolve().parents[2] / "config" / "policy.default.yaml"
USER_CONFIG_FILE = Path("~/.config/chittam/config.yaml").expanduser()

_WILDCARD_CHARS = frozenset("*?[")
_HARD_BLOCKS: tuple[tuple[str, str], ...] = tuple(
    (block, block.rstrip("/"))
    for block in ("/.ssh/", "/.aws/", "/.gnupg/", "/tax/", "/taxes/")
)

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


def _compile_combined_regex(patterns: list[str], flags: int = 0) -> re.Pattern[str] | None:
    if not patterns:
        return None
    parts = [f"(?:{fnmatch.translate(pat)})" for pat in patterns]
    return re.compile("|".join(parts), flags)


class CompiledPrivacyPolicy:
    """Pre-compiled, zero-allocation privacy policy matcher for fast directory walks."""

    def __init__(self, policy: PrivacyPolicy) -> None:
        self.excluded_dirs = frozenset(policy.exclude_dir_names)
        self.file_patterns: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
            (pat, re.compile(fnmatch.translate(pat), re.IGNORECASE))
            for pat in policy.exclude_file_patterns
        )
        self.file_prefilter = _compile_combined_regex(
            list(policy.exclude_file_patterns),
            re.IGNORECASE,
        )

        home_resolved = str(Path.home().resolve())
        exact_paths: dict[str, str] = {}
        prefix_list: list[tuple[str, str, str]] = []
        wildcard_list: list[tuple[re.Pattern[str], str]] = []
        wildcard_raw_globs: list[str] = []

        def _add_glob_variant(glob_pat: str, raw_glob: str, check_prefix: bool) -> None:
            if check_prefix and glob_pat.endswith("/**"):
                prefix = glob_pat[:-3]
                if not any(ch in prefix for ch in _WILDCARD_CHARS):
                    prefix_list.append((prefix, prefix + os.sep, raw_glob))
                    return
            if not any(ch in glob_pat for ch in _WILDCARD_CHARS):
                exact_paths.setdefault(glob_pat, raw_glob)
                return
            wildcard_list.append((re.compile(fnmatch.translate(glob_pat)), raw_glob))
            wildcard_raw_globs.append(glob_pat)

        for raw_glob in policy.exclude_path_globs:
            exp_glob = str(Path(raw_glob).expanduser())
            is_tilde = raw_glob.startswith("~/")
            _add_glob_variant(exp_glob, raw_glob, check_prefix=is_tilde or exp_glob.endswith("/**"))
            if is_tilde:
                alt_glob = os.path.join(home_resolved, raw_glob[2:])
                if alt_glob != exp_glob:
                    _add_glob_variant(alt_glob, raw_glob, check_prefix=True)

        self.exact_paths = exact_paths
        self.prefix_paths: tuple[tuple[str, str, str], ...] = tuple(prefix_list)
        self.wildcard_globs: tuple[tuple[re.Pattern[str], str], ...] = tuple(wildcard_list)
        self.wildcard_prefilter = _compile_combined_regex(wildcard_raw_globs)

    def check(
        self,
        path_str: str,
        *,
        raw_path_str: str | None = None,
        name: str | None = None,
        is_dir: bool = False,
    ) -> tuple[bool, str]:
        """Check whether an already-resolved path violates privacy, credential, or noise rules."""
        # 1. Directory component exclusions
        if self.excluded_dirs:
            dir_str = path_str if is_dir else path_str.rpartition(os.sep)[0]
            if dir_str:
                for part in dir_str.split(os.sep):
                    if part in self.excluded_dirs:
                        return True, f"excluded_dir:{part}"

        # 2. Filename / secret pattern exclusions
        if self.file_prefilter is not None:
            fname = name if name is not None else path_str.rpartition(os.sep)[2]
            if fname and self.file_prefilter.match(fname):
                for pat, rx in self.file_patterns:
                    if rx.match(fname):
                        return True, f"excluded_file_pattern:{pat}"

        # 3. Pre-expanded full path glob exclusions
        if path_str in self.exact_paths:
            return True, f"excluded_glob:{self.exact_paths[path_str]}"
        if raw_path_str and raw_path_str != path_str and raw_path_str in self.exact_paths:
            return True, f"excluded_glob:{self.exact_paths[raw_path_str]}"

        for prefix, prefix_sep, raw_glob in self.prefix_paths:
            if path_str == prefix or path_str.startswith(prefix_sep):
                return True, f"excluded_prefix:{raw_glob}"
            if raw_path_str and raw_path_str != path_str:
                if raw_path_str == prefix or raw_path_str.startswith(prefix_sep):
                    return True, f"excluded_glob:{raw_glob}"

        if self.wildcard_prefilter is not None:
            matched_resolved = bool(self.wildcard_prefilter.match(path_str))
            matched_raw = bool(
                raw_path_str and raw_path_str != path_str and self.wildcard_prefilter.match(raw_path_str)
            )
            if matched_resolved or matched_raw:
                for rx, raw_glob in self.wildcard_globs:
                    if (matched_resolved and rx.match(path_str)) or (
                        matched_raw and raw_path_str is not None and rx.match(raw_path_str)
                    ):
                        return True, f"excluded_glob:{raw_glob}"

        # 4. Hard-coded safety guardrails even if user edits config
        lowered = path_str.lower()
        for block, stripped in _HARD_BLOCKS:
            if block in lowered or lowered.endswith(stripped):
                return True, f"hard_privacy_guard:{block}"

        return False, ""


@functools.lru_cache(maxsize=64)
def _get_cached_privacy_policy(
    exclude_dirs: tuple[str, ...],
    exclude_files: tuple[str, ...],
    exclude_globs: tuple[str, ...],
    home_env: str,
) -> CompiledPrivacyPolicy:
    policy = PrivacyPolicy(
        exclude_dir_names=list(exclude_dirs),
        exclude_file_patterns=list(exclude_files),
        exclude_path_globs=list(exclude_globs),
    )
    return CompiledPrivacyPolicy(policy)


def compile_privacy_policy(policy: PrivacyPolicy) -> CompiledPrivacyPolicy:
    """Return a cached CompiledPrivacyPolicy keyed by exact rule tuples and $HOME."""
    return _get_cached_privacy_policy(
        tuple(policy.exclude_dir_names),
        tuple(policy.exclude_file_patterns),
        tuple(policy.exclude_path_globs),
        os.environ.get("HOME", ""),
    )


def is_path_excluded(
    path: Path | str,
    policy: PrivacyPolicy,
    *,
    already_resolved: bool = False,
    is_dir: bool | None = None,
) -> tuple[bool, str]:
    """Return (True, reason) if path violates privacy, credential, or noise rules."""
    matcher = compile_privacy_policy(policy)
    if already_resolved:
        path_str = os.fspath(path)
        raw_str = path_str
        resolved_is_dir = bool(is_dir)
    else:
        p = Path(path).expanduser()
        raw_str = str(p)
        try:
            resolved = p.resolve(strict=False)
        except OSError:
            resolved = p
        path_str = str(resolved)
        resolved_is_dir = resolved.is_dir() if is_dir is None else bool(is_dir)

    return matcher.check(path_str, raw_path_str=raw_str, is_dir=resolved_is_dir)


def redact_secrets(text: str) -> str:
    """Scrub API keys, bearer/GitHub tokens, and private keys before indexing."""
    out = text
    for pattern, repl in _SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out
