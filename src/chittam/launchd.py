"""macOS launchd background sync management for Chittam."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from chittam.models import AppConfig

PLIST_LABEL = "com.vamfi.chittam.sync"
DEFAULT_PLIST_PATH = Path(f"~/Library/LaunchAgents/{PLIST_LABEL}.plist")
DEFAULT_START_INTERVAL = 1800
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_chittam_cli_command() -> str:
    """Resolve the absolute path to the `chittam` CLI binary."""
    candidates = [
        _REPO_ROOT / ".venv" / "bin" / "chittam",
        Path(sys.executable).resolve().parent / "chittam",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    which_bin = shutil.which("chittam")
    if which_bin:
        return str(Path(which_bin).expanduser())
    return str(_REPO_ROOT / ".venv" / "bin" / "chittam")


def build_launchd_plist(
    cfg: AppConfig,
    *,
    start_interval: int = DEFAULT_START_INTERVAL,
    command: str | None = None,
) -> dict[str, Any]:
    """Construct the launchd plist dictionary for periodic Chittam sync."""
    cli_bin = command or resolve_chittam_cli_command()
    data_dir = Path(cfg.data_dir).expanduser()
    log_path = data_dir / "sync.log"
    bin_dir = str(Path(cli_bin).parent)
    env_path = f"{bin_dir}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

    return {
        "Label": PLIST_LABEL,
        "ProgramArguments": [cli_bin, "sync", "--quiet"],
        "StartInterval": int(start_interval),
        "RunAtLoad": False,
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
        "EnvironmentVariables": {
            "PATH": env_path,
        },
        "ProcessType": "Background",
    }


def _validate_plist_file(plist_path: Path) -> dict[str, Any]:
    """Validate XML plist via stdlib plistlib and macOS `plutil -lint`."""
    raw = plist_path.read_bytes()
    parsed = plistlib.loads(raw)
    if not isinstance(parsed, dict) or parsed.get("Label") != PLIST_LABEL:
        raise ValueError(f"Invalid plist payload in {plist_path}")

    plutil = shutil.which("plutil") or "/usr/bin/plutil"
    if Path(plutil).exists():
        proc = subprocess.run(
            [plutil, "-lint", str(plist_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plutil -lint failed for {plist_path}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
    return parsed


def _bootout_quietly(plist_path: Path) -> None:
    """Unload any existing launchd job without failing if not currently loaded."""
    launchctl = shutil.which("launchctl") or "/bin/launchctl"
    if not Path(launchctl).exists():
        return
    uid = os.getuid()
    subprocess.run(
        [launchctl, "bootout", f"gui/{uid}/{PLIST_LABEL}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if plist_path.exists():
        subprocess.run(
            [launchctl, "unload", "-w", str(plist_path)],
            capture_output=True,
            text=True,
            check=False,
        )


def install_launchd(
    cfg: AppConfig,
    *,
    plist_path: Path | None = None,
    load: bool = True,
    start_interval: int = DEFAULT_START_INTERVAL,
) -> Path:
    """Write, validate, and optionally load the Chittam periodic sync LaunchAgent."""
    target = (plist_path or DEFAULT_PLIST_PATH).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.data_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    payload = build_launchd_plist(cfg, start_interval=start_interval)
    xml_bytes = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=False)

    prior_bytes = target.read_bytes() if target.exists() else None
    tmp_path = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_bytes(xml_bytes)
        os.replace(tmp_path, target)
        _validate_plist_file(target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if prior_bytes is not None:
            target.write_bytes(prior_bytes)
        else:
            target.unlink(missing_ok=True)
        raise

    if load:
        launchctl = shutil.which("launchctl") or "/bin/launchctl"
        if not Path(launchctl).exists():
            raise RuntimeError("launchctl binary not found; cannot load LaunchAgent")
        uid = os.getuid()
        _bootout_quietly(target)
        boot = subprocess.run(
            [launchctl, "bootstrap", f"gui/{uid}", str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if boot.returncode != 0:
            legacy = subprocess.run(
                [launchctl, "load", "-w", str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            if legacy.returncode != 0:
                raise RuntimeError(
                    "Failed to load launchd agent via bootstrap or load: "
                    f"{boot.stderr.strip()} | {legacy.stderr.strip()}"
                )

    return target


def uninstall_launchd(
    *,
    plist_path: Path | None = None,
    unload: bool = True,
) -> bool:
    """Unload and remove the Chittam LaunchAgent plist if present."""
    target = (plist_path or DEFAULT_PLIST_PATH).expanduser()
    existed = target.exists()
    if unload:
        _bootout_quietly(target)
    if existed:
        target.unlink(missing_ok=True)
    return existed


def launchd_status(*, plist_path: Path | None = None) -> dict[str, Any]:
    """Inspect installation and runtime status of the Chittam LaunchAgent."""
    target = (plist_path or DEFAULT_PLIST_PATH).expanduser()
    installed = target.exists()
    start_interval: int | None = None
    program_arguments: list[str] = []
    log_path: str | None = None

    if installed:
        try:
            parsed = plistlib.loads(target.read_bytes())
            if isinstance(parsed, dict):
                start_interval = parsed.get("StartInterval")
                program_arguments = list(parsed.get("ProgramArguments") or [])
                log_path = parsed.get("StandardOutPath")
        except Exception:
            pass

    loaded = False
    pid: int | None = None
    last_exit_status: int | None = None
    launchctl = shutil.which("launchctl") or "/bin/launchctl"
    if Path(launchctl).exists():
        uid = os.getuid()
        res = subprocess.run(
            [launchctl, "print", f"gui/{uid}/{PLIST_LABEL}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            loaded = True
            for line in res.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("pid = "):
                    try:
                        pid = int(stripped.split("=", 1)[1].strip())
                    except ValueError:
                        pass
                elif stripped.startswith("last exit code = "):
                    raw_code = stripped.split("=", 1)[1].strip()
                    try:
                        last_exit_status = int(raw_code)
                    except ValueError:
                        pass
        else:
            lst = subprocess.run(
                [launchctl, "list", PLIST_LABEL],
                capture_output=True,
                text=True,
                check=False,
            )
            if lst.returncode == 0:
                loaded = True

    return {
        "label": PLIST_LABEL,
        "plist_path": str(target),
        "installed": installed,
        "loaded": loaded,
        "pid": pid,
        "last_exit_status": last_exit_status,
        "start_interval": start_interval,
        "program_arguments": program_arguments,
        "log_path": log_path,
    }
