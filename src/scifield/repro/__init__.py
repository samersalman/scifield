"""Reproducibility utilities — capture provenance sidecars for output artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scifield import __version__


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def record_run(
    artifact_path: Path,
    inputs: dict[str, Path],
    config: dict[str, Any],
) -> Path:
    """Write a sidecar JSON next to artifact_path; return the sidecar path."""
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    input_hashes = {name: _hash_file(Path(path)) for name, path in inputs.items()}
    payload = {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "config_hash": config_hash,
        "config": config,
        "input_hashes": input_hashes,
        "software_versions": {
            "python": platform.python_version(),
            "scifield": __version__,
            "platform": platform.platform(),
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }
    sidecar_path = Path(str(artifact_path) + ".run.json")
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return sidecar_path
