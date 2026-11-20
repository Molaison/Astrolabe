"""Git provenance shared by the host launcher and the run lifecycle."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


GIT_PROVENANCE_ENV = "ASTROLABE_GIT_PROVENANCE"
_GIT_TIMEOUT_SECONDS = 30.0


def _git(root: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "--no-optional-locks", "-C", str(root), *args),
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=_GIT_TIMEOUT_SECONDS,
        check=False,
    )


def collect_checkout_git(root: Path) -> dict[str, Any]:
    """Read commit, branch and tracked changes from this checkout's own Git metadata."""
    record: dict[str, Any] = {
        "available": False,
        "root": str(root),
        "commit": None,
        "branch": None,
        "dirty": None,
        "dirty_entries": None,
        "dirty_scope": (
            "tracked modifications; untracked files and submodule worktrees excluded"
        ),
        "reason": None,
    }
    try:
        if not (root / ".git").exists():
            record["reason"] = "no .git in source root (installed or copied source tree)"
            return record
        if shutil.which("git") is None:
            record["reason"] = "git executable not found on PATH"
            return record
        # A caller's Git process can export paths into a different checkout.
        env = os.environ.copy()
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
            env.pop(key, None)
        commit = _git(root, env, "rev-parse", "HEAD")
        if commit.returncode != 0:
            record["reason"] = f"git rev-parse HEAD failed: {commit.stderr.strip()[:200]}"
            return record
        branch = _git(root, env, "rev-parse", "--abbrev-ref", "HEAD")
        dirty = _git(
            root, env, "status", "--porcelain", "--ignore-submodules=all", "--untracked-files=no",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        record["reason"] = f"git invocation failed: {exc}"
        return record

    record["available"] = True
    record["commit"] = commit.stdout.strip() or None
    if branch.returncode == 0 and branch.stdout.strip():
        record["branch"] = branch.stdout.strip()
    if dirty.returncode != 0:
        record["reason"] = f"git status failed: {dirty.stderr.strip()[:200]}"
    else:
        entries = [line for line in dirty.stdout.splitlines() if line.strip()]
        record["dirty"] = bool(entries)
        record["dirty_entries"] = len(entries)
    return record


def encode_host_git_provenance(host_root: Path, container_root: str) -> str:
    """Capture the mounted checkout before entering the container."""
    record = collect_checkout_git(host_root)
    if not record["available"]:
        return ""
    record.update(root=container_root, host_root=str(host_root), source="host_launch")
    # Apptainer uses commas to separate --env assignments.
    return base64.urlsafe_b64encode(json.dumps(record).encode()).decode("ascii")


def read_host_git_provenance(root: Path) -> dict[str, Any] | None:
    """Use launch metadata only for the checkout imported in the active runtime."""
    encoded = os.environ.get(GIT_PROVENANCE_ENV)
    if os.environ.get("ASTROLABE_RUNTIME_ACTIVE") != "1" or not encoded:
        return None
    try:
        record = json.loads(base64.b64decode(encoded, altchars=b"-_", validate=True))
        if (
            isinstance(record, dict)
            and record.get("available") is True
            and isinstance(record.get("commit"), str)
            and record["commit"]
            and Path(record["root"]).resolve() == root.resolve()
        ):
            record["root"] = str(root)
            return record
    except (KeyError, TypeError, ValueError, OSError, RuntimeError):
        pass
    return None
