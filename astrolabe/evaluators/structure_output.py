"""Stable evaluator structures and legacy root-level projections."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote


def structure_directory(out_dir: Path, alias: str | None = None) -> Path:
    root = Path(out_dir) / "structures"
    directory = root / "evaluators" / ("alias-" + quote(str(alias), safe="")) if alias is not None else root
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def project_structure(path: Path, out_dir: Path, alias: str | None = None) -> Path:
    """Atomically replace the legacy root copy; never alias/link its contents."""
    path = Path(path)
    if alias is None:
        return path
    root = Path(out_dir) / "structures"
    root.mkdir(parents=True, exist_ok=True)
    target = root / path.name
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=root)
    try:
        with os.fdopen(fd, "wb") as dest, path.open("rb") as source:
            shutil.copyfileobj(source, dest)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return path
