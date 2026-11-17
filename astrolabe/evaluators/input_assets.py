"""Stable, atomically written assets shared by native preparation requests."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile


def write_text_if_changed(path: Path, text: str) -> Path:
    """Leave unchanged input files untouched, including their modification time."""
    path = Path(path)
    payload = text.encode("utf-8")
    try:
        if path.read_bytes() == payload:
            return path
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def materialize_msa(directory: Path, text: str) -> Path:
    """Keep previous job receipts valid while sharing identical MSA contents."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return write_text_if_changed(Path(directory) / f"{digest}.a3m", text)
