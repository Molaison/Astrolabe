"""Content proofs and manifest-last publication for rebuildable posthoc outputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


def content_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(paths: list[Path], context: Any) -> str:
    entries = []

    def visit(path: Path, ancestors: frozenset[Path]) -> None:
        if path.is_dir():
            resolved = path.resolve()
            if resolved in ancestors:
                raise ValueError(f"Cyclic posthoc input directory: {path}")
            entries.append((str(path), "directory"))
            for child in sorted(path.iterdir()):
                visit(child, ancestors | {resolved})
        else:
            entries.append((str(path), content_digest(path) if path.is_file() else "missing"))

    for root in sorted(set(paths)):
        visit(root, frozenset())
    data = json.dumps([context, entries], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".posthoc-manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Publication:
    """Stage all files; a manifest, never file existence, defines the output set.

    Before promotion, failure leaves the old complete bundle untouched. During
    multi-file promotion a PARTIAL manifest fences a mixed set, including after
    process death. Only the final manifest replacement announces completion.
    """

    def __init__(self, manifest: Path):
        self.manifest = manifest
        manifest.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix=".posthoc-stage-", dir=manifest.parent))
        self.files: dict[Path, Path] = {}
        self.promoting = False

    def path(self, final: Path) -> Path:
        # Replace an output symlink itself, never its possibly original input.
        final = final.expanduser().absolute()
        if final not in self.files:
            self.files[final] = self.stage / f"{len(self.files)}{final.suffix}"
        return self.files[final]

    def publish(self, *, status: str = "COMPLETE", **metadata: Any) -> None:
        outputs = [{"path": str(final), "sha256": content_digest(staged)}
                   for final, staged in self.files.items()]
        payload = {"version": 1, **metadata, "status": "PARTIAL", "outputs": outputs}
        _write_manifest(self.manifest, payload)
        self.promoting = True
        for final, staged in self.files.items():
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
        payload["status"] = status
        _write_manifest(self.manifest, payload)

    def failed(self, exc: BaseException) -> None:
        # An old manifest remains authoritative only while its files are intact.
        intact = False
        if self.manifest.is_file() and not self.promoting:
            try:
                prior = json.loads(self.manifest.read_text())
                intact = (prior.get("status") == "COMPLETE"
                          and all(content_digest(Path(item["path"])) == item["sha256"]
                                  for item in prior["outputs"]))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if not intact:
            _write_manifest(self.manifest, {
                "version": 1, "status": "PARTIAL", "outputs": [],
                "error": f"{type(exc).__name__}: {exc}",
            })

    def close(self) -> None:
        shutil.rmtree(self.stage, ignore_errors=True)
