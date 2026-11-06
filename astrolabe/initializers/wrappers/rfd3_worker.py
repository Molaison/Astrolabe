#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


_DAEMON_FACTORY = "astrolabe.initializers.wrappers.rfd3_worker:daemon_factory"


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip() or "rfd3")


def _prepend_rfd3_import_roots(repo_path: str | Path) -> Path:
    script_dir = Path(__file__).resolve().parent
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != script_dir
    ]
    sys.modules.pop("rfd3", None)

    repo_root = Path(repo_path).expanduser().resolve()
    for import_root in (
        repo_root / "src",
        repo_root / "models" / "rfd3" / "src",
    ):
        import_root_str = str(import_root)
        if import_root.exists() and import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)
    return repo_root


def _load_engine(repo_path: str | Path, engine_config: dict[str, Any]) -> Any:
    _prepend_rfd3_import_roots(repo_path)
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    config = RFD3InferenceConfig(**dict(engine_config))
    return RFD3InferenceEngine(**config)


def _capture_rng_state() -> dict[str, Any]:
    import random

    import numpy as np

    import torch

    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return {
        "python_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state().clone(),
        "cuda_states": cuda_states,
    }


def _restore_rng_state(snapshot: dict[str, Any]) -> None:
    import random

    import numpy as np

    import torch

    random.setstate(snapshot["python_state"])
    np.random.set_state(snapshot["numpy_state"])
    torch.set_rng_state(snapshot["torch_state"].clone())
    cuda_states = snapshot.get("cuda_states")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.clone() for state in cuda_states])


def _write_engine_outputs(
    engine: Any,
    request: dict[str, Any],
    post_init_rng_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from biotite.structure.io.pdb import PDBFile

    if post_init_rng_state is not None:
        _restore_rng_state(post_init_rng_state)
    outputs_by_example = engine.run(
        inputs=str(Path(request["input_path"])),
        n_batches=int(request["n_batches"]),
        out_dir=None,
    )
    if not outputs_by_example:
        raise RuntimeError("RFD3 inference produced no outputs.")

    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_outputs = []
    for output_list in outputs_by_example.values():
        for output in output_list:
            example_id = str(output.example_id)
            base_path = output_dir / _safe_label(example_id)
            structure_path = base_path.with_suffix(".pdb")
            pdb_file = PDBFile()
            pdb_file.set_structure(output.atom_array)
            pdb_file.write(structure_path)

            metadata_path = None
            if output.metadata:
                metadata_path = base_path.with_suffix(".json")
                metadata_path.write_text(
                    json.dumps(output.metadata, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
            manifest_outputs.append(
                {
                    "example_id": example_id,
                    "structure_path": str(structure_path),
                    "metadata_path": str(metadata_path) if metadata_path else None,
                }
            )

    response = {"outputs": manifest_outputs}
    Path(request["response_path"]).write_text(
        json.dumps(response, indent=2) + "\n",
        encoding="utf-8",
    )
    return response


def run_once(request: dict[str, Any]) -> dict[str, Any]:
    repo_path = request.get("repo_path") or Path.cwd()
    engine = _load_engine(repo_path, dict(request["config"]))
    return _write_engine_outputs(
        engine,
        request,
        post_init_rng_state=_capture_rng_state(),
    )


def daemon_factory(settings: dict[str, Any]):
    repo_path = str(settings["repo_path"])
    engine_config = dict(settings["engine_config"])
    engine = _load_engine(repo_path, engine_config)
    post_init_rng_state = _capture_rng_state()

    def infer(request: dict[str, Any]) -> dict[str, Any]:
        if Path(str(request.get("repo_path", repo_path))).resolve() != Path(
            repo_path
        ).resolve():
            raise ValueError("RFD3 request repository does not match the loaded engine")
        if dict(request["config"]) != engine_config:
            raise ValueError(
                "RFD3 request engine settings do not match the loaded engine owner"
            )
        return _write_engine_outputs(
            engine,
            request,
            post_init_rng_state=post_init_rng_state,
        )

    return infer


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        raise SystemExit("usage: rfd3_worker.py REQUEST.json")

    request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    run_once(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
