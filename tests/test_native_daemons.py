from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import sys
from textwrap import dedent

import pytest

from astrolabe.mutators.native_daemons import NativeSequenceRunner


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepend_pythonpath(monkeypatch, path: Path) -> None:
    existing = os.environ.get("PYTHONPATH")
    value = str(path) if not existing else f"{path}{os.pathsep}{existing}"
    monkeypatch.setenv("PYTHONPATH", value)


def _write_factory(root: Path, *, device_metadata: str) -> None:
    root.mkdir()
    (root / "device_contract_factory.py").write_text(
        dedent(
            f"""
            from pathlib import Path
            from astrolabe.model_execution import kernel_handler


            def loader(settings):
                output = Path(settings["runner_config"]["out_folder"])
                metadata = {{"output_dir": str(output)}}
                device = {device_metadata!r}
                if device is not None:
                    metadata["device"] = device
                return lambda payload: {{"ok": True}}, metadata


            def factory(settings):
                return kernel_handler(settings, loader)
            """
        ).lstrip(),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("device_metadata", "message"),
    [
        (None, "must report its actual execution device"),
        ("", "must report its actual execution device"),
    ],
)
def test_native_sequence_runner_rejects_missing_device_metadata(
    tmp_path: Path,
    monkeypatch,
    device_metadata: str | None,
    message: str,
) -> None:
    _write_factory(tmp_path / "factory", device_metadata=device_metadata)
    _prepend_pythonpath(monkeypatch, tmp_path / "factory")
    with pytest.raises(ValueError, match=message):
        NativeSequenceRunner(
            backend="device-contract-test",
            factory="device_contract_factory:factory",
            runner_config={"out_folder": str(tmp_path / "output")},
        )


def test_real_cpu_ligandmpnn_worker_init_reports_cpu_device(
    tmp_path: Path,
) -> None:
    case = Path(
        "/case/config/001_binder_interface/01_astrolabe_beam/"
        "12_config_hierarchy/03_D5056_D15.yaml"
    )
    ligandmpnn_root = Path("/models/ligandmpnn")
    af3_root = Path("/models/alphafold3")
    if not case.is_file() or not ligandmpnn_root.is_dir() or not af3_root.is_dir():
        pytest.skip("licensed real model fixtures are unavailable")
    output = tmp_path / "worker"
    result = tmp_path / "worker-result.json"
    code = dedent(
        """
        import json
        import os
        from pathlib import Path
        import sys

        from omegaconf import OmegaConf
        import astrolabe.protein_beam as beam
        from astrolabe.protein_beam import load_config, worker_init

        case = Path(sys.argv[1])
        output = Path(sys.argv[2])
        result = Path(sys.argv[3])
        output.mkdir(parents=True)
        cfg = load_config(case)
        worker_init(
            OmegaConf.to_yaml(cfg, resolve=False),
            str(output),
            str(output / "unused.sock"),
        )
        mutator = beam._RUNTIME[2]
        try:
            result.write_text(json.dumps({
                "runner_type": type(mutator.runner).__name__,
                "device": str(mutator.runner.device),
                "pid": mutator.runner.pid,
                "worker_parent_pid": os.getpid(),
            }))
        finally:
            mutator.shutdown()
        """
    ).lstrip()
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing
        else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        JAX_PLATFORMS="cpu",
        GH_CUDA_PREINITIALIZE="0",
        GH_WEIGHT_AF3=str(af3_root),
        GH_WEIGHT_LIGANDMPNN=str(ligandmpnn_root),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(case), str(output), str(result)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(result.read_text())
    assert evidence["runner_type"] == "NativeSequenceRunner"
    assert evidence["device"] == "cpu"
    assert evidence["pid"] != evidence["worker_parent_pid"]
