"""Real CLI and spawn-worker integration; Dummy produces no model prediction."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("sharing", ["shared", "replicated"])
def test_isolated_cli_completes_and_resumes_without_new_evaluations(tmp_path, sharing):
    pytest.importorskip("torch", reason="execution suite requires the scientific runtime")
    output = tmp_path / "output"
    command = [
        str(ROOT / "scripts/astrolabe"), "--config-dir", str(ROOT / "config"),
        "run", "--task", "beam_refinement", "--no-chdir", "env=env",
        "evaluator.evaluator_terms.ESMFold._type=dummy",
        "execution.devices=[]", "execution.cpu_slots=2",
        "execution.ram_bytes=2147483648", "execution.max_pending_bytes=67108864",
        f"search.execution.model_sharing={sharing}",
        f"out_dir={output}", f"raw_model_results_dir={tmp_path / 'raw'}",
        "task_name=integrated_cli", "obabel_path=null",
        "ananas_path=null", "dssp_path=null", "usalign_path=null",
    ]
    environment = {
        **os.environ, "ASTROLABE_RUNTIME_ACTIVE": "1", "ASTROLABE_PY": sys.executable,
        "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
        "GH_CUDA_PREINITIALIZE": "0", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    for name in ("ASTROLABE_RESOURCE_SOCKET", "ASTROLABE_RESIDENT_MODEL_REGISTRY"):
        environment.pop(name, None)

    def invoke(arguments):
        result = subprocess.run(
            arguments, cwd=ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
        )
        assert result.returncode == 0, result.stdout

    invoke([*command[:7], "--dry-run", *command[7:]])
    assert not output.exists()
    invoke(command)
    summaries = list(output.rglob("03_final_beam.json"))
    assert len(summaries) == 1
    summary_path = summaries[0]
    summary = json.loads(summary_path.read_text())
    assert summary["search"]["evaluations"] == 3
    assert len(summary["evaluations"]) == 3
    run_root = summary_path.parent
    events = (run_root / "01_evaluations.jsonl").read_bytes()
    receipts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (run_root / ".prediction_workers/receipts").glob("*.json")
    }
    assert len([name for name in receipts if name.endswith(".score.json")]) == 3
    resume = [f"out_dir={run_root}" if argument.startswith("out_dir=") else argument
              for argument in command]
    invoke([*resume, "search.resume=true"])
    resumed_events = (run_root / "01_evaluations.jsonl").read_bytes()
    assert resumed_events.startswith(events)
    # Shared owners publish a fresh endpoint on resume. That operational event
    # is allowed; scientific events, evaluations and receipts remain unchanged.
    appended = [json.loads(line) for line in resumed_events[len(events):].splitlines()]
    assert all(row["event"] == "resident_models_shared" for row in appended)
    assert len(appended) <= (1 if sharing == "shared" else 0)
    assert json.loads(summary_path.read_text())["evaluations"] == summary["evaluations"]
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (run_root / ".prediction_workers/receipts").glob("*.json")
    } == receipts
