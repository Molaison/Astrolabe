#!/usr/bin/env python3
"""Execute a model-free native CLI workflow inside a candidate runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true", help="Use the installed package and its bundled configuration.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "ASTROLABE_RUNTIME_ACTIVE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "GH_CUDA_PREINITIALIZE": "0",
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for name in ("ASTROLABE_RESOURCE_SOCKET", "ASTROLABE_RESIDENT_MODEL_REGISTRY"):
        env.pop(name, None)

    with tempfile.TemporaryDirectory(prefix="astrolabe-runtime-") as directory:
        work = Path(directory)
        output = work / "output"
        config_args = [] if args.installed else ["--config-dir", str(root / "config")]
        if args.installed:
            env.pop("PYTHONPATH", None)
        subprocess.run(
            [
                sys.executable, "-m", "astrolabe", *config_args,
                "run", "--task", "beam_refinement", "--no-chdir", "env=env",
                "evaluator.evaluator_terms.ESMFold._type=dummy",
                "execution.devices=[]", "execution.cpu_slots=2",
                "execution.ram_bytes=2147483648", "execution.max_pending_bytes=67108864",
                "search.execution.model_sharing=replicated", f"out_dir={output}",
                f"raw_model_results_dir={work / 'raw'}", "obabel_path=null",
                "ananas_path=null", "dssp_path=null", "usalign_path=null",
            ],
            cwd=work if args.installed else root, env=env, check=True, timeout=120,
        )
        summaries = list(output.rglob("03_final_beam.json"))
        if len(summaries) != 1:
            raise RuntimeError(f"Expected one native run summary, found {len(summaries)}")
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        if summary["search"]["evaluations"] != 3 or len(summary["evaluations"]) != 3:
            raise RuntimeError("Native run did not complete the three planned evaluations")
    print("Native CPU workflow completed three evaluations.")


if __name__ == "__main__":
    main()
