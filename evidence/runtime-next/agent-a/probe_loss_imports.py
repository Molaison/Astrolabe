"""Inventory imports/registry around a real direct W0 run; no timing claim."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
repo, out = args.repo.resolve(), args.out.resolve()
if out.exists():
    raise SystemExit("refusing to overwrite " + str(out))
out.mkdir(parents=True)
sys.path.insert(0, str(repo))
os.chdir(repo / "scripts/validation/fixtures")
os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void", JAX_PLATFORMS="cpu",
                  OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
os.environ["ASTROLABE_RUNTIME_TRACE"] = str(out / "runtime.jsonl")
from astrolabe.app import Astrolabe, _load_run_module
from scripts.validation import runtime_workflow_benchmark as benchmark
from omegaconf import OmegaConf

_load_run_module()
from astrolabe.losses.base import loss_registry
registry = {name: asdict(loss_registry.get_metadata(name)) for name in sorted(loss_registry.valid_losses)}
before = "scipy.spatial" in sys.modules
base = benchmark.compose(repo / "scripts/validation/fixtures/runtime_cpu.yaml", repo / "config")
config = benchmark.mode_config(base, "direct", out)
OmegaConf.save(OmegaConf.create(config), out / "config.yaml")
Astrolabe(OmegaConf.create(config)).run()
identities = {str((row["repeat_index"], row["seed"])) for row in benchmark.planned_trajectories(base)}
rows, failures = benchmark.read_completed(out, identities)
result = {"scope": "real local direct W0 import/registry/scientific probe; no performance or resident certification",
          "repository": str(repo),
          "head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
          "sources": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in
                      ("astrolabe/losses/interaction.py", "astrolabe/losses/repeat_symmetry.py")},
          "protocol_sha256": benchmark.fingerprint(benchmark.scientific_projection(base)),
          "spatial_after_runner_import": before, "spatial_after_full_direct_run": "scipy.spatial" in sys.modules,
          "registered_losses": registry, "scientific_rows": rows, "failures": failures}
(out / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({key: value for key, value in result.items() if key not in ("registered_losses", "scientific_rows")}, indent=2))
