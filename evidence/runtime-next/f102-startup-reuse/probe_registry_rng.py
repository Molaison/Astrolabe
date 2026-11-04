"""CPU-only registry/requirements/RNG inspection; no scientific run or PyRosetta init."""
from pathlib import Path
import builtins
import copy
import dataclasses
import hashlib
import inspect
import json
import os
import pickle
import random
import subprocess
import sys

repo, output = (Path(value).resolve() for value in sys.argv[1:3])
assert not output.exists(), output
sys.path.insert(0, str(repo))
os.chdir(repo)
import numpy as np
import torch

assert not torch.cuda.is_available(), "This local review must not use CUDA"
random.seed(4903)
np.random.seed(8911)
torch.manual_seed(10301)


def rng():
    return {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }


real_import, attempts = builtins.__import__, []


def observe_import(name, *args, **kwargs):
    if name == "pyrosetta" or name.startswith("pyrosetta."):
        attempts.append(name)
    return real_import(name, *args, **kwargs)


builtins.__import__ = observe_import
initial_rng = rng()
from astrolabe.app import _load_run_module
_load_run_module()
from astrolabe.losses.base import loss_registry
from astrolabe.losses import interaction
from astrolabe.runtime_requirements import _pyrosetta_loss_types_from_source, task_requirements
from omegaconf import OmegaConf

registry = {
    name: {"metadata": dataclasses.asdict(loss_registry.get_metadata(name)),
           "module": loss_registry.get_function(name).__module__,
           "signature": str(inspect.signature(loss_registry.get_function(name)))}
    for name in sorted(loss_registry.valid_losses)
}
assert len(registry) == 169, len(registry)
result = {
    "scope": "CPU-only metadata, requirement classification, missing dependency and RNG; zero model or scientific evaluations",
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "repo": str(repo), "registry": registry, "registry_count": len(registry),
    "rng_initial": initial_rng, "rng_after_app_load": rng(),
    "pyrosetta_attempts_after_app_load": list(attempts),
    "pyrosetta_loaded_after_app_load": "pyrosetta" in sys.modules,
    "spatial_loaded_after_app_load": "scipy.spatial" in sys.modules,
    "pyrosetta_loss_types": sorted(_pyrosetta_loss_types_from_source()),
    "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
}
# Inspect the actual requirements contract for every registered loss, without
# evaluating any loss or launching its evaluator/mutator.
template = OmegaConf.to_container(OmegaConf.load(repo / "benchmarks/runtime/workloads/cpu_scientific_control.yaml"), resolve=False)
requirements = {}
for name in sorted(registry):
    config = copy.deepcopy(template)
    config["loss"]["loss_terms"] = {"selected_loss": {"_type": name}}
    config["stages"] = {"default": copy.deepcopy(template["stages"]["default"])}
    config["stages"]["default"]["weighted_score_expression"] = "selected_loss"
    requirements[name] = task_requirements(config)
result["requirements_by_loss"] = requirements
result["rng_after_requirement_classification"] = rng()

# This environment intentionally has no installed proprietary PyRosetta. The
# native SIF path is a separate, not-yet-run independent review gate.
assert interaction.pyrosetta is None
failures = []
for _ in range(2):
    before = rng()
    try:
        interaction._require_pyrosetta()
    except ImportError as exc:
        failures.append({"type": type(exc).__name__, "message": str(exc),
                         "cause_type": type(exc.__cause__).__name__,
                         "cause_message": str(exc.__cause__),
                         "attempts": list(attempts)})
    else:
        raise AssertionError("Local missing dependency must fail")
    assert rng() == before, "Missing dependency altered an RNG stream"
result["missing_dependency"] = failures
result["rng_final"] = rng()
assert result["rng_final"] == initial_rng, "Inspection altered an RNG stream"
assert not result["spatial_loaded_after_app_load"]
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"head": result["head"], "registry": len(registry),
                  "requirements_cases": len(requirements), "rng_preserved": True,
                  "pyrosetta_attempts_after_app_load": result["pyrosetta_attempts_after_app_load"],
                  "scope": result["scope"]}))
