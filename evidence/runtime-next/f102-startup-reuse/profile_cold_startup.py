"""One bounded cold diagnostic; never runs a scientific workload or native init."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import sys
import time

# This diagnostic owns no child processes. Four invocations have at most 80 s
# aggregate Python CPU time; the parent reserves the remaining 10 s for wrappers.
resource.setrlimit(resource.RLIMIT_CPU, (19, 20))
resource.setrlimit(resource.RLIMIT_FSIZE, (4_000_000, 4_000_000))

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True, type=Path)
parser.add_argument("--variant", required=True, choices=("before", "candidate"))
args = parser.parse_args()
repo = args.repo.resolve()
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU must be explicitly hidden"
assert os.environ.get("NVIDIA_VISIBLE_DEVICES") in ("", "void", "none"), "GPU exposure forbidden"
assert os.environ.get("OMP_NUM_THREADS") == "1"
assert os.environ.get("MKL_NUM_THREADS") == "1"
assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"

expected = {
    "before": {
        "astrolabe/losses/interaction.py": "a1b3881bedc8370d56338004bc13cd5707d88750647855b50131bd554d57c0d8",
        "astrolabe/runtime_requirements.py": "ce388250da3f323b9181fb3f4ce517da54f6d2985373f9e6c4032715204a2c2e",
    },
    "candidate": {
        "astrolabe/losses/interaction.py": "a78914c2a6b5e410b9f07935ead280f15f9d714488c3655d8cd958a5febb322c",
        "astrolabe/runtime_requirements.py": "c60dcfa62fcd5ea39246bff5373ebc548bfc106ac9b12d2d242b8da676a6d469",
    },
}
actual = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in expected[args.variant]}
assert actual == expected[args.variant], "Profile source differs from the reviewed candidate"
sys.path.insert(0, str(repo))
os.chdir(repo)


def emit(phase, wall_start, cpu_start, **extra):
    print(json.dumps({"phase": phase, "variant": args.variant,
                      "wall_seconds": time.perf_counter() - wall_start,
                      "cpu_seconds": time.process_time() - cpu_start,
                      "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "pyrosetta_loaded": "pyrosetta" in sys.modules,
                      "spatial_loaded": "scipy.spatial" in sys.modules,
                      "scientific_evaluations": 0, **extra}), flush=True)


wall, cpu = time.perf_counter(), time.process_time()
from astrolabe.app import _load_run_module
_load_run_module()
emit("cold_load_run_module", wall, cpu)

# The real configuration persistence caller invokes this first scan once per
# cold process. Do not clear its cache or run repeated hidden warmup scans.
from astrolabe.runtime_requirements import _pyrosetta_loss_types_from_source
assert _pyrosetta_loss_types_from_source.cache_info().currsize == 0
wall, cpu = time.perf_counter(), time.process_time()
loss_types = _pyrosetta_loss_types_from_source()
emit("first_requirement_scan", wall, cpu, loss_types=sorted(loss_types))
from astrolabe.losses.base import loss_registry
assert len(loss_registry.valid_losses) == 169
print(json.dumps({"status": "DIAGNOSTIC_COMPLETE", "variant": args.variant,
                  "python": sys.version, "registry_count": len(loss_registry.valid_losses),
                  "source_sha256": actual, "scientific_evaluations": 0,
                  "native_initialization": False, "forward_or_sampling": False,
                  "scope": "Cold imports and first source scan only; not an E2E throughput result"}), flush=True)
