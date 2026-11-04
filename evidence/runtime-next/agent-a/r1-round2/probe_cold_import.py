"""Read-only cold runner load. Never starts a trajectory or evaluates a model."""
import json
import sys
import time

repository = sys.argv[1]
sys.path.insert(0, repository)
started_wall, started_cpu = time.perf_counter(), time.process_time()
from astrolabe.app import _load_run_module
runner = _load_run_module()
result = {
    "repository": repository,
    "runner_source": runner.__file__,
    "python": sys.version,
    "scope": "one cold import only; no scientific workloads or GPU execution",
    "scientific_evaluation_count": 0,
    "load_wall_seconds": time.perf_counter() - started_wall,
    "load_process_cpu_seconds": time.process_time() - started_cpu,
    "loaded_packages": {name: name in sys.modules for name in
                        ("numpy", "scipy", "scipy.spatial", "jax", "jaxlib", "torch", "Bio", "pandas", "rdkit")},
}
print(json.dumps(result, indent=2))
