#!/usr/bin/env bash
# Root runs this only in a reviewed, capable CPU environment. No GPU/model benchmark.
set -euo pipefail
: "${ASTROLABE_TEST_PYTHON:?set an absolute path to the reviewed CPU Python interpreter}"
: "${ASTROLABE_TEST_OUT:?set a new absolute evidence directory}"
case "$ASTROLABE_TEST_PYTHON" in /*) ;; *) echo "Python path must be absolute" >&2; exit 2;; esac
case "$ASTROLABE_TEST_OUT" in /*) ;; *) echo "Evidence path must be absolute" >&2; exit 2;; esac
ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
mkdir "$ASTROLABE_TEST_OUT"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void HIP_VISIBLE_DEVICES=''
export ROCR_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu GH_CUDA_PREINITIALIZE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 ASTROLABE_EXPECT_GUARDRAILS=1
git rev-parse HEAD > "$ASTROLABE_TEST_OUT/source-head.txt"
"$ASTROLABE_TEST_PYTHON" -m pip freeze --all > "$ASTROLABE_TEST_OUT/pip-freeze.txt"
"$ASTROLABE_TEST_PYTHON" benchmarks/runtime/validate.py preflight --out "$ASTROLABE_TEST_OUT/preflight"
"$ASTROLABE_TEST_PYTHON" - <<'PY' > "$ASTROLABE_TEST_OUT/cpu-capacity.json"
import json
from astrolabe.runtime.admission import available_cpus
cpus = available_cpus()
print(json.dumps({"available_cpu_slots": cpus, "required": 2}))
assert cpus >= 2, "Saturation overlap contract needs at least two allocated/effective CPU slots"
PY
# Preserve the existing complete regression entry point unchanged.
timeout 600 "$ASTROLABE_TEST_PYTHON" benchmarks/runtime/validate.py regressions --out "$ASTROLABE_TEST_OUT/regressions" > "$ASTROLABE_TEST_OUT/regression-entry.log" 2>&1
# New recovered contracts are not silently substituted into the frozen measurement code.
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" timeout 300 "$ASTROLABE_TEST_PYTHON" -m pytest \
  tests/test_runtime_ingress.py tests/test_runtime_request_isolation.py tests/test_runtime_saturation.py \
  -q -ra --junitxml="$ASTROLABE_TEST_OUT/recovered-contracts.xml" \
  --basetemp="$ASTROLABE_TEST_OUT/pytest-tmp" > "$ASTROLABE_TEST_OUT/recovered-contracts.log" 2>&1
"$ASTROLABE_TEST_PYTHON" - "$ASTROLABE_TEST_OUT/recovered-contracts.xml" <<'PY' > "$ASTROLABE_TEST_OUT/recovered-contracts-gate.json"
import json, sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
cases = list(root.iter("testcase"))
result = {"tests": len(cases), "failures": sum(c.find("failure") is not None for c in cases),
          "errors": sum(c.find("error") is not None for c in cases),
          "skipped": sum(c.find("skipped") is not None for c in cases)}
print(json.dumps(result))
assert result["tests"] > 0 and not any(result[k] for k in ("failures", "errors", "skipped")), result
PY
