#!/usr/bin/env python3
"""Compare complete independent native control outputs; no benchmark statistics."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    b, c = [json.loads(p.read_text()) for p in (args.baseline, args.candidate)]
    fields = ["protocol_sha256", "probe_sha256", "dependencies", "native_version", "registry_order", "registry",
        "requirements", "pyrosetta_loss_classification", "native_symbols", "geometry_before", "geometry_after",
        "metric_values", "registered_loss_values", "initialized_before_explicit_init", "initialized_after_explicit_init", "cuda_initialized"]
    checks = {field + "_exact": field in b and field in c and b[field] == c[field] for field in fields}
    checks["baseline_pass"] = b.get("status") == "PASS"
    checks["candidate_pass"] = c.get("status") == "PASS"
    for label in ("production_seeded", "after_registry_and_requirements", "after_first_require", "after_native_init", "after_pose_build", "after_native_losses"):
        checks["rng_" + label + "_exact"] = label in b.get("rng", {}) and label in c.get("rng", {}) and b["rng"][label] == c["rng"][label]
    checks["expected_baseline_head"] = b.get("source", {}).get("head") == b.get("protocol", {}).get("baseline_commit")
    checks["expected_candidate_head"] = c.get("source", {}).get("head") == c.get("protocol", {}).get("candidate_commit")
    checks["candidate_cold_pyrosetta_deferred"] = c.get("cold_modules", {}).get("pyrosetta") is False
    checks["candidate_cold_spatial_deferred"] = c.get("cold_modules", {}).get("scipy.spatial") is False
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "scope": "native CPU scientific correctness only",
        "checks": checks, "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.baseline, args.candidate)},
        "registry_count": [len(x.get("registry", {})) for x in (b, c)],
        "losses": {"baseline": b.get("registered_loss_values"), "candidate": c.get("registered_loss_values")},
        "cpu_seconds": [x.get("observed_cpu_seconds") for x in (b, c)],
        "cold_import_rng_diagnostic_equal": b.get("rng", {}).get("after_cold_import_diagnostic") == c.get("rng", {}).get("after_cold_import_diagnostic")}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
