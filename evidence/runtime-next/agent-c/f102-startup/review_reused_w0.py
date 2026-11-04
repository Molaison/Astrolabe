#!/usr/bin/env python3
"""Reuse the fixed independent W0 reader against native-host c90 versus 9d raw.

Only expected HEADs and the raw directory layout are changed in memory.
No benchmark, production file, workload or original raw artifact is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types


READER_SHA256 = "eda968172261e7b266278a4616c65d17b8b70582f3e670b41b7249dff45f5a3a"
HEADS = {
    "baseline": "9d62b3eb86cb75a2266af49dc48cb851e69e0341",
    "candidate": "c90fadfdb82e2ba81b596b8d1cc93c4a25e783c4",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reader", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--prior", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    original = a.reader.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    assert digest == READER_SHA256, "independent reader bytes changed"
    old, new = 'comparison = root / "comparison"', 'comparison = root'
    source = original.decode()
    assert source.count(old) == 1
    source = source.replace(old, new)
    module = types.ModuleType("independent_reused_w0_reader")
    module.__file__ = str(a.reader.resolve())
    exec(compile(source, str(a.reader.resolve()), "exec"), module.__dict__)
    module.HEADS = HEADS.copy()
    code = module.audit(a)
    report = json.loads((a.out / "report.json").read_text())
    aggregate = {}
    for mode in module.MODES:
        before = report["groups"]["baseline/" + mode]
        after = report["groups"]["candidate/" + mode]
        ratio = after["completed_trajectories_per_second"] / before["completed_trajectories_per_second"]
        aggregate[mode] = {
            "baseline_completed_trajectories_per_second": before["completed_trajectories_per_second"],
            "candidate_completed_trajectories_per_second": after["completed_trajectories_per_second"],
            "aggregate_effective_throughput_change_percent": 100 * (ratio - 1),
            "five_pair_faster_count": report["same_mode_pairs"][mode]["faster_pairs"],
            "strict_range_separation": report["same_mode_pairs"][mode]["strict_range_separation"],
        }
    invocation = {"reader_sha256": digest, "expected_heads_override": HEADS,
        "in_memory_layout_substitution": {"old": old, "new": new},
        "reader_logic_otherwise_unchanged": True, "new_benchmark_run": False,
        "scope": "all original CSV/provenance/config/runtime/telemetry rows, independent aggregate calculation",
        "arguments": {k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(a).items()},
        "aggregate_same_mode": aggregate,
        "conclusion": "No stable additional performance gain if ranges overlap or fewer than five pairs improve; retain broader runtime fixes on correctness evidence only."}
    (a.out / "invocation.json").write_text(json.dumps(invocation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit_status": report["status"], "aggregate_same_mode": aggregate}, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
