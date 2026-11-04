#!/usr/bin/env python3
"""Read final-head persisted W0 artifacts using the fixed independent reader.

Only the expected candidate HEAD is overridden in memory. No benchmark runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types

READER_SHA256 = "eda968172261e7b266278a4616c65d17b8b70582f3e670b41b7249dff45f5a3a"
HEADS = {
    "baseline": "d66ee714f2e755948534f95d9d125287ce8dbd91",
    "candidate": "1bdaf3a53227c9197b930dc00a01b752acd90067",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--prior", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    source = a.reader.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    assert digest == READER_SHA256, "independent reader bytes changed"
    module = types.ModuleType("independent_final_ci_reader")
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
            "aggregate_effective_throughput_change_percent": 100 * (ratio - 1),
            "faster_pairs": report["same_mode_pairs"][mode]["faster_pairs"],
            "strict_range_separation": report["same_mode_pairs"][mode]["strict_range_separation"],
        }
    record = {
        "reader_sha256": digest, "expected_heads_override": HEADS,
        "reader_logic_and_layout_unchanged": True, "benchmark_run": False,
        "arguments": {k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(a).items()},
        "aggregate_same_mode": aggregate,
        "scope": "Final exact HEAD CPU W0 only; not a native-SIF or GPU experiment",
    }
    (a.out / "reader-invocation.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "errors": report["errors"], "same_mode": aggregate}, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
