#!/usr/bin/env python3
"""Replay explicit config fixtures through the native Astrolabe shell entry.

No model runs. Invalid files use normal `run`; valid controls use `--dry-run`.
The injected test observer rejects scientific imports and child creation after
CLI entry. A success therefore proves a startup boundary, not model compatibility.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def load_cases(source):
    root = source / "tests/fixtures/formal_config"
    cases = []
    for manifest in sorted(root.glob("*/manifest.json")):
        data = json.loads(manifest.read_text())
        for row in data.get("cases", []) if isinstance(data, dict) else data:
            row = dict(row)
            row["group"] = manifest.parent.name
            row["id"] = row.get("id", Path(row["file"]).stem)
            row["config"] = manifest.parent / row["file"]
            cases.append(row)
    if not cases:
        raise ValueError("No explicit config cases found")
    return cases


def run_case(row, source, entrypoint, scratch, logroot, base_env):
    name = row["group"] + "-" + row["id"]
    directory = scratch / name
    directory.mkdir()
    observation = directory / "boundary.jsonl"
    output = directory / "run-output"
    # Corpus asset placeholders are repository-relative. Keep that explicit:
    # standalone's normal default otherwise chdirs to the YAML's directory.
    command = [str(entrypoint), "run", "--standalone", str(row["config"]), "--no-chdir"]
    expected = row.get("expected_error")
    if expected is None:
        command.append("--dry-run")
    command += ["env=env", "out_dir=" + str(output), "task_name=" + name]
    env = dict(base_env, ASTROLABE_CONFIG_BOUNDARY_LOG=str(observation))
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=source, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    timed_out = False
    try:
        text, _ = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        text, _ = process.communicate()
    elapsed = time.monotonic() - start
    traces = [json.loads(line) for line in observation.read_text().splitlines()] if observation.exists() else []
    events = [event for trace in traces for event in trace["events"]]
    forbidden = [event for event in events if event["event"].startswith("forbidden_")]
    cli_observed = any(event["event"] == "cli_entered" for event in events)
    good_exit = process.returncode == (0 if expected is None else 2)
    needles = expected if isinstance(expected, list) else [expected] if expected else []
    diagnostic = all(needle in text for needle in needles)
    no_output = not output.exists()
    passed = good_exit and diagnostic and no_output and cli_observed and not forbidden and not timed_out
    logfile = logroot / (name + ".log")
    logfile.write_text(text)
    return {
        **{key: value for key, value in row.items() if key != "config"},
        "config": str(row["config"]),
        "config_sha256": hashlib.sha256(row["config"].read_bytes()).hexdigest(),
        "command": command, "seconds": elapsed, "returncode": process.returncode,
        "expected_diagnostic_found": diagnostic, "cli_observed": cli_observed,
        "forbidden_events": forbidden, "output_absent": no_output,
        "timed_out": timed_out, "passed": passed, "log": str(logfile),
        "boundary": traces,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--entrypoint", type=Path)
    parser.add_argument("--fixture-source", type=Path,
                        help="Read corpus/probe from this checkout while testing --source")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    source = args.source.resolve()
    fixture_source = (args.fixture_source or source).resolve()
    entrypoint = args.entrypoint or source / "scripts/astrolabe"
    logroot = args.report.resolve().parent / (args.report.stem + "-cases")
    logroot.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="formal-config-"))
    env = dict(os.environ, ASTROLABE_RUNTIME_ACTIVE="1", ASTROLABE_PY=sys.executable,
               ASTROLABE_MODEL_EXECUTION="daemon",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               GH_CUDA_PREINITIALIZE="0", JAX_PLATFORMS="cpu",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join([
        str(fixture_source / "tests/fixtures/formal_config/sentinel"), str(source),
        env.get("PYTHONPATH", ""),
    ])
    cases = [row for row in load_cases(fixture_source) if args.filter in row["group"] + "-" + row["id"]]
    if not cases:
        raise ValueError("Filter matched no cases")
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        records = list(executor.map(
            lambda row: run_case(row, source, entrypoint, scratch, logroot, env), cases,
        ))
    # Run a known-good CPU sequence config without dry-run. The observer must
    # catch it reaching the application runtime, proving the boundary is live.
    control = next(row for row in load_cases(fixture_source)
                   if row["group"] == "startup" and row["id"] == "valid_minimal")
    control = dict(control, id="probe_sensitivity", expected_error="BOUNDARY_CONTROL_NOT_CONFIG_ERROR")
    control_result = run_case(control, source, entrypoint, scratch, logroot, env)
    probe_sensitive = (
        control_result["cli_observed"]
        and bool(control_result["forbidden_events"])
        and not control_result["passed"]
        and not control_result["timed_out"]
    )
    summary = {
        "source": str(source), "python": sys.executable, "entrypoint": str(entrypoint),
        "controlled_environment": {
            name: env[name] for name in (
                "ASTROLABE_RUNTIME_ACTIVE", "ASTROLABE_MODEL_EXECUTION",
                "CUDA_VISIBLE_DEVICES", "GH_CUDA_PREINITIALIZE", "JAX_PLATFORMS",
            )
        },
        "source_sha256": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([*(source / "astrolabe").rglob("*.py"),
                                source / "run.py", source / "scripts/astrolabe"])
        },
        "seconds": time.monotonic() - start,
        "passed": sum(row["passed"] for row in records), "total": len(records),
        "invalid": sum(row.get("expected_error") is not None for row in records),
        "valid_controls": sum(row.get("expected_error") is None for row in records),
        "probe_sensitive": probe_sensitive, "probe_control": control_result,
        "cases": records,
    }
    args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"cases", "probe_control", "source_sha256"}}))
    for row in records:
        if not row["passed"]:
            print("FAIL", row["group"], row["id"], row["log"])
    return int(summary["passed"] != summary["total"] or not probe_sensitive)


if __name__ == "__main__":
    raise SystemExit(main())
