#!/usr/bin/env python3
"""Replay daily task YAML via the real shell CLI in fresh CPU processes.

Every invalid case must exit 2, contain its diagnostic and cross no runtime
boundary. Controls use dry-run. Timings include process/CLI startup and targeted
test observation; they are not model-startup performance measurements.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def path_state(path):
    if path.is_symlink():
        return {"link": os.readlink(path)}
    if not path.exists():
        return {"absent": True}
    if path.is_file():
        return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "mode": path.stat().st_mode}
    return {"entries": {str(p.relative_to(path)): path_state(p)
                        for p in sorted(path.iterdir())}}


def execute(command, cwd, env, trace, log):
    start = time.monotonic()
    proc = subprocess.Popen(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    timeout = False
    try:
        text, _ = proc.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        timeout = True
        os.killpg(proc.pid, signal.SIGKILL)
        text, _ = proc.communicate()
    elapsed = time.monotonic() - start
    log.write_text(text)
    events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    return dict(command=command, seconds=elapsed, returncode=proc.returncode,
                timed_out=timeout, events=events, log=str(log)), text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    source = args.source.resolve()
    root = source / "tests/fixtures/daily_task_config"
    manifest = json.loads((root / "manifest.json").read_text())
    logroot = args.report.resolve().with_suffix("")
    logroot.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="daily-task-", dir=logroot))
    entrypoint = args.entrypoint or source / "scripts/astrolabe"
    env = dict(os.environ, ASTROLABE_RUNTIME_ACTIVE="1", ASTROLABE_PY=sys.executable,
               ASTROLABE_MODEL_EXECUTION="daemon", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", GH_CUDA_PREINITIALIZE="0", JAX_PLATFORMS="cpu",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join([
        str(source / "tests/daily_config_probe"), str(source), env.get("PYTHONPATH", "")])
    def source_identity():
        paths = [*source.glob("astrolabe/**/*.py"), *source.glob("config/**/*.yaml"),
                 source / "scripts/astrolabe", Path(__file__).resolve(),
                 source / "tests/daily_config_probe/sitecustomize.py", root / "manifest.json",
                 *root.glob("**/*"), *source.glob("tests/fixtures/formal_config/**/*")]
        return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(set(paths)) if path.is_file()}
    identity_before = source_identity()

    def run(row):
        ident = row["id"]
        directory = scratch / ident
        directory.mkdir()
        output = directory / "output"
        trace = logroot / (ident + ".trace.jsonl")
        trace.unlink(missing_ok=True)
        config = root / row["file"]
        command = [str(entrypoint), "run", "--standalone", str(config), "--no-chdir"]
        if row["valid"]:
            command.append("--dry-run")
        preserve = row.get("preserve_fields", [])
        command += ["env=env"]
        if "out_dir" not in preserve:
            command += ["out_dir=" + str(output)]
        if "task_name" not in preserve:
            command += ["task_name=" + ident]
        # Snapshot the actual configured targets, including preserved paths.
        # Type-invalid path values have no target; the global write bomb still
        # prevents a consumer from coercing them into an accidental path.
        from omegaconf import OmegaConf
        raw = OmegaConf.to_container(OmegaConf.load(config), resolve=False)
        protected = [output]
        for field in ("out_dir", "raw_model_results_dir"):
            value = raw.get(field)
            if field == "out_dir" and field not in preserve:
                continue
            if isinstance(value, str) and value.strip() and "${" not in value:
                target = Path(value).expanduser()
                protected.append(target if target.is_absolute() else source / target)
        before = {str(path): path_state(path) for path in protected}
        case_env = dict(env, ASTROLABE_DAILY_PROBE=str(trace), **row.get("env", {}))
        result, text = execute(command, source, case_env, trace, logroot / (ident + ".log"))
        after = {str(path): path_state(path) for path in protected}
        events = result["events"]
        forbidden = [e for e in events if e["event"].startswith("forbidden_")]
        errors = [e for e in events if e["event"] in {"stage_error", "stage_findings"}]
        needles = row.get("expected_error") or []
        if isinstance(needles, str):
            needles = [needles]
        observed = any(e["event"] == "cli_entered" for e in events)
        found = all(s in text for s in needles)
        stage = errors[0]["stage"] if errors else ("composition" if result["returncode"] == 2 else None)
        stage_match = row["valid"] or stage == row["expected_stage"]
        entered = next((e["at"] for e in events if e["event"] == "stage_enter" and e["stage"] == "startup"), None)
        stopped = next((e["at"] for e in events if e["event"] == "stage_error" and e["stage"] == "startup"), None)
        passed = (observed and not forbidden and not result["timed_out"] and found
                  and result["returncode"] == (0 if row["valid"] else 2) and not output.exists()
                  and stage_match and before == after)
        return dict(**row, **result, config=str(config),
                    config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                    cli_observed=observed, forbidden=forbidden, diagnostic_found=found,
                    failure_stage=stage, validation_seconds=(stopped-entered) if stopped and entered else None,
                    stage_match=stage_match, protected_before=before, protected_after=after,
                    output_absent=not output.exists(), passed=passed)

    rows = [row for row in manifest["cases"] if args.filter in row["id"]]
    if not rows:
        raise ValueError("No cases selected")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, rows))

    calibrations = []
    # Each barrier must fail loudly if reached, independently of malformed YAML.
    snippets = {
        "model_import": "import astrolabe.cli; import torch",
        "evaluator_factory": "import astrolabe.cli; from astrolabe.evaluator_startup import initialize_plan; initialize_plan([], None, None, workers=1, seed=None)",
        "model_owner": "import astrolabe.cli; from astrolabe.model_daemon import OwnedModelDaemon; OwnedModelDaemon(None)",
        "evaluator_import": "import astrolabe.cli; import astrolabe.evaluators.utils",
        "child_process": "import astrolabe.cli; import subprocess; subprocess.run(['true'])",
        "output_directory": "import astrolabe.cli; import os; os.mkdir(" + repr(str(scratch / "blocked-directory")) + ")",
        "output_file": "import astrolabe.cli; open(" + repr(str(scratch / "blocked-file")) + ", 'w')",
    }
    expected_events = {
        "model_import": ("forbidden_import", "module", "torch"),
        "evaluator_factory": ("forbidden_factory", "target", "astrolabe.evaluator_startup.initialize_plan"),
        "model_owner": ("forbidden_factory", "target", "astrolabe.model_daemon.OwnedModelDaemon.__init__"),
        "evaluator_import": ("forbidden_evaluator_import", "module", "astrolabe.evaluators.utils"),
        "child_process": ("forbidden_effect", "operation", "subprocess.Popen"),
        "output_directory": ("forbidden_effect", "operation", "os.mkdir"),
        "output_file": ("forbidden_write", "path", str(scratch / "blocked-file")),
    }
    for name, code in snippets.items():
        trace = logroot / (name + ".trace.jsonl")
        trace.unlink(missing_ok=True)
        result, _ = execute([sys.executable, "-c", code], source,
                            dict(env, ASTROLABE_DAILY_PROBE=str(trace)), trace,
                            logroot / (name + ".log"))
        result["name"] = name
        event, key, value = expected_events[name]
        result["passed"] = result["returncode"] != 0 and any(
            e["event"] == event and e.get(key) == value for e in result["events"])
        calibrations.append(result)
    # A valid normal CLI run must actually reach a protected runtime boundary.
    normal = dict(next(r for r in manifest["cases"] if r["valid"]), id="normal_control",
                  valid=False, expected_error=[])
    result = run(normal)
    result["passed"] = bool(result["forbidden"]) and result["cli_observed"]
    calibrations.append(result)
    identity_after = source_identity()
    report = dict(source=str(source), python=sys.executable, jobs=args.jobs,
                  source_hashes=identity_before, source_stable=identity_before == identity_after,
                  manifest_sha256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
                  cases=results, calibrations=calibrations, fields=manifest["fields"],
                  passed=all(r["passed"] for r in results+calibrations)
                         and identity_before == identity_after)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=sum(r["passed"] for r in results), total=len(results),
                          calibrations=sum(r["passed"] for r in calibrations),
                          report=str(args.report))))
    for row in results:
        if not row["passed"]:
            print("FAIL", row["id"], row["failure_stage"], row["returncode"], row["log"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
