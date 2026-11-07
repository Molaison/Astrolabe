#!/usr/bin/env python3
"""Check the bounded recovery model and its correspondence to real Runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "verification/runtime_recovery"
TEST = ROOT / "tests/test_runtime_recovery_contract.py"
RUNTIME = "astrolabe/design/runtime.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command, *, cwd, log, env=None):
    with log.open("w", encoding="utf-8") as stream:
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, start_new_session=True) as child:
            try:
                code = child.wait(timeout=120)
            except subprocess.TimeoutExpired as error:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise RuntimeError(f"Check timed out; no acceptance: {log}") from error
    return code, log.read_text(encoding="utf-8")


def check_model(jar, output, name, *, budget=1, replay=False, refund=False, persist=True,
                violated=None):
    destination = output / name
    destination.mkdir()
    shutil.copyfile(MODEL / "RuntimeRecovery.tla", destination / "RuntimeRecovery.tla")
    (destination / "RuntimeRecovery.cfg").write_text(
        "SPECIFICATION Spec\nCONSTANTS\n Slots = {first, second}\n"
        f" Budget = {budget}\n ReplayUnknown = {str(replay).upper()}\n"
        f" RefundUnknown = {str(refund).upper()}\n"
        f" PersistIssue = {str(persist).upper()}\n"
        "INVARIANTS TypeOK AtMostOnce BudgetNeverRefunded IssuedBeforeInvoke\n"
        "CHECK_DEADLOCK FALSE\n", encoding="utf-8",
    )
    command = ["java", "-Xmx256m", "-XX:+UseParallelGC", "-cp", str(jar),
               "tlc2.TLC", "-workers", "1", "-metadir", str(destination / "states"),
               "-config", "RuntimeRecovery.cfg", "RuntimeRecovery.tla"]
    code, text = execute(command, cwd=destination, log=destination / "tlc.log")
    expected = (code != 0 and f"Invariant {violated} is violated." in text
                if violated else code == 0 and
                "Model checking completed. No error has been found." in text)
    if not expected:
        raise RuntimeError(f"Unexpected model result ({code}): {destination / 'tlc.log'}")
    return {"case": name, "exit_code": code, "expected_violation": violated,
            "log": str(destination / "tlc.log"), "command": command}


def check_code(source, output, name, *, expected_failure=None):
    log = output / f"{name}.log"
    command = [
        sys.executable, "-c",
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "import pytest; sys.exit(pytest.main(sys.argv[1:]))",
        str(source), str(TEST), "-q", "-p", "no:cacheprovider",
        f"--junitxml={output / (name + '.xml')}",
    ]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="",
               PYTEST_PLUGINS="", CUDA_VISIBLE_DEVICES="",
               ASTROLABE_RECOVERY_SOURCE=str(source / RUNTIME))
    code, text = execute(command, cwd=source, log=log, env=env)
    cases = ET.parse(output / (name + ".xml")).findall(".//testcase")
    failures = [case for case in cases if case.find("failure") is not None]
    invalid = [case for case in cases
               if case.find("error") is not None or case.find("skipped") is not None]
    expected = (code == 1 and not invalid and
                any(case.attrib["name"].startswith(expected_failure) for case in failures)
                if expected_failure else code == 0 and bool(cases) and
                not failures and not invalid)
    if not expected:
        raise RuntimeError(f"Unexpected real-code result ({code}): {log}")
    return {"case": name, "exit_code": code, "expected_failure": expected_failure,
            "tests": len(cases), "failures": len(failures),
            "log": str(log), "command": command}


def self_test_code(output):
    mutations = (
        ("replay", 'previous["status"] in ("issued", "unknown")',
         'previous["status"] in ("deliberately_disabled",)',
         "test_crash_resume_never_replays"),
        ("refund", 'used = sum(row["boundary"] == boundary for row in self._records("attempt").values())',
         'used = sum(row["boundary"] == boundary and row["status"] == "completed" for row in self._records("attempt").values())',
         "test_budget_survives_unknown"),
        ("issue", 'previous["status"] = "issued"\n        self._write("attempt", slot, previous)',
         'previous["status"] = "issued"',
         "test_crash_resume_never_replays[invoke-"),
    )
    results = []
    for name, before, after, failure in mutations:
        with tempfile.TemporaryDirectory(prefix=f"astrolabe-{name}-") as temporary:
            source = Path(temporary)
            shutil.copytree(ROOT / "astrolabe", source / "astrolabe",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            path = source / RUNTIME
            text = path.read_text(encoding="utf-8")
            if text.count(before) != 1:
                raise RuntimeError(f"Mutation target drifted; inspect {RUNTIME}: {name}")
            path.write_text(text.replace(before, after), encoding="utf-8")
            results.append(check_code(source, output, f"code_{name}",
                                      expected_failure=failure))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jar", type=Path, default=os.environ.get("TLA2TOOLS_JAR"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true",
                        help="Also require broken models and real code variants to fail.")
    parser.add_argument("--behavior-only", action="store_true",
                        help="Run real code tests without the source-sync gate during diagnosis.")
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="astrolabe-recovery-"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "checks": [], "source": str(ROOT),
              "mode": "behavior-only" if args.behavior_only else
                      "self-test" if args.self_test else "acceptance",
              "verification_sha256": {
                  str(path.relative_to(ROOT)): digest(path)
                  for path in (Path(__file__).resolve(), TEST,
                               MODEL / "RuntimeRecovery.tla", MODEL / "coverage.json")
              }}
    try:
        coverage = json.loads((MODEL / "coverage.json").read_text(encoding="utf-8"))
        actual = {name: digest(ROOT / name) for name in coverage["files"]}
        report.update(sync_commit=coverage["sync_commit"], source_sha256=actual)
        if args.behavior_only:
            report["checks"].append(check_code(ROOT, output, "code"))
        else:
            stale = [name for name, sha in actual.items() if sha != coverage["files"][name]]
            if stale:
                raise RuntimeError(f"Model/source sync needs review: {', '.join(stale)}")
            jar = Path(args.jar).resolve() if args.jar else output / "tla2tools.jar"
            if not jar.exists() and not args.jar:
                with urllib.request.urlopen(coverage["tool"]["url"], timeout=60) as response:
                    jar.write_bytes(response.read())
            if digest(jar) != coverage["tool"]["sha256"]:
                raise RuntimeError(f"TLC jar checksum mismatch: {jar}")
            report["tool"] = coverage["tool"]
            for budget in (1, 2):
                report["checks"].append(check_model(jar, output, f"budget_{budget}", budget=budget))
            report["checks"].append(check_code(ROOT, output, "code"))
            if args.self_test:
                report["checks"].append(check_model(
                    jar, output, "model_replay", replay=True, violated="AtMostOnce"))
                report["checks"].append(check_model(
                    jar, output, "model_refund", refund=True, violated="BudgetNeverRefunded"))
                report["checks"].append(check_model(
                    jar, output, "model_issue", persist=False, violated="IssuedBeforeInvoke"))
                report["checks"].extend(self_test_code(output))
                report["checks"].append(check_code(ROOT, output, "code_restored"))
        report["status"] = "passed"
    except (OSError, ValueError, RuntimeError, ET.ParseError) as error:
        report["error"] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        (output / "result.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{report['status']}: {output / 'result.json'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
