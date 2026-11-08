"""Supervise an explicit inventory of independent scientific commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .session import IndependentCommand, resource_session, run_independent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    options = parser.parse_args(argv)
    manifest = json.loads(options.manifest.read_text())
    if manifest.get("schema") != "astrolabe.independent-commands/1":
        parser.error("Expected astrolabe.independent-commands/1 manifest")
    commands = []
    for item in manifest["commands"]:
        if (not isinstance(item.get("argv"), list) or not item["argv"]
                or any(not isinstance(arg, str) for arg in item["argv"])):
            parser.error("Each command needs a nonempty argv list of strings")
        commands.append(IndependentCommand(
            identity=item["identity"], argv=tuple(item["argv"]),
            output_dir=Path(item["output_dir"]),
            cwd=Path(item["cwd"]) if item.get("cwd") else None,
            env=item.get("env"),
        ))
    if not commands:
        parser.error("The command inventory is empty")
    if options.report.exists():
        parser.error("Refusing to overwrite an existing command report")
    with resource_session(manifest.get("execution")) as client:
        results = run_independent(
            commands, max_trajectories=manifest.get("max_trajectories", len(commands)),
            timeout=manifest.get("command_timeout_seconds"),
        )
        snapshot = client.snapshot() if client is not None else None
    report = {"schema": manifest["schema"], "results": results,
              "resource_snapshot": snapshot,
              "all_completed": all(item["state"] == "completed" for item in results)}
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
