"""Argparse entry point for offline profiling diagnostics (no model imports)."""

from __future__ import annotations

import argparse
import json
import sys

from .report import compare_sessions, report_session


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="astrolabe profile", description="Offline observational profiling diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Write summary.json, report.html and trace.json")
    report.add_argument("directory", metavar="DIR")
    report.add_argument("--output", metavar="DIR", help="Output directory (default: session directory)")
    compare = commands.add_parser("compare", help="Diagnostic comparison; stdout JSON unless --output is provided")
    compare.add_argument("baseline", metavar="BASE")
    compare.add_argument("candidate", metavar="CANDIDATE")
    compare.add_argument("--output", metavar="DIR", help="Also write comparison.json and comparison.html")
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            result = report_session(args.directory, args.output)
        else:
            result = compare_sessions(args.baseline, args.candidate, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError) as exc:
        print(f"profiling {args.command}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
