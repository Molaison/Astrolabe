#!/usr/bin/env python3
"""Inspect the pinned Anthropic optimization kits.

Installation is container-only. This manifest helper intentionally has no
host-Python installation path; build a SIF with
``anthropic_kits_apptainer.py`` and let the definition's ``%post`` install
the pinned software inside that image.

Examples:
  python scripts/containers/anthropic_kits.py check
  python scripts/containers/anthropic_kits.py list
  python scripts/containers/anthropic_kits_apptainer.py --kit esm_if1 --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astrolabe.optimized_backends import (  # noqa: E402
    MANIFEST_PATH,
    OptimizedBackendError,
    load_manifest as _load_manifest,
    selected_kit_specs,
    validate_manifest_source,
)


def load_manifest() -> dict:
    return _load_manifest(MANIFEST_PATH)


def selected_kits(manifest: dict, names: list[str] | None, software_ids: list[str] | None) -> list[tuple[str, dict]]:
    return selected_kit_specs(manifest, names, software_ids)


def check_source(manifest: dict) -> list[dict]:
    return validate_manifest_source(manifest)


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "list"))
    parser.add_argument("--kit", action="append", dest="kits", help="Anthropic kit name; repeat to select multiple")
    parser.add_argument("--software", action="append", dest="software_ids", help="Current GH software id; repeat to select multiple")
    args = parser.parse_args()
    try:
        manifest = load_manifest()
        rows = check_source(manifest)
    except (OptimizedBackendError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    if args.command in {"check", "list"}:
        if args.command == "list":
            print_json({"source": manifest["source"], "kits": rows, "unmapped": manifest.get("unmapped", {})})
        else:
            print_json({"status": "ready", "sourceRevision": manifest["source"]["revision"], "kits": rows})
        return 0

    raise AssertionError("unreachable: install is not a host-Python command")


if __name__ == "__main__":
    sys.exit(main())
