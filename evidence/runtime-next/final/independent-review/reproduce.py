#!/usr/bin/env python3
"""Replay the final offline CI audit using the byte-identical committed reader.

Requires the cloned repository, extracted CI artifacts, Git, and OmegaConf.
No benchmark or production entry point is executed. Output must be a new path.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys


EXPECTED_READER_SHA256 = "eda968172261e7b266278a4616c65d17b8b70582f3e670b41b7249dff45f5a3a"
FINAL_HEAD = "9d62b3eb86cb75a2266af49dc48cb851e69e0341"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--prior", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    script = args.repo / "evidence/runtime-next/agent-c/review_r1_ci.py"
    if hashlib.sha256(script.read_bytes()).hexdigest() != EXPECTED_READER_SHA256:
        raise ValueError("reader bytes differ from the independently verified reader")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("independent_final_ci_reader", script)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    # This updates only the explicitly reviewed revision expectation. The
    # unchanged audit still checks its source bytes, provenance and all data.
    reader.HEADS["candidate"] = FINAL_HEAD
    return reader.audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
