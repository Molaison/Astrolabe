#!/usr/bin/env python3
"""Bounded real-NA-MPNN subprocess versus resident-daemon acceptance canary."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation.model_gpu_throughput import (  # noqa: E402
    HarnessError,
    _atomic_json,
    _flatten_arrays,
    _measure,
    _sha256,
    compare_outputs,
    counted_torch_load,
)


SCHEMA = "astrolabe.na-mpnn-gpu-acceptance/v1"
REPORT_NAME = "acceptance.json"
PREFLIGHT_NAME = "preflight.json"
DAEMON_FACTORY = "scripts.validation.na_mpnn_gpu_acceptance:daemon_factory"

REPO_HEAD = "835974f3d420de18b3eb1dc66f25fbd17c53cc09"
RUN_PY_SHA256 = (
    "f83959cc5f365a53dadae93d6e46c60f232832aa21b731ea1a6a9331b28a0f2a"
)
CHECKPOINT_SHA256 = (
    "5fb12951a0ad3ee44b91f33b7dd19114e755b49eb577d9eb1be71b24b7e49e4c"
)
CHECKPOINT_SIZE_BYTES = 27_667_482
DEFAULT_CHECKPOINT = Path("/ext/na_mpnn/models/design_model/s_19137.pt")

INPUT_SHA256 = (
    "3cce305d43bb4db3156ae756690c2ea427e8f6fcaa9838e434f281b7a9b2dc2e"
)
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "rna-archived-gpu-acceptance"
    / "gpu_v4"
    / "baseline"
    / "configured-output"
    / "001_seed_67101"
    / "structures"
    / "step_000_ABC_model.cif"
)
DEFAULT_REPO = Path("/ext/na_mpnn")
SEEDS = (267850363, 267850364)
ARMS = ("subprocess", "daemon")
REQUESTS_PER_ARM = 2
HARD_REQUEST_LIMIT = 4
TEMPERATURE = 0.3
BATCH_SIZE = 1
NUMBER_OF_BATCHES = 1
FIXED_RESIDUES = tuple(
    [f"B{index}" for index in range(1, 97)]
    + [f"C{index}" for index in range(1, 97)]
)
EXPECTED_ENCODED_RESIDUES = tuple(
    [f"A{index}" for index in range(1, 41)]
    + list(FIXED_RESIDUES)
)
EXPECTED_DESIGN_MASK = tuple(
    [1] * 40 + [0] * 192
)
EXPECTED_RESIDUES_PER_CHAIN = {"A": 40, "B": 96, "C": 96}
EXPECTED_ATOM_COUNT = 2399
EXPECTED_RNA_SEQUENCE = "UCGGCAGCUGCCAUCUGACU" * 2

_PROTEIN_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
_RNA_TO_ONE = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
}
_OUTPUT_RNA_TO_ONE = {
    "a": "A",
    "b": "A",
    "c": "C",
    "d": "C",
    "g": "G",
    "h": "G",
    "t": "U",
    "u": "U",
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
}
_EXPECTED_BUNDLE_KEYS = {
    "predicted_ppm",
    "true_sequence",
    "chain_labels",
    "mask",
    "protein_mask",
    "dna_mask",
    "rna_mask",
    "encoded_residues",
    "encoded_residues_dict",
    "restype_to_int",
}
REQUIRED_RUNNER_FLAGS = (
    "--mode",
    "--model_type",
    "--pdb_path",
    "--out_folder",
    "--checkpoint_na_mpnn",
    "--temperature",
    "--batch_size",
    "--number_of_batches",
    "--design_na_only",
    "--parse_na_only",
    "--na_shared_tokens",
    "--output_pdbs",
    "--output_sequences",
    "--output_specificity",
    "--seed",
    "--chains_to_design",
    "--fixed_residues",
    "--parse_these_chains_only",
)


class AcceptanceError(HarnessError):
    """The bounded NA-MPNN acceptance cannot be proven."""


@dataclass(frozen=True)
class RequestBudget:
    per_arm: Mapping[str, int]

    def __post_init__(self) -> None:
        if set(self.per_arm) != set(ARMS):
            raise ValueError(f"request budget arms must be {ARMS}")
        counts = {arm: int(self.per_arm[arm]) for arm in ARMS}
        if counts != {"subprocess": 2, "daemon": 2}:
            raise ValueError("request budget must be subprocess=2 daemon=2")
        if sum(counts.values()) != HARD_REQUEST_LIMIT:
            raise ValueError("request budget total must be exactly four")

    def claim(self, arm: str, accepted: Mapping[str, int]) -> int:
        if arm not in self.per_arm:
            raise ValueError(f"unknown NA-MPNN arm {arm!r}")
        if sum(int(value) for value in accepted.values()) >= HARD_REQUEST_LIMIT:
            raise AcceptanceError("NA-MPNN total hard budget of four is exhausted")
        next_count = int(accepted.get(arm, 0)) + 1
        if next_count > int(self.per_arm[arm]):
            raise AcceptanceError(
                f"NA-MPNN hard budget exceeded for {arm}: "
                f"{next_count}>{self.per_arm[arm]}"
            )
        return next_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Absent or empty directory for the report and raw evidence.",
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate real inputs/source/public entrypoint without CUDA or inference.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.expanduser().resolve()
    args.repo = args.repo.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.input = args.input.expanduser().resolve()
    args.python_bin = str(args.python_bin)
    if args.out_dir.exists():
        if not args.out_dir.is_dir():
            parser.error(f"--out-dir is not a directory: {args.out_dir}")
        if any(args.out_dir.iterdir()) and not args.preflight_only:
            parser.error(f"--out-dir must be absent or empty: {args.out_dir}")
    if not args.input.is_file():
        parser.error(f"Input structure is unavailable: {args.input}")
    if not Path(args.python_bin).is_file():
        parser.error(f"Python executable is unavailable: {args.python_bin}")
    return args


def _manifest_digest(records: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _repo_source_manifest(repo: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in sorted(repo.rglob("*.py")):
        if (
            ".git" in path.parts
            or "__pycache__" in path.parts
            or not path.is_file()
        ):
            continue
        records[str(path.relative_to(repo))] = _sha256(path)
    if not records:
        raise AcceptanceError(f"NA-MPNN source tree contains no Python files: {repo}")
    return records


def _execution_source_manifest() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "astrolabe" / "model_daemon.py",
        ROOT / "astrolabe" / "model_execution.py",
        ROOT / "astrolabe" / "mutators" / "na_mpnn.py",
        ROOT / "astrolabe" / "mutators" / "wrappers" / "na_mpnn_runner.py",
        ROOT / "astrolabe" / "mutators" / "wrappers" / "na_mpnn_daemon.py",
        ROOT / "scripts" / "validation" / "model_gpu_throughput.py",
    )
    return {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
        _sha256(path)
        for path in paths
    }


def input_facts(path: Path) -> dict[str, Any]:
    from Bio.PDB import MMCIFParser

    structure = MMCIFParser(QUIET=True).get_structure("na-mpnn-input", str(path))
    atom_count = sum(1 for _ in structure.get_atoms())
    chains: dict[str, list[str]] = {}
    for chain in structure[0]:
        names = [residue.get_resname().strip() for residue in chain]
        try:
            sequence = "".join(_PROTEIN_TO_ONE[name] for name in names)
            chain_type = "protein"
        except KeyError:
            try:
                sequence = "".join(_RNA_TO_ONE[name] for name in names)
                chain_type = "rna"
            except KeyError as exc:
                raise AcceptanceError(
                    f"Unsupported residue {exc.args[0]!r} in chain {chain.id}"
                ) from exc
        chains[str(chain.id)] = {
            "type": chain_type,
            "residues": len(names),
            "sequence": sequence,
        }
    residue_counts = {chain: int(item["residues"]) for chain, item in chains.items()}
    if atom_count != EXPECTED_ATOM_COUNT:
        raise AcceptanceError(
            f"Input atom count is {atom_count}, expected {EXPECTED_ATOM_COUNT}"
        )
    if residue_counts != EXPECTED_RESIDUES_PER_CHAIN:
        raise AcceptanceError(
            f"Input residue counts are {residue_counts}, "
            f"expected {EXPECTED_RESIDUES_PER_CHAIN}"
        )
    if (
        chains["A"]["type"] != "rna"
        or chains["A"]["sequence"] != EXPECTED_RNA_SEQUENCE
    ):
        raise AcceptanceError(
            "Input chain A must be the exact repeated 28490 RNA sequence"
        )
    if chains["B"]["type"] != "protein" or chains["C"]["type"] != "protein":
        raise AcceptanceError("Input chains B and C must be protein")
    digest = _sha256(path)
    if digest != INPUT_SHA256:
        raise AcceptanceError(
            f"Input SHA-256 is {digest}, expected {INPUT_SHA256}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "atom_count": atom_count,
        "chains": chains,
        "residue_counts": residue_counts,
    }


def request_plans() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "arm": arm,
            "request_index": index,
            "seed": seed,
            "temperature": TEMPERATURE,
            "batch_size": BATCH_SIZE,
            "number_of_batches": NUMBER_OF_BATCHES,
            "chains_to_design": "A",
            "parse_these_chains_only": "ABC",
            "design_na_only": 1,
            "parse_na_only": 0,
            "na_shared_tokens": 0,
            "fixed_residues": list(FIXED_RESIDUES),
            "design_mask": list(EXPECTED_DESIGN_MASK),
        }
        for arm in ARMS
        for index, seed in enumerate(SEEDS)
    )


def canonicalize_sequence(sequence: str) -> dict[str, Any]:
    chains = str(sequence).split("/")
    if len(chains) != 3:
        raise AcceptanceError(
            f"NA-MPNN FASTA must contain A/B/C chains, got {len(chains)}"
        )
    try:
        rna = "".join(_OUTPUT_RNA_TO_ONE[token] for token in chains[0])
    except KeyError as exc:
        raise AcceptanceError(
            f"Unsupported NA-MPNN RNA token {exc.args[0]!r}"
        ) from exc
    if len(rna) != 40 or set(rna) - set("ACGU"):
        raise AcceptanceError("Selected RNA sequence is not 40 nt over ACGU")
    protein_b, protein_c = chains[1:]
    if len(protein_b) != 96 or len(protein_c) != 96:
        raise AcceptanceError("Selected fixed protein chains are not 96 aa each")
    return {
        "raw": str(sequence),
        "canonical": "/".join((rna, protein_b, protein_c)),
        "rna": rna,
        "protein_b": protein_b,
        "protein_c": protein_c,
    }


def _validate_probability_bundle(
    bundle: Mapping[str, Any],
    *,
    seed: int,
    input_data: Mapping[str, Any],
) -> dict[str, Any]:
    if set(bundle) != _EXPECTED_BUNDLE_KEYS:
        raise AcceptanceError(
            "NA-MPNN probability bundle keys changed: "
            f"missing={sorted(_EXPECTED_BUNDLE_KEYS - set(bundle))} "
            f"extra={sorted(set(bundle) - _EXPECTED_BUNDLE_KEYS)}"
        )
    expected = {
        "predicted_ppm": (np.dtype("float64"), (232, 33)),
        "true_sequence": (np.dtype("int64"), (232,)),
        "chain_labels": (np.dtype("int32"), (232,)),
        "mask": (np.dtype("int32"), (232,)),
        "protein_mask": (np.dtype("int32"), (232,)),
        "dna_mask": (np.dtype("int32"), (232,)),
        "rna_mask": (np.dtype("int32"), (232,)),
    }
    for key, (dtype, shape) in expected.items():
        value = np.asarray(bundle[key])
        if value.dtype != dtype or value.shape != shape:
            raise AcceptanceError(
                f"{key} is {value.dtype}{value.shape}, expected {dtype}{shape}"
            )
    encoded = tuple(str(item) for item in bundle["encoded_residues"])
    if encoded != EXPECTED_ENCODED_RESIDUES:
        raise AcceptanceError("NA-MPNN encoded residue order changed")
    if not np.array_equal(np.asarray(bundle["mask"]), np.ones(232, dtype=np.int32)):
        raise AcceptanceError("NA-MPNN residue presence mask is not all ones")
    rna_mask = np.asarray(bundle["rna_mask"], dtype=np.int32)
    protein_mask = np.asarray(bundle["protein_mask"], dtype=np.int32)
    dna_mask = np.asarray(bundle["dna_mask"], dtype=np.int32)
    if not np.array_equal(rna_mask, np.asarray(list(EXPECTED_DESIGN_MASK), dtype=np.int32)):
        raise AcceptanceError("NA-MPNN RNA mask does not match chain A")
    if not np.array_equal(
        protein_mask,
        np.asarray([0] * 40 + [1] * 192, dtype=np.int32),
    ):
        raise AcceptanceError("NA-MPNN protein mask does not match fixed B/C chains")
    if not np.array_equal(dna_mask, np.zeros(232, dtype=np.int32)):
        raise AcceptanceError("NA-MPNN DNA mask is nonzero in the RNA-only canary")
    if tuple(int(item) for item in np.asarray(bundle["chain_labels"])) != tuple(
        [0] * 40 + [1] * 96 + [2] * 96
    ):
        raise AcceptanceError("NA-MPNN chain labels do not match A/B/C order")
    fixed = set(FIXED_RESIDUES)
    derived_design_mask = [
        int(residue.startswith("A") and residue not in fixed)
        for residue in encoded
    ]
    if tuple(derived_design_mask) != EXPECTED_DESIGN_MASK:
        raise AcceptanceError("Derived design mask violates fixed B/C residues")
    return {
        "seed": int(seed),
        "encoded_residues": list(encoded),
        "design_mask": derived_design_mask,
        "rna_mask": rna_mask.tolist(),
        "protein_mask": protein_mask.tolist(),
        "dna_mask": dna_mask.tolist(),
        "chain_labels": np.asarray(bundle["chain_labels"]).tolist(),
    }


def capture_runner_result(
    *,
    runner,
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
    input_data: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = result.get("probabilities")
    if not isinstance(bundle, Mapping):
        raise AcceptanceError("NAMPNNRunner returned no probability bundle")
    mask = _validate_probability_bundle(
        bundle,
        seed=int(plan["seed"]),
        input_data=input_data,
    )
    fasta_path = Path(result["fasta_path"])
    entries = runner._parse_fasta(fasta_path)
    selected_entries = [
        entry
        for entry in entries
        if entry["metadata"].get("overall_confidence") is not None
    ]
    if len(selected_entries) != 1:
        raise AcceptanceError(
            f"Expected one selected NA-MPNN FASTA entry, got {len(selected_entries)}"
        )
    selected = selected_entries[0]
    if selected["sequence"] != result["sequence"]:
        raise AcceptanceError("NAMPNNRunner selected FASTA sequence changed")
    canonical_entries = []
    for entry in selected_entries:
        metadata_seed = entry["metadata"].get("seed")
        if metadata_seed is None or int(float(metadata_seed)) != int(plan["seed"]):
            raise AcceptanceError("Selected FASTA entry does not preserve the request seed")
        canonical = canonicalize_sequence(entry["sequence"])
        if canonical["protein_b"] != input_data["chains"]["B"]["sequence"]:
            raise AcceptanceError("Fixed chain B changed in selected NA-MPNN output")
        if canonical["protein_c"] != input_data["chains"]["C"]["sequence"]:
            raise AcceptanceError("Fixed chain C changed in selected NA-MPNN output")
        canonical_entries.append(
            {
                "header": entry["header"],
                "metadata": dict(entry["metadata"]),
                "sequence": canonical,
            }
        )
    deterministic = {
        "probability_bundle": dict(bundle),
        "all_fasta_entries": entries,
        "selected_fasta_entries": canonical_entries,
        "fixed_mask": mask,
    }
    return {
        "deterministic": deterministic,
        "job_dir": str(result.get("job_dir") or ""),
        "fasta_path": str(fasta_path),
        "selected_raw": str(result["sequence"]),
    }


def _public_environment(repo: Path) -> dict[str, str]:
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH")
    paths = [str(repo), str(repo / "inference")]
    if inherited:
        paths.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def probe_public_entrypoint(repo: Path, python_bin: str) -> dict[str, Any]:
    environment = _public_environment(repo)
    commands = (
        [python_bin, "inference/run.py", "--help"],
        [
            python_bin,
            "-c",
            "import inference.run; import data_utils; import model_utils",
        ],
    )
    records = []
    for command in commands:
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(repo),
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise AcceptanceError(
                f"Public NA-MPNN entrypoint failed ({' '.join(command)}): {detail}"
            )
        if command[-1] == "--help":
            missing = [
                flag for flag in REQUIRED_RUNNER_FLAGS if flag not in completed.stdout
            ]
            if missing:
                raise AcceptanceError(
                    "Public NA-MPNN subprocess interface is missing flags: "
                    + ", ".join(missing)
                )
        records.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "duration_s": time.perf_counter() - started,
            }
        )
    return {"model_requests": 0, "commands": records}


def preflight(
    *,
    repo: Path,
    checkpoint: Path,
    input_path: Path,
    python_bin: str,
    probe_entrypoint: bool = True,
) -> dict[str, Any]:
    from astrolabe.mutators.wrappers.na_mpnn_daemon import verify_vendor
    from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner

    repo = repo.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    input_path = input_path.expanduser().resolve()
    verified_repo = verify_vendor(repo)
    if verified_repo != repo:
        raise AcceptanceError("verify_vendor returned another NA-MPNN repository")
    if not checkpoint.is_file():
        raise AcceptanceError(f"NA-MPNN checkpoint is unavailable: {checkpoint}")
    checkpoint_sha256 = _sha256(checkpoint)
    checkpoint_size = checkpoint.stat().st_size
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise AcceptanceError(
            f"NA-MPNN checkpoint SHA-256 is {checkpoint_sha256}, "
            f"expected {CHECKPOINT_SHA256}"
        )
    if checkpoint_size != CHECKPOINT_SIZE_BYTES:
        raise AcceptanceError(
            f"NA-MPNN checkpoint size is {checkpoint_size}, "
            f"expected {CHECKPOINT_SIZE_BYTES}"
        )
    run_py = repo / "inference" / "run.py"
    if _sha256(run_py) != RUN_PY_SHA256:
        raise AcceptanceError("Patched NA-MPNN inference/run.py changed")
    facts = input_facts(input_path)
    source_before = _repo_source_manifest(repo)
    execution_source_before = _execution_source_manifest()
    input_sha_before = facts["sha256"]
    checkpoint_sha_before = checkpoint_sha256
    if not probe_entrypoint:
        public_entrypoint = {
            "model_requests": 0,
            "commands": [],
            "probed": False,
        }
    else:
        public_entrypoint = {
            "probed": True,
            **probe_public_entrypoint(repo, python_bin),
        }
    source = _repo_source_manifest(repo)
    execution_source = _execution_source_manifest()
    if source != source_before:
        raise AcceptanceError("NA-MPNN source changed during entrypoint preflight")
    if execution_source != execution_source_before:
        raise AcceptanceError("NA-MPNN execution source changed during preflight")
    if _sha256(input_path) != input_sha_before:
        raise AcceptanceError("NA-MPNN input changed during entrypoint preflight")
    if _sha256(checkpoint) != checkpoint_sha_before:
        raise AcceptanceError("NA-MPNN checkpoint changed during entrypoint preflight")
    signature = str(inspect.signature(NAMPNNRunner.design_sequences))
    return {
        "schema": f"{SCHEMA}.preflight",
        "verified": True,
        "repo": {
            "path": str(repo),
            "head": REPO_HEAD,
            "run_py_sha256": RUN_PY_SHA256,
            "source_manifest": source,
            "source_manifest_sha256": _manifest_digest(source),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
        },
        "input": facts,
        "execution_source": {
            "manifest": execution_source,
            "manifest_sha256": _manifest_digest(execution_source),
        },
        "public_entrypoint": public_entrypoint,
        "frozen_during_preflight": True,
        "runner_design_sequences_signature": signature,
        "request_budget": {
            "hard_total": HARD_REQUEST_LIMIT,
            "per_arm": {"subprocess": 2, "daemon": 2},
            "af3_requests": 0,
        },
        "request_plans": list(request_plans()),
    }


class _AuditedResidentHandler:
    def __init__(
        self,
        handler: Callable[[Mapping[str, Any]], Any],
        *,
        resident_model: Any,
        capture_path: Path,
        checkpoint_path: Path,
        checkpoint_reads: Sequence[str],
        maximum_requests: int = 2,
    ) -> None:
        self.handler = handler
        self.pid = os.getpid()
        self.model_token = id(resident_model)
        self.capture_path = capture_path.expanduser().resolve()
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.checkpoint_reads = tuple(
            str(Path(path).expanduser().resolve()) for path in checkpoint_reads
        )
        if self.checkpoint_reads != (str(self.checkpoint_path),):
            raise AcceptanceError(
                "Resident NA-MPNN checkpoint load trace must contain one exact read"
            )
        self.maximum_requests = int(maximum_requests)
        self.request_count = 0
        self._lock = threading.Lock()
        try:
            self.device = str(next(resident_model.parameters()).device)
        except (AttributeError, StopIteration):
            self.device = str(getattr(resident_model, "device", "unknown"))

    def infer(self, payload: Mapping[str, Any]) -> Any:
        with self._lock:
            if self.request_count >= self.maximum_requests:
                raise AcceptanceError("Resident NA-MPNN daemon request budget exhausted")
            self.request_count += 1
            request_count = self.request_count
        started = time.perf_counter()
        status = "completed"
        error = None
        try:
            result = self.handler(payload)
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record = {
                "pid": self.pid,
                "model_token": self.model_token,
                "device": self.device,
                "model_load_count": len(self.checkpoint_reads),
                "checkpoint_reads": list(self.checkpoint_reads),
                "request_count": request_count,
                "request_delta": 1,
                "status": status,
                "error": error,
                "job_name": payload.get("job_name"),
                "job_dir": payload.get("job_dir"),
                "duration_s": time.perf_counter() - started,
            }
            self.capture_path.parent.mkdir(parents=True, exist_ok=True)
            with self.capture_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return result


def daemon_factory(settings: Mapping[str, Any]) -> _AuditedResidentHandler:
    import torch
    from astrolabe.mutators.wrappers.na_mpnn_daemon import NAMPNNHandler

    settings = dict(settings)
    capture_path = Path(settings.pop("_acceptance_capture_path"))
    checkpoint_path = Path(settings["checkpoint"])
    checkpoint_reads: list[str] = []
    with counted_torch_load(torch, checkpoint_reads):
        handler = NAMPNNHandler(settings)
    resident = handler.resident.get("model")
    if resident is None:
        raise AcceptanceError("NA-MPNN daemon handler has no resident model")
    return _AuditedResidentHandler(
        handler,
        resident_model=resident,
        capture_path=capture_path,
        checkpoint_path=checkpoint_path,
        checkpoint_reads=checkpoint_reads,
        maximum_requests=2,
    )


def _audited_daemon_runner_class():
    from astrolabe.model_daemon import DaemonSpec, OwnedModelDaemon
    from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner

    class AuditedDaemonRunner(NAMPNNRunner):
        def __init__(self, args, *, output_root: Path, capture_path: Path) -> None:
            super().__init__(args, output_root=output_root)
            if self.execution_mode != "daemon":
                raise ValueError("AuditedDaemonRunner requires execution_mode=daemon")
            self.acceptance_capture_path = capture_path.expanduser().resolve()
            self.records: list[dict[str, Any]] = []

        def _request_daemon(self, payload) -> None:
            with self._daemon_lock:
                if self._owned_daemon is None:
                    settings = dict(self._daemon_settings())
                    settings["_acceptance_capture_path"] = str(
                        self.acceptance_capture_path
                    )
                    self._owned_daemon = OwnedModelDaemon(
                        DaemonSpec(
                            backend="na_mpnn",
                            factory=DAEMON_FACTORY,
                            settings=settings,
                            python_bin=self.python,
                            cwd=str(self.repo_path),
                            env=self._patched_env(),
                        ),
                        startup_timeout=self.startup_timeout,
                        request_timeout=self.request_timeout,
                        shutdown_timeout=self.shutdown_timeout,
                        max_pending=self.max_pending,
                    )
                daemon = self._owned_daemon
            record = daemon.request(payload)
            self.records.append(dict(record))

        def status(self) -> dict[str, Any]:
            if self._owned_daemon is None:
                raise AcceptanceError("NA-MPNN daemon has not started")
            return self._owned_daemon.status()

    return AuditedDaemonRunner


def _runner_config(
    *,
    repo: Path,
    checkpoint: Path,
    python_bin: str,
    execution_mode: str,
) -> dict[str, Any]:
    return {
        "repo_path": str(repo),
        "python_bin": python_bin,
        "checkpoint": str(checkpoint),
        "temperature": TEMPERATURE,
        "batch_size": BATCH_SIZE,
        "number_of_batches": NUMBER_OF_BATCHES,
        "design_na_only": 1,
        "parse_na_only": 0,
        "na_shared_tokens": 0,
        "output_pdbs": 0,
        "seed": SEEDS[0],
        "extra_args": {},
        "execution_mode": execution_mode,
        "startup_timeout": 300,
        "shutdown_timeout": 30,
        "max_pending": 1,
    }


def _counted_subprocess_runner_class():
    from astrolabe.mutators.wrappers.na_mpnn_runner import NAMPNNRunner

    class CountedSubprocessRunner(NAMPNNRunner):
        def __init__(self, args, *, output_root: Path) -> None:
            super().__init__(args, output_root=output_root)
            if self.execution_mode != "subprocess":
                raise ValueError(
                    "CountedSubprocessRunner requires execution_mode=subprocess"
                )
            self.model_request_count = 0

        def _run_subprocess(self, payload) -> None:
            if self.model_request_count >= REQUESTS_PER_ARM:
                raise AcceptanceError("NA-MPNN subprocess request budget exhausted")
            self.model_request_count += 1
            super()._run_subprocess(payload)

    return CountedSubprocessRunner


def _request_features() -> dict[str, Any]:
    return {
        "chains_to_design": "A",
        "fixed_residues": list(FIXED_RESIDUES),
        "parse_these_chains_only": "ABC",
    }


def _read_daemon_capture(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AcceptanceError(f"Resident NA-MPNN capture is missing: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records


def validate_resident_evidence(
    records: Sequence[Mapping[str, Any]],
    statuses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != REQUESTS_PER_ARM or len(statuses) != REQUESTS_PER_ARM:
        raise AcceptanceError("Resident NA-MPNN evidence must contain two requests")
    pids = {int(item["pid"]) for item in records}
    tokens = {int(item["model_token"]) for item in records}
    loads = {int(item["model_load_count"]) for item in records}
    request_counts = [int(item["request_count"]) for item in records]
    request_deltas = [int(item["request_delta"]) for item in records]
    devices = {str(item["device"]) for item in records}
    status_pids = {int(item["pid"]) for item in statuses}
    status_request_counts = [int(item["request_count"]) for item in statuses]
    status_load_counts = {int(item["model_load_count"]) for item in statuses}
    checkpoint_read_sets = {
        tuple(str(Path(path).expanduser().resolve()) for path in item["checkpoint_reads"])
        for item in records
    }
    if (
        len(pids) != 1
        or len(tokens) != 1
        or loads != {1}
        or request_counts != [1, 2]
        or request_deltas != [1, 1]
        or pids != status_pids
        or status_request_counts != [1, 2]
        or status_load_counts != {1}
        or len(checkpoint_read_sets) != 1
    ):
        raise AcceptanceError(
            "Resident NA-MPNN daemon residency evidence is inconsistent"
        )
    return {
        "pid": next(iter(pids)),
        "model_token": next(iter(tokens)),
        "device": next(iter(devices)),
        "model_load_count": 1,
        "checkpoint_reads": list(next(iter(checkpoint_read_sets))),
        "request_counts": request_counts,
        "request_deltas": request_deltas,
    }


def _rng_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _gpu_facts(torch_module: Any) -> dict[str, Any]:
    properties = torch_module.cuda.get_device_properties(torch_module.cuda.current_device())
    executable = shutil.which("nvidia-smi")
    inventory = []
    if executable is not None:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            inventory = [
                [part.strip() for part in line.split(",", 3)]
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
    return {
        "torch_version": torch_module.__version__,
        "torch_cuda_version": torch_module.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "current_device": torch_module.cuda.current_device(),
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "") or ""),
        "major": properties.major,
        "minor": properties.minor,
        "total_memory_bytes": properties.total_memory,
        "nvidia_smi_inventory": inventory,
    }


@contextmanager
def _counted_subprocess_run(
    records: list[dict[str, Any]],
) -> Iterator[None]:
    import astrolabe.mutators.wrappers.na_mpnn_runner as runner_module

    original = runner_module.subprocess.run

    def observed(command, *args, **kwargs):
        started = time.perf_counter()
        completed = original(command, *args, **kwargs)
        records.append(
            {
                "command": list(command),
                "duration_s": time.perf_counter() - started,
                "returncode": getattr(completed, "returncode", None),
            }
        )
        return completed

    runner_module.subprocess.run = observed
    try:
        yield
    finally:
        runner_module.subprocess.run = original


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from astrolabe.model_execution import capture_random_state

    if not torch.cuda.is_available():
        raise AcceptanceError("The NA-MPNN canary requires a visible CUDA device")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preflight_report = preflight(
        repo=args.repo,
        checkpoint=args.checkpoint,
        input_path=args.input,
        python_bin=args.python_bin,
        probe_entrypoint=True,
    )
    _atomic_json(args.out_dir / PREFLIGHT_NAME, preflight_report)
    input_data = preflight_report["input"]
    source_before = preflight_report["repo"]["source_manifest"]
    execution_before = preflight_report["execution_source"]["manifest"]
    input_sha_before = input_data["sha256"]
    checkpoint_sha_before = preflight_report["checkpoint"]["sha256"]
    accepted = {"subprocess": 0, "daemon": 0}
    budget = RequestBudget({"subprocess": 2, "daemon": 2})

    torch.cuda.current_device()
    torch.cuda.synchronize()
    caller_rng_before = capture_random_state()
    caller_rng_before_sha = _rng_digest(caller_rng_before)

    CountedSubprocessRunner = _counted_subprocess_runner_class()
    subprocess_runner = CountedSubprocessRunner(
        _runner_config(
            repo=args.repo,
            checkpoint=args.checkpoint,
            python_bin=args.python_bin,
            execution_mode="subprocess",
        ),
        output_root=args.out_dir / "subprocess",
    )
    subprocess_outputs: list[dict[str, Any]] = []
    subprocess_times: list[float] = []
    subprocess_commands: list[dict[str, Any]] = []
    with _counted_subprocess_run(subprocess_commands):
        for index, seed in enumerate(SEEDS):
            accepted["subprocess"] = budget.claim("subprocess", accepted)
            subprocess_runner.extra_args["seed"] = seed
            result, elapsed = _measure(
                lambda: subprocess_runner.design_sequences(
                    structure_path=args.input,
                    job_name=f"subprocess_{index:02d}_seed_{seed}",
                    **_request_features(),
                ),
                synchronize=torch.cuda.synchronize,
            )
            subprocess_times.append(elapsed)
            subprocess_outputs.append(
                capture_runner_result(
                    runner=subprocess_runner,
                    result=result,
                    plan=request_plans()[index],
                    input_data=input_data,
                )
            )
    if subprocess_runner.model_request_count != REQUESTS_PER_ARM:
        raise AcceptanceError("Subprocess NA-MPNN hard request count is not two")
    if len(subprocess_commands) != REQUESTS_PER_ARM:
        raise AcceptanceError("Observed NA-MPNN subprocess process count is not two")

    daemon_capture = args.out_dir / "daemon" / "resident-requests.jsonl"
    AuditedDaemonRunner = _audited_daemon_runner_class()
    daemon_runner = AuditedDaemonRunner(
        _runner_config(
            repo=args.repo,
            checkpoint=args.checkpoint,
            python_bin=args.python_bin,
            execution_mode="daemon",
        ),
        output_root=args.out_dir / "daemon",
        capture_path=daemon_capture,
    )
    daemon_outputs: list[dict[str, Any]] = []
    daemon_times: list[float] = []
    daemon_statuses: list[dict[str, Any]] = []
    try:
        for index, seed in enumerate(SEEDS):
            accepted["daemon"] = budget.claim("daemon", accepted)
            daemon_runner.extra_args["seed"] = seed
            result, elapsed = _measure(
                lambda: daemon_runner.design_sequences(
                    structure_path=args.input,
                    job_name=f"daemon_{index:02d}_seed_{seed}",
                    **_request_features(),
                ),
                synchronize=torch.cuda.synchronize,
            )
            daemon_times.append(elapsed)
            daemon_outputs.append(
                capture_runner_result(
                    runner=daemon_runner,
                    result=result,
                    plan=request_plans()[REQUESTS_PER_ARM + index],
                    input_data=input_data,
                )
            )
            daemon_statuses.append(daemon_runner.status())
    finally:
        daemon_runner.shutdown()

    caller_rng_after = capture_random_state()
    caller_rng_after_sha = _rng_digest(caller_rng_after)
    if caller_rng_before_sha != caller_rng_after_sha:
        raise AcceptanceError("NA-MPNN requests changed caller RNG state")
    if accepted != {"subprocess": 2, "daemon": 2}:
        raise AcceptanceError(f"NA-MPNN request budget mismatch: {accepted}")

    daemon_records = _read_daemon_capture(daemon_capture)
    if len(daemon_records) != REQUESTS_PER_ARM:
        raise AcceptanceError("Resident NA-MPNN daemon request count is not two")
    resident_summary = validate_resident_evidence(daemon_records, daemon_statuses)

    comparisons: list[dict[str, Any]] = []
    subprocess_arrays: dict[str, np.ndarray] = {}
    daemon_arrays: dict[str, np.ndarray] = {}
    for index, (expected, actual) in enumerate(zip(subprocess_outputs, daemon_outputs)):
        comparison = compare_outputs(
            expected["deterministic"],
            actual["deterministic"],
        )
        comparison["request_index"] = index
        comparisons.append(comparison)
        for key, value in _flatten_arrays(expected["deterministic"]).items():
            subprocess_arrays[f"request_{index}.{key}"] = value
        for key, value in _flatten_arrays(actual["deterministic"]).items():
            daemon_arrays[f"request_{index}.{key}"] = value
    if not all(item["equal"] for item in comparisons):
        raise AcceptanceError(
            "Subprocess and resident-daemon NA-MPNN outputs differ: "
            f"{[item for item in comparisons if not item['equal']]}"
        )

    subprocess_array_path = args.out_dir / "subprocess_probabilities.npz"
    daemon_array_path = args.out_dir / "daemon_probabilities.npz"
    np.savez_compressed(subprocess_array_path, **subprocess_arrays)
    np.savez_compressed(daemon_array_path, **daemon_arrays)

    source_after = _repo_source_manifest(args.repo)
    execution_after = _execution_source_manifest()
    if source_after != source_before:
        raise AcceptanceError("Frozen NA-MPNN repository source changed during inference")
    if execution_after != execution_before:
        raise AcceptanceError("Astrolabe NA-MPNN execution source changed")
    if _sha256(args.input) != input_sha_before:
        raise AcceptanceError("NA-MPNN root input changed during inference")
    if _sha256(args.checkpoint) != checkpoint_sha_before:
        raise AcceptanceError("NA-MPNN checkpoint changed during inference")

    report = {
        "schema": SCHEMA,
        "verified": True,
        "scope": (
            "Exact within-run real NA-MPNN subprocess-vs-resident-daemon "
            "probabilities, selected FASTA sequences, fixed mask, requested "
            "seeds, and caller RNG. Timings are descriptive; no speed threshold."
        ),
        "created_unix_s": time.time(),
        "design": {
            "temperature": TEMPERATURE,
            "batch_size": BATCH_SIZE,
            "number_of_batches": NUMBER_OF_BATCHES,
            "seeds": list(SEEDS),
            "chains_to_design": "A",
            "parse_these_chains_only": "ABC",
            "fixed_residues": list(FIXED_RESIDUES),
            "design_mask": list(EXPECTED_DESIGN_MASK),
            "rna_output_alphabet": "ACGU",
        },
        "request_budget": {
            "hard_total": HARD_REQUEST_LIMIT,
            "accepted": accepted,
            "af3_requests": 0,
        },
        "input": input_data,
        "repo": preflight_report["repo"],
        "checkpoint": preflight_report["checkpoint"],
        "execution_source": preflight_report["execution_source"],
        "public_entrypoint": preflight_report["public_entrypoint"],
        "gpu": _gpu_facts(torch),
        "requests": {
            "subprocess": [
                {
                    **request_plans()[index],
                    "duration_s": subprocess_times[index],
                    "job_dir": subprocess_outputs[index]["job_dir"],
                    "fasta_path": subprocess_outputs[index]["fasta_path"],
                    "process": subprocess_commands[index],
                }
                for index in range(REQUESTS_PER_ARM)
            ],
            "daemon": [
                {
                    **request_plans()[REQUESTS_PER_ARM + index],
                    "duration_s": daemon_times[index],
                    "job_dir": daemon_outputs[index]["job_dir"],
                    "fasta_path": daemon_outputs[index]["fasta_path"],
                    "resident_capture": daemon_records[index],
                    "status": daemon_statuses[index],
                }
                for index in range(REQUESTS_PER_ARM)
            ],
        },
        "resident_daemon": resident_summary,
        "caller_rng": {
            "equal": True,
            "before_sha256": caller_rng_before_sha,
            "after_sha256": caller_rng_after_sha,
        },
        "exact_comparison": comparisons,
        "raw_arrays": {
            "subprocess": str(subprocess_array_path),
            "daemon": str(daemon_array_path),
            "subprocess_sha256": _sha256(subprocess_array_path),
            "daemon_sha256": _sha256(daemon_array_path),
        },
        "performance": {
            "measured_requests": {
                "subprocess_s": subprocess_times,
                "daemon_s": daemon_times,
            },
            "descriptive_only": True,
            "speedup_claim": None,
            "statistical_claim": None,
        },
        "frozen_after_run": {
            "repo_source_manifest_sha256": _manifest_digest(source_after),
            "execution_source_manifest_sha256": _manifest_digest(execution_after),
            "input_sha256": _sha256(args.input),
            "checkpoint_sha256": _sha256(args.checkpoint),
        },
    }
    _atomic_json(args.out_dir / REPORT_NAME, report)
    return report


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = preflight(
        repo=args.repo,
        checkpoint=args.checkpoint,
        input_path=args.input,
        python_bin=args.python_bin,
        probe_entrypoint=True,
    )
    _atomic_json(args.out_dir / PREFLIGHT_NAME, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        run_preflight(args)
        print(args.out_dir / PREFLIGHT_NAME)
        return 0
    report = _run(args)
    print(args.out_dir / REPORT_NAME)
    if not report["verified"]:
        raise AcceptanceError("NA-MPNN GPU acceptance report is not verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
