from __future__ import annotations

import logging
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from typing import Dict, List, Sequence

import numpy as np


logger = logging.getLogger(__name__)


class NAMPNNRunner:
    """Wrapper around baker-laboratory/NA-MPNN inference script."""

    def __init__(self, args, *, output_root: pathlib.Path) -> None:
        self.args = args
        self.repo_path = self._resolve_repo_path(self._get("repo_path"))
        self.python = str(self._get("python_bin", sys.executable))
        self.checkpoint = self._resolve_checkpoint(self._get("checkpoint"))
        self.temperature = float(self._get("temperature", 0.1))
        self.batch_size = int(self._get("batch_size", 1))
        self.number_of_batches = int(self._get("number_of_batches", 1))
        self.design_na_only = int(self._get("design_na_only", 0))
        self.parse_na_only = int(self._get("parse_na_only", 0))
        self.na_shared_tokens = int(self._get("na_shared_tokens", 0))
        self.output_pdbs = int(self._get("output_pdbs", 0))
        self.job_name = self._get("job_name")
        self.extra_args = {str(key).strip().replace("-", "_"): value
                           for key, value in dict(self._get("extra_args", {}) or {}).items()}
        if {"pdb_path", "out_folder"}.intersection(self.extra_args):
            raise ValueError("NA-MPNN input/output paths belong to the scientific request, not extra_args")
        if self.extra_args.get("model_type", "na_mpnn") != "na_mpnn" or self.extra_args.get("mode", "design") != "design":
            raise ValueError("NA-MPNN mutator requires native na_mpnn/design mode")
        if self.extra_args.get("checkpoint_na_mpnn"):
            self.checkpoint = self._resolve_checkpoint(self.extra_args["checkpoint_na_mpnn"])
        for key, cast in (("temperature", float), ("batch_size", int), ("number_of_batches", int),
                          ("design_na_only", int), ("parse_na_only", int), ("na_shared_tokens", int), ("output_pdbs", int)):
            if self.extra_args.get(key) is not None:
                setattr(self, key, cast(self.extra_args[key]))
        self.output_root = pathlib.Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        fixed_seed = int(self.extra_args.get("seed", 0) or 0) != 0
        self.daemon = bool(self._get("daemon", os.environ.get("ASTROLABE_MUTATOR_WORKER") == "1" and fixed_seed))
        self.startup_timeout = float(self._get("startup_timeout", 600))
        self.request_timeout = self._get("request_timeout")
        self._lock = threading.RLock()
        self._proc = None
        self._ready = None
        self._buffer = bytearray()
        self._closed = False
        self._started = False
        self._bootstrap = None
        # A one-shot subprocess's successful exit is not a model load counter.
        self.model_load_count = None

    def _extra_arguments(self):
        result = []
        for key, value in self.extra_args.items():
            if value is None:
                continue
            # Native NA-MPNN argparse spells these options with underscores.
            key = str(key).strip().replace("-", "_")
            if key:
                result.extend(["--" + key, str(value)])
        return result

    def _ensure_daemon(self):
        from astrolabe.mutators.wrappers.subprocess_protocol import read_message, stop_process
        with self._lock:
            if self._closed:
                raise RuntimeError("NA-MPNN runner is closed")
            if self._proc is not None and self._proc.poll() is None:
                return
            if self._started:
                raise RuntimeError("NA-MPNN process exited; implicit reload would alter request lifecycle")
            self._bootstrap = tempfile.TemporaryDirectory(prefix="nampnn-init-", dir=self.output_root)
            root = pathlib.Path(self._bootstrap.name)
            argv = ["--mode", "design", "--model_type", "na_mpnn", "--pdb_path", "",
                    "--out_folder", str(root / "initial"), "--checkpoint_na_mpnn", str(self.checkpoint),
                    "--batch_size", str(self.batch_size), "--temperature", str(self.temperature),
                    "--na_shared_tokens", str(self.na_shared_tokens), "--output_pdbs", "0"]
            argv.extend(self._extra_arguments())
            spec = root / "spec.json"
            spec.write_text(json.dumps({"native_path": str(self.repo_path / "inference/run.py"), "argv": argv}))
            script = pathlib.Path(__file__).with_name("na_mpnn_daemon.py")
            self._proc = subprocess.Popen([self.python, str(script), str(spec)],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=None, cwd=str(self.repo_path), env=self._patched_env(),
                                          text=True, bufsize=1)
            self._started = True
            try:
                ready = read_message(self._proc, self._buffer, timeout=self.startup_timeout)
                if (ready.get("type") != "ready" or ready.get("pid") != self._proc.pid
                        or ready.get("model_load_count") != 1 or ready.get("weights_loaded") is not True):
                    raise RuntimeError("invalid NA-MPNN loaded-model handshake")
                self._ready = ready
                self.model_load_count = ready["model_load_count"]
            except BaseException:
                stop_process(self._proc)
                self._closed = True
                self._bootstrap.cleanup()
                raise

    def _infer_resident(self, argv):
        from astrolabe.mutators.wrappers.subprocess_protocol import read_message
        from astrolabe.runtime.client import OutcomeUnknown
        with self._lock:
            self._ensure_daemon()
            request_id = uuid.uuid4().hex
            try:
                self._proc.stdin.write(json.dumps({"type": "infer", "id": request_id, "argv": argv}) + "\n")
                self._proc.stdin.flush()
                response = read_message(self._proc, self._buffer, timeout=self.request_timeout)
            except (OSError, EOFError, TimeoutError, ValueError) as exc:
                self.close()
                raise OutcomeUnknown("NA-MPNN result is unknown; request was not replayed") from exc
            if response.get("id") != request_id:
                self.close()
                raise OutcomeUnknown("NA-MPNN response identity mismatch; request was not replayed")
            if response.get("type") != "result":
                self.close()
                raise RuntimeError("NA-MPNN inference failed: " + str(response.get("error")))
            if response.get("model_load_count") != self.model_load_count:
                self.close()
                raise RuntimeError("NA-MPNN unexpectedly reloaded weights during a request")

    def close(self):
        from astrolabe.mutators.wrappers.subprocess_protocol import stop_process
        with self._lock:
            self._closed = True
            stop_process(self._proc)
            self._proc = None
            self._ready = None
            if self._bootstrap is not None:
                self._bootstrap.cleanup()
                self._bootstrap = None

    shutdown = close

    def design_sequences(
        self,
        *,
        structure_path: pathlib.Path,
        job_name: str,
        chains_to_design: str | Sequence[str] | None = None,
        fixed_residues: Sequence[str] | None = None,
        parse_these_chains_only: str | Sequence[str] | None = None,
    ) -> Dict[str, object]:
        if self._closed:
            raise RuntimeError("NA-MPNN runner is closed")
        structure_path = pathlib.Path(structure_path).expanduser().resolve()
        if not structure_path.exists():
            raise FileNotFoundError(f"NA-MPNN input structure not found: {structure_path}")

        # A successful process that produced no new file must never consume a
        # stale FASTA/specificity result from an earlier scientific request.
        job_dir = pathlib.Path(tempfile.mkdtemp(prefix=self._safe_name(job_name) + "-", dir=self.output_root))

        cmd = [
            self.python,
            str(self.repo_path / "inference" / "run.py"),
            "--mode",
            "design",
            "--model_type",
            "na_mpnn",
            "--pdb_path",
            str(structure_path),
            "--out_folder",
            str(job_dir),
            "--checkpoint_na_mpnn",
            str(self.checkpoint),
            "--temperature",
            str(self.temperature),
            "--batch_size",
            str(self.batch_size),
            "--number_of_batches",
            str(self.number_of_batches),
            "--design_na_only",
            str(self.design_na_only),
            "--parse_na_only",
            str(self.parse_na_only),
            "--na_shared_tokens",
            str(self.na_shared_tokens),
            "--output_pdbs",
            str(self.output_pdbs),
            # Always output sequence + specificity when running as a mutator.
            "--output_sequences",
            "1",
            "--output_specificity",
            "1",
        ]

        if chains_to_design:
            cmd.extend(["--chains_to_design", self._normalize_chain_list(chains_to_design)])
        if fixed_residues:
            cmd.extend(["--fixed_residues", self._normalize_residue_list(fixed_residues)])
        if parse_these_chains_only:
            cmd.extend(
                [
                    "--parse_these_chains_only",
                    self._normalize_parse_chain_spec(parse_these_chains_only),
                ]
            )

        cmd.extend(self._extra_arguments())

        logger.info("Running NA-MPNN design for %s", structure_path.name)
        if self.daemon:
            self._infer_resident(cmd[2:])
        else:
            subprocess.run(cmd, check=True, cwd=str(self.repo_path), env=self._patched_env())

        expected_fasta = job_dir / "seqs" / f"{self._structure_basename(structure_path)}.fa"
        fasta_files = [expected_fasta]
        if not expected_fasta.exists():
            raise RuntimeError(f"NA-MPNN did not produce expected FASTA output: {expected_fasta}.")

        best_entry, best_fasta_path = self._select_best_entry(fasta_files)
        best_entry["job_dir"] = job_dir
        best_entry["fasta_path"] = best_fasta_path
        best_entry["probabilities"] = self._load_specificity_matrix(job_dir, structure_path)
        return best_entry

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get(self, key, default=None):
        if hasattr(self.args, key):
            value = getattr(self.args, key)
            if value is None:
                return default
            return value
        if isinstance(self.args, dict):
            value = self.args.get(key, default)
            if value is None:
                return default
            return value
        return default

    def _resolve_repo_path(self, path_value) -> pathlib.Path:
        if path_value is None:
            raise ValueError("NAMPNN `repo_path` must be provided in the configuration.")
        path = pathlib.Path(path_value).expanduser()
        if not path.is_absolute():
            path = (pathlib.Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"NAMPNN repository path '{path}' not found. "
                "Initialize vendor submodules (recommended) or point `na_mpnn.repo_path` to a valid checkout."
            )
        return path

    def _resolve_checkpoint(self, checkpoint_value):
        if checkpoint_value is None:
            checkpoint_value = self.repo_path / "models" / "design_model" / "s_19137.pt"
        checkpoint = pathlib.Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (self.repo_path / checkpoint).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"NAMPNN checkpoint file '{checkpoint}' does not exist.")
        return checkpoint

    def _patched_env(self):
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        repo_str = str(self.repo_path)
        env["PYTHONPATH"] = f"{repo_str}:{pythonpath}" if pythonpath else repo_str
        return env

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()) or "job"

    def _select_best_entry(
        self, fasta_files: List[pathlib.Path]
    ) -> tuple[Dict[str, object], pathlib.Path]:
        best = None
        best_score = float("-inf")
        best_path = None
        for fasta_path in fasta_files:
            for entry in self._parse_fasta(fasta_path):
                confidence = entry["metadata"].get("overall_confidence")
                if confidence is None:
                    continue
                try:
                    score = float(confidence)
                except (TypeError, ValueError):
                    continue
                if score > best_score:
                    best = entry
                    best_score = score
                    best_path = fasta_path
        if best is None or best_path is None:
            raise RuntimeError(
                "No designed sequences with `overall_confidence` found in NA-MPNN FASTA outputs."
            )
        return best, best_path

    def _parse_fasta(self, fasta_path: pathlib.Path) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        header = None
        seq_lines: List[str] = []
        with open(fasta_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        entries.append(
                            {
                                "header": header,
                                "sequence": "".join(seq_lines),
                                "metadata": self._parse_header(header),
                            }
                        )
                    header = line[1:].strip()
                    seq_lines = []
                else:
                    seq_lines.append(line)
        if header is not None:
            entries.append(
                {
                    "header": header,
                    "sequence": "".join(seq_lines),
                    "metadata": self._parse_header(header),
                }
            )
        return entries

    @staticmethod
    def _parse_header(header: str) -> Dict[str, object]:
        metadata: Dict[str, object] = {}
        parts = [part.strip() for part in header.split(",")]
        for part in parts[1:]:
            tokens = part.replace(",", " ").split()
            for token in tokens:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.strip().lower()
                value = value.strip().strip(",")
                try:
                    metadata[key] = float(value)
                except ValueError:
                    metadata[key] = value
        return metadata

    def _load_specificity_matrix(
        self, job_dir: pathlib.Path, structure_path: pathlib.Path
    ) -> Dict[str, object] | None:
        spec_dir = job_dir / "specificity"
        spec_path = spec_dir / f"{self._structure_basename(structure_path)}.npz"
        if not spec_path.exists():
            raise FileNotFoundError(f"NA-MPNN specificity output missing at {spec_path}.")
        data = np.load(spec_path, allow_pickle=True)
        bundle: Dict[str, object] = {}
        for key in data.files:
            value = data[key]
            if key == "restype_to_int":
                bundle[key] = value.item()
            elif key in {"encoded_residues", "chain_letters"}:
                bundle[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                bundle[key] = value.copy()
            else:
                bundle[key] = value
        return bundle

    @staticmethod
    def _structure_basename(structure_path: pathlib.Path) -> str:
        name = structure_path.name
        for suffix in (".pdb", ".cif", ".mmcif"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @staticmethod
    def _normalize_chain_list(value: str | Sequence[str]) -> str:
        if isinstance(value, str):
            return value
        return ",".join(str(item).strip() for item in value if str(item).strip())

    @staticmethod
    def _normalize_parse_chain_spec(value: str | Sequence[str]) -> str:
        if isinstance(value, str):
            return value
        return "".join(str(item).strip() for item in value if str(item).strip())

    @staticmethod
    def _normalize_residue_list(value: Sequence[str]) -> str:
        return " ".join(str(item).strip() for item in value if str(item).strip())
