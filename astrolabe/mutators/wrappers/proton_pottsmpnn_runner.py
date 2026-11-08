from __future__ import annotations

import atexit
import json
import logging
import math
import operator
import os
import re
import selectors
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .proton_pottsmpnn_daemon import (
    _validated_encoded_binder_tokens,
    _validated_state_patterns,
)

_SHARED_RUNNER_LOCK = threading.RLock()
_SHARED_RUNNERS: dict[str, tuple[str, "ProtonPottsMPNNRunner"]] = {}
_SHARED_ATEXIT_REGISTERED = False


def _plain_config(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise ValueError("Proton-PottsMPNN shared runner config must be a mapping.")
    return dict(config)


def _register_runtime_owner(runner) -> None:
    """Bind lazy Proton resources to the active main run, when present."""
    from astrolabe.core import global_config

    try:
        owner = global_config.get_const("resource_owner")
    except RuntimeError:
        return  # Standalone posthoc callers own and close their runner.
    if owner is not None:
        owner.add(runner, "proton_pottsmpnn")


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _runner_is_alive(runner: "ProtonPottsMPNNRunner") -> bool:
    if runner._closed:
        return False
    if runner._started and runner._proc is not None:
        return runner._proc.poll() is None
    return True


def _unregister_shared_runner(runner: "ProtonPottsMPNNRunner") -> None:
    key = runner._shared_registry_key
    if not key:
        return
    with _SHARED_RUNNER_LOCK:
        registered = _SHARED_RUNNERS.get(key)
        if registered is not None and registered[1] is runner:
            _SHARED_RUNNERS.pop(key, None)


def _close_shared_runners_at_exit() -> None:
    with _SHARED_RUNNER_LOCK:
        runners = [entry[1] for entry in _SHARED_RUNNERS.values()]
        _SHARED_RUNNERS.clear()
    for runner in runners:
        try:
            runner._shared_registry_key = None
            runner.close()
        except Exception:
            pass


def get_shared_proton_runner(
    config: Mapping[str, Any] | Any | None = None,
    *,
    key: str = "default",
) -> "ProtonPottsMPNNRunner":
    """Return one live Proton runner per logical key.

    A config is required only to create the first runner. Repeating the same
    resolved config reuses it; a different config for the same key is rejected.
    Without a config, a live registered runner is returned or a clear error is
    raised. Closing the runner unregisters it, so a dead runner is never reused.
    """
    global _SHARED_ATEXIT_REGISTERED
    registry_key = str(key or "").strip()
    if not registry_key:
        raise ValueError("Proton-PottsMPNN shared runner key must be nonempty.")
    with _SHARED_RUNNER_LOCK:
        existing = _SHARED_RUNNERS.get(registry_key)
        if existing is not None:
            fingerprint, runner = existing
            if _runner_is_alive(runner):
                if config is None:
                    _register_runtime_owner(runner)
                    return runner
                candidate = _plain_config(config)
                if _config_fingerprint(candidate) != fingerprint:
                    raise ValueError(
                        "Conflicting Proton-PottsMPNN config for shared runner "
                        f"key {registry_key!r}; close the existing runner before "
                        "registering different settings."
                    )
                _register_runtime_owner(runner)
                return runner
            _SHARED_RUNNERS.pop(registry_key, None)
        if config is None:
            raise RuntimeError(
                "No live shared Proton-PottsMPNN runner is registered for "
                f"key {registry_key!r}; initialize it first with a config."
            )
        candidate = _plain_config(config)
        runner = ProtonPottsMPNNRunner(candidate)
        runner._shared_registry_key = registry_key
        _SHARED_RUNNERS[registry_key] = (
            _config_fingerprint(candidate),
            runner,
        )
        if not _SHARED_ATEXIT_REGISTERED:
            atexit.register(_close_shared_runners_at_exit)
            _SHARED_ATEXIT_REGISTERED = True
        return runner


def _timing(phase: str, started: float, proc, request_id=None) -> None:
    if os.environ.get("GH_PROTON_TIMING") == "1":
        logging.getLogger(__name__).info(
            "event=proton_timing phase=%s duration_ms=%.3f epoch=%.6f pid=%s request_id=%s",
            phase, (time.perf_counter() - started) * 1000, time.time(),
            proc.pid, request_id,
        )


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


_CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _canonical_sequence(value: Any, label: str) -> str:
    sequence = str(value or "").strip().upper()
    invalid = sorted(set(sequence) - _CANONICAL_AA)
    if not sequence:
        raise ValueError(f"Proton-PottsMPNN {label} must be nonempty.")
    if invalid:
        raise ValueError(
            f"Proton-PottsMPNN {label} must contain canonical amino acids only; "
            f"invalid={invalid}."
        )
    return sequence


def _validated_histidine_positions(
    binder_sequence: str, histidine_positions: Any
) -> list[int]:
    if not isinstance(histidine_positions, (list, tuple)):
        raise ValueError(
            "Proton-PottsMPNN histidine_positions must be a nonempty list of "
            "0-based binder positions."
        )
    positions: list[int] = []
    for raw in histidine_positions:
        if isinstance(raw, bool):
            raise ValueError("Proton-PottsMPNN histidine_positions cannot contain booleans.")
        try:
            position = operator.index(raw)
        except TypeError as exc:
            raise ValueError(
                "Proton-PottsMPNN histidine_positions must contain integers."
            ) from exc
        if position < 0 or position >= len(binder_sequence):
            raise ValueError(
                "Proton-PottsMPNN histidine_positions must be within binder "
                f"sequence length {len(binder_sequence)}; got {position}."
            )
        if binder_sequence[position] != "H":
            raise ValueError(
                "Proton-PottsMPNN histidine_positions must identify His residues "
                f"in binder_sequence; position {position} is "
                f"{binder_sequence[position]!r}."
            )
        positions.append(position)
    if not positions:
        raise ValueError(
            "Proton-PottsMPNN histidine_positions must contain at least one position."
        )
    if len(positions) != len(set(positions)):
        raise ValueError("Proton-PottsMPNN histidine_positions must be unique.")
    return positions


class ProtonPottsMPNNRunner:
    """Persistent JSONL adapter for the pinned Proton-PottsMPNN design engine."""

    _ENGINE_REL = (
        "foundry/models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py"
    )
    _DEFAULT_CHECKPOINT_REL = (
        "checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt"
    )

    def __init__(self, config: Mapping[str, Any] | Any):
        cfg = _plain_config(config or {})
        self._config_fingerprint = _config_fingerprint(cfg)
        self._shared_registry_key: str | None = None
        if not _as_bool(cfg.get("daemon", True)):
            raise ValueError(
                "Proton-PottsMPNN only supports persistent daemon execution; "
                "per-call model loading is disabled."
            )

        self.python_bin = self._required_executable(
            cfg.get("python_bin"), "python_bin"
        )
        self.repo_path = self._required_dir(cfg.get("repo_path"), "repo_path")
        engine_path = self.repo_path / self._ENGINE_REL
        if not engine_path.is_file():
            raise FileNotFoundError(
                f"Proton-PottsMPNN engine source does not exist: {engine_path}"
            )
        self.checkpoint_path = self._resolve_checkpoint(cfg.get("checkpoint_path"))
        self.hbplus_bin = self._required_executable(
            cfg.get("hbplus_bin"), "hbplus_bin"
        )

        self.extended_vocab = str(cfg.get("extended_vocab") or "v6").strip()
        if self.extended_vocab != "v6":
            raise ValueError(
                "Proton-PottsMPNN requires extended_vocab='v6' for the pinned "
                f"30-token checkpoint; got {self.extended_vocab!r}."
            )
        self.device = str(cfg.get("device") or "auto").strip()
        if re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device) is None:
            raise ValueError(
                "Proton-PottsMPNN device must be auto, cpu, cuda, or cuda:<index>; "
                f"got {self.device!r}."
            )
        self.design_mask_adapter = str(cfg.get("design_mask_adapter") or "none")
        if self.design_mask_adapter not in {"none", "preserve"}:
            raise ValueError("design_mask_adapter must be 'none' or 'preserve'.")

        self.seed = int(cfg.get("seed", 0) if cfg.get("seed") is not None else 0)
        self.n_jobs = int(cfg.get("n_jobs", 1) or 1)
        if self.n_jobs < 1:
            raise ValueError("Proton-PottsMPNN n_jobs must be at least 1.")

        output_dir = cfg.get("output_dir") or "proton_pottsmpnn_outputs"
        self.output_dir = self._resolve_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout = float(cfg.get("startup_timeout", 600.0))
        request_timeout = cfg.get("request_timeout")
        self.request_timeout = (
            None
            if request_timeout in (None, "", "None", "null")
            else float(request_timeout)
        )
        daemon_script = cfg.get("daemon_script_path")
        if daemon_script is None:
            daemon_script = Path(__file__).with_name("proton_pottsmpnn_daemon.py")
        self.daemon_script_path = self._required_file(
            daemon_script, "daemon_script_path"
        )

        self._proc: subprocess.Popen[str] | None = None
        self._started = False
        self._closed = False
        self._request_index = 0
        self._lock = threading.RLock()
        _register_runtime_owner(self)

    @staticmethod
    def _resolve_path(value: str | os.PathLike[str]) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    @classmethod
    def _required_file(cls, value: Any, name: str) -> Path:
        if value in (None, "", "None", "null", "???"):
            raise ValueError(f"Proton-PottsMPNN {name} must be configured.")
        path = cls._resolve_path(value)
        if not path.is_file():
            raise FileNotFoundError(
                f"Proton-PottsMPNN {name} does not exist: {path}"
            )
        return path

    @classmethod
    def _required_executable(cls, value: Any, name: str) -> Path:
        path = cls._required_file(value, name)
        if not os.access(path, os.X_OK):
            raise PermissionError(
                f"Proton-PottsMPNN {name} is not executable: {path}"
            )
        return path

    @classmethod
    def _required_dir(cls, value: Any, name: str) -> Path:
        if value in (None, "", "None", "null", "???"):
            raise ValueError(f"Proton-PottsMPNN {name} must be configured.")
        path = cls._resolve_path(value)
        if not path.is_dir():
            raise FileNotFoundError(
                f"Proton-PottsMPNN {name} does not exist: {path}"
            )
        return path

    def _resolve_checkpoint(self, value: Any) -> Path:
        if value in (None, "", "None", "null", "???"):
            value = self.repo_path / self._DEFAULT_CHECKPOINT_REL
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            cwd_path = (Path.cwd() / path).resolve()
            path = cwd_path if cwd_path.exists() else (self.repo_path / path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Proton-PottsMPNN checkpoint_path does not exist: {path}"
            )
        return path

    def _command(self) -> list[str]:
        command = [
            str(self.python_bin),
            "-u",
            str(self.daemon_script_path),
            "--repo-path",
            str(self.repo_path),
            "--checkpoint",
            str(self.checkpoint_path),
            "--hbplus-bin",
            str(self.hbplus_bin),
            "--extended-vocab",
            self.extended_vocab,
            "--device",
            self.device,
            "--output-dir",
            str(self.output_dir),
        ]
        if getattr(self, "design_mask_adapter", "none") == "preserve":
            command.extend(["--design-mask-adapter", "preserve"])
        return command

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("APPTAINERENV_PYTHONPATH", None)
        isolated_pythonpath = env.get("GH_PROTON_POTTSMPNN_PYTHONPATH", "").strip()
        if isolated_pythonpath:
            env["PYTHONPATH"] = isolated_pythonpath
        return env

    def _ensure_daemon(self) -> subprocess.Popen[str]:
        if self._closed:
            raise RuntimeError("Proton-PottsMPNNRunner is closed.")
        if self._proc is not None:
            if self._proc.poll() is None:
                return self._proc
            raise RuntimeError(
                "Proton-PottsMPNN daemon is no longer running; "
                "implicit model restart is disabled."
            )
        if self._started:
            raise RuntimeError(
                "Proton-PottsMPNN daemon is no longer running; "
                "implicit model restart is disabled."
            )

        started = time.perf_counter()
        proc = subprocess.Popen(
            self._command(),
            cwd=str(self.repo_path),
            env=self._subprocess_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._proc = proc
        self._started = True
        try:
            ready = self._read_response(proc, self.startup_timeout)
        except Exception:
            proc.terminate()
            proc.wait()
            raise
        if ready.get("type") != "ready" or ready.get("model_load_count") != 1:
            proc.terminate()
            proc.wait()
            raise RuntimeError(
                f"Unexpected Proton-PottsMPNN daemon startup response: {ready}"
            )
        _timing("daemon_ready", started, proc)
        return proc

    @staticmethod
    def _read_response(
        proc: subprocess.Popen[str], timeout: float | None
    ) -> dict[str, Any]:
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(timeout):
                raise TimeoutError(
                    "Timed out waiting for Proton-PottsMPNN daemon response."
                )
        finally:
            selector.close()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(
                "Proton-PottsMPNN daemon exited unexpectedly with code "
                f"{proc.poll()}."
            )
        return json.loads(line)

    @staticmethod
    def _write_request(proc: subprocess.Popen[str], payload: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        proc.stdin.flush()

    @staticmethod
    def _plain_mapping(value: Any, name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if not isinstance(value, Mapping):
            raise ValueError(f"Proton-PottsMPNN {name} must be a mapping.")
        return dict(value)

    def design_sequences(self, **features: Any) -> dict[str, Any]:
        pdb_path = self._required_file(features.get("pdb_path"), "pdb_path")
        if pdb_path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError(
                "Proton-PottsMPNN requires evaluator structures in PDB or mmCIF format."
            )
        binder_chain = str(features.get("binder_chain") or "").strip()
        if not binder_chain:
            raise ValueError("Proton-PottsMPNN binder_chain must be configured.")
        binder_sequence = str(features.get("binder_sequence") or "").strip().upper()
        if not binder_sequence:
            raise ValueError("Proton-PottsMPNN binder_sequence must be provided.")
        chain_sequences = self._plain_mapping(
            features.get("chain_sequences"), "chain_sequences"
        )
        if binder_chain not in {str(k) for k in chain_sequences}:
            raise ValueError(
                f"Proton-PottsMPNN chain_sequences is missing binder chain {binder_chain!r}."
            )
        fixed_positions = sorted(
            {int(p) for p in (features.get("fixed_positions") or [])}
        )
        if any(p < 0 or p >= len(binder_sequence) for p in fixed_positions):
            raise ValueError(
                "Proton-PottsMPNN fixed_positions must be 0-based and within the "
                f"binder sequence length {len(binder_sequence)}."
            )
        max_mutations = int(features.get("max_mutations") or 0)
        if max_mutations <= 0:
            raise ValueError("Proton-PottsMPNN max_mutations must be greater than zero.")
        criteria = self._plain_mapping(features.get("criteria"), "criteria")
        n_jobs = int(features.get("n_jobs", self.n_jobs) or self.n_jobs)

        request_id = uuid.uuid4().hex
        with self._lock:
            payload = {
                "type": "infer",
                "id": request_id,
                "pdb_path": str(pdb_path),
                "binder_chain": binder_chain,
                "binder_sequence": binder_sequence,
                "chain_sequences": {
                    str(k): str(v).strip().upper()
                    for k, v in chain_sequences.items()
                },
                "fixed_positions": fixed_positions,
                "criteria": criteria,
                "max_mutations": max_mutations,
                "n_jobs": n_jobs,
                "seed": self.seed + self._request_index,
            }
            self._request_index += 1
            proc = self._ensure_daemon()
            started = time.perf_counter()
            self._write_request(proc, payload)
            response = self._read_response(proc, self.request_timeout)
            _timing("request_roundtrip", started, proc, request_id)
        if response.get("type") == "error" and response.get("id") == request_id:
            raise RuntimeError(
                "Proton-PottsMPNN daemon error: "
                f"{response.get('error_type')}: {response.get('error')}"
            )
        if response.get("type") != "result" or response.get("id") != request_id:
            raise RuntimeError(f"Unexpected Proton-PottsMPNN daemon response: {response}")
        selected = response.get("selected")
        if not isinstance(selected, Mapping):
            raise RuntimeError("Proton-PottsMPNN daemon returned no selected design.")
        canonical_sequence = str(selected.get("canonical_sequence") or "").strip()
        if not canonical_sequence:
            raise RuntimeError(
                "Proton-PottsMPNN selected design is missing canonical_sequence."
            )
        if len(canonical_sequence) != len(binder_sequence):
            raise RuntimeError(
                "Proton-PottsMPNN selected canonical_sequence length "
                f"{len(canonical_sequence)} does not match binder length "
                f"{len(binder_sequence)}."
            )
        return {
            "selected": dict(selected),
            "designs_considered": int(response.get("designs_considered") or 0),
            "raw_output_path": response.get("raw_output_path"),
            "seed_energies": response.get("seed_energies"),
            "request_id": request_id,
            "daemon_pid": response.get("pid"),
            "model_load_count": response.get("model_load_count"),
            "request_count": response.get("request_count"),
        }

    def score_states(
        self,
        *,
        pdb_path: str | os.PathLike[str],
        binder_chain: str,
        binder_sequence: str,
        chain_sequences: Mapping[str, str],
        histidine_positions: list[int],
        state_patterns: list[str | list[str]] | Mapping[str, str | list[str]] | None = None,
        encoded_binder_tokens: list[str] | None = None,
    ) -> dict[str, Any]:
        """Score one all-His neutral/protonated pair without redesigning.

        ``histidine_positions`` are 0-based binder-sequence positions. Every
        selected residue is set to HIS-S in one clone and HIS-P in the other;
        all other encoded states remain identical.

        Explicit ``state_patterns`` are S/P strings or S/P lists, optionally a
        mapping of names to patterns (<=16 unique patterns, <=4 centers). Their
        order follows ``histidine_positions``. ``encoded_binder_tokens`` replaces
        the full encoded binder on this same fixed-geometry context; it does not
        change the structure-validation identity ``binder_sequence``. Energies
        are not probabilities/pKa or physical geometry validation. Legacy gap
        remains E(all-P)-E(all-S), including when explicit states are supplied.
        """
        path = self._required_file(pdb_path, "pdb_path")
        if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError(
                "Proton-PottsMPNN score_states requires PDB or mmCIF input."
            )
        chain = str(binder_chain or "").strip()
        if not chain:
            raise ValueError("Proton-PottsMPNN score_states binder_chain must be configured.")
        sequence = _canonical_sequence(binder_sequence, "binder_sequence")
        chain_map = self._plain_mapping(chain_sequences, "chain_sequences")
        normalized_chains = {
            str(chain_id): _canonical_sequence(sequence_value, f"chain_sequences[{chain_id!r}]")
            for chain_id, sequence_value in chain_map.items()
        }
        if chain not in normalized_chains:
            raise ValueError(
                f"Proton-PottsMPNN score_states chain_sequences is missing binder "
                f"chain {chain!r}."
            )
        if normalized_chains[chain] != sequence:
            raise ValueError(
                "Proton-PottsMPNN score_states binder_sequence does not match "
                f"chain_sequences[{chain!r}]."
            )
        encoded_tokens, scored_sequence = _validated_encoded_binder_tokens(encoded_binder_tokens, sequence)
        positions = _validated_histidine_positions(scored_sequence, histidine_positions)
        pattern_input = dict(state_patterns) if isinstance(state_patterns, Mapping) else state_patterns
        patterns = _validated_state_patterns(pattern_input, len(positions))
        audit = patterns is not None or encoded_tokens is not None

        request_id = uuid.uuid4().hex
        with self._lock:
            payload = {
                "type": "score_states",
                "id": request_id,
                "pdb_path": str(path),
                "binder_chain": chain,
                "binder_sequence": sequence,
                "chain_sequences": normalized_chains,
                "histidine_positions": positions,
            }
            if audit:
                if patterns is not None:
                    payload["state_patterns"] = pattern_input
                if encoded_tokens is not None:
                    payload["encoded_binder_tokens"] = encoded_tokens
                payload["audit_parameters"] = {
                    "repo_path": str(self.repo_path),
                    "checkpoint_path": str(self.checkpoint_path),
                    "extended_vocab": self.extended_vocab,
                    "device": self.device,
                }
            proc = self._ensure_daemon()
            started = time.perf_counter()
            self._write_request(proc, payload)
            response = self._read_response(proc, self.request_timeout)
            _timing("score_states_roundtrip", started, proc, request_id)

        if response.get("type") == "error" and response.get("id") == request_id:
            raise RuntimeError(
                "Proton-PottsMPNN score_states daemon error: "
                f"{response.get('error_type')}: {response.get('error')}"
            )
        if response.get("type") != "result" or response.get("id") != request_id:
            raise RuntimeError(
                f"Unexpected Proton-PottsMPNN score_states response: {response}"
            )
        neutral = response.get("neutral_energy")
        protonated = response.get("protonated_energy")
        gap = response.get("gap")
        numeric = {}
        for label, value in (
            ("neutral_energy", neutral),
            ("protonated_energy", protonated),
            ("gap", gap),
        ):
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Proton-PottsMPNN score_states returned invalid {label}: {value!r}"
                ) from exc
            if not math.isfinite(parsed):
                raise RuntimeError(
                    f"Proton-PottsMPNN score_states returned non-finite {label}: {value!r}"
                )
            numeric[label] = parsed
        if not math.isclose(numeric["gap"], numeric["protonated_energy"] - numeric["neutral_energy"], rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError(
                "Proton-PottsMPNN score_states returned inconsistent gap: "
                f"{numeric['gap']} != {numeric['protonated_energy']} - {numeric['neutral_energy']}"
            )
        if response.get("n_centers") != len(positions):
            raise RuntimeError(
                "Proton-PottsMPNN score_states returned wrong n_centers: "
                f"{response.get('n_centers')} != {len(positions)}"
            )
        if response.get("histidine_positions") != positions:
            raise RuntimeError(
                "Proton-PottsMPNN score_states returned different histidine_positions."
            )
        if response.get("binder_sequence") != sequence:
            raise RuntimeError(
                "Proton-PottsMPNN score_states returned different binder_sequence."
            )
        if response.get("outside_centers_unchanged") is not True:
            raise RuntimeError(
                "Proton-PottsMPNN score_states did not preserve non-centre states."
            )
        raw_output_path = response.get("raw_output_path")
        if not raw_output_path or not Path(raw_output_path).is_file():
            raise RuntimeError(
                "Proton-PottsMPNN score_states returned no existing raw_output_path."
            )
        audit_fields = {}
        if audit:
            requested = patterns if patterns is not None else [
                {"name": "S" * len(positions), "pattern": "S" * len(positions)},
                {"name": "P" * len(positions), "pattern": "P" * len(positions)},
            ]
            # Treat the daemon as a protocol boundary: reject a partial or
            # different-context audit rather than passing it to scientific code.
            try:
                if (response["same_context"] is not True
                        or response["audit_source"] != "fixed_geometry_potts_context"
                        or response["energy_method"] != "ctx.scorer.H_of"
                        or response["scored_binder_sequence"] != scored_sequence
                        or response["state_names"] != [spec["name"] for spec in requested]
                        or response["state_patterns"] != requested):
                    raise ValueError("identity, patterns, or same context mismatch")
                baseline = response["scored_encoded_sequence"]
                native = response["native_encoded_sequence"]
                vocab = response["token_to_idx"]
                binder_indices = response["context_binder_positions"]
                centers = response["context_histidine_positions"]
                if (not isinstance(baseline, list) or not baseline
                        or len(native) != len(baseline)
                        or len(binder_indices) != len(sequence)
                        or len(set(binder_indices)) != len(binder_indices)
                        or any(type(p) is not int or not 0 <= p < len(baseline) for p in binder_indices)
                        or centers != [binder_indices[p] for p in positions]):
                    raise ValueError("context mapping mismatch")
                noncenters = [p for p in range(len(baseline)) if p not in centers]
                if response["noncenter_context_positions"] != noncenters:
                    raise ValueError("noncenter mapping mismatch")

                def check_tokens(values, tokens):
                    if (len(values) != len(baseline) or len(tokens) != len(baseline)
                            or any(type(v) is not int or vocab[t] != v for v, t in zip(values, tokens))):
                        raise ValueError("encoded tokens mismatch")

                check_tokens(baseline, response["scored_tokens"])
                check_tokens(native, response["native_tokens"])
                binder_tokens = [response["scored_tokens"][p] for p in binder_indices]
                if response["scored_binder_tokens"] != binder_tokens:
                    raise ValueError("binder tokens mismatch")
                if encoded_tokens is not None:
                    if binder_tokens != encoded_tokens:
                        raise ValueError("encoded override mismatch")
                    if any(baseline[p] != native[p] for p in range(len(baseline)) if p not in set(binder_indices)):
                        raise ValueError("override changed nonbinder states")
                elif baseline != native:
                    raise ValueError("baseline changed native states")
                states = response["states"]
                if not isinstance(states, list) or len(states) != len(requested):
                    raise ValueError("incomplete states")
                for state, spec in zip(states, requested):
                    if state["name"] != spec["name"] or state["pattern"] != spec["pattern"]:
                        raise ValueError("state identity mismatch")
                    center_tokens = [f"HIS-{s}" for s in spec["pattern"]]
                    expected = list(baseline)
                    for p, token in zip(centers, center_tokens):
                        expected[p] = vocab[token]
                    if state["encoded_sequence"] != expected or state["center_tokens"] != center_tokens:
                        raise ValueError("state changed noncenters or selected states")
                    check_tokens(state["encoded_sequence"], state["tokens"])
                    if type(state["energy"]) not in (int, float) or not math.isfinite(state["energy"]):
                        raise ValueError("non-finite or invalid state energy")
                    endpoint = {"S" * len(positions): numeric["neutral_energy"],
                                "P" * len(positions): numeric["protonated_energy"]}
                    if spec["pattern"] in endpoint and state["energy"] != endpoint[spec["pattern"]]:
                        raise ValueError("legacy endpoint energy mismatch")
                audit_fields = {key: response[key] for key in (
                    "state_names", "states", "state_patterns", "scored_binder_sequence",
                    "context_binder_positions", "context_histidine_positions",
                    "scored_encoded_sequence", "scored_tokens", "scored_binder_tokens",
                    "native_encoded_sequence", "native_tokens", "token_to_idx",
                    "noncenter_context_positions", "same_context", "audit_source",
                    "energy_method", "energy_semantics",
                )}
            except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
                raise RuntimeError(f"Proton-PottsMPNN score_states invalid state audit: {exc}") from exc
        return {
            **numeric,
            "n_centers": len(positions),
            "histidine_positions": positions,
            "binder_sequence": sequence,
            "outside_centers_unchanged": True,
            "raw_output_path": raw_output_path,
            "request_id": request_id,
            "daemon_pid": response.get("pid"),
            "model_load_count": response.get("model_load_count"),
            "request_count": response.get("request_count"),
            **audit_fields,
        }

    def close(self) -> None:
        if self._closed:
            _unregister_shared_runner(self)
            return
        proc = None
        try:
            with self._lock:
                proc = self._proc
                if proc is not None:
                    if proc.poll() is not None:
                        self._closed = True
                        raise RuntimeError(
                            "Proton-PottsMPNN daemon exited before shutdown; "
                            f"return code {proc.returncode}."
                        )
                    self._write_request(proc, {"type": "shutdown"})
                    response = self._read_response(proc, 30.0)
                    if response.get("type") != "ok":
                        raise RuntimeError(
                            "Unexpected Proton-PottsMPNN daemon shutdown response: "
                            f"{response}"
                        )
                    return_code = proc.wait(timeout=30.0)
                    if return_code != 0:
                        raise RuntimeError(
                            "Proton-PottsMPNN daemon shutdown returned code "
                            f"{return_code}."
                        )
                self._closed = True
        finally:
            if self._closed or proc is None or proc.poll() is not None:
                _unregister_shared_runner(self)

    shutdown = close
