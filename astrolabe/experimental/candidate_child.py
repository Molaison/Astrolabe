"""Independent process wrapper for one durable candidate execution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from omegaconf import OmegaConf

from astrolabe.design.runtime import ContractConflict
from astrolabe.experimental.candidate_batch_runner import (
    CHILD_RECEIPT_SCHEMA,
    CandidateExecutionSpec,
    CandidateResult,
    CandidateTask,
    ProcessCandidatePayload,
    ProcessCandidateReceipt,
    ProcessCandidateOutputReader,
    _candidate_content_id,
    _owned_process_alive,
    _process_identity,
    _read_candidate_receipt,
    _repo_root,
    _sha256_file,
    _write_candidate_receipt,
)


def _payload(receipt: ProcessCandidateReceipt) -> ProcessCandidatePayload:
    return ProcessCandidatePayload(
        config_yaml=receipt.config_yaml,
        out_dir=receipt.out_dir,
        task_name=receipt.task_name,
        stdout_log=receipt.stdout_log,
        stderr_log=receipt.stderr_log,
        sequences=dict(receipt.sequences),
    )


def _task(receipt: ProcessCandidateReceipt) -> CandidateTask:
    return CandidateTask(
        candidate_id=receipt.job_id,
        generation=receipt.generation,
        payload={"sequences": dict(receipt.sequences)},
    )


def _validate_admitted(
    receipt: ProcessCandidateReceipt,
    spec: CandidateExecutionSpec,
) -> None:
    if receipt.schema != CHILD_RECEIPT_SCHEMA:
        raise ContractConflict("Unsupported candidate receipt schema")
    if (
        receipt.job_id != spec.job_id
        or receipt.generation != spec.generation
        or receipt.attempt != spec.attempt
        or receipt.actual_seed != spec.actual_seed
        or receipt.content_id != spec.content_id
    ):
        raise ContractConflict("Candidate admission belongs to another request")
    loaded = OmegaConf.load(spec.config_yaml)
    final_config = OmegaConf.to_container(loaded, resolve=False)
    if final_config != receipt.final_config:
        raise ContractConflict("Candidate final config changed after admission")
    recomputed = _candidate_content_id(
        job_id=receipt.job_id,
        generation=receipt.generation,
        attempt=receipt.attempt,
        actual_seed=receipt.actual_seed,
        task_name=receipt.task_name,
        out_dir=receipt.out_dir,
        final_config=receipt.final_config,
        scientific_config=receipt.scientific_config,
        source_manifest=receipt.source_manifest,
        input_assets_manifest=receipt.input_assets_manifest,
    )
    if recomputed != receipt.content_id:
        raise ContractConflict("Candidate receipt content identity is inconsistent")


def _spec(receipt: ProcessCandidateReceipt) -> CandidateExecutionSpec:
    return CandidateExecutionSpec(
        job_id=receipt.job_id,
        generation=receipt.generation,
        attempt=receipt.attempt,
        actual_seed=receipt.actual_seed,
        task_name=receipt.task_name,
        out_dir=receipt.out_dir,
        work_root=str(Path(receipt.config_yaml).parent),
        config_yaml=receipt.config_yaml,
        stdout_log=receipt.stdout_log,
        stderr_log=receipt.stderr_log,
        receipt_path=str(Path(receipt.config_yaml).parent / "candidate_receipt.json"),
        sequences=dict(receipt.sequences),
        final_config=receipt.final_config,
        scientific_config=receipt.scientific_config,
        source_manifest=receipt.source_manifest,
        input_assets_manifest=receipt.input_assets_manifest,
        content_id=receipt.content_id,
    )


def _output_manifest(task_root: Path) -> tuple[dict[str, object], ...]:
    if not task_root.is_dir():
        return ()
    files = []
    for path in sorted(item for item in task_root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(task_root).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return tuple(files)


def _terminal(
    receipt: ProcessCandidateReceipt,
    *,
    status: str,
    result: CandidateResult | None,
    error: str | None,
    provenance: dict[str, object] | None,
    output_manifest: tuple[dict[str, object], ...] = (),
) -> ProcessCandidateReceipt:
    return ProcessCandidateReceipt(
        schema=receipt.schema,
        status=status,
        job_id=receipt.job_id,
        generation=receipt.generation,
        attempt=receipt.attempt,
        actual_seed=receipt.actual_seed,
        content_id=receipt.content_id,
        task_name=receipt.task_name,
        config_yaml=receipt.config_yaml,
        out_dir=receipt.out_dir,
        stdout_log=receipt.stdout_log,
        stderr_log=receipt.stderr_log,
        sequences=receipt.sequences,
        final_config=receipt.final_config,
        scientific_config=receipt.scientific_config,
        source_manifest=receipt.source_manifest,
        input_assets_manifest=receipt.input_assets_manifest,
        issued_process=receipt.issued_process,
        cli_provenance=provenance,
        output_manifest=output_manifest,
        result=result,
        error=error,
    )


def run(*, receipt_path: Path, runner: Path, timeout_s: float | None) -> int:
    receipt = _read_candidate_receipt(receipt_path)
    if receipt.status != "admitted":
        return 0
    _validate_admitted(receipt, _spec(receipt))

    issued = ProcessCandidateReceipt(
        schema=receipt.schema,
        status="issued",
        job_id=receipt.job_id,
        generation=receipt.generation,
        attempt=receipt.attempt,
        actual_seed=receipt.actual_seed,
        content_id=receipt.content_id,
        task_name=receipt.task_name,
        config_yaml=receipt.config_yaml,
        out_dir=receipt.out_dir,
        stdout_log=receipt.stdout_log,
        stderr_log=receipt.stderr_log,
        sequences=receipt.sequences,
        final_config=receipt.final_config,
        scientific_config=receipt.scientific_config,
        source_manifest=receipt.source_manifest,
        input_assets_manifest=receipt.input_assets_manifest,
        issued_process=_process_identity(),
    )
    # The issued proof is durable before the model child can be created.
    _write_candidate_receipt(receipt_path, issued)

    stdout_path = Path(receipt.stdout_log)
    stderr_path = Path(receipt.stderr_log)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(runner),
        "run",
        "--standalone",
        receipt.config_yaml,
        "--no-chdir",
    ]
    started = time.time()
    model_done = False
    model_identity = None
    model_env = os.environ.copy()
    capture_root = model_env.get("ASTROLABE_ACCEPTANCE_CAPTURE_ROOT")
    if capture_root:
        capture_dir = Path(capture_root) / receipt.job_id
        capture_dir.mkdir(parents=True, exist_ok=True)
        model_env["ASTROLABE_CANDIDATE_ID"] = receipt.job_id
        model_env["ASTROLABE_ACCEPTANCE_CAPTURE"] = str(
            capture_dir / "capture.pickle"
        )
        model_env["ASTROLABE_ACCEPTANCE_CLAIMS"] = str(capture_dir / "claims")
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=str(_repo_root()),
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
                env=model_env,
            )
            model_identity = _process_identity(process.pid)
            try:
                returncode = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                if _owned_process_alive(model_identity):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                provenance = {
                    "started_unix_s": started,
                    "finished_unix_s": time.time(),
                    "returncode": process.returncode,
                    "timed_out": True,
                    "model_process": model_identity,
                }
                _write_candidate_receipt(
                    receipt_path,
                    _terminal(
                        issued,
                        status="unknown",
                        result=None,
                        error="candidate child timeout; outcome remains charged",
                        provenance=provenance,
                    ),
                )
                return 0
            model_done = True
    except OSError as exc:
        if model_identity is not None:
            _write_candidate_receipt(
                receipt_path,
                _terminal(
                    issued,
                    status="unknown",
                    result=None,
                    error=f"candidate child process failed: {exc}",
                    provenance={
                        "started_unix_s": started,
                        "finished_unix_s": time.time(),
                        "timed_out": False,
                        "model_process": model_identity,
                    },
                ),
            )
        else:
            result = CandidateResult(
                task=_task(receipt),
                status="failed",
                error=f"candidate CLI could not start: {exc}",
                payload=_payload(receipt),
            )
            _write_candidate_receipt(
                receipt_path,
                _terminal(
                    issued,
                    status="failed",
                    result=result,
                    error=result.error,
                    provenance=None,
                ),
            )
        return 0
    except BaseException as exc:
        if model_identity is None:
            result = CandidateResult(
                task=_task(receipt),
                status="failed",
                error=f"candidate wrapper failed before issue: {type(exc).__name__}: {exc}",
                payload=_payload(receipt),
            )
            _write_candidate_receipt(
                receipt_path,
                _terminal(
                    issued,
                    status="failed",
                    result=result,
                    error=result.error,
                    provenance=None,
                ),
            )
        else:
            _write_candidate_receipt(
                receipt_path,
                _terminal(
                    issued,
                    status="unknown",
                    result=None,
                    error=f"candidate wrapper error: {type(exc).__name__}: {exc}",
                    provenance={
                        "started_unix_s": started,
                        "finished_unix_s": time.time(),
                        "timed_out": False,
                        "model_process": model_identity,
                    },
                ),
            )
        return 0

    provenance = {
        "started_unix_s": started,
        "finished_unix_s": time.time(),
        "returncode": returncode,
        "timed_out": False,
        "model_process": model_identity,
    }
    if not model_done or returncode != 0:
        result = CandidateResult(
            task=_task(receipt),
            status="failed",
            error=(
                f"candidate CLI failed rc={returncode}; "
                f"stderr={receipt.stderr_log}"
            ),
            payload=_payload(receipt),
        )
        _write_candidate_receipt(
            receipt_path,
            _terminal(
                issued,
                status="failed",
                result=result,
                error=result.error,
                provenance=provenance,
            ),
        )
        return 0
    try:
        score = ProcessCandidateOutputReader(Path(receipt.out_dir)).read_best_score(
            receipt.task_name
        )
        manifest = _output_manifest(Path(receipt.out_dir) / receipt.task_name)
        if not manifest:
            raise ValueError("completed candidate CLI produced no output manifest")
        result = CandidateResult(
            task=_task(receipt),
            status="completed",
            score=score.weighted_score,
            duration_s=float(provenance["finished_unix_s"]) - started,
            payload=ProcessCandidatePayload(
                config_yaml=receipt.config_yaml,
                out_dir=receipt.out_dir,
                task_name=receipt.task_name,
                stdout_log=receipt.stdout_log,
                stderr_log=receipt.stderr_log,
                sequences=score.sequences,
            ),
        )
        _write_candidate_receipt(
            receipt_path,
            _terminal(
                issued,
                status="completed",
                result=result,
                error=None,
                provenance=provenance,
                output_manifest=manifest,
            ),
        )
        return 0
    except BaseException as exc:
        _write_candidate_receipt(
            receipt_path,
            _terminal(
                issued,
                status="unknown",
                result=None,
                error=f"candidate output validation failed: {exc}",
                provenance=provenance,
            ),
        )
        return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrolabe-candidate-child")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--timeout-s", default="none")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    timeout_s = (
        None
        if str(args.timeout_s).casefold() == "none"
        else float(args.timeout_s)
    )
    return run(
        receipt_path=Path(args.receipt).expanduser().resolve(),
        runner=Path(args.runner).expanduser().resolve(),
        timeout_s=timeout_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
