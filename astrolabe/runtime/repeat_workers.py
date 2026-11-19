"""Run the application's repeat loop in one isolated process per selected GPU."""
from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import yaml
from omegaconf import OmegaConf

from astrolabe.design.repeats import RepeatRun
from .session import _resolved_devices, _terminate_group, _visibility_tokens
from .telemetry import NvidiaSMISampler


_ALLOCATION_MASKS = ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS")
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def gpu_selectors(raw: str) -> tuple[str, ...]:
    message = "--gpus expects a comma-separated list of GPU indices or GPU UUIDs (e.g. 0,1)."
    try:
        tokens = _visibility_tokens(raw)
    except ValueError as error:
        raise ValueError(message) from error
    if not tokens or any(not re.fullmatch(r"[0-9]+|GPU-[0-9a-fA-F-]+", token) for token in tokens):
        raise ValueError(message)
    tokens = tuple(str(int(token)) if token.isdecimal() else token for token in tokens)
    if len(set(tokens)) != len(tokens):
        raise ValueError("--gpus names the same GPU more than once.")
    return tokens


def validate_gpu_repeat_config(cfg, gpus: tuple[str, ...]) -> None:
    if OmegaConf.select(cfg, "execution.devices", default="allocated") != "allocated":
        raise ValueError("Use execution.devices=allocated with --gpus so each worker can own its assigned GPU.")
    if len(gpus) > 1 and os.environ.get("ASTROLABE_RESOURCE_SOCKET"):
        raise ValueError("--gpus with multiple GPUs requires separate worker allocations. "
                         "The inherited resource broker requires a shared CUDA mask.")


def _gpu_identities(selectors: tuple[str, ...]) -> list[str]:
    try:
        rows = NvidiaSMISampler()._query("--query-gpu=index,uuid")
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"Cannot read local GPU identities with nvidia-smi: {error}") from error
    inventory = {uuid.strip(): {"index": int(index), "uuid": uuid.strip()} for index, uuid in rows}
    try:
        selected = [device["uuid"] for device in _resolved_devices(selectors, inventory)]
    except ValueError as error:
        raise ValueError(f"--gpus {','.join(selectors)}: {error}. See nvidia-smi -L.") from error
    for key in _ALLOCATION_MASKS:
        if key in os.environ:
            try:
                mask = _visibility_tokens(os.environ[key])
                allowed = {device["uuid"] for device in _resolved_devices(mask, inventory)}
            except ValueError as error:
                raise ValueError(f"Cannot resolve the GPU allocation in {key}: {error}") from error
            denied = [token for token, uuid in zip(selectors, selected) if uuid not in allowed]
            if denied:
                raise ValueError(f"--gpus selects devices outside {key}: {', '.join(denied)}")
    return selected


def run_gpu_repeats(cfg, repeats: tuple[RepeatRun, ...], selectors: tuple[str, ...]) -> int:
    devices = _gpu_identities(selectors)
    source_root = str(Path(__file__).resolve().parents[2])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, env.get("PYTHONPATH"))))
    children: list[tuple[str, subprocess.Popen]] = []
    stop_signal = None
    force_stop = False

    def request_stop(signum, _frame):
        nonlocal stop_signal, force_stop
        if stop_signal is not None:
            if not force_stop:
                force_stop = True
                raise KeyboardInterrupt
            return
        stop_signal = signum
        for _, child in children:
            with suppress(ProcessLookupError):
                child.send_signal(signum)

    previous = {signum: signal.signal(signum, request_stop) for signum in _STOP_SIGNALS}
    failures = []
    try:
        # Transfer the composed config and selected repeat identities through
        # stdin. Children use the same Python runtime and working directory.
        raw_config = OmegaConf.to_container(cfg, resolve=False)
        for slot, device in enumerate(devices[:len(repeats)]):
            if stop_signal is not None:
                break
            work = []
            for repeat in repeats[slot::len(devices)]:
                entry = asdict(repeat)
                if repeat.output is not None:
                    entry["output"] = os.fspath(repeat.output)
                work.append(entry)
            payload = yaml.safe_dump({"config": raw_config, "repeats": work}, sort_keys=False)
            name = selectors[slot]
            child = subprocess.Popen(
                [sys.executable, "-P", "-u", "-m", "astrolabe.runtime.repeat_workers"],
                env=dict(env, CUDA_VISIBLE_DEVICES=device),
                stdin=subprocess.PIPE, text=True, encoding="utf-8", start_new_session=True,
            )
            children.append((name, child))
            if stop_signal is not None:
                with suppress(ProcessLookupError):
                    child.send_signal(stop_signal)
            with suppress(BrokenPipeError), child.stdin:
                child.stdin.write(payload)

        for name, child in children:
            if code := child.wait():
                _terminate_group(child)
                failures.append(f"GPU {name} (exit {code})")
    except BaseException as error:
        for _, child in children:
            try:
                _terminate_group(child)
            except BaseException as cleanup_error:
                error.add_note(f"Repeat worker {child.pid} cleanup failed: {cleanup_error}")
        interrupted = force_stop and isinstance(error, KeyboardInterrupt)
        if not interrupted or getattr(error, "__notes__", ()):
            raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    if failures:
        print(f"[astrolabe] Repeat workers failed: {', '.join(failures)}. "
              "Check the worker output and each run's logs.", file=sys.stderr)
    if stop_signal is not None:
        print("[astrolabe] GPU repeat batch interrupted.", file=sys.stderr)
        return 128 + stop_signal
    if failures:
        return 1
    print(f"[astrolabe] Completed {len(repeats)}/{len(repeats)} repeats on {len(children)} GPU(s).",
          file=sys.stderr)
    return 0


def _worker_main() -> None:
    from hydra.core.utils import setup_globals
    from astrolabe.cli import _run_from_config

    payload = yaml.safe_load(sys.stdin)
    setup_globals()
    cfg = OmegaConf.create(payload["config"])
    repeats = tuple(RepeatRun(**entry) for entry in payload["repeats"])
    _run_from_config(cfg, autoselect_gpu=False, planned_repeats=repeats)


if __name__ == "__main__":
    _worker_main()
