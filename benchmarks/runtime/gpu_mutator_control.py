#!/usr/bin/env python3
"""Frozen, bounded SAME-revision pretrained ProteinMPNN GPU mode control.

``freeze`` is CPU-only; ``run`` requires a new single-GPU Slurm allocation.
The actual public build_runner/design_sequences/close methods are used. The
private client handle is read only to observe release, never to execute work.
Raw output and RNG trees use Astrolabe's existing non-pickle tree codec.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_PROTOCOL = Path(__file__).with_name("workloads") / "proteinmpnn_gpu_control.json"
BROKER_ENV = ("ASTROLABE_EXECUTION_ENDPOINT", "ASTROLABE_EXECUTION_TOKEN", "ASTROLABE_MUTATOR_WORKER")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def trial_plan(protocol):
    return [{"index": i * 2 + j, "group": i, "seed": seed, "mode": mode,
             "id": f"g{i:02d}-seed{seed}-{mode}"}
            for i, (seed, modes) in enumerate(zip(protocol["groups"], protocol["mode_order"]))
            for j, mode in enumerate(modes)]


def validate_protocol(protocol):
    """Reject silent reductions, omitted native kwargs and changed science."""
    require(protocol["schema"] == 1, "unknown protocol schema")
    require(protocol["groups"] == [17, 18, 19, 20, 21], "seed schedule changed")
    orders = [["legacy", "resident"] if i % 2 == 0 else ["resident", "legacy"] for i in range(5)]
    require(protocol["mode_order"] == orders, "interleaved mode order changed")
    require(protocol["requests_per_trial"] == 3 and protocol["warmup_requests"] == 0,
            "request count/warmup changed")
    require(protocol["assets"]["pdb"]["residues"] == 354, "full input required")
    config, decode = protocol["constructor"], protocol["decode"]
    require(config["model_type"] == "protein_mpnn" and config["number_of_batches"] == 1,
            "only the frozen ProteinMPNN checkpoint and batch are certified")
    require(config["omit_AA"] == ["C"] and config["temperature"] == 0.3,
            "sampling protocol changed")
    require(config["parse_atoms_with_zero_occupancy"] == 1, "parser protocol changed")
    require(decode["parse_these_chains_only"] == ["A", "B"] and decode["chains_to_design"] == "A,B",
            "chain parser requires a real list and both full chains")
    require(decode["batch_size"] == 1 and decode["distribution_semantics"] == "legacy_order_conditioned_marginals",
            "native decode semantics changed")
    require(not decode["return_joint_sequence"] and decode["reference_decoding_ranks"] is None,
            "return/decode contract changed")
    require(not any(decode[key] for key in ("fixed_residues", "symmetry_residues", "symmetry_weights",
            "fixed_residues_multi", "pdb_path_multi", "redesigned_residues_multi", "redesigned_residues",
            "bias_AA_per_residue_multi", "bias_AA_per_residue", "omit_AA_per_residue_multi",
            "omit_AA_per_residue", "homo_oligomer")), "this control designs all positions without ties or per-residue edits")
    tree = ast.parse((ROOT / "astrolabe/mutators/wrappers/ligandmpnn_run.py").read_text())
    native = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "design_sequences")
    names = {node.arg for node in native.args.args if node.arg != "self"}
    require(set(decode) == names, f"native kwargs must be explicit: {sorted(names ^ set(decode))}")
    gates = protocol["scientific_gates"]
    require((gates["rtol"], gates["atol"], gates["probability_sum_atol"]) == (1e-5, 1e-6, 1e-5),
            "scientific tolerances changed")
    budget = protocol["budget"]
    require(budget["gpus"] == 1 and budget["allocation_seconds"] <= 900 and
            budget["maximum_gpu_hours"] <= .25 and 0 < budget["experiment_seconds"] <= 840 and
            0 < budget["trial_seconds"] <= 150, "allocation/time ceiling exceeded")
    require(budget["max_output_bytes"] == 260000000 and budget["telemetry_sample_seconds"] == .5,
            "output/telemetry budget changed")


def verify_file(path, expected):
    path = Path(path).resolve()
    require(path.is_file(), f"asset missing: {path}")
    require(path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"],
            f"asset size/hash mismatch: {path}")
    return str(path)


def require_private_allocation(environment):
    require(not any(environment.get(key) for key in BROKER_ENV), "requires a fresh private runner, without broker/worker credentials")
    require(bool(environment.get("SLURM_JOB_ID")), "GPU run requires a dedicated Slurm allocation")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    require(bool(visible) and len(visible.split(",")) == 1 and visible not in ("all", "-1", "void"),
            "exactly one scheduler-bound CUDA device is required")
    require(environment.get("NVIDIA_VISIBLE_DEVICES", "") != "all", "all-device container visibility is forbidden")
    allocated = environment.get("ASTROLABE_ALLOCATED_CUDA_DEVICES")
    require(not allocated or allocated == visible, "runtime allocation differs from scheduler binding")


def require_gpu_uuid(value):
    require(isinstance(value, str) and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value),
            "allocation must supply one actual GPU UUID, verified by the resource owner")


def validate_execution_prefix(plan, rows):
    require(isinstance(rows, list) and len(rows) <= len(plan), "unexpected extra trials")
    for expected, actual in zip(plan, rows):
        require(isinstance(actual, dict) and all(type(actual.get(key)) is type(value) and actual.get(key) == value
                                                for key, value in expected.items()),
                "trial identity is not the strict frozen ordered prefix (duplicate, reorder or foreign trial)")
        seconds = actual.get("wall_seconds")
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
                "trial wall time must be positive and finite")


def constructor_config(protocol, *, checkpoint, out_folder, seed):
    config = copy.deepcopy(protocol["constructor"])
    config.update(checkpoint_protein_mpnn=str(checkpoint), out_folder=str(out_folder), seed=seed)
    return config


def freeze(args):
    """Read real weights/structure on CPU before the GPU allocation begins."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "freeze must run in an explicitly GPU-hidden CPU environment")
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    require(not git("diff", "HEAD", "--name-only"), "freeze requires a clean tracked candidate")
    output = args.output.resolve()
    require(not output.exists(), "freeze output must be new; existing evidence is never overwritten")
    assets = {key: verify_file(getattr(args, key) or item["path"], item)
              for key, item in protocol["assets"].items()}
    for relative, digest in protocol["native_sources"].items():
        require(sha256(ROOT / relative) == digest, f"audited native source changed: {relative}")
    output.mkdir(parents=True)
    shutil.copyfile(assets["pdb"], output / "input.pdb")
    assets["pdb"] = verify_file(output / "input.pdb", protocol["assets"]["pdb"])
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from astrolabe.evaluators.torch_rng import capture_settings
    from astrolabe.evaluators.tree_codec import dump_tree
    from astrolabe.mutators.wrappers.ligandmpnn_run import LigandMPNN
    from astrolabe.mutators.wrappers.ligandmpnn_utils import alphabet, parse_PDB
    require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), "CPU preflight must not initialize/use CUDA")
    require(capture_settings() == protocol["numerical_settings"], "actual framework numerical defaults differ from protocol")
    parsed, _, _, icodes, _ = parse_PDB(assets["pdb"], device="cpu", chains=["A", "B"],
                                      parse_all_atoms=False, parse_atoms_with_zero_occupancy=True)
    expected = protocol["assets"]["pdb"]
    chain_order = [chain for chain, count in expected["chain_order"] for _ in range(count)]
    require(list(parsed["chain_letters"]) == chain_order, "native parsed chain order/length changed")
    require(dict(Counter(chain_order)) == expected["chain_counts"], "chain counts changed")
    require(tuple(parsed["X"].shape) == (354, 4, 3) and bool((parsed["mask"] == 1).all()),
            "full finite-backbone mask is required")
    native_sequence = "".join(alphabet[int(index)] for index in parsed["S"])
    require(hashlib.sha256(native_sequence.encode()).hexdigest() == expected["native_sequence_sha256"],
            "native sequence changed")
    require(bool(torch.isfinite(parsed["X"]).all()), "non-finite input coordinates")
    dump_tree(output / "parsed_input", {"protein": parsed, "icodes": tuple(icodes)})
    config = constructor_config(protocol, checkpoint=assets["checkpoint"], out_folder=output / "cpu-preflight-native", seed=17)
    config["device"] = "cpu"
    runner = LigandMPNN(OmegaConf.create(config))
    try:
        weights = {"load_count": runner.model_load_count,
                   "parameters": sum(p.numel() for p in runner.model.parameters()),
                   "devices": sorted({str(p.device) for p in runner.model.parameters()}),
                   "dtypes": sorted({str(p.dtype) for p in runner.model.parameters()}),
                   "forward_calls": 0, "sample_calls": 0}
        require(weights == {"load_count": 1, "parameters": 1660485, "devices": ["cpu"],
                            "dtypes": ["torch.float32"], "forward_calls": 0, "sample_calls": 0},
                "pretrained CPU constructor/parameter contract changed")
    finally:
        runner.close()
    require(not torch.cuda.is_initialized(), "CPU preflight unexpectedly initialized CUDA")
    source_paths = git("ls-files", "astrolabe", "scripts/validation", "benchmarks/runtime").splitlines()
    sources = {name: sha256(ROOT / name) for name in source_paths if (ROOT / name).is_file()}
    frozen = {"schema": 1, "protocol": protocol, "protocol_sha256": sha256(args.protocol),
              "root": str(ROOT), "head": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
              "assets": assets, "source_sha256": sources, "plan": trial_plan(protocol),
              "output": str(output), "created_unix": time.time(),
              "cpu_preflight": {"python": sys.version, "executable": sys.executable,
                                "torch": str(torch.__version__), "numpy": np.__version__,
                                "cuda_build": torch.version.cuda, "cuda_initialized": False,
                                "weights": weights, "native_sequence": native_sequence,
                                "settings": capture_settings(), "mask_sum": int(parsed["mask"].sum()),
                                "chain_order": chain_order,
                                "parsed_input_sha256": {p.name: sha256(p) for p in (output / "parsed_input").iterdir()}}}
    write_json(output / "frozen.json", frozen)
    print(json.dumps({"frozen": str(output / "frozen.json"), "sha256": sha256(output / "frozen.json"),
                      "status": "CPU_PREFLIGHT_ONLY", "model_loads_planned": 10, "decode_calls_planned": 30}))


def load_frozen(path, expected_sha=None):
    path = Path(path).resolve()
    if expected_sha:
        require(sha256(path) == expected_sha, "frozen manifest changed")
    frozen = json.loads(path.read_text())
    validate_protocol(frozen["protocol"])
    require(frozen["plan"] == trial_plan(frozen["protocol"]), "trial plan differs from frozen protocol")
    require(Path(frozen["root"]).resolve() == ROOT, "candidate path changed after freeze")
    require(git("rev-parse", "HEAD") == frozen["head"], "candidate HEAD changed after freeze")
    for relative, digest in frozen["source_sha256"].items():
        require(sha256(ROOT / relative) == digest, f"candidate source changed after freeze: {relative}")
    for key, path_value in frozen["assets"].items():
        verify_file(path_value, frozen["protocol"]["assets"][key])
    for filename, digest in frozen["cpu_preflight"]["parsed_input_sha256"].items():
        require(sha256(path.parent / "parsed_input" / filename) == digest, "CPU parsed input evidence changed")
    return frozen


def exact_tree(left, right):
    """Value/type/shape/dtype exact comparison for RNG and discrete results."""
    import numpy as np
    import torch
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict):
        return set(left) == set(right) and all(exact_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact_tree(a, b) for a, b in zip(left, right))
    return bool(left == right)


def validate_output(result, protocol):
    """Validate actual public return fields; do not invent returned masks."""
    import torch
    required = {"S", "sampling_probs", "log_probs", "logits", "decoding_order", "chain_letters", "alphabet", "distribution_semantics"}
    require(isinstance(result, dict) and set(result) == required, "unexpected native public output schema")
    shapes = {"S": (1, 354), "decoding_order": (1, 354), "sampling_probs": (1, 354, 20),
              "log_probs": (1, 354, 21), "logits": (1, 354, 21)}
    for key, shape in shapes.items():
        value = result[key]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and value.device.type == "cpu",
                f"{key} is not a full CPU-ready output")
        expected_dtype = torch.int64 if key in ("S", "decoding_order") else torch.float32
        require(value.dtype == expected_dtype and bool(torch.isfinite(value).all()), f"invalid {key} dtype/non-finite output")
    order = result["decoding_order"][0]
    require(torch.equal(order.sort().values, torch.arange(354)), "decode order is not a full permutation")
    sequence = result["S"]
    require(bool(((sequence >= 0) & (sequence < 20)).all()), "invalid sampled residue")
    alphabet = tuple("ACDEFGHIKLMNPQRSTVWY")
    require(result["alphabet"] == alphabet, "native alphabet/order changed")
    expected_chains = tuple(chain for chain, count in protocol["assets"]["pdb"]["chain_order"] for _ in range(count))
    require(result["chain_letters"] == expected_chains, "returned chain order changed")
    require(result["distribution_semantics"] == protocol["decode"]["distribution_semantics"], "returned distribution semantics changed")
    probs = result["sampling_probs"]
    require(bool((probs >= 0).all()) and torch.allclose(probs.sum(-1), torch.ones((1, 354)),
            rtol=0, atol=protocol["scientific_gates"]["probability_sum_atol"]), "sampling probabilities are not normalized")
    c_index = alphabet.index("C")
    require(not bool((sequence == c_index).any()) and bool((probs[..., c_index] == 0).all()), "omitted C sampled/has nonzero probability")
    selected = result["log_probs"].gather(-1, sequence.unsqueeze(-1)).squeeze(-1)
    # This is the native get_score formula on the all-one frozen mask, evaluated
    # independently from returned log_probs. It is not a returned native score.
    score = float(-selected.mean())
    return {"sequence": "".join(alphabet[int(x)] for x in sequence[0]),
            "derived_native_mean_nll": score, "score_source": "CPU formula on returned log_probs; all-one input mask",
            "probability_sum_max_abs_error": float((probs.sum(-1) - 1).abs().max())}


def validate_lifecycle(state):
    require(state.get("model_load_count") == 1 and state.get("weights_loaded") is True and
            state.get("gpu_resident") is True and state.get("ready") is True and
            state.get("devices") == ["cuda:0"] and state.get("dtypes") == ["torch.float32"],
            "model is not one loaded float32 GPU instance")


def validate_rng(state, protocol):
    import torch
    require(isinstance(state, dict) and set(state) == {"cpu", "cuda", "python", "numpy", "settings"},
            "incomplete scientific RNG state")
    for key in ("cpu", "cuda"):
        value = state[key]
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.uint8
                and value.ndim == 1 and value.numel() > 0, f"missing actual {key} RNG bytes")
    require(state["settings"] == protocol["numerical_settings"], "RNG numerical settings changed")


def trial(args):
    require_private_allocation(os.environ)
    require_gpu_uuid(args.allocated_gpu_uuid)
    frozen = load_frozen(args.frozen, args.frozen_sha256)
    item = frozen["plan"][args.index]
    protocol = frozen["protocol"]
    directory = Path(frozen["output"]) / "trials" / item["id"]
    directory.mkdir(parents=True, exist_ok=True)
    require(not (directory / "trial.json").exists(), "trial evidence already exists; replay is prohibited")
    result = {**item, "status": "RUNNING", "requests": [], "valid_completed_requests": 0,
              "frozen_sha256": args.frozen_sha256, "head": frozen["head"], "release": None}
    write_json(directory / "trial.json", result)
    runner = client = None
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    from astrolabe.evaluators.lifecycle import cpu_tree
    from astrolabe.evaluators.torch_rng import capture, capture_settings
    from astrolabe.evaluators.tree_codec import dump_tree
    from astrolabe.mutators.resident import build_runner, execution_mode
    from astrolabe.mutators.wrappers.ligandmpnn_run import LigandMPNN  # Same native import before seed in both modes.
    from astrolabe.mutators.wrappers.resident_daemon import lifecycle_snapshot
    try:
        require(torch.version.cuda is not None and torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "real CUDA is required; CPU fallback is never a GPU result")
        torch.cuda.set_device(0)
        device = "cuda:0"
        props = torch.cuda.get_device_properties(0)
        result["device"] = {"name": props.name, "uuid": str(getattr(props, "uuid", "unavailable")),
                            "total_memory": props.total_memory, "capability": list(torch.cuda.get_device_capability(0)),
                            "visible": os.environ["CUDA_VISIBLE_DEVICES"]}
        actual_uuid = result["device"]["uuid"].lower().removeprefix("gpu-")
        require(actual_uuid == args.allocated_gpu_uuid.lower().removeprefix("gpu-"),
                "actual CUDA device UUID differs from the independently verified allocation")
        result["allocated_gpu_uuid"] = args.allocated_gpu_uuid
        result["environment"] = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                                 "cuda_build": torch.version.cuda, "numpy": np.__version__,
                                 "slurm_job_id": os.environ["SLURM_JOB_ID"]}
        require(all(result["environment"][key] == frozen["cpu_preflight"][key]
                    for key in ("python", "executable", "torch", "cuda_build", "numpy")),
                "GPU environment differs from the real CPU preflight environment")
        require(capture_settings() == protocol["numerical_settings"], "numerical defaults changed; no automatic precision tuning")
        random.seed(item["seed"])
        np.random.seed(item["seed"])
        torch.manual_seed(item["seed"])
        dump_tree(directory / "rng_before_constructor", capture(device))
        config = constructor_config(protocol, checkpoint=frozen["assets"]["checkpoint"],
                                    out_folder=directory / "native", seed=item["seed"])
        result["constructor"] = config
        kwargs = copy.deepcopy(protocol["decode"])
        kwargs["pdb_path"] = frozen["assets"]["pdb"]
        result["decode"] = kwargs
        start = time.perf_counter()
        with execution_mode("legacy" if item["mode"] == "legacy" else "daemon"):
            runner = build_runner("ligandmpnn", OmegaConf.create(config))
        torch.cuda.synchronize(0)
        result["construction_seconds"] = time.perf_counter() - start
        if item["mode"] == "resident":
            client = runner._client  # Observational reference survives public close.
            result["worker_pid"] = client.process.pid
        state = lifecycle_snapshot(runner, "ligandmpnn") if item["mode"] == "legacy" else dict(runner.lifecycle)
        validate_lifecycle(state)
        result["after_constructor_lifecycle"] = state
        previous_rng = capture(device)
        validate_rng(previous_rng, protocol)
        require(previous_rng["settings"] == protocol["numerical_settings"], "constructor changed numerical settings")
        dump_tree(directory / "rng_after_constructor", previous_rng)
        write_json(directory / "trial.json", result)
        for index in range(protocol["requests_per_trial"]):
            record = {"index": index, "phase": "first" if index == 0 else "steady", "status": "RUNNING"}
            result["requests"].append(record)
            write_json(directory / "trial.json", result)
            before = capture(device)
            require(exact_tree(before, previous_rng), "RNG changed between consecutive requests")
            request_dir = directory / f"request-{index:02d}"
            dump_tree(request_dir / "rng_before", before)
            start = time.perf_counter()
            raw = runner.design_sequences(**kwargs)
            output = cpu_tree(raw)
            del raw
            torch.cuda.synchronize(0)
            record["cpu_ready_call_seconds"] = time.perf_counter() - start
            previous_rng = capture(device)
            validate_rng(previous_rng, protocol)
            require(not exact_tree(before["cuda"], previous_rng["cuda"]), "native GPU sampling did not advance CUDA RNG")
            require(previous_rng["settings"] == protocol["numerical_settings"], "request changed numerical settings")
            # Save actual data even when later shape/science checks reject it.
            dump_tree(request_dir / "output", output)
            dump_tree(request_dir / "rng_after", previous_rng)
            record["science"] = validate_output(output, protocol)
            state = lifecycle_snapshot(runner, "ligandmpnn") if item["mode"] == "legacy" else dict(runner.lifecycle)
            validate_lifecycle(state)
            record.update(status="COMPLETE", lifecycle=state)
            result["valid_completed_requests"] += 1
            del output
            write_json(directory / "trial.json", result)
        result["status"] = "COMPLETE"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        close_start = time.perf_counter()
        try:
            if runner is not None:
                runner.close()
                if item["mode"] == "resident":
                    release = dict(client.lifecycle)
                    release["worker_exit_code"] = client.process.poll()
                    require(release.get("release_confirmed") is True and release["worker_exit_code"] is not None,
                            "private worker release is not confirmed")
                    result["release"] = release
                else:
                    require(runner.model is None and runner._closed, "legacy model close did not clear its owner")
                    result["release"] = {"model_cleared": True, "owner_process_exit_required": True,
                                         "note": "Final GPU context/resource release requires controller-observed process exit."}
            else:
                result["release"] = {"model_handle_created": False, "owner_process_exit_required": True}
        except BaseException as exc:
            result["status"] = "FAILED"
            result["release_error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        result["close_seconds"] = time.perf_counter() - close_start
        write_json(directory / "trial.json", result)
    return 0 if result["status"] == "COMPLETE" else 1


def compare_pair(legacy, resident, protocol):
    """Preserve numeric deviations even if the predeclared gate fails."""
    import numpy as np
    float_keys = set(protocol["scientific_gates"]["float_keys"])
    report = {"pass": True, "fields": {}}
    for key in sorted(set(legacy) | set(resident)):
        if key not in legacy or key not in resident:
            item = {"pass": False, "reason": "missing key"}
        elif key not in float_keys:
            item = {"pass": exact_tree(legacy[key], resident[key]), "comparison": "exact"}
        else:
            a, b = legacy[key].numpy(), resident[key].numpy()
            compatible = a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all() and np.isfinite(b).all()
            if compatible:
                delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                scale = np.maximum(np.abs(a.astype(np.float64)), np.finfo(np.float64).tiny)
                item = {"pass": bool(np.allclose(a, b, rtol=protocol["scientific_gates"]["rtol"],
                                                 atol=protocol["scientific_gates"]["atol"], equal_nan=False)),
                        "bitexact": a.tobytes() == b.tobytes(), "max_abs": float(delta.max(initial=0)),
                        "max_relative_to_legacy": float(np.minimum(delta / scale, np.finfo(np.float64).max).max(initial=0))}
            else:
                item = {"pass": False, "reason": "shape/dtype differs or non-finite values"}
        report["fields"][key] = item
        report["pass"] = report["pass"] and item["pass"]
    return report


def aggregate(frozen_path, execution):
    from astrolabe.evaluators.tree_codec import load_tree
    frozen = json.loads(Path(frozen_path).read_text())
    protocol = frozen["protocol"]
    root = Path(frozen["output"])
    validate_execution_prefix(frozen["plan"], execution["trials"])
    report = {"status": "INVALID", "scope": protocol["scope"], "head": frozen["head"],
              "frozen_sha256": sha256(frozen_path), "pairs": [], "trials": execution["trials"],
              "attempted_trials": len(execution["trials"]), "planned_trials": 10,
              "planned_design_requests": 30, "completed_design_requests": 0,
              "all_attempt_wall_seconds": sum(row["wall_seconds"] for row in execution["trials"]),
              "unstarted_trials": 10 - len(execution["trials"]),
              "certification": "GPU mutator control only; existing private process mechanism; no shared broker/evaluator/MCMC/revision attribution"}
    rows = {row["id"]: row for row in execution["trials"]}
    complete = len(rows) == 10
    for item in frozen["plan"]:
        row = rows.get(item["id"])
        if row is None:
            complete = False
            continue
        path = root / "trials" / item["id"] / "trial.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        row["child_status"] = data.get("status", "MISSING")
        requests = data.get("requests", [])
        row["valid_completed_requests"] = sum(request.get("status") == "COMPLETE" for request in requests)
        report["completed_design_requests"] += row["valid_completed_requests"]
        measurement = row.get("measurement", {})
        envelope = measurement.get("measurement_envelope_seconds")
        measured_ok = (measurement.get("stop_reason") is None and measurement.get("seconds") == row["wall_seconds"]
                       and type(envelope) in (int, float) and math.isfinite(envelope) and envelope >= row["wall_seconds"])
        row["valid"] = (row["observation"].get("release_confirmed") is True and row["observation"].get("exit_code") == 0
                        and not row["observation"].get("cleanup_attempted", True)
                        and row["observation"].get("observed_tree_empty") is True
                        and not row["observation"].get("remaining_identities", ["unknown"])
                        and data.get("status") == "COMPLETE" and len(requests) == 3 and row["valid_completed_requests"] == 3
                        and data.get("valid_completed_requests") == 3
                        and measured_ok
                        and data.get("frozen_sha256") == report["frozen_sha256"])
        complete = complete and row["valid"]
    for group in range(5):
        items = [item for item in frozen["plan"] if item["group"] == group]
        by_mode = {item["mode"]: item for item in items}
        if not all(rows.get(item["id"], {}).get("valid") for item in items):
            continue
        dirs = {mode: root / "trials" / item["id"] for mode, item in by_mode.items()}
        pair = {"group": group, "seed": items[0]["seed"], "pass": True, "requests": []}
        metadata = {mode: json.loads((directory / "trial.json").read_text()) for mode, directory in dirs.items()}
        for field in ("device", "environment"):
            equal = metadata["legacy"][field] == metadata["resident"][field]
            pair[field + "_exact"] = equal
            pair["pass"] = pair["pass"] and equal
        for stage in ("rng_before_constructor", "rng_after_constructor"):
            equal = exact_tree(load_tree(dirs["legacy"] / stage), load_tree(dirs["resident"] / stage))
            pair[stage + "_exact"] = equal
            pair["pass"] = pair["pass"] and equal
        for index in range(3):
            rd = {mode: directory / f"request-{index:02d}" for mode, directory in dirs.items()}
            left, right = (load_tree(rd[mode] / "output") for mode in ("legacy", "resident"))
            gate = compare_pair(left, right, protocol)
            try:
                validate_output(left, protocol)
                validate_output(right, protocol)
                for mode in ("legacy", "resident"):
                    before = load_tree(rd[mode] / "rng_before")
                    after = load_tree(rd[mode] / "rng_after")
                    previous = load_tree(dirs[mode] / "rng_after_constructor" if index == 0
                                         else dirs[mode] / f"request-{index - 1:02d}" / "rng_after")
                    validate_rng(before, protocol)
                    validate_rng(after, protocol)
                    require(exact_tree(before, previous), "RNG discontinuity between requests")
                    require(not exact_tree(before["cuda"], after["cuda"]), "CUDA RNG did not advance during sampling")
                gate["independent_output_rng_gate"] = True
            except (ValueError, TypeError, KeyError) as exc:
                gate["independent_output_rng_gate"] = False
                gate["independent_gate_error"] = str(exc)
                gate["pass"] = False
            for stage in ("rng_before", "rng_after"):
                equal = exact_tree(load_tree(rd["legacy"] / stage), load_tree(rd["resident"] / stage))
                gate[stage + "_exact"] = equal
                gate["pass"] = gate["pass"] and equal
            pair["requests"].append(gate)
            pair["pass"] = pair["pass"] and gate["pass"]
        report["pairs"].append(pair)
        complete = complete and pair["pass"]
    report["status"] = "PASS" if complete and len(report["pairs"]) == 5 and execution.get("status") == "COMPLETE" else "INVALID"
    report["resource_telemetry_status"] = "COMPLETE" if len(rows) == 10 and all(
        row.get("measurement", {}).get("peak_process_tree_rss_bytes") is not None and
        row.get("measurement", {}).get("peak_device_vram_bytes") is not None for row in rows.values()) else "PARTIAL_UNKNOWN"
    report["resource_telemetry_note"] = "Raw timelines and measured/unknown values remain in trials[].measurement. Missing readings are never zero. Device-wide readings rely on root's exclusive allocation evidence."
    report["per_mode"] = {}
    for mode in ("legacy", "resident"):
        selected = [row for row in report["trials"] if row["mode"] == mode]
        seconds = sum(row["wall_seconds"] for row in selected)
        count = sum(row.get("valid_completed_requests", 0) for row in selected)
        report["per_mode"][mode] = {"attempted_trials": len(selected), "completed_design_requests": count,
                                    "all_attempt_wall_seconds": seconds,
                                    "diagnostic_completed_per_second": count / seconds if seconds else None}
        if report["status"] == "PASS":
            data = [json.loads((root / "trials" / row["id"] / "trial.json").read_text()) for row in selected]
            report["per_mode"][mode].update(
                effective_design_requests_per_second=count / seconds,
                median_workflow_seconds=statistics.median(row["wall_seconds"] for row in selected),
                median_construction_seconds=statistics.median(row["construction_seconds"] for row in data),
                median_first_request_seconds=statistics.median(row["requests"][0]["cpu_ready_call_seconds"] for row in data),
                median_steady_request_seconds=statistics.median(request["cpu_ready_call_seconds"] for row in data for request in row["requests"][1:]))
    if report["status"] == "PASS":
        ratios = [rows[f"g{i:02d}-seed{seed}-resident"]["wall_seconds"] / rows[f"g{i:02d}-seed{seed}-legacy"]["wall_seconds"]
                  for i, seed in enumerate(protocol["groups"])]
        report["resident_to_legacy_workflow_ratios"] = ratios
        report["median_resident_to_legacy_workflow_ratio"] = statistics.median(ratios)
        report["resident_faster_in_all_five_groups"] = all(ratio < 1 for ratio in ratios)
    return report


def run(args):
    require_private_allocation(os.environ)
    require_gpu_uuid(args.allocated_gpu_uuid)
    frozen = load_frozen(args.frozen, args.frozen_sha256)
    from astrolabe.runtime.telemetry import process_namespace_supported
    from scripts.validation.runtime_workflow_benchmark import run_once
    require(process_namespace_supported(), "process namespace unsupported; release cannot be certified")
    output = Path(frozen["output"])
    require(not (output / "execution.json").exists(), "run already attempted; never overwrite/retry frozen evidence")
    budget = frozen["protocol"]["budget"]
    start = time.monotonic()
    deadline = start + budget["experiment_seconds"]
    execution = {"status": "RUNNING", "frozen_sha256": args.frozen_sha256, "trials": [],
                 "allocated_gpu_uuid": args.allocated_gpu_uuid,
                 "scheduler": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
                    "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")},
                 "start_unix": time.time(), "budget": budget}
    write_json(output / "execution.json", execution)
    try:
        for item in frozen["plan"]:
            require(time.monotonic() < deadline - 7, "overall budget exhausted before next trial")
            used_output = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
            remaining_output = budget["max_output_bytes"] - used_output
            require(remaining_output > 0, "cumulative output budget exhausted before next trial")
            directory = output / "trials" / item["id"]
            directory.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()), "_trial", "--frozen", str(args.frozen.resolve()),
                       "--frozen-sha256", args.frozen_sha256, "--index", str(item["index"]),
                       "--allocated-gpu-uuid", args.allocated_gpu_uuid]
            before = time.monotonic()
            row = {**item, "command": command, "wall_seconds": 0.0, "observation": {},
                   "output_budget_before_trial": {"used_bytes": used_output, "remaining_bytes": remaining_output}}
            execution["trials"].append(row)
            write_json(output / "execution.json", execution)
            measurement = run_once(command, directory, ROOT, args.allocated_gpu_uuid,
                                   min(deadline, before + budget["trial_seconds"]),
                                   budget["telemetry_sample_seconds"], remaining_output, repository=ROOT)
            observation = measurement["process_release"]
            row["wall_seconds"] = measurement["seconds"]
            row["observation"] = observation
            row["measurement"] = measurement
            row["output_bytes_after_trial"] = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
            write_json(output / "execution.json", execution)
            require(not measurement["stop_reason"] and row["output_bytes_after_trial"] <= budget["max_output_bytes"] and
                    observation["release_confirmed"] is True and observation["exit_code"] == 0,
                    f"trial failed or release unknown: {item['id']}; later trials are not launched")
            if item["index"] % 2 == 1:
                interim = aggregate(args.frozen, execution)
                write_json(output / f"comparison-through-group-{item['group']:02d}.json", interim)
                require(len(interim["pairs"]) == item["group"] + 1 and all(pair["pass"] for pair in interim["pairs"]),
                        "paired scientific gate failed; later groups are not launched")
        execution["status"] = "COMPLETE"
    except BaseException as exc:
        execution["status"] = "FAILED"
        execution["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if execution["trials"] and not execution["trials"][-1]["observation"]:
            execution["trials"][-1]["wall_seconds"] = time.monotonic() - before
    finally:
        execution["experiment_wall_seconds"] = time.monotonic() - start
        execution["experiment_gpu_hours"] = execution["experiment_wall_seconds"] / 3600
        execution["cost_scope"] = "Inner experiment only; full allocation startup/shutdown GPU cost requires root Slurm start/end accounting."
        write_json(output / "execution.json", execution)
    report = aggregate(args.frozen, execution)
    write_json(output / "comparison.json", report)
    print(json.dumps({"status": report["status"], "completed_design_requests": report["completed_design_requests"],
                      "comparison": str(output / "comparison.json")}))
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("freeze", help="CPU-only real pretrained load/input gate; no forward or GPU")
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--pdb", type=Path)
    for command in ("run", "_trial", "compare"):
        p = subs.add_parser(command)
        p.add_argument("--frozen", type=Path, required=True)
        p.add_argument("--frozen-sha256", required=True)
        if command == "_trial":
            p.add_argument("--index", type=int, required=True)
        if command in ("run", "_trial"):
            p.add_argument("--allocated-gpu-uuid", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
        return 0
    if args.command == "_trial":
        return trial(args)
    if args.command == "run":
        return run(args)
    frozen = load_frozen(args.frozen, args.frozen_sha256)
    execution = json.loads((Path(frozen["output"]) / "execution.json").read_text())
    report = aggregate(args.frozen, execution)
    destination = Path(frozen["output"]) / "comparison-independent.json"
    require(not destination.exists(), "independent comparison already exists")
    write_json(destination, report)
    print(json.dumps({"status": report["status"], "output": str(destination)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
