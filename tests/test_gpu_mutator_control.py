"""CPU-native tensor/negative controls for the GPU measurement harness.

These synthetic outputs test rejection/measurement logic only. They do not
execute pretrained inference and are never GPU/backend certification evidence.
"""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gpu_mutator_control", ROOT / "benchmarks/runtime/gpu_mutator_control.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


@pytest.fixture
def protocol():
    return json.loads(control.DEFAULT_PROTOCOL.read_text())


def scientific_output():
    probabilities = torch.ones((1, 354, 20), dtype=torch.float32) / 19
    probabilities[..., 1] = 0
    return {"S": torch.zeros((1, 354), dtype=torch.int64), "sampling_probs": probabilities,
            "log_probs": torch.full((1, 354, 21), -float(np.log(21)), dtype=torch.float32),
            "logits": torch.zeros((1, 354, 21), dtype=torch.float32),
            "decoding_order": torch.arange(354).reshape(1, -1),
            "chain_letters": tuple("A" * 177 + "B" * 177),
            "alphabet": tuple("ACDEFGHIKLMNPQRSTVWY"),
            "distribution_semantics": "legacy_order_conditioned_marginals"}


def synthetic_rng(protocol, index):
    from astrolabe.evaluators.torch_rng import capture
    state = capture("cpu")  # Actual CPU/Python/NumPy states; explicitly synthetic CUDA test bytes.
    state["cuda"] = torch.tensor([index, 1, 2, 3], dtype=torch.uint8)
    state["settings"] = protocol["numerical_settings"]
    return state


def test_real_native_signature_hashes_and_fixed_plan(protocol):
    control.validate_protocol(protocol)
    assert len(control.trial_plan(protocol)) == 10
    assert sum(protocol["requests_per_trial"] for _ in control.trial_plan(protocol)) == 30
    for name, digest in protocol["native_sources"].items():
        assert control.sha256(ROOT / name) == digest
    for group in range(5):
        items = control.trial_plan(protocol)[group * 2:group * 2 + 2]
        assert {item["mode"] for item in items} == {"legacy", "resident"}
        assert {item["seed"] for item in items} == {17 + group}


@pytest.mark.parametrize("mutation", [
    lambda p: p["decode"].update(parse_these_chains_only="A,B"),
    lambda p: p["decode"].pop("symmetry_weights"),
    lambda p: p["decode"].update(batch_size=2),
    lambda p: p["constructor"].update(omit_AA=[]),
    lambda p: p["constructor"].update(number_of_batches=2),
    lambda p: p.update(requests_per_trial=1),
    lambda p: p.update(warmup_requests=1),
    lambda p: p["groups"].reverse(),
    lambda p: p["scientific_gates"].update(rtol=.01),
    lambda p: p["budget"].update(experiment_seconds=901),
])
def test_scientific_protocol_reductions_rejected(protocol, mutation):
    mutation(protocol)
    with pytest.raises((ValueError, KeyError)):
        control.validate_protocol(protocol)


@pytest.mark.parametrize("environment", [
    {}, {"SLURM_JOB_ID": "9", "CUDA_VISIBLE_DEVICES": ""},
    {"SLURM_JOB_ID": "9", "CUDA_VISIBLE_DEVICES": "0,1"},
    {"SLURM_JOB_ID": "9", "CUDA_VISIBLE_DEVICES": "0", "NVIDIA_VISIBLE_DEVICES": "all"},
    {"SLURM_JOB_ID": "9", "CUDA_VISIBLE_DEVICES": "0", "ASTROLABE_EXECUTION_TOKEN": "shared"},
    {"SLURM_JOB_ID": "9", "CUDA_VISIBLE_DEVICES": "0", "ASTROLABE_ALLOCATED_CUDA_DEVICES": "1"},
])
def test_shared_or_cpu_or_unallocated_device_rejected(environment):
    with pytest.raises(ValueError):
        control.require_private_allocation(environment)


def test_allocation_guard_accepts_only_explicit_private_binding():
    # This does not prove hardware allocation; actual run independently checks CUDA and Slurm evidence.
    control.require_private_allocation({"SLURM_JOB_ID": "9", "CUDA_VISIBLE_DEVICES": "GPU-test"})


def test_real_cpu_tensor_output_and_codec(protocol, tmp_path):
    from astrolabe.evaluators.tree_codec import dump_tree, load_tree
    data = scientific_output()
    assert control.validate_output(data, protocol)["sequence"] == "A" * 354
    dump_tree(tmp_path / "output", data)
    restored = load_tree(tmp_path / "output")
    assert control.exact_tree(data, restored)
    assert control.compare_pair(data, restored, protocol)["pass"]


@pytest.mark.parametrize("mutation", [
    lambda o: o["S"].fill_(1),  # Forbidden cysteine.
    lambda o: o["decoding_order"].fill_(0),
    lambda o: o["sampling_probs"].fill_(0),
    lambda o: o["sampling_probs"].__setitem__((0, 0, 0), float("nan")),
    lambda o: o.update(S=o["S"][:, :-1]),
    lambda o: o.update(logits=o["logits"].half()),
    lambda o: o.update(chain_letters=tuple("B" * 177 + "A" * 177)),
    lambda o: o.update(mask=torch.ones(354)),  # No invented return field.
])
def test_malformed_or_scientifically_changed_outputs_rejected(protocol, mutation):
    data = scientific_output()
    mutation(data)
    with pytest.raises(ValueError):
        control.validate_output(data, protocol)


def test_fixed_tolerance_and_discrete_negative_controls(protocol):
    left = scientific_output()
    right = copy.deepcopy(left)
    right["logits"][0, 1, 2] = 5e-7
    allowed = control.compare_pair(left, right, protocol)
    assert allowed["pass"] and not allowed["fields"]["logits"]["bitexact"]
    assert allowed["fields"]["logits"]["max_abs"] > 0
    right["logits"][0, 1, 2] = 2e-6
    assert not control.compare_pair(left, right, protocol)["pass"]
    right = copy.deepcopy(left)
    right["S"][0, 2] = 2
    assert not control.compare_pair(left, right, protocol)["pass"]


def test_rng_requires_cuda_state_and_exact_bytes(protocol):
    left = synthetic_rng(protocol, 1)
    control.validate_rng(left, protocol)
    right = copy.deepcopy(left)
    assert control.exact_tree(left, right)
    right["cuda"][0] += 1
    assert not control.exact_tree(left, right)
    right["cuda"] = None
    with pytest.raises(ValueError, match="cuda"):
        control.validate_rng(right, protocol)


def test_corrupt_checkpoint_bytes_rejected(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"genuine-evidence-control")
    expected = {"bytes": path.stat().st_size, "sha256": control.sha256(path)}
    control.verify_file(path, expected)
    path.write_bytes(b"changed-evidence-control")
    with pytest.raises(ValueError, match="hash mismatch"):
        control.verify_file(path, expected)


@pytest.fixture
def synthetic_run(tmp_path, protocol):
    from astrolabe.evaluators.tree_codec import dump_tree
    frozen_path = tmp_path / "frozen.json"
    frozen = {"output": str(tmp_path), "protocol": protocol, "head": "synthetic-contract-only",
              "plan": control.trial_plan(protocol)}
    control.write_json(frozen_path, frozen)
    frozen_sha = control.sha256(frozen_path)
    rows = []
    for item in frozen["plan"]:
        directory = tmp_path / "trials" / item["id"]
        directory.mkdir(parents=True)
        rows.append({**item, "wall_seconds": 2., "observation": {"release_confirmed": True, "exit_code": 0,
            "cleanup_attempted": False, "observed_tree_empty": True, "remaining_identities": []},
            "measurement": {"seconds": 2., "stop_reason": None, "measurement_envelope_seconds": 2.5,
                            "peak_process_tree_rss_bytes": None, "peak_device_vram_bytes": None}})
        metadata = {"status": "COMPLETE", "valid_completed_requests": 3, "frozen_sha256": frozen_sha,
                    "construction_seconds": .5, "device": {"fixture": "synthetic CUDA metadata"},
                    "environment": {"fixture": "CPU test only"},
                    "requests": [{"status": "COMPLETE", "cpu_ready_call_seconds": .2} for _ in range(3)]}
        control.write_json(directory / "trial.json", metadata)
        dump_tree(directory / "rng_before_constructor", synthetic_rng(protocol, 0))
        dump_tree(directory / "rng_after_constructor", synthetic_rng(protocol, 1))
        for index in range(3):
            root = directory / f"request-{index:02d}"
            dump_tree(root / "output", scientific_output())
            dump_tree(root / "rng_before", synthetic_rng(protocol, index + 1))
            dump_tree(root / "rng_after", synthetic_rng(protocol, index + 2))
    return frozen_path, {"trials": rows, "status": "COMPLETE"}


def test_complete_pairwise_contract_and_raw_counts(synthetic_run):
    frozen, execution = synthetic_run
    report = control.aggregate(frozen, execution)
    assert report["status"] == "PASS"  # Synthetic harness contract, never actual GPU certification.
    assert report["completed_design_requests"] == 30
    assert len(report["pairs"]) == 5
    assert report["all_attempt_wall_seconds"] == 20
    assert not report["resident_faster_in_all_five_groups"]
    assert report["per_mode"]["legacy"]["effective_design_requests_per_second"] == 1.5
    assert report["resource_telemetry_status"] == "PARTIAL_UNKNOWN"


@pytest.mark.parametrize("mutation", [
    lambda e: e["trials"].pop(),
    lambda e: e["trials"][0]["observation"].update(release_confirmed=False),
    lambda e: e["trials"][0]["observation"].update(cleanup_attempted=True),
    lambda e: e["trials"][0]["observation"].update(exit_code=1),
    lambda e: e["trials"][0]["observation"].update(remaining_identities=[[123, 456]]),
    lambda e: e["trials"][0]["measurement"].update(stop_reason="output_budget_exhausted"),
    lambda e: e["trials"][0]["measurement"].update(measurement_envelope_seconds=1.),
    lambda e: e.update(status="FAILED"),
])
def test_failed_missing_or_unknown_release_cannot_certify_throughput(synthetic_run, mutation):
    frozen, execution = synthetic_run
    mutation(execution)
    report = control.aggregate(frozen, execution)
    assert report["status"] == "INVALID"
    assert "effective_design_requests_per_second" not in report["per_mode"]["legacy"]
    assert report["all_attempt_wall_seconds"] == sum(row["wall_seconds"] for row in execution["trials"])


@pytest.mark.parametrize("mutation", [
    lambda e: e["trials"].append(copy.deepcopy(e["trials"][0])),
    lambda e: e["trials"].__setitem__(1, copy.deepcopy(e["trials"][0])),
    lambda e: e["trials"].reverse(),
    lambda e: e["trials"][0].update(seed=999),
    lambda e: e["trials"][0].update(index=True),
    lambda e: e["trials"][0].update(mode="other"),
    lambda e: e["trials"][0].update(id="foreign"),
    lambda e: e["trials"][0].update(wall_seconds=0),
    lambda e: e["trials"][0].update(wall_seconds=-2),
    lambda e: e["trials"][0].update(wall_seconds=float("nan")),
    lambda e: e["trials"][0].update(wall_seconds=float("inf")),
])
def test_corrupted_execution_identity_or_timing_rejected(synthetic_run, mutation):
    frozen, execution = synthetic_run
    mutation(execution)
    with pytest.raises(ValueError):
        control.aggregate(frozen, execution)


def test_equal_but_invalid_probability_and_rng_continuity_still_fail(synthetic_run, protocol):
    from astrolabe.evaluators.tree_codec import dump_tree
    frozen, execution = synthetic_run
    for mode in ("legacy", "resident"):
        directory = frozen.parent / "trials" / f"g00-seed17-{mode}" / "request-01"
        data = scientific_output()
        data["sampling_probs"].fill_(0)
        dump_tree(directory / "output", data)
        dump_tree(directory / "rng_before", synthetic_rng(protocol, 9))
    report = control.aggregate(frozen, execution)
    assert report["status"] == "INVALID"
    assert not report["pairs"][0]["requests"][1]["independent_output_rng_gate"]


def test_one_returned_sequence_or_rng_difference_blocks_all_certification(synthetic_run, protocol):
    from astrolabe.evaluators.tree_codec import dump_tree
    frozen, execution = synthetic_run
    directory = frozen.parent / "trials" / "g03-seed20-resident" / "request-02"
    data = scientific_output()
    data["S"][0, 10] = 2
    dump_tree(directory / "output", data)
    dump_tree(directory / "rng_after", synthetic_rng(protocol, 8))
    report = control.aggregate(frozen, execution)
    assert report["status"] == "INVALID"
    assert report["completed_design_requests"] == 30
    assert not report["pairs"][3]["requests"][2]["pass"]
