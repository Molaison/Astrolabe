from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.evaluators.registry import EvaluatorRegistry
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.registry import MutatorRegistry


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validation"
    / "typed_native_gpu_acceptance.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "typed_native_gpu_acceptance",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def acceptance_module():
    return _load_module()


@pytest.fixture(autouse=True)
def isolated_runtime_registries(monkeypatch, acceptance_module):
    import astrolabe.mutators.base as mutator_base

    # Registry tests may have imported this module before restoring an isolated
    # registry. Reload here so decorators repopulate the actual test registry.
    importlib.reload(mutator_base)

    monkeypatch.setattr(
        EvaluatorRegistry,
        "_registry",
        dict(EvaluatorRegistry._registry),
    )
    monkeypatch.setattr(
        EvaluatorRegistry,
        "_operation_factories",
        dict(EvaluatorRegistry._operation_factories or {}),
    )
    monkeypatch.setattr(
        MutatorRegistry,
        "_registry",
        dict(MutatorRegistry._registry),
    )
    monkeypatch.setattr(
        MutatorRegistry,
        "_operation_factories",
        dict(MutatorRegistry._operation_factories or {}),
    )
    global_config.reset()

    class StructuralStatus:
        def __init__(self) -> None:
            self.request_count = 0
            self.completed_count = 0
            self.failed_count = 0

        def status(self):
            return {
                "pid": os.getpid(),
                "backend": "alphafold3",
                "state": "running",
                "request_count": self.request_count,
                "completed_count": self.completed_count,
                "failed_count": self.failed_count,
            }

    class StructuralRunner:
        def __init__(self) -> None:
            self._client = SimpleNamespace(owner=StructuralStatus())

        def run_inference(self, *_args, **_kwargs):
            owner = self._client.owner
            owner.request_count += 1
            owner.completed_count += 1
            return None

    class StructuralAF3:
        def __init__(self, output_root: Path) -> None:
            self.output_root = Path(output_root)
            self.model_runner = StructuralRunner()
            self.shutdown_calls = 0

        def evaluate(self, inputs, runtime_config=None):
            step = int((runtime_config or {}).get("step", 0))
            plddt = {}
            complex_paths = {}
            pae = {}
            for assembly, item in inputs.items():
                self.model_runner.run_inference(item, assembly)
                sequence = "".join(chain.sequence for chain in item.chains)
                digest = hashlib.sha256(
                    f"{step}:{assembly}:{sequence}".encode("ascii")
                ).hexdigest()[:12]
                output = (
                    self.output_root
                    / f"structural-{assembly}-{step:03d}-{digest}.cif"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    f"data_structural\n# assembly={assembly} "
                    f"step={step} sequence={sequence}\n",
                    encoding="ascii",
                )
                values = np.asarray(
                    [
                        (ord(token) % 17) / 17.0
                        for token in sequence
                    ],
                    dtype=np.float32,
                )
                plddt[assembly] = [values]
                complex_paths[assembly] = output
                pae[assembly] = [
                    np.abs(
                        values[:, None] - values[None, :]
                    ).astype(np.float32)
                ]
            return StandardScores(
                names=list(inputs),
                plddt=plddt,
                complex_paths=complex_paths,
                pae=pae,
            )

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    def fake_factory(
        term_conf,
        *,
        alias,
        raw_model_results_dir,
        oligo_names,
        evaluator_conf,
    ):
        del term_conf, alias, oligo_names, evaluator_conf
        return StructuralAF3(raw_model_results_dir)

    EvaluatorRegistry.register("typed_acceptance_fake", fake_factory)
    yield
    global_config.reset()


def _config(acceptance_module, root: Path):
    from omegaconf import open_dict

    cfg = acceptance_module.acceptance_config(root)
    with open_dict(cfg):
        cfg.evaluator.evaluator_terms.AF3._type = "typed_acceptance_fake"
    return cfg


def _assignment(record: dict, sequence: str) -> tuple[str, dict]:
    matches = [
        (variable, value)
        for variable, value in record["assignments"].items()
        if value.get("sequence") == sequence
    ]
    assert len(matches) == 1
    return matches[0]


def test_root_config_persists_active_science_without_unused_rf3_env(
    acceptance_module,
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("GH_WEIGHT_RF3", raising=False)
    root = tmp_path / "active-config"
    cfg = _config(acceptance_module, root)

    acceptance_module._initialize_root(root, cfg)
    stored = acceptance_module._read_json(root / "config.json")
    assert "rf3" not in stored
    assert stored["evaluator"]["enabled_aliases"] == ["AF3"]
    assert stored["evaluator"]["evaluator_terms"]["AF3"]["_type"] == (
        "typed_acceptance_fake"
    )
    assert stored["mutator"]["enabled_aliases"] == ["MCMC"]
    assert stored["mutator"]["mutator_terms"]["MCMC"]["_type"] == "MCMC"

    restored = acceptance_module.load_acceptance_config(root)
    assert restored.evaluator.evaluator_terms.AF3._type == "typed_acceptance_fake"
    assert restored.mutator.mutator_terms.MCMC._type == "MCMC"


def test_native_request_budget_rejects_before_runner_callback(
    acceptance_module,
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "budget"
    cfg = _config(acceptance_module, root)
    acceptance_module._initialize_root(root, cfg)
    acceptance_module._atomic_json(
        root / "native_request_budget.json",
        {
            "schema": acceptance_module.NATIVE_REQUEST_BUDGET_SCHEMA,
            "limit": 4,
            "used": 4,
        },
    )
    with pytest.raises(
        acceptance_module.AcceptanceError,
        match="before model callback",
    ):
        acceptance_module._reserve_native_request(root)
    assert not (root / "native_inference_requests.jsonl").exists()

    acceptance_module._atomic_json(
        root / "native_request_budget.json",
        {
            "schema": acceptance_module.NATIVE_REQUEST_BUDGET_SCHEMA,
            "limit": 4,
            "used": 0,
        },
    )
    global_budget = tmp_path / "global-native-request-budget.json"
    acceptance_module._atomic_json(
        global_budget,
        {
            "schema": acceptance_module.NATIVE_REQUEST_BUDGET_SCHEMA,
            "limit": 20,
            "used": 20,
        },
    )
    monkeypatch.setenv(
        acceptance_module.GLOBAL_NATIVE_REQUEST_BUDGET_ENV,
        str(global_budget),
    )
    monkeypatch.setenv(
        acceptance_module.GLOBAL_NATIVE_REQUEST_LIMIT_ENV,
        "20",
    )
    with pytest.raises(
        acceptance_module.AcceptanceError,
        match="before model callback",
    ):
        acceptance_module._reserve_native_request(root)
    assert acceptance_module._read_json(
        root / "native_request_budget.json"
    )["used"] == 0


def test_native_request_count_requires_owner_status_deltas(
    acceptance_module,
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "owner-status-required"
    monkeypatch.setattr(
        acceptance_module,
        "_native_owner_status",
        lambda _evaluator: None,
    )
    with pytest.raises(
        acceptance_module.AcceptanceError,
        match="owner status deltas are required",
    ):
        acceptance_module.run_typed_acceptance(
            root,
            config=_config(acceptance_module, root),
        )


def test_prepare_builds_real_operation_registry_without_loading(
    acceptance_module,
    tmp_path,
):
    root = tmp_path / "prepare"
    prepared = acceptance_module.prepare_typed_acceptance(
        root,
        config=_config(acceptance_module, root),
    )
    assert prepared["status"] == "prepared"
    assert prepared["model_owner_loaded"] == {
        "evaluator": False,
        "mutator": False,
    }
    assert prepared["expected_native_requests_per_complete_trajectory"] == 4
    assert (root / "prepare.json").is_file()
    assert not (root / "native_inference_requests.jsonl").exists()


def test_real_registry_codes_runtime_designrun_and_manifest_closure(
    acceptance_module,
    tmp_path,
):
    root = tmp_path / "uninterrupted"
    cfg = _config(acceptance_module, root)
    result = acceptance_module.run_typed_acceptance(root, config=cfg)

    assert result["route"] == {
        "frontdoor": "Astrolabe(DesignRun)",
        "trial": "RefinementTrial",
        "policy": "PhasedBeamSearchPolicy",
        "runtime": "Runtime",
        "loss": "LossEngine",
    }
    assert result["profile"]["seed"] == 17
    assert result["profile"]["num_samples"] == 1
    assert result["profile"]["num_recycles"] == 1
    assert result["profile"]["empty_msa"] is True
    assert result["prediction_invocations"]["count"] == 2
    assert result["observed_gpu_requests"]["expected"] == 4
    assert result["observed_gpu_requests"]["observed"] == 4
    assert result["observed_gpu_requests"]["source"] == (
        "NativeModelClient owner request/completed deltas"
    )
    assert result["observed_gpu_requests"]["native_status_request_delta"] == 4
    assert result["observed_gpu_requests"]["native_status_completed_delta"] == 4
    assert result["observed_gpu_requests"]["native_status_entries"] == 2
    assert result["observed_gpu_requests"]["run_inference_callbacks"] == 4
    assert result["structure_samples"]["expected"] == 4
    assert result["structure_samples"]["count"] == 4
    completed_invocations = acceptance_module._read_jsonl(
        root / "prediction_invocations.jsonl"
    )
    assert all(
        item["native_status_after"]["process"]["pid"] == os.getpid()
        for item in completed_invocations
        if item["event"] == "prediction_invocation_completed"
    )
    assert result["policy"] == {
        "root_evaluations": 1,
        "evaluations": 2,
        "proposals": 1,
        "duplicates": 0,
        "generation": 1,
        "finished": True,
    }
    assert result["fixture"]["assemblies"] == ["A", "B"]

    root_record, child_record = result["evaluations"]
    shared_name, root_shared = _assignment(root_record["candidate"], "GHIK")
    child_shared = child_record["candidate"]["assignments"][shared_name]
    assert child_shared["sequence"] != "GHIK"

    fixed = [
        domain
        for domain in result["fixture"]["domains"]
        if domain["fixed_tokens"]
    ]
    assert sorted(tuple(item["fixed_tokens"]) for item in fixed) == [
        (0, 1, 2),
        (0, 1, 2),
    ]
    root_assignments = root_record["candidate"]["assignments"]
    child_assignments = child_record["candidate"]["assignments"]
    for variable, value in root_assignments.items():
        if variable == shared_name:
            continue
        assert child_assignments[variable] == value

    prediction_attempts = [
        item
        for item in result["attempts"]
        if item["round"].endswith("/predict")
    ]
    mutation_attempts = [
        item for item in result["attempts"] if item["round"] == "propose"
    ]
    assert len(prediction_attempts) == 2
    assert len(mutation_attempts) == 1
    assert mutation_attempts[0]["status"] == "completed"
    for attempt in prediction_attempts:
        assert attempt["status"] == "completed"
        assert "folding-input" in attempt["manifest_roles"]
        assert "runtime-config" in attempt["manifest_roles"]
        assert "model" in attempt["manifest_roles"]
        assert len(attempt["folding"]) == 2
        folding = {item["assembly"]: item for item in attempt["folding"]}
        assert set(folding) == {"A", "B"}
        assert folding["A"]["seed"] == [17]
        assert folding["B"]["seed"] == [17]
        shared_sequence = (
            "GHIK"
            if attempt["round"] == "root/predict"
            else child_shared["sequence"]
        )
        assert [chain["sequence"] for chain in folding["A"]["chains"]] == [
            shared_sequence + "ACD"
        ]
        assert [chain["sequence"] for chain in folding["B"]["chains"]] == [
            shared_sequence + "MNP"
        ]
        assert all(
            chain["unpaired_msa"] in (None, "")
            for item in folding.values()
            for chain in item["chains"]
        )
        assert all(
            chain["paired_msa"] in (None, "")
            for item in folding.values()
            for chain in item["chains"]
        )

    assert len(result["rng"]["receipts"]) == 2
    for record in result["evaluations"]:
        assert len(record["structures"]) == 2
        array_path = root / record["arrays"]["path"]
        assert array_path.is_file()
        assert hashlib.sha256(array_path.read_bytes()).hexdigest() == (
            record["arrays"]["sha256"]
        )
    assert (root / "runtime" / "journal.sqlite").is_file()
    assert (root / "runtime" / "selected.json").is_file()


def test_receipt_safe_crash_resume_matches_exact_uninterrupted_evidence(
    acceptance_module,
    tmp_path,
):
    uninterrupted = tmp_path / "uninterrupted"
    baseline = acceptance_module.run_typed_acceptance(
        uninterrupted,
        config=_config(acceptance_module, uninterrupted),
    )

    for point in ("root", "proposal", "prediction", "outcome"):
        crashed = tmp_path / point
        with pytest.raises(acceptance_module.ControlledCrash) as raised:
            acceptance_module.run_typed_acceptance(
                crashed,
                config=_config(acceptance_module, crashed),
                crash_after=point,
            )
        assert raised.value.point == point
        marker = acceptance_module._read_json(
            crashed / "controlled_crash.json"
        )
        assert marker["durable_before_exit"] is True
        assert marker["exit_mode"] == "controlled_exception"

        resumed = acceptance_module.run_typed_acceptance(
            crashed,
            config=acceptance_module.load_acceptance_config(crashed),
        )
        comparison = acceptance_module.compare_acceptance(uninterrupted, crashed)
        assert comparison["matched"] is True
        assert resumed["prediction_invocations"]["count"] == 2
        assert resumed["observed_gpu_requests"]["observed"] == 4
        assert resumed["structure_samples"]["count"] == 4
        assert all(comparison["checks"].values())
        assert acceptance_module._read_json(
            crashed / "native_request_budget.json"
        )["used"] == 4
        assert resumed["rng"]["final"] == baseline["rng"]["final"]
        assert sorted(
            (item["round"], item["before"], item["after"])
            for item in resumed["rng"]["receipts"]
        ) == sorted(
            (item["round"], item["before"], item["after"])
            for item in baseline["rng"]["receipts"]
        )


@pytest.mark.parametrize(
    ("point", "expected_requests"),
    (("root", 2), ("proposal", 2), ("prediction", 4), ("outcome", 4)),
)
def test_hard_cut_calls_process_exit_only_after_durable_receipt(
    acceptance_module,
    tmp_path,
    monkeypatch,
    point,
    expected_requests,
):
    root = tmp_path / f"hard-cut-{point}"
    exits = []

    def observed_exit(code):
        exits.append(int(code))
        raise SystemExit(code)

    monkeypatch.setattr(acceptance_module.os, "_exit", observed_exit)
    with pytest.raises(SystemExit) as raised:
        acceptance_module.run_typed_acceptance(
            root,
            config=_config(acceptance_module, root),
            crash_after=point,
            hard_exit=True,
        )

    assert raised.value.code == 75
    assert exits == [75]
    marker = acceptance_module._read_json(root / "controlled_crash.json")
    assert marker["durable_before_exit"] is True
    assert marker["exit_mode"] == "os_exit_75"
    assert marker["point"] == point

    with sqlite3.connect(root / "runtime" / "journal.sqlite") as connection:
        rows = connection.execute(
            "SELECT value FROM records WHERE kind='attempt'"
        ).fetchall()
    attempts = [json.loads(value) for (value,) in rows]
    predictions = [
        item
        for item in attempts
        if item["component"] == "AF3" and item["status"] == "completed"
    ]
    assert len(predictions) == expected_requests // 2
    native_requests = acceptance_module._read_jsonl(
        root / "native_inference_requests.jsonl"
    )
    assert len(native_requests) == expected_requests
    assert acceptance_module._read_json(
        root / "native_request_budget.json"
    )["used"] == expected_requests
