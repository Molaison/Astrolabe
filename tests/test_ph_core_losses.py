from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.evaluators.utils import StandardScores
from astrolabe.core import global_config
from astrolabe.losses import ph_core
from astrolabe.losses.base import LossEngine, loss_registry


class _FakePose:
    def __init__(self, content: str, apo_path: Path):
        self.content = content
        self.apo_path = apo_path


class _FakeBackend:
    def __init__(self, gap=2.0):
        self.gap = gap
        self.calls = []

    def score_states(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "gap": self.gap,
            "neutral_energy": -3.0,
            "protonated_energy": -1.0,
            "center_states": {str(index): "C" for index in kwargs["histidine_positions"]},
        }


@pytest.fixture
def runtime_context(monkeypatch):
    state = global_config.RuntimeState()
    monkeypatch.setattr(global_config, "_current_state", state)
    state.init_const("stage_manager", SimpleNamespace(current_stage="test"))
    state.init_const("protomers", {})
    state.set_variable("step", 1)
    return global_config.build_runtime_context(state)


@pytest.fixture(autouse=True)
def _clear_ph_core_caches():
    ph_core._clear_network_cache()
    ph_core._release_cached.cache_clear()
    yield
    ph_core._clear_network_cache()
    ph_core._release_cached.cache_clear()


def _eval_result(
    tmp_path: Path,
    content: str = "complex-v1",
    filename: str = "prediction.cif",
):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    result = StandardScores(names=["AB"], plddt={}, complex_paths={"AB": path})
    return result, path


def _fake_evidence(pose: _FakePose, *, centers, quality, unpaired=0, status="network"):
    selected = None
    edges = []
    if status == "network":
        edges = [
            {
                "histidine_residue": centers[0],
                "partner_residue": centers[1] if len(centers) > 1 else 3,
                "partner_type": "HIS" if len(centers) > 1 else "ARG",
                "histidine_ring_atom": "ND1",
                "partner_heavy_atom": "ND1" if len(centers) > 1 else "NH1",
                "energy": -0.7,
                "mean_atomic_depth": 3.0,
                "energy_quality": 1.0,
                "depth_quality": 0.75,
                "quality": quality,
                "histidine_original_index": centers[0] + 10,
                "partner_original_index": (centers[1] if len(centers) > 1 else 3) + 10,
            }
        ]
        selected = {
            "quality": quality,
            "n_centers": len(centers),
            "histidine_positions": list(centers),
            "histidine_original_indices": [index + 10 for index in centers],
            "min_energy": -0.7,
            "edges": edges,
        }
    return {
        "schema": "ph_core_network_v1",
        "source_path": str(pose.content),
        "source_sha256": pose.content,
        "normalized_apo_path": str(pose.apo_path),
        "binder_chain": "A",
        "binder_sequence": "AHHAARG",
        "binder_length": 7,
        "histidine_positions_apo": list(centers),
        "histidine_positions_zero_based": [index - 1 for index in centers],
        "original_residue_mapping": [
            {
                "apo_index": index,
                "original_pose_index": index + 10,
                "original_chain": "D",
                "original_number": index + 10,
                "original_insertion_code": "",
            }
            for index in range(1, 8)
        ],
        "parameters": {},
        "neutral_histidine_protocol": "test",
        "status": status,
        "neutral": float(quality) if status == "network" else 0.0,
        "unpaired": unpaired,
        "selected_component": selected,
        "all_components": [selected] if selected else [],
        "unpaired_histidines": [],
        "edge_count": len(edges),
    }


def _patch_fake_network(monkeypatch, tmp_path: Path, builder):
    monkeypatch.setattr(
        ph_core,
        "_load_pose",
        lambda path: _FakePose(
            Path(path).read_text(encoding="utf-8"),
            tmp_path
            / (
                "normalized-apo-"
                + Path(path).read_text(encoding="utf-8").replace("/", "_")
                + ".pdb"
            ),
        ),
    )
    monkeypatch.setattr(ph_core, "_build_network_evidence", builder)


def test_registered_with_expected_interface():
    assert "ph_core_network" in loss_registry.valid_losses
    metadata = loss_registry.get_metadata("ph_core_network")
    assert metadata.required_params == ["binder_chain"]
    assert set(metadata.optional_params) == {
        "metric",
        "hbond_scale",
        "depth_scale",
        "probe_radius",
        "resolution",
        "gap_scale",
        "gap_reference",
        "runner_key",
        "runner_config",
    }


def test_hh_component_passes_both_zero_based_centres_to_shared_backend(monkeypatch, tmp_path):
    result, _path = _eval_result(tmp_path)
    calls = []

    def builder(pose, **_kwargs):
        return _fake_evidence(pose, centers=[2, 5], quality=0.75)

    backend = _FakeBackend(gap=2.0)

    def get_runner(*, config, key):
        calls.append((config, key))
        return backend

    _patch_fake_network(monkeypatch, tmp_path, builder)
    monkeypatch.setattr(ph_core, "_get_shared_proton_runner", get_runner)

    value = ph_core.ph_core_network(
        result,
        "A",
        metric="release",
        runner_key="shared",
        runner_config={"seed": 7},
    )["AB"]

    expected = 0.75 * np.tanh((2.0 / 2) / 1.0)
    assert value == pytest.approx(expected)
    assert calls == [({"seed": 7}, "shared")]
    assert backend.calls[0]["histidine_positions"] == [1, 4]
    assert backend.calls[0]["binder_chain"] == "A"
    assert backend.calls[0]["chain_sequences"] == {"A": "AHHAARG"}
    audit = result.ph_core_network_audits["AB"]
    assert audit["selected_component"]["histidine_positions"] == [2, 5]
    assert audit["selected_component"]["histidine_original_indices"] == [12, 15]
    assert audit["gap"] == 2.0
    assert audit["state_energies"] == {"neutral": -3.0, "protonated": -1.0}
    json.dumps(audit)


def test_metrics_share_network_and_release_cache(monkeypatch, tmp_path):
    result, _path = _eval_result(tmp_path)
    build_calls = []

    def builder(pose, **_kwargs):
        build_calls.append(pose.content)
        return _fake_evidence(pose, centers=[2, 5], quality=0.8, unpaired=1)

    backend = _FakeBackend(gap=1.5)
    _patch_fake_network(monkeypatch, tmp_path, builder)
    monkeypatch.setattr(
        ph_core,
        "_get_shared_proton_runner",
        lambda **_kwargs: backend,
    )

    assert ph_core.ph_core_network(result, "A", metric="neutral")["AB"] == pytest.approx(0.8)
    assert ph_core.ph_core_network(result, "A", metric="unpaired")["AB"] == 1.0
    first = ph_core.ph_core_network(result, "A", metric="release")["AB"]
    second = ph_core.ph_core_network(result, "A", metric="release")["AB"]

    assert first == pytest.approx(second)
    assert len(build_calls) == 1
    assert len(backend.calls) == 1


def test_three_metric_audits_are_preserved_without_release_overwrite(monkeypatch, tmp_path):
    result, _path = _eval_result(tmp_path)
    _patch_fake_network(
        monkeypatch,
        tmp_path,
        lambda pose, **_kwargs: _fake_evidence(
            pose,
            centers=[2, 5],
            quality=0.8,
            unpaired=1,
        ),
    )
    backend = _FakeBackend(gap=2.0)
    monkeypatch.setattr(
        ph_core,
        "_get_shared_proton_runner",
        lambda **_kwargs: backend,
    )

    ph_core.ph_core_network(result, "A", metric="neutral")
    ph_core.ph_core_network(result, "A", metric="release")
    ph_core.ph_core_network(result, "A", metric="unpaired")

    record = result.ph_core_network_audits["AB"]
    assert set(record["metrics"]) == {"neutral", "release", "unpaired"}
    release = record["metrics"]["release"]
    assert release["gap"] == 2.0
    assert release["state_energies"] == {"neutral": -3.0, "protonated": -1.0}
    assert release["proton_result"]["neutral_energy"] == -3.0
    # The top-level latest record remains backward compatible.
    assert record["metric"] == "unpaired"


def test_content_change_recomputes_network_and_migrates_centres(monkeypatch, tmp_path):
    result, path = _eval_result(tmp_path, "complex-v1")
    observed = []

    def builder(pose, **_kwargs):
        observed.append(pose.content)
        centers = [2, 5] if pose.content == "complex-v1" else [4]
        return _fake_evidence(
            pose,
            centers=centers,
            quality=0.8 if len(centers) == 2 else 0.6,
        )

    backend = _FakeBackend()
    _patch_fake_network(monkeypatch, tmp_path, builder)
    monkeypatch.setattr(
        ph_core,
        "_get_shared_proton_runner",
        lambda **_kwargs: backend,
    )
    ph_core.ph_core_network(result, "A", metric="release")
    path.write_text("complex-v2", encoding="utf-8")
    ph_core.ph_core_network(result, "A", metric="release")

    assert observed == ["complex-v1", "complex-v2"]
    assert backend.calls[0]["histidine_positions"] == [1, 4]
    assert backend.calls[1]["histidine_positions"] == [3]


def test_arg_lys_bridge_joins_histidines_into_one_component():
    edges = [
        {
            "histidine_residue": 2,
            "partner_residue": 8,
            "energy": -0.8,
            "quality": 0.8,
        },
        {
            "histidine_residue": 5,
            "partner_residue": 8,
            "energy": -0.6,
            "quality": 0.6,
        },
    ]

    components = ph_core._component_candidates(None, {2, 5}, edges)

    assert len(components) == 1
    assert components[0]["histidine_positions"] == [2, 5]
    assert components[0]["partner_positions"] == [8]
    assert components[0]["component_positions"] == [2, 5, 8]
    assert components[0]["n_centers"] == 2
    assert components[0]["quality"] == pytest.approx(0.6)


def test_mtime_is_audit_only_and_does_not_break_content_cache(monkeypatch, tmp_path):
    result, path = _eval_result(tmp_path, "same-content")
    build_calls = []

    def builder(pose, **_kwargs):
        build_calls.append(pose.content)
        return _fake_evidence(pose, centers=[2, 5], quality=0.8)

    _patch_fake_network(monkeypatch, tmp_path, builder)
    first = ph_core.ph_core_network(result, "A", metric="neutral")["AB"]
    first_mtime = result.ph_core_network_audits["AB"]["source_mtime_ns"]

    requested_mtime = first_mtime + 2_000_000_000
    os.utime(path, ns=(path.stat().st_atime_ns, requested_mtime))
    observed_mtime = path.stat().st_mtime_ns
    assert observed_mtime != first_mtime
    second = ph_core.ph_core_network(result, "A", metric="neutral")["AB"]
    second_mtime = result.ph_core_network_audits["AB"]["source_mtime_ns"]

    assert first == second == pytest.approx(0.8)
    assert len(build_calls) == 1
    assert second_mtime == observed_mtime


def test_no_network_returns_zero_without_backend(monkeypatch, tmp_path):
    result, _path = _eval_result(tmp_path)

    def builder(pose, **_kwargs):
        return _fake_evidence(pose, centers=[], quality=0.0, unpaired=2, status="no_network")

    def forbidden_backend(**_kwargs):
        raise AssertionError("no-network release must not call Proton")

    _patch_fake_network(monkeypatch, tmp_path, builder)
    monkeypatch.setattr(ph_core, "_get_shared_proton_runner", forbidden_backend)

    assert ph_core.ph_core_network(result, "A", metric="neutral")["AB"] == 0.0
    assert ph_core.ph_core_network(result, "A", metric="release")["AB"] == 0.0
    assert result.ph_core_network_audits["AB"]["gap"] is None
    assert result.ph_core_network_audits["AB"]["original_G"] is None
    assert result.ph_core_network_audits["AB"]["status"] == "no_network"


def test_unpaired_survives_no_network_branch(monkeypatch, tmp_path):
    result, _path = _eval_result(tmp_path)

    def builder(pose, **_kwargs):
        return _fake_evidence(pose, centers=[], quality=0.0, unpaired=2, status="no_network")

    _patch_fake_network(monkeypatch, tmp_path, builder)
    monkeypatch.setattr(
        ph_core,
        "_get_shared_proton_runner",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unpaired must not call Proton")
        ),
    )

    assert ph_core.ph_core_network(result, "A", metric="unpaired")["AB"] == 2.0
    assert ph_core.ph_core_network(result, "A", metric="neutral")["AB"] == 0.0
    assert ph_core.ph_core_network(result, "A", metric="release")["AB"] == 0.0


def test_nonfinite_backend_gap_raises(monkeypatch, tmp_path):
    result, _path = _eval_result(tmp_path)
    _patch_fake_network(
        monkeypatch,
        tmp_path,
        lambda pose, **_kwargs: _fake_evidence(pose, centers=[2, 5], quality=0.8),
    )
    monkeypatch.setattr(
        ph_core,
        "_get_shared_proton_runner",
        lambda **_kwargs: _FakeBackend(gap=float("nan")),
    )

    with pytest.raises(ValueError, match="non-finite gap"):
        ph_core.ph_core_network(result, "A", metric="release")


def test_score_expression_and_acceptance_gate_use_scalar_only(monkeypatch, tmp_path, runtime_context):
    result, _path = _eval_result(tmp_path)
    _patch_fake_network(
        monkeypatch,
        tmp_path,
        lambda pose, **_kwargs: _fake_evidence(pose, centers=[2, 5], quality=0.8),
    )
    engine = LossEngine(
        {
            "loss_terms": {
                "core": {
                    "_type": "ph_core_network",
                    "binder_chain": "A",
                    "metric": "neutral",
                },
                "unpaired": {
                    "_type": "ph_core_network",
                    "binder_chain": "A",
                    "metric": "unpaired",
                },
            }
        },
        runtime_context=runtime_context,
        validate_config=False,
    )
    engine.setup(
        "core",
        None,
        None,
        acceptance_gate="core >= 0.8 and unpaired == 0",
        validate_config=False,
    )

    score, detail = engine.compute_loss(result, runtime_config=None)

    assert score == pytest.approx(0.8)
    assert detail["core"] == pytest.approx(0.8)
    assert detail["unpaired"] == 0.0
    assert engine.passes_acceptance_gate(detail) is True


def test_example_objective_release_direction_and_no_his_gate(monkeypatch, tmp_path, runtime_context):
    positive, positive_path = _eval_result(
        tmp_path,
        "positive-g",
        filename="positive.cif",
    )
    negative, negative_path = _eval_result(
        tmp_path,
        "negative-g",
        filename="negative.cif",
    )
    no_his, no_his_path = _eval_result(
        tmp_path,
        "no-his",
        filename="no-his.cif",
    )

    def builder(pose, **_kwargs):
        if pose.content == "no-his":
            return _fake_evidence(
                pose,
                centers=[],
                quality=0.0,
                unpaired=0,
                status="no_network",
            )
        return _fake_evidence(
            pose,
            centers=[2, 5],
            quality=0.8,
            unpaired=1,
        )

    class DirectionalBackend:
        def __init__(self):
            self.paths = []

        def score_states(self, **kwargs):
            self.paths.append(kwargs["pdb_path"])
            gap = 2.0 if "positive-g" in Path(kwargs["pdb_path"]).name else -2.0
            return {
                "gap": gap,
                "neutral_energy": -3.0,
                "protonated_energy": -1.0,
            }

    backend = DirectionalBackend()
    _patch_fake_network(monkeypatch, tmp_path, builder)
    monkeypatch.setattr(
        ph_core,
        "_get_shared_proton_runner",
        lambda **_kwargs: backend,
    )
    engine = LossEngine(
        {
            "loss_terms": {
                "core_neutral": {
                    "_type": "ph_core_network",
                    "binder_chain": "A",
                    "metric": "neutral",
                },
                "core_release": {
                    "_type": "ph_core_network",
                    "binder_chain": "A",
                    "metric": "release",
                },
                "core_unpaired": {
                    "_type": "ph_core_network",
                    "binder_chain": "A",
                    "metric": "unpaired",
                },
            }
        },
        runtime_context=runtime_context,
        validate_config=False,
    )
    engine.setup(
        "0.5 + 0.2 * core_neutral + 0.1 * core_release - 0.1 * core_unpaired",
        None,
        None,
        acceptance_gate="core_neutral >= 0 and core_unpaired >= 0",
        validate_config=False,
    )

    positive_score, positive_detail = engine.compute_loss(
        positive,
        runtime_config=None,
    )
    negative_score, negative_detail = engine.compute_loss(
        negative,
        runtime_config=None,
    )
    no_his_score, no_his_detail = engine.compute_loss(
        no_his,
        runtime_config=None,
    )

    assert positive_detail["core_neutral"] == negative_detail["core_neutral"] == pytest.approx(0.8)
    assert positive_detail["core_unpaired"] == negative_detail["core_unpaired"] == pytest.approx(1.0)
    assert positive_detail["core_release"] > 0.0
    assert negative_detail["core_release"] < 0.0
    assert positive_score > negative_score
    assert len(backend.paths) == 2
    assert no_his_score == pytest.approx(0.5)
    assert no_his_detail["core_release"] == 0.0
    assert engine.passes_acceptance_gate(no_his_detail) is True


def test_real_pyrosetta_no_histidine_is_evidence_based_and_backend_free(tmp_path):
    pr = pytest.importorskip("pyrosetta")
    from astrolabe.losses.design_screening import _ensure_pyrosetta

    _ensure_pyrosetta()
    pose = pr.pose_from_sequence("AAA")
    pr.rosetta.core.pose.append_pose_to_pose(pose, pr.pose_from_sequence("AAA"), True)
    info = pr.rosetta.core.pose.PDBInfo(pose)
    for index in range(1, 7):
        info.chain(index, "A" if index <= 3 else "B")
        info.number(index, index + 20)
    pose.pdb_info(info)
    path = tmp_path / "no_his.pdb"
    pose.dump_pdb(str(path))
    result = StandardScores(names=["AB"], plddt={}, complex_paths={"AB": path})

    def forbidden_backend(**_kwargs):
        raise AssertionError("no-histidine release must not call Proton")

    original_getter = ph_core._get_shared_proton_runner
    ph_core._get_shared_proton_runner = forbidden_backend
    try:
        assert ph_core.ph_core_network(result, "B", metric="neutral")["AB"] == 0.0
        assert ph_core.ph_core_network(result, "B", metric="release")["AB"] == 0.0
    finally:
        ph_core._get_shared_proton_runner = original_getter

    audit = result.ph_core_network_audits["AB"]
    assert audit["status"] == "no_network"
    assert audit["histidine_positions_apo"] == []
    assert audit["original_residue_mapping"][0]["original_chain"] == "B"
    assert audit["original_residue_mapping"][0]["original_number"] == 24
    normalized = ph_core._load_pose(audit["normalized_apo_path"])
    assert normalized.pdb_info().chain(1) == "A"
    assert normalized.pdb_info().number(1) == 1
