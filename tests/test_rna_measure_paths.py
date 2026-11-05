from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from astrolabe.design.native_scores import restore_scores, snapshot_scores
from astrolabe.design.store import DataStore
from astrolabe.evaluators.utils import StandardScores
from astrolabe.rna_beam import _measure_native_scores, _resolve_bivalent_raw_output


_CIF = """data_model
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 0.0 0.0 0.0
#
"""


def test_restored_bivalent_scores_resolve_original_alphafold_run_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "configured-output"
    raw_root = output / "raw_model_results"
    run_name = "step_000_ABC"
    raw_run = raw_root / "alphafold3" / run_name
    raw_model = raw_run / f"{run_name}_model.cif"
    raw_confidence = raw_run / f"{run_name}_confidences.json"
    raw_run.mkdir(parents=True)
    raw_model.write_text(_CIF, encoding="utf-8")
    raw_confidence.write_text(
        json.dumps({"chain_iptm": [0.5, 0.5, 0.5]}),
        encoding="utf-8",
    )
    scores = StandardScores(
        names=["ABC"],
        plddt={"ABC": [np.asarray([0.8, 0.7], dtype=np.float32)]},
        complex_paths={"ABC": raw_model},
    )

    data = DataStore()
    snapshot = snapshot_scores(scores, data)
    restored = restore_scores(
        snapshot,
        data,
        output / ".native_refinement" / "measurement-input_evaluate_0",
    )

    restored_model = restored.complex_paths["ABC"]
    assert restored_model.name == f"ABC-{run_name}_model.cif"
    output = _resolve_bivalent_raw_output(restored_model, "ABC", raw_root)
    assert output.identifier == run_name
    measured_model = output.model_path
    measured_confidence = output.confidence_path
    assert measured_model.is_file()
    assert measured_confidence.is_file()
    assert measured_model.stem.removesuffix("_model") == (
        measured_confidence.stem.removesuffix("_confidences")
    )
    restored_confidence = restored_model.with_name(
        f"{restored_model.stem.removesuffix('_model')}_confidences.json"
    )
    assert restored_confidence.name == (
        f"ABC-{run_name}_confidences.json"
    )


def _restored_model(
    tmp_path: Path,
    *,
    assembly: str,
    run_name: str,
) -> tuple[Path, Path]:
    output = tmp_path / "configured-output"
    raw_root = output / "raw_model_results"
    raw_run = raw_root / "alphafold3" / run_name
    raw_model = raw_run / f"{run_name}_model.cif"
    raw_confidence = raw_run / f"{run_name}_confidences.json"
    raw_run.mkdir(parents=True)
    raw_model.write_text(_CIF, encoding="utf-8")
    raw_confidence.write_text(
        json.dumps({"chain_iptm": [0.5, 0.5, 0.5]}),
        encoding="utf-8",
    )
    scores = StandardScores(
        names=[assembly],
        plddt={assembly: [np.asarray([0.8, 0.7], dtype=np.float32)]},
        complex_paths={assembly: raw_model},
    )
    data = DataStore()
    snapshot = snapshot_scores(scores, data)
    restored = restore_scores(
        snapshot,
        data,
        output / ".native_refinement" / "measurement-input_evaluate_0",
    )
    return restored.complex_paths[assembly], raw_root


def test_transport_prefix_is_removed_once_for_dashed_assembly(
    tmp_path: Path,
) -> None:
    model, raw_root = _restored_model(
        tmp_path,
        assembly="A-B",
        run_name="A-B-step_000",
    )

    output = _resolve_bivalent_raw_output(model, "A-B", raw_root)

    assert model.name == "A-B-A-B-step_000_model.cif"
    assert output.identifier == "A-B-step_000"
    assert output.model_path == (
        raw_root / "alphafold3/A-B-step_000/A-B-step_000_model.cif"
    )


def test_transport_trailing_metadata_does_not_change_atom_identity(
    tmp_path: Path,
) -> None:
    model, raw_root = _restored_model(
        tmp_path,
        assembly="ABC",
        run_name="step_000",
    )
    with model.open("a", encoding="utf-8") as handle:
        handle.write('_dummy.title ""\n')

    output = _resolve_bivalent_raw_output(model, "ABC", raw_root)

    assert output.identifier == "step_000"
    assert model.read_bytes() != output.model_path.read_bytes()


@pytest.mark.parametrize(
    "extra",
    [
        '_dummy.title "not-empty"\n',
        '_unknown.title ""\n',
    ],
)
def test_transport_rejects_noncanonical_or_unknown_metadata(
    tmp_path: Path,
    extra: str,
) -> None:
    model, raw_root = _restored_model(
        tmp_path,
        assembly="ABC",
        run_name="step_000",
    )
    with model.open("a", encoding="utf-8") as handle:
        handle.write(extra)

    with pytest.raises(ValueError, match=r"non-empty _dummy.title|no content-matching"):
        _resolve_bivalent_raw_output(model, "ABC", raw_root)


def test_same_named_transport_and_run_identifier_fails_as_ambiguous(
    tmp_path: Path,
) -> None:
    model, raw_root = _restored_model(
        tmp_path,
        assembly="ABC",
        run_name="step_000",
    )
    original = raw_root / "alphafold3/step_000/step_000_model.cif"
    transport_looking = raw_root / "alphafold3/ABC-step_000"
    transport_looking.mkdir()
    (transport_looking / "ABC-step_000_model.cif").write_bytes(
        original.read_bytes()
    )
    (transport_looking / "ABC-step_000_confidences.json").write_text(
        '{"chain_iptm": [0.5, 0.5, 0.5]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous AlphaFold 3 raw output"):
        _resolve_bivalent_raw_output(model, "ABC", raw_root)


def test_full_measurement_body_restores_metadata_and_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, raw_root = _restored_model(
        tmp_path,
        assembly="ABC",
        run_name="step_000",
    )
    raw_model = raw_root / "alphafold3/step_000/step_000_model.cif"
    data = DataStore()
    scores = StandardScores(
        names=["ABC"],
        plddt={"ABC": [np.asarray([0.8, 0.7], dtype=np.float32)]},
        complex_paths={"ABC": raw_model},
    )
    restored = restore_scores(
        snapshot_scores(scores, data),
        data,
        tmp_path / "measurement-input_full",
    )
    metrics = {
        "dual_interface": 0.5,
        "repeat_cohesion": 0.75,
        "weak_arm_plddt": 0.25,
        "interchain_clashes": 0,
        "structure_review_pass": True,
    }

    def fake_measure(model_path: Path, confidence_path: Path):
        assert model_path == raw_root / "alphafold3/step_000/step_000_model.cif"
        assert confidence_path.is_file()
        return dict(metrics)

    monkeypatch.setattr(
        "astrolabe.analysis.rna_geometry.measure_bivalent_model",
        fake_measure,
    )
    measurement = _measure_native_scores(
        scores=restored,
        candidate_label="ACGU",
        oligo="ABC",
        raw_model_results_dir=raw_root,
        objective_weights={
            "weak_interface": 2.0,
            "repeat_contacts": 1.0,
            "repeat_plddt": 0.5,
            "clashes": 0.5,
        },
        bivalent=True,
    )

    sidecar = Path(measurement.metrics["confidence_path"])
    assert measurement.metrics["rna_sequence"] == "ACGU"
    assert measurement.metrics["mean_plddt"] == pytest.approx(0.75)
    assert measurement.score == pytest.approx(1.875)
    assert sidecar.is_file()
    assert sidecar.read_bytes() == (
        raw_root / "alphafold3/step_000/step_000_confidences.json"
    ).read_bytes()
