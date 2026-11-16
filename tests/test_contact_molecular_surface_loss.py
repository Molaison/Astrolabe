from __future__ import annotations

from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.losses.interaction import contact_molecular_surface
from astrolabe.losses.interaction import contact_molecular_surface_norm
from astrolabe.losses.interaction import _chain_relative_residue_indices
from astrolabe.losses.interaction import interface_buried_unsatisfied_hbonds
from astrolabe.losses.interaction import ligand_buried_unsat_hbonds
from astrolabe.losses.interaction import normalize_contact_molecular_surface


def _eval_result(oligo_name: str = "AB", pose=None) -> SimpleNamespace:
    return SimpleNamespace(
        names=[oligo_name],
        complex_poses=SimpleNamespace(poses={oligo_name: object() if pose is None else pose}),
    )


def test_contact_molecular_surface_returns_raw_value(monkeypatch) -> None:
    monkeypatch.setattr(
        "astrolabe.losses.interaction._require_pyrosetta",
        lambda: None,
    )
    monkeypatch.setattr(
        "astrolabe.losses.interaction._compute_contact_molecular_surface_between_chains",
        lambda pose, chain_a, chain_b, **kwargs: 250.0,
    )

    out = contact_molecular_surface(_eval_result(), backend="rosetta")

    assert out["AB"] == pytest.approx(250.0)


def test_contact_molecular_surface_norm_hits_target_value_at_400() -> None:
    assert normalize_contact_molecular_surface(400.0) == pytest.approx(0.8)


def test_contact_molecular_surface_norm_is_monotonic_and_saturating() -> None:
    score_250 = normalize_contact_molecular_surface(250.0)
    score_400 = normalize_contact_molecular_surface(400.0)
    score_550 = normalize_contact_molecular_surface(550.0)
    score_700 = normalize_contact_molecular_surface(700.0)

    assert 0.0 < score_250 < score_400 < score_550 < score_700 < 1.0
    assert (score_400 - score_250) > (score_550 - score_400)
    assert (score_550 - score_400) > (score_700 - score_550)


def test_contact_molecular_surface_norm_uses_runtime_loss(monkeypatch) -> None:
    monkeypatch.setattr(
        "astrolabe.losses.interaction._require_pyrosetta",
        lambda: None,
    )
    monkeypatch.setattr(
        "astrolabe.losses.interaction._compute_contact_molecular_surface_between_chains",
        lambda pose, chain_a, chain_b, **kwargs: 400.0,
    )

    out = contact_molecular_surface_norm(_eval_result(), backend="rosetta")

    assert out["AB"] == pytest.approx(0.8)


@pytest.mark.parametrize("options", [
    {"midpoint": 0}, {"midpoint": float("inf")}, {"midpoint_value": 1.5},
    {"midpoint_value": float("nan")}, {"lower_scale": -2}, {"upper_scale": "invalid"},
])
def test_normalization_settings_are_validated_before_computing_surface(monkeypatch, options):
    monkeypatch.setattr("astrolabe.losses.interaction.contact_molecular_surface",
                        lambda *args, **kwargs: pytest.fail("invalid parameters reached expensive calculation"))
    result = _eval_result()
    with pytest.raises(ValueError):
        contact_molecular_surface_norm(result, **options)
    assert getattr(result, "loss_backend_records", None) is None


def test_normalized_surface_records_formula_without_changing_prior_raw_evidence(monkeypatch):
    monkeypatch.setattr("astrolabe.losses.interaction._compute_contact_molecular_surface_between_chains",
                        lambda *args, **kwargs: 300.0)
    result = _eval_result()
    contact_molecular_surface(result, backend="rosetta")
    normalized = contact_molecular_surface_norm(result, backend="rosetta", midpoint=300.0,
                                                midpoint_value=0.7, lower_scale=100.0, upper_scale=50.0)
    raw_record, norm_record = result.loss_backend_records["AB"]
    assert "normalization" not in raw_record
    definition = norm_record["normalization"]
    assert definition["midpoint"] == 300.0
    assert definition["midpoint_value"] == 0.7
    assert definition["lower_scale"] == 100.0
    assert definition["upper_scale"] == 50.0
    assert definition["formula"]
    assert definition["output_units"] == "dimensionless"
    assert definition["raw_value"] == 300.0
    assert definition["output_value"] == normalized["AB"] == pytest.approx(0.7)


def test_contact_molecular_surface_passes_chain_relative_residue_range(
    monkeypatch,
) -> None:
    captured = {}

    monkeypatch.setattr(
        "astrolabe.losses.interaction._require_pyrosetta",
        lambda: None,
    )

    def fake_compute(pose, chain_a, chain_b, **kwargs):
        captured.update(kwargs)
        return 500.0

    monkeypatch.setattr(
        "astrolabe.losses.interaction._compute_contact_molecular_surface_between_chains",
        fake_compute,
    )

    out = contact_molecular_surface(
        _eval_result(pose=_FakePose(["A"] * 10 + ["B"] * 350)),
        backend="rosetta",
        protomer_a="A",
        protomer_b="B",
        protomer_b_residues="138-343",
        distance_weight=1.0,
    )

    assert out["AB"] == pytest.approx(500.0)
    assert captured["selected_residues_b"] == list(range(148, 354))
    assert captured["distance_weight"] == 1.0


def test_contact_molecular_surface_treats_target_chains_as_one_group(
    monkeypatch,
) -> None:
    captured = {}

    monkeypatch.setattr(
        "astrolabe.losses.interaction._require_pyrosetta",
        lambda: None,
    )

    def fake_compute(pose, chain_a, chain_b, **kwargs):
        captured["chain_a"] = chain_a
        captured["chain_b"] = chain_b
        return 600.0

    monkeypatch.setattr(
        "astrolabe.losses.interaction._compute_contact_molecular_surface_between_chains",
        fake_compute,
    )

    out = contact_molecular_surface(
        _eval_result("ABCD"),
        backend="rosetta",
        protomer_a="A",
        protomer_b=["B", "C", "D"],
    )

    assert out["ABCD"] == pytest.approx(600.0)
    assert captured["chain_a"] == ["A"]
    assert captured["chain_b"] == ["B", "C", "D"]


def test_contact_molecular_surface_loads_pose_from_complex_path(monkeypatch) -> None:
    loaded_pose = object()
    captured = {}

    monkeypatch.setattr(
        "astrolabe.losses.interaction._require_pyrosetta",
        lambda: None,
    )

    def fake_load_pose(path):
        captured["path"] = path
        return loaded_pose

    def fake_compute(pose, chain_a, chain_b, **kwargs):
        captured["pose"] = pose
        return 450.0

    monkeypatch.setattr(
        "astrolabe.losses.interaction.load_pose",
        fake_load_pose,
    )
    monkeypatch.setattr(
        "astrolabe.losses.interaction._compute_contact_molecular_surface_between_chains",
        fake_compute,
    )
    eval_result = SimpleNamespace(
        names=["ABCD"],
        complex_poses=None,
        complex_paths={"ABCD": "model.cif"},
    )

    out = contact_molecular_surface(
        eval_result,
        backend="rosetta",
        protomer_a="A",
        protomer_b=["B", "C", "D"],
    )

    assert out["ABCD"] == pytest.approx(450.0)
    assert captured["path"] == "model.cif"
    assert captured["pose"] is loaded_pose


def test_interface_buns_uses_grouped_target_and_complex_path(monkeypatch) -> None:
    loaded_pose = object()
    captured = {}

    monkeypatch.setattr(
        "astrolabe.losses.interaction._require_pyrosetta",
        lambda: None,
    )
    monkeypatch.setattr(
        "astrolabe.losses.interaction.load_pose",
        lambda path: loaded_pose,
    )

    def fake_compute(pose, binder_chains, target_chains, **kwargs):
        captured["pose"] = pose
        captured["binder_chains"] = binder_chains
        captured["target_chains"] = target_chains
        return 8.0

    monkeypatch.setattr(
        "astrolabe.losses.interaction._compute_interface_buried_unsatisfied_hbonds",
        fake_compute,
    )
    eval_result = SimpleNamespace(
        names=["ABCD"],
        complex_poses=None,
        complex_paths={"ABCD": "model.cif"},
    )

    out = interface_buried_unsatisfied_hbonds(
        eval_result,
        backend="rosetta",
        protomer_a="A",
        protomer_b=["B", "C", "D"],
    )

    assert out["ABCD"] == pytest.approx(8.0)
    assert captured["pose"] is loaded_pose
    assert captured["binder_chains"] == ["A"]
    assert captured["target_chains"] == ["B", "C", "D"]


class _FakePdbInfo:
    def __init__(self, chains):
        self._chains = chains

    def chain(self, idx):
        return self._chains[idx - 1]


class _FakePose:
    def __init__(self, chains):
        self._chains = chains

    def pdb_info(self):
        return _FakePdbInfo(self._chains)

    def total_residue(self):
        return len(self._chains)


def test_chain_relative_residue_indices_map_to_pose_indices() -> None:
    pose = _FakePose(["A", "A", "B", "B", "B", "B"])

    assert _chain_relative_residue_indices(pose, "B", "1-2,4") == [3, 4, 6]


def test_ligand_buried_unsat_hbonds_keeps_filter_failures_visible(
    monkeypatch,
) -> None:
    global_config.reset()
    global_config.init_const(
        "protomers",
        {
            "A": SimpleNamespace(type=ProtomerTypes.PROTEIN),
            "B": SimpleNamespace(type=ProtomerTypes.SMILES),
        },
    )

    class _FailingFilter:
        def report_sm(self, _pose):
            raise RuntimeError("dalphaball missing")

    monkeypatch.setattr(
        "astrolabe.losses.interaction._require_pyrosetta",
        lambda: None,
    )
    monkeypatch.setattr(
        "astrolabe.losses.interaction._build_ligand_environment_pose",
        lambda *_args, **_kwargs: SimpleNamespace(total_residue=lambda: 2),
    )
    monkeypatch.setattr(
        "astrolabe.losses.interaction.pyrosetta",
        SimpleNamespace(
            rosetta=SimpleNamespace(
                protocols=SimpleNamespace(
                    rosetta_scripts=SimpleNamespace(
                        XmlObjects=SimpleNamespace(
                            static_get_filter=lambda _xml: _FailingFilter()
                        )
                    )
                )
            )
        ),
    )

    try:
        with pytest.raises(RuntimeError, match="dalphaball missing"):
            ligand_buried_unsat_hbonds(
                SimpleNamespace(complex_poses=SimpleNamespace(poses={"AB": object()}))
            )
    finally:
        global_config.reset()
