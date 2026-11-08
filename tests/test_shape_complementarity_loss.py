from __future__ import annotations

from types import SimpleNamespace

import pytest

from astrolabe.losses import interaction
from astrolabe.losses import shape_complementarity


@pytest.mark.parametrize("spatial", [False, True, "failure"])
def test_compute_shape_complementarity_between_chains_returns_filter_value(
    monkeypatch, spatial,
) -> None:
    fake_sub_pose = SimpleNamespace(
        total_residue=lambda: 10,
        num_chains=lambda: 2,
    )

    class FakeShapeComplementarityFilter:
        def jump_id(self, jump_id: int) -> None:
            assert jump_id == 1

        def use_rosetta_radii(self, enabled: bool) -> None:
            assert enabled is True

        def report_sm(self, pose) -> float:
            assert pose is fake_sub_pose
            assert spatial is not True
            return 0.42

    monkeypatch.setattr(interaction, "_require_pyrosetta", lambda: None)
    monkeypatch.setattr(
        interaction,
        "_isolate_chains",
        lambda pose, keep_chains: fake_sub_pose,
    )
    monkeypatch.setattr(
        interaction,
        "ShapeComplementarityFilter",
        FakeShapeComplementarityFilter,
    )
    monkeypatch.setattr(shape_complementarity, "supports_spatial_sc", lambda pose: bool(spatial))

    def compute_spatial(pose):
        assert pose is fake_sub_pose
        if spatial == "failure":
            raise RuntimeError("No molecular surface dots")
        return 0.42

    monkeypatch.setattr(shape_complementarity, "spatial_shape_complementarity", compute_spatial)

    out = interaction._compute_shape_complementarity_between_chains(
        object(),
        "A",
        "B",
    )

    assert out == pytest.approx(0.42)


@pytest.mark.parametrize("strict", [False, True])
def test_spatial_failure_preserves_native_failure_handling(monkeypatch, strict):
    pose = SimpleNamespace(total_residue=lambda: 10, num_chains=lambda: 2)
    monkeypatch.setattr(interaction, "_require_pyrosetta", lambda: None)
    monkeypatch.setattr(interaction, "_isolate_chains", lambda *_: pose)
    monkeypatch.setattr(interaction, "ShapeComplementarityFilter", lambda: SimpleNamespace(
        jump_id=lambda _: None, use_rosetta_radii=lambda _: None, report_sm=lambda _: -1.0))
    monkeypatch.setattr(shape_complementarity, "supports_spatial_sc", lambda _: True)

    def fail(_):
        raise RuntimeError("No molecular surface dots")

    monkeypatch.setattr(shape_complementarity, "spatial_shape_complementarity", fail)
    if strict:
        with pytest.raises(ValueError, match="Invalid shape complementarity"):
            interaction._compute_shape_complementarity_between_chains(pose, "A", "B", strict=True)
    else:
        assert interaction._compute_shape_complementarity_between_chains(pose, "A", "B") == -1.0
