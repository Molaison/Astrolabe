"""Regression tests for the vendored Proton-PottsMPNN design-mask preservation.

These tests require the isolated Python 3.12 runtime with atomworks/biotite
(the default Astrolabe image does not provide them), and must import ``mpnn``
from this worktree's ``vendor/proton_pottsmpnn`` tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_MPNN_SRC = (
    REPO_ROOT
    / "vendor"
    / "proton_pottsmpnn"
    / "foundry"
    / "models"
    / "mpnn"
    / "src"
)

try:
    import numpy as np
    from atomworks.ml.utils.token import get_token_starts
    from biotite.structure import AtomArray
    import mpnn.utils.inference as inference
except ImportError as exc:  # pragma: no cover - default image lacks atomworks/mpnn
    pytest.skip(
        f"requires the isolated proton_pottsmpnn Python 3.12 runtime: {exc}",
        allow_module_level=True,
    )

if not Path(inference.__file__).resolve().is_relative_to(VENDOR_MPNN_SRC.resolve()):
    raise AssertionError(
        "tests must import mpnn from the worktree vendor snapshot, got "
        f"{inference.__file__}"
    )


def _atom_array(residues: list[tuple[str, int, str, str]]) -> AtomArray:
    array = AtomArray(len(residues))
    array.chain_id = [chain for chain, _, _, _ in residues]
    array.res_id = [res_id for _, res_id, _, _ in residues]
    array.ins_code = [ins_code for _, _, ins_code, _ in residues]
    array.res_name = [res_name for _, _, _, res_name in residues]
    array.atom_name = ["CA"] * len(residues)
    array.element = ["C"] * len(residues)
    array.coord = np.zeros((len(residues), 3), dtype=np.float32)
    return array


def _residue_mask(atom_array: AtomArray) -> dict[tuple[str, int, str], bool]:
    tokens = atom_array[get_token_starts(atom_array)]
    return {
        (str(chain), int(res_id), str(ins_code)): bool(value)
        for chain, res_id, ins_code, value in zip(
            tokens.chain_id,
            tokens.res_id,
            tokens.ins_code,
            tokens.mpnn_designed_residue_mask,
        )
    }


def _parse_stub(extra_residues: tuple[tuple[str, int, str, str], ...] = ()):
    """Simulate atomworks ``parse_atom_array``: rebuild atoms, drop extra annotations.

    The real parser rebuilds residues from CCD templates and adds missing atoms,
    which drops ``mpnn_designed_residue_mask`` and changes atom order/count.
    """

    def _parse(atom_array, *args, **kwargs):  # noqa: ARG001
        tokens = atom_array[get_token_starts(atom_array)]
        records = [
            (str(chain), int(res_id), str(ins_code), str(res_name))
            for chain, res_id, ins_code, res_name in zip(
                tokens.chain_id,
                tokens.res_id,
                tokens.ins_code,
                tokens.res_name,
            )
        ]
        records = list(reversed(records)) + list(extra_residues)
        n_atoms = 2 * len(records)
        parsed = AtomArray(n_atoms)
        parsed.chain_id = [chain for chain, _, _, _ in records for _ in range(2)]
        parsed.res_id = [res_id for _, res_id, _, _ in records for _ in range(2)]
        parsed.ins_code = [ins_code for _, _, ins_code, _ in records for _ in range(2)]
        parsed.res_name = [res_name for _, _, _, res_name in records for _ in range(2)]
        parsed.atom_name = ["CA", "O"] * len(records)
        parsed.element = ["C", "O"] * len(records)
        parsed.coord = np.zeros((n_atoms, 3), dtype=np.float32)
        return {"assemblies": {"1": [parsed]}}

    return _parse


def test_caller_design_mask_survives_lossy_parse(monkeypatch) -> None:
    atom_array = _atom_array(
        [
            ("A", 1, "", "ALA"),
            ("A", 2, "", "GLY"),
            ("A", 2, "A", "SER"),
            ("B", 1, "", "LEU"),
        ]
    )
    atom_array.set_annotation(
        "mpnn_designed_residue_mask", np.array([True, False, True, False])
    )
    monkeypatch.setattr(inference, "parse_atom_array", _parse_stub())

    result = inference.MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atom_array, input_dict={"designed_chains": ["A"]}
    )

    assert _residue_mask(result.atom_array) == {
        ("A", 1, ""): True,
        ("A", 2, ""): False,
        ("A", 2, "A"): True,
        ("B", 1, ""): False,
    }


def test_without_caller_mask_input_dict_scope_is_unchanged(monkeypatch) -> None:
    atom_array = _atom_array(
        [
            ("A", 1, "", "ALA"),
            ("A", 2, "", "GLY"),
            ("B", 1, "", "LEU"),
        ]
    )
    monkeypatch.setattr(inference, "parse_atom_array", _parse_stub())

    scoped = inference.MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atom_array, input_dict={"designed_chains": ["A"]}
    )
    assert _residue_mask(scoped.atom_array) == {
        ("A", 1, ""): True,
        ("A", 2, ""): True,
        ("B", 1, ""): False,
    }

    unscoped = inference.MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atom_array, input_dict={}
    )
    assert (
        "mpnn_designed_residue_mask"
        not in unscoped.atom_array.get_annotation_categories()
    )


def test_insertion_codes_distinguish_residues(monkeypatch) -> None:
    atom_array = _atom_array(
        [
            ("A", 10, "", "ALA"),
            ("A", 10, "A", "GLY"),
            ("A", 10, "B", "SER"),
        ]
    )
    atom_array.set_annotation(
        "mpnn_designed_residue_mask", np.array([False, True, True])
    )
    monkeypatch.setattr(inference, "parse_atom_array", _parse_stub())

    result = inference.MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atom_array, input_dict={}
    )

    assert _residue_mask(result.atom_array) == {
        ("A", 10, ""): False,
        ("A", 10, "A"): True,
        ("A", 10, "B"): True,
    }


def test_unmappable_or_ambiguous_residues_raise(monkeypatch) -> None:
    atom_array = _atom_array([("A", 1, "", "ALA"), ("B", 1, "", "LEU")])
    atom_array.set_annotation("mpnn_designed_residue_mask", np.array([True, False]))
    monkeypatch.setattr(
        inference, "parse_atom_array", _parse_stub(extra_residues=(("A", 99, "", "ALA"),))
    )

    with pytest.raises(ValueError, match="no matching input residue identity"):
        inference.MPNNInferenceInput.from_atom_array_and_dict(
            atom_array=atom_array, input_dict={}
        )

    duplicated = _atom_array(
        [("A", 1, "", "ALA"), ("A", 2, "", "GLY"), ("A", 1, "", "SER")]
    )
    duplicated.set_annotation(
        "mpnn_designed_residue_mask", np.array([True, False, True])
    )
    with pytest.raises(ValueError, match="not uniquely identified"):
        inference.MPNNInferenceInput.from_atom_array_and_dict(
            atom_array=duplicated, input_dict={}
        )
