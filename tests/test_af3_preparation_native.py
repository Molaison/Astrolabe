"""Optional native boundary checks; no inference or model weights required."""
from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("alphafold3.model.features")
from alphafold3 import structure
from alphafold3.common import folding_input
from alphafold3.constants import mmcif_names, residue_names
from alphafold3.model import features
from alphafold3.model.atom_layout import atom_layout

_ROOT = Path(__file__).resolve().parents[1]
if "astrolabe.af3_preparation" in sys.modules:
    _PREPARATION = sys.modules["astrolabe.af3_preparation"]
else:
    _SPEC = importlib.util.spec_from_file_location(
        "astrolabe.af3_preparation", _ROOT / "astrolabe/af3_preparation.py")
    _PREPARATION = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _PREPARATION
    _SPEC.loader.exec_module(_PREPARATION)



def test_real_template_sequence_entity_mapping_and_coordinate_changes():
    """Exercise the actual native mmCIF parser and Templates implementation."""
    fixture = _ROOT / "examples/paper/inputs/assemblies/01_C3_seed2503_step024_reference.cif"
    target = structure.from_mmcif(fixture.read_text(), include_bonds=False)
    chain_id = next(iter(target.chain_single_letter_sequence()))
    target = target.filter(chain_id=chain_id)
    if target.release_date is None:
        target = target.copy_and_update_globals(release_date=datetime.date(2000, 1, 1))
    text = target.to_mmcif()
    shifted_text = target.copy_and_update_coords(target.coords + 1.0).to_mmcif()

    def arguments(seq_a, *, mmcif=text, offset=0):
        sequences = (seq_a, "AACAVA")
        templates = [
            folding_input.Template(mmcif=mmcif,
                                   query_to_template_map={i: i + offset for i in range(6)}),
            folding_input.Template(mmcif=text, query_to_template_map={i: i + 1 for i in range(6)})]
        chains = [folding_input.ProteinChain(id=cid, sequence=seq, ptms=[], templates=[template],
                                             unpaired_msa="", paired_msa="")
                  for cid, seq, template in zip(("A", "B"), sequences, templates, strict=True)]
        fold = folding_input.Input(name="native_template_regression", chains=chains, rng_seeds=[17])
        return {
            "all_tokens": atom_layout.AtomLayout(
                atom_name=np.array(["CA"] * 12, dtype=object),
                atom_element=np.array(["C"] * 12, dtype=object),
                res_id=np.tile(np.arange(1, 7), 2),
                chain_id=np.array(["A"] * 6 + ["B"] * 6, dtype=object),
                chain_type=np.array([mmcif_names.PROTEIN_CHAIN] * 12, dtype=object),
                res_name=np.array([residue_names.PROTEIN_COMMON_ONE_TO_THREE[aa]
                                   for aa in "".join(sequences)], dtype=object)),
            "standard_token_idxs": np.arange(12),
            "padding_shapes": features.PaddingShapes(
                num_tokens=12, msa_size=32, num_chains=2, num_templates=4, num_atoms=128),
            "fold_input": fold, "max_templates": 4, "logging_name": "native_template_regression"}

    native = features.Templates.compute_features
    cache = _PREPARATION.AF3PreparationCache()
    cases = [
        arguments("GACWVA"), arguments("AACWVA"),  # Gly -> Ala
        arguments("AACAVA"),                       # Trp -> Ala and entity merge
        arguments("AACWVA"),                       # entity split
        arguments("AACWVA", offset=1),             # mapping invalidation
        arguments("AACWVA", mmcif=shifted_text),    # actual coordinate invalidation
    ]
    for case in cases:
        expected = native(**case)
        actual = cache.template_features(native, case, protein_type=folding_input.ProteinChain)
        for field in ("aatype", "atom_positions", "atom_mask"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
        actual.atom_positions[:] = -999.0
    assert cache.statistics["templates_misses"] == 4
    assert cache.statistics["templates_hits"] == 2


def test_scope_is_local_and_environment_opt_out(monkeypatch):
    from alphafold3.data import msa_features
    cache = _PREPARATION.AF3PreparationCache()
    rows = ["ACDEFG", "ACDkEFG"]
    native = msa_features.extract_msa_features
    expected = native(rows, mmcif_names.PROTEIN_CHAIN)
    with _PREPARATION.preparation_cache_scope(cache):
        actual = msa_features.extract_msa_features(rows, mmcif_names.PROTEIN_CHAIN)
    after_scope = cache.statistics.copy()
    msa_features.extract_msa_features(rows, mmcif_names.PROTEIN_CHAIN)
    assert cache.statistics == after_scope
    monkeypatch.setenv("ALPHAFOLD3_MSA_CACHE", "0")
    with _PREPARATION.preparation_cache_scope(cache):
        msa_features.extract_msa_features(rows, mmcif_names.PROTEIN_CHAIN)
    assert cache.statistics == after_scope
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
