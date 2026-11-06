from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from astrolabe.evaluators.structure_output import project_structure, structure_directory
from astrolabe.posthoc.io_utils import index_structures


def test_same_step_aliases_keep_outputs_after_root_projection_changes(tmp_path: Path):
    """CPU doubles for two writers publishing the same canonical PDB name."""
    name = "step_003_AB_model.pdb"
    af2 = structure_directory(tmp_path, "AF2/one") / name
    esm = structure_directory(tmp_path, "ESM fold") / name
    root = tmp_path / "structures" / name

    af2.write_text("AF2\n")
    assert project_structure(af2, tmp_path, "AF2/one") == af2
    esm.write_text("ESMFOLD\n")
    assert project_structure(esm, tmp_path, "ESM fold") == esm

    assert af2 == tmp_path / "structures/evaluators/alias-AF2%2Fone" / name
    assert esm == tmp_path / "structures/evaluators/alias-ESM%20fold" / name
    assert af2.read_text() == "AF2\n"
    assert esm.read_text() == "ESMFOLD\n"
    assert root.read_text() == "ESMFOLD\n"
    assert root.stat().st_ino not in {af2.stat().st_ino, esm.stat().st_ino}
    assert index_structures(tmp_path / "structures")[(3, "AB")] == root

    project_structure(af2, tmp_path, "AF2/one")
    assert root.read_text() == "AF2\n"
    assert esm.read_text() == "ESMFOLD\n"


def test_direct_evaluator_preserves_root_structure_path(tmp_path: Path):
    root = structure_directory(tmp_path)
    path = root / "step_000_AB_model.cif"
    path.write_text("data_test\n")
    assert project_structure(path, tmp_path) == path
    assert index_structures(root)[(0, "AB")] == path


def test_af2_and_esmfold_writers_keep_distinct_score_paths(tmp_path: Path, monkeypatch):
    """Exercise both evaluators with CPU doubles when their heavy imports exist."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorflow")
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "alphafold2"))
    from alphafold.common import protein, residue_constants
    from astrolabe.core import global_config
    from astrolabe.core.protomer_config_parser import ProtomerTypes
    from astrolabe.evaluators.alphafold2 import AlphaFold2Evaluator
    from astrolabe.evaluators.esmfold import ESMFoldEvaluator

    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("protomers", {
        name: SimpleNamespace(type=ProtomerTypes.PROTEIN, length=1) for name in "AB"
    })
    global_config.set_variable("step", 3)
    monkeypatch.setattr(protein, "to_pdb", lambda _: "AF2\n")

    af2 = AlphaFold2Evaluator(model_runners={})
    af2.alias = "AF2/one"
    af2._apply_plddt_to_b_factors = lambda prot, _: prot
    predicted = SimpleNamespace(
        aatype=np.zeros(2), b_factors=np.zeros((2, residue_constants.atom_type_num))
    )
    af2_scores = af2.prepare_output(
        {"AB": [{"seed": 1, "prediction": {
            "plddt": np.full(2, 80.0, dtype=np.float32), "ranking_confidence": 0.8,
        }, "protein": predicted}]},
        {"AB": SimpleNamespace(name="AB", chains=[SimpleNamespace(sequence="A") for _ in "AB"])},
        {},
    )

    class FakeESM:
        def infer(self, _sequences):
            return {
                "chain_index": torch.tensor([[0, 1]]),
                "plddt": torch.full((1, 2, 1), 80.0),
                "mean_plddt": torch.tensor([80.0]),
                "ptm": torch.tensor([0.7]),
                "predicted_aligned_error": torch.zeros((1, 2, 2)),
            }

        def output_to_pdb(self, _output):
            return ["ESMFOLD\n"]

    esm = ESMFoldEvaluator(model=FakeESM())
    esm.alias = "ESM fold"
    esm_scores = esm.run_function({"AB": "A:B"})

    af2_path = af2_scores.complex_paths["AB"]
    esm_path = esm_scores.complex_paths["AB"]
    root = tmp_path / "structures/step_003_AB_model.pdb"
    assert af2_path.read_text() == "AF2\n"
    assert esm_path.read_text() == root.read_text() == "ESMFOLD\n"
    assert af2_path.parent.name == "alias-AF2%2Fone"
    assert esm_path.parent.name == "alias-ESM%20fold"
    assert index_structures(root.parent)[(3, "AB")] == root
    global_config.reset()
