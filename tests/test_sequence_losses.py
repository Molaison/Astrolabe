from __future__ import annotations

from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes


def _monkeypatch_global_const(monkeypatch, name: str, value) -> None:
    monkeypatch.setattr(global_config, f"_{name}", value, raising=True)
    monkeypatch.setitem(global_config._initialized_constants, name, True)


def _protein_protomer(*, sequence: str, mutated_sequence) -> SimpleNamespace:
    return SimpleNamespace(
        type=ProtomerTypes.PROTEIN,
        sequence=sequence,
        mutated_sequence=mutated_sequence,
    )


class _HashableEvalResult:
    names = ("ABC",)


def test_loss_engine_recomputes_when_working_sequence_changes(monkeypatch):
    from astrolabe.losses.base import LossEngine

    protomer = _protein_protomer(sequence="AAAA", mutated_sequence="AAAA")
    _monkeypatch_global_const(monkeypatch, "protomers", {"A": protomer})
    engine = LossEngine({"loss_terms": {"fraction": {
        "_type": "sequence_residue_fraction",
        "included_protomers": "A", "residues": "X",
    }}}, validate_config=False)
    engine.setup("fraction", None, None, validate_config=False)
    scores = _HashableEvalResult()
    _, initial = engine._extract_scores_with_registry(scores)
    protomer.mutated_sequence = "AXXX"
    _, updated = engine._extract_scores_with_registry(scores)
    assert initial["fraction"] == 0.0
    assert updated["fraction"] == 0.75


def test_sequence_residue_fraction_counts_x_from_current_protomer_sequence(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAAA", mutated_sequence="AXXXX"),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    score = sequence_residue_fraction(
        eval_result={},
        included_protomers="A",
        residues="X",
    )

    assert score == 0.8


def test_sequence_residue_fraction_supports_multiple_residues(monkeypatch) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="WAAA"),
            "B": _protein_protomer(sequence="BBBB", mutated_sequence="XXAA"),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    score = sequence_residue_fraction(
        eval_result={},
        included_protomers="A, B",
        residues="W,X",
    )

    assert score == 0.375


def test_sequence_residue_fraction_rejects_non_protein_protomer(monkeypatch) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": SimpleNamespace(
                type=ProtomerTypes.SMILES,
                sequence="CCO",
                mutated_sequence="CCO",
            ),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    with pytest.raises(ValueError, match="protein protomer"):
        sequence_residue_fraction(
            eval_result={},
            included_protomers="A",
            residues="X",
        )


@pytest.mark.parametrize("mutated_sequence", [None, 123, "", "   "])
def test_sequence_residue_fraction_rejects_invalid_mutated_sequence(
    monkeypatch, mutated_sequence
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence=mutated_sequence),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    with pytest.raises(ValueError, match="mutated_sequence"):
        sequence_residue_fraction(
            eval_result={},
            included_protomers="A",
            residues="X",
        )


def test_sequence_residue_fraction_rejects_missing_mutated_sequence_attribute(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": SimpleNamespace(
                type=ProtomerTypes.PROTEIN,
                sequence="AAAA",
            ),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    with pytest.raises(ValueError, match="mutated_sequence"):
        sequence_residue_fraction(
            eval_result={},
            included_protomers="A",
            residues="X",
        )


@pytest.mark.parametrize("residues", ["ALA", "X,ALA", ""])
def test_sequence_residue_fraction_rejects_malformed_residue_tokens(
    monkeypatch, residues: str
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AXXX"),
        },
    )

    from astrolabe.losses.sequence import sequence_residue_fraction

    with pytest.raises(ValueError, match="residue"):
        sequence_residue_fraction(
            eval_result={},
            included_protomers="A",
            residues=residues,
        )


def test_sequence_region_identity_returns_one_for_identical_region(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="ABCD"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="ABCD"),
            "C": _protein_protomer(sequence="AAAA", mutated_sequence="ABCD"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    score = sequence_region_identity(
        eval_result={},
        included_protomers="A,B,C",
        positions="1-4",
    )

    assert score == 1.0


def test_sequence_region_identity_uses_mean_pairwise_identity(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="AACC"),
            "C": _protein_protomer(sequence="AAAA", mutated_sequence="BBAA"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    score = sequence_region_identity(
        eval_result={},
        included_protomers="A,B,C",
        positions="1-4",
    )

    assert score == pytest.approx(1.0 / 3.0)


def test_sequence_region_identity_parses_ranges_and_singletons(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="ABCD"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="AXCY"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    score = sequence_region_identity(
        eval_result={},
        included_protomers="A,B",
        positions="2-3,4",
    )

    assert score == pytest.approx(1.0 / 3.0)


def test_sequence_edge_region_identity_compares_directed_edge_barcodes(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="ABCD"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="WXYZ"),
            "C": _protein_protomer(sequence="AAAA", mutated_sequence="ABYZ"),
        },
    )

    from astrolabe.losses.sequence import sequence_edge_region_identity

    score = sequence_edge_region_identity(
        eval_result={},
        edges="A>B,B>C,C>A",
        m_positions="1-2",
        n_positions="3-4",
    )

    assert score == pytest.approx(1.0 / 3.0)


def test_sequence_edge_region_identity_rejects_duplicate_edges(monkeypatch) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="BBBB"),
        },
    )

    from astrolabe.losses.sequence import sequence_edge_region_identity

    with pytest.raises(ValueError, match="duplicates"):
        sequence_edge_region_identity(
            eval_result={},
            edges="A>B,A>B",
            m_positions="1",
            n_positions="2",
        )


@pytest.mark.parametrize("positions", ["", "0", "2-1", "A", "1-"])
def test_sequence_region_identity_rejects_invalid_positions(
    monkeypatch, positions: str
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    with pytest.raises(ValueError, match="position"):
        sequence_region_identity(
            eval_result={},
            included_protomers="A,B",
            positions=positions,
        )


def test_sequence_region_identity_rejects_out_of_range_positions(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    with pytest.raises(ValueError, match="outside protomer length"):
        sequence_region_identity(
            eval_result={},
            included_protomers="A,B",
            positions="1,5",
        )


def test_sequence_region_identity_rejects_unequal_protomer_lengths(
    monkeypatch,
) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAA", mutated_sequence="AAA"),
        },
    )

    from astrolabe.losses.sequence import sequence_region_identity

    with pytest.raises(ValueError, match="equal-length"):
        sequence_region_identity(
            eval_result={},
            included_protomers="A,B",
            positions="1-3",
        )


def test_sequence_region_identity_works_through_loss_engine(monkeypatch) -> None:
    _monkeypatch_global_const(
        monkeypatch,
        "protomers",
        {
            "A": _protein_protomer(sequence="AAAA", mutated_sequence="AAAA"),
            "B": _protein_protomer(sequence="AAAA", mutated_sequence="AACC"),
            "C": _protein_protomer(sequence="AAAA", mutated_sequence="BBAA"),
        },
    )

    from astrolabe.losses.base import LossEngine

    loss_config = {
        "loss_terms": {
            "ABC_patch_barcode_identity": {
                "_type": "sequence_region_identity",
                "included_protomers": "A,B,C",
                "positions": "1-4",
            },
        },
        "expression_terms": {
            "ABC_patch_barcode_score": (
                "max(0.0, 1.0 - abs(ABC_patch_barcode_identity - 0.35) / 0.35)"
            ),
        },
    }
    engine = LossEngine(loss_config)
    engine.setup(
        weighted_score_expression="ABC_patch_barcode_score",
        early_stopping=None,
        acceptance_threshold=None,
    )

    score, env = engine.compute_loss(
        eval_result=_HashableEvalResult(),
        runtime_config={},
    )

    assert env["ABC_patch_barcode_identity"] == pytest.approx(1.0 / 3.0)
    assert score == pytest.approx(1.0 - abs((1.0 / 3.0) - 0.35) / 0.35)


def test_rna_defect_preserves_occurrences_weights_and_current_sequences(monkeypatch):
    import sys

    from astrolabe.losses.rna_secondary import nupack_ensemble_defect, rna_ensemble_defect

    protomer = SimpleNamespace(type=ProtomerTypes.RNA, sequence="AAAA", mutated_sequence="GCGC")
    _monkeypatch_global_const(monkeypatch, "protomers", {
        "A": protomer,
        "B": SimpleNamespace(type=ProtomerTypes.RNA, mutated_sequence="AUAU"),
    })
    requests = []

    def defect(*, strands, structure, model):
        requests.append((strands, structure, model))
        return 0.2 if len(strands) == 2 else 0.6

    monkeypatch.setitem(sys.modules, "nupack", SimpleNamespace(Model=lambda **kw: kw, defect=defect))
    targets = {
        "dimer": {"strands": ["A", "A"], "structure": "((((+))))", "weight": 1},
        "trimer": {"strands": ["B", "A", "B"], "structure": "(((.+).((+))))", "weight": 3},
    }
    settings = dict(material="rna", celsius=25, sodium=1, magnesium=0, ensemble="nostacking")
    assert rna_ensemble_defect(None, targets, **settings) == pytest.approx(0.5)
    assert [request[0] for request in requests] == [["GCGC", "GCGC"], ["AUAU", "GCGC", "AUAU"]]
    assert [request[1] for request in requests] == [spec["structure"] for spec in targets.values()]
    assert requests[0][2] == settings
    protomer.mutated_sequence = "GGGG"
    assert nupack_ensemble_defect({}, {"dimer": targets["dimer"]}) == 0.2
    assert requests[-1][0] == ["GGGG", "GGGG"]

    from astrolabe.losses.base import LossEngine

    engine = LossEngine({"loss_terms": {"defect": {
        "_type": "rna_ensemble_defect", "targets": targets, **settings,
    }}})
    engine.setup("-defect", None, None)
    score, terms = engine.compute_loss(_HashableEvalResult(), runtime_config={})
    assert terms["defect"] == pytest.approx(0.5)
    assert score == pytest.approx(-0.5)


def test_rna_probability_and_ensemble_energy_use_public_backend_quantities(monkeypatch):
    import sys

    from astrolabe.losses.rna_secondary import (
        calculate_rna_ensemble_free_energy,
        calculate_rna_ensemble_defect,
        calculate_rna_structure_probability,
        rna_ensemble_free_energy,
        rna_structure_probability,
    )

    monkeypatch.setitem(sys.modules, "nupack", SimpleNamespace(
        Model=lambda **kw: kw,
        structure_probability=lambda **kw: 0.8,
        pfunc=lambda **kw: (12345, -4.5),
    ))
    _monkeypatch_global_const(monkeypatch, "protomers", {
        "A": SimpleNamespace(type=ProtomerTypes.RNA, mutated_sequence="GCGC"),
    })
    targets = {"monomer": {"strands": ["A"], "structure": "...."}}
    assert calculate_rna_structure_probability(["GCGC"], "....") == 0.8
    assert calculate_rna_ensemble_free_energy(["GCGC"]) == -4.5
    assert rna_structure_probability(None, targets) == 0.8
    assert rna_ensemble_free_energy(None, targets) == -4.5

    model = SimpleNamespace()

    def fold_compound(sequence, details):
        assert sequence == "GCGC"
        assert details.temperature == 25
        assert details.compute_bpp == 1
        return SimpleNamespace(
            mfe=lambda: ("....", -3.0), exp_params_rescale=lambda energy: None,
            pf=lambda: ("....", -4.5), ensemble_defect=lambda structure: 0.2,
            pr_structure=lambda structure: 0.8,
        )

    monkeypatch.setitem(sys.modules, "RNA", SimpleNamespace(md=lambda: model, fold_compound=fold_compound))
    settings = dict(backend="viennarna", celsius=25, ensemble="vienna")
    assert calculate_rna_ensemble_defect(["GCGC"], "....", **settings) == 0.2
    assert calculate_rna_structure_probability(["GCGC"], "....", **settings) == 0.8
    assert calculate_rna_ensemble_free_energy(["GCGC"], **settings) == -4.5


@pytest.mark.parametrize("metric", ["ensemble_defect", "structure_probability"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_rna_secondary_rejects_invalid_backend_probabilities(monkeypatch, metric, value):
    import sys

    from astrolabe.losses.rna_secondary import calculate_rna_secondary

    monkeypatch.setitem(sys.modules, "nupack", SimpleNamespace(
        Model=lambda **kw: kw, defect=lambda **kw: value,
        structure_probability=lambda **kw: value,
    ))
    with pytest.raises(ValueError):
        calculate_rna_secondary(["ACGU"], "....", metric=metric)


@pytest.mark.parametrize("sequence,structure,occurrences,weight,kind", [
    ("ACTU", "....", ["A"], 1, ProtomerTypes.RNA),
    ("ACGN", "....", ["A"], 1, ProtomerTypes.RNA),
    (None, "....", ["A"], 1, ProtomerTypes.RNA),
    ("ACGU", "...", ["A"], 1, ProtomerTypes.RNA),
    ("ACGU", "..+..", ["A"], 1, ProtomerTypes.RNA),
    ("ACGU", "([)]", ["A"], 1, ProtomerTypes.RNA),
    ("ACGU", ")..(", ["A"], 1, ProtomerTypes.RNA),
    ("ACGU", "(...", ["A"], 1, ProtomerTypes.RNA),
    ("ACGU", "....", "A", 1, ProtomerTypes.RNA),
    ("ACGU", "....", ["missing"], 1, ProtomerTypes.RNA),
    ("ACGU", "....", ["A"], -1, ProtomerTypes.RNA),
    ("ACGU", "....", ["A"], 0, ProtomerTypes.RNA),
    ("ACGU", "....", ["A"], 1, ProtomerTypes.DNA),
])
def test_rna_secondary_validates_targets_before_import(
    monkeypatch, sequence, structure, occurrences, weight, kind,
):
    import sys

    from astrolabe.losses.rna_secondary import rna_ensemble_defect

    monkeypatch.setitem(sys.modules, "nupack", None)
    _monkeypatch_global_const(monkeypatch, "protomers", {
        "A": SimpleNamespace(type=kind, mutated_sequence=sequence),
    })
    with pytest.raises(ValueError):
        rna_ensemble_defect(None, {
            "target": {"strands": occurrences, "structure": structure, "weight": weight},
        })


@pytest.mark.parametrize("backend,settings", [
    ("nupack", {"sodium": 0.15}),
    ("nupack", {"magnesium": 0.01}),
    ("viennarna", {"sodium": 1.0}),
    ("viennarna", {"ensemble": "stacking"}),
    ("unknown", {}),
])
def test_rna_secondary_rejects_unsupported_settings(monkeypatch, backend, settings):
    import sys

    from astrolabe.losses.rna_secondary import calculate_rna_ensemble_defect

    monkeypatch.setitem(sys.modules, "nupack", None)
    monkeypatch.setitem(sys.modules, "RNA", None)
    with pytest.raises(ValueError):
        calculate_rna_ensemble_defect(["ACGU"], "....", backend=backend, **settings)


def test_rna_secondary_missing_backend_is_honest_and_vienna_rejects_multistrand(monkeypatch):
    import sys

    from astrolabe.losses.rna_secondary import calculate_rna_ensemble_defect

    for backend, module in [("nupack", "nupack"), ("viennarna", "RNA")]:
        monkeypatch.setitem(sys.modules, module, None)
        with pytest.raises(ImportError, match="optional"):
            calculate_rna_ensemble_defect(["ACGU"], "....", backend=backend)
    with pytest.raises(ValueError, match="single strand"):
        calculate_rna_ensemble_defect(["ACGU", "ACGU"], "((((+))))", backend="viennarna")
