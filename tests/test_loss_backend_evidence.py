from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.decision_maker import _snapshot_result
from astrolabe.design.native_scores import NATIVE_SCORES_ENVELOPE_CODEC, restore_scores, snapshot_scores
from astrolabe.design.store import DataStore
from astrolabe.evaluators.evidence import merge_evidence_metadata
from astrolabe.evaluators.utils import StandardScores
from astrolabe.execution_audit import build_execution_audit
from astrolabe.losses.base import LossEngine, loss_registry


def _scores(tmp_path, *names):
    paths = {}
    for name in names:
        path = tmp_path / f"{name}.pdb"
        path.write_text("END\n")
        paths[name] = path
    return StandardScores(list(names), {name: [0.8] for name in names}, paths)


def _record(method="geometric"):
    return {"metric": "hbond", "backend": "independent", "method": method,
            "selection": {"chains_a": ["A"], "chains_b": ["B"]}}


def test_loss_definition_is_metadata_and_survives_filter_merge_and_storage(tmp_path):
    original = _scores(tmp_path, "AB", "AC")
    original.loss_backend_records = {"AB": [_record()], "AC": [_record("rosetta")]}
    assert "loss_backend_records" not in StandardScores.score_field_names()
    assert original.clone().loss_backend_records == original.loss_backend_records
    filtered = original.filter_by_oligomer(["AB"])
    assert filtered.loss_backend_records == {"AB": [_record()]}
    filtered.loss_backend_records["AB"][0]["selection"]["chains_a"].append("C")
    assert original.loss_backend_records["AB"][0]["selection"]["chains_a"] == ["A"]
    merged = merge_evidence_metadata(_scores(tmp_path, "AB", "AC"),
                                     [original.filter_by_oligomer(["AB"]),
                                      original.filter_by_oligomer(["AC"])])
    assert merged.loss_backend_records == original.loss_backend_records
    data = DataStore(tmp_path / "store")
    blob = snapshot_scores(merged, data)
    data.persist((blob,))
    restored = restore_scores(blob, DataStore(tmp_path / "store"), tmp_path / "restored")
    assert restored.loss_backend_records == original.loss_backend_records


def test_assessment_keeps_the_scoring_definition_after_live_result_changes(tmp_path):
    scores = _scores(tmp_path, "AB")
    scores.loss_backend_records = {"AB": [_record()]}
    record = SimpleNamespace(assessment_id=1, attempt_id=1, accepted_revision=0,
        kind="mutation", step=1, accepted=False, stage="default", weighted_score=0.5,
        score_details={"hb": 3.0}, sequences={}, evaluation_result=scores,
        evidence_snapshot=_snapshot_result(scores))
    scores.loss_backend_records["AB"][0]["method"] = "rosetta"
    audit = build_execution_audit(run_id="test", status="completed",
                                  decisions=SimpleNamespace(assessments=[record]))
    assert audit["assessments"][0]["loss_backend_records"]["AB"][0]["method"] == "geometric"


def test_earlier_v2_snapshot_does_not_invent_missing_backend_provenance(tmp_path):
    scores = _scores(tmp_path, "AB")
    data = DataStore()
    blob = snapshot_scores(scores, data)
    envelope = data.get(blob, NATIVE_SCORES_ENVELOPE_CODEC)
    previous = replace(envelope, fields=tuple(
        field for field in envelope.fields if field.name != "loss_backend_records"
    ))
    old_blob = data.put(previous, NATIVE_SCORES_ENVELOPE_CODEC)
    restored = restore_scores(old_blob, data, tmp_path / "restored")
    assert restored.loss_backend_records is None
    assert restored.plddt == scores.plddt


@pytest.mark.parametrize("oligomer_specific", [None, ["AB"]])
def test_loss_aliases_keep_their_own_evidence_without_recomputing_on_cache_hit(
    tmp_path, oligomer_specific,
):
    global_config.reset()
    calls = []

    @loss_registry.register(name="__backend_evidence__", optional_params=["method"],
                            supports_oligomer_specific=True)
    def metric(result, method="geometric"):
        calls.append(method)
        result.loss_backend_records = {name: [_record(method)] for name in result.names}
        return {name: 3.0 for name in result.names}

    terms = {alias: {"_type": "__backend_evidence__", "method": method}
             for alias, method in (("fast_hb", "geometric"), ("native_hb", "rosetta"))}
    if oligomer_specific is not None:
        for term in terms.values():
            term["_for"] = oligomer_specific
    engine = LossEngine({"loss_terms": terms})
    engine.setup("fast_hb", None, None)
    result = _scores(tmp_path, "AB", "AC")
    first = engine._extract_scores_with_registry(result)
    records = result.loss_backend_records
    assert set(records) == ({"AB"} if oligomer_specific else {"AB", "AC"})
    assert {row["loss_alias"]: row["method"] for row in records["AB"]} == {
        "fast_hb": "geometric", "native_hb": "rosetta"}
    result.loss_backend_records = None
    assert engine._extract_scores_with_registry(result) == first
    assert len(calls) == 2
    assert result.loss_backend_records == records
    result.loss_backend_records["AB"][0]["method"] = "caller changed"
    engine._extract_scores_with_registry(result)
    assert result.loss_backend_records == records
    assert len(calls) == 2


def test_failed_loss_does_not_publish_partial_backend_evidence(tmp_path):
    global_config.reset()

    @loss_registry.register(name="__partial_backend_evidence__", optional_params=[])
    def metric(result):
        result.loss_backend_records = {"AB": [_record()]}
        raise ValueError("invalid second oligomer")

    engine = LossEngine({"loss_terms": {"hb": {"_type": "__partial_backend_evidence__"}}})
    engine.setup("hb", None, None)
    result = _scores(tmp_path, "AB", "AC")
    with pytest.raises(ValueError, match="invalid second oligomer"):
        engine._extract_scores_with_registry(result)
    assert result.loss_backend_records is None


def test_reselecting_an_alias_removes_its_previous_oligomer_evidence(tmp_path):
    global_config.reset()

    @loss_registry.register(name="__reselect_backend_evidence__", optional_params=[],
                            supports_oligomer_specific=True)
    def metric(result):
        result.loss_backend_records = {name: [_record()] for name in result.names}
        return {name: 3.0 for name in result.names}

    engine = LossEngine({"loss_terms": {"hb": {
        "_type": "__reselect_backend_evidence__", "_for": ["AB"],
    }}})
    result = _scores(tmp_path, "AB", "AC")
    engine.setup("hb", None, None)
    engine.score_context(result)
    assert set(result.loss_backend_records) == {"AB"}
    spec, function, params = engine.loss_specs["hb"]
    engine.loss_specs["hb"] = (spec, function, {**params, "oligomer_specific": ["AC"]})
    assert engine.score_context(result)["AC"]["hb"] == 3.0
    assert set(result.loss_backend_records) == {"AC"}
    engine.loss_specs["hb"] = (spec, function, {**params, "oligomer_specific": ["NO_MATCH"]})
    assert engine.score_context(result) == {"hb": 0.0}
    assert result.loss_backend_records is None


def test_cache_hit_preserves_later_gate_expression_evidence(tmp_path):
    global_config.reset()
    calls = []

    @loss_registry.register(name="__cached_backend_evidence__", optional_params=[])
    def cached_metric(result):
        calls.append("score")
        result.loss_backend_records = {name: [_record("score")] for name in result.names}
        return {name: 3.0 for name in result.names}

    @loss_registry.register(name="__gate_backend_evidence__", optional_params=[])
    def gate_metric(result):
        calls.append("gate")
        result.loss_backend_records = {name: [_record("gate")] for name in result.names}
        return {name: 2.0 for name in result.names}

    engine = LossEngine({"loss_terms": {"hb": {"_type": "__cached_backend_evidence__"}}})
    engine.setup("hb", None, None)
    result = _scores(tmp_path, "AB")
    engine.score_context(result)
    engine.score_context_for_expressions(result, ["__gate_backend_evidence__"])
    expected = result.loss_backend_records
    assert len(expected["AB"]) == 2
    engine.score_context(result)
    assert calls == ["score", "gate"]
    assert sorted(result.loss_backend_records["AB"], key=lambda row: row["loss_alias"]) == sorted(
        expected["AB"], key=lambda row: row["loss_alias"],
    )


def test_real_file_loss_engine_scores_and_caches_without_pyrosetta(tmp_path, monkeypatch):
    struc = pytest.importorskip("biotite.structure")
    pdbx = pytest.importorskip("biotite.structure.io.pdbx")
    for package in ("shape_complementarity", "py_contact_ms", "hydride"):
        pytest.importorskip(package)
    from astrolabe.losses import independent, interaction
    import run

    source = Path(__file__).resolve().parents[1] / "examples/input/9TP_cid.cif"
    atoms = independent.load_structure(source)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    file = pdbx.CIFFile()
    pdbx.set_structure(file, atoms)
    path = tmp_path / "protein.cif"
    file.write(path)
    monkeypatch.setattr(interaction, "_require_pyrosetta",
                        lambda: pytest.fail("independent loss acquired a PyRosetta runtime"))
    global_config.reset()
    try:
        terms = {alias: {"_type": name, "backend": "independent"} for alias, name in (
            ("sc", "protomer_shape_complementarity"),
            ("cms", "contact_molecular_surface"),
            ("hb", "interface_hbond_count"),
            ("buns", "interface_buried_unsatisfied_hbonds"),
        )}
        engine = LossEngine({"loss_terms": terms})
        engine.setup("sc", None, None)
        assert not run._loss_computer_needs_complex_poses(engine)
        result = StandardScores(["AB"], {"AB": [0.8]}, {"AB": path})
        with pytest.warns(UserWarning, match="not numerically equivalent"):
            scores = engine.score_context(result)
        assert -1 <= scores["sc"] <= 1
        assert all(scores[name] >= 0 for name in ("cms", "hb", "buns"))
        assert result.complex_poses is None
        assert {row["loss_alias"] for row in result.loss_backend_records["AB"]} == set(terms)
        assert all(row["backend"] == "independent" for row in result.loss_backend_records["AB"])
        assert engine.score_context(result) == scores
    finally:
        global_config.reset()
