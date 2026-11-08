"""F21: model-free next-operation equivalence and transactional fault injection."""
from copy import deepcopy
import hashlib
import json
import random
import sys
from types import SimpleNamespace

import pytest

from astrolabe.core import global_config
from astrolabe.model_daemon import RequestOutcomeUnknown
from astrolabe.mutators.dplm import DPLMMutator
from astrolabe.mutators.random_substitution import PoreBlockingSubstitutionMutator
from astrolabe.mutators.rfd4 import RFD4Mutator
from astrolabe.mutators.rfd4_types import AtomRef, ConditionPlan, HBondCondition
from astrolabe.mutators.state_protocol import digest
from astrolabe.structure_processors.base import StructureProcessor
from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner


BINDING = {"assets": "frozen-test-assets", "inputs": "native-input-identity"}


@pytest.fixture(autouse=True)
def runtime(tmp_path):
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("is_debug", False)
    global_config.init_const("segments", {})
    yield
    global_config.reset()


def wire(state):
    return json.loads(json.dumps(state if isinstance(state, dict) else state.to_dict()))


def changed(state, mutate):
    record = wire(state)
    mutate(record["payload"])
    record["sha256"] = digest({key: value for key, value in record.items() if key != "sha256"})
    return record


def dplm(tmp_path, monkeypatch, **overrides):
    args = dict(python_bin=sys.executable, repo_path=str(tmp_path),
                work_dir=str(tmp_path / "work"), response_cache=False,
                response_cache_dir=str(tmp_path / "cache"),
                num_samples=3, prefetch_save=False, quality_filter_enable=False)
    args.update(overrides)
    owner = DPLMMutator(**args)
    calls = []

    def infer(tasks):
        calls.append(deepcopy(tasks))
        return {task["name"]: ["ACDE", "AGDE", "ATDE"] for task in tasks}

    monkeypatch.setattr(owner.runner, "design_sequences", infer)
    return owner, calls


FEATURES = {"A": {"combined_sequence": "AADE", "masked_sequence": "A<mask>DE",
                   "mask_positions": [1], "mutation_quota": 1,
                   "mapping": [("A", i) for i in range(4)]}}


def test_f21_dplm_reject_consumes_and_restored_next_proposal_matches(tmp_path, monkeypatch):
    owner, calls = dplm(tmp_path, monkeypatch)
    assert owner.compute_mutation(FEATURES)["A"]["samples"] == ["ACDE"]
    state = owner.export_state()
    # Scientific rejection does not put the proposed sample back into the bank.
    before_rng = random.getstate()
    expected = owner.compute_mutation(FEATURES)
    restored, restored_calls = dplm(tmp_path, monkeypatch)
    restored.restore_state(wire(state))
    assert restored.compute_mutation(FEATURES) == expected
    assert restored._prefetch["A"].cursor == 2
    assert restored._prefetch["A"].bank_id == owner._prefetch["A"].bank_id
    assert len(calls) == 1 and restored_calls == []
    assert random.getstate() == before_rng


@pytest.mark.parametrize("fault", ["cursor", "mask", "quota", "samples", "bank", "mapping", "version", "identity", "identity_type", "checksum"])
def test_dplm_corrupt_restore_is_transactional(tmp_path, monkeypatch, fault):
    owner, _ = dplm(tmp_path, monkeypatch)
    owner.compute_mutation(FEATURES)
    state = owner.snapshot_state(binding=BINDING)
    edits = {"cursor": ("cursor", 99), "mask": ("mask_positions", [8]),
             "quota": ("mutation_quota", 8), "samples": ("samples", [True]),
             "bank": ("bank_id", "wrong"), "mapping": ("native_inputs", {})}
    if fault in edits:
        key, value = edits[fault]
        record = changed(state, lambda p: p["banks"]["A"].update({key: value}))
    elif fault == "identity_type":
        record = wire(state)
        record["identity"]["invalidate"] = 1  # Python considers 1 == True; JSON identity must not.
        record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    else:
        record = wire(state)
        record[{"version": "version", "identity": "identity", "checksum": "sha256"}[fault]] = "bad"
    before = owner.snapshot_state(binding=BINDING).to_dict()
    with pytest.raises(ValueError):
        owner.restore_state(record, binding=BINDING)
    assert owner.snapshot_state(binding=BINDING).to_dict() == before


def test_dplm_shared_queue_and_unknown_are_not_exportable(tmp_path, monkeypatch):
    owner, _ = dplm(tmp_path, monkeypatch, response_cache=True)
    with pytest.raises(ValueError, match="response_cache=false"):
        owner.snapshot_state(binding=BINDING)
    owner.runner.response_cache = False
    owner.runner._unknown = RequestOutcomeUnknown("lost after send")
    with pytest.raises(RequestOutcomeUnknown):
        owner.snapshot_state(binding=BINDING)


def test_f21_pore_skip_does_not_advance_and_restored_activation_matches(monkeypatch):
    import astrolabe.mutators.random_substitution as module
    seen = []

    def select(*a, **kw):
        seen.append(global_config.pore_bottleneck_activation_index)
        return {"A": [seen[-1] % 3]}

    monkeypatch.setattr(module, "prepare_exclude_sites", select)
    owner = PoreBlockingSubstitutionMutator("pore()", active_step_interval=2)
    global_config.set_variable("step", 0)
    owner.prepare_mutation_input({}, None, {})
    state = owner.export_state()
    global_config.set_variable("step", 1)
    assert not owner.prepare_mutation_input({}, None, {})["pore_blocker_active"]
    assert owner._active_invocations == 1
    global_config.set_variable("step", 2)
    expected = owner.prepare_mutation_input({}, None, {})
    restored = PoreBlockingSubstitutionMutator("pore()", active_step_interval=2)
    restored.restore_state(wire(state))
    assert restored.prepare_mutation_input({}, None, {}) == expected
    assert seen == [0, 1, 1]
    before = restored.snapshot_state(binding=BINDING)
    with pytest.raises(ValueError):
        restored.restore_state(changed(before, lambda p: p.update(active_invocations=-1)), binding=BINDING)
    assert restored.snapshot_state(binding=BINDING) == before


def rfd4():
    # Constructor-free scientific owner: neither native environment nor model.
    owner = RFD4Mutator.__new__(RFD4Mutator)
    owner.assembly, owner.binder_chains, owner.target_chains = "AB", ("A",), ("B",)
    owner.feedback = {"repair_unsatisfied_attempts": True}
    owner.position_selector, owner.mutation_weight, owner.seed = None, 1.0, 7
    owner.backend = SimpleNamespace(generation={"mode": "inpaint"})
    from collections import Counter
    owner._failed_hbond_attempts, owner._hbond_attempt_history, owner._observed_requests = Counter(), [], {}
    return owner


def test_f21_rfd4_duplicate_conflict_restored_feedback_and_references(tmp_path, monkeypatch):
    import astrolabe.mutators.rfd4_feedback as module
    atom = SimpleNamespace(atom=AtomRef("B", 1, "N"), role="DONOR", source_present=True,
                           satisfied=False, qualifying_hbond_count=0, sasa=.5,
                           nearest_other_group_heavy_atom_distance=3.)
    calls = []

    def analyze(*a, **kw):
        calls.append(a)
        return SimpleNamespace(polar_atoms=(atom,))

    monkeypatch.setattr(module, "analyze_interface", analyze)
    path = tmp_path / "generated.pdb"
    path.write_text("immutable native structure")
    generated = SimpleNamespace(request_digest="one", structure_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
                                structure_path=str(path))
    request = SimpleNamespace(plan=ConditionPlan((), (), (HBondCondition(atom.atom, atom.role),)))
    owner = rfd4()
    owner._observe_hbond_attempt(request, generated)
    state = owner.export_state()
    owner._observe_hbond_attempt(request, generated)
    assert len(calls) == 1
    generated.structure_digest = "conflicting-bytes"
    with pytest.raises(ValueError, match="Conflicting"):
        owner._observe_hbond_attempt(request, generated)
    generated.structure_digest = state["payload"]["history"][0]["generated_structure_sha256"]
    restored = rfd4()
    restored.restore_state(wire(state))
    restored._observe_hbond_attempt(request, generated)
    assert len(calls) == 1
    # The next native condition proposal depends on restored failure counts.
    from astrolabe.mutators.rfd4_feedback import (
        InterfaceFeedback, PolarAtomObservation, ResidueObservation, plan_conditions,
    )
    from astrolabe.mutators.rfd4_types import ChainSpec, ResidueRef
    chains = (ChainSpec("A", "binder", "SS"), ChainSpec("B", "target", "SS"))
    residues = tuple(ResidueObservation(ResidueRef(chain.chain, i), "SER", i, i, "",
                                       1., 0, 1, ((float(i), 0., 0.),), 2.)
                     for chain in chains for i in (1, 2))
    polar = tuple(PolarAtomObservation(AtomRef("B", i, "N"), "SER", "N", "DONOR",
                                      (float(i), 0., 0.), .5, 2., True, True, False, 0)
                  for i in (1, 2))
    feedback = InterfaceFeedback(("A",), ("B",), residues, (), polar, (), 0., ())

    def next_plan(mutator):
        return plan_conditions(chains, (ResidueRef("A", 1), ResidueRef("A", 2)), 1,
                               feedback=feedback, policy={"local_repair": True},
                               failed_hbond_attempts=dict(mutator._failed_hbond_attempts))

    assert next_plan(restored) == next_plan(owner)
    generated.request_digest = "two"
    owner._observe_hbond_attempt(request, generated)
    restored._observe_hbond_attempt(request, generated)
    assert restored.snapshot_state(binding=BINDING) == owner.snapshot_state(binding=BINDING)
    assert restored._failed_hbond_attempts["B:1:N"] == 2
    before = restored.snapshot_state(binding=BINDING)
    for edit in (lambda p: p["failures"].update({"B:1:N": 999}),
                 lambda p: p["observed"].update(one="foreign")):
        with pytest.raises(ValueError):
            restored.restore_state(changed(before, edit), binding=BINDING)
        assert restored.snapshot_state(binding=BINDING) == before
    path.write_text("changed on disk")
    with pytest.raises(ValueError):
        restored.restore_state(wire(before), binding=BINDING)
    assert restored._failed_hbond_attempts["B:1:N"] == 2


def test_rfd4_mid_observation_failure_leaves_state_unchanged_and_retry_matches_baseline(tmp_path, monkeypatch):
    import astrolabe.mutators.rfd4_feedback as module
    atoms = tuple(SimpleNamespace(atom=AtomRef("B", i, "N"), role="DONOR", source_present=True,
                                  satisfied=False, qualifying_hbond_count=0, sasa=.5,
                                  nearest_other_group_heavy_atom_distance=3.) for i in (1, 2))
    monkeypatch.setattr(module, "analyze_interface", lambda *a, **kw: SimpleNamespace(polar_atoms=atoms))
    path = tmp_path / "generated.pdb"
    path.write_text("immutable native structure")
    generated = SimpleNamespace(request_digest="prior", structure_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
                                structure_path=str(path))
    request = SimpleNamespace(plan=ConditionPlan((), (), tuple(HBondCondition(a.atom, a.role) for a in atoms)))
    owner = rfd4()
    owner._observe_hbond_attempt(request, generated)
    before = owner.export_state()
    live_fields = (owner._failed_hbond_attempts, owner._hbond_attempt_history, owner._observed_requests)

    class FaultAfterFirstResult:
        def __bool__(self):
            raise RuntimeError("injected second HBond result failure")

    # The first unsatisfied result has already updated the temporary counter
    # when interpreting the second result raises.
    atoms[1].satisfied = FaultAfterFirstResult()
    generated.request_digest = "retryable"
    with pytest.raises(RuntimeError, match="injected second HBond"):
        owner._observe_hbond_attempt(request, generated)
    assert owner.export_state() == before
    assert all(current is original for current, original in zip(
        (owner._failed_hbond_attempts, owner._hbond_attempt_history, owner._observed_requests), live_fields))
    assert "retryable" not in owner._observed_requests

    atoms[1].satisfied = False
    baseline = rfd4()
    baseline.restore_state(wire(before))
    baseline._observe_hbond_attempt(request, generated)
    owner._observe_hbond_attempt(request, generated)
    assert owner.export_state() == baseline.export_state()
    assert dict(owner._failed_hbond_attempts) == {"B:1:N": 2, "B:2:N": 2}
    assert len(owner._hbond_attempt_history) == 2


class CacheOwner(StructureProcessor):
    def _run_impl(self, input_file):
        return input_file


def test_predocker_bank_rng_consumption_and_corrupt_restore(tmp_path):
    paths = [tmp_path / f"pose-{i}.pdb" for i in range(3)]
    for i, path in enumerate(paths):
        path.write_text(f"pose {i}")
    owner = CacheOwner()
    owner._cache["native-input"] = str(paths[0])
    owner._set_cache_candidates("native-input", [str(p) for p in paths])
    owner._cache_rng.seed(17)
    owner.take_cache("native-input")
    state = owner.snapshot_bank_state(binding=BINDING)
    expected = [owner.take_cache("native-input") for _ in range(8)]
    restored = CacheOwner()
    restored.restore_bank_state(wire(state), binding=BINDING)
    assert [restored.take_cache("native-input") for _ in range(8)] == expected
    assert restored._cache_consumptions == owner._cache_consumptions
    before = restored.snapshot_bank_state(binding=BINDING)
    with pytest.raises(ValueError):
        restored.restore_bank_state(changed(before, lambda p: p["consumptions"].update(foreign=1)), binding=BINDING)
    assert restored.snapshot_bank_state(binding=BINDING) == before


def rapidock(tmp_path):
    (tmp_path / "weights.pt").write_bytes(b"fake native assets")
    return RAPiDockRunner(dict(python_bin=sys.executable, repo_path=str(tmp_path), model_dir=str(tmp_path),
                              ckpt="weights.pt", work_dir=str(tmp_path / "work"),
                              response_cache_dir=str(tmp_path / "cache")))


def test_rapidock_bank_and_consume_identity_restored_without_model(tmp_path, monkeypatch):
    protein, peptide = tmp_path / "protein.pdb", tmp_path / "peptide.pdb"
    protein.write_text("protein input")
    peptide.write_text("peptide input")
    paths = [tmp_path / f"sample-{i}.pdb" for i in range(3)]
    for i, path in enumerate(paths):
        path.write_text(f"sample {i}")
    owner = rapidock(tmp_path)
    key = owner._cache_key(protein_pdb=str(protein), peptide_pdb=str(peptide))
    owner._response_cache_mem[key] = [str(p) for p in paths]
    args = dict(name="AB", protein_pdb=str(protein), peptide_pdb=str(peptide))
    assert owner.dock_candidate(**args) == str(paths[0])
    state = owner.export_state()
    expected = owner.dock_candidate(**args)
    restored = rapidock(tmp_path)
    monkeypatch.setattr(restored, "_ensure_daemon", lambda: pytest.fail("model initialization"))
    monkeypatch.setattr(restored, "_request_bank", lambda **kw: pytest.fail("inference replay"))
    restored.restore_state(wire(state))
    assert restored.dock_candidate(**args) == expected
    assert restored._proc is None and restored._cursor_mem[key] == 2
    before = restored.snapshot_state(binding=BINDING)
    with pytest.raises(ValueError):
        restored.restore_state(changed(before, lambda p: p["banks"][key].update(cursor=99)), binding=BINDING)
    assert restored.snapshot_state(binding=BINDING) == before
    restored._unknown = RequestOutcomeUnknown("lost response")
    with pytest.raises(RequestOutcomeUnknown):
        restored.restore_state(wire(before), binding=BINDING)
    with pytest.raises(RequestOutcomeUnknown):
        restored.dock_candidate(**args)


def test_predocker_unknown_never_becomes_soft_failure_or_cache_replay(tmp_path, monkeypatch):
    owner = CacheOwner()
    error = RequestOutcomeUnknown("sent, response lost")
    calls = []

    def fail(path):
        calls.append(path)
        raise error

    monkeypatch.setattr(owner, "_run_impl", fail)
    for _ in range(2):
        with pytest.raises(RequestOutcomeUnknown):
            owner.run("native-input")
    assert calls == ["native-input"]
    assert owner._cache == {} and owner._last_operation["status"] == "outcome_unknown"
    with pytest.raises(RequestOutcomeUnknown):
        owner.snapshot_bank_state(binding=BINDING)


def test_predocker_known_scientific_failure_remains_distinct(tmp_path, monkeypatch):
    owner = CacheOwner()
    artifact = tmp_path / "penalty.pdb"
    artifact.write_text("scientific penalty geometry")

    def penalty(path):
        owner._known_scientific_failure = {"reason": "no_feasible_pose"}
        return str(artifact)

    monkeypatch.setattr(owner, "_run_impl", penalty)
    owner.run("native-input")
    assert owner._last_operation["status"] == "known_scientific_failure"
    state = owner.snapshot_bank_state(binding=BINDING)
    restored = CacheOwner()
    restored.restore_bank_state(wire(state), binding=BINDING)
    monkeypatch.setattr(restored, "_run_impl", lambda path: pytest.fail("replayed known failure"))
    restored.run("native-input")
    assert restored._last_operation["status"] == "known_scientific_failure"


def test_haddock_unknown_does_not_fall_through_to_another_anchor(tmp_path, monkeypatch):
    from astrolabe.structure_processors.haddock3 import HADDOCK3NucleicDockingProcessor
    owner = HADDOCK3NucleicDockingProcessor.__new__(HADDOCK3NucleicDockingProcessor)
    owner.binder_chain, owner.skip_if_contact = "A", False
    source = tmp_path / "input.pdb"
    source.write_text("input")
    for name, impl in {
        "_resolve_target_chains": lambda path: ("B", "C"),
        "_resolve_anchor_chains": lambda chains: chains,
        "_prepare_work_dir": lambda path: tmp_path / "anchors",
        "_write_chain_subset": lambda *a: None,
        "_write_restraints": lambda *a: "restraints",
        "_write_config": lambda *a: "config",
    }.items():
        monkeypatch.setattr(owner, name, impl)
    calls = []

    def unknown(*args):
        calls.append(args)
        raise RequestOutcomeUnknown("anchor sent")

    monkeypatch.setattr(owner, "_run_haddock", unknown)
    with pytest.raises(RequestOutcomeUnknown):
        owner._run_impl(str(source))
    assert len(calls) == 1


def test_vina_geometry_retry_rejects_wrapped_unknown(monkeypatch):
    from astrolabe.structure_processors.autodock import AutoDockVinaLigandPoseRescue, DockingPenaltyError
    owner = AutoDockVinaLigandPoseRescue.__new__(AutoDockVinaLigandPoseRescue)

    def unknown(*args, **kwargs):
        try:
            raise RequestOutcomeUnknown("native result unknown")
        except RequestOutcomeUnknown as error:
            raise DockingPenaltyError("geometry", reason="alpha_shape_pose_filter_failed") from error

    monkeypatch.setattr(owner, "_merge_ligand_with_receptor", unknown)
    monkeypatch.setattr(owner, "_should_retry_alpha_shape_filter", lambda e: pytest.fail("unknown retried"))
    with pytest.raises(DockingPenaltyError):
        owner._merge_ligand_with_receptor_with_retry("docked", "receptor", receptor_pdbqt_path="r",
                                                    ligand_pdbqt_path="l", binding_site={})


def test_rapidock_predocker_preserves_unknown_exception(monkeypatch):
    from astrolabe.structure_processors.peptide_docker import RAPiDockPeptideDockingProcessor
    owner = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)
    error = RequestOutcomeUnknown("sent")

    def unknown(path):
        raise error

    monkeypatch.setattr(owner, "_split_complex_chains", unknown)
    with pytest.raises(RequestOutcomeUnknown) as caught:
        owner._run_impl("input")
    assert caught.value is error
