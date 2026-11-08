from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design.generated_structure import (
    GENERATED_STRUCTURE_CODEC, GENERATED_STRUCTURE_REFERENCE_CODECS,
    bind_generated_structure, materialize_generated_structure,
    publish_generated_structure, read_generated_structures,
)
from astrolabe.design.legacy import LEGACY_REFERENCE_CODECS, LegacyContext
from astrolabe.design.native_mutation import make_mutator_operation
from astrolabe.design.native_scores import snapshot_scores
from astrolabe.design.native_support import NativeBindings
from astrolabe.design.operations import INVOKE
from astrolabe.design.runtime import ContractConflict, Runtime
from astrolabe.design.store import DataStore, JSONCodec, design_codec
from astrolabe.design.types import (
    Artifact, Blob, Evidence, EvidenceUse, History, Proposal,
    ProposeRequest, Requirement, Sample, SamplingPolicy, key_for, manifest,
)
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import build_mutators
from astrolabe.mutators.rfd4 import RFD4Mutator, structure_sequences
from astrolabe.mutators.rfd4_types import ChainSpec, ConditionPlan, GeneratedStructure, GenerationResult, ResidueRef


def _structure(path, sequences, *, shift=0.0):
    from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure
    from Bio.SeqUtils import seq3

    structure = Structure.Structure("test")
    model = Model.Model(0)
    structure.add(model)
    for chain_index, (chain_id, sequence) in enumerate(sequences.items()):
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for position, token in enumerate(sequence, 1):
            residue = Residue.Residue((" ", position, " "), seq3(token).upper(), " ")
            chain.add(residue)
            residue.add(Atom.Atom("CA", np.asarray([position * 3.8, chain_index * 6.0, shift]),
                                  0.0, 1.0, " ", " CA ", position, element="C"))
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(path))
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import astrolabe.mutators.rfd4_backend as backend_module

    global_config.reset()
    for name, value in (("segments", {}), ("protomers", {}), ("mutable_protomers", {}),
                        ("segment_manager", SegmentManager()), ("named_files", {}),
                        ("exclude_AA", ["W"])):
        global_config.init_const(name, value)
    global_config.set_variable("mutation_num", 2)
    a = Protomer("A", "ACDE", "p")
    b = Protomer("B", "+GG", "p")
    global_config.protomers.update(A=a, B=b)
    global_config.mutable_protomers.update(A=a)
    source = _structure(tmp_path / "accepted.cif", {"A": "ACDE", "B": "GG"})
    weights = tmp_path / "weights"
    weights.write_bytes(b"weights-v1")
    software = tmp_path / "backend.py"
    software.write_bytes(b"backend-v1")
    plans = []
    feedback_calls = []
    feedback_module = ModuleType("astrolabe.mutators.rfd4_feedback")

    def analyze_interface(*args, **kwargs):
        feedback_calls.append((args, kwargs))
        return "measured-feedback"

    def plan_conditions(chains, eligible, budget, **kwargs):
        plans.append((chains, eligible, budget, kwargs))
        sites = sorted(eligible, key=lambda site: -(kwargs.get("site_scores") or {}).get(site, 0))
        released = tuple(sites[:budget])
        fixed = tuple(ResidueRef(chain.chain, position)
                      for chain in chains if chain.chain == "B"
                      for position in range(1, len(chain.sequence) + 1))
        return ConditionPlan(released, fixed, excluded_amino_acids=kwargs["excluded_amino_acids"])

    feedback_module.analyze_interface = analyze_interface
    feedback_module.plan_conditions = plan_conditions
    monkeypatch.setitem(sys.modules, feedback_module.__name__, feedback_module)

    class FakeBackend:
        def __init__(self, **kwargs):
            self.calls = []
            self.override = None
            self.shift = 0.0

        def fingerprint(self):
            return {"weights": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "software": hashlib.sha256(software.read_bytes()).hexdigest()}

        def generate(self, request):
            self.calls.append(request)
            sequences = {chain.chain: list(chain.sequence) for chain in request.chains}
            for site in request.plan.released:
                original = sequences[site.chain][site.position - 1]
                sequences[site.chain][site.position - 1] = "T" if original != "T" else "A"
            sequences = {chain: "".join(sequence) for chain, sequence in sequences.items()}
            sequences.update(self.override or {})
            root = tmp_path / f"cache-{len(self.calls)}"
            root.mkdir()
            output = _structure(root / "generated.cif", sequences, shift=self.shift)
            annotated = root / "annotated.cif"
            annotated.write_bytes(Path(request.structure_path).read_bytes())
            metadata = root / "manifest.json"
            metadata.write_text(json.dumps({"request": asdict(request), "fingerprint": self.fingerprint()}))
            return GeneratedStructure(str(output), str(annotated), str(metadata), tuple(sequences.items()),
                                      key_for(asdict(request)).digest,
                                      hashlib.sha256(output.read_bytes()).hexdigest(), False)

        def shutdown(self):
            pass

    monkeypatch.setattr(backend_module, "RFD4Backend", FakeBackend)
    mutator = RFD4Mutator(assembly="AB", binder_chains=["A"], target_chains=["B"],
                          python_bin="unused", source_dir="unused", model_dir="unused", cache_dir="unused")
    scores = StandardScores(names=["AB"], plddt={}, complex_paths={"AB": source})
    yield SimpleNamespace(mutator=mutator, scores=scores, source=source, plans=plans,
                          feedback_calls=feedback_calls, weights=weights, software=software, root=tmp_path)
    global_config.reset()


def test_rfd4_stages_all_generated_differences_without_probability_sampling(setup):
    data = DataStore()
    codec = JSONCodec()
    config = {"step": 4}
    prepared = setup.mutator.prepare_native_mutation(global_config.mutable_protomers, setup.scores, config)
    reference = data.put(prepared, codec)
    assert "structure_path" not in prepared["request"]
    setup.scores.complex_paths["AB"] = f"{setup.source.parent}/./{setup.source.name}"
    alias = setup.mutator.prepare_native_mutation(global_config.mutable_protomers, setup.scores, config)
    assert data.put(alias, codec) == reference
    relocated = setup.root / "relocated" / setup.source.name
    relocated.parent.mkdir()
    shutil.copyfile(setup.source, relocated)
    setup.scores.complex_paths["AB"] = str(relocated)
    moved = setup.mutator.prepare_native_mutation(global_config.mutable_protomers, setup.scores, config)
    assert data.put(moved, codec) == reference
    _structure(relocated, {"A": "ACDE", "B": "GG"}, shift=1.0)
    changed = setup.mutator.prepare_native_mutation(global_config.mutable_protomers, setup.scores, config)
    assert changed["input_hex"] != prepared["input_hex"]
    assert data.put(changed, codec) != reference
    restored = data.get(reference, codec)
    result = setup.mutator.mutate_prepared(restored, global_config.mutable_protomers, setup.scores, config)
    assert isinstance(result, GenerationResult)
    assert setup.plans[0][2] == 2
    request = setup.mutator.backend.calls[0]
    assert request.plan.excluded_amino_acids == ("W",)
    assert len(request.plan.released) == 2
    assert global_config.protomers["A"].sequence == "ACDE"
    assert global_config.protomers["A"].mutated_sequence == "TTDE"
    assert global_config.protomers["B"].mutated_sequence == "GG"
    assert len(global_config.segments["A_1"].mutations) == 2
    assert structure_sequences(result.generated.structure_path) == dict(result.generated.sequences)
    assert global_config.segments["A_1"].probs is None


@pytest.mark.parametrize("empty_selector", [False, True])
def test_rfd4_zero_budget_or_empty_selector_never_generates(setup, monkeypatch, empty_selector):
    if empty_selector:
        import astrolabe.mutators.rfd4 as module

        def exclude(*args, **kwargs):
            assert kwargs["fallback_to_all_mutable_if_selector_empty"] is False
            return {"A": list(range(4))}

        monkeypatch.setattr(module, "prepare_exclude_sites", exclude)
    else:
        global_config.set_variable("mutation_num", 0)
    assert setup.mutator.mutate(global_config.mutable_protomers, setup.scores, {}) is None
    assert not setup.mutator.backend.calls
    assert not setup.plans


@pytest.mark.parametrize("violation", ["source", "pending", "preupdate", "shared", "ptm", "bias", "probability"])
def test_rfd4_rejects_unrepresentable_sources_before_inference(setup, violation):
    config = {}
    if violation == "source":
        _structure(setup.source, {"A": "AAAA", "B": "GG"})
    elif violation == "pending":
        global_config.segments["A_1"].mutations = [(0, "A", "T", 0.0, 0.0)]
    elif violation == "preupdate":
        config["mutator_stack_pre_update"] = True
    elif violation == "shared":
        global_config.protomers["C"] = Protomer("C", "A_1", "p")
    elif violation == "ptm":
        global_config.protomers["A"]._ptm_dict = {0: "MSE"}
    elif violation == "bias":
        config["aa_bias"] = {"T": 1.0}
    else:
        config["transition_consistency_gate_context"] = object()
    with pytest.raises(ValueError):
        setup.mutator.mutate(global_config.mutable_protomers, setup.scores, config)
    assert not setup.mutator.backend.calls


@pytest.mark.parametrize("override", [{"A": "TTTT"}, {"B": "AG"}, {"A": "T"}, {"A": "WWDE"}])
def test_rfd4_rejects_changed_fixed_targets_or_nonreleased_sequences(setup, override):
    setup.mutator.backend.override = override
    with pytest.raises(ValueError):
        setup.mutator.mutate(global_config.mutable_protomers, setup.scores, {})
    assert not any(segment.mutations for segment in global_config.segments.values())


def test_rfd4_named_site_scores_and_fresh_feedback_reach_preinference_plan(setup):
    setup.mutator.feedback = {"enabled": True, "chemistry": {"buried_sasa_cutoff": 2.0},
                              "retention": {"energy_top_k": 1}}
    setup.scores.per_proto_score_terms = {"A": {"badness": [np.asarray([1., 3., 2., 9.])]}}
    config = {"per_residue_scores": {"mutator_score_terms": {"badness": 1.0}}}
    setup.mutator.mutate(global_config.mutable_protomers, setup.scores, config)
    assert len(setup.feedback_calls) == 1
    assert setup.feedback_calls[0][0][0] == str(setup.source)
    assert setup.plans[0][3]["feedback"] == "measured-feedback"
    assert setup.plans[0][3]["policy"] == {"energy_top_k": 1, "binder_chains": ["A"], "target_chains": ["B"]}
    assert set(setup.mutator.backend.calls[0].plan.released) == {ResidueRef("A", 2), ResidueRef("A", 4)}
    assert global_config.protomers["A"].mutated_sequence == "ATDT"


def test_rfd4_static_plan_uses_current_parent_and_validates_partition(setup):
    released = (ResidueRef("A", 1), ResidueRef("A", 2))
    fixed = (ResidueRef("A", 3), ResidueRef("A", 4),
             ResidueRef("B", 1), ResidueRef("B", 2))
    path = setup.root / "initial-plan.json"
    record = {"source_sha256": "initial", "plan": asdict(ConditionPlan(
        released, fixed, excluded_amino_acids=("W",),
    ))}
    path.write_text(json.dumps(record))
    setup.mutator.feedback = {"enabled": True, "fixed_plan_path": str(path)}
    request = setup.mutator.prepare_generation(
        global_config.mutable_protomers, setup.scores, {"step": 2},
    )
    assert request.structure_path == str(setup.source)
    assert request.plan.released == released and request.plan.fixed_coordinates == fixed
    assert not setup.feedback_calls and not setup.plans
    assert json.loads(request.plan.feedback_json)["condition_update"]["mode"] == "fixed_initial_identities"
    record["plan"]["fixed_coordinates"] = record["plan"]["fixed_coordinates"][:-1]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Fixed RFD4 plan"):
        setup.mutator.prepare_generation(global_config.mutable_protomers, setup.scores, {})


def test_proposal_feedback_counts_only_measured_failures_and_is_idempotent(setup, monkeypatch):
    from astrolabe.mutators.rfd4_types import AtomRef, HBondCondition
    atom = SimpleNamespace(atom=AtomRef("B", 1, "N"), role="DONOR", source_present=True,
                           satisfied=False, qualifying_hbond_count=0, sasa=.5,
                           nearest_other_group_heavy_atom_distance=3.)
    module = sys.modules["astrolabe.mutators.rfd4_feedback"]
    monkeypatch.setattr(module, "analyze_interface", lambda *a, **kw: SimpleNamespace(polar_atoms=(atom,)))
    setup.mutator.feedback = {"repair_unsatisfied_attempts": True}
    request = SimpleNamespace(plan=ConditionPlan((), (), (HBondCondition(atom.atom, atom.role),)))
    generated = SimpleNamespace(request_digest="one", structure_digest="raw-one", structure_path="unused")
    setup.mutator._observe_hbond_attempt(request, generated)
    setup.mutator._observe_hbond_attempt(request, generated)
    assert dict(setup.mutator._failed_hbond_attempts) == {"B:1:N": 1}
    assert len(setup.mutator._hbond_attempt_history) == 1
    atom.satisfied = True
    atom.qualifying_hbond_count = 1
    generated.request_digest = "two"
    setup.mutator._observe_hbond_attempt(request, generated)
    assert dict(setup.mutator._failed_hbond_attempts) == {"B:1:N": 1}
    assert setup.mutator._hbond_attempt_history[-1]["outcomes"][0]["satisfied"] is True


def test_rfd4_fixed_target_copies_are_kept_in_complete_physical_mapping(setup):
    setup.mutator.assembly = "ABB"
    setup.mutator.target_chains = ("B", "C")
    _structure(setup.source, {"A": "ACDE", "B": "GG", "C": "GG"})
    setup.scores.complex_paths = {"ABB": setup.source}
    result = setup.mutator.mutate(global_config.mutable_protomers, setup.scores, {})
    assert [(chain.chain, chain.protomer) for chain in result.chains] == [("A", "A"), ("B", "B"), ("C", "B")]
    assert dict(result.generated.sequences) == {"A": "TTDE", "B": "GG", "C": "GG"}


def test_rfd4_registered_factory_and_unsupported_aa_bias(setup):
    config = OmegaConf.create({"mutator_terms": {"joint": {
        "_type": "rfd4", "assembly": "AB", "binder_chains": ["A"], "target_chains": ["B"],
        "python_bin": "unused", "source_dir": "unused", "model_dir": "unused", "cache_dir": "unused",
    }}})
    assert isinstance(build_mutators(config, setup.root)["joint"], RFD4Mutator)
    config.LigandMPNN = {"bias_AA": "W:-1"}
    with pytest.raises(ValueError, match="bias"):
        build_mutators(config, setup.root)


def test_generated_structure_registry_initial_sequence_and_debug_cache(setup, monkeypatch):
    import run
    from astrolabe.evaluators.generated_structure import GeneratedStructureEvaluator
    from astrolabe.evaluators.utils import construct_evaluators
    from astrolabe.utils.base import construct_standard_input

    config = OmegaConf.create({"enabled_aliases": ["generated"], "evaluator_terms": {
        "generated": {"_type": "generated_structure", "initial_structures": {"AB": str(setup.source)}},
        "unused": {"_type": "not_installed"},
    }})
    evaluator = construct_evaluators(config, setup.root, enabled_aliases=config.enabled_aliases)["generated"]
    assert isinstance(evaluator, GeneratedStructureEvaluator)
    inputs = construct_standard_input(["AB"], 17)
    args = SimpleNamespace(seed=17, skip_complex_poses=True, per_residue_scores=None)
    cache = setup.root / "initial_eval_rslt.pkl"
    cache.write_bytes(b"invalid old prediction cache")
    monkeypatch.setenv("DEBUG", "1")
    monkeypatch.setattr(run, "_post_eval_hook_config", lambda *_: pytest.fail("must not repack generated coordinates"))
    scores = run.load_or_evaluate_initial(args, evaluator, inputs, {}, str(setup.root / "initial"))
    assert scores.names == ["AB"] and scores.complex_paths == {"AB": setup.source}
    assert scores.plddt == {} and scores.iptm is None and scores.ptm is None
    assert cache.read_bytes() == b"invalid old prediction cache"
    _structure(setup.source, {"A": "AAAA", "B": "GG"})
    with pytest.raises(ValueError, match="sequence"):
        evaluator.evaluate(inputs, {"step": 0})


def test_generated_structure_runner_uses_durable_current_conformer_and_clears_noop(setup, monkeypatch):
    import run
    from astrolabe.evaluators.generated_structure import GeneratedStructureEvaluator
    from astrolabe.utils.base import construct_standard_input

    global_config.init_const("out_dir", setup.root / "results")
    evaluator = GeneratedStructureEvaluator(initial_structures={"AB": setup.source})
    runtime = {}
    args = SimpleNamespace(seed=17, skip_complex_poses=True, per_residue_scores=None)
    monkeypatch.setattr(run, "_post_eval_hook_config", lambda *_: pytest.fail("must not repack generated coordinates"))
    paths, digests = [], []
    for step in (1, 2):
        global_config.segment_manager.clear_mutations()
        setup.mutator.backend.shift = float(step)
        run.execute_mutation_step([setup.mutator], setup.scores, runtime, step)
        record = runtime["generated_structure_records"]["AB"]
        assert record["step"] == step
        assert Path(record["index_path"]).is_relative_to(setup.root / "results" / "generated_structures")
        shutil.rmtree(setup.root / f"cache-{step}")
        # The real mutation -> evaluation work-item route retains the handoff.
        scores = run.perform_evaluation(args, evaluator, runtime, step, ["AB"])
        paths.append(scores.complex_paths["AB"])
        digests.append(hashlib.sha256(paths[-1].read_bytes()).hexdigest())
        assert structure_sequences(paths[-1]) == {"A": "TTDE", "B": "GG"}
        assert scores.plddt == {} and scores.iptm is None
    assert paths[0] != paths[1] and digests[0] != digests[1]
    assert paths[0].exists()
    # Even a sequence-matching old record must not be accepted as a new step.
    with pytest.raises(ValueError, match="Stale"):
        evaluator.evaluate(construct_standard_input(["AB"], 17), {**runtime, "step": 3})
    global_config.segment_manager.clear_mutations()
    global_config.set_variable("mutation_num", 0)
    run.execute_mutation_step([setup.mutator], setup.scores, runtime, 3)
    assert runtime["generated_structure_records"] == {}
    with pytest.raises(ValueError, match="Missing current-step"):
        evaluator.evaluate(construct_standard_input(["AB"], 17), runtime)


def test_generated_structure_accepts_only_explicit_single_ccd_target(setup):
    from Bio.PDB import Atom, Chain, MMCIFIO, MMCIFParser, Residue
    from astrolabe.evaluators.generated_structure import GeneratedStructureEvaluator
    from astrolabe.utils.base import construct_standard_input

    _structure(setup.source, {"A": "ACDE"})
    structure = MMCIFParser(QUIET=True).get_structure("ligand", str(setup.source))
    chain = Chain.Chain("B")
    residue = Residue.Residue(("H_CFF", 1, " "), "CFF", " ")
    for index, name in enumerate(("O11", "O13", "N9")):
        residue.add(Atom.Atom(name, np.asarray([index, 4.0, 0.0]), 0.0, 1.0, " ",
                             name, index + 10, element=name[0]))
    chain.add(residue)
    structure[0].add(chain)
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(setup.source))
    chains = (ChainSpec("A", "A", "ACDE"), ChainSpec("B", "B", "CFF", kind="ligand"))
    assert structure_sequences(setup.source, chains) == {"A": "ACDE", "B": "CFF"}
    with pytest.raises(ValueError, match="chemical components"):
        structure_sequences(setup.source)
    old_segment = global_config.segments.pop("B_1")
    global_config.segment_manager.segments.pop("B_1")
    global_config.segment_manager.unique_segments.pop(id(old_segment))
    global_config.segment_manager.primary_names.pop(id(old_segment))
    global_config.protomers["B"] = Protomer("B", "+CFF", "c")
    inputs = construct_standard_input(["AB"], 17)
    evaluator = GeneratedStructureEvaluator(initial_structures={"AB": setup.source})
    scores = evaluator.evaluate(inputs, {"step": 0})
    assert scores.complex_paths["AB"] == setup.source
    with pytest.raises(ValueError, match="chemical components"):
        structure_sequences(setup.source, (chains[0], ChainSpec("B", "B", "ATP", kind="ligand")))


@pytest.mark.parametrize("corruption", ["assembly", "attempt", "request_digest", "structure_digest", "sequences", "bytes", "coordinate_sequence", "missing"])
def test_generated_structure_rejects_invalid_current_handoff(setup, corruption):
    import run
    from astrolabe.evaluators.generated_structure import GeneratedStructureEvaluator
    from astrolabe.utils.base import construct_standard_input

    global_config.init_const("out_dir", setup.root / "results")
    runtime = {}
    run.execute_mutation_step([setup.mutator], setup.scores, runtime, 1)
    record = runtime["generated_structure_records"]["AB"]
    index_path = Path(record["index_path"])
    index = json.loads(index_path.read_text())
    path = index_path.parent / "rfd4.generated.cif"
    if corruption == "bytes":
        path.write_bytes(path.read_bytes() + b"\n")
    elif corruption == "coordinate_sequence":
        _structure(path, {"A": "AAAA", "B": "GG"})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record["structure_digest"] = index["structure_digest"] = digest
        index["files"][path.name] = {"sha256": digest, "size": path.stat().st_size}
    elif corruption == "missing":
        path.unlink()
    elif corruption == "sequences":
        index["sequences"] = [["A", "AAAA"], ["B", "GG"]]
    else:
        index[corruption] = "wrong"
    index_path.write_text(json.dumps(index))
    evaluator = GeneratedStructureEvaluator(initial_structures={"AB": setup.source})
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluator.evaluate(construct_standard_input(["AB"], 17), runtime)


def test_generated_structure_recovery_rejected_before_builder(setup, monkeypatch):
    import run

    # Rejection must precede model setup and any portable recovery construction.
    monkeypatch.setattr(run, "_shutdown_pipeline_resources", lambda: None)
    config = OmegaConf.create({"recovery": {"mode": "resume"}, "evaluator": {"evaluator_terms": {
        "generated": {"_type": "generated_structure", "initial_structures": {"AB": str(setup.source)}},
    }}})
    with pytest.raises(ValueError, match="generated_structure evaluator does not support recovery"):
        run._run_pipeline(config)


def test_rfd4_generated_artifact_can_be_inspected_without_the_raw_cache(setup):
    data = DataStore(setup.root / "data")
    context = LegacyContext()
    base = context.capture_candidate(assemblies=("AB",), data=data)
    result = setup.mutator.mutate(global_config.mutable_protomers, setup.scores, {})
    global_config.segment_manager.apply_mutations()
    proposed = context.capture_candidate(assemblies=("AB",), data=data).candidate
    sample = Sample(key_for("protocol"), "0", 7, "0")
    artifact = bind_generated_structure(
        result, data=data, system=base.system, base=base.candidate,
        proposed=proposed, sample=sample, inputs=manifest(()),
    )
    proposal = Proposal(key_for("proposal"), (base.candidate.identity,), base.candidate.identity,
                        proposed, (), key_for("policy"), (artifact,))
    reference = data.put(proposal, design_codec)
    data.persist((reference,))
    shutil.rmtree(Path(result.generated.structure_path).parent)
    reopened = DataStore(setup.root / "data")
    for codec in (design_codec, *LEGACY_REFERENCE_CODECS, *GENERATED_STRUCTURE_REFERENCE_CODECS):
        reopened.register(codec)
    restored = reopened.get(reference, design_codec)
    record, = read_generated_structures(restored, reopened)
    assert dict(record.sequences) == {"A": "TTDE", "B": "GG"}
    assert record.base_candidate == base.candidate.identity
    paths = materialize_generated_structure(restored, reopened, setup.root / "restored")
    assert structure_sequences(paths[0]) == {"A": "TTDE", "B": "GG"}
    assert json.loads(paths[-1].read_text())["validation"] == "generated_unrefolded"
    wrong_occurrence = replace(artifact.occurrences[0], chain="Z")
    invalid = replace(proposal, artifacts=(replace(artifact, occurrences=(wrong_occurrence,)),))
    with pytest.raises(ValueError, match="identity/ownership"):
        read_generated_structures(invalid, data)


def _native(setup, data):
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("AB",), data=data)
    bindings = NativeBindings(context, translation, data, setup.root / "work")
    sample = Sample(key_for("protocol"), "0", 17, "0")
    sampling = SamplingPolicy(key_for("sampling"), data.put({}, JSONCodec()))
    content = snapshot_scores(setup.scores, data)
    inputs = manifest(())
    artifact = Artifact(key_for("scores"), translation.candidate.identity, translation.system.occurrences,
                        sample, content, (), inputs)
    evidence = Evidence(key_for("evidence"), translation.candidate.identity, (artifact,), (), inputs)
    request = ProposeRequest(translation.system, (translation.candidate,), translation.candidate,
                             (EvidenceUse(evidence, "current", translation.candidate.identity),),
                             (), None, sampling, sample)
    operation = make_mutator_operation(
        "joint", lambda: setup.mutator, bindings, sampling_policy=sampling,
        version="test-rfd4", model_identity=key_for("rfd4-model"), prepare_native=True,
    )
    ready = operation.initialization.invoke(operation.initialization.prepare(operation.initialization_request))
    operation.initialization.on_result(ready)
    return operation, request, translation


def test_rfd4_native_artifact_retains_structure_and_full_blob_closure_after_restart(setup):
    data = DataStore(setup.root / "data")
    operation, request, translation = _native(setup, data)
    runtime_root = setup.root / "runtime"
    requirements = (Requirement("joint", INVOKE, "required"),)
    with Runtime((operation,), root=runtime_root, data=data) as runtime:
        runtime.preflight(requirements)
        with runtime.bind("trial", History(key_for(()), ())):
            proposals = runtime.call("propose", operation, request)
    proposal = proposals.items[0]
    assert proposal.candidate != request.working
    assert len(proposal.artifacts) == 1
    assert proposal.artifacts[0].content.codec == GENERATED_STRUCTURE_CODEC.identity
    record, = read_generated_structures(proposal, data)
    assert record.validation == "generated_unrefolded"
    assert record.sample == request.sample
    assert len(proposal.artifacts[0].occurrences) == 2
    for root in setup.root.glob("cache-*"):
        shutil.rmtree(root)
    reopened = DataStore(setup.root / "data")
    cold_operation, cold_request, _ = _native(setup, reopened)
    with Runtime((cold_operation,), root=runtime_root, data=reopened) as runtime:
        runtime.preflight(requirements)
        with runtime.bind("trial", History(key_for(()), ())):
            restored = runtime.call("propose", cold_operation, cold_request).items[0]
    assert len(setup.mutator.backend.calls) == 1
    assert read_generated_structures(restored, reopened) == (record,)
    files = materialize_generated_structure(restored, reopened, setup.root / "inspection")
    assert len(files) == 4
    assert structure_sequences(files[0]) == {"A": "TTDE", "B": "GG"}
    assert json.loads(files[-1].read_text())["validation"] == "generated_unrefolded"
    with pytest.raises(ValueError, match="ownership"):
        read_generated_structures(replace(restored, base=key_for("wrong-base")), reopened)


def test_rfd4_input_or_weights_change_invalidates_outer_prepared_identity(setup):
    setup.mutator.seed = 3
    config = {"step": 2}
    assert setup.mutator.prepare_generation(global_config.mutable_protomers, setup.scores, config).seed == 5
    global_config.init_const("seed", 17)
    assert setup.mutator.prepare_generation(global_config.mutable_protomers, setup.scores, config).seed == 22
    global_config.update_const("seed", 29)
    assert setup.mutator.prepare_generation(global_config.mutable_protomers, setup.scores, config).seed == 34
    assert setup.mutator.prepare_generation(global_config.mutable_protomers, setup.scores,
                                            {"step": 2, "seed": 17}).seed == 22
    assert setup.mutator.prepare_generation(global_config.mutable_protomers, setup.scores,
                                            {"step": 2, "seed": 0}).seed == 5
    data = DataStore(setup.root / "data")
    operation, request, _ = _native(setup, data)
    first = operation.prepare(request)
    prepared_blob = next(dependency for dependency in first.inputs.dependencies
                         if dependency.role == "native-prepared")
    assert data.get(Blob(prepared_blob.identity, JSONCodec().identity), JSONCodec())["request"]["seed"] == 20
    assert not setup.mutator.backend.calls
    setup.weights.write_bytes(b"weights-v2")
    second = operation.prepare(request)
    assert first.inputs.identity != second.inputs.identity
    with pytest.raises(ValueError, match="assets changed"):
        operation.invoke(first)
    assert not setup.mutator.backend.calls
    _structure(setup.source, {"A": "ACDE", "B": "GG"}, shift=1.0)
    third = operation.prepare(request)
    assert third.inputs.identity == second.inputs.identity


@pytest.mark.parametrize("asset", ["weights", "software"])
def test_rfd4_outer_runtime_rejects_asset_changes_before_serving_a_cached_proposal(setup, asset):
    data = DataStore(setup.root / "data")
    operation, request, _ = _native(setup, data)
    with Runtime((operation,), root=setup.root / "runtime", data=data) as runtime:
        runtime.preflight((Requirement("joint", INVOKE, "required"),))
        with runtime.bind("trial", History(key_for(()), ())):
            first = runtime.call("propose", operation, request)
            assert runtime.call("propose", operation, request) == first
            assert len(setup.mutator.backend.calls) == 1
            operation.bindings = replace(operation.bindings, work_root=setup.root / "relocated-work")
            assert runtime.call("propose", operation, request) == first
            assert len(setup.mutator.backend.calls) == 1
            getattr(setup, asset).write_bytes(b"changed-in-place")
            with pytest.raises(ContractConflict):
                runtime.call("propose", operation, request)
            assert len(setup.mutator.backend.calls) == 1


def test_rfd4_same_sequence_conformers_have_distinct_proposal_identities(setup):
    data = DataStore(setup.root / "data")
    operation, request, _ = _native(setup, data)
    first = operation.invoke(operation.prepare(request)).items[0]
    setup.mutator.backend.shift = 2.0
    second = operation.invoke(operation.prepare(request)).items[0]
    assert first.candidate == second.candidate
    assert first.identity != second.identity
    a, = read_generated_structures(first, data)
    b, = read_generated_structures(second, data)
    assert a.structure_digest != b.structure_digest


def test_rfd4_cli_copies_survive_cache_cleanup_and_legacy_returns_are_ignored(setup):
    result = setup.mutator.mutate(global_config.mutable_protomers, setup.scores, {})
    index = publish_generated_structure(result, setup.root / "published", attempt="round-3")
    shutil.rmtree(Path(result.generated.structure_path).parent)
    record = json.loads(index.read_text())
    assert record["attempt"] == "round-3"
    assert record["validation"] == "generated_unrefolded"
    for filename, metadata in record["files"].items():
        assert hashlib.sha256((index.parent / filename).read_bytes()).hexdigest() == metadata["sha256"]
    for value in (None, {}, ("sequence", "legacy"), [1, 2]):
        assert publish_generated_structure(value, setup.root / "legacy") is None
    assert not (setup.root / "legacy").exists()
