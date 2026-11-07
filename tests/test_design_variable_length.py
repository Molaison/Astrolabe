"""CPU-only contracts for bounded whole-sequence candidates and their layout."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, ProtomerTypes, Segment, SegmentManager
from astrolabe.design.generated_structure import bind_generated_structure, read_generated_structures
from astrolabe.design.legacy import (
    LegacyContext, UnsupportedLegacyState, resolve_legacy_index_map,
    restore_native_sequence, translate_legacy_state,
)
from astrolabe.design.native_flow import NativeFlow
from astrolabe.design.native_mutation import NativeMutationOperation, UnsupportedNativeMutation
from astrolabe.design.store import DataStore, JSONCodec, design_codec
from astrolabe.design.types import (
    Assignment, Chemical, Domain, Polymer, Proposal, Sample,
    key_for, make_candidate, make_system, manifest,
)
from astrolabe.mutators.rfd4_types import ChainSpec, GeneratedStructure, GenerationResult


@pytest.fixture(autouse=True)
def clean_globals():
    global_config.reset()
    for name in ("segments", "protomers", "mutable_protomers", "named_files"):
        global_config.init_const(name, {})
    global_config.init_const("segment_manager", SegmentManager())
    yield
    global_config.reset()


def state(*, bounds=(2, 8)):
    protein = Protomer("A", "ACDE", "p")
    ligand = Protomer("L", "+CFF", "c")
    global_config.protomers.update(A=protein, L=ligand)
    global_config.mutable_protomers.update(A=protein)
    segment = global_config.segments["A_1"]
    segment.length_bounds = bounds
    data, context = DataStore(), LegacyContext()
    translation = context.capture_candidate(assemblies=("AL",), data=data)
    return SimpleNamespace(protein=protein, segment=segment, data=data, context=context,
                           translation=translation, manager=global_config.segment_manager)


def edit(s, sequence, *, base=None):
    return s.translation.system.edit(base or s.translation.candidate,
        (Assignment("A_1", Polymer("protein", tuple(sequence), ())),), data=s.data)


def capture(s):
    return translate_legacy_state(assemblies=("AL",), data=s.data)


def assert_layout(s, sequence):
    assert s.protein.sequence == sequence
    assert s.protein.components_slices == [slice(0, len(sequence))]
    assert s.protein.mutable_mask.tolist() == [True] * len(sequence)
    assert len(s.protein.tied_positions_dict) == len(sequence)
    assert s.protein.ptm_dict == {}


def test_grow_accept_shrink_reject_next_accept_keeps_logical_identity():
    s = state()
    initial = s.translation
    assert_layout(s, "ACDE")  # Prime masks and tied-position caches before growth.
    for sequence, accept in (("ACDEFG", True), ("AC", False), ("GHIKL", True)):
        previous = capture(s)
        accepted_slices = list(s.protein.components_slices)
        s.segment.stage_sequence(sequence)
        assert s.protein.mutated_sequence == sequence
        assert s.protein.sequence == "".join(previous.candidate.assignments[0].value.tokens)
        assert s.protein.components_slices == accepted_slices
        proposed = edit(s, sequence, base=previous.candidate)
        if accept:
            s.manager.apply_mutations()
            assert capture(s).candidate == proposed
        else:
            s.manager.clear_mutations()
            assert capture(s).candidate == previous.candidate
            assert_layout(s, "ACDEFG")
        current = capture(s)
        assert current.system == initial.system
        assert current.system.occurrences == initial.system.occurrences
        assert s.segment.pending_sequence is None
        ligand = next(item for item in current.system.occurrences if item.variable == "L_1")
        index = resolve_legacy_index_map(ligand, current.candidate, s.data)
        assert index["assembly_range"] == [s.protein.length, s.protein.length + 1]
        assert index["protomer_range"] == [0, 1]
    assert_layout(s, "GHIKL")
    with pytest.raises(ValueError, match="pre-update"):
        s.manager.begin_pre_update_step()


def test_apply_initial_equal_slots_and_restore_full_layout():
    s = state()
    initial = s.translation.candidate
    longer = edit(s, "ACDEFG")
    s.context.apply_candidate(longer, translation=s.translation, data=s.data)
    assert_layout(s, "ACDEFG")
    s.segment.stage_sequence("AC")
    snapshot = s.context.capture_state()
    s.context.apply_candidate(initial, translation=s.translation, data=s.data)
    assert_layout(s, "ACDE")
    assert s.segment.pending_sequence is None
    assert capture(s).candidate == initial
    s.context.restore_state(snapshot)
    assert_layout(s, "ACDEFG")
    assert s.segment.pending_sequence == "AC"
    s.manager.clear_mutations()
    # Applying an assignment equal to the translation slot must still overwrite globals.
    with s.context.activate(initial, translation=s.translation, data=s.data):
        assert_layout(s, "ACDE")
    assert_layout(s, "ACDEFG")
    restore_native_sequence("ACDE", s.protein, global_config.segments)
    assert_layout(s, "ACDE")


@pytest.mark.parametrize("sequence", ["A", "ACDEFGHIK", "ACX", "AC[SEP]", ""])
def test_out_of_bounds_and_noncanonical_proposals_do_not_change_pending(sequence):
    s = state()
    s.segment.stage_sequence("ACDEF")
    with pytest.raises(ValueError):
        s.segment.stage_sequence(sequence)
    assert s.segment.pending_sequence == "ACDEF"
    with pytest.raises(ValueError):
        edit(s, sequence)
    assert_layout(s, "ACDE")


@pytest.mark.parametrize("attribute,value", [
    ("length_bounds", None), ("is_fixed", True), ("ptm_dict", {1: "SEP"}),
    ("type", ProtomerTypes.RNA), ("mutations", [(0, "A", "G", 0., 0.)]),
])
def test_stage_sequence_requires_explicit_mutable_unmodified_protein(attribute, value):
    segment = Segment("ACDE", {}, length_bounds=(2, 8))
    setattr(segment, attribute, value)
    with pytest.raises(ValueError):
        segment.stage_sequence("ACDEF")
    assert segment.pending_sequence is None
    assert segment.sequence == "ACDE"


def test_default_fixed_length_and_domain_identity_guards():
    s = state(bounds=None)
    with pytest.raises(ValueError, match="token count"):
        edit(s, "ACDEF")
    fixed = make_system(s.translation.system.variables, s.translation.system.occurrences,
                        s.translation.system.topology, (Domain("A_1", (0,), (), None),))
    base = make_candidate(fixed, s.translation.candidate.assignments)
    with pytest.raises(ValueError, match="fixed token"):
        fixed.edit(base, (Assignment("A_1", Polymer("protein", tuple("GCDE"), ())),))
    for domain in (Domain("A_1", (0,), (), None, (2, 8)),
                   Domain("A_1", (), (), None, (8, 2)),
                   Domain("A_1", (), (), None, (True, 8))):
        with pytest.raises(ValueError):
            make_system(fixed.variables, fixed.occurrences, fixed.topology, (domain,))
    bounded = make_system(fixed.variables, fixed.occurrences, fixed.topology,
                          (Domain("A_1", (), (), None, (2, 8)),))
    assert bounded.identity != s.translation.system.identity
    with pytest.raises(ValueError, match="unmodified canonical"):
        make_candidate(bounded, (Assignment("A_1", Polymer("protein", tuple("ACDE"), ((0, "SEP"),))),
                                 s.translation.candidate.assignments[1]))


def test_fixed_length_domain_retains_pre_extension_identity_and_checkpoint_bytes():
    # Captured from c2eea23, before the variable-length extension.
    legacy = (b'{"$dataclass":"astrolabe.design.types.Domain","fields":'
              b'{"allowed_values":null,"fixed_atoms":{"$tuple":[]},'
              b'"fixed_tokens":{"$tuple":[0]},"variable":"A_1"}}')
    fixed = Domain("A_1", (0,), (), None)
    assert key_for(fixed).digest == "70dd868b6e8b6902f6aa2aa45d66ab9ee4069b1e0cc95598e4d4a4ad915721a3"
    assert design_codec.decode(legacy) == fixed
    assert design_codec.encode(fixed) == legacy
    bounded = Domain("A_1", (), (), None, (2, 8))
    assert design_codec.decode(design_codec.encode(bounded)) == bounded
    assert b"length_bounds" in design_codec.encode(bounded)
    assert key_for(bounded) != key_for(Domain("A_1", (), (), None))
    with pytest.raises(ValueError, match="fields"):
        design_codec.decode(legacy.replace(b'"allowed_values":null,', b""))


def test_fixed_length_activation_restores_initial_assignment_from_changed_live_state():
    s = state(bounds=None)
    initial = s.translation.candidate
    s.segment.sequence = "GHIK"
    s.manager.refresh_protomers()
    with s.context.activate(initial, translation=s.translation, data=s.data):
        assert s.protein.sequence == "ACDE"
    assert s.protein.sequence == "GHIK"


@pytest.mark.parametrize("boundary", ["fixed", "ptm", "locked", "homology", "shared", "multisegment", "repeat", "smiles"])
def test_translation_rejects_unsupported_variable_topology(boundary):
    s = state()
    assemblies = ("AL",)
    if boundary == "fixed":
        s.segment.is_fixed = True
    elif boundary == "ptm":
        s.segment.ptm_dict = {1: "SEP"}
    elif boundary == "locked":
        s.protein.add_locked_positions([1])
    elif boundary == "homology":
        s.protein._homology = {"unpairedMsa": ">query\nACDE\n"}
    elif boundary in {"shared", "multisegment"}:
        s.protein.components.append("A_1")
    elif boundary == "repeat":
        assemblies = ("AAL",)
    elif boundary == "smiles":
        global_config.segments["L_1"].type = ProtomerTypes.SMILES
    with pytest.raises(UnsupportedLegacyState):
        translate_legacy_state(assemblies=assemblies, data=s.data)


def _unused(*args, **kwargs):
    raise AssertionError("no inference or scientific initialization is allowed")


class FlowStub(SimpleNamespace):
    pass


@pytest.mark.parametrize("boundary", [None, "mixed", "preupdate", "recovery"])
def test_native_flow_declares_before_initial_translation(boundary):
    s = state(bounds=None)
    calls = []

    def declare():
        calls.append("declare")
        s.segment.length_bounds = (2, 8)

    mutators = {"bounded": SimpleNamespace(declare_variable_topology=declare)}
    if boundary == "mixed":
        mutators["substitution"] = object()
    stages = FlowStub(all_oligos=("AL",), runtime_context=SimpleNamespace(mutators=mutators))
    args = SimpleNamespace(steps=3, seed=17, search={"resume": boundary == "recovery"})
    config = {"mutator_stack_pre_update": boundary == "preupdate"}
    kwargs = dict(initial_evaluation_result=None, initial_loader=_unused,
                  initialize=_unused, advance=_unused, data=s.data)
    if boundary is not None:
        with pytest.raises(ValueError, match="does not support"):
            NativeFlow(args, FlowStub(), stages, config, **kwargs)
    else:
        flow = NativeFlow(args, FlowStub(), stages, config, **kwargs)
        assert flow.system.domains[0].length_bounds == (2, 8)
        assert flow.system == capture(s).system
        flow.close()
    assert calls == ["declare"]


def test_mutation_context_rejects_reconstruction_before_invoking_owner():
    state()
    operation = SimpleNamespace(mutation_context=object())
    with pytest.raises(UnsupportedNativeMutation, match="substitution-only"):
        NativeMutationOperation._validate_context_representability(operation)


def test_generated_artifact_resolves_source_and_target_lengths_with_fixed_ligand(tmp_path):
    from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure
    from Bio.SeqUtils import seq3

    s = state()
    proposed = edit(s, "ACDEFG")
    structure = Structure.Structure("generated")
    model = Model.Model(0)
    structure.add(model)
    protein, ligand = Chain.Chain("A"), Chain.Chain("B")
    model.add(protein)
    model.add(ligand)
    for i, token in enumerate("ACDEFG", 1):
        residue = Residue.Residue((" ", i, " "), seq3(token).upper(), " ")
        protein.add(residue)
        residue.add(Atom.Atom("CA", np.array([i * 3.8, 0., 0.]), 0., 1., " ", " CA ", i, element="C"))
    residue = Residue.Residue(("H_CFF", 1, " "), "CFF", " ")
    ligand.add(residue)
    residue.add(Atom.Atom("C1", np.array([0., 3., 0.]), 0., 1., " ", " C1 ", 7, element="C"))
    path, annotated, metadata = (tmp_path / name for name in ("generated.cif", "input.cif", "metadata.json"))
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(path))
    annotated.write_text("source bytes")
    metadata.write_text("{}")
    chains = (ChainSpec("A", "A", "ACDE"), ChainSpec("B", "L", "CFF", kind="ligand"))
    generated = GeneratedStructure(str(path), str(annotated), str(metadata),
        (("A", "ACDEFG"), ("B", "CFF")), "request", hashlib.sha256(path.read_bytes()).hexdigest(), False)
    result = GenerationResult("AL", chains, generated)
    sample = Sample(key_for("test"), "replicate", 17, "1")
    artifact = bind_generated_structure(result, data=s.data, system=s.translation.system,
        base=s.translation.candidate, proposed=proposed, sample=sample, inputs=manifest(()))
    proposal = Proposal(key_for("proposal"), (s.translation.candidate.identity,),
        s.translation.candidate.identity, proposed, (), sample.protocol, (artifact,))
    record, = read_generated_structures(proposal, s.data)
    assert record.sequences == generated.sequences
    assert record.chains[0].sequence == "ACDE"
    assert record.proposed_candidate == proposed.identity
    bad_source = replace(result, chains=(replace(chains[0], sequence="ACD"), chains[1]))
    with pytest.raises(ValueError, match="differs from candidate"):
        bind_generated_structure(bad_source, data=s.data, system=s.translation.system,
            base=s.translation.candidate, proposed=proposed, sample=sample, inputs=manifest(()))
    bad_ligand = replace(result, generated=replace(generated, sequences=(("A", "ACDEFG"), ("B", "ATP"))))
    with pytest.raises(ValueError, match="fixed CCD"):
        bind_generated_structure(bad_ligand, data=s.data, system=s.translation.system,
            base=s.translation.candidate, proposed=proposed, sample=sample, inputs=manifest(()))
    old_graph = s.translation.candidate.assignments[1].value.graph
    payload = s.data.get(old_graph, JSONCodec(old_graph.codec))
    new_graph = s.data.put(dict(payload, value="ATP"), JSONCodec(old_graph.codec))
    with pytest.raises(ValueError, match="chemical graph"):
        s.translation.system.edit(proposed, (Assignment("L_1", Chemical(new_graph)),), data=s.data)


def test_rfd4_generated_evaluator_grow_accept_shrink_reject_next_accepted_input(tmp_path, monkeypatch):
    """Exercise real preparation, pending sequences, publication and evaluator input."""
    from dataclasses import asdict
    import json
    from pathlib import Path

    from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure
    from Bio.SeqUtils import seq3

    from astrolabe.design.generated_structure import publish_generated_structure
    from astrolabe.evaluators.generated_structure import GeneratedStructureEvaluator
    from astrolabe.mutators import rfd4_backend, rfd4_feedback
    from astrolabe.mutators.rfd4 import RFD4Mutator, structure_sequences
    from astrolabe.mutators.rfd4_reconstruction import read_motif_lineage
    from astrolabe.utils.base import construct_standard_input

    def write_structure(path, sequence, shift=0.):
        structure = Structure.Structure("cpu-fixture")
        model = Model.Model(0)
        structure.add(model)
        for chain_id in ("A", "B"):
            chain = Chain.Chain(chain_id)
            model.add(chain)
            residues = list(enumerate(sequence, 1)) if chain_id == "A" else [(1, "CFF")]
            for position, token in residues:
                ligand = chain_id == "B"
                residue = Residue.Residue(("H_CFF" if ligand else " ", position, " "),
                    "CFF" if ligand else seq3(token).upper(), " ")
                chain.add(residue)
                name = "C1" if ligand else "CA"
                residue.add(Atom.Atom(name, np.array([position * 3.8, 6. if ligand else 0., shift]),
                    0., 1., " ", f" {name:3}", position, element="C"))
        writer = MMCIFIO()
        writer.set_structure(structure)
        writer.save(str(path))
        return path

    class FakeBackend:
        def __init__(self, **kwargs):
            self.generation = kwargs["generation"]
            self.inputs = []
            self.outputs = []

        def fingerprint(self):
            return {"fixture": "cpu-variable-length-v1"}

        def generate(self, request):
            source = Path(request.structure_path)
            self.inputs.append((source.read_bytes(), request.chains,
                                read_motif_lineage(source)))
            sequence = ("ACDEFG", "AC", "GHIKL")[len(self.inputs) - 1]
            directory = tmp_path / f"backend-round-{len(self.inputs)}"
            directory.mkdir()
            output = write_structure(directory / "generated.cif", sequence, float(len(self.inputs)))
            annotated = directory / "annotated.cif"
            annotated.write_bytes(source.read_bytes())
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            metadata = directory / "manifest.json"
            metadata.write_text(json.dumps({
                "request": asdict(request), "checksums": {"generated.cif": digest},
                "validation": {"motif_mapping": []},
            }))
            generated = GeneratedStructure(str(output), str(annotated), str(metadata),
                (("A", sequence), ("B", "CFF")), key_for(asdict(request)).digest, digest, False)
            self.outputs.append(generated)
            return generated

        def shutdown(self):
            pass

    feedback_inputs = []

    def analyze_interface(path, binder_chains, target_chains, **kwargs):
        feedback_inputs.append(Path(path).read_bytes())
        return SimpleNamespace(binder_chains=tuple(binder_chains), residues=(),
                               interfacial_hbonds=(), polar_atoms=())

    monkeypatch.setattr(rfd4_backend, "RFD4Backend", FakeBackend)
    monkeypatch.setattr(rfd4_feedback, "analyze_interface", analyze_interface)
    s = state(bounds=None)
    mutator = RFD4Mutator(assembly="AL", binder_chains=["A"], target_chains=["B"],
        python_bin="unused", source_dir="unused", model_dir="unused", cache_dir="unused",
        generation={"mode": "inpaint"}, position_selector="None",
        feedback={"enabled": True, "reconstruction": {"length_min": 2, "length_max": 8}})
    mutator.declare_variable_topology()
    initial = capture(s)
    source = write_structure(tmp_path / "initial.cif", "ACDE")
    evaluator = GeneratedStructureEvaluator(initial_structures={"AL": source})
    accepted_scores = evaluator.evaluate(construct_standard_input(["AL"], 17), {"step": 0})
    accepted_bytes, accepted_sequence = source.read_bytes(), "ACDE"
    round_one_bytes = None
    published = []

    for step, (sequence, accept) in enumerate((("ACDEFG", True), ("AC", False), ("GHIKL", True)), 1):
        previous = capture(s)
        runtime = {"step": step, "seed": 17}
        if step == 2:
            prepared = mutator.prepare_native_mutation(global_config.mutable_protomers, accepted_scores, runtime)
            result = mutator.mutate_prepared(prepared, global_config.mutable_protomers, accepted_scores, runtime)
        else:
            result = mutator.mutate(global_config.mutable_protomers, accepted_scores, runtime)
        assert isinstance(result, GenerationResult)
        assert result.chains[0].sequence == accepted_sequence
        assert s.segment.pending_sequence == sequence
        assert s.segment.mutations in (None, [])
        assert s.protein.mutated_sequence == sequence
        assert_layout(s, accepted_sequence)
        assert capture(s).candidate == previous.candidate
        source_bytes, chains, lineage = mutator.backend.inputs[-1]
        assert source_bytes == accepted_bytes
        assert chains[0].sequence == accepted_sequence
        assert chains[1].sequence == "CFF" and lineage == {}
        assert feedback_inputs[-1] == accepted_bytes
        if step == 3:
            assert source_bytes == round_one_bytes
            assert source_bytes != Path(mutator.backend.outputs[1].structure_path).read_bytes()
            assert chains[0].sequence == "ACDEFG"

        attempt = f"{step}:rfd4"
        index = publish_generated_structure(result, tmp_path / "published", attempt=attempt)
        published.append(index)
        generated_path = index.parent / "rfd4.generated.cif"
        assert read_motif_lineage(generated_path) == {}
        assert structure_sequences(generated_path, result.chains) == {"A": sequence, "B": "CFF"}
        runtime["generated_structure_records"] = {"AL": {
            "step": step, "attempt": attempt, "index_path": str(index),
            "request_digest": result.generated.request_digest,
            "structure_digest": result.generated.structure_digest,
        }}
        proposed_scores = evaluator.evaluate(construct_standard_input(["AL"], 17), runtime)
        assert proposed_scores.complex_paths["AL"].read_bytes() == generated_path.read_bytes()
        if accept:
            s.manager.apply_mutations()
            accepted_scores = proposed_scores
            accepted_sequence, accepted_bytes = sequence, generated_path.read_bytes()
            if step == 1:
                round_one_bytes = accepted_bytes
        else:
            s.manager.clear_mutations()
            assert capture(s).candidate == previous.candidate
            assert accepted_scores.complex_paths["AL"].read_bytes() == round_one_bytes
            assert accepted_sequence == "ACDEFG"
        assert s.segment.pending_sequence is None
        assert_layout(s, accepted_sequence)
        assert capture(s).system == initial.system
    assert len(mutator.backend.inputs) == len(published) == 3
    assert_layout(s, "GHIKL")
    assert source.read_bytes() == mutator.backend.inputs[0][0]
    mutator.shutdown()
