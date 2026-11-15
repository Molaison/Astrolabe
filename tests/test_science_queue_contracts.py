import copy
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators import proton_pottsmpnn as proton
from astrolabe.mutators import vdm
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.rfd4 import RFD4Mutator
from astrolabe.mutators.vdm_proton import VdmProtonMutator


class SelectedRunner:
    def __init__(self, sequence):
        self.sequence = sequence
        self.calls = []

    def design_sequences(self, **features):
        self.calls.append(copy.deepcopy(features))
        tokens = {**proton.STANDARD_AA_TO_TOKEN, "H": "HIS-S"}
        return {"selected": {"canonical_sequence": self.sequence,
                             "extended_tokens": [tokens[a] for a in self.sequence]}}

    def close(self):
        pass


class QueueSource(Mutator):
    name = "injected_queue_source"

    def __init__(self, position=0, target="V", entropy=0.0):
        self.position, self.target, self.entropy = position, target, entropy
        self.calls = []

    def mutate(self, protomers, scores, runtime):
        proto = protomers["A"]
        self.calls.append(proto.sequence)
        global_config.segment_manager.add_mutations(
            [(proto.components[0], (self.position, proto.sequence[self.position],
                                    self.target, self.entropy, 0.0))], 1)


@pytest.fixture
def science(monkeypatch, tmp_path):
    global_config.reset()
    for name, value in (("segments", {}), ("protomers", {}), ("mutable_protomers", {}),
                        ("named_files", {}), ("segment_manager", SegmentManager()),
                        ("exclude_AA", [])):
        global_config.init_const(name, value)
    global_config.set_variable("mutation_num", 2)
    proto = Protomer("A", "AAAA", "p")
    global_config.protomers["A"] = proto
    global_config.mutable_protomers["A"] = proto
    calls = []

    def selector(*args, **kwargs):
        calls.append(args[1].names)
        return {}

    monkeypatch.setattr(proton, "prepare_exclude_sites", selector)
    monkeypatch.setattr(vdm, "prepare_exclude_sites", selector)
    scores = StandardScores(names=["A"], plddt={}, complex_paths={"A": tmp_path / "source.pdb"})
    yield SimpleNamespace(proto=proto, manager=global_config.segment_manager,
                          scores=scores, selectors=calls, root=tmp_path)
    global_config.reset()


def plain(sequence="AALA", **kwargs):
    runner = SelectedRunner(sequence)
    mutator = proton.ProtonPottsMPNNMutator(
        sequence_model_runner=runner,
        criteria={"method": "greedy_energy_block", "infill_scope": "chain", "center_count": 0},
        **kwargs)
    return mutator, runner


@pytest.mark.parametrize("runtime,names", [
    ({"ignored_complexes": ["A"]}, ["A"]),
    ({"negative_design_complexes": ["A"]}, ["A"]),
    ({"ignored_complexes": ["A"], "negative_design_complexes": ["AA"]}, ["A", "AA"]),
])
def test_all_contexts_excluded_never_selects_or_models(science, runtime, names):
    science.scores.names = names
    mutator, runner = plain()
    assert mutator._active_scores(science.scores, runtime).names == []
    with pytest.raises(ValueError, match="exactly one active complex"):
        mutator.mutate(global_config.protomers, science.scores, runtime)
    assert science.selectors == []
    assert runner.calls == []


@pytest.mark.parametrize("field", ["ignored_complexes", "negative_design_complexes"])
def test_partial_exclusions_keep_the_real_active_context(science, field):
    science.scores.names = ["AA", "A"]
    mutator, runner = plain()
    mutator.mutate(global_config.protomers, science.scores, {field: ["AA"]})
    assert science.selectors == [["A"]]
    assert len(runner.calls) == 1
    assert science.proto.mutated_sequence == "AALA"


@pytest.mark.parametrize("order", ["first", "last"])
@pytest.mark.parametrize("network_only", [False, True])
def test_plain_stacks_reject_both_orders_before_first_model(science, order, network_only, monkeypatch):
    from astrolabe.mutators import proton_network_geometry
    monkeypatch.setattr(proton_network_geometry, "build_network_pair_preferences", lambda *a, **k: ([], {}))
    mutator, runner = plain("AAAA", network_only=network_only)
    source = QueueSource()
    stack = [mutator, source] if order == "first" else [source, mutator]
    with pytest.raises(ValueError, match="pre.update"):
        run.execute_mutation_step(stack, science.scores, {}, 1)
    assert source.calls == []
    assert runner.calls == []
    assert science.selectors == []


@pytest.mark.parametrize("dirty", ["nonoverlap", "alias", "pending_sequence"])
def test_direct_plain_rejects_dirty_physical_input_before_selector(science, dirty):
    segment = global_config.segments[science.proto.components[0]]
    if dirty == "pending_sequence":
        segment.length_bounds = (4, 4)
        segment.stage_sequence("VAAA")
    else:
        name = science.proto.components[0]
        if dirty == "alias":
            global_config.segments["alias"] = segment
            science.manager.add_segment("alias", segment)
            name = "alias"
        science.manager.add_mutations([(name, (0, "A", "V", 0.0, 0.0))], 1)
    before = copy.deepcopy((segment.mutations, segment.pending_sequence))
    mutator, runner = plain(reference_sequence="AAAA", max_total_mutations=1)
    with pytest.raises(ValueError, match="queue|pending"):
        mutator.mutate(global_config.protomers, science.scores, {})
    assert science.selectors == []
    assert runner.calls == []
    assert (segment.mutations, segment.pending_sequence) == before


@pytest.mark.parametrize("change", ["sequence", "queue", "pending_sequence", "mutation_budget"])
@pytest.mark.parametrize("boundary", ["model", "output"])
def test_plain_detects_input_change_after_preparation(science, change, boundary):
    mutator, runner = plain()
    features = mutator.prepare_mutation_input(global_config.protomers, science.scores, {})
    output = None if boundary == "model" else mutator.compute_mutation(features)
    segment = global_config.segments[science.proto.components[0]]
    if change == "sequence":
        segment.sequence = "VAAA"
    elif change == "queue":
        segment.add_mutations((0, "A", "V", 0.0, 0.0))
    elif change == "pending_sequence":
        segment.length_bounds = (4, 4)
        segment.stage_sequence("VAAA")
    else:
        global_config.set_variable("mutation_num", 3)
    before = copy.deepcopy((segment.sequence, segment.mutations, segment.pending_sequence))
    with pytest.raises(ValueError, match="changed|queue|pending|budget"):
        if boundary == "model":
            mutator.compute_mutation(features)
        else:
            mutator.prepare_mutation_output(output, global_config.protomers, science.scores, {})
    assert len(runner.calls) == (0 if boundary == "model" else 1)
    assert (segment.sequence, segment.mutations, segment.pending_sequence) == before


@pytest.mark.parametrize("failure", ["raise", "truncate", "wrong_site"])
def test_plain_enqueue_is_atomic_and_delivers_exact_genotype(science, failure, monkeypatch):
    mutator, runner = plain("VALA")
    original = science.manager.add_mutations

    def broken(records, count):
        original(records[:1], 1)
        if failure == "raise":
            raise OSError("partial write")
        if failure == "wrong_site":
            name = science.proto.components[0]
            original([(name, (1, "A", "L", 0.0, 0.0))], 1)

    monkeypatch.setattr(science.manager, "add_mutations", broken)
    segment = global_config.segments[science.proto.components[0]]
    before = copy.deepcopy(segment.mutations)
    with pytest.raises((ValueError, OSError)):
        mutator.mutate(global_config.protomers, science.scores, {})
    assert len(runner.calls) == 1
    assert segment.mutations == before
    assert science.proto.mutated_sequence == "AAAA"
    assert mutator.last_mutation_telemetry["oligos"] == {}


def test_plain_positive_preserves_shared_segment_and_rng(science):
    segment = global_config.segments[science.proto.components[0]]
    global_config.segments["alias"] = segment
    science.manager.add_segment("alias", segment)
    mutator, runner = plain(reference_sequence="AAAA", max_total_mutations=1)
    rng = random.getstate()
    mutator.mutate(global_config.protomers, science.scores, {})
    assert random.getstate() == rng
    assert len(runner.calls) == 1
    assert science.proto.sequence == "AAAA"
    assert science.proto.mutated_sequence == "AALA"
    assert global_config.segments["alias"] is segment
    assert len(segment.mutations) == 1


@pytest.mark.parametrize("order", ["first", "last"])
@pytest.mark.parametrize("accepted", [False, True])
def test_valid_preupdate_preserves_order_and_accept_reject_boundary(science, order, accepted):
    source = QueueSource()
    mutator, runner = plain("AALA" if order == "first" else "VALA",
                            reference_sequence="AAAA", max_total_mutations=2)
    stack = [mutator, source] if order == "first" else [source, mutator]
    runtime = {"mutator_stack_pre_update": True}
    segment = global_config.segments[science.proto.components[0]]
    run.execute_mutation_step(stack, science.scores, runtime, 1)
    assert source.calls == (["AALA"] if order == "first" else ["AAAA"])
    assert runner.calls[0]["binder_sequence"] == ("AAAA" if order == "first" else "VAAA")
    assert science.proto.mutated_sequence == "VALA"
    if accepted:
        science.manager.apply_mutations()
    run._finalize_mutator_stack_pre_update(runtime, accepted)
    assert science.proto.sequence == ("VALA" if accepted else "AAAA")
    assert segment is global_config.segments[science.proto.components[0]]
    assert not segment.mutations


@pytest.mark.parametrize("order", ["first", "last"])
def test_entropy_leftover_stops_before_next_model(science, order):
    source = QueueSource(entropy=2.0)
    mutator, runner = plain()
    stack = [mutator, source] if order == "first" else [source, mutator]
    runtime = {"mutator_stack_pre_update": True,
               "mutator_stack_pre_update_max_entropy": -1 if order == "first" else 0.5}
    try:
        with pytest.raises(ValueError, match="queue|pending"):
            run.execute_mutation_step(stack, science.scores, runtime, 1)
        assert len(runner.calls) == (1 if order == "first" else 0)
        assert len(source.calls) == (0 if order == "first" else 1)
    finally:
        run._finalize_mutator_stack_pre_update(runtime, False)
    assert science.proto.sequence == "AAAA"


@pytest.mark.parametrize("downstream", [1, 2])
def test_reference_cap_survives_downstream_staged_edits(science, downstream):
    mutator, runner = plain(reference_sequence="AAAA", max_total_mutations=1)
    source = QueueSource()
    tail = QueueSource(3, "G")
    runtime = {"mutator_stack_pre_update": True}
    try:
        with pytest.raises(ValueError, match="original.parent|reference|max_total"):
            run.execute_mutation_step([mutator, source, tail][:downstream + 1], science.scores, runtime, 1)
        assert len(runner.calls) == 1
        assert len(source.calls) == 1
        assert tail.calls == []
    finally:
        run._finalize_mutator_stack_pre_update(runtime, False)
    assert science.proto.sequence == "AAAA"


@pytest.mark.parametrize("order", ["first", "last"])
@pytest.mark.parametrize("preupdate", [False, True])
def test_real_rfd4_remains_exclusive_before_any_model(science, order, preupdate, monkeypatch):
    from astrolabe.mutators import rfd4_backend
    calls = []
    monkeypatch.setattr(rfd4_backend, "RFD4Backend",
                        lambda **kwargs: SimpleNamespace(generate=lambda request: calls.append(request)))
    rfd = RFD4Mutator(assembly="AB", binder_chains=["A"], target_chains=["B"],
                      python_bin="unused", source_dir="unused", model_dir="unused", cache_dir="unused")
    source = QueueSource()
    stack = [rfd, source] if order == "first" else [source, rfd]
    with pytest.raises(ValueError, match="cannot run in a mutation stack"):
        run.execute_mutation_step(stack, science.scores, {"mutator_stack_pre_update": preupdate}, 1)
    assert source.calls == []
    assert calls == []


def structure(path, sequences):
    from Bio.PDB import Atom, Chain, Model, PDBIO, Residue, Structure
    from Bio.SeqUtils import seq3

    model = Model.Model(0)
    root = Structure.Structure("source")
    root.add(model)
    for chain_index, (chain_id, sequence) in enumerate(sequences.items()):
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for position, aa in enumerate(sequence, 1):
            residue = Residue.Residue((" ", position, " "), seq3(aa).upper(), " ")
            chain.add(residue)
            for atom_index, atom in enumerate(("N", "CA", "C", "O")):
                residue.add(Atom.Atom(atom, np.array([position * 4.0, chain_index * 8.0, atom_index * 0.5]),
                                     0.0, 1.0, " ", f"{atom:>4}", position * 4 + atom_index,
                                     element=atom[0]))
    writer = PDBIO()
    writer.set_structure(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer.save(str(path))
    return path


def vdm_case(science, monkeypatch, unchanged=False):
    global_config.segments.clear()
    global_config.update_const("segment_manager", SegmentManager())
    science.manager = global_config.segment_manager
    binder = Protomer("A", "A +K AA" if unchanged else "AAAA", "p")
    target = Protomer("B", "+GG", "p")
    global_config.protomers.update(A=binder, B=target)
    global_config.mutable_protomers.update(A=binder, B=target)
    science.proto = binder
    global_config.set_variable("mutation_num", 4)
    path = structure(science.root / "p10_shard_134_rep33_1" / "parent.pdb",
                     {"A": binder.sequence, "B": target.sequence})
    science.scores = StandardScores(names=["AB"], plddt={}, complex_paths={"AB": path})
    producer = vdm.VdmPairMutator(library_path=science.root / "injected-library",
                                  site_1={"position": 1, "residue_type": "H"},
                                  site_2={"position": 2, "residue_type": "K",
                                          "sequence": "fixed" if unchanged else "mutable"})
    producer.preview_dir = science.root / "previews"
    producer.mutation_weight = 0.25 if unchanged else 0.5

    def proposal(**kwargs):
        current = kwargs["binder_sequence"]
        sites = []
        for index, (aa, spec) in enumerate(zip("HK", kwargs["sites"])):
            policy = spec["sequence"]
            sites.append(dict(site_index=index, chain="A", position_zero_based=index,
                              position_one_based=index + 1, residue_id=[" ", index + 1, " "],
                              original_residue_type=current[index], residue_type=aa,
                              library_role="his" if index == 0 else "partner", sequence=policy,
                              sidechain_policy=spec["sidechain"]))
        edits = [dict(position_zero_based=p, position_one_based=p + 1, **{"from": a, "to": b})
                 for p, (a, b) in enumerate(zip(current, "HKAA")) if a != b]
        metadata = dict(structure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        binder_chain="A", original_binder_sequence=current,
                        effective_mutable_positions=kwargs["mutable_positions"],
                        target_sequences_before={"B": "GG"}, target_sequences_after={"B": "GG"},
                        sites=sites, edits=edits, library_roles=["his", "partner"],
                        source={"his": {"resname": "HIS"}, "partner": {"resname": "LYS"}},
                        placed_sidechains=[dict(site, coordinates={}) for site in sites])
        return dict(status="proposed", designed_sequence="HKAA", metadata=metadata)

    monkeypatch.setattr(vdm, "_load_proposer", lambda: proposal)
    runner = SelectedRunner("HKLA")
    consumer = VdmProtonMutator(sequence_model_runner=runner, microenvironment_mutations=1)
    consumer.mutation_weight = 0.75 if unchanged else 0.5
    return producer, consumer, runner


@pytest.mark.parametrize("unchanged", [False, True])
def test_real_vdm_producer_preview_roles_and_consumer_queue(science, monkeypatch, unchanged):
    producer, consumer, runner = vdm_case(science, monkeypatch, unchanged)
    runtime = {}
    run.execute_mutation_step([producer, consumer], science.scores, runtime, 1)
    assert len(runner.calls) == 1
    features = runner.calls[0]
    assert features["binder_sequence"] == "HKAA"
    assert features["chain_sequences"] == {"A": "HKAA", "B": "GG"}
    assert features["criteria"]["explicit_centers"] == [{"res_id": 1, "protonation_type": "HIS-S"}]
    assert 1 in features["fixed_positions"]
    assert runtime["network_pair_roles"] == {"A": {0: "H", 1: "K"}}
    assert science.proto.sequence == ("AKAA" if unchanged else "AAAA")
    assert science.proto.mutated_sequence == "HKLA"
    assert global_config.protomers["B"].mutated_sequence == "GG"
    assert sum(len(s.mutations or []) for s in science.manager.unique_segments.values()) == (2 if unchanged else 3)
    audit = consumer.last_mutation_telemetry["oligos"]["AB"]["vdm_proton"]
    assert audit["pair_sequence"] == "HKAA"
    assert audit["microenvironment_edits"] == [(2, "A", "L")]
    assert audit["native_output"]["selected"]["extended_tokens"][0] == "HIS-S"


@pytest.mark.parametrize("cap", [2, 3])
def test_vdm_producer_reference_cap_survives_consumer_delivery(science, monkeypatch, cap):
    producer, consumer, runner = vdm_case(science, monkeypatch)
    producer.reference_sequence = "AAAA"
    producer.max_total_mutations = cap
    runtime = {}
    if cap == 2:
        with pytest.raises(ValueError, match="original-parent reference mutation budget"):
            run.execute_mutation_step([producer, consumer], science.scores, runtime, 1)
    else:
        run.execute_mutation_step([producer, consumer], science.scores, runtime, 1)
    assert len(runner.calls) == 1
    assert runner.calls[0]["binder_sequence"] == "HKAA"
    proposal = producer.last_mutation_telemetry["oligos"]["AB"]
    assert proposal["designed_sequence"] == "HKAA"
    assert proposal["total_mutations"] == 2
    assert science.proto.sequence == "AAAA"
    assert science.proto.mutated_sequence == "HKLA"
    assert global_config.protomers["B"].mutated_sequence == "GG"
    assert runtime["network_pair_roles"] == {"A": {0: "H", 1: "K"}}
    assert sum(len(s.mutations or []) for s in science.manager.unique_segments.values()) == 3
    audit = consumer.last_mutation_telemetry["oligos"]["AB"]["vdm_proton"]
    assert audit["pair_sequence"] == "HKAA"
    assert audit["after_sequence"] == "HKLA"
    assert audit["microenvironment_edits"] == [(2, "A", "L")]
    assert audit["native_output"]["selected"]["extended_tokens"][0] == "HIS-S"


@pytest.mark.parametrize("corruption", ["partial", "unrelated", "pending_sequence"])
def test_vdm_real_handoff_rejects_dirty_queue_before_model(science, monkeypatch, corruption):
    producer, consumer, runner = vdm_case(science, monkeypatch)
    runtime = {}
    producer.mutate(global_config.mutable_protomers, science.scores, runtime)
    segment = global_config.segments[science.proto.components[0]]
    if corruption == "partial":
        segment.mutations.pop()
    elif corruption == "unrelated":
        segment.add_mutations((3, "A", "V", 0.0, 0.0))
    else:
        segment.clear_mutations()
        segment.length_bounds = (4, 4)
        segment.stage_sequence("HKAA")
    before = copy.deepcopy((segment.mutations, segment.pending_sequence))
    with pytest.raises(ValueError, match="pair|queue|pending"):
        consumer.mutate(global_config.mutable_protomers, science.scores, runtime)
    assert runner.calls == []
    assert (segment.mutations, segment.pending_sequence) == before


@pytest.mark.parametrize("violation", ["his_token", "unchanged_fixed_partner"])
def test_vdm_consumer_keeps_microstate_and_unchanged_fixed_role(science, monkeypatch, violation):
    producer, consumer, runner = vdm_case(science, monkeypatch, unchanged=True)
    runtime = {}
    producer.mutate(global_config.mutable_protomers, science.scores, runtime)
    features = consumer.prepare_mutation_input(global_config.mutable_protomers, science.scores, runtime)
    output = consumer.compute_mutation(features)
    if violation == "his_token":
        output["AB"]["selected"]["extended_tokens"][0] = "HIS-P"
    else:
        output["AB"]["selected"]["canonical_sequence"] = "HRLA"
        output["AB"]["selected"]["extended_tokens"][1] = "ARG"
    before = {id(s): copy.deepcopy(s.mutations) for s in science.manager.unique_segments.values()}
    with pytest.raises(ValueError):
        consumer.prepare_mutation_output(output, global_config.mutable_protomers, science.scores, runtime)
    assert {id(s): s.mutations for s in science.manager.unique_segments.values()} == before
    assert science.proto.mutated_sequence == "HKAA"


def cached_pairset(science, producer, runtime, features):
    def record(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def save(name, value):
        path = science.root / name
        path.write_text(json.dumps(value))
        return path

    preview = Path(runtime["vdm_pair_preview"]["structure_path"])
    source = Path(features["AB"]["structure_path"])
    original = dict(mutation_input=features, proposal=producer.last_mutation_telemetry["oligos"]["AB"],
                    selected_mutation_budget=2, global_mutation_budget=4,
                    input_structure=str(source), input_sha256=record(source)["sha256"],
                    template_preview=str(preview), template_preview_sha256=record(preview)["sha256"])
    handoff = dict(schema="vdm_pairset_handoff", status="READY", source_snapshot={},
                   p10_proposal=record(save("proposal.json", original)), p10_joint_preview=record(preview),
                   p10_roles=record(save("roles.json", runtime["vdm_pair_roles"])),
                   original_p10_overlay=record(save("overlay.json", {})),
                   original_p10_effective_config=record(save("config.json", {})))
    path = save("handoff.json", handoff)
    return path, record(path)["sha256"]


@pytest.mark.parametrize("queued", [False, True])
def test_cached_native_vdm_packet_delivers_full_pair_and_microenvironment(science, monkeypatch, queued):
    producer, _, _ = vdm_case(science, monkeypatch)
    runtime = {}
    features = producer.prepare_mutation_input(global_config.mutable_protomers, science.scores, runtime)
    output = producer.compute_mutation(features)
    producer.prepare_mutation_output(output, global_config.mutable_protomers, science.scores, runtime)
    path, digest = cached_pairset(science, producer, runtime, features)
    if not queued:
        science.manager.clear_mutations()
    runner = SelectedRunner("HKLA")
    consumer = VdmProtonMutator(sequence_model_runner=runner, microenvironment_mutations=1,
                                pairset_handoff=path, pairset_handoff_sha256=digest)
    consumer.mutate(global_config.mutable_protomers, science.scores, {})
    assert len(runner.calls) == 1
    assert runner.calls[0]["vdm_proton"]["mode"] == "cached_pairset_handoff"
    assert science.proto.sequence == "AAAA"
    assert science.proto.mutated_sequence == "HKLA"
    assert sum(len(s.mutations or []) for s in science.manager.unique_segments.values()) == 3


@pytest.mark.parametrize("accepted", [False, True])
def test_preupdate_plain_in_middle_keeps_full_working_input(science, accepted):
    global_config.set_variable("mutation_num", 3)
    head, tail = QueueSource(), QueueSource(3, "G")
    mutator, runner = plain("VALA", reference_sequence="AAAA", max_total_mutations=3)
    runtime = {"mutator_stack_pre_update": True}
    run.execute_mutation_step([head, mutator, tail], science.scores, runtime, 1)
    assert runner.calls[0]["binder_sequence"] == "VAAA"
    assert tail.calls == ["VALA"]
    assert science.proto.mutated_sequence == "VALG"
    if accepted:
        science.manager.apply_mutations()
    run._finalize_mutator_stack_pre_update(runtime, accepted)
    assert science.proto.sequence == ("VALG" if accepted else "AAAA")


@pytest.mark.parametrize("boundary", ["model", "output"])
def test_vdm_detects_producer_queue_change_after_preparation(science, monkeypatch, boundary):
    producer, consumer, runner = vdm_case(science, monkeypatch)
    runtime = {}
    producer.mutate(global_config.mutable_protomers, science.scores, runtime)
    features = consumer.prepare_mutation_input(global_config.mutable_protomers, science.scores, runtime)
    output = None if boundary == "model" else consumer.compute_mutation(features)
    global_config.segments[science.proto.components[0]].mutations.pop()
    with pytest.raises(ValueError, match="pair|queue|changed"):
        if boundary == "model":
            consumer.compute_mutation(features)
        else:
            consumer.prepare_mutation_output(output, global_config.mutable_protomers, science.scores, runtime)
    assert len(runner.calls) == (0 if boundary == "model" else 1)


@pytest.mark.parametrize("failure", ["raise", "truncate"])
def test_vdm_consumer_atomic_rollback_retains_complete_producer_queue(science, monkeypatch, failure):
    producer, consumer, runner = vdm_case(science, monkeypatch)
    consumer.microenvironment_mutations = 2
    runner.sequence = "HKLV"
    runtime = {}
    producer.mutate(global_config.mutable_protomers, science.scores, runtime)
    before = {id(s): copy.deepcopy(s.mutations) for s in science.manager.unique_segments.values()}
    original = science.manager.add_mutations

    def broken(records, count):
        original(records[:1], 1)
        if failure == "raise":
            raise OSError("partial consumer write")

    monkeypatch.setattr(science.manager, "add_mutations", broken)
    with pytest.raises((ValueError, OSError)):
        consumer.mutate(global_config.mutable_protomers, science.scores, runtime)
    assert {id(s): s.mutations for s in science.manager.unique_segments.values()} == before
    assert science.proto.mutated_sequence == "HKAA"
    assert "network_pair_roles" not in runtime


def cached_legacy(science, producer, runtime, features):
    def record(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save(name, value):
        path = science.root / name
        path.write_text(json.dumps(value))
        return path

    preview = Path(runtime["vdm_pair_preview"]["structure_path"])
    source = Path(features["AB"]["structure_path"])
    original = save("legacy-original.json", dict(mutation_input=features,
                    proposal=producer.last_mutation_telemetry["oligos"]["AB"]))
    common = dict(parent_id="parent", structure_path=str(source), structure_sha256=record(source))
    parent = dict(common, role="vdm_parent", sequence="AAAA", source_structure=str(source),
                  source_structure_sha256=record(source))
    candidate = dict(common, item_id="candidate", role="vdm_candidate", sequence="HKAA",
                     source_structure=str(preview), source_structure_sha256=record(preview))
    manifest = save("legacy-manifest.json", {"items": [parent, candidate]})
    packet = dict(candidate_id="candidate", route="vdm_pair", parent_structure_sha256=record(source),
                  binder_sequence="HKAA", template_preview_path=str(preview),
                  template_preview_sha256=record(preview), pair_positions_1based=[1, 2],
                  original_proposal_path=str(original), original_proposal_sha256=record(original))
    packets = save("legacy-packets.json", {"packets": [packet]})
    summary, raw = save("hbond-summary.json", {}), save("hbond-raw.json", {})
    handoff = save("legacy-handoff.json", dict(scoring_input_draft=str(manifest),
                    scoring_input_sha256=record(manifest), proposal_packet=str(packets),
                    proposal_packet_sha256=record(packets), hbond_summary=str(summary),
                    hbond_summary_sha256=record(summary), raw_hbond_result=str(raw),
                    raw_hbond_result_sha256=record(raw),
                    candidates=[dict(candidate_id="candidate", pair_positions_1based=[1, 2])]))
    return manifest, handoff


@pytest.mark.parametrize("queued", [False, True])
def test_legacy_cached_packet_retains_full_native_queue(science, monkeypatch, queued):
    producer, _, _ = vdm_case(science, monkeypatch)
    runtime = {}
    features = producer.prepare_mutation_input(global_config.mutable_protomers, science.scores, runtime)
    output = producer.compute_mutation(features)
    producer.prepare_mutation_output(output, global_config.mutable_protomers, science.scores, runtime)
    manifest, handoff = cached_legacy(science, producer, runtime, features)
    if not queued:
        science.manager.clear_mutations()
    runner = SelectedRunner("HKLA")
    consumer = VdmProtonMutator(sequence_model_runner=runner, microenvironment_mutations=1,
                                proposal_manifest=manifest, proposal_id="candidate", hbond_handoff=handoff)
    consumer.mutate(global_config.mutable_protomers, science.scores, {})
    assert len(runner.calls) == 1
    assert runner.calls[0]["vdm_proton"]["mode"] == "cached_packet"
    assert science.proto.mutated_sequence == "HKLA"
    assert sum(len(s.mutations or []) for s in science.manager.unique_segments.values()) == 3
