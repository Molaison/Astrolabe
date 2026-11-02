"""Independent CPU tests using the fork's actual scientific classes and codecs."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from astrolabe.core import global_config
from astrolabe.core.folding_input import Template
from astrolabe.core.protomer_config_parser import Protomer, Segment, SegmentManager
from astrolabe.design import legacy
from astrolabe.design.store import ArrayCodec, DataStore, JSONCodec, design_codec
from astrolabe.design.types import (
    Assignment, Domain, Polymer, Sample, key_for, make_candidate, make_system,
)
from astrolabe.evaluators.utils import StandardScores


@pytest.fixture(autouse=True)
def scientific_state():
    global_config.reset()
    for name in ("segments", "protomers", "mutable_protomers", "named_files"):
        global_config.init_const(name, {})
    global_config.init_const("segment_manager", SegmentManager())
    yield
    global_config.reset()


def add(name, sequence):
    protomer = Protomer(name, sequence, "p")
    global_config.protomers[name] = protomer
    return protomer


def test_owner_restrictions_are_resolved_and_preserved_when_applying():
    add("A", "AC")
    data = DataStore()
    original = legacy.translate_legacy_state(assemblies=("A",), data=data)
    restriction = data.put({"allowed_tokens": {"1": ["C", "G"]}}, JSONCodec())
    system = make_system(
        original.system.variables, original.system.occurrences, original.system.topology,
        (Domain("A_1", (), (), restriction),),
    )
    base = make_candidate(system, original.candidate.assignments, data=data)
    translation = replace(original, system=system, candidate=base)
    context = legacy.LegacyContext()
    permitted = legacy.candidate_for_mutation(
        translation, (Assignment("A_1", Polymer("protein", ("A", "G"), ())),), data=data,
    )
    context.apply_candidate(permitted, translation=translation, data=data)
    assert global_config.segments["A_1"].sequence == "AG"
    with pytest.raises(ValueError, match="not allowed"):
        legacy.candidate_for_mutation(
            translation, (Assignment("A_1", Polymer("protein", ("A", "T"), ())),), data=data,
        )
    # Keeping a permitted identity while swapping payloads cannot bypass domains.
    forged = replace(permitted, assignments=(
        Assignment("A_1", Polymer("protein", ("A", "T"), ())),
    ))
    with pytest.raises(ValueError, match="not allowed"):
        context.apply_candidate(forged, translation=translation, data=data)
    assert global_config.segments["A_1"].sequence == "AG"
    context.apply_candidate(base, translation=translation, data=data)
    assert global_config.segments["A_1"].sequence == "AC"


def test_apply_rolls_back_mid_write_and_preserves_alias_identity(monkeypatch):
    add("A", "AC")
    add("B", "AT")
    first, second = global_config.segments["A_1"], global_config.segments["B_1"]
    global_config.segments["shared"] = first
    first.probs = [np.asarray([0.2, 0.8])]
    first.scores = [0.7]
    first.mutations = [(0, "A", "G", 1.0, 0.7)]
    manager = global_config.segment_manager
    manager.begin_pre_update_step()
    data = DataStore()
    context = legacy.LegacyContext()
    original = context.capture_candidate(assemblies=("AB",), data=data)
    proposal = legacy.candidate_for_mutation(original, tuple(
        Assignment(name, Polymer("protein", ("G", "G"), ()))
        for name in ("A_1", "B_1")
    ), data=data)
    original_setattr = Segment.__setattr__

    def fail_on_second(self, name, value):
        if self is second and name == "sequence" and value == "GG":
            raise RuntimeError("second write failed")
        original_setattr(self, name, value)

    monkeypatch.setattr(Segment, "__setattr__", fail_on_second)
    with pytest.raises(RuntimeError, match="second write failed"):
        context.apply_candidate(proposal, translation=original, data=data)
    assert first.sequence == "AC" and second.sequence == "AT"
    assert global_config.segments["shared"] is first
    assert manager.unique_segments[id(first)] is first
    np.testing.assert_array_equal(first.probs[0], [0.2, 0.8])
    assert first.scores == [0.7]
    assert first.mutations == [(0, "A", "G", 1.0, 0.7)]
    assert manager._pre_update_snapshot is not None


def test_ptm_and_fixed_segments_reject_edits_before_native_mutation():
    add("A", "AC")
    add("B", "+GT")
    global_config.segments["A_1"].ptm_dict = {1: "PHOS"}
    data = DataStore()
    original = legacy.translate_legacy_state(assemblies=("AB",), data=data)
    for replacement, reason in (
        (Assignment("A_1", Polymer("protein", ("A", "G"), ((1, "PHOS"),))), "modified token"),
        (Assignment("A_1", Polymer("protein", ("A", "C"), ())), "PTM"),
        (Assignment("B_1", Polymer("protein", ("G", "G"), ())), "fixed token"),
    ):
        with pytest.raises(ValueError, match=reason):
            legacy.candidate_for_mutation(original, (replacement,), data=data)
    assert global_config.segments["A_1"].ptm_dict == {1: "PHOS"}
    assert global_config.segments["B_1"].sequence == "GT"


def test_template_evidence_and_identity_survive_a_fresh_process(tmp_path):
    protomer = add("A", "AC")
    protomer._homology = {"templates": [Template(
        mmcif="data_synthetic_template\n# test fixture\n", query_to_template_map={1: 3, 0: 2},
    )]}
    data = DataStore(tmp_path / "store")
    original = legacy.translate_legacy_state(assemblies=("A",), data=data)
    sample = Sample(key_for("test/protocol"), "replicate-1", 17, "prediction-0")
    path = tmp_path / "selected.cif"
    path.write_bytes(b"data_synthetic_structure\n# test fixture\n")
    scores = StandardScores(
        names=["A"], complex_paths={"A": path},
        plddt={"A": [np.array([0.6, np.nan], dtype=np.float32)]},
        per_proto_scores={"A": [np.array([0.4, 0.5])]},
    )
    evidence = legacy.standard_scores_to_evidence(
        scores, original.system, original.candidate, data,
        samples={"A": (sample,)}, artifact_sample_by_assembly={"A": sample},
    )
    reference = data.put((original.system, original.candidate, evidence), design_codec)
    data.persist((reference,))
    path.unlink()
    program = r'''
import json, sys
from pathlib import Path
import numpy as np
from astrolabe.design.legacy import LEGACY_REFERENCE_CODECS, LegacyFileBytesCodec, decode_legacy_templates
from astrolabe.design.store import DataStore, ArrayCodec, JSONCodec, design_codec
from astrolabe.design.types import Blob, Key
store = DataStore(Path(sys.argv[1]))
for codec in LEGACY_REFERENCE_CODECS:
    store.register(codec)
reference = Blob(Key(sys.argv[2]), design_codec.identity)
system, candidate, evidence = store.get(reference, design_codec)
assert evidence.evaluated_candidate == candidate.identity
assert evidence.identity == evidence.with_measurements(()).identity
assert candidate.system == system.identity
artifact, = evidence.artifacts
assert artifact.candidate == candidate.identity
assert store.get(artifact.content, LegacyFileBytesCodec()).startswith(b"data_synthetic_structure")
plddt = next(m for m in evidence.measurements if ".plddt." in m.quantity)
assert plddt.subjects[0].sample == artifact.sample
assert plddt.subjects[0].artifact == artifact.identity
assert plddt.missing == "partial"
assert store.get(plddt.values, ArrayCodec()).dtype == np.float32
assert store.get(plddt.valid_mask, ArrayCodec()).tolist() == [True, False]
aggregate = next(m for m in evidence.measurements if m.quantity.startswith("legacy.aggregate"))
assert aggregate.subjects[0].sample is None
assert aggregate.subjects[0].occurrences == ()
topology = store.get(system.topology, JSONCodec("astrolabe.legacy.topology/v1"))
template, = decode_legacy_templates(topology["protomers"]["A"]["homology"], data=store)["templates"]
assert template.query_to_template_map == {0: 2, 1: 3}
assert template.mmcif.startswith("data_synthetic_template")
print(json.dumps([system.identity.digest, candidate.identity.digest, evidence.identity.digest]))
'''
    completed = subprocess.run(
        [sys.executable, "-c", program, str(data.root), reference.identity.digest],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
    )
    assert json.loads(completed.stdout) == [
        original.system.identity.digest, original.candidate.identity.digest, evidence.identity.digest,
    ]
    again = legacy.translate_legacy_state(assemblies=("A",), data=data)
    assert again.system.identity == original.system.identity
    assert again.candidate.identity == original.candidate.identity
    protomer._homology["templates"][0] = Template(mmcif="data_changed\n", query_to_template_map={0: 2})
    changed = legacy.translate_legacy_state(assemblies=("A",), data=data)
    assert changed.system.identity != original.system.identity


def test_applying_captured_state_restores_fixed_flag_and_ptm():
    add("A", "+AC")
    segment = global_config.segments["A_1"]
    segment.ptm_dict = {1: "PHOS"}
    data = DataStore()
    original = legacy.translate_legacy_state(assemblies=("A",), data=data)
    segment.ptm_dict = {}
    segment.is_fixed = False
    legacy.LegacyContext().apply_candidate(
        original.candidate, translation=original, data=data,
    )
    assert segment.is_fixed
    assert segment.ptm_dict == {1: "PHOS"}
    segment.is_fixed = False
    legacy.LegacyContext().apply_candidate(
        original.candidate, translation=original, data=data,
    )
    assert segment.is_fixed
