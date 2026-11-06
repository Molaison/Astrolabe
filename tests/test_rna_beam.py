"""Numerical search bookkeeping and real shared-segment contract checks.

These tests do not stand in for NA-MPNN or AlphaFold 3 inference.
"""

import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.design.repeats import RepeatRun
from astrolabe.rna_beam import (
    _expand_repeated_protomer,
    _normalize_bivalent_sequence_statement,
    _restore_sequence,
    _sequence_key,
    _validate_bivalent_rna,
    _validate_sequence_statements,
    _validate_shared_rna,
    score_bivalent_rna,
    validate_bivalent_objective_weights,
    proposal_seed,
    beam_expansion_order,
    run_configured_refinement,
    run_rna_beam,
)


def test_random_rna_initialization_is_rejected_before_parsing():
    for statement in ("87", "80-90@shared", "+87", "ACGU 83"):
        with pytest.raises(ValueError, match="supplied candidate"):
            _validate_sequence_statements({"A": statement})
    _validate_sequence_statements({"A": "ACGU@shared", "B": "shared"})


@pytest.mark.parametrize("chains", ["ABCD", "ABCDE", "AAAA", "AAAAA"])
def test_native_segments_share_queued_changes_and_restore_parent(chains):
    from astrolabe.core import global_config
    from astrolabe.core.protomer_config_parser import Protomer, SegmentManager

    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("named_files", {})
    sequence = "ACGU" * 21 + "ACG"
    try:
        for name in dict.fromkeys(chains):
            statement = sequence + "@shared" if name == "A" else "shared"
            global_config.protomers[name] = Protomer(name, statement, "rna")
        assert _validate_shared_rna(global_config.protomers, global_config.segments, chains) == chains
        physical_chains = _expand_repeated_protomer(chains)
        assert physical_chains == "ABCDE"[:len(chains)]
        assert _validate_shared_rna(global_config.protomers, global_config.segments) == physical_chains
        shared = global_config.segments["shared"]
        shared.add_mutations((0, "A", "U", 0.0, 0.0))
        assert {p.mutated_sequence for p in global_config.protomers.values()} == {"U" + sequence[1:]}
        global_config.segment_manager.apply_mutations()
        assert {p.sequence for p in global_config.protomers.values()} == {"U" + sequence[1:]}
        _restore_sequence(sequence, global_config.protomers["A"], global_config.segments)
        assert {p.sequence for p in global_config.protomers.values()} == {sequence}
        assert shared.mutations == []
        # Equal strings in independent segments do not implement synchronization.
        global_config.protomers["B"] = Protomer("B_independent", sequence, "rna")
        with pytest.raises(ValueError, match="same ordered segments"):
            _validate_shared_rna(global_config.protomers, global_config.segments)
    finally:
        global_config.reset()


def test_native_bivalent_repeats_share_initialization_mutation_and_restore():
    import numpy as np
    from astrolabe.core import global_config
    from astrolabe.core.protomer_config_parser import Protomer, SegmentManager

    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("named_files", {})
    try:
        np.random.seed(29)
        rna = Protomer("A", _normalize_bivalent_sequence_statement("20@repeat repeat"), "rna")
        target = "ACDEFGHIKLMNPQRSTVWY" * 4 + "ACDEFGHIKLMNPQRS"
        global_config.protomers.update(A=rna, B=Protomer("B", "+" + target, "protein"),
                                      C=Protomer("C", "+" + target, "protein"))
        assert _validate_bivalent_rna(global_config.protomers, global_config.segments, "ABC") == "ABC"
        original = _sequence_key(rna, global_config.segments)
        assert len(original) == 20
        assert rna.sequence == original * 2
        replacement = "A" if original[0] != "A" else "C"
        global_config.segments["repeat"].add_mutations((0, original[0], replacement, 0.0, 0.0))
        changed = replacement + original[1:]
        assert _sequence_key(rna, global_config.segments, mutated=True) == changed
        assert rna.mutated_sequence == changed * 2
        _restore_sequence(changed, rna, global_config.segments)
        assert rna.sequence == changed * 2
        _restore_sequence(original, rna, global_config.segments)
        assert rna.sequence == original * 2
        assert global_config.protomers["B"].sequence == target
        assert global_config.protomers["C"].sequence == target
    finally:
        global_config.reset()


def test_bivalent_objective_requires_explicit_weights_and_saturates_clash_penalty():
    weights = {"weak_interface": 2, "repeat_contacts": 3, "repeat_plddt": 4, "clashes": 0.25}
    metrics = {"dual_interface": 0.6, "repeat_cohesion": 0.5, "weak_arm_plddt": 0.4, "interchain_clashes": 1}
    assert score_bivalent_rna(metrics, weights) == pytest.approx(1.2 + 1.5 + 1.6 - 0.25)
    metrics["interchain_clashes"] = 100
    assert score_bivalent_rna(metrics, weights) == pytest.approx(1.2 + 1.5 + 1.6 - 0.25)
    for invalid in (None, {}, {**weights, "extra": 1}, {**weights, "clashes": -1}, {**weights, "clashes": math.nan}):
        with pytest.raises(ValueError):
            validate_bivalent_objective_weights(invalid)


def test_archived_proposal_seed_and_balanced_partial_generation():
    assert proposal_seed(67101, 1) == 267850363
    assert proposal_seed(67101, 2) == 267850364
    assert beam_expansion_order(["parent0", "parent1", "parent2"], 4)[:5] == [
        ("parent0", 0), ("parent1", 0), ("parent2", 0), ("parent0", 1), ("parent1", 1),
    ]


def test_configured_refinement_translates_canonical_search_once(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    import astrolabe.rna_beam as module

    calls = []

    def execute(cfg, **kwargs):
        calls.append((cfg, kwargs))
        return {"ok": True}

    monkeypatch.setattr(module, "_run_rna_scientific_composition", execute)
    cfg = OmegaConf.create(
        {
            "search": {
                "algorithm": "branching",
                "owner": "archived_refinement",
                "policy": {
                    "width": 3,
                    "proposals_per_parent": 2,
                    "max_evaluations": 11,
                },
                "measurement": "confidence",
                "initialization": "configured",
            }
        }
    )
    output = tmp_path / "configured"
    result = run_configured_refinement(
        cfg,
        RepeatRun(index=0, total=1, seed=71, output=output),
    )
    assert result == {"ok": True}
    translated, kwargs = calls[0]
    assert "search" not in translated
    assert translated.beam_search.initialization == "configured"
    persisted = kwargs.pop("persisted_config")
    assert persisted.search.algorithm == "branching"
    assert persisted.search.owner == "archived_refinement"
    assert persisted.search.initialization == "configured"
    assert persisted.seed == 71
    assert persisted.out_dir == str(output)
    assert persisted.repeats == 1
    assert kwargs == {
        "out_dir": output,
        "seed": 71,
        "beam_width": 3,
        "proposals_per_parent": 2,
        "max_evaluations": 11,
        "objective_weights": None,
        "resume": False,
    }


def test_configured_bivalent_start_uses_same_native_boundary(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    import astrolabe.rna_beam as module

    calls = []

    def execute(cfg, **kwargs):
        calls.append((cfg, kwargs))
        return {"ok": True}

    monkeypatch.setattr(module, "_run_rna_scientific_composition", execute)
    weights = {
        "weak_interface": 2,
        "repeat_contacts": 1,
        "repeat_plddt": 0.5,
        "clashes": 0.5,
    }
    cfg = OmegaConf.create(
        {
            "search": {
                "width": 8,
                "proposals_per_parent": 4,
                "max_evaluations": 200,
                "measurement": "bivalent_geometry",
                "algorithm": "branching",
                "owner": "archived_refinement",
                "initialization": "archived",
                "objective_weights": weights,
            }
        }
    )
    run_configured_refinement(
        cfg,
        RepeatRun(
            index=0,
            total=1,
            seed=83,
            output=tmp_path / "bivalent",
            start={"sequence": "ACGUACGUACGUACGUACGU"},
        ),
    )
    translated, kwargs = calls[0]
    assert translated.beam_search.initialization == "archived"
    assert translated.beam_search.archived_starts[0].sequence == "ACGUACGUACGUACGUACGU"
    assert translated.beam_search.archived_start == 0
    assert kwargs["objective_weights"] == {
        key: float(value) for key, value in sorted(weights.items())
    }
    persisted = kwargs["persisted_config"]
    assert persisted.search.algorithm == "branching"
    assert persisted.search.owner == "archived_refinement"
    assert persisted.search.start.sequence == "ACGUACGUACGUACGUACGU"
    assert list(persisted.search.archived_starts) == [persisted.search.start]
    assert persisted.search.archived_start == 0
    assert persisted.seed == 83


def test_legacy_public_entry_keeps_legacy_config_behavior(tmp_path, monkeypatch):
    import astrolabe.rna_beam as module

    calls = []

    def execute(cfg, **kwargs):
        calls.append((cfg, kwargs))
        return {"ok": True}

    monkeypatch.setattr(module, "_run_rna_scientific_composition", execute)
    cfg = {"legacy": True}
    result = run_rna_beam(
        cfg,
        out_dir=tmp_path / "legacy",
        seed=71,
        beam_width=1,
        proposals_per_parent=1,
        max_evaluations=1,
    )
    assert result == {"ok": True}
    assert calls[0][0] is cfg
    assert "persisted_config" not in calls[0][1]


@pytest.mark.parametrize("extra", [
    {"unknown_control": True},
    {"policy": {"pipeline": True}},
    {"policy": {"width": 3}},
])
@pytest.mark.parametrize("entry", ["factory", "public"])
def test_archived_factory_rejects_ignored_or_conflicting_controls(tmp_path, monkeypatch, extra, entry):
    from omegaconf import OmegaConf
    import astrolabe.rna_beam as module

    def unexpected(*args, **kwargs):
        pytest.fail("invalid controls reached scientific initialization")

    if entry == "factory":
        monkeypatch.setattr(module, "_run_rna_scientific_composition", unexpected)
    cfg = OmegaConf.create({"search": {
        "algorithm": "branching", "owner": "archived_refinement",
        "width": 1, "proposals_per_parent": 1, "max_evaluations": 3,
        "measurement": "confidence", **extra,
    }})
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="unsupported|conflicting"):
        if entry == "factory":
            run_configured_refinement(cfg, RepeatRun(index=0, total=1, seed=71, output=output))
        else:
            run_rna_beam(cfg, out_dir=output, seed=71)
    assert not output.exists()


@pytest.mark.parametrize("entry", ["configured", "legacy", "canonical_public"])
def test_rna_public_entry_uses_generic_branching_and_writes_legacy_outputs(
    tmp_path,
    monkeypatch,
    entry,
):
    from omegaconf import OmegaConf, open_dict

    from astrolabe.core import global_config
    from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
    import astrolabe.evaluators.utils as evaluator_utils
    import astrolabe.mutators.base as mutator_base
    import astrolabe.utils.base as utils_base
    import run as run_module

    output = tmp_path / "rna-run"
    model_dir = tmp_path / "models" / "alphafold3"
    model_dir.mkdir(parents=True)
    (model_dir / "af3.bin").write_bytes(b"fake AF3 asset")
    nampnn_checkpoint = tmp_path / "models" / "nampnn.pt"
    nampnn_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    nampnn_checkpoint.write_bytes(b"fake NAMPNN asset")
    constructors = []
    constructor_config_exists = []
    shutdowns = []
    proposal_ordinal = 0
    sequence = "ACGU" * 21 + "ACG"
    cfg = OmegaConf.create(
        {
            "seqs": {"A": sequence + "@shared"},
            "oligo": ["AAAA"],
            "structure_start": None,
            "search": {
                "algorithm": "branching",
                "owner": "archived_refinement",
                "width": 1,
                "proposals_per_parent": 1,
                "max_evaluations": 3,
                "measurement": "confidence",
                "initialization": "configured",
            },
            "repeat_plan": {"output": "ignored/{repeat:03d}_seed_{seed}"},
            "evaluator": {
                "evaluator_terms": {
                    "AF3": {
                        "_type": "alphafold3",
                        "num_seeds": 1,
                        "num_samples": 1,
                    }
                },
                "alphafold3": {"model_dir": str(model_dir)},
            },
            "mutator": {
                "mutator_terms": {
                    "MPNN": {
                        "_type": "nampnn",
                    }
                },
                "NAMPNN": {"checkpoint": str(nampnn_checkpoint)},
            },
        }
    )

    def initialize(_args):
        return None

    def setup(_args):
        global_config.reset()
        global_config.init_const("segments", {})
        global_config.init_const("protomers", {})
        global_config.init_const("mutable_protomers", {})
        global_config.init_const("segment_manager", SegmentManager())
        global_config.init_const("named_files", {})
        protomer = Protomer("A", sequence + "@shared", "rna")
        global_config.protomers["A"] = protomer
        global_config.mutable_protomers["A"] = protomer

    class Mutator:
        def __init__(self):
            self.runner = SimpleNamespace(
                batch_size=None,
                number_of_batches=None,
                extra_args={},
                job_name=None,
            )
            self.position_selector = None
            self.max_mutations_per_proto = None

        def mutate(self, protomers, scores, runtime_config):
            nonlocal proposal_ordinal
            proposal_ordinal += 1
            segment = global_config.segments["shared"]
            position = proposal_ordinal - 1
            token = segment.sequence[position]
            replacement = next(item for item in "ACGU" if item != token)
            segment.sequence = (
                segment.sequence[:position]
                + replacement
                + segment.sequence[position + 1 :]
            )

        def shutdown(self):
            shutdowns.append("mutator")

    class Evaluator:
        def evaluate(self, inputs, runtime_config):
            step = int(global_config.get_variable("step"))
            assembly = inputs[0][0]
            model = output / "raw_model_results" / "fake" / f"{step}.cif"
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_text(f"model:{step}\n", encoding="utf-8")
            value = np.asarray([0.4 + step / 10, 0.6 + step / 10], dtype=np.float32)
            return __import__(
                "astrolabe.evaluators.utils",
                fromlist=["StandardScores"],
            ).StandardScores(
                names=[assembly],
                plddt={assembly: [value]},
                complex_paths={assembly: model},
            )

        def shutdown(self):
            shutdowns.append("evaluator")

    def merge(*args, **kwargs):
        return SimpleNamespace(num_seeds=1, num_samples=1)

    def build_mutators(*args, **kwargs):
        constructor_config_exists.append((output / "04_config.yaml").is_file())
        constructors.append("mutator")
        return {"MPNN": Mutator()}

    def construct_evaluators(*args, **kwargs):
        constructor_config_exists.append((output / "04_config.yaml").is_file())
        constructors.append("evaluator")
        return {"AF3": Evaluator()}

    monkeypatch.setattr(run_module, "initialize_global_config", initialize)
    monkeypatch.setattr(run_module, "setup_protomers", setup)
    monkeypatch.setattr(evaluator_utils, "_merge_conf_with_defaults", merge)
    monkeypatch.setattr(evaluator_utils, "construct_evaluators", construct_evaluators)
    monkeypatch.setattr(mutator_base, "build_mutators", build_mutators)
    monkeypatch.setattr(utils_base, "construct_standard_input", lambda names, seed: (tuple(names), seed))

    try:
        if entry == "configured":
            result = run_configured_refinement(
                cfg,
                RepeatRun(index=0, total=1, seed=71, output=output),
            )
        else:
            if entry == "legacy":
                del cfg.search
            result = run_rna_beam(
                cfg, out_dir=output, seed=71, beam_width=1,
                proposals_per_parent=1, max_evaluations=3,
            )
        assert constructors == ["mutator", "evaluator"]
        assert constructor_config_exists == [True, True]
        assert shutdowns == ["evaluator", "mutator"]
        assert result["evaluations"] == 3
        assert result["proposals"] == 2
        assert len(result["beam"]) == 1
        assert (output / "01_evaluations.jsonl").is_file()
        assert (output / "02_evaluations.csv").is_file()
        assert (output / "03_final_beam.json").is_file()
        assert (output / "04_config.yaml").is_file()
        persisted = OmegaConf.load(output / "04_config.yaml")
        import yaml

        plain = yaml.safe_load((output / "04_config.yaml").read_text())
        assert plain["search"]["owner"] == "archived_refinement"
        assert persisted.search.algorithm == "branching"
        assert persisted.search.owner == "archived_refinement"
        assert persisted.search.initialization == "configured"
        assert persisted.search.measurement == "confidence"
        assert persisted.seed == 71
        assert persisted.out_dir == str(output)
        assert persisted.repeats == 1
        if entry == "configured":
            assert persisted.repeat_index == 0
        assert persisted.get("repeat_plan") is None
        assert persisted.get("beam_search") is None
        from astrolabe.configuration import selected_search_target

        assert selected_search_target(persisted) == ("branching", "archived_refinement")
        events = [
            __import__("json").loads(line)
            for line in (output / "01_evaluations.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert [event["event"] for event in events[:6]] == [
            "search_started",
            "evaluation_started",
            "evaluation_completed",
            "proposal_started",
            "proposal_completed",
            "evaluation_started",
        ]
        if entry == "configured":
            with open_dict(cfg.search):
                cfg.search.resume = True
            resumed = run_configured_refinement(
                cfg,
                RepeatRun(index=0, total=1, seed=71, output=output),
            )
            assert resumed == result
            assert constructors == ["mutator", "evaluator"]
            assert shutdowns == ["evaluator", "mutator"]
            resumed_events = [
                __import__("json").loads(line)
                for line in (output / "01_evaluations.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            assert sum(
                event["event"] == "search_started"
                for event in resumed_events
            ) == 1
            assert sum(
                event["event"] == "search_completed"
                for event in resumed_events
            ) == 1
    finally:
        global_config.reset()


def _fake_rna_config(tmp_path, *, max_evaluations=3):
    from omegaconf import OmegaConf

    model_dir = tmp_path / "models" / "alphafold3"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "af3.bin").write_bytes(b"fake AF3 asset")
    nampnn_checkpoint = tmp_path / "models" / "nampnn.pt"
    nampnn_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    nampnn_checkpoint.write_bytes(b"fake NAMPNN asset")
    sequence = "ACGU" * 21 + "ACG"
    return OmegaConf.create(
        {
            "seqs": {"A": sequence + "@shared"},
            "oligo": ["AAAA"],
            "structure_start": None,
            "search": {
                "algorithm": "branching",
                "owner": "archived_refinement",
                "width": 1,
                "proposals_per_parent": 1,
                "max_evaluations": max_evaluations,
                "measurement": "confidence",
                "initialization": "configured",
            },
            "repeat_plan": {"output": "ignored/{repeat:03d}_seed_{seed}"},
            "evaluator": {
                "evaluator_terms": {
                    "AF3": {
                        "_type": "alphafold3",
                        "num_seeds": 1,
                        "num_samples": 1,
                    }
                },
                "alphafold3": {"model_dir": str(model_dir)},
            },
            "mutator": {
                "mutator_terms": {"MPNN": {"_type": "nampnn"}},
                "NAMPNN": {"checkpoint": str(nampnn_checkpoint)},
            },
        }
    )


def _install_fake_rna_components(
    monkeypatch,
    *,
    output,
    sequence,
    constructors,
    constructor_config_exists,
    shutdowns,
    proposals,
    evaluations,
    af3_num_samples=1,
    af3_num_recycles=10,
    merge_calls=None,
):
    from astrolabe.core import global_config
    from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
    import astrolabe.evaluators.utils as evaluator_utils
    import astrolabe.mutators.base as mutator_base
    import astrolabe.utils.base as utils_base
    import run as run_module

    def initialize(_args):
        return None

    def setup(_args):
        global_config.reset()
        global_config.init_const("segments", {})
        global_config.init_const("protomers", {})
        global_config.init_const("mutable_protomers", {})
        global_config.init_const("segment_manager", SegmentManager())
        global_config.init_const("named_files", {})
        protomer = Protomer("A", sequence + "@shared", "rna")
        global_config.protomers["A"] = protomer
        global_config.mutable_protomers["A"] = protomer

    class Mutator:
        def __init__(self):
            self.runner = SimpleNamespace(
                batch_size=None,
                number_of_batches=None,
                extra_args={},
                job_name=None,
            )
            self.position_selector = None
            self.max_mutations_per_proto = None

        def mutate(self, protomers, scores, runtime_config):
            proposals[0] += 1
            segment = global_config.segments["shared"]
            position = proposals[0] - 1
            token = segment.sequence[position]
            replacement = next(item for item in "ACGU" if item != token)
            random.random()
            np.random.random()
            segment.sequence = (
                segment.sequence[:position]
                + replacement
                + segment.sequence[position + 1 :]
            )

        def shutdown(self):
            shutdowns.append("mutator")

    class Evaluator:
        def evaluate(self, inputs, runtime_config):
            step = int(global_config.get_variable("step"))
            evaluations.append(step)
            assembly = inputs[0][0]
            model = output / "raw_model_results" / "fake" / f"{step}.cif"
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_text(f"model:{step}\n", encoding="utf-8")
            offset = 0.01 * random.random()
            value = np.asarray(
                [0.4 + step / 10 + offset + np.random.random() / 1000, 0.6 + step / 10],
                dtype=np.float32,
            )
            return __import__(
                "astrolabe.evaluators.utils",
                fromlist=["StandardScores"],
            ).StandardScores(
                names=[assembly],
                plddt={assembly: [value]},
                complex_paths={assembly: model},
            )

        def shutdown(self):
            shutdowns.append("evaluator")

    class Mutators(dict):
        def shutdown(self):
            return None

    class Evaluators(dict):
        def shutdown(self):
            return None

    def merge(*args, **kwargs):
        if merge_calls is not None:
            merge_calls.append((args[0], args[2]))
        return SimpleNamespace(
            num_seeds=1,
            num_samples=af3_num_samples,
            num_recycles=af3_num_recycles,
        )

    def build_mutators(*args, **kwargs):
        constructor_config_exists.append((output / "04_config.yaml").is_file())
        constructors.append("mutator")
        value = Mutators({"MPNN": Mutator()})
        return value

    def construct_evaluators(*args, **kwargs):
        constructor_config_exists.append((output / "04_config.yaml").is_file())
        constructors.append("evaluator")
        value = Evaluators({"AF3": Evaluator()})
        return value

    monkeypatch.setattr(run_module, "initialize_global_config", initialize)
    monkeypatch.setattr(run_module, "setup_protomers", setup)
    monkeypatch.setattr(evaluator_utils, "_merge_conf_with_defaults", merge)
    monkeypatch.setattr(evaluator_utils, "construct_evaluators", construct_evaluators)
    monkeypatch.setattr(mutator_base, "build_mutators", build_mutators)
    monkeypatch.setattr(
        utils_base,
        "construct_standard_input",
        lambda names, seed: (tuple(names), seed),
    )


def test_rna_selected_helpers_use_projection_without_mutating_science(tmp_path):
    from omegaconf import OmegaConf, open_dict

    import astrolabe.rna_beam as module
    from astrolabe.design.configured_refinement import (
        _scientific_config_payload,
        _selected_asset_manifest,
    )

    cfg = _fake_rna_config(tmp_path, max_evaluations=2)
    cfg.homologies = {
        "B": {
            "templates": [
                {
                    "mmcif": "data_template\n#\n",
                    "queryIndices": [0],
                    "templateIndices": [0],
                }
            ]
        }
    }
    with open_dict(cfg.evaluator.evaluator_terms.AF3):
        cfg.evaluator.evaluator_terms.AF3.num_samples = 5
        cfg.evaluator.evaluator_terms.AF3.num_recycles = 10
    args = copy.deepcopy(cfg)
    args._set_flag("allow_objects", True)
    with open_dict(args):
        args.out_dir = tmp_path / "run"
        args.raw_model_results_dir = tmp_path / "run" / "raw_model_results"
        args.seed = 83
        args.mutator.enabled_aliases = ["MPNN"]
        args.evaluator.enabled_aliases = ["AF3"]
    before = OmegaConf.to_container(args, resolve=False)

    scientific = module._rna_scientific_config(
        args,
        seed=83,
        beam_width=1,
        proposals_per_parent=1,
        max_evaluations=2,
        objective_weights=None,
        initialization="configured",
    )
    projection = module._rna_selection_projection(
        scientific,
        evaluator_alias="AF3",
        mutator_alias="MPNN",
    )
    payload = _scientific_config_payload(projection)
    manifest = _selected_asset_manifest(projection)
    selected = module._rna_selected_scientific_identity(
        scientific,
        evaluator_alias="AF3",
        mutator_alias="MPNN",
    )
    resources = module._rna_resource_content(
        scientific,
        evaluator_alias="AF3",
        mutator_alias="MPNN",
    )

    assert OmegaConf.to_container(args, resolve=False) == before
    assert scientific["seed"] == payload["seed"] == 83
    af3 = scientific["evaluator"]["evaluator_terms"]["AF3"]
    assert af3["num_samples"] == 5
    assert af3["num_recycles"] == 10
    assert projection.stages.default.evaluator == "AF3"
    assert list(projection.stages.default.mutators) == ["MPNN"]
    assert {
        (item.role, item.type, item.state)
        for item in manifest.components
        if item.role in {"mutator:MPNN", "evaluator:AF3"}
    } == {
        ("mutator:MPNN", "nampnn", "hashed"),
        ("evaluator:AF3", "alphafold3", "hashed"),
    }

    changed = copy.deepcopy(args)
    with open_dict(changed.evaluator.evaluator_terms.AF3):
        changed.evaluator.evaluator_terms.AF3.num_samples = 6
    changed_scientific = module._rna_scientific_config(
        changed,
        seed=83,
        beam_width=1,
        proposals_per_parent=1,
        max_evaluations=2,
        objective_weights=None,
        initialization="configured",
    )
    changed_selected = module._rna_selected_scientific_identity(
        changed_scientific,
        evaluator_alias="AF3",
        mutator_alias="MPNN",
    )
    changed_resources = module._rna_resource_content(
        changed_scientific,
        evaluator_alias="AF3",
        mutator_alias="MPNN",
    )
    assert changed_selected != selected
    assert changed_resources == resources


def test_rna_two_evaluation_cpu_run_preserves_sample_and_recycle_settings(
    tmp_path,
    monkeypatch,
):
    from omegaconf import open_dict

    from astrolabe.core import global_config

    cfg = _fake_rna_config(tmp_path, max_evaluations=2)
    with open_dict(cfg.evaluator.evaluator_terms.AF3):
        cfg.evaluator.evaluator_terms.AF3.num_samples = 5
        cfg.evaluator.evaluator_terms.AF3.num_recycles = 10
    output = tmp_path / "two-evaluations"
    constructors = []
    constructor_config_exists = []
    shutdowns = []
    proposals = [0]
    evaluations = []
    merge_calls = []
    _install_fake_rna_components(
        monkeypatch,
        output=output,
        sequence="ACGU" * 21 + "ACG",
        constructors=constructors,
        constructor_config_exists=constructor_config_exists,
        shutdowns=shutdowns,
        proposals=proposals,
        evaluations=evaluations,
        af3_num_samples=5,
        af3_num_recycles=10,
        merge_calls=merge_calls,
    )

    try:
        result = run_configured_refinement(
            cfg,
            RepeatRun(index=0, total=1, seed=83, output=output),
        )
    finally:
        global_config.reset()

    assert constructors == ["mutator", "evaluator"]
    assert evaluations == [0, 1]
    assert proposals[0] == 1
    assert result["evaluations"] == 2
    assert result["proposals"] == 1
    assert result["af3_samples_per_evaluation"] == 5
    assert merge_calls
    assert merge_calls[0][1].num_samples == 5
    assert merge_calls[0][1].num_recycles == 10


@pytest.mark.parametrize("change", ["asset", "config", "source"])
def test_rna_resume_rejects_changed_identity_before_model_construction(
    tmp_path,
    monkeypatch,
    change,
):
    from omegaconf import open_dict

    import astrolabe.rna_beam as module
    from astrolabe.core import global_config
    from astrolabe.design.runtime import ContractConflict

    cfg = _fake_rna_config(tmp_path, max_evaluations=1)
    output = tmp_path / "resume"
    constructors = []
    constructor_config_exists = []
    shutdowns = []
    proposals = [0]
    evaluations = []
    _install_fake_rna_components(
        monkeypatch,
        output=output,
        sequence="ACGU" * 21 + "ACG",
        constructors=constructors,
        constructor_config_exists=constructor_config_exists,
        shutdowns=shutdowns,
        proposals=proposals,
        evaluations=evaluations,
    )

    try:
        run_configured_refinement(
            cfg,
            RepeatRun(index=0, total=1, seed=71, output=output),
        )
        assert constructors == ["mutator", "evaluator"]
        constructors.clear()
        constructor_config_exists.clear()
        evaluations.clear()

        if change == "asset":
            (tmp_path / "models" / "alphafold3" / "af3.bin").write_bytes(
                b"changed AF3 asset"
            )
        elif change == "config":
            cfg.evaluator.evaluator_terms.AF3.num_samples = 2
        else:
            monkeypatch.setattr(
                module,
                "_rna_source_content",
                lambda: ("changed scientific source",),
            )
        with open_dict(cfg.search):
            cfg.search.resume = True

        with pytest.raises(
            ContractConflict,
            match="scientific contract|scientific config",
        ):
            run_configured_refinement(
                cfg,
                RepeatRun(index=0, total=1, seed=71, output=output),
            )
        assert constructors == []
        assert evaluations == []
    finally:
        global_config.reset()


def test_rna_source_identity_covers_native_execution_layers():
    import astrolabe.rna_beam as module

    roles = {record.role for record in module._rna_source_content()}
    assert {
        "source/extra/runtime",
        "source/extra/control_journal",
        "source/extra/branching",
        "source/extra/native_refinement",
        "source/extra/native_refinement_runtime",
        "source/extra/configured_refinement",
        "source/components/nampnn",
        "source/components/nampnn_runner",
        "source/components/alphafold3",
        "source/components/alphafold3_wrapper",
    } <= roles


def test_rna_resume_after_raw_prediction_reuses_receipt_without_reinference(
    tmp_path,
    monkeypatch,
):
    from omegaconf import open_dict

    from astrolabe.core import global_config
    from astrolabe.design.runtime import Runtime

    sequence = "ACGU" * 21 + "ACG"
    cfg = _fake_rna_config(tmp_path, max_evaluations=3)
    baseline = tmp_path / "baseline"
    constructors = []
    constructor_config_exists = []
    shutdowns = []
    proposals = [0]
    evaluations = []
    _install_fake_rna_components(
        monkeypatch,
        output=baseline,
        sequence=sequence,
        constructors=constructors,
        constructor_config_exists=constructor_config_exists,
        shutdowns=shutdowns,
        proposals=proposals,
        evaluations=evaluations,
    )

    try:
        baseline_result = run_configured_refinement(
            cfg,
            RepeatRun(index=0, total=1, seed=71, output=baseline),
        )
        baseline_evaluations = tuple(evaluations)
        baseline_proposals = proposals[0]
        baseline_rng = (random.getstate(), np.random.get_state())
        baseline_beam = tuple(
            (entry["sequence"], entry["score"], entry["metrics"])
            for entry in baseline_result["beam"]
        )

        split = tmp_path / "split"
        constructors = []
        constructor_config_exists = []
        shutdowns = []
        proposals = [0]
        evaluations = []
        _install_fake_rna_components(
            monkeypatch,
            output=split,
            sequence=sequence,
            constructors=constructors,
            constructor_config_exists=constructor_config_exists,
            shutdowns=shutdowns,
            proposals=proposals,
            evaluations=evaluations,
        )
        original_call = Runtime.call
        crashed = [False]

        def crash_after_prediction(
            self,
            named_round,
            operation,
            request,
            *,
            work_id=None,
        ):
            result = original_call(
                self,
                named_round,
                operation,
                request,
                work_id=work_id,
            )
            if (
                named_round == "native-refinement/evaluate/1/prediction"
                and not crashed[0]
            ):
                crashed[0] = True
                raise RuntimeError("injected crash after raw prediction")
            return result

        monkeypatch.setattr(Runtime, "call", crash_after_prediction)
        with pytest.raises(RuntimeError, match="injected crash"):
            run_configured_refinement(
                cfg,
                RepeatRun(index=0, total=1, seed=71, output=split),
            )
        assert evaluations == [0, 1]

        with open_dict(cfg.search):
            cfg.search.resume = True
        resumed = run_configured_refinement(
            cfg,
            RepeatRun(index=0, total=1, seed=71, output=split),
        )
        resumed_rng = (random.getstate(), np.random.get_state())
        resumed_beam = tuple(
            (entry["sequence"], entry["score"], entry["metrics"])
            for entry in resumed["beam"]
        )
        events = [
            __import__("json").loads(line)
            for line in (split / "01_evaluations.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]

        assert evaluations == list(baseline_evaluations)
        assert proposals[0] == baseline_proposals == 2
        assert resumed_beam == baseline_beam
        assert resumed_rng[0] == baseline_rng[0]
        assert resumed_rng[1][0] == baseline_rng[1][0]
        assert np.array_equal(resumed_rng[1][1], baseline_rng[1][1])
        assert resumed_rng[1][2:] == baseline_rng[1][2:]
        for name in ("evaluation_started", "evaluation_completed"):
            identifiers = [
                event["evaluation_id"]
                for event in events
                if event["event"] == name
            ]
            assert identifiers == [0, 1, 2]
    finally:
        global_config.reset()


def _compose_paper_recipe(name, input_name=None):
    from pathlib import Path
    from omegaconf import OmegaConf
    from astrolabe import cli

    root = Path(__file__).resolve().parents[1]
    cfg = cli._compose_hydra_config(
        config_dir=root / "config", config_name="paper/" + name, overrides=["env=env"],
    )
    if input_name:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "examples/paper/inputs/nucleic" / (input_name + ".yaml")))
    return cfg


@pytest.mark.parametrize("polymer,target", [("rna", "4kqy"), ("dna", "1tup")])
def test_methods_recipe_uses_dplm_initialization_and_conditional_haddock(polymer, target):
    name = "protein_binder_" + polymer
    cfg = _compose_paper_recipe(name, name + "_" + target)
    assert cfg.initializer.mode == "dplm"
    assert cfg.paper_protocol.initial_sequence_generator is None
    assert list(cfg.mutator.mutator_terms.NAMPNN.predocker) == ["haddock_na"]
    docking = cfg.predocker.predocker_terms.haddock_na
    assert docking._type == "haddock3_nucleic"
    assert docking.skip_if_contact is True and docking.require_contact is True
    assert docking.contact_cutoff == 4.0
    assert list(docking.target_chains) == (["B"] if polymer == "rna" else ["B", "C"])
    assert cfg.evaluator.protenix.model_name == "protenix_base_default_v1.0.0"
    assert cfg.stages.complex.weighted_score_expression == "iptm"
    assert cfg.paper_protocol.representative == "accepted_complex_max_iptm"
    assert cfg.steps == 20 and cfg.paper_protocol.planned_runs_per_target == 80
    alias = _compose_paper_recipe(name + "_haddock")
    assert alias.initializer.mode == "dplm"
    assert list(alias.mutator.mutator_terms.NAMPNN.predocker) == ["haddock_na"]


@pytest.mark.parametrize("polymer", ["rna", "dna"])
def test_archived_recipe_preserves_original_nucleic_binder_initialization(polymer):
    from omegaconf import OmegaConf

    cfg = _compose_paper_recipe("protein_binder_" + polymer + "_archived")
    assert cfg.initializer.mode == "random"
    assert cfg.repeat_plan.initial_sequence.owner == "python_random"
    assert cfg.repeat_plan.initial_sequence.alphabet == "ADEFGHIKLMNPQRSTVWY"
    assert list(cfg.repeat_plan.initial_sequence.length) == [60, 80]
    assert OmegaConf.select(cfg, "mutator.mutator_terms.NAMPNN.predocker") is None
    assert cfg.paper_protocol.variant == "archived"


def test_methods_recipe_bivalent_defaults_to_random_and_preserves_archived_starts():
    cfg = _compose_paper_recipe("bivalent_rna")
    assert cfg.search.initialization == "random"
    assert cfg.seqs.A == "20@repeat repeat"
    assert cfg.seqs.B == cfg.seqs.C and len(cfg.seqs.B.lstrip("+")) == 96
    assert cfg.search.max_evaluations == 200
    assert len(cfg.search.archived_starts) == 3
    archived = _compose_paper_recipe("bivalent_rna_archived")
    assert archived.search.initialization == "archived"
    assert archived.seqs.A == "UCGGCAGCUGCCAUCUGACU@repeat repeat"
    assert [entry.source_step for entry in archived.search.archived_starts] == [269, 613, 774]
    assert archived.search.objective_weights == cfg.search.objective_weights
    assert archived.evaluator.alphafold3.num_samples == cfg.evaluator.alphafold3.num_samples == 5


@pytest.mark.parametrize("polymer,steps", [("rna", 20), ("dna", 40)])
def test_methods_recipe_aptamer_budget_and_same_state_objective(polymer, steps):
    cfg = _compose_paper_recipe(polymer + "_aptamer", polymer + "_aptamer_1urn_u1a")
    assert cfg.seqs.A == "28-36" and cfg.seqs.B.startswith("+")
    assert cfg.initializer.mode == "random"
    assert cfg.steps == steps and cfg.paper_protocol.planned_runs_per_target == 80
    assert cfg.stages.default.mutation_num == "100%"
    assert cfg.stages.default.weighted_score_expression == "aptamer_iptm + aptamer_plddt + 0.4 * aptamer_ptm"
    assert cfg.loss.loss_terms.aptamer_plddt.binder_chain == "A"
    assert cfg.paper_protocol.representative == "max_objective_first_20"
    assert list(cfg.paper_protocol.analysis_step_range) == [0, 19]


@pytest.mark.parametrize("chains", [4, 5])
def test_methods_recipe_oligomer_keeps_supplied_synchronized_start_and_budget(chains):
    cfg = _compose_paper_recipe(f"rna_oligomer_{chains}")
    assert len(cfg.seqs.A) == 87 and set(cfg.seqs.A) <= set("ACGU")
    assert list(cfg.oligo) == ["A" * chains]
    assert cfg.search.width == 8
    assert cfg.search.proposals_per_parent == 4
    assert cfg.search.max_evaluations == 1000
