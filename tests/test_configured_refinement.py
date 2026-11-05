from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf, open_dict

import run
from astrolabe import app as app_module
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design.configured_refinement import (
    _asset_manifest_identity,
    _generic_proposal_seed,
    _model_component_assets,
    _scientific_config_payload,
    _selected_asset_manifest,
    _search_settings,
    run_configured_refinement,
)
from astrolabe.design.repeats import RepeatRun
from astrolabe.design.runtime import OutcomeUnknown
from astrolabe.evaluators.utils import StandardScores
from astrolabe.stage_manager import StageManager


@pytest.fixture(autouse=True)
def _clean_globals():
    global_config.reset()
    yield
    global_config.reset()


def _base_config(root: Path) -> dict:
    return {
        "task_name": "generic_refinement",
        "out_dir": str(root / "outputs"),
        "raw_model_results_dir": str(root / "raw"),
        "seed": 71,
        "repeats": 1,
        "seqs": {"A": "AC", "B": "GG", "C": "+GG"},
        "types": {"A": "protein", "B": "protein", "C": "protein"},
        "homologies": {},
        "named_files": {},
        "debug": False,
        "structure_start": None,
        "enable_pyrosetta": False,
        "skip_complex_poses": True,
        "use_metropolis_acceptance": False,
        "per_residue_scores": None,
        "post_eval_hooks": None,
        "initializer": None,
        "mutator": {
            "mutator_terms": {
                "first": {"_type": "fake"},
                "second": {"_type": "fake"},
            },
            "selector_expression": None,
            "stack_pre_update": False,
            "stack_pre_update_max_entropy": None,
            "state_conflict_diagnostics": None,
        },
        "evaluator": {
            "evaluator_terms": {"fold": {"_type": "fake"}},
        },
        "stages": {
            "default": {
                "oligo": ["AB", "C"],
                "weighted_score_expression": "mean_plddt",
                "mutators": ["first", "second"],
                "mutation_num": 1,
                "evaluator": "fold",
                "acceptance_gate": None,
                "acceptance_threshold": None,
                "early_stopping": None,
                "prefix_early_stopping": None,
                "transition_consistency_gate": None,
            }
        },
        "loss": {"loss_terms": {}, "expression_terms": {}},
        "search": {
            "algorithm": "branching",
            "owner": "native_refinement",
            "width": 1,
            "proposals_per_parent": 1,
            "max_evaluations": 2,
            "measurement": "weighted_score",
            "summary_format": "json",
        },
    }


def _cfg(raw: dict):
    cfg = OmegaConf.create(raw)
    cfg._set_flag("allow_objects", True)
    return cfg


class _FakeLoss:
    def __init__(self, *, loss_specs=None) -> None:
        self.loss_specs = dict(loss_specs or {})
        self.loss_names = ["mean_plddt"]

    def setup(
        self,
        weighted_score_expression,
        early_stopping,
        acceptance_threshold,
        acceptance_gate=None,
        validate_config=True,
        prefix_early_stopping=None,
    ) -> None:
        del validate_config
        self.weighted_score_expression = weighted_score_expression
        self.early_stopping = early_stopping
        self.acceptance_threshold = acceptance_threshold
        self.acceptance_gate = acceptance_gate
        self.prefix_early_stopping = prefix_early_stopping

    def compute_loss(self, eval_result, runtime_config):
        del runtime_config
        values = []
        for assembly_values in eval_result.plddt.values():
            for value in assembly_values:
                values.append(float(np.asarray(value, dtype=float).mean()))
        score = float(np.mean(values))
        return score, {
            "mean_plddt": score,
            "AB": {"mean_plddt": score},
            "C": {"mean_plddt": score},
        }


class _FakeMutator:
    def __init__(self, alias: str, probe: "_Probe") -> None:
        self.alias = alias
        self.name = alias
        self.probe = probe
        self.probe.mutator_constructor_calls += 1

    def mutate(self, protomers, scores, runtime_config):
        del scores, runtime_config
        self.probe.mutator_calls.append(
            {
                "alias": self.alias,
                "instance_id": id(self),
                "A": protomers["A"].mutated_sequence,
                "B": protomers["B"].mutated_sequence,
            }
        )
        if self.alias == "first":
            segment = global_config.segments[protomers["A"].components[0]]
            target = _next_residue(segment.sequence[0])
            mutation = (0, segment.sequence[0], target, 0.0, 1.0)
        else:
            segment = global_config.segments[protomers["B"].components[0]]
            target = _next_residue(segment.sequence[0])
            mutation = (0, segment.sequence[0], target, 0.0, 1.0)
        global_config.segment_manager.add_mutations(
            [(segment_name_for(segment), mutation)],
            1,
        )

    def shutdown(self) -> None:
        self.probe.mutator_shutdowns += 1


def segment_name_for(segment) -> str:
    for name, candidate in global_config.segments.items():
        if candidate is segment:
            return name
    raise AssertionError("segment is not registered")


def _next_residue(value: str) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return alphabet[(alphabet.index(value) + 1) % len(alphabet)]


class _FakeEvaluator:
    def __init__(self, root: Path, probe: "_Probe") -> None:
        self.alias = "fold"
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.probe = probe
        self.probe.evaluator_constructor_calls += 1

    def evaluate(self, inputs, runtime_config):
        self.probe.evaluation_attempts += 1
        del runtime_config
        if self.probe.fail_evaluation:
            raise RuntimeError("fake evaluator failure")
        sequences = {
            assembly: [chain.sequence for chain in value.chains]
            for assembly, value in inputs.items()
        }
        self.probe.evaluator_calls.append(
            {
                "sequences": sequences,
                "seed": 71,
                "instance_id": id(self),
            }
        )
        score = 0.25
        if sequences["AB"][0].startswith("C"):
            score += 0.25
        elif sequences["AB"][0].startswith("D"):
            score += 0.5
        if sequences["AB"][1].startswith("T"):
            score += 0.25
        elif sequences["AB"][1].startswith("V"):
            score += 0.5
        paths = {}
        for assembly in inputs:
            path = self.root / f"model_{len(self.probe.evaluator_calls):02d}_{assembly}.cif"
            path.write_text(f"model {assembly}\n", encoding="ascii")
            paths[assembly] = path
        return StandardScores(
            names=list(inputs),
            plddt={
                assembly: [np.asarray([score, score], dtype=np.float32)]
                for assembly in inputs
            },
            complex_paths=paths,
        )

    def shutdown(self) -> None:
        self.probe.evaluator_shutdowns += 1


class _Probe:
    def __init__(self, root: Path, *, fail_evaluation=False, loss_specs=None) -> None:
        self.root = root
        self.fail_evaluation = fail_evaluation
        self.loss_specs = loss_specs or {}
        self.prepare_calls = 0
        self.prepare_apply_flags = []
        self.component_load_calls = 0
        self.loader_calls = 0
        self.mutator_calls = []
        self.evaluator_calls = []
        self.evaluation_attempts = 0
        self.mutator_constructor_calls = 0
        self.evaluator_constructor_calls = 0
        self.mutator_shutdowns = 0
        self.evaluator_shutdowns = 0
        self.mutators = {}
        self.evaluator = None
        self.stage_manager = None

    def prepare(self, args, *, apply_initialization=True):
        self.prepare_calls += 1
        self.prepare_apply_flags.append(apply_initialization)
        manager = SegmentManager()
        global_config.init_const("segments", {})
        global_config.init_const("segment_manager", manager)
        global_config.init_const("named_files", {})
        global_config.init_const("out_dir", args.out_dir)
        global_config.init_const("raw_model_results_dir", args.raw_model_results_dir)
        global_config.init_const("skip_complex_poses", bool(args.skip_complex_poses))
        global_config.init_const("enable_pyrosetta", bool(args.enable_pyrosetta))
        global_config.init_const("pre_eval_hooks", None)

        protomers = {}
        mutable_protomers = {}
        for name, statement in args.seqs.items():
            protomer = Protomer(
                name,
                str(statement),
                str(args.types.get(name, "protein")),
                getattr(args, "homologies", {}).get(name),
            )
            protomers[name] = protomer
            if protomer.type.value in {"p", "r", "d"}:
                mutable_protomers[name] = protomer
        global_config.init_const("protomers", protomers)
        global_config.init_const("mutable_protomers", mutable_protomers)

        return run.PreparedPipelineState(
            runtime_context=global_config.build_runtime_context(),
            initializer_mode="random",
            initializer_result=None if not apply_initialization else object(),
        )

    def load(self, args, prepared):
        self.component_load_calls += 1
        if prepared.initializer_mode != "random":
            raise AssertionError("unexpected prepared initializer")
        loss = _FakeLoss(loss_specs=self.loss_specs)
        stage_manager = StageManager(
            args.stages,
            loss_computer=loss,
            runtime_context=prepared.runtime_context,
        )
        self.mutators = {
            alias: _FakeMutator(alias, self)
            for alias in args.mutator.enabled_aliases
        }
        self.evaluator = _FakeEvaluator(self.root, self)
        self.stage_manager = stage_manager
        global_config.init_const("loss_computer", loss)
        global_config.init_const("stage_manager", stage_manager)
        global_config.init_const("mutators", self.mutators)
        global_config.init_const("evaluators", {"fold": self.evaluator})
        return loss, stage_manager, {}


def _install_probe(monkeypatch, probe: _Probe) -> None:
    monkeypatch.setattr(
        run,
        "prepare_pipeline_state",
        probe.prepare,
    )
    monkeypatch.setattr(
        run,
        "load_pipeline_components",
        probe.load,
    )
    def load_run_module():
        probe.loader_calls += 1
        return run

    monkeypatch.setattr(app_module, "_load_run_module", load_run_module)


def _events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_generic_builder_reuses_stack_and_evaluator_across_complete_assignments(
    tmp_path,
    monkeypatch,
):
    raw = _base_config(tmp_path)
    raw["search"]["max_evaluations"] = 3
    probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, probe)
    repeat = RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs")

    summary = run_configured_refinement(_cfg(raw), repeat)

    output = Path(summary["out_dir"])
    assert probe.loader_calls == 1
    assert probe.prepare_calls == 1
    assert probe.prepare_apply_flags == [True]
    assert probe.component_load_calls == 1
    assert summary["components"]["load_order"] == ["mutator", "evaluator"]
    assert summary["components"]["mutators"] == ["first", "second"]
    assert summary["components"]["evaluator"] == "fold"
    assert probe.mutator_constructor_calls == 2
    assert probe.evaluator_constructor_calls == 1
    assert len(probe.mutator_calls) == 4
    assert [call["alias"] for call in probe.mutator_calls] == [
        "first",
        "second",
        "first",
        "second",
    ]
    assert len({call["instance_id"] for call in probe.mutator_calls[::2]}) == 1
    assert len({call["instance_id"] for call in probe.mutator_calls[1::2]}) == 1
    assert len(probe.evaluator_calls) == 3
    assert len({call["instance_id"] for call in probe.evaluator_calls}) == 1
    assert all(call["seed"] == 71 for call in probe.evaluator_calls)

    root, child, grandchild = probe.evaluator_calls
    assert root["sequences"] == {"AB": ["AC", "GG"], "C": ["GG"]}
    assert child["sequences"] == {"AB": ["CC", "HG"], "C": ["GG"]}
    assert grandchild["sequences"] == {"AB": ["DC", "IG"], "C": ["GG"]}
    assert probe.mutator_shutdowns == 2
    assert probe.evaluator_shutdowns == 1

    assert summary["routing"] == {
        "algorithm": "branching",
        "owner": "native_refinement",
    }
    assert summary["seed"] == 71
    assert summary["repeats"] == 1
    assert summary["search"]["evaluations"] == 3
    assert summary["search"]["proposals"] == 2
    assert summary["search"]["proposal_cap"] == 2
    assignments = summary["beam"][0]["candidate"]["assignments"]
    assert assignments["A_1"]["sequence"] == "DC"
    assert assignments["B_1"]["sequence"] == "IG"
    assert assignments["C_1"]["sequence"] == "GG"
    assert set(summary["beam"][0]["structure_paths"]) == {"AB", "C"}
    assert summary["beam"][0]["loss"]["mean"]["mean_plddt"] == pytest.approx(0.75)
    assert summary["beam"][0]["loss"]["by_assembly"]["AB"]["mean_plddt"] == pytest.approx(
        0.75
    )
    assert summary["evaluations"][2]["candidate_identity"] == (
        summary["beam"][0]["candidate"]["identity"]
    )
    assert summary["evaluations"][2]["parent_id"] == 1
    assert len({row["system_identity"] for row in summary["evaluations"]}) == 1
    assert "candidate" not in summary["evaluations"][2]
    assert isinstance(summary["search"]["stop_reason"], str)

    saved_text = (output / "config.yaml").read_text(encoding="utf-8")
    saved = OmegaConf.load(output / "config.yaml")
    safe_saved = yaml.safe_load(saved_text)
    assert saved.seed == 71
    assert saved.repeats == 1
    assert saved.out_dir == str(output)
    assert safe_saved["out_dir"] == str(output)
    assert saved.search.algorithm == "branching"
    assert saved.search.owner == "native_refinement"
    assert saved.search.measurement == "weighted_score"
    assert saved.search.resume is False
    assert saved.search.width == 1
    assert saved.search.proposals_per_parent == 1
    assert saved.search.max_evaluations == 3
    assert saved.search.proposal_cap == 2

    events = _events(output / "01_evaluations.jsonl")
    proposal = next(item for item in events if item["event"] == "proposal_started")
    assert proposal["seed"] == _generic_proposal_seed(71, 1)
    assert proposal["seed"] != (71 * 100003 + 1) % 2147483647
    assert (output / "03_final_beam.json").is_file()
    assert json.loads((output / "provenance.json").read_text())["state"] == "completed"


def test_completed_outdir_resume_uses_cached_control_without_model_loads(
    tmp_path,
    monkeypatch,
):
    raw = _base_config(tmp_path)
    first_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, first_probe)
    first = run_configured_refinement(
        _cfg(raw),
        RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
    )

    output = Path(first["out_dir"])
    config_path = output / "config.yaml"
    provenance_path = output / "provenance.json"
    config_bytes = config_path.read_bytes()
    provenance_bytes = provenance_path.read_bytes()
    journal_bytes = (output / "01_evaluations.jsonl").read_bytes()
    resume_cfg = OmegaConf.load(config_path)
    with open_dict(resume_cfg):
        resume_cfg.out_dir = str(output)
    with open_dict(resume_cfg.search):
        resume_cfg.search.resume = True

    resumed_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, resumed_probe)
    resumed = run_configured_refinement(
        resume_cfg,
        RepeatRun(index=0, total=1, seed=71, output=output),
    )

    assert resumed_probe.prepare_calls == 1
    assert resumed_probe.prepare_apply_flags == [False]
    assert resumed_probe.component_load_calls == 0
    assert resumed_probe.mutator_constructor_calls == 0
    assert resumed_probe.evaluator_constructor_calls == 0
    assert resumed_probe.mutator_calls == []
    assert resumed_probe.evaluator_calls == []
    assert resumed["search"] == first["search"]
    assert resumed["beam"] == first["beam"]
    assert config_path.read_bytes() == config_bytes
    assert provenance_path.read_bytes() == provenance_bytes
    assert (output / "01_evaluations.jsonl").read_bytes() == journal_bytes
    assert [
        path for path in output.parent.iterdir() if path.is_dir()
    ] == [output]


def test_resume_rejects_changed_scientific_config_before_models_or_overwrites(
    tmp_path,
    monkeypatch,
):
    first_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, first_probe)
    first = run_configured_refinement(
        _cfg(_base_config(tmp_path)),
        RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
    )

    output = Path(first["out_dir"])
    config_path = output / "config.yaml"
    provenance_path = output / "provenance.json"
    config_bytes = config_path.read_bytes()
    provenance_bytes = provenance_path.read_bytes()
    resume_cfg = OmegaConf.load(config_path)
    with open_dict(resume_cfg):
        resume_cfg.out_dir = str(output)
    with open_dict(resume_cfg.search):
        resume_cfg.search.resume = True
        resume_cfg.search.max_evaluations = 3

    resumed_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, resumed_probe)
    with pytest.raises(ValueError, match="Scientific configuration changed"):
        run_configured_refinement(
            resume_cfg,
            RepeatRun(index=0, total=1, seed=71, output=output),
        )

    assert resumed_probe.prepare_calls == 0
    assert resumed_probe.component_load_calls == 0
    assert config_path.read_bytes() == config_bytes
    assert provenance_path.read_bytes() == provenance_bytes


@pytest.mark.parametrize(
    "configure",
    (
        lambda raw, value: raw["evaluator"]["evaluator_terms"]["fold"].update(
            {"checkpoint": value}
        ),
        lambda raw, value: raw["mutator"]["mutator_terms"]["first"].update(
            {"runner": {"checkpoint": value}}
        ),
        lambda raw, value: raw.update(
            {"initializer": {"mode": "aa_set", "aa_set": value}}
        ),
        lambda raw, value: (
            raw["stages"]["default"].update(
                {"weighted_score_expression": "objective"}
            ),
            raw["loss"]["loss_terms"].update(
                {"objective": {"_type": "ptm", "reference": value}}
            ),
        ),
    ),
)
def test_scientific_identity_covers_each_selected_component_config(
    tmp_path,
    configure,
):
    first = _base_config(tmp_path)
    second = deepcopy(first)
    for raw in (first, second):
        raw["evaluator"]["enabled_aliases"] = ["fold"]
        raw["mutator"]["enabled_aliases"] = ["first", "second"]
    configure(first, "/models/first")
    configure(second, "/models/second")

    assert (
        _scientific_config_payload(_cfg(first))
        != _scientific_config_payload(_cfg(second))
    )


def test_scientific_identity_keeps_inputs_and_drops_lifecycle_paths(tmp_path):
    first = _base_config(tmp_path)
    second = deepcopy(first)
    second["out_dir"] = str(tmp_path / "another-output")
    second["raw_model_results_dir"] = str(tmp_path / "another-raw")
    second["task_name"] = "another_task"
    second["logging"] = {"console_level": "DEBUG"}
    second["evaluator"]["scheduling"] = {"cpu_postprocess_workers": 7}
    second["search"]["resume"] = True
    second["search"]["summary_format"] = None

    assert (
        _scientific_config_payload(_cfg(first))
        == _scientific_config_payload(_cfg(second))
    )

    second["named_files"] = {"motif": str(tmp_path / "fixed-input.pdb")}
    assert (
        _scientific_config_payload(_cfg(first))
        != _scientific_config_payload(_cfg(second))
    )


def test_container_af3_cold_scientific_identity_ignores_inactive_rf3(
    tmp_path,
    monkeypatch,
):
    from astrolabe.argparser import parse_arguments
    from astrolabe.configuration import load_composed_config

    monkeypatch.delenv("GH_WEIGHT_RF3", raising=False)
    model_dir = tmp_path / "af3-assets"
    model_dir.mkdir()
    selected = model_dir / "af3.bin"
    selected.write_bytes(b"selected AF3 asset v1")

    cfg = load_composed_config(
        Path(__file__).resolve().parents[1]
        / "config"
        / "container_af3_smoke.yaml"
    )
    cfg.out_dir = str(tmp_path / "runs")
    cfg.seed = 17
    cfg.weight_path.alphafold3 = str(model_dir)
    args = parse_arguments(cfg)

    payload = _scientific_config_payload(args)
    assert "rf3" not in payload
    assert payload["evaluator"]["alphafold3"]["model_dir"] == str(model_dir)

    first = _asset_manifest_identity(_selected_asset_manifest(args))
    selected.write_bytes(b"selected AF3 asset v2")
    second = _asset_manifest_identity(_selected_asset_manifest(args))
    assert first != second


def test_resume_rejects_changed_selected_model_config_before_models(
    tmp_path,
    monkeypatch,
):
    raw = _base_config(tmp_path)
    raw["evaluator"]["evaluator_terms"]["fold"]["checkpoint"] = (
        "/models/original.pt"
    )
    first_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, first_probe)
    first = run_configured_refinement(
        _cfg(raw),
        RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
    )

    output = Path(first["out_dir"])
    resume_cfg = OmegaConf.load(output / "config.yaml")
    with open_dict(resume_cfg):
        resume_cfg.out_dir = str(output)
    with open_dict(resume_cfg.evaluator.evaluator_terms.fold):
        resume_cfg.evaluator.evaluator_terms.fold.checkpoint = (
            "/models/changed.pt"
        )
    with open_dict(resume_cfg.search):
        resume_cfg.search.resume = True

    resumed_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, resumed_probe)
    with pytest.raises(ValueError, match="Scientific configuration changed"):
        run_configured_refinement(
            resume_cfg,
            RepeatRun(index=0, total=1, seed=71, output=output),
        )

    assert resumed_probe.prepare_calls == 0
    assert resumed_probe.component_load_calls == 0


def _asset_config(root: Path) -> dict:
    raw = _base_config(root)
    model_dir = root / "af3-model"
    (model_dir / "nested").mkdir(parents=True)
    (model_dir / "params.npz").write_bytes(b"af3-params-v1")
    (model_dir / "nested" / "confidence.npz").write_bytes(b"af3-confidence-v1")
    raw["mutator"]["mutator_terms"] = {"first": {"_type": "MCMC"}}
    raw["stages"]["default"]["mutators"] = ["first"]
    raw["evaluator"]["evaluator_terms"]["fold"]["_type"] = "Alphafold3"
    raw["evaluator"]["alphafold3"] = {"model_dir": str(model_dir)}
    return raw


def test_selected_asset_manifest_hashes_af3_model_dir_and_marks_mcmc_weightless(
    tmp_path,
):
    raw = _asset_config(tmp_path)
    manifest = _selected_asset_manifest(_cfg(raw))
    records = manifest.record()

    assert records["complete"] is True
    assert {
        (item["role"], item["type"], item["state"])
        for item in records["components"]
    } >= {
        ("evaluator:fold", "alphafold3", "hashed"),
        ("mutator:first", "mcmc", "weight_free"),
    }
    assert [
        Path(item["path"]).name for item in records["files"]
    ] == ["confidence.npz", "params.npz"]
    assert all(len(item["sha256"]) == 64 for item in records["files"])


def test_rosetta_crystal_reference_and_params_are_content_identified(tmp_path):
    params = tmp_path / "ligand.params"
    reference = tmp_path / "reference.pdb"
    alternate = tmp_path / "alternate.pdb"
    for path in (params, reference, alternate):
        path.write_text(path.name)
    mapping = {"states": {"state": {
        "params": str(params), "reference": str(reference),
        "references": {"AB": str(alternate)},
    }}}
    component, files = _model_component_assets("evaluator:fold", "rosetta_crystal", mapping)
    assert component.state == "hashed"
    assert {Path(item.path) for item in files} == {params, reference, alternate}
    before = {item.path: item.sha256 for item in files}
    reference.write_text("changed reference coordinates")
    _, changed = _model_component_assets("evaluator:fold", "rosetta_crystal", mapping)
    assert {item.path: item.sha256 for item in changed}[str(reference)] != before[str(reference)]
    with pytest.raises(ValueError, match="lacks params or reference"):
        _model_component_assets("evaluator:fold", "rosetta_crystal", {"states": {"s": {}}})


def test_resume_rejects_same_path_asset_content_change_before_parse_or_models(
    tmp_path,
    monkeypatch,
):
    raw = _asset_config(tmp_path)
    first_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, first_probe)
    first = run_configured_refinement(
        _cfg(raw),
        RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
    )

    output = Path(first["out_dir"])
    resume_cfg = OmegaConf.load(output / "config.yaml")
    with open_dict(resume_cfg):
        resume_cfg.out_dir = str(output)
    with open_dict(resume_cfg.search):
        resume_cfg.search.resume = True

    resumed_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, resumed_probe)
    resumed = run_configured_refinement(
        resume_cfg,
        RepeatRun(index=0, total=1, seed=71, output=output),
    )
    assert resumed == first
    assert resumed_probe.prepare_calls == 1
    assert resumed_probe.component_load_calls == 0

    config_bytes = (output / "config.yaml").read_bytes()
    provenance_bytes = (output / "provenance.json").read_bytes()
    (tmp_path / "af3-model" / "params.npz").write_bytes(b"af3-params-v2")

    changed_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, changed_probe)
    with pytest.raises(ValueError, match="asset content changed"):
        run_configured_refinement(
            resume_cfg,
            RepeatRun(index=0, total=1, seed=71, output=output),
        )

    assert changed_probe.prepare_calls == 0
    assert changed_probe.component_load_calls == 0
    assert (output / "config.yaml").read_bytes() == config_bytes
    assert (output / "provenance.json").read_bytes() == provenance_bytes


def test_resume_rejects_asset_manifest_that_cannot_prove_model_weights(
    tmp_path,
    monkeypatch,
):
    raw = _asset_config(tmp_path)
    raw["evaluator"]["alphafold3"] = {}
    probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, probe)

    with pytest.raises(ValueError, match="alphafold3 model_dir"):
        run_configured_refinement(
            _cfg(raw),
            RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
        )

    assert probe.prepare_calls == 0
    assert probe.component_load_calls == 0


def test_resume_rejects_saved_incomplete_asset_manifest_before_parse(
    tmp_path,
    monkeypatch,
):
    raw = _base_config(tmp_path)
    raw["evaluator"]["evaluator_terms"]["fold"]["_type"] = "unverified_model"
    first_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, first_probe)
    first = run_configured_refinement(
        _cfg(raw),
        RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
    )

    output = Path(first["out_dir"])
    resume_cfg = OmegaConf.load(output / "config.yaml")
    with open_dict(resume_cfg):
        resume_cfg.out_dir = str(output)
    with open_dict(resume_cfg.search):
        resume_cfg.search.resume = True

    resumed_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, resumed_probe)
    with pytest.raises(ValueError, match="selected assets cannot be content-verified"):
        run_configured_refinement(
            resume_cfg,
            RepeatRun(index=0, total=1, seed=71, output=output),
        )

    assert resumed_probe.prepare_calls == 0
    assert resumed_probe.component_load_calls == 0


def test_outdir_resume_after_completed_child_measurement_does_not_replay_models(
    tmp_path,
    monkeypatch,
):
    from astrolabe.design.native_refinement_runtime import DurableNativeRefinementOwner

    original_terminal = DurableNativeRefinementOwner._terminal
    crashed = False

    def crash_after_child_measurement(owner, job):
        nonlocal crashed
        terminal = original_terminal(owner, job)
        if not crashed and job.phase == "evaluate" and job.id == 1:
            crashed = True
            raise RuntimeError("injected configured-refinement cut")
        return terminal

    monkeypatch.setattr(
        DurableNativeRefinementOwner,
        "_terminal",
        crash_after_child_measurement,
    )
    first_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, first_probe)
    with pytest.raises(RuntimeError, match="injected configured-refinement cut"):
        run_configured_refinement(
            _cfg(_base_config(tmp_path)),
            RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
        )

    [output] = list((tmp_path / "outputs").glob("*/001_run"))
    resume_cfg = OmegaConf.load(output / "config.yaml")
    with open_dict(resume_cfg):
        resume_cfg.out_dir = str(output)
    with open_dict(resume_cfg.search):
        resume_cfg.search.resume = True

    resumed_probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, resumed_probe)
    resumed = run_configured_refinement(
        resume_cfg,
        RepeatRun(index=0, total=1, seed=71, output=output),
    )

    assert resumed_probe.prepare_calls == 1
    assert resumed_probe.prepare_apply_flags == [False]
    assert resumed_probe.component_load_calls == 0
    assert resumed_probe.mutator_constructor_calls == 0
    assert resumed_probe.evaluator_constructor_calls == 0
    assert resumed_probe.evaluator_calls == []
    assert resumed["search"]["evaluations"] == 2
    assert resumed["search"]["proposals"] == 1
    assert json.loads((output / "provenance.json").read_text())["state"] == "completed"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"structure_start": {"mode": "soft_select"}}, "structure_start"),
        ({"enable_pyrosetta": True}, "live PyRosetta poses"),
        (
            {"mutator": {"stack_pre_update": True}},
            "stack_pre_update",
        ),
        (
            {"initializer": {"mode": "dplm"}},
            "model-backed initialization",
        ),
        (
            {"stages": {"default": {"acceptance_threshold": "plddt > 0.8"}}},
            "stage controls",
        ),
        (
            {
                "stages": {
                    "stage_1": {
                        "_entrance": "True",
                        "oligo": ["AB"],
                        "weighted_score_expression": "mean_plddt",
                        "mutators": ["first"],
                        "mutation_num": 1,
                        "evaluator": "fold",
                    }
                }
            },
            "exactly one static stage",
        ),
    ],
)
def test_preflight_rejects_unpreserved_controls_before_initialization(
    tmp_path,
    monkeypatch,
    mutation,
    message,
):
    raw = _base_config(tmp_path)
    for key, value in mutation.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key].update(deepcopy(value))
        else:
            raw[key] = deepcopy(value)
    if "stages" in mutation and "default" not in mutation["stages"]:
        raw["stages"].update(deepcopy(mutation["stages"]))

    probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, probe)
    with pytest.raises(ValueError, match=message):
        run_configured_refinement(
            _cfg(raw),
            RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
        )
    assert probe.prepare_calls == 0
    assert probe.component_load_calls == 0
    assert probe.evaluator_calls == []
    assert not (tmp_path / "outputs").exists()


def test_loss_pose_requirement_fails_before_initialization_or_output(
    tmp_path,
    monkeypatch,
):
    raw = _base_config(tmp_path)
    raw["loss"]["loss_terms"] = {"pose_term": {"_type": "sap_score"}}
    raw["stages"]["default"]["weighted_score_expression"] = "pose_term"
    probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, probe)

    with pytest.raises(ValueError, match="requires live complex poses"):
        run_configured_refinement(
            _cfg(raw),
            RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
        )

    assert probe.prepare_calls == 0
    assert probe.component_load_calls == 0
    assert probe.evaluator_calls == []
    assert not (tmp_path / "outputs").exists()


def test_prediction_failure_propagates_without_retry_and_resources_close(
    tmp_path,
    monkeypatch,
):
    probe = _Probe(tmp_path / "raw" / "models", fail_evaluation=True)
    _install_probe(monkeypatch, probe)

    with pytest.raises(OutcomeUnknown) as raised:
        run_configured_refinement(
            _cfg(_base_config(tmp_path)),
            RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
        )
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "fake evaluator failure" in str(raised.value.__cause__)

    assert probe.prepare_calls == 1
    assert probe.component_load_calls == 1
    assert probe.evaluation_attempts == 1
    assert probe.evaluator_calls == []
    assert probe.mutator_shutdowns == 2
    assert probe.evaluator_shutdowns == 1
    [output] = list((tmp_path / "outputs").glob("*/001_run"))
    events = _events(output / "01_evaluations.jsonl")
    assert [item["event"] for item in events] == [
        "search_started",
        "evaluation_started",
        "evaluation_failed",
        "search_failed",
    ]
    assert not (output / "03_final_beam.json").exists()
    assert json.loads((output / "provenance.json").read_text())["state"] == "failed"


def test_search_fields_are_required_instead_of_defaulted(tmp_path, monkeypatch):
    raw = _base_config(tmp_path)
    del raw["search"]["max_evaluations"]
    probe = _Probe(tmp_path / "raw" / "models")
    _install_probe(monkeypatch, probe)

    with pytest.raises(ValueError, match=r"max_evaluations.*positive integer"):
        run_configured_refinement(
            _cfg(raw),
            RepeatRun(index=0, total=1, seed=71, output=tmp_path / "outputs"),
        )
    assert probe.prepare_calls == 0
    assert probe.component_load_calls == 0


def test_proposal_cap_is_explicit_and_defaults_to_one_fewer_evaluation(tmp_path):
    cfg = _cfg(_base_config(tmp_path))
    assert _search_settings(cfg).proposal_cap == 1
    with open_dict(cfg.search):
        cfg.search.proposal_cap = 3
    assert _search_settings(cfg).proposal_cap == 3
    with open_dict(cfg.search):
        cfg.search.proposal_cap = 0
    with pytest.raises(ValueError, match="proposal_cap.*positive integer"):
        _search_settings(cfg)


def test_initialize_pipeline_components_preserves_prepare_then_load_wrapper(
    monkeypatch,
):
    args = object()
    prepared = object()
    expected = (object(), object(), object())
    calls = []

    def prepare(value):
        calls.append(("prepare", value))
        return prepared

    def load(value, state):
        calls.append(("load", value, state))
        return expected

    monkeypatch.setattr(run, "prepare_pipeline_state", prepare)
    monkeypatch.setattr(run, "load_pipeline_components", load)

    assert run.initialize_pipeline_components(args) == expected
    assert calls == [("prepare", args), ("load", args, prepared)]
