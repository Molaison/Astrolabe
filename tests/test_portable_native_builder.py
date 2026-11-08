from __future__ import annotations

from types import ModuleType, SimpleNamespace
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError
import pytest

import run as run_module
from astrolabe import run_config_lifecycle
from astrolabe.configuration import load_composed_config
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design import portable_native_builder as builder_module
from astrolabe.design.native_mutation_state import (
    NATIVE_MUTATION_CONTEXT_SCHEMA,
    MutationContextPolicy,
)
from astrolabe.design.native_registry import (
    NativeComponentContext,
    NativeOperationHooks,
)
from astrolabe.design.portable_native_builder import (
    UnsupportedPortableNativeProfile,
    af3_mcmc_single_child_template,
    build_portable_native_recipe,
    content_manifest,
    read_recovery_policy,
    scientific_config_payload,
    source_manifest,
    validate_portable_native_profile,
)
from astrolabe.design.portable_native_flow import PortableNativeMath
from astrolabe.design.portable_native_math import build_portable_native_math
from astrolabe.design.runtime import ContractConflict
from astrolabe.design.types import key_for
from astrolabe.evaluators.registry import EvaluatorRegistry
from astrolabe.evaluators.utils import StandardScores
from astrolabe.initializers.base import InitializerResult
from astrolabe.mutators.registry import MutatorRegistry


@pytest.fixture(autouse=True)
def _clean_global_config():
    global_config.reset()
    yield
    global_config.reset()


class _ColdFailureEvaluator:
    def __init__(self) -> None:
        self.loads = 0

    def evaluate(self, *_args, **_kwargs):
        raise AssertionError("builder tests must not evaluate")

    def shutdown(self):
        pass


class _ColdFailureMutator:
    def __init__(self) -> None:
        self.loads = 0

    def mutate(self, *_args, **_kwargs):
        raise AssertionError("builder tests must not mutate")

    def shutdown(self):
        pass


def _evaluator_hook(_model, _request):
    return None


def _mutator_hook(
    _model,
    runtime_config,
    *,
    request,
    sample,
    step,
    work_root,
):
    del request, sample, step, work_root
    return runtime_config


def _math_factory(context):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("builder tests must not execute decision math")

    return PortableNativeMath(
        identity=key_for(
            (
                "test/portable-native-builder/math/v1",
                context.contract.config_identity,
            )
        ),
        finalize_root=forbidden,
        decide_step=forbidden,
    )


def _args(tmp_path: Path, *, recovery_resume: bool = False):
    return OmegaConf.create(
        {
            "task_name": "portable-builder-test",
            "out_dir": str(tmp_path / "run"),
            "raw_model_results_dir": str(tmp_path / "raw"),
            "seed": 17,
            "repeats": 1,
            "steps": 2,
            "types": {"A": "p"},
            "seqs": {"A": "AAAA"},
            "named_files": {},
            "structure_start": None,
            "initial_temperature": 0.05,
            "use_metropolis_acceptance": False,
            "enable_pyrosetta": False,
            "skip_complex_poses": True,
            "allow_mmcif_pdb_fallback": False,
            "obabel_path": None,
            "molfile_to_params": None,
            "dssp_path": None,
            "usalign_path": None,
            "ananas_path": None,
            "ligand_resnames": None,
            "predocker": {"predocker_terms": {}},
            "per_residue_scores": {
                "enabled": False,
                "terms": {},
                "mutator_score_terms": {},
                "allow_missing_mutator_score_terms": False,
            },
            "pre_eval_hooks": {},
            "post_eval_hooks": {},
            "initializer": {"mode": "random", "pssm": None},
            "evaluator": {
                "enabled_aliases": ["fold"],
                "evaluator_terms": {
                    "fold": {"_type": "portable_builder_evaluator"},
                },
            },
            "mutator": {
                "enabled_aliases": ["mutate"],
                "selector_expression": None,
                "selectors": {"mutate": "None"},
                "state_conflict_diagnostics": {
                    "enabled": False,
                    "output_dir": None,
                },
                "stack_pre_update": False,
                "stack_pre_update_max_entropy": None,
                "mutator_terms": {
                    "mutate": {"_type": "MCMC"},
                },
            },
            "loss": {
                "loss_terms": {},
                "expression_terms": {},
            },
            "stages": {
                "default": {
                    "evaluator": "fold",
                    "oligo": ["A"],
                    "mutators": ["mutate"],
                    "mutation_num": 1,
                    "weighted_score_expression": "plddt",
                    "acceptance_threshold": None,
                    "early_stopping": None,
                }
            },
            "repeat_plan": {"count": 1},
            "search": None,
            "unused_rf3": {},
            "rf3": {
                "repo_path": "${unused_rf3.repo_path}",
            },
            "recovery": {
                "mode": "durable",
                "resume": recovery_resume,
            },
        }
    )


def _initialize_legacy() -> None:
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    manager = SegmentManager()
    global_config.init_const("segment_manager", manager)
    protomer = Protomer("A", "AAAA", "p")
    global_config.protomers["A"] = protomer
    global_config.mutable_protomers["A"] = protomer


def _install_run_module(
    monkeypatch,
    tmp_path: Path,
    *,
    calls: list,
    initializer_calls: list,
    forbidden_calls: list | None = None,
):
    forbidden_calls = [] if forbidden_calls is None else forbidden_calls
    evaluator_loads = _ColdFailureEvaluator()
    mutator_loads = _ColdFailureMutator()

    def evaluator_factory():
        evaluator_loads.loads += 1
        return evaluator_loads

    def mutator_factory():
        mutator_loads.loads += 1
        return mutator_loads

    EvaluatorRegistry.register("portable_builder_evaluator", evaluator_factory)
    MutatorRegistry.register("portable_builder_mutator", mutator_factory)

    def parse_arguments(
        cfg,
        *,
        existing_run_dir=None,
        persist_config=True,
    ):
        calls.append(
            (
                "parse_arguments",
                existing_run_dir,
                persist_config,
            )
        )
        return cfg

    def prepare_pipeline_state(args, *, apply_initialization=True):
        calls.append(("prepare_pipeline_state", apply_initialization))
        initializer_calls.append(apply_initialization)
        _initialize_legacy()
        result = (
            InitializerResult(mode="random", sequences={"A": "AAAA"})
            if apply_initialization
            else None
        )
        if apply_initialization:
            global_config.init_const("initializer_result", result)
        return run_module.PreparedPipelineState(
            runtime_context=global_config.build_runtime_context(),
            initializer_mode="random",
            initializer_result=result,
            initializer_pending=False,
        )

    original_prepare_pipeline_math = run_module.prepare_pipeline_math

    def prepare_pipeline_math(args, prepared):
        calls.append(("prepare_pipeline_math",))
        return original_prepare_pipeline_math(args, prepared)

    def forbidden(name):
        def fail(*_args, **_kwargs):
            forbidden_calls.append(name)
            raise AssertionError(f"cold builder called {name}")

        return fail

    monkeypatch.setattr(run_module, "build_mutators", forbidden("build_mutators"))
    monkeypatch.setattr(
        run_module,
        "construct_evaluators",
        forbidden("construct_evaluators"),
    )
    monkeypatch.setattr(
        run_module,
        "initialize_predockers",
        forbidden("initialize_predockers"),
    )
    monkeypatch.setattr(run_module, "parse_arguments", parse_arguments)
    monkeypatch.setattr(
        run_module,
        "prepare_pipeline_state",
        prepare_pipeline_state,
    )
    monkeypatch.setattr(
        run_module,
        "prepare_pipeline_math",
        prepare_pipeline_math,
    )
    return evaluator_loads, mutator_loads


def _hooks():
    mutation_context = MutationContextPolicy(
        "test/portable-builder/mutation-context/v1",
        NATIVE_MUTATION_CONTEXT_SCHEMA,
    )
    return {
        "fold": NativeOperationHooks(
            configure=_evaluator_hook,
            identity="test/portable-builder/fold/v1",
        ),
        "mutate": NativeOperationHooks(
            configure=_mutator_hook,
            identity="test/portable-builder/mutate/v1",
            mutation_context=mutation_context,
        ),
    }


def _build(
    monkeypatch,
    tmp_path: Path,
    *,
    cfg=None,
    calls=None,
    initializer_calls=None,
    source_paths=None,
    asset_paths=None,
    operation_hooks=None,
    math_factory=_math_factory,
    forbidden_calls=None,
):
    calls = [] if calls is None else calls
    initializer_calls = [] if initializer_calls is None else initializer_calls
    cfg = _args(tmp_path) if cfg is None else cfg
    loads = _install_run_module(
        monkeypatch,
        tmp_path,
        calls=calls,
        initializer_calls=initializer_calls,
        forbidden_calls=forbidden_calls,
    )
    recipe = build_portable_native_recipe(
        cfg,
        math_factory=math_factory,
        operation_hooks=operation_hooks,
        source_paths=source_paths,
        asset_paths=asset_paths,
        run_module=run_module,
    )
    return recipe, loads


def _select_asset(cfg, path: Path):
    cfg.evaluator.evaluator_terms.fold.checkpoint = str(path)
    return cfg


def test_recovery_contract_is_explicit_and_closed():
    assert read_recovery_policy(
        OmegaConf.create(
            {"recovery": {"mode": "durable", "resume": False}}
        )
    ).resume is False
    with pytest.raises(UnsupportedPortableNativeProfile, match="explicit"):
        read_recovery_policy(OmegaConf.create({}))
    with pytest.raises(UnsupportedPortableNativeProfile, match="unknown"):
        read_recovery_policy(
            OmegaConf.create(
                {
                    "recovery": {
                        "mode": "durable",
                        "resume": False,
                        "fallback": True,
                    }
                }
            )
        )


def test_static_preflight_rejects_before_prepare_or_component_load(
    monkeypatch,
    tmp_path: Path,
):
    cfg = _args(tmp_path)
    cfg.structure_start = {"path": str(tmp_path / "start.cif")}
    calls = []
    initializer_calls = []
    _install_run_module(
        monkeypatch,
        tmp_path,
        calls=calls,
        initializer_calls=initializer_calls,
    )
    with pytest.raises(
        UnsupportedPortableNativeProfile,
        match="structure_start",
    ):
        build_portable_native_recipe(
            cfg,
            math_factory=_math_factory,
            operation_hooks=_hooks(),
            run_module=run_module,
        )
    assert calls == [("parse_arguments", None, True)]
    assert initializer_calls == []


def test_builder_prepares_math_cold_and_persists_root_receipt(
    monkeypatch,
    tmp_path: Path,
):
    asset = tmp_path / "selected.pt"
    asset.write_bytes(b"asset-v1")
    cfg = _select_asset(_args(tmp_path), asset)
    calls = []
    initializer_calls = []
    forbidden_calls = []
    recipe, (evaluator, mutator) = _build(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        calls=calls,
        initializer_calls=initializer_calls,
        asset_paths={"weights": asset},
        forbidden_calls=forbidden_calls,
    )
    try:
        assert calls == [
            ("parse_arguments", None, True),
            ("prepare_pipeline_state", True),
            ("prepare_pipeline_math",),
        ]
        assert forbidden_calls == []
        assert initializer_calls == [True]
        assert evaluator.loads == 0
        assert mutator.loads == 0
        assert recipe.rounds == 1
        assert recipe.flow.initial == recipe.flow.system.edit(
            recipe.flow.initial,
            recipe.flow.initial.assignments,
            data=recipe.flow.data,
        )
        assert callable(recipe.flow.evaluator.configure)
        assert callable(recipe.flow.evaluator.native_invoke)
        assert callable(recipe.flow.mutator.configure)
        assert callable(recipe.flow.mutator.native_invoke)
        assert recipe.flow.mutator.mutation_context is not None
        assert recipe.flow.evaluator.configure_identity
        assert recipe.flow.evaluator.native_invoke_identity
        assert recipe.flow.mutator.native_invoke_identity
        assert recipe.contract.system_identity == recipe.flow.system.identity
        assert any(
            item["sha256"] == hashlib_file(asset)
            for item in recipe.asset_manifest["files"]
        )
        assert recipe.supplemental_asset_manifest[0].sha256 == hashlib_file(asset)
        prepared = (
            Path(recipe.args.out_dir)
            / ".framework"
            / "portable-native"
            / "prepared.json"
        )
        assert prepared.is_file()
        assert (Path(recipe.args.out_dir) / ".framework" / "journal.sqlite").is_file()
    finally:
        recipe.close()


def test_selected_component_config_excludes_process_bookkeeping(tmp_path: Path):
    cfg = _args(tmp_path)
    cfg.mutator.state_conflict_diagnostics = {
        "enabled": False,
        "output_dir": str(tmp_path / "inactive-diagnostics"),
    }
    selected = builder_module._selected_scientific_payload(cfg)
    evaluator = builder_module._selected_component_config(
        selected["evaluator"],
        family="evaluator",
    )
    mutator = builder_module._selected_component_config(
        selected["mutator"],
        family="mutator",
    )
    assert "state_conflict_diagnostics" in cfg.mutator
    assert "state_conflict_diagnostics" not in evaluator
    assert "state_conflict_diagnostics" not in mutator

    active = _args(tmp_path)
    active.mutator.state_conflict_diagnostics = {
        "enabled": True,
        "output_dir": str(tmp_path / "active-diagnostics"),
    }
    active_selected = builder_module._selected_scientific_payload(active)
    active_mutator = builder_module._selected_component_config(
        active_selected["mutator"],
        family="mutator",
    )
    assert active.mutator.state_conflict_diagnostics.enabled is True
    assert active_mutator.state_conflict_diagnostics.enabled is True
    assert active_mutator.state_conflict_diagnostics.output_dir == str(
        tmp_path / "active-diagnostics"
    )

    persisted = run_config_lifecycle.build_persisted_run_config(active)
    assert persisted.mutator.state_conflict_diagnostics.enabled is True
    persisted_selected = builder_module._selected_scientific_payload(persisted)
    persisted_mutator = builder_module._selected_component_config(
        persisted_selected["mutator"],
        family="mutator",
    )
    assert persisted_mutator.state_conflict_diagnostics == (
        active_mutator.state_conflict_diagnostics
    )


def test_selected_component_config_constructs_real_factories(
    monkeypatch,
    tmp_path: Path,
):
    model_dir = tmp_path / "alphafold3"
    model_dir.mkdir()
    (model_dir / "af3.bin").write_bytes(b"cpu-factory-probe")

    cfg = _args(tmp_path)
    cfg.evaluator.evaluator_terms.fold._type = "Alphafold3"
    cfg.evaluator.alphafold3 = {
        "model_dir": str(model_dir),
        "flash_attention_implementation": "cudnn",
        "num_seeds": 1,
        "num_recycles": 3,
        "num_samples": 2,
        "save_embeddings": False,
        "run_data_pipeline": False,
        "run_inference": True,
        "execution": "daemon",
    }
    cfg.mutator.MCMC = {
        "mutation_method": "frequency_adjusted",
        "select_positions": "plddt",
    }

    selected = builder_module._selected_scientific_payload(cfg)
    assert selected["evaluator"]["base_key"] == "alphafold3"
    assert selected["mutator"]["base_key"] == "MCMC"
    evaluator_config = builder_module._selected_component_config(
        selected["evaluator"],
        family="evaluator",
    )
    mutator_config = builder_module._selected_component_config(
        selected["mutator"],
        family="mutator",
    )
    assert "alphafold3" in evaluator_config
    assert "Alphafold3" not in evaluator_config
    assert "MCMC" in mutator_config

    _initialize_legacy()
    data = builder_module.DataStore(tmp_path / "native-factory-data")
    context = builder_module.LegacyContext()
    translation = context.capture_candidate(assemblies=("A",), data=data)
    bindings = builder_module.NativeBindings(
        context,
        translation,
        data,
        tmp_path / "native-factory-work",
    )
    sampling = builder_module.SamplingPolicy(
        key_for(("native-factory-probe/sampling/v1",)),
        data.put({"mutation_num": 1}, builder_module.JSONCodec()),
    )
    component_context = NativeComponentContext(
        bindings=bindings,
        validated_evaluator_conf=evaluator_config,
        validated_mutator_conf=mutator_config,
        raw_model_results_dir=tmp_path / "raw",
        sampling_policy=sampling,
    )

    from astrolabe.model_execution import capture_random_state

    daemon_specs = []
    daemon_requests = []

    class DaemonStartBoundary:
        def __init__(self, spec, **_kwargs):
            daemon_specs.append(spec)

        def request(self, payload):
            daemon_requests.append(payload)
            if payload.get("operation") != "describe":
                raise AssertionError("factory probe must not run inference")
            return {
                "metadata": {},
                "random_state": capture_random_state(),
            }

        def shutdown(self):
            pass

        @property
        def pid(self):
            return None

    monkeypatch.setattr(
        "astrolabe.model_daemon.OwnedModelDaemon",
        DaemonStartBoundary,
    )

    evaluator_operation = EvaluatorRegistry.build_operation(
        "Alphafold3",
        alias="fold",
        options={},
        context=component_context,
    )
    evaluator = evaluator_operation.factory()
    try:
        assert evaluator.model_runner._settings == {
            "model_dir": str(model_dir),
            "flash_attention_implementation": "cudnn",
            "num_samples": 2,
            "num_recycles": 3,
            "save_embeddings": False,
        }
        assert daemon_specs[0].settings["config"]["num_samples"] == 2
        assert daemon_specs[0].settings["config"]["num_recycles"] == 3
        assert [
            payload.get("operation") for payload in daemon_requests
        ] == ["describe"]
    finally:
        evaluator.shutdown()

    mutator_operation = MutatorRegistry.build_operation(
        "MCMC",
        alias="mutate",
        options={},
        context=component_context,
    )
    mutator = mutator_operation.factory()
    assert mutator.mutation_method == "frequency_adjusted"
    assert mutator.select_positions == "plddt"


def hashlib_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_manifest(runtime):
    return {
        "revision": runtime.scientific_contract.digest,
        "operations": {
            alias: operation.descriptor.identity.digest
            for alias, operation in runtime.operations.items()
        },
        "budgets": dict(runtime.budgets),
    }


def test_resume_preserves_full_runtime_manifest_without_model_load(
    monkeypatch,
    tmp_path: Path,
):
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"weights")
    first_cfg = _select_asset(_args(tmp_path), asset)
    first_calls = []
    first_initializers = []
    first, _ = _build(
        monkeypatch,
        tmp_path,
        cfg=first_cfg,
        calls=first_calls,
        initializer_calls=first_initializers,
        asset_paths={"weights": asset},
    )
    initial_manifest = _runtime_manifest(first.runtime)
    root = Path(first.args.out_dir)
    initializer_candidate = first.flow.initial
    persisted = run_config_lifecycle.build_persisted_run_config(first.args)
    OmegaConf.save(persisted, root / "config.yaml")
    (root / "provenance.json").write_text(
        '{"state":"failed"}\n',
        encoding="utf-8",
    )
    first.close()
    global_config.reset()

    resumed_cfg = _args(tmp_path, recovery_resume=True)
    resumed_cfg.out_dir = str(root)
    _select_asset(resumed_cfg, asset)
    resume_calls = []
    resume_initializers = []
    resume_forbidden = []
    resumed, (evaluator, mutator) = _build(
        monkeypatch,
        tmp_path,
        cfg=resumed_cfg,
        calls=resume_calls,
        initializer_calls=resume_initializers,
        asset_paths={"weights": asset},
        forbidden_calls=resume_forbidden,
    )
    try:
        assert resume_calls == [
            ("parse_arguments", root, False),
            ("prepare_pipeline_state", False),
            ("prepare_pipeline_math",),
        ]
        assert resume_forbidden == []
        assert resume_initializers == [False]
        assert resumed.flow.initial == initializer_candidate
        assert resumed.contract.config_identity == first.contract.config_identity
        assert _runtime_manifest(resumed.runtime) == initial_manifest
        assert evaluator.loads == 0
        assert mutator.loads == 0
    finally:
        resumed.close()


def test_source_and_asset_content_not_paths_drive_identity(
    monkeypatch,
    tmp_path: Path,
):
    source = tmp_path / "selected_source.py"
    asset = tmp_path / "selected_asset.bin"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    asset.write_bytes(b"one")
    initial_source = key_for(
        source_manifest(run_module, {"fixture": source})
    )
    initial_asset = key_for(content_manifest({"weight": asset}, prefix="asset"))

    source.write_text("VALUE = 2\n", encoding="utf-8")
    asset.write_bytes(b"two")
    changed_source = key_for(
        source_manifest(run_module, {"fixture": source})
    )
    changed_asset = key_for(content_manifest({"weight": asset}, prefix="asset"))
    assert changed_source != initial_source
    assert changed_asset != initial_asset


def test_application_source_manifest_is_repository_wide_python_tree(
    tmp_path: Path,
):
    monkeypatch_run = tmp_path / "run.py"
    monkeypatch_run.write_text("RUN = 1\n", encoding="utf-8")
    source_root = tmp_path / "astrolabe"
    for relative in (
        "utils/base.py",
        "losses/base.py",
        "losses/confidence.py",
        "per_residue_scores.py",
        "core/protomer_config_parser.py",
    ):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    shim = ModuleType("test_application_run")
    shim.__file__ = str(monkeypatch_run)
    records = source_manifest(shim)
    application = next(
        record
        for record in records
        if record.role == "application/astrolabe"
    )
    entries = {path for path, _digest, _size in application.entries}
    assert {
        "utils/base.py",
        "losses/base.py",
        "losses/confidence.py",
        "per_residue_scores.py",
        "core/protomer_config_parser.py",
    }.issubset(entries)


def _af3_source_args(tmp_path: Path):
    cfg = _args(tmp_path)
    cfg.evaluator.evaluator_terms.fold._type = "alphafold3"
    return cfg


def _source_module_shim(tmp_path: Path) -> ModuleType:
    run_path = tmp_path / "source" / "run.py"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text("RUN = 1\n", encoding="utf-8")
    source_root = run_path.parent / "astrolabe"
    source_root.mkdir()
    (source_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    shim = ModuleType("test_runtime_package_run")
    shim.__file__ = str(run_path)
    return shim


def test_selected_af3_hashes_runtime_package_and_explicit_repo(
    tmp_path: Path,
    monkeypatch,
):
    shim = _source_module_shim(tmp_path)
    package = tmp_path / "site-packages" / "alphafold3"
    package.mkdir(parents=True)
    (package / "model.py").write_bytes(b"runtime-v1\n")
    (package / "resources.json").write_bytes(b'{"version":1}\n')
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "model.cpython-311.pyc").write_bytes(b"cache")
    custom = tmp_path / "custom-af3"
    custom.mkdir()
    (custom / "source.py").write_bytes(b"custom-repo\n")

    stale_vendor = (
        Path(shim.__file__).parent
        / "vendor"
        / "alphafold3"
        / "src"
    )
    stale_vendor.mkdir(parents=True)
    (stale_vendor / "stale.py").write_bytes(b"must-not-be-authority\n")

    cfg = _af3_source_args(tmp_path)
    cfg.evaluator.evaluator_terms.fold.repo_path = str(custom)
    monkeypatch.setattr(
        builder_module.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(
                submodule_search_locations=[str(package)],
            )
            if name == "alphafold3"
            else None
        ),
    )

    records = source_manifest(shim, args=cfg)
    runtime = next(
        record for record in records if record.role == "runtime/alphafold3"
    )
    assert runtime.path == str(package.resolve())
    assert {entry[0] for entry in runtime.entries} == {
        "model.py",
        "resources.json",
    }
    assert not any(
        "/vendor/alphafold3" in record.path for record in records
    )
    assert any(
        record.role == (
            "selected/evaluator/"
            "selected.evaluator.term.repo_path"
        )
        and record.path == str(custom.resolve())
        for record in records
    )

    initial_identity = key_for(records)
    (package / "model.py").write_bytes(b"runtime-v2\n")
    changed_identity = key_for(source_manifest(shim, args=cfg))
    assert changed_identity != initial_identity


def test_selected_af3_without_runtime_package_fails_preflight(
    tmp_path: Path,
    monkeypatch,
):
    shim = _source_module_shim(tmp_path)
    cfg = _af3_source_args(tmp_path)
    monkeypatch.setattr(
        builder_module.importlib.util,
        "find_spec",
        lambda name: None,
    )
    with pytest.raises(
        UnsupportedPortableNativeProfile,
        match="runtime alphafold3 source package has no directory origin",
    ):
        source_manifest(shim, args=cfg)


def test_selected_af3_without_runtime_fails_before_state_or_models(
    tmp_path: Path,
    monkeypatch,
):
    cfg = _af3_source_args(tmp_path)
    calls = []
    initializer_calls = []
    forbidden = []
    _install_run_module(
        monkeypatch,
        tmp_path,
        calls=calls,
        initializer_calls=initializer_calls,
        forbidden_calls=forbidden,
    )
    monkeypatch.setattr(
        builder_module.importlib.util,
        "find_spec",
        lambda name: None,
    )
    with pytest.raises(
        UnsupportedPortableNativeProfile,
        match="runtime alphafold3 source package has no directory origin",
    ):
        build_portable_native_recipe(
            cfg,
            math_factory=_math_factory,
            run_module=run_module,
        )
    assert calls == [("parse_arguments", None, True)]
    assert initializer_calls == []
    assert forbidden == []


def test_selected_af3_hashes_the_active_runtime_package_when_available(
    tmp_path: Path,
):
    spec = builder_module.importlib.util.find_spec("alphafold3")
    locations = (
        tuple(
            Path(item).resolve()
            for item in (spec.submodule_search_locations or ())
        )
        if spec is not None
        else ()
    )
    if not locations:
        pytest.skip("active interpreter has no alphafold3 runtime package")

    cfg = _af3_source_args(tmp_path)
    records = source_manifest(run_module, args=cfg)
    runtime = [
        record
        for record in records
        if record.role == "runtime/alphafold3"
        or record.role.startswith("runtime/alphafold3/")
    ]
    assert runtime
    assert {Path(record.path) for record in runtime} == set(locations)
    assert sum(len(record.entries) for record in runtime) > 0


@pytest.mark.parametrize(
    "changed_source",
    (
        "utils/base.py",
        "losses/confidence.py",
    ),
)
def test_resume_rejects_coarse_source_change_before_state_or_models(
    monkeypatch,
    tmp_path: Path,
    changed_source: str,
):
    project = tmp_path / "source"
    run_path = project / "run.py"
    run_path.parent.mkdir(parents=True)
    run_path.write_text("RUN = 1\n", encoding="utf-8")
    source_root = project / "astrolabe"
    for relative in (
        "utils/base.py",
        "losses/confidence.py",
    ):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"VALUE = {relative!r}\n", encoding="utf-8")
    monkeypatch.setattr(run_module, "__file__", str(run_path))

    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"weights")
    first_cfg = _select_asset(_args(tmp_path), asset)
    first, _ = _build(
        monkeypatch,
        tmp_path,
        cfg=first_cfg,
        asset_paths={"weights": asset},
    )
    root = Path(first.args.out_dir)
    OmegaConf.save(
        run_config_lifecycle.build_persisted_run_config(first.args),
        root / "config.yaml",
    )
    (root / "provenance.json").write_text(
        '{"state":"failed"}\n',
        encoding="utf-8",
    )
    first.close()
    global_config.reset()

    (source_root / changed_source).write_text(
        f"VALUE = 'changed {changed_source}'\n",
        encoding="utf-8",
    )
    resume_cfg = OmegaConf.create(
        OmegaConf.to_container(first.args, resolve=False)
    )
    resume_cfg.out_dir = str(root)
    resume_cfg.recovery.resume = True
    calls = []
    initializer_calls = []
    forbidden = []
    _install_run_module(
        monkeypatch,
        tmp_path,
        calls=calls,
        initializer_calls=initializer_calls,
        forbidden_calls=forbidden,
    )
    with pytest.raises(
        ContractConflict,
        match="selected source changed",
    ):
        build_portable_native_recipe(
            resume_cfg,
            math_factory=_math_factory,
            asset_paths={"weights": asset},
            run_module=run_module,
        )
    assert calls == [("parse_arguments", root, False)]
    assert initializer_calls == []
    assert forbidden == []


def _selected_file_cfg(
    tmp_path: Path,
    *,
    kind: str,
    path: Path,
):
    cfg = _args(tmp_path)
    if kind == "homology":
        cfg.homologies = {"A": f"{path}:A"}
    elif kind == "loss":
        cfg.loss.loss_terms.source_tm = {
            "_type": "tmscore",
            "template_files": [str(path)],
        }
        cfg.stages.default.weighted_score_expression = "source_tm"
    elif kind == "user-ccd":
        cfg.user_ccd = str(path)
    elif kind == "evaluator-reference":
        cfg.evaluator.evaluator_terms.fold.initial_guess = {
            "enabled": True,
            "references": {
                "AAAA": {"path": str(path)},
            },
        }
    elif kind == "evaluator-constraint":
        cfg.evaluator.evaluator_terms.fold.constraint_path = str(path)
    else:
        raise AssertionError(kind)
    return cfg


@pytest.mark.parametrize(
    ("kind", "filename"),
    (
        ("homology", "selected.json"),
        ("loss", "selected.pdb"),
        ("user-ccd", "selected.cif"),
        ("evaluator-reference", "reference.pdb"),
        ("evaluator-constraint", "constraints.json"),
    ),
)
def test_resume_rejects_same_path_selected_file_content_change(
    monkeypatch,
    tmp_path: Path,
    kind: str,
    filename: str,
):
    selected_file = tmp_path / filename
    selected_file.write_bytes(b"selected-v1")
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"weights")
    cfg = _selected_file_cfg(
        tmp_path,
        kind=kind,
        path=selected_file,
    )
    cfg = _select_asset(cfg, asset)
    first, _ = _build(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        asset_paths={"weights": asset},
    )
    root = Path(first.args.out_dir)
    OmegaConf.save(
        run_config_lifecycle.build_persisted_run_config(first.args),
        root / "config.yaml",
    )
    (root / "provenance.json").write_text(
        '{"state":"failed"}\n',
        encoding="utf-8",
    )
    first.close()
    global_config.reset()

    selected_file.write_bytes(b"selected-v2")
    resume_cfg = OmegaConf.create(
        OmegaConf.to_container(first.args, resolve=False)
    )
    resume_cfg.out_dir = str(root)
    resume_cfg.recovery.resume = True
    calls = []
    initializer_calls = []
    forbidden = []
    _install_run_module(
        monkeypatch,
        tmp_path,
        calls=calls,
        initializer_calls=initializer_calls,
        forbidden_calls=forbidden,
    )
    with pytest.raises(
        ContractConflict,
        match="selected asset content changed",
    ):
        build_portable_native_recipe(
            resume_cfg,
            math_factory=_math_factory,
            asset_paths={"weights": asset},
            run_module=run_module,
        )
    assert calls == [
        ("parse_arguments", root, False),
        ("prepare_pipeline_state", False),
    ]
    assert initializer_calls == [False]
    assert forbidden == []


def test_missing_selected_file_fails_before_state_preparation(
    monkeypatch,
    tmp_path: Path,
):
    missing = tmp_path / "missing.a3m"
    cfg = _args(tmp_path)
    cfg.homologies = {"A": str(missing)}
    calls = []
    initializer_calls = []
    _install_run_module(
        monkeypatch,
        tmp_path,
        calls=calls,
        initializer_calls=initializer_calls,
    )
    with pytest.raises(
        UnsupportedPortableNativeProfile,
        match="selected input dependency is missing",
    ):
        build_portable_native_recipe(
            cfg,
            math_factory=_math_factory,
            run_module=run_module,
        )
    assert calls == [("parse_arguments", None, True)]
    assert initializer_calls == []


def test_dormant_config_is_outside_the_canonical_identity(tmp_path: Path):
    cfg = _args(tmp_path)
    cfg.rf3.repo_path = "${missing_dormant_rf3.repo_path}"
    cfg.unused_rf3 = {"different": "${another.unused.value}"}
    payload = scientific_config_payload(cfg)
    assert "unused_rf3" not in payload
    assert "rf3" not in payload


def test_raw_runtime_markers_and_dormant_roots_do_not_change_identity(
    tmp_path: Path,
):
    first = _args(tmp_path)
    second = OmegaConf.create(OmegaConf.to_container(first, resolve=False))
    first.autodock = {"binary": "${not.selected.autodock}"}
    first.carbondesign = {"enabled": True}
    first._raw_model_results_dir_auto_managed = True
    first._raw_model_results_dir_marker = str(tmp_path / "raw-a/.marker")
    second._raw_model_results_dir_auto_managed = False
    second._raw_model_results_dir_marker = None
    second.out_dir = str(tmp_path / "another-output")
    second.task_name = "another-task-name"
    # The persisted lifecycle may omit a redundant root fallback.  The stage
    # field remains the authoritative mutation count.
    second.mutation_num = 10

    first_payload = scientific_config_payload(first)
    second_payload = scientific_config_payload(second)
    assert first_payload == second_payload
    assert key_for(first_payload) == key_for(second_payload)


def test_parsed_and_persisted_semantic_views_have_the_same_identity(
    tmp_path: Path,
):
    parsed = _args(tmp_path)
    parsed._raw_model_results_dir_auto_managed = True
    parsed._raw_model_results_dir_marker = str(tmp_path / "raw/.marker")
    persisted = run_config_lifecycle.build_persisted_run_config(parsed)
    assert scientific_config_payload(parsed) == scientific_config_payload(persisted)


def test_active_scientific_fields_change_the_closed_identity(tmp_path: Path):
    base = _args(tmp_path)
    original = key_for(scientific_config_payload(base))
    mutations = (
        ("seed", lambda cfg: setattr(cfg, "seed", 18)),
        ("steps", lambda cfg: setattr(cfg, "steps", 3)),
        ("sequence", lambda cfg: cfg.seqs.__setitem__("A", "AAAAA")),
        (
            "acceptance",
            lambda cfg: cfg.stages.default.__setitem__(
                "acceptance_threshold",
                "plddt > 0.5",
            ),
        ),
        (
            "initializer",
            lambda cfg: cfg.initializer.__setitem__("mode", "aa_set")
            or cfg.initializer.__setitem__("aa_set", ["A"]),
        ),
        (
            "evaluator-default",
            lambda cfg: cfg.evaluator.__setitem__(
                "portable_builder_evaluator",
                {"num_recycles": 2},
            ),
        ),
        (
            "evaluator-term",
            lambda cfg: cfg.evaluator.evaluator_terms.fold.__setitem__(
                "num_samples",
                2,
            ),
        ),
        (
            "mutator-default",
            lambda cfg: cfg.mutator.__setitem__(
                "MCMC",
                {"mutation_method": "frequency_adjusted"},
            ),
        ),
        (
            "mutator-term",
            lambda cfg: cfg.mutator.mutator_terms.mutate.__setitem__(
                "temperature",
                0.2,
            ),
        ),
        (
            "metropolis-temperature",
            lambda cfg: setattr(cfg, "use_metropolis_acceptance", True),
        ),
    )
    for label, mutate in mutations:
        changed = OmegaConf.create(
            OmegaConf.to_container(base, resolve=False)
        )
        mutate(changed)
        changed_key = key_for(scientific_config_payload(changed))
        assert changed_key != original, label


def test_explicit_unreferenced_loss_matches_loss_engine_required_set(
    tmp_path: Path,
):
    from astrolabe.losses.base import LossEngine

    cfg = _args(tmp_path)
    cfg.loss.loss_terms.explicit_ptm = {"_type": "ptm"}
    payload = scientific_config_payload(cfg)
    selected_loss = payload["selected"]["loss"]
    assert selected_loss["loss_terms"] == {
        "explicit_ptm": {"_type": "ptm"}
    }

    stage = payload["selected"]["stage"]
    engine = LossEngine(
        selected_loss,
        runtime_context=object(),
        validate_config=False,
    )
    engine.setup(
        weighted_score_expression=stage["weighted_score_expression"],
        early_stopping=stage.get("early_stopping"),
        acceptance_threshold=stage.get("acceptance_threshold"),
        acceptance_gate=stage.get("acceptance_gate"),
        prefix_early_stopping=stage.get("prefix_early_stopping"),
        validate_config=False,
    )
    assert set(engine.included_loss_terms) == {
        "plddt",
        "explicit_ptm",
    }


def test_inactive_temperature_is_not_required_by_runtime_initialization(
    monkeypatch,
    tmp_path: Path,
):
    cfg = _args(tmp_path)
    with open_dict(cfg):
        cfg.exclude_AA = []
        del cfg.initial_temperature
    run_module.initialize_global_config(cfg)
    assert global_config.initial_temperature is None
    global_config.reset()

    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"weights")
    _select_asset(cfg, asset)
    recipe, _ = _build(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        asset_paths={"weights": asset},
    )
    try:
        assert scientific_config_payload(cfg)["run"]["initial_temperature"] is None
    finally:
        recipe.close()


def test_active_metropolis_temperature_remains_required(tmp_path: Path):
    cfg = _args(tmp_path)
    with open_dict(cfg):
        cfg.use_metropolis_acceptance = True
        del cfg.initial_temperature
    with pytest.raises(ConfigAttributeError, match="initial_temperature"):
        run_module.initialize_global_config(cfg)


def test_inactive_af3_datapipeline_is_removed_from_process_runtime_view(
    tmp_path: Path,
):
    cfg = _args(tmp_path)
    cfg.evaluator.evaluator_terms.fold._type = "Alphafold3"
    cfg.evaluator.evaluator_terms.fold.run_data_pipeline = False
    cfg.evaluator.alphafold3 = {
        "run_data_pipeline": False,
        "datapipeline": {
            "jackhmmer_binary_path": "${hmmer.jackhmmer}",
        },
    }

    builder_module._fill_inactive_runtime_defaults(cfg)

    assert "datapipeline" not in cfg.evaluator.alphafold3
    assert cfg.evaluator.evaluator_terms.fold.run_data_pipeline is False


def test_active_af3_datapipeline_keeps_its_resolvable_contract(
    tmp_path: Path,
):
    cfg = _args(tmp_path)
    cfg.evaluator.evaluator_terms.fold._type = "Alphafold3"
    cfg.evaluator.evaluator_terms.fold.run_data_pipeline = True
    cfg.evaluator.alphafold3 = {
        "run_data_pipeline": True,
        "datapipeline": {
            "jackhmmer_binary_path": "${hmmer.jackhmmer}",
        },
    }
    cfg.hmmer = {"jackhmmer": "/tools/jackhmmer"}

    builder_module._fill_inactive_runtime_defaults(cfg)

    assert OmegaConf.to_container(
        cfg.evaluator.alphafold3.datapipeline,
        resolve=False,
    )["jackhmmer_binary_path"] == (
        "${hmmer.jackhmmer}"
    )


def _tool_loss_cfg(tmp_path: Path, loss_type: str):
    cfg = _args(tmp_path)
    cfg.stages.default.weighted_score_expression = "plddt"
    if loss_type == "tmscore":
        template = tmp_path / "template.pdb"
        template.write_bytes(b"template\n")
        cfg.loss.loss_terms.explicit_tool = {
            "_type": "tmscore",
            "template_files": [str(template)],
        }
    elif loss_type == "min_dssp_fractions":
        cfg.loss.loss_terms.explicit_tool = {
            "_type": "min_dssp_fractions",
            "EHL_fractions": "0.2,0.2,0.2",
        }
    elif loss_type == "ananas_rmsd_raw":
        cfg.loss.loss_terms.explicit_tool = {
            "_type": "ananas_rmsd_raw",
            "point_group": "C",
            "order": 2,
        }
    else:
        raise AssertionError(loss_type)
    return cfg


@pytest.mark.parametrize(
    ("loss_type", "tool_key"),
    (
        ("tmscore", "usalign_path"),
        ("min_dssp_fractions", "dssp_path"),
        ("ananas_rmsd_raw", "ananas_path"),
    ),
)
def test_unbound_active_loss_executable_fails_before_state_or_models(
    monkeypatch,
    tmp_path: Path,
    loss_type: str,
    tool_key: str,
):
    cfg = _tool_loss_cfg(tmp_path, loss_type)
    setattr(cfg, tool_key, None)
    calls = []
    initializer_calls = []
    forbidden = []
    _install_run_module(
        monkeypatch,
        tmp_path,
        calls=calls,
        initializer_calls=initializer_calls,
        forbidden_calls=forbidden,
    )
    monkeypatch.setattr(
        builder_module.shutil,
        "which",
        lambda _name: None,
    )
    monkeypatch.delenv("GH_USALIGN", raising=False)
    monkeypatch.delenv("GH_DSSP", raising=False)
    monkeypatch.delenv("GH_ANANAS", raising=False)

    with pytest.raises(
        UnsupportedPortableNativeProfile,
        match=tool_key,
    ):
        build_portable_native_recipe(
            cfg,
            math_factory=_math_factory,
            run_module=run_module,
        )
    assert calls == [("parse_arguments", None, True)]
    assert initializer_calls == []
    assert forbidden == []


def test_bound_active_loss_executable_is_content_hashed(
    monkeypatch,
    tmp_path: Path,
):
    cfg = _tool_loss_cfg(tmp_path, "min_dssp_fractions")
    executable = tmp_path / "mkdssp"
    executable.write_bytes(b"test executable\n")
    executable.chmod(0o755)
    cfg.dssp_path = str(executable)
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"weights")
    _select_asset(cfg, asset)
    recipe, _ = _build(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        asset_paths={"weights": asset},
    )
    try:
        record = next(
            item
            for item in recipe.supplemental_asset_manifest
            if item.role == "loss-tools/dssp_path"
        )
        assert record.path == str(executable.resolve())
        assert record.sha256 == hashlib_file(executable)
    finally:
        recipe.close()


def test_plddt_loss_does_not_probe_unused_external_tools(
    monkeypatch,
    tmp_path: Path,
):
    cfg = _args(tmp_path)
    cfg.stages.default.weighted_score_expression = "plddt"
    assert builder_module._active_loss_tool_keys(cfg) == ()
    monkeypatch.setattr(
        builder_module.shutil,
        "which",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("unused external tool was probed")
        ),
    )
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"weights")
    _select_asset(cfg, asset)
    recipe, _ = _build(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        asset_paths={"weights": asset},
    )
    recipe.close()


def test_cold_build_from_real_release_config_ignores_unused_rf3(
    monkeypatch,
    tmp_path: Path,
):
    cfg = load_composed_config("config/release_protein.yaml")
    model_dir = tmp_path / "af3-model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"cold-cfg-model")
    with open_dict(cfg):
        cfg.task_name = "portable-builder-real-cfg"
        cfg.out_dir = str(tmp_path / "run")
        cfg.raw_model_results_dir = str(tmp_path / "raw")
        cfg.steps = 2
        cfg.repeats = 1
        cfg.evaluator.enabled_aliases = ["Alphafold3"]
        cfg.evaluator.evaluator_terms.Alphafold3.model_dir = str(model_dir)
        cfg.mutator.enabled_aliases = ["MCMC"]
        cfg.unused_rf3 = {}
        cfg.rf3.repo_path = "${unused_rf3.repo_path}"
        cfg.recovery = {"mode": "durable", "resume": False}

    forbidden_calls = []
    recipe, (evaluator, mutator) = _build(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        forbidden_calls=forbidden_calls,
    )
    try:
        assert evaluator.loads == 0
        assert mutator.loads == 0
        assert forbidden_calls == []
        assert recipe.selected_evaluator_alias == "Alphafold3"
        assert recipe.selected_mutator_alias == "MCMC"
        assert any(
            Path(item["path"]).name == "weights.bin"
            for item in recipe.asset_manifest["files"]
        )
        assert "rf3" not in scientific_config_payload(cfg)
        assert "unused_rf3" not in scientific_config_payload(cfg)
    finally:
        recipe.close()


def test_selected_interpolation_failure_remains_actionable(
    monkeypatch,
    tmp_path: Path,
):
    cfg = _args(tmp_path)
    cfg.evaluator.evaluator_terms.fold.checkpoint = (
        "${missing_selected.asset.path}"
    )
    calls = []
    initializer_calls = []
    _install_run_module(
        monkeypatch,
        tmp_path,
        calls=calls,
        initializer_calls=initializer_calls,
    )
    with pytest.raises(
        UnsupportedPortableNativeProfile,
        match="selected|resolve",
    ):
        build_portable_native_recipe(
            cfg,
            math_factory=_math_factory,
            operation_hooks=_hooks(),
            run_module=run_module,
        )
    assert initializer_calls == []


def test_default_decisions_adapter_is_constructed_cold(
    monkeypatch,
    tmp_path: Path,
):
    asset = tmp_path / "selected.pt"
    asset.write_bytes(b"selected-asset")
    cfg = _select_asset(_args(tmp_path), asset)
    recipe, _ = _build(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        asset_paths={"weights": asset},
        math_factory=None,
    )
    try:
        assert recipe.flow.math.requires_mutation_context is True
        assert recipe.flow.evaluator.owner.ready is False
        assert recipe.flow.mutator.owner.ready is False
    finally:
        recipe.close()


def test_default_evaluator_hook_runs_shared_finalizer_at_global_step(
    monkeypatch,
    tmp_path: Path,
):
    finalized = []
    original_finalizer = run_module._finalize_evaluation_result

    def spy_finalizer(args, scores, loss_computer=None):
        finalized.append((args, scores, loss_computer))
        return original_finalizer(args, scores, loss_computer=loss_computer)

    monkeypatch.setattr(
        run_module,
        "_finalize_evaluation_result",
        spy_finalizer,
    )
    asset = tmp_path / "selected.pt"
    asset.write_bytes(b"selected-asset")
    cfg = _select_asset(_args(tmp_path), asset)
    cfg.per_residue_scores.enabled = True
    cfg.per_residue_scores.terms = {"plddt": {"_type": "plddt"}}
    recipe, _ = _build(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        asset_paths={"weights": asset},
    )
    observed_steps = []

    class Evaluator:
        def evaluate(self, inputs, runtime_config):
            del inputs
            observed_steps.append(global_config.step)
            return StandardScores(
                names=["A"],
                plddt={"A": [np.asarray([0.7, 0.8], dtype=np.float32)]},
                complex_paths={},
                per_proto_scores={
                    "A": [np.asarray([0.7, 0.8], dtype=np.float32)]
                },
            )

    try:
        hook = recipe.flow.evaluator
        raw_scores = hook.native_invoke(
            Evaluator(),
            {"A": object()},
            {"step": 4},
        )
        assert global_config.step == 4
        assert observed_steps == [4]
        assert finalized == [
            (recipe.args, raw_scores, global_config.loss_computer)
        ]
        assert type(raw_scores) is StandardScores
        np.testing.assert_array_equal(
            raw_scores.per_proto_score_terms["A"]["plddt"][0],
            np.asarray([0.7, 0.8], dtype=np.float32),
        )
    finally:
        recipe.close()


def test_default_mutator_hook_uses_shared_single_alias_step(
    monkeypatch,
    tmp_path: Path,
):
    executed = []
    original_execute = run_module.execute_mutation_step

    def spy_execute(mutators, scores, runtime_config, step):
        executed.append((mutators, scores, runtime_config, step))
        return original_execute(mutators, scores, runtime_config, step)

    monkeypatch.setattr(run_module, "execute_mutation_step", spy_execute)
    asset = tmp_path / "selected.pt"
    asset.write_bytes(b"selected-asset")
    cfg = _select_asset(_args(tmp_path), asset)
    recipe, _ = _build(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        asset_paths={"weights": asset},
    )

    class Mutator:
        alias = "mutate"
        name = "portable-test-mutator"
        last_mutation_telemetry = {"metrics": {"test": 1.0}}

        def __init__(self):
            self.calls = []

        def mutate(self, protomers, scores, runtime_config):
            self.calls.append((protomers, scores, runtime_config))
            runtime_config["mutator_selector_score_contexts"].append(
                {"mutator": "mutate", "selected": ["A"]}
            )

    scores = StandardScores(
        names=["A"],
        plddt={"A": [np.asarray([0.7, 0.8], dtype=np.float32)]},
        complex_paths={},
    )
    mutator = Mutator()
    runtime_config = {"step": 2}
    try:
        hook = recipe.flow.mutator
        hook.configure(
            mutator,
            runtime_config,
            request=None,
            sample=None,
            step=2,
            work_root=tmp_path,
        )
        hook.native_invoke(
            mutator,
            global_config.mutable_protomers,
            scores,
            runtime_config,
        )
        assert global_config.step == 2
        assert global_config.mutation_num == 1
        assert mutator.mutation_weight == 1.0
        assert len(executed) == 1
        mutators, observed_scores, observed_config, observed_step = executed[0]
        assert mutators == [mutator]
        assert observed_scores is scores
        assert observed_config is runtime_config
        assert observed_step == 2
        assert runtime_config["mutator_selector_score_contexts"] == [
            {"mutator": "mutate", "selected": ["A"]}
        ]
        assert mutator.calls == [
            (global_config.mutable_protomers, scores, runtime_config)
        ]
    finally:
        recipe.close()


def test_portable_decisions_require_typed_mutation_context_before_model_load(
    monkeypatch,
    tmp_path: Path,
):
    asset = tmp_path / "selected.pt"
    asset.write_bytes(b"selected-asset")
    cfg = _select_asset(_args(tmp_path), asset)
    calls = []
    initializer_calls = []
    evaluator_loads, mutator_loads = _install_run_module(
        monkeypatch,
        tmp_path,
        calls=calls,
        initializer_calls=initializer_calls,
    )
    hooks = _hooks()
    hooks["mutate"] = NativeOperationHooks(
        configure=_mutator_hook,
        identity="test/portable-builder/mutate-without-context/v1",
    )

    with pytest.raises(
        UnsupportedPortableNativeProfile,
        match="MutationContextPolicy",
    ):
        build_portable_native_recipe(
            cfg,
            operation_hooks=hooks,
            asset_paths={"weights": asset},
            run_module=run_module,
        )

    assert evaluator_loads.loads == 0
    assert mutator_loads.loads == 0


def test_math_factory_reuses_builder_run_module_object(
    monkeypatch,
    tmp_path: Path,
):
    asset = tmp_path / "selected.pt"
    asset.write_bytes(b"selected-asset")
    cfg = _select_asset(_args(tmp_path), asset)
    contexts = []

    def math_factory(context):
        contexts.append(context)
        return build_portable_native_math(
            context,
            run_module=run_module,
        )

    recipe, _ = _build(
        monkeypatch,
        tmp_path,
        cfg=cfg,
        asset_paths={"weights": asset},
        math_factory=math_factory,
    )
    try:
        assert contexts[0].run_module is run_module
        with pytest.raises(ContractConflict, match="same module object"):
            build_portable_native_math(
                contexts[0],
                run_module=SimpleNamespace(),
            )
    finally:
        recipe.close()


def test_af3_mcmc_template_is_one_root_and_one_child_without_gpu_claim():
    template = af3_mcmc_single_child_template()
    assert template["steps"] == 2
    assert template["stages"]["default"]["mutators"] == ["MCMC"]
    assert template["stages"]["default"]["evaluator"] == "af3"
    evaluator = template["evaluator"]["evaluator_terms"]["af3"]
    assert evaluator["_type"] == "Alphafold3"
    assert evaluator["num_seeds"] == 1
    assert evaluator["num_recycles"] == 1
    assert evaluator["num_samples"] == 1
    assert evaluator["run_data_pipeline"] is False
    assert template["recovery"] == {"mode": "durable", "resume": False}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda args: args.stages.default.update(
                {"transition_consistency_gate": {"enabled": True}}
            ),
            "transition controls",
        ),
        (
            lambda args: setattr(args, "enable_pyrosetta", True),
            "PyRosetta",
        ),
        (
            lambda args: setattr(args, "skip_complex_poses", False),
            "skip_complex_poses",
        ),
        (
            lambda args: args.initializer.update({"mode": "dplm"}),
            "paid-initializer",
        ),
        (
            lambda args: args.mutator.update({"stack_pre_update": True}),
            "stack_pre_update",
        ),
        (
            lambda args: setattr(
                args,
                "search",
                {"algorithm": "branching", "owner": "other"},
            ),
            "search controls",
        ),
    ],
)
def test_unsupported_capabilities_fail_static_preflight(
    tmp_path: Path,
    mutation,
    message: str,
):
    args = _args(tmp_path)
    mutation(args)
    with pytest.raises(UnsupportedPortableNativeProfile, match=message):
        validate_portable_native_profile(args, run_module)
