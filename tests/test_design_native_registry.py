from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design.legacy import LegacyContext
from astrolabe.design.native_registry import (
    NativeComponentContext,
    NativeOperationHooks,
)
from astrolabe.design.native_mutation_state import (
    NATIVE_MUTATION_CONTEXT_CODEC,
    NATIVE_MUTATION_CONTEXT_SCHEMA,
    MutationContextPolicy,
)
from astrolabe.design.runtime import ContractConflict, Runtime
from astrolabe.design.native_support import NativeBindings
from astrolabe.design.store import DataStore, JSONCodec
from astrolabe.design.types import (
    Assignment,
    Polymer,
    PredictRequest,
    Sample,
    SamplingPolicy,
    key_for,
    make_candidate,
    make_system,
)
from astrolabe.evaluators.registry import EvaluatorRegistry
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.registry import MutatorRegistry


JSON = JSONCodec()


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch):
    monkeypatch.setattr(EvaluatorRegistry, "_registry", dict(EvaluatorRegistry._registry))
    monkeypatch.setattr(MutatorRegistry, "_registry", dict(MutatorRegistry._registry))
    monkeypatch.setattr(
        EvaluatorRegistry,
        "_operation_factories",
        dict(EvaluatorRegistry._operation_factories or {}),
    )
    monkeypatch.setattr(
        MutatorRegistry,
        "_operation_factories",
        dict(MutatorRegistry._operation_factories or {}),
    )
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())
    yield
    global_config.reset()


def _sampling(data: DataStore) -> SamplingPolicy:
    return SamplingPolicy(
        key_for("sampling/policy"),
        data.put({"mutation_count": 1}, JSON),
    )


def _minimal_bindings(tmp_path: Path) -> NativeBindings:
    data = DataStore(tmp_path / "data")
    topology = data.put({"topology": "single"}, JSON)
    system = make_system(("A",), (), topology)
    candidate = make_candidate(
        system,
        (Assignment("A", Polymer("protein", ("A",), ())),),
    )
    return NativeBindings(
        SimpleNamespace(),
        SimpleNamespace(system=system, candidate=candidate),
        data,
        tmp_path / "work",
    )


def _context(
    bindings: NativeBindings,
    *,
    evaluator_terms=None,
    mutator_terms=None,
    schemas=None,
    operation_hooks=None,
) -> NativeComponentContext:
    evaluator_conf = (
        OmegaConf.create({"evaluator_terms": evaluator_terms})
        if evaluator_terms is not None
        else None
    )
    mutator_conf = (
        OmegaConf.create({"mutator_terms": mutator_terms})
        if mutator_terms is not None
        else None
    )
    return NativeComponentContext(
        bindings=bindings,
        validated_evaluator_conf=evaluator_conf,
        validated_mutator_conf=mutator_conf,
        raw_model_results_dir=bindings.work_root / "raw",
        sampling_policy=_sampling(bindings.data),
        options_schemas=schemas or {},
        operation_hooks=operation_hooks or {},
    )


def test_explicit_operation_factory_has_priority_without_native_context(
    monkeypatch,
):
    marker = object()

    def override(*, alias, options, context):
        assert alias == "custom"
        assert options == {"x": 1}
        assert context is None
        return marker

    EvaluatorRegistry.register_operation("custom", override)
    assert (
        EvaluatorRegistry.build_operation(
            "custom",
            alias="custom",
            options={"x": 1},
            context=None,
        )
        is marker
    )


def test_unknown_registry_name_and_untyped_options_fail_before_load(tmp_path):
    bindings = _minimal_bindings(tmp_path)
    loads = []

    def fake_factory(*args, **kwargs):
        loads.append("load")
        raise AssertionError("factory must not run")

    EvaluatorRegistry.register("native_fake", fake_factory)
    context = _context(
        bindings,
        evaluator_terms={"fold": {"_type": "native_fake"}},
    )
    with pytest.raises(ValueError, match="Unknown EvaluatorRegistry"):
        EvaluatorRegistry.build_operation(
            "missing",
            alias="fold",
            options={},
            context=context,
        )
    with pytest.raises(ValueError, match="explicit options schema"):
        EvaluatorRegistry.build_operation(
            "native_fake",
            alias="fold",
            options={"unknown": True},
            context=context,
        )
    assert loads == []


def test_explicit_options_schema_is_required_and_validated_before_load(tmp_path):
    bindings = _minimal_bindings(tmp_path)
    loads = []

    def fake_factory(*args, **kwargs):
        loads.append("load")
        raise AssertionError("factory must not run during build")

    EvaluatorRegistry.register("native_schema", fake_factory)
    context = _context(
        bindings,
        evaluator_terms={"fold": {"_type": "native_schema"}},
        schemas={
            "fold": {
                "type": "object",
                "properties": {"temperature": {"type": "number"}},
                "required": ["temperature"],
                "additionalProperties": False,
            }
        },
    )
    operation = EvaluatorRegistry.build_operation(
        "native_schema",
        alias="fold",
        options={"temperature": 0.25},
        context=context,
    )
    assert operation.descriptor.options_schema.effective_options_json == (
        '{"temperature":0.25}'
    )
    with pytest.raises(Exception, match="unknown|Additional properties"):
        EvaluatorRegistry.build_operation(
            "native_schema",
            alias="fold",
            options={"temperature": 0.25, "unexpected": True},
            context=context,
        )
    assert loads == []


def test_selected_identity_ignores_inactive_terms(tmp_path):
    bindings = _minimal_bindings(tmp_path)

    def fake_factory(*args, **kwargs):
        raise AssertionError("factory must not run during build")

    EvaluatorRegistry.register("native_identity", fake_factory)
    first_context = _context(
        bindings,
        evaluator_terms={
            "fold": {"_type": "native_identity", "recycles": 1},
            "inactive": {"_type": "native_identity", "recycles": 99},
        },
    )
    second_context = _context(
        bindings,
        evaluator_terms={
            "fold": {"_type": "native_identity", "recycles": 1},
            "inactive": {"_type": "native_identity", "recycles": 100},
        },
    )
    first = EvaluatorRegistry.build_operation(
        "native_identity",
        alias="fold",
        options={},
        context=first_context,
    )
    second = EvaluatorRegistry.build_operation(
        "native_identity",
        alias="fold",
        options={},
        context=second_context,
    )
    assert first.model_identity == second.model_identity


def test_container_af3_cold_build_ignores_missing_inactive_rf3_and_hashes_assets(
    tmp_path,
    monkeypatch,
):
    from astrolabe.argparser import parse_arguments
    from astrolabe.configuration import load_composed_config

    monkeypatch.delenv("GH_WEIGHT_RF3", raising=False)
    model_dir = tmp_path / "alphafold3-assets"
    model_dir.mkdir()
    weights = model_dir / "af3.bin"
    weights.write_bytes(b"selected AF3 asset v1")

    cfg = load_composed_config(
        Path(__file__).resolve().parents[1]
        / "config"
        / "container_af3_smoke.yaml"
    )
    cfg.out_dir = str(tmp_path / "runs")
    cfg.seed = 17
    cfg.weight_path.alphafold3 = str(model_dir)
    args = parse_arguments(cfg)

    bindings = _minimal_bindings(tmp_path)
    native_context = NativeComponentContext(
        bindings=bindings,
        validated_evaluator_conf=args.evaluator,
        validated_mutator_conf=args.mutator,
        raw_model_results_dir=tmp_path / "raw",
        sampling_policy=_sampling(bindings.data),
    )
    first = EvaluatorRegistry.build_operation(
        str(args.evaluator.evaluator_terms.Alphafold3._type),
        alias="Alphafold3",
        options={},
        context=native_context,
    )
    again = EvaluatorRegistry.build_operation(
        str(args.evaluator.evaluator_terms.Alphafold3._type),
        alias="Alphafold3",
        options={},
        context=native_context,
    )
    mutator = MutatorRegistry.build_operation(
        str(args.mutator.mutator_terms.MCMC._type),
        alias="MCMC",
        options={
            "protomer_scope": "mutable",
            "mutation_weight": 1.0,
        },
        context=NativeComponentContext(
            bindings=bindings,
            validated_evaluator_conf=args.evaluator,
            validated_mutator_conf=args.mutator,
            raw_model_results_dir=tmp_path / "raw",
            sampling_policy=_sampling(bindings.data),
            options_schemas={
                "mcmc": {
                    "type": "object",
                    "properties": {
                        "protomer_scope": {
                            "type": "string",
                            "enum": ["mutable", "all"],
                        },
                        "mutation_weight": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                        },
                    },
                    "required": ["protomer_scope", "mutation_weight"],
                    "additionalProperties": False,
                }
            },
        ),
    )

    assert first.owner._value is None
    assert again.descriptor.identity == first.descriptor.identity
    assert mutator.owner._value is None

    weights.write_bytes(b"selected AF3 asset v2")
    changed = EvaluatorRegistry.build_operation(
        str(args.evaluator.evaluator_terms.Alphafold3._type),
        alias="Alphafold3",
        options={},
        context=native_context,
    )
    assert changed.descriptor.identity != first.descriptor.identity


def test_empty_options_build_every_real_registered_family_type_without_load(
    tmp_path,
):
    bindings = _minimal_bindings(tmp_path)
    import astrolabe.evaluators.utils  # noqa: F401
    import astrolabe.mutators.base  # noqa: F401

    for name in EvaluatorRegistry.list_available():
        context = _context(
            bindings,
            evaluator_terms={"selected": {"_type": name}},
        )
        operation = EvaluatorRegistry.build_operation(
            name,
            alias="selected",
            options={},
            context=context,
        )
        assert operation.descriptor.alias == "selected"
        assert operation.owner._value is None

    for name in MutatorRegistry.list_available():
        context = _context(
            bindings,
            mutator_terms={"selected": {"_type": name}},
        )
        operation = MutatorRegistry.build_operation(
            name,
            alias="selected",
            options={},
            context=context,
        )
        assert operation.descriptor.alias == "selected"
        assert operation.owner._value is None


@pytest.mark.parametrize("family", ["evaluator", "mutator"])
@pytest.mark.parametrize("has_asset", [False, True])
def test_native_resume_requires_content_verified_assets(tmp_path, family, has_asset):
    bindings = _minimal_bindings(tmp_path)

    def factory(*args, **kwargs):
        raise AssertionError("model must not load during resume preflight")

    name = "native_resume_asset_test"
    registry = EvaluatorRegistry if family == "evaluator" else MutatorRegistry
    registry.register(name, factory)
    term = {"_type": name}
    if has_asset:
        weights = tmp_path / "weights.bin"
        weights.write_bytes(b"frozen selected model")
        term["checkpoint_path"] = str(weights)
    context = _context(
        bindings,
        evaluator_terms={"selected": term} if family == "evaluator" else None,
        mutator_terms={"selected": term} if family == "mutator" else None,
    )

    def operation():
        return registry.build_operation(name, alias="selected", options={}, context=context)

    root = tmp_path / "runtime"
    first = Runtime((operation(),), root=root, data=bindings.data)
    first.close()
    if has_asset:
        resumed = Runtime((operation(),), root=root, data=bindings.data)
        resumed.close()
    else:
        with pytest.raises(ContractConflict, match="assets cannot be content-verified"):
            Runtime((operation(),), root=root, data=bindings.data)


def _evaluator_configure(_model, _request):
    return {"hook": "evaluator"}


def _mutator_configure(
    _model,
    runtime_config,
    *,
    request,
    sample,
    step,
    work_root,
):
    del request, sample, step, work_root
    return {**runtime_config, "hook": "mutator"}


def _changed_evaluator_configure(_model, _request):
    return {"hook": "evaluator", "revision": 2}


def test_operation_hooks_are_alias_scoped_cold_and_rebuild_identically(tmp_path):
    bindings = _minimal_bindings(tmp_path)
    loads = []
    calls = []

    def fake_evaluator(*args, **kwargs):
        loads.append("evaluator")
        raise AssertionError("hook validation must remain cold")

    def fake_mutator(*args, **kwargs):
        loads.append("mutator")
        raise AssertionError("hook validation must remain cold")

    def observed_evaluator_configure(model, request):
        calls.append("evaluator")
        return _evaluator_configure(model, request)

    def observed_mutator_configure(model, runtime_config, **kwargs):
        calls.append("mutator")
        return _mutator_configure(model, runtime_config, **kwargs)

    EvaluatorRegistry.register("native_hooked_evaluator", fake_evaluator)
    MutatorRegistry.register("native_hooked_mutator", fake_mutator)
    hooks = {
        "FOLD": NativeOperationHooks(
            configure=observed_evaluator_configure,
            identity="test/hook/evaluator/v1",
        ),
        "mutate": NativeOperationHooks(
            configure=observed_mutator_configure,
            identity="test/hook/mutator/v1",
        ),
        "inactive": NativeOperationHooks(
            configure=_changed_evaluator_configure,
            identity="test/hook/inactive/v1",
        ),
    }
    context = _context(
        bindings,
        evaluator_terms={"fold": {"_type": "native_hooked_evaluator"}},
        mutator_terms={"mutate": {"_type": "native_hooked_mutator"}},
        operation_hooks=hooks,
    )

    evaluator = EvaluatorRegistry.build_operation(
        "native_hooked_evaluator",
        alias="fold",
        options={},
        context=context,
    )
    mutator = MutatorRegistry.build_operation(
        "native_hooked_mutator",
        alias="mutate",
        options={},
        context=context,
    )
    evaluator_again = EvaluatorRegistry.build_operation(
        "native_hooked_evaluator",
        alias="fold",
        options={},
        context=context,
    )
    mutator_again = MutatorRegistry.build_operation(
        "native_hooked_mutator",
        alias="mutate",
        options={},
        context=context,
    )

    assert loads == []
    assert calls == []
    assert evaluator.configure is observed_evaluator_configure
    assert mutator.configure is observed_mutator_configure
    assert evaluator.configure_identity
    assert evaluator_again.descriptor.identity == evaluator.descriptor.identity
    assert mutator_again.descriptor.identity == mutator.descriptor.identity

    no_hook_context = _context(
        bindings,
        evaluator_terms={"fold": {"_type": "native_hooked_evaluator"}},
        operation_hooks={
            "inactive": NativeOperationHooks(
                configure=_changed_evaluator_configure,
                identity="test/hook/inactive/v2",
            )
        },
    )
    no_hook = EvaluatorRegistry.build_operation(
        "native_hooked_evaluator",
        alias="fold",
        options={},
        context=no_hook_context,
    )
    baseline_context = _context(
        bindings,
        evaluator_terms={"fold": {"_type": "native_hooked_evaluator"}},
    )
    baseline = EvaluatorRegistry.build_operation(
        "native_hooked_evaluator",
        alias="fold",
        options={},
        context=baseline_context,
    )
    assert no_hook.descriptor.identity == baseline.descriptor.identity
    assert no_hook.configure is None
    assert loads == []


def test_hook_identity_and_source_changes_descriptors_and_block_old_resume(
    tmp_path,
):
    bindings = _minimal_bindings(tmp_path)

    def fake_evaluator(*args, **kwargs):
        raise AssertionError("descriptor checks must not construct models")

    def fake_mutator(*args, **kwargs):
        raise AssertionError("descriptor checks must not construct models")

    EvaluatorRegistry.register("native_resume_evaluator", fake_evaluator)
    MutatorRegistry.register("native_resume_mutator", fake_mutator)

    def build(evaluator_hook, mutator_hook):
        context = _context(
            bindings,
            evaluator_terms={
                "fold": {"_type": "native_resume_evaluator"},
            },
            mutator_terms={
                "mutate": {"_type": "native_resume_mutator"},
            },
            operation_hooks={
                "fold": evaluator_hook,
                "mutate": mutator_hook,
            },
        )
        return (
            EvaluatorRegistry.build_operation(
                "native_resume_evaluator",
                alias="fold",
                options={},
                context=context,
            ),
            MutatorRegistry.build_operation(
                "native_resume_mutator",
                alias="mutate",
                options={},
                context=context,
            ),
        )

    original = build(
        NativeOperationHooks(
            configure=_evaluator_configure,
            identity="test/resume/evaluator/v1",
        ),
        NativeOperationHooks(
            configure=_mutator_configure,
            identity="test/resume/mutator/v1",
        ),
    )
    changed_identity = build(
        NativeOperationHooks(
            configure=_evaluator_configure,
            identity="test/resume/evaluator/v2",
        ),
        NativeOperationHooks(
            configure=_mutator_configure,
            identity="test/resume/mutator/v2",
        ),
    )
    changed_source = build(
        NativeOperationHooks(
            configure=_changed_evaluator_configure,
            identity="test/resume/evaluator/v1",
        ),
        NativeOperationHooks(
            configure=_mutator_configure,
            identity="test/resume/mutator/v1",
        ),
    )

    assert (
        changed_identity[0].descriptor.identity
        != original[0].descriptor.identity
    )
    assert (
        changed_source[0].descriptor.identity
        != original[0].descriptor.identity
    )
    assert (
        changed_identity[1].descriptor.identity
        != original[1].descriptor.identity
    )

    runtime_root = tmp_path / "runtime"
    runtime = Runtime((original[0],), root=runtime_root, data=bindings.data)
    runtime.close()
    for changed in (
        changed_identity[0],
        changed_source[0],
    ):
        with pytest.raises(ContractConflict, match="scientific contract"):
            Runtime((changed,), root=runtime_root, data=bindings.data)

    mutator_runtime_root = tmp_path / "mutator-runtime"
    mutator_runtime = Runtime(
        (original[1],),
        root=mutator_runtime_root,
        data=bindings.data,
    )
    mutator_runtime.close()
    with pytest.raises(ContractConflict, match="scientific contract"):
        Runtime(
            (changed_identity[1],),
            root=mutator_runtime_root,
            data=bindings.data,
        )


def test_hook_without_source_fails_before_model_construction(tmp_path):
    bindings = _minimal_bindings(tmp_path)
    loads = []

    def fake_evaluator(*args, **kwargs):
        loads.append("load")
        raise AssertionError("source validation must precede construction")

    EvaluatorRegistry.register("native_builtin_hook", fake_evaluator)
    context = _context(
        bindings,
        evaluator_terms={"fold": {"_type": "native_builtin_hook"}},
        operation_hooks={
            "fold": NativeOperationHooks(
                configure=len,
                identity="test/hook/builtin/v1",
            )
        },
    )
    with pytest.raises(ValueError, match="source is unavailable"):
        EvaluatorRegistry.build_operation(
            "native_builtin_hook",
            alias="fold",
            options={},
            context=context,
        )
    assert loads == []


def test_mutator_hook_attaches_frozen_mutation_context_policy(tmp_path):
    bindings = _minimal_bindings(tmp_path)

    def fake_mutator(*args, **kwargs):
        raise AssertionError("descriptor checks must not construct models")

    MutatorRegistry.register("native_context_mutator", fake_mutator)
    policy = MutationContextPolicy(
        "test/native-registry-mutation-context/v1",
        NATIVE_MUTATION_CONTEXT_SCHEMA,
    )
    context = _context(
        bindings,
        mutator_terms={"mutate": {"_type": "native_context_mutator"}},
        operation_hooks={
            "mutate": NativeOperationHooks(
                configure=_mutator_configure,
                identity="test/native-registry-hook/v1",
                mutation_context=policy,
            )
        },
    )

    operation = MutatorRegistry.build_operation(
        "native_context_mutator",
        alias="mutate",
        options={},
        context=context,
    )

    assert operation.mutation_context is policy
    assert NATIVE_MUTATION_CONTEXT_CODEC.identity in {
        codec.identity for codec in operation.reference_codecs
    }
    assert operation.owner._value is None


def test_frozen_invoker_hook_changes_descriptor_and_stays_cold(tmp_path):
    bindings = _minimal_bindings(tmp_path)
    loads = []

    def fake_evaluator(*args, **kwargs):
        loads.append("evaluator")
        raise AssertionError("invoker descriptor checks must remain cold")

    def fake_mutator(*args, **kwargs):
        loads.append("mutator")
        raise AssertionError("invoker descriptor checks must remain cold")

    def evaluator_invoke_a(evaluator, inputs, runtime_config):
        return evaluator.evaluate(inputs, runtime_config)

    def evaluator_invoke_b(evaluator, inputs, runtime_config):
        value = evaluator.evaluate(inputs, runtime_config)
        return value

    def mutator_invoke_a(mutator, protomers, scores, runtime_config):
        return mutator.mutate(protomers, scores, runtime_config)

    def mutator_invoke_b(mutator, protomers, scores, runtime_config):
        value = mutator.mutate(protomers, scores, runtime_config)
        return value

    EvaluatorRegistry.register("native_invoker_evaluator", fake_evaluator)
    MutatorRegistry.register("native_invoker_mutator", fake_mutator)

    def build(evaluator_invoker, mutator_invoker):
        context = _context(
            bindings,
            evaluator_terms={
                "fold": {"_type": "native_invoker_evaluator"},
            },
            mutator_terms={
                "mutate": {"_type": "native_invoker_mutator"},
            },
            operation_hooks={
                "fold": NativeOperationHooks(
                    configure=_evaluator_configure,
                    identity="test/invoker/evaluator-configure/v1",
                    invoke=evaluator_invoker,
                    invoke_identity="test/invoker/evaluator/v1",
                ),
                "mutate": NativeOperationHooks(
                    configure=_mutator_configure,
                    identity="test/invoker/mutator-configure/v1",
                    invoke=mutator_invoker,
                    invoke_identity="test/invoker/mutator/v1",
                ),
            },
        )
        return (
            EvaluatorRegistry.build_operation(
                "native_invoker_evaluator",
                alias="fold",
                options={},
                context=context,
            ),
            MutatorRegistry.build_operation(
                "native_invoker_mutator",
                alias="mutate",
                options={},
                context=context,
            ),
        )

    first = build(evaluator_invoke_a, mutator_invoke_a)
    repeated = build(evaluator_invoke_a, mutator_invoke_a)
    changed_source = build(evaluator_invoke_b, mutator_invoke_b)

    assert first[0].descriptor.identity == repeated[0].descriptor.identity
    assert first[1].descriptor.identity == repeated[1].descriptor.identity
    assert first[0].descriptor.identity != changed_source[0].descriptor.identity
    assert first[1].descriptor.identity != changed_source[1].descriptor.identity
    assert first[0].owner.identity != changed_source[0].owner.identity
    assert first[1].owner.identity != changed_source[1].owner.identity
    assert first[0].native_invoke is evaluator_invoke_a
    assert first[1].native_invoke is mutator_invoke_a
    assert {
        codec.identity for codec in first[0].reference_codecs
    } == {
        codec.identity for codec in changed_source[0].reference_codecs
    }
    assert {
        codec.identity for codec in first[1].reference_codecs
    } == {
        codec.identity for codec in changed_source[1].reference_codecs
    }
    assert loads == []
    assert all(
        operation.owner._value is None
        for pair in (first, repeated, changed_source)
        for operation in pair
    )


def test_invoker_hook_identity_is_required_with_callback():
    with pytest.raises(ValueError, match="invoke and invoke_identity"):
        NativeOperationHooks(
            configure=_evaluator_configure,
            identity="test/invoker/configure/v1",
            invoke=lambda evaluator, inputs, runtime_config: None,
        )
    with pytest.raises(ValueError, match="invoke and invoke_identity"):
        NativeOperationHooks(
            configure=_evaluator_configure,
            identity="test/invoker/configure/v1",
            invoke_identity="test/invoker/v1",
        )


class _FakeEvaluator:
    def __init__(self, path: Path):
        self.path = path
        self.calls = []

    def evaluate(self, inputs, runtime_config=None):
        self.calls.append(dict(inputs))
        self.path.write_bytes(b"synthetic registry structure\n")
        return StandardScores(
            names=list(inputs),
            plddt={"A": [np.asarray([0.3, 0.7], dtype=np.float32)]},
            complex_paths={"A": self.path},
        )


@pytest.mark.parametrize(
    ("assemblies", "expected"),
    (
        (("A", "B"), ("A", "B")),
        (("AB",), ("AB",)),
    ),
)
def test_registered_evaluator_receives_translation_assembly_identities(
    tmp_path,
    assemblies,
    expected,
):
    global_config.protomers["A"] = Protomer("A", "AC", "p")
    global_config.protomers["B"] = Protomer("B", "DE", "p")
    data = DataStore(tmp_path / "data")
    legacy = LegacyContext()
    translation = legacy.capture_candidate(assemblies=assemblies, data=data)
    bindings = NativeBindings(legacy, translation, data, tmp_path / "work")
    observed = []

    def recording_factory(
        term_conf,
        *,
        alias,
        raw_model_results_dir,
        oligo_names,
        evaluator_conf,
    ):
        del term_conf, alias, raw_model_results_dir, evaluator_conf
        observed.append(tuple(oligo_names))
        return _FakeEvaluator(tmp_path / "constructor-argument.cif")

    EvaluatorRegistry.register("native_constructor_record", recording_factory)
    native_context = _context(
        bindings,
        evaluator_terms={
            "fold": {"_type": "native_constructor_record"},
        },
    )
    operation = EvaluatorRegistry.build_operation(
        "native_constructor_record",
        alias="fold",
        options={},
        context=native_context,
    )
    assert observed == []
    operation.initialization.invoke(
        operation.initialization.prepare(operation.initialization_request)
    )
    assert tuple(item.identity for item in translation.assemblies) == expected
    assert observed == [expected]


def test_registered_evaluator_loads_only_at_bootstrap_and_serves_typed_request(
    tmp_path,
):
    global_config.protomers["A"] = Protomer("A", "AC", "p")
    data = DataStore(tmp_path / "data")
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=("A",), data=data)
    bindings = NativeBindings(context, translation, data, tmp_path / "work")
    loads = []
    output = tmp_path / "prediction.cif"

    def fake_evaluator(
        term_conf,
        *,
        alias,
        raw_model_results_dir,
        oligo_names,
        evaluator_conf,
    ):
        loads.append((alias, dict(term_conf)))
        return _FakeEvaluator(output)

    EvaluatorRegistry.register("native_fake_eval", fake_evaluator)
    native_context = _context(
        bindings,
        evaluator_terms={"fold": {"_type": "native_fake_eval"}},
    )
    operation = EvaluatorRegistry.build_operation(
        "native_fake_eval",
        alias="fold",
        options={},
        context=native_context,
    )
    assert loads == []
    assert operation.owner._value is None

    request = PredictRequest(
        system=translation.system,
        candidate=translation.candidate,
        assemblies=("A",),
        conditioning=(),
        sources=(),
        samples=(
            Sample(
                protocol=key_for("protocol"),
                replicate="0",
                seed=7,
                index="1",
            ),
        ),
    )
    prepared_init = operation.initialization.prepare(
        operation.initialization_request
    )
    ready = operation.initialization.invoke(prepared_init)
    operation.initialization.on_result(ready)
    assert len(loads) == 1
    assert loads[0][0] == "fold"

    evidence = operation.invoke(operation.prepare(request))
    assert evidence.evaluated_candidate == translation.candidate.identity
    assert output.is_file()
    assert len(loads) == 1


class _FakeMutator:
    def mutate(self, protomers, scores, runtime_config):
        return None

    def shutdown(self):
        return None


def test_registered_mutator_is_lazy_and_binds_context_sampling_policy(tmp_path):
    bindings = _minimal_bindings(tmp_path)
    loads = []

    def fake_mutator(
        term_conf,
        *,
        alias,
        mutator_conf,
        raw_model_results_dir,
    ):
        loads.append(alias)
        return _FakeMutator()

    MutatorRegistry.register("native_fake_mutator", fake_mutator)
    context = _context(
        bindings,
        mutator_terms={"mutate": {"_type": "native_fake_mutator"}},
    )
    operation = MutatorRegistry.build_operation(
        "native_fake_mutator",
        alias="mutate",
        options={},
        context=context,
    )
    assert loads == []
    assert operation.sampling_policy == context.sampling_policy
    operation.initialization.invoke(
        operation.initialization.prepare(operation.initialization_request)
    )
    assert loads == ["mutate"]
