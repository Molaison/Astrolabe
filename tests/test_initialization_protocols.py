"""F22: deterministic native-init, MSA publish and provider pollution faults.

All native/provider modules are doubles. No scientific backend is loaded here.
"""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import random
import sys
from types import SimpleNamespace

import pytest

from astrolabe import optimized_backends as optimized
from astrolabe.core import global_config
from astrolabe.pre_eval_hooks import af3_antibody_msa_cache as msa
from astrolabe.utils import pyrosetta_init as native


@pytest.fixture(autouse=True)
def isolated_owners(monkeypatch):
    monkeypatch.setattr(native, "_OWNER", native.PyRosettaInitializationOwner())
    for name in ("_ACTIVATION_CACHE", "_PROCESS_ACTIVE", "_PROCESS_STATES", "_PROCESS_FAILURES"):
        monkeypatch.setattr(optimized, name, {})


def fake_native(*, initialized=False, failure=None):
    calls = []
    module = SimpleNamespace(
        initialized=initialized, version=lambda: "test-native-v1",
        _rosetta_database_from_env=lambda: "/test/database",
    )

    def init(options):
        calls.append(options)
        module.initialized = True  # irreversible side effect before a failure
        if failure is not None:
            raise failure

    module.init = init
    module.is_initialized = lambda: module.initialized
    return module, calls


def test_same_explicit_init_reuses_immutable_effective_receipt_and_rng(tmp_path):
    module, calls = fake_native()
    params = tmp_path / "ligand.params"
    params.write_text("NAME LIG\n")
    options = f"-mute all -constant_seed -jran 701 -extra_res_fa {params}"
    rng = random.getstate()
    receipt = native.ensure_pyrosetta_initialized(module, options, explicit=True)
    again = native.ensure_pyrosetta_initialized(module, options, explicit=True, require_attested=True)
    assert again is receipt
    assert calls == [options]
    assert random.getstate() == rng
    assert receipt.options == tuple(options.split())
    assert receipt.params == ((str(params), hashlib.sha256(params.read_bytes()).hexdigest()),)
    assert (receipt.version, receipt.database, receipt.state) == ("test-native-v1", "/test/database", "READY")
    with pytest.raises(FrozenInstanceError):
        receipt.version = "invented"


@pytest.mark.parametrize("changed", ["-beta", "-constant_seed -jran 702", "-ignore_zero_occupancy false"])
def test_conflicting_explicit_options_fail_without_second_init_or_score(changed):
    module, calls = fake_native()
    native.ensure_pyrosetta_initialized(module, "-mute all -constant_seed -jran 701", explicit=True)
    scored = []
    with pytest.raises(native.PyRosettaInitializationError, match="conflicting explicit"):
        native.ensure_pyrosetta_initialized(module, f"-mute all {changed}", explicit=True)
        scored.append("score")
    assert len(calls) == 1
    assert scored == []


@pytest.mark.parametrize("failure", [RuntimeError("partial init"), KeyboardInterrupt("cut init")])
def test_partial_init_poison_survives_native_initialized_flag_and_runtime_reset(failure):
    module, calls = fake_native(failure=failure)
    with pytest.raises(type(failure)):
        native.ensure_pyrosetta_initialized(module, "-mute all")
    global_config.reset()
    with pytest.raises(native.PyRosettaInitializationError, match="POISONED"):
        native.ensure_pyrosetta_initialized(module, "-mute all -beta")
    assert module.initialized
    assert calls == ["-mute all"]
    assert native._OWNER.receipt is None


def test_external_init_is_legacy_compatible_but_not_attested():
    module, calls = fake_native(initialized=True)
    receipt = native.ensure_pyrosetta_initialized(module, "-beta", explicit=True)
    assert receipt.state == "INITIALIZED_UNATTESTED"
    assert receipt.options is None
    with pytest.raises(native.PyRosettaInitializationError, match="UNATTESTED"):
        native.ensure_pyrosetta_initialized(module, "-beta", require_attested=True)
    assert calls == []


def test_params_change_or_missing_requirement_fails_before_score(tmp_path):
    module, calls = fake_native()
    params = tmp_path / "ligand.params"
    params.write_text("NAME LIG\n")
    options = f"-mute all -extra_res_fa {params}"
    receipt = native.ensure_pyrosetta_initialized(module, options)
    params.write_text("NAME OTHER\n")
    with pytest.raises(native.PyRosettaInitializationError, match="params identity"):
        native.ensure_pyrosetta_initialized(module, options)
    assert native._OWNER.receipt is receipt
    assert len(calls) == 1


def test_preinit_missing_params_is_correctable_without_native_side_effects(tmp_path):
    module, calls = fake_native()
    params = tmp_path / "missing.params"
    options = f"-extra_res_fa {params}"
    with pytest.raises(FileNotFoundError):
        native.ensure_pyrosetta_initialized(module, options)
    assert calls == []
    params.write_text("NAME LIG\n")
    native.ensure_pyrosetta_initialized(module, options, params=[params])
    assert calls == [options]


@pytest.mark.parametrize("identity", ["version", "database"])
def test_native_identity_drift_rejects_reuse(identity):
    module, calls = fake_native()
    native.ensure_pyrosetta_initialized(module, "-mute all")
    if identity == "version":
        module.version = lambda: "different"
    else:
        module._rosetta_database_from_env = lambda: "/different/database"
    with pytest.raises(native.PyRosettaInitializationError, match=identity):
        native.ensure_pyrosetta_initialized(module, "-mute all")
    assert len(calls) == 1


def test_default_first_party_ensures_reuse_stronger_attested_owner(monkeypatch, tmp_path):
    import run
    from astrolabe.evaluators.dummy import DummyEvaluator
    from astrolabe.losses import design_screening
    from astrolabe.post_eval_hooks import rosetta_minimize
    from astrolabe.posthoc.rosetta_direct_channel import init_pyrosetta_once

    module, calls = fake_native()
    monkeypatch.setitem(sys.modules, "pyrosetta", module)
    monkeypatch.setattr(run, "setup_extra_residues", lambda **kwargs: None)
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    options = "-mute all -beta -constant_seed -jran 701"
    native.ensure_pyrosetta_initialized(module, options, explicit=True)
    run._init_pyrosetta_runtime({})
    design_screening._ensure_pyrosetta()
    assert rosetta_minimize._ensure_pyrosetta_initialized() is module
    assert DummyEvaluator._ensure_pyrosetta_initialized() is module
    assert init_pyrosetta_once() is module
    assert calls == [options]
    # Explicitly passing even a default value must not be mistaken for ensure-only.
    with pytest.raises(native.PyRosettaInitializationError, match="conflicting explicit"):
        design_screening.physical_interface_scores(
            tmp_path / "never_loaded.pdb", chain_a="A", chain_b="B",
            init_opts="-ignore_unrecognized_res -ignore_zero_occupancy -mute all",
        )
    global_config.reset()


def test_run_new_generated_params_are_requirements_not_silently_ignored(monkeypatch, tmp_path):
    import run
    module, calls = fake_native()
    monkeypatch.setitem(sys.modules, "pyrosetta", module)
    monkeypatch.setattr(run, "setup_extra_residues", lambda **kwargs: None)
    global_config.reset()
    global_config.init_const("out_dir", tmp_path)
    native.ensure_pyrosetta_initialized(module, "-mute all")
    (tmp_path / "raw_model_results").mkdir()
    (tmp_path / "raw_model_results/lig.params").write_text("NAME LIG\n")
    with pytest.raises(RuntimeError, match="params identity"):
        run._init_pyrosetta_runtime({})
    assert calls == ["-mute all"]
    global_config.reset()


def test_crystal_rechecks_explicit_seed_even_after_local_handle_cache(monkeypatch, tmp_path):
    from astrolabe.evaluators.rosetta_crystal import RosettaCrystalEvaluator
    module, calls = fake_native()
    monkeypatch.setitem(sys.modules, "pyrosetta", module)
    params = tmp_path / "lig.params"
    params.write_text("NAME LIG\n")
    evaluator = RosettaCrystalEvaluator(
        states={"minus2": {"params": str(params)}}, raw_model_results_dir=tmp_path,
    )
    assert evaluator._ensure_pyrosetta() is module
    assert evaluator._ensure_pyrosetta() is module
    evaluator.random_seed += 1
    with pytest.raises(native.PyRosettaInitializationError, match="conflicting explicit"):
        evaluator._ensure_pyrosetta()
    assert len(calls) == 1
    assert "-constant_seed -jran 701" in calls[0]


@pytest.mark.parametrize("bad", ["{broken", json.dumps({"sequence": "AAAA"}),
    json.dumps({"chain_id": "A", "sequence": "AAAA", "templates": [
        {"queryIndices": [99], "templateIndices": [0]}]})])
def test_msa_mid_load_failure_never_publishes_partial_entries_and_can_retry(tmp_path, bad):
    cache = msa.AntibodyMsaCache(cache_dir=tmp_path)
    old = msa.MsaCacheEntry(chain_id="A", sequence="CCCC", unpaired_msa=">old\nCCCC\n", paired_msa=None, templates=())
    cache.add_entry(old)
    previous = cache._entries
    (tmp_path / "00_good.json").write_text(json.dumps({
        "chain_id": "A", "sequence": "AAAA", "unpaired_msa": ">q\nAAAA\n", "paired_msa": ">stale\nAAAA\n",
    }))
    broken = tmp_path / "01_bad.json"
    broken.write_text(bad)
    with pytest.raises((ValueError, KeyError)):
        cache.load()
    assert not cache._loaded
    assert cache._entries is previous
    assert cache.find_nearest(chain_id="A", sequence="AAAA", max_distance=0) is None
    assert cache.find_nearest(chain_id="A", sequence="CCCC", max_distance=0)[0] is old
    broken.unlink()
    cache.load()
    assert cache._loaded
    hit, distance = cache.find_nearest(chain_id="A", sequence="AAAT", max_distance=1)
    assert distance == 1 and hit.sequence == "AAAA" and hit.paired_msa is None
    chain = msa.ProteinChain(id="A", sequence="AAAT", ptms=[], paired_msa=">old\nAAAA\n")
    replacement = msa._replace_chain_payload(chain, hit, preserve_existing_templates=False, expected_template_count=None)
    assert replacement.unpaired_msa == ">q\nAAAT\n"
    assert replacement.paired_msa == ""


@pytest.mark.parametrize("outcome", [None, {"active": True},
    {"active": True, "applied": "deferred"}, {"active": True, "mode": "fast", "applied": "configured"},
    RuntimeError("after patch"), KeyboardInterrupt("cut hook")])
def test_provider_side_effect_then_invalid_return_or_verify_failure_stays_poisoned_after_reset(monkeypatch, outcome):
    selection = optimized.build_kit_selection("protenix", "exact", verify_source=False)
    calls = []
    patched = []

    def hook(mode, **kwargs):
        assert optimized._PROCESS_STATES[selection.activation_api.module] == "APPLYING"
        calls.append(mode)
        patched.append("module patched")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setitem(sys.modules, selection.activation_api.module, SimpleNamespace(enable=hook))
    error = KeyboardInterrupt if isinstance(outcome, KeyboardInterrupt) else optimized.ActivationNotActiveError
    with pytest.raises(error):
        optimized.activate_provider(selection)
    optimized.reset_activation_state()
    for request in (selection, replace(selection, mode="fast"), replace(selection, mode="off")):
        with pytest.raises(optimized.ProviderPoisonedError, match="POISONED"):
            optimized.activate_provider(request)
    assert calls == ["exact"] and patched == ["module patched"]


def test_provider_active_reset_reuses_receipt_without_reapplying_and_rejects_off(monkeypatch):
    selection = optimized.build_kit_selection("protenix", "exact", verify_source=False)
    calls = []

    def hook(mode, **kwargs):
        calls.append(mode)
        return {"active": True, "applied": "configured", "mode": mode}

    monkeypatch.setitem(sys.modules, selection.activation_api.module, SimpleNamespace(enable=hook))
    report = optimized.activate_provider(selection)
    optimized.reset_activation_state()
    assert optimized.activate_provider(selection) is report
    for request in (replace(selection, mode="fast"), replace(selection, mode="off")):
        with pytest.raises(optimized.ActivationConflictError):
            optimized.activate_provider(request)
    assert calls == ["exact"]


def test_provider_prepare_failure_is_correctable_before_hook_side_effects(monkeypatch):
    selection = optimized.build_kit_selection("protenix", "exact", verify_source=False)
    module = SimpleNamespace()
    monkeypatch.setitem(sys.modules, selection.activation_api.module, module)
    with pytest.raises(optimized.UnsupportedActivationError, match="absent"):
        optimized.activate_provider(selection)
    optimized.reset_activation_state()
    module.enable = lambda *args, **kwargs: {"active": True, "applied": "configured"}
    assert optimized.activate_provider(selection).active
