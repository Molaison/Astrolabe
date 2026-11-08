"""Actual AF3 wrapper functions with synthetic devices; no model inference."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def wrapper():
    pytest.importorskip("alphafold3")
    from astrolabe.evaluators.wrappers import alphafold3
    return alphafold3


@pytest.mark.parametrize("device,backend,index", [(2, "gpu", 2), ("cpu", "cpu", 0)])
def test_setup_places_on_configured_device_and_binds_cycles(wrapper, monkeypatch, device, backend, index):
    enumerated = []
    devices = [SimpleNamespace(platform=backend, id=i) for i in range(3)]

    def local_devices(*, backend):
        enumerated.append(backend)
        return devices

    def runner(**fields):
        return SimpleNamespace(
            **fields,
            set_cyclic_asym_ids=lambda value: cycles.append(tuple(value)),
            set_cyclic_relative_patches=lambda value: patches.append(tuple(map(tuple, value))),
        )

    cycles, patches = [], []
    monkeypatch.setattr(wrapper.jax, "local_devices", local_devices)
    monkeypatch.setattr(wrapper, "_configure_refstructure_env", lambda _: None)
    monkeypatch.setattr(wrapper, "_configure_msa_env", lambda _: None)
    monkeypatch.setattr(wrapper, "make_model_config", lambda **fields: fields)
    monkeypatch.setattr(wrapper, "ModelRunner", runner)
    config = OmegaConf.create(dict(
        gpu_device=device, model_dir="unchanged", flash_attention_implementation="cudnn",
        num_samples=3, num_recycles=7, save_embeddings=False,
        cyclic_asym_ids=[1], cyclic_relative_patches=[[1, 2]],
    ))
    loaded = wrapper.alphafold3_setup(config)
    assert enumerated == [backend]
    assert loaded.device is devices[index]
    assert loaded.config["num_diffusion_samples"] == 3
    assert loaded.config["num_recycles"] == 7
    assert cycles == [(1,)] and patches == [((1, 2),)]


@pytest.mark.parametrize("unavailable", [False, True])
def test_explicit_device_unavailable_fails_before_constructing_model(wrapper, monkeypatch, unavailable):
    def devices(**kwargs):
        if unavailable:
            raise RuntimeError("no GPU")
        return [object()]

    monkeypatch.setattr(wrapper.jax, "local_devices", devices)
    monkeypatch.setattr(wrapper, "_configure_refstructure_env", lambda _: None)
    monkeypatch.setattr(wrapper, "_configure_msa_env", lambda _: None)
    with pytest.raises((RuntimeError, ValueError), match="Configured AF3 device"):
        wrapper.alphafold3_setup(OmegaConf.create({"gpu_device": 2}))


def test_weights_placed_and_completed_once_before_model_execution(wrapper, monkeypatch):
    owner = wrapper.ModelRunner.__new__(wrapper.ModelRunner)
    owner._device = object()
    owner.model_params = object()
    placed = object()
    events = []
    monkeypatch.setattr(wrapper.hk, "transform", lambda fn: SimpleNamespace(apply=fn))
    monkeypatch.setattr(wrapper.jax, "device_put", lambda value, device:
                        events.append(("place", value, device)) or placed)
    monkeypatch.setattr(wrapper.jax, "block_until_ready", lambda value:
                        events.append(("ready", value)))
    monkeypatch.setattr(wrapper.jax, "jit", lambda fn, **kwargs: fn)
    forward = owner._model
    assert owner._model is forward
    assert forward.args == (placed,)
    assert events == [("place", owner.model_params, owner._device), ("ready", placed)]


def test_seed_transfer_completes_before_model_and_extraction(wrapper, monkeypatch):
    from astrolabe.evaluators.runtime_stages import complete_stage
    from astrolabe import inference_backends
    from astrolabe.inference_pipeline import Pending

    owner = wrapper.ModelRunner.__new__(wrapper.ModelRunner)
    owner._device = object()
    seed, placed_seed = object(), object()
    events = []
    owner.prepare_features = lambda batch: batch
    owner.upload_features = lambda features: features
    owner.run_model = lambda features, key: events.append(("infer", key)) or features
    owner.materialize_result = lambda value: events.append(("materialize",)) or value
    owner.extract_structures = lambda batch, value, name: events.append(("extract", name)) or [value]
    monkeypatch.setattr(wrapper.jax, "device_put", lambda value, device:
                        events.append(("seed_upload", value, device)) or placed_seed)
    monkeypatch.setattr(inference_backends, "jax_pending", lambda value, **kwargs:
                        Pending(value, lambda: events.append(("complete",))))
    value = {"features": 1}
    for stage in owner.inference_stages(seed, target_name="target"):
        value = complete_stage(stage, value)
    assert value == [{"features": 1}]
    assert events == [
        ("seed_upload", seed, owner._device), ("complete",), ("infer", placed_seed),
        ("complete",), ("materialize",), ("extract", "target"),
    ]


def test_predict_structure_uses_owner_seed_and_extraction(wrapper, monkeypatch):
    from astrolabe.evaluators import alphafold3

    fold_input = SimpleNamespace(rng_seeds=[17], user_ccd=None, name="target")
    batch, extracted, calls = {"value": 1}, [object()], []
    owner = SimpleNamespace(run_inference_and_extract_for_seed=lambda example, seed, name:
                            calls.append((example, seed, name)) or extracted)
    monkeypatch.setattr(alphafold3, "jax", SimpleNamespace())
    monkeypatch.setattr(alphafold3.chemical_components, "Ccd", lambda **kwargs: None)
    monkeypatch.setattr(alphafold3.featurisation, "featurise_input", lambda **kwargs: [batch])
    monkeypatch.setattr(alphafold3, "allow_all_x_protein_chains_during_featurisation",
                        lambda _: nullcontext())
    monkeypatch.setattr(alphafold3.AlphaFold3Evaluator, "_apply_query_residue_index_jumps",
                        lambda examples, *_: examples)
    result = alphafold3.AlphaFold3Evaluator.predict_structure(fold_input, owner, (128,))
    assert calls == [(batch, 17, "target")]
    assert result[0].seed == 17
    assert result[0].inference_results is extracted
