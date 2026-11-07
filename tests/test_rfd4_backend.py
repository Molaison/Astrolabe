"""Critical cache publication and native boundary checks, without model inference."""
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import os
import sys

import pytest

from astrolabe.mutators.rfd4_backend import RFD4Backend, native_interaction_metrics, validate_generation
from astrolabe.mutators.rfd4_types import ChainSpec, ConditionPlan, GenerationRequest, ResidueRef


def _backend(tmp_path, monkeypatch):
    source, model = tmp_path / "upstream", tmp_path / "model"
    for name in ("src/rfproteina/generate.py", "configs/inference.yaml", "pyproject.toml"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    model.mkdir()
    for name in ("model.safetensors", "config.json"):
        (model / name).write_text("fixture")
    calls = tmp_path / "model_calls"
    class Client:
        def __init__(self, *args, **kwargs):
            with calls.open("a") as stream:
                stream.write("load\n")
        def request(self, payload):
            root = Path(payload["root"])
            assert not (tmp_path / "cache" / "generated.cif").exists()
            (root / "native.cif").write_text("generated full structure")
            with calls.open("a") as stream:
                stream.write("generate\n")
            return {"structure_path": str(root / "native.cif")}
        def shutdown(self):
            pass
    monkeypatch.setattr("astrolabe.model_execution.NativeModelClient", Client)
    backend = RFD4Backend(python_bin=sys.executable, source_dir=source,
                          model_dir=model, cache_dir=tmp_path / "cache")
    def fingerprint():
        return {"source": hashlib.sha256((source / "src/rfproteina/generate.py").read_bytes()).hexdigest(),
                "weight": hashlib.sha256((model / "model.safetensors").read_bytes()).hexdigest(),
                "generation": dict(backend.generation)}
    def cpu(operation, payload):
        root = Path(payload["root"])
        if operation == "prepare":
            (root / "annotated_input.cif").write_text("annotated full input")
            return {"mapping": ["A", "B"]}
        assert operation == "validate"
        if (root / "generated.cif").read_text() != "generated full structure":
            raise ValueError("invalid generated structure")
        return {"mapping": ["A", "B"], "sequences": [["A", "A"], ["B", "G"]]}
    monkeypatch.setattr(backend, "fingerprint", fingerprint)
    monkeypatch.setattr(backend, "_cpu", cpu)
    path = tmp_path / "input.pdb"
    path.write_text("conformer one")
    request = GenerationRequest(str(path), "AB", (ChainSpec("A", "binder", "A"), ChainSpec("B", "target", "G")),
                                ConditionPlan((ResidueRef("A", 1),), (ResidueRef("B", 1),)), 12)
    return backend, request, calls


@pytest.mark.parametrize("inherited", ["", "/foreign/compiler-cache"])
def test_worker_uses_task_owned_compiler_caches(tmp_path, monkeypatch, inherited):
    backend, _, _ = _backend(tmp_path, monkeypatch)
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.setenv(name, inherited)
    environment = backend._env()
    for name, directory in (("TRITON_CACHE_DIR", "triton"),
                            ("TORCHINDUCTOR_CACHE_DIR", "torchinductor")):
        expected = backend.cache_dir / ".compiler" / directory
        assert environment[name] == str(expected)
        assert expected.is_dir()
        assert os.environ[name] == inherited


def test_cache_hit_never_starts_native_model_and_corruption_is_preserved(tmp_path, monkeypatch):
    backend, request, calls = _backend(tmp_path, monkeypatch)
    first = backend.generate(request)
    backend.shutdown()
    assert not first.cache_hit
    def forbidden(*args, **kwargs):
        raise AssertionError("Cache hit attempted model startup")
    monkeypatch.setattr("astrolabe.model_execution.NativeModelClient", forbidden)
    hit = backend.generate(request)
    assert hit.cache_hit and hit.structure_digest == first.structure_digest
    assert calls.read_text().splitlines() == ["load", "generate"]
    Path(hit.structure_path).write_text("corrupted output")
    with pytest.raises(RuntimeError, match="checksum"):
        backend.generate(request)
    assert Path(hit.structure_path).read_text() == "corrupted output"


def test_content_seed_plan_and_actual_asset_bytes_invalidate_cache(tmp_path, monkeypatch):
    backend, request, calls = _backend(tmp_path, monkeypatch)
    results = [backend.generate(request)]
    Path(request.structure_path).write_text("conformer two, same sequence")
    results.append(backend.generate(request))
    results.append(backend.generate(replace(request, seed=13)))
    results.append(backend.generate(replace(request, plan=replace(request.plan, feedback_json='{"energy":-1}'))))
    backend.shutdown()
    (backend.model_dir / "model.safetensors").write_text("new actual weights")
    results.append(backend.generate(request))
    assert len({result.request_digest for result in results}) == 5
    assert all(not result.cache_hit for result in results)


def test_variable_length_motif_geometry_and_lineage_are_cache_identity(tmp_path, monkeypatch):
    backend, request, calls = _backend(tmp_path, monkeypatch)
    backend.generation = validate_generation({"mode": "inpaint"})
    reconstruction = {
        "chain": "A", "length_min": 2, "length_max": 5,
        "motifs": [{"id": "lineage-one", "source": {"chain": "A", "position": 2}, "atoms": ["CA"]}],
    }
    request = replace(request, chains=(ChainSpec("A", "binder", "AAA"),
                                       ChainSpec("B", "target", "CFF", "ligand")),
                      plan=ConditionPlan(tuple(ResidueRef("A", i) for i in range(1, 4)),
                                         (ResidueRef("B", 1),), reconstruction=reconstruction))
    variants = [reconstruction, {**reconstruction, "length_max": 6},
                {**reconstruction, "motifs": [{**reconstruction["motifs"][0], "atoms": ["CA", "C"]}]},
                {**reconstruction, "motifs": [{**reconstruction["motifs"][0], "id": "lineage-two"}]}]
    outputs = [backend.generate(replace(request, plan=replace(request.plan, reconstruction=value)))
               for value in variants]
    assert len({r.request_digest for r in outputs}) == 4
    backend.shutdown()
    monkeypatch.setattr("astrolabe.model_execution.NativeModelClient",
                        lambda *a, **k: pytest.fail("cached reconstruction started a model"))
    assert backend.generate(request).cache_hit
    assert calls.read_text().splitlines().count("generate") == 4


def test_reconstruction_contract_rejects_before_model_start(tmp_path, monkeypatch):
    backend, request, calls = _backend(tmp_path, monkeypatch)
    request = replace(request, plan=replace(request.plan, reconstruction={
        "chain": "A", "length_min": 2, "length_max": 5, "motifs": []}))
    with pytest.raises(ValueError, match="bounded full-noise"):
        backend.generate(request)
    assert not calls.exists()


def test_failed_writer_has_no_visible_cache_entry(tmp_path, monkeypatch):
    backend, request, _ = _backend(tmp_path, monkeypatch)
    class Failure:
        def __init__(self, *args, **kwargs):
            pass
        def request(self, payload):
            raise RuntimeError("interrupted native generation")
        def shutdown(self):
            pass
    monkeypatch.setattr("astrolabe.model_execution.NativeModelClient", Failure)
    with pytest.raises(RuntimeError, match="interrupted"):
        backend.generate(request)
    entries = list(backend.cache_dir.iterdir())
    assert any(path.name.startswith(".") for path in entries)
    assert not any(path.is_dir() and not path.name.startswith(".") for path in entries)


def test_two_processes_publish_once_under_flock(tmp_path, monkeypatch):
    backend, request, calls = _backend(tmp_path, monkeypatch)
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    def run():
        try:
            result = backend.generate(request)
            queue.put((result.request_digest, result.cache_hit))
        except BaseException as error:
            queue.put(str(error))
    children = [ctx.Process(target=run) for _ in range(2)]
    for child in children:
        child.start()
    for child in children:
        child.join(15)
        assert child.exitcode == 0
    results = [queue.get(timeout=2) for _ in children]
    assert all(isinstance(result, tuple) for result in results), results
    assert results[0][0] == results[1][0]
    assert sorted(result[1] for result in results) == [False, True]
    assert calls.read_text().splitlines() == ["load", "generate"]


@pytest.mark.parametrize("settings", [
    {"typo": 1}, {"partial_t_backbone": 0.8, "partial_trajectory_fraction_steps": 0.5},
    {"nsteps": 30}, {"partial_t_latent": float("nan")}, {"partial_t_backbone": 1.0},
    {"partial_trajectory_fraction_steps": 0.001}, {"precision": "invented"},
    {"mode": "unknown"}, {"mode": None},
    {"mode": "inpaint", "nsteps": 30},
    {"mode": "inpaint", "recycle_strategy": "invented"},
])
def test_invalid_or_incompatible_native_settings_fail_explicitly(settings):
    with pytest.raises(ValueError):
        validate_generation(settings)


@pytest.mark.parametrize("settings", [None, {}, {"mode": "partial"}])
def test_partial_defaults_preserve_historical_generation_mapping(settings):
    assert validate_generation(settings) == {
        "nsteps": 60, "recycle_strategy": "medium/front_loaded", "precision": "bf16-mixed",
        "partial_t_backbone": 0.8, "partial_t_latent": 0.8,
    }


@pytest.mark.parametrize("settings", [
    {"partial_t_backbone": 0.6}, {"partial_t_latent": 0.7},
    {"partial_trajectory_fraction_steps": 0.5},
])
def test_explicit_partial_preserves_historical_custom_settings(settings):
    assert validate_generation({"mode": "partial", **settings}) == validate_generation(settings)


def test_inpaint_resolves_native_full_noise_settings():
    assert validate_generation({"mode": "inpaint"}) == {
        "mode": "inpaint", "nsteps": 60, "recycle_strategy": "medium/front_loaded",
        "precision": "bf16-mixed",
    }


@pytest.mark.parametrize("key", [
    "partial_t_backbone", "partial_t_latent", "partial_trajectory_fraction_steps",
])
@pytest.mark.parametrize("value", [None, 0.5])
def test_inpaint_rejects_supplied_partial_settings_even_null(key, value):
    with pytest.raises(ValueError, match="inpaint mode does not accept partial"):
        validate_generation({"mode": "inpaint", key: value})


def test_inpaint_mode_is_retained_in_cache_identity(tmp_path, monkeypatch):
    backend, request, calls = _backend(tmp_path, monkeypatch)
    partial = backend.generate(request)
    backend.shutdown()
    backend.generation = validate_generation({"mode": "partial"})
    assert backend.generate(request).cache_hit
    backend.generation = validate_generation({"mode": "inpaint"})
    inpaint = backend.generate(request)
    backend.shutdown()
    assert not inpaint.cache_hit
    assert inpaint.request_digest != partial.request_digest
    manifest = json.loads(Path(inpaint.manifest_path).read_text())
    assert manifest["fingerprint"]["generation"] == validate_generation({"mode": "inpaint"})
    assert backend.generate(request).cache_hit
    assert calls.read_text().splitlines() == ["load", "generate", "load", "generate"]


def test_fraction_does_not_silently_retain_default_modality_times():
    result = validate_generation({"partial_trajectory_fraction_steps": 0.5})
    assert result["partial_trajectory_fraction_steps"] == 0.5
    assert "partial_t_backbone" not in result and "partial_t_latent" not in result


def test_fixed_ligand_request_identity_and_single_residue(tmp_path, monkeypatch):
    backend, request, _ = _backend(tmp_path, monkeypatch)
    ligand = ChainSpec("B", "target", "CFF", kind="ligand")
    request = replace(request, chains=(request.chains[0], ligand))
    record, source = backend._request(request)
    assert ligand.residue_count == 1
    assert request.chains[0].residue_count == 1
    assert record["chains"][1] == {"chain": "B", "protomer": "target", "sequence": "CFF", "kind": "ligand"}
    assert record["source_sha256"] == hashlib.sha256(source).hexdigest()
    variants = [request,
                replace(request, chains=(request.chains[0], replace(ligand, kind="protein"))),
                replace(request, chains=(request.chains[0], replace(ligand, sequence="ATP")))]
    digests = {hashlib.sha256(json.dumps(backend._request(value)[0], sort_keys=True).encode()).hexdigest()
               for value in variants}
    assert len(digests) == len(variants)
    with pytest.raises(ValueError, match="never be released"):
        backend._request(replace(request, plan=ConditionPlan((ResidueRef("B", 1),), ())))
    with pytest.raises(ValueError, match="one-based ordinals"):
        backend._request(replace(request, plan=ConditionPlan((), (ResidueRef("B", 2),))))


@pytest.mark.parametrize("kind,ccd", [("rna", "CFF"), ("ligand", ""), ("ligand", "cff"),
                                       ("ligand", "CFF ATP"), ("ligand", "[CFF]")])
def test_invalid_ligand_declarations_fail(tmp_path, monkeypatch, kind, ccd):
    backend, request, _ = _backend(tmp_path, monkeypatch)
    request = replace(request, chains=(request.chains[0], ChainSpec("B", "target", ccd, kind=kind)))
    with pytest.raises(ValueError):
        backend._request(request)


def test_native_interaction_metrics_uses_cpu_protocol_without_backend(tmp_path, monkeypatch):
    import subprocess
    expected = {"hbond_atom_pair_count_coi": 2.0, "hydrophobic_atom_pair_count_coi": 4.0,
                "hbond_donor_conditioning_adherence": None}
    def run(command, **kwargs):
        assert command[-1] == "interaction_metrics"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert json.loads(kwargs["input"]) == {"structure_path": str(tmp_path / "input.cif")}
        return subprocess.CompletedProcess(command, 0,
                                           ("RFD4_RESULT=" + json.dumps(expected) + "\n").encode(), b"")
    monkeypatch.setattr("astrolabe.mutators.rfd4_backend.subprocess.run", run)
    assert native_interaction_metrics(tmp_path / "input.cif", python_bin=sys.executable,
                                      source_dir=tmp_path / "upstream") == expected
