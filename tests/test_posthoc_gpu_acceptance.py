from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.posthoc import refold, refold_receipts


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    REPO_ROOT / "scripts" / "validation" / "posthoc_gpu_acceptance.py"
)
CONTINUATION_PATH = (
    REPO_ROOT / "scripts" / "validation" / "posthoc_continuation.py"
)


def _load_harness():
    name = "posthoc_gpu_acceptance_test_module"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_continuation():
    name = "posthoc_continuation_test_module"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CONTINUATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


def _prepare(harness, root: Path) -> dict:
    return harness.prepare_acceptance(
        root / "acceptance",
        repo_root=REPO_ROOT,
        model_dir=Path("/models/alphafold3"),
    )


def _manifest(harness, root: Path) -> tuple[Path, dict]:
    path = root / "acceptance" / "inputs" / "manifest.jsonl"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write_harmless_cache_marker(harness, root: Path, item: dict) -> Path:
    caller_root = root / "acceptance" / "caller" / "runs"
    run_dir = harness._expected_run_dir(caller_root, item)
    marker = harness._model_done_marker(run_dir, item)
    marker.parent.mkdir(parents=True)
    marker.write_text('{"fixture": "CPU cache-marking only"}\n', encoding="utf-8")
    structure = harness._model_structure(run_dir, item)
    structure.parent.mkdir(parents=True, exist_ok=True)
    structure.write_text("data_cpu_cache_fixture\n#\n", encoding="utf-8")
    return marker


def _stub_receipt_assets(monkeypatch) -> None:
    monkeypatch.setattr(
        refold_receipts,
        "selected_asset_record",
        lambda *, backend, selected_config: {
            "schema": "test.refold-assets/v1",
            "complete": True,
            "components": [],
            "files": [
                {
                    "role": f"evaluator:{backend}",
                    "path": f"test://{backend}",
                    "sha256": "0" * 64,
                    "size_bytes": 0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        refold_receipts,
        "numeric_source_record",
        lambda backend, repo_root=None: {
            "schema": "test.refold-source/v1",
            "backend": backend,
            "files": [],
        },
    )


def _run_generated_sitecustomize(
    harness,
    root: Path,
    *,
    stub_mode: str,
    invalid_input: bool = False,
    jax_rng: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    capture = root / "capture"
    harness._install_capture(capture)
    package = root / "stubs" / "astrolabe"
    evaluators = package / "evaluators"
    wrappers = evaluators / "wrappers"
    wrappers.mkdir(parents=True)
    for path in (
        package / "__init__.py",
        evaluators / "__init__.py",
        wrappers / "__init__.py",
    ):
        path.write_text("", encoding="utf-8")
    (evaluators / "alphafold3.py").write_text(
        "\n".join(
            [
                "class AlphaFold3Evaluator:",
                "    def evaluate(self, inputs=None, runtime_config=None):",
                "        return {'plddt': 0.9}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (wrappers / "alphafold3.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "from pathlib import Path",
                "import numpy as np",
                "",
                "class ModelRunner:",
                "    def run_inference(self, featurised_example, rng_key):",
                "        root = Path(os.environ['ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE_DIR'])",
                "        with (root / 'stub_inner_calls.jsonl').open('a', encoding='utf-8') as handle:",
                "            handle.write(json.dumps({'called': True}) + '\\n')",
                "        if os.environ['TEST_STUB_MODE'] == 'raise':",
                "            raise RuntimeError('stub inner failure')",
                "        if os.environ['TEST_STUB_MODE'] == 'raw_object':",
                "            return {'object_result': np.asarray([object()], dtype=object)}",
                "        return {",
                "            'identifier': 'af3-stub',",
                "            'positions': np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32),",
                "            'confidence': np.float32(0.75),",
                "        }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_lines = [
            "import json",
            "import os",
            "from pathlib import Path",
            "import numpy as np",
            "from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner",
            "features = {",
            "    'token_features': {",
            "        'token_index': np.arange(20, dtype=np.int32),",
            "        'aatype': np.arange(20, dtype=np.int32),",
            "        'token_mask': np.ones((20,), dtype=np.bool_),",
            "    },",
            "    'atom_features': {",
            "        'atom_positions': np.zeros((20, 37, 3), dtype=np.float32),",
            "    },",
            "    'msa': np.tile(np.arange(20, dtype=np.int32), (4, 1)),",
            "    'metadata': {'name': 'single20', 'blob': b'stub' },",
            "}",
            "if " + ("True" if invalid_input else "False") + ":",
            "    features['object_dtype'] = np.asarray([object()], dtype=object)",
        ]
    if jax_rng:
        script_lines.extend(
            [
                "import jax",
                "rng_key = jax.random.PRNGKey(17)",
            ]
        )
    else:
        script_lines.append(
            "rng_key = np.asarray([[17, 0], [42, 1]], dtype=np.uint32)"
        )
    script_lines.extend(
        [
            "result = ModelRunner().run_inference(features, rng_key)",
            "Path('entrypoint_result.json').write_text(json.dumps({'ran': True, 'identifier': result['identifier']}))",
        ]
    )
    script = "\n".join(script_lines)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join([str(capture), str(root / "stubs")]),
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE": "1",
            "ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE_DIR": str(capture),
            "ASTROLABE_POSTHOC_ACCEPTANCE_STAGE": "entrypoint-cpu",
            "ASTROLABE_POSTHOC_ACCEPTANCE_MAX_REQUESTS": "1",
            "JAX_PLATFORMS": "cpu",
            "TEST_STUB_MODE": stub_mode,
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return completed, capture


def _run_generated_sitecustomize_with_real_af3(
    harness,
    root: Path,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    pytest.importorskip("alphafold3")
    prepared = _prepare(harness, root)
    prepared_path = root / "acceptance" / "prepared.json"
    capture = root / "real-af3-capture"
    harness._install_capture(capture)
    feature_work = root / "real-af3-feature-work"
    script = "\n".join(
        [
            "import json",
            "import os",
            "from pathlib import Path",
            "import sys",
            "",
            "import jax",
            "import numpy as np",
            "from alphafold3.constants import chemical_components",
            "from alphafold3.data import featurisation",
            "import sitecustomize as generated",
            "from astrolabe.core import global_config",
            "from astrolabe.evaluators.alphafold3 import AlphaFold3Evaluator",
            "from astrolabe.evaluators.wrappers.alphafold3 import ModelRunner",
            "from astrolabe.posthoc import refold",
            "from astrolabe.utils.af3_compat import allow_all_x_protein_chains_during_featurisation",
            "from astrolabe.utils.base import construct_standard_input",
            "",
            "prepared_path = Path(sys.argv[1])",
            "work = Path(sys.argv[2])",
            "prepared = json.loads(prepared_path.read_text(encoding='utf-8'))",
            "base = prepared_path.parent",
            "item = json.loads((base / prepared['inputs']['manifest']).read_text(encoding='utf-8').strip())",
            "config_path = base / prepared['inputs']['config']",
            "cfg = refold._load_cfg_for_refold(config_path)",
            "runtime = refold._load_refold_runtime_overrides(cfg)",
            "context = refold._load_refold_input_context(cfg, item=item)",
            "homologies = refold._resolve_homology_value(",
            "    v=item.get('homologies') or {},",
            "    relpath_base=None,",
            "    path_maps=(),",
            "    require_abs=True,",
            ")",
            "work.mkdir(parents=True, exist_ok=True)",
            "refold._init_refold_global_config(out_dir=work, **runtime, **context)",
            "refold._build_protomers(seqs=item['seqs'], types=item['types'], homologies=homologies)",
            "global_config.set_variable('step', int(item['step']))",
            "inputs = construct_standard_input([str(item['oligo'])], seed=int(cfg.seed))",
            "evaluator = AlphaFold3Evaluator(None, work / 'unused_evaluator_output', cfg.evaluator.alphafold3)",
            "prepared_inputs = evaluator.prepare_input(inputs)",
            "fold_input = prepared_inputs[str(item['oligo'])]",
            "ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)",
            "with allow_all_x_protein_chains_during_featurisation(fold_input):",
            "    examples = featurisation.featurise_input(",
            "        fold_input=fold_input,",
            "        buckets=evaluator.buckets[str(item['oligo'])],",
            "        ccd=ccd,",
            "        verbose=False,",
            "        **evaluator.kwargs[str(item['oligo'])],",
            "    )",
            "assert len(examples) == 1",
            "real_bundle = examples[0]",
            "",
            "def stub_inner(self, featurised_example, rng_key):",
            "    assert featurised_example is real_bundle",
            "    calls = Path(os.environ['ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE_DIR']) / 'stub_inner_calls.jsonl'",
            "    with calls.open('a', encoding='utf-8') as handle:",
            "        handle.write(json.dumps({'called': True}) + '\\n')",
            "    return {",
            "        'identifier': np.asarray([17], dtype=np.int64),",
            "        'positions': np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32),",
            "    }",
            "",
            "generated._original_inference = stub_inner",
            "runner = object.__new__(ModelRunner)",
            "rng_key = jax.random.PRNGKey(17)",
            "result = ModelRunner.run_inference(runner, real_bundle, rng_key)",
            "(work / 'entrypoint_result.json').write_text(",
            "    json.dumps({'identifier': result['identifier'].tolist()}),",
            "    encoding='utf-8',",
            ")",
        ]
    )
    script_path = root / "real_af3_entrypoint.py"
    script_path.write_text(script + "\n", encoding="utf-8")
    environment = os.environ.copy()
    python_path = [str(capture), str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE": "1",
            "ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE_DIR": str(capture),
            "ASTROLABE_POSTHOC_ACCEPTANCE_STAGE": "entrypoint-real-af3-cpu",
            "ASTROLABE_POSTHOC_ACCEPTANCE_MAX_REQUESTS": "1",
            "JAX_PLATFORMS": "cpu",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(prepared_path),
            str(feature_work),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return completed, capture


def test_prepare_freezes_exact_single20aa_manifest_and_no_runtime_output(
    harness,
    tmp_path,
):
    prepared = _prepare(harness, tmp_path)
    root = tmp_path / "acceptance"
    manifest_path, item = _manifest(harness, tmp_path)

    assert prepared["fixture"] == {
        "name": "single-20aa",
        "sequence": harness.SEQUENCE,
        "types": {"A": "p"},
        "oligo": "A",
    }
    assert prepared["scientific_settings"] == {
        "seed": 17,
        "num_seeds": 1,
        "num_samples": 1,
        "num_recycles": 1,
        "run_data_pipeline": False,
        "run_inference": True,
    }
    assert item["seqs"] == {"A": "ACDEFGHIKLMNPQRSTVWY"}
    assert item["types"] == {"A": "p"}
    assert item["needed_evaluators"] == ["af3"]
    assert item["inputs_hash"] == harness._manifest_inputs_hash(item)
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 1

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(root / prepared["inputs"]["config"])
    assert cfg.seed == 17
    assert cfg.evaluator.alphafold3.num_samples == 1
    assert cfg.evaluator.alphafold3.num_recycles == 1
    assert cfg.evaluator.alphafold3.num_seeds == 1
    assert cfg.evaluator.alphafold3.run_inference is True
    assert cfg.evaluator.alphafold3.run_data_pipeline is False
    assert cfg.evaluator.alphafold3.execution == "in_process"
    assert cfg.evaluator.evaluator_terms.Alphafold3.num_samples == 1
    assert cfg.evaluator.evaluator_terms.Alphafold3.num_recycles == 1

    public = prepared["commands"]["public_caller"]
    assert public[1:5] == ["-m", "astrolabe.cli", "posthoc", "refold-run"]
    assert public[public.index("--manifest") + 1] == str(manifest_path)
    assert public[public.index("--evaluators") + 1] == "af3"
    assert public[public.index("--workers") + 1] == "1"
    assert not (root / "caller").exists()
    assert not (root / "baseline").exists()
    assert not (root / "evidence" / "requests").exists()

    source = json.loads(
        (root / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        "scripts/validation/posthoc_gpu_acceptance.py"
        in source["files"]
    )
    assert source["fingerprint"] == prepared["source_fingerprint"]

    assessment = prepared["cache_assessment"]
    assert assessment["identical_manifest_same_out_root"][
        "expected_model_requests"
    ] == 0
    assert assessment["identical_manifest_same_out_root"][
        "requires_committed_receipt"
    ]
    assert assessment["changed_manifest_or_config_same_out_root"] == {
        "detected_by_committed_receipt": True,
        "classification": "REJECTED_WITHOUT_REPLAY",
    }


def test_refold_run_json_redirects_runtime_stdout_to_stderr(
    monkeypatch,
    capsys,
):
    from astrolabe import cli

    summary = {
        "manifest_jsonl": "/tmp/manifest.jsonl",
        "out_root": "/tmp/out",
        "evaluators": ["af3"],
        "processed": 1,
        "skipped": 0,
        "failed": 0,
        "dry_run": False,
        "groups": 1,
        "resource_groups": 1,
        "ran_af3": 1,
        "ran_protenix": 0,
    }

    def fake_run_refold(**_kwargs):
        print("Featurising data with seed 17.")
        print("Featurising data with seed 17 took 0.18 seconds.")
        return summary

    monkeypatch.setattr(refold, "run_refold", fake_run_refold)
    namespace = SimpleNamespace(
        manifest="/tmp/manifest.jsonl",
        out_root="/tmp/out",
        evaluators=["af3"],
        relpath_base=None,
        path_map=[],
        keep_protenix_structures=False,
        config_yaml=None,
        workers=1,
        dry_run=False,
        format="json",
    )

    assert cli._cmd_refold_run(namespace) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == summary
    assert captured.out.lstrip().startswith("{")
    assert "Featurising data" not in captured.out
    assert "Featurising data with seed 17." in captured.err
    assert "Featurising data with seed 17 took 0.18 seconds." in captured.err


def test_committed_caller_stdout_extractor_is_strict(tmp_path):
    continuation = _load_continuation()
    valid = tmp_path / "valid.stdout.json"
    valid.write_text(
        "\n".join(
            [
                "Featurising data with seed 17.",
                "Featurising data with seed 17 took 0.18 seconds.",
                "{",
                '  "ran_af3": 1,',
                '  "processed": 1,',
                '  "failed": 0',
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    payload, evidence = continuation.parse_committed_caller_stdout(valid)
    assert payload == {"ran_af3": 1, "processed": 1, "failed": 0}
    assert evidence["allowed_prefix_line_count"] == 2
    assert evidence["json_start_line"] == 3
    assert evidence["json_trailing_non_whitespace"] is False

    trailing = tmp_path / "trailing.stdout.json"
    trailing.write_text(valid.read_text() + "unexpected\n", encoding="utf-8")
    with pytest.raises(
        continuation.ContinuationError,
        match="non-whitespace residue",
    ):
        continuation.parse_committed_caller_stdout(trailing)

    unknown_prefix = tmp_path / "unknown-prefix.stdout.json"
    unknown_prefix.write_text(
        "\n".join(
            [
                "Unapproved runtime line",
                "Featurising data with seed 17 took 0.18 seconds.",
                '{"ran_af3": 1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        continuation.ContinuationError,
        match="line 1 is not the known AF3 progress line",
    ):
        continuation.parse_committed_caller_stdout(unknown_prefix)


def test_prepare_is_idempotent_but_manifest_drift_is_rejected(
    harness,
    tmp_path,
):
    first = _prepare(harness, tmp_path)
    second = _prepare(harness, tmp_path)
    assert second == first

    manifest_path, item = _manifest(harness, tmp_path)
    item["seqs"]["A"] = "AAAAAAAAAAAAAAAAAAAA"
    item["inputs_hash"] = harness._manifest_inputs_hash(item)
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(harness.FrozenInputChanged, match="manifest changed"):
        harness.verify_frozen_inputs(tmp_path / "acceptance")


def test_array_comparison_checks_dtype_shape_equal_nan(harness, tmp_path):
    left = {
        "number": np.asarray([1.0, np.nan], dtype=np.float32),
        "zero": np.asarray([0], dtype=np.int32),
    }
    right = {
        "number": np.asarray([1.0, np.nan], dtype=np.float32),
        "zero": np.asarray([0], dtype=np.int32),
    }
    shared = harness._shared_validation()
    left_path = tmp_path / "left.npz"
    right_path = tmp_path / "right.npz"
    shared._write_npz(left_path, left)
    shared._write_npz(right_path, right)
    equal = harness.compare_npz(left_path, right_path)
    assert equal["equal"] is True
    assert equal["array_count"] == 2
    assert equal["failures"] == []
    assert equal["max_abs_difference"] is None

    wrong_dtype = dict(right)
    wrong_dtype["number"] = right["number"].astype(np.float64)
    wrong_dtype_path = tmp_path / "wrong_dtype.npz"
    shared._write_npz(wrong_dtype_path, wrong_dtype)
    dtype_result = harness.compare_npz(left_path, wrong_dtype_path)
    assert dtype_result["equal"] is False
    assert dtype_result["failures"][0]["reason"] == "dtype"

    wrong_shape = dict(right)
    wrong_shape["zero"] = np.asarray([0, 1], dtype=np.int32)
    wrong_shape_path = tmp_path / "wrong_shape.npz"
    shared._write_npz(wrong_shape_path, wrong_shape)
    shape_result = harness.compare_npz(left_path, wrong_shape_path)
    assert shape_result["equal"] is False
    assert shape_result["failures"][0]["reason"] == "shape"

    wrong_value = dict(right)
    wrong_value["zero"] = np.asarray([1], dtype=np.int32)
    wrong_value_path = tmp_path / "wrong_value.npz"
    shared._write_npz(wrong_value_path, wrong_value)
    value_result = harness.compare_npz(left_path, wrong_value_path)
    assert value_result["equal"] is False
    assert value_result["failures"][0]["reason"] == "value"

    empty_left = tmp_path / "empty-left.npz"
    empty_right = tmp_path / "empty-right.npz"
    np.savez(empty_left)
    np.savez(empty_right)
    empty_result = harness.compare_npz(empty_left, empty_right)
    assert empty_result["equal"] is False
    assert empty_result["array_count"] == 0
    assert empty_result["failures"] == [
        {"key": None, "reason": "empty-array-archive"}
    ]


def test_runtime_preflight_uses_container_marker_and_wheel_origin(
    harness,
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    site_packages = tmp_path / "site-packages"
    af3 = site_packages / "alphafold3"
    nvrtc = site_packages / "nvidia" / "cuda_nvrtc" / "lib"
    cudnn = site_packages / "nvidia" / "cudnn" / "lib"
    for path in (af3, nvrtc, cudnn):
        path.mkdir(parents=True)
    (nvrtc / "libnvrtc.so.12").write_bytes(b"nvrtc")
    (cudnn / "libcudnn.so.9").write_bytes(b"cudnn")
    locations = {
        "alphafold3": af3,
        "nvidia.cuda_nvrtc": nvrtc.parent,
        "nvidia.cudnn": cudnn.parent,
    }
    monkeypatch.setattr(
        harness.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(
                submodule_search_locations=[str(locations[name])],
            )
            if name in locations
            else None
        ),
    )
    monkeypatch.setattr(
        harness.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        harness.ctypes.util,
        "find_library",
        lambda _name: None,
    )
    for module_name in (
        "astrolabe.evaluators.alphafold3",
        "astrolabe.evaluators.wrappers.alphafold3",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    host_sif = tmp_path / "host-only.sif"
    marker = str(host_sif)
    monkeypatch.setenv("APPTAINER_CONTAINER", marker)
    prepared = {
        "sif": marker,
        "sif_sha256": "a" * 64,
        "sif_provenance": {
            "file": "/evidence/sif_provenance.json",
            "file_sha256": "b" * 64,
            "path": marker,
            "sha256": "a" * 64,
        },
        "sif_python": sys.executable,
    }

    result = harness._runtime_preflight(prepared)

    assert result["status"] == "PASS"
    assert result["container"]["marker_name"] == "APPTAINER_CONTAINER"
    assert result["container"]["reported_host_path_exists_in_container"] is False
    assert result["runtime_source_origin"]["directories"] == [str(af3)]
    assert result["runtime_libraries"]["nvrtc"]["loaded_path"] == str(
        (nvrtc / "libnvrtc.so.12").resolve()
    )
    assert result["runtime_libraries"]["cudnn"]["loaded_path"] == str(
        (cudnn / "libcudnn.so.9").resolve()
    )
    assert result["model_execution_modules_loaded"] == []
    assert result["model_weights_loaded"] is False


def test_generated_sitecustomize_complete_entrypoint_success(
    harness,
    tmp_path,
):
    completed, capture = _run_generated_sitecustomize(
        harness,
        tmp_path,
        stub_mode="success",
    )

    assert completed.returncode == 0, completed.stderr
    evidence = harness._request_evidence(capture)
    assert evidence["rpc_delivered"] == 1
    assert evidence["request_reservations"] == 1
    assert evidence["inner_run_inference_invocations"] == 1
    assert evidence["inner_run_inference_completed"] == 1
    assert evidence["entrypoint_completed"] == 1
    assert len(evidence["completed_entrypoint_events"]) == 1
    assert evidence["observed_model_requests"] == 1
    assert evidence["failed_or_denied_model_requests"] == 0
    events = [
        json.loads(line)
        for line in (capture / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    names = [event["event"] for event in events]
    assert names.index("model_rpc_delivered") < names.index(
        "model_request_reserved"
    ) < names.index("model_inner_invocation_started") < names.index(
        "model_inner_invocation_completed"
    ) < names.index("model_entrypoint_completed")
    completion = next(
        event
        for event in events
        if event["event"] == "model_entrypoint_completed"
    )
    assert Path(completion["rng_key_path"]).is_file()
    assert Path(completion["result_path"]).is_file()
    assert Path(completion["input_snapshot_path"]).is_file()
    snapshot = json.loads(
        Path(completion["input_snapshot_path"]).read_text(encoding="utf-8")
    )
    assert snapshot["scope"]["unsupported_objects"] == "fail closed; never dropped"
    assert snapshot["scope"]["raw_buffers_in_json"] is False
    assert (
        snapshot["tree"]["items"]["token_features"]["items"]["token_index"][
            "dtype"
        ]
        == "<i4"
    )
    assert len(
        (capture / "stub_inner_calls.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 1
    assert json.loads((tmp_path / "entrypoint_result.json").read_text()) == {
        "ran": True,
        "identifier": "af3-stub",
    }


def test_generated_sitecustomize_captures_real_jax_rng(
    harness,
    tmp_path,
):
    assert importlib.util.find_spec("jax") is not None
    completed, capture = _run_generated_sitecustomize(
        harness,
        tmp_path,
        stub_mode="success",
        jax_rng=True,
    )

    assert completed.returncode == 0, completed.stderr
    evidence = harness._request_evidence(capture)
    completion = evidence["completed_entrypoint_events"][0]
    with np.load(completion["rng_key_path"], allow_pickle=False) as archive:
        assert archive.files == ["rng_key"]
        assert archive["rng_key"].dtype == np.dtype(np.uint32)
        assert archive["rng_key"].shape == (2,)
        np.testing.assert_array_equal(
            archive["rng_key"],
            np.asarray([0, 17], dtype=np.uint32),
        )


def test_generated_sitecustomize_inner_exception_is_physical_failure(
    harness,
    tmp_path,
):
    completed, capture = _run_generated_sitecustomize(
        harness,
        tmp_path,
        stub_mode="raise",
    )

    assert completed.returncode != 0
    assert "stub inner failure" in completed.stderr
    evidence = harness._request_evidence(capture)
    assert evidence["rpc_delivered"] == 1
    assert evidence["request_reservations"] == 1
    assert evidence["inner_run_inference_invocations"] == 1
    assert evidence["inner_run_inference_completed"] == 0
    assert evidence["inner_run_inference_failed"] == 1
    assert evidence["entrypoint_completed"] == 0
    assert evidence["observed_model_requests"] == 1
    assert evidence["failed_or_denied_model_requests"] == 1
    assert len(
        (capture / "stub_inner_calls.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 1


def test_generated_sitecustomize_digest_failure_has_no_inner_invocation(
    harness,
    tmp_path,
):
    completed, capture = _run_generated_sitecustomize(
        harness,
        tmp_path,
        stub_mode="success",
        invalid_input=True,
    )

    assert completed.returncode != 0
    assert "unsupported native object" in completed.stderr
    evidence = harness._request_evidence(capture)
    assert evidence["rpc_delivered"] == 1
    assert evidence["request_reservations"] == 1
    assert evidence["inner_run_inference_invocations"] == 0
    assert evidence["inner_run_inference_completed"] == 0
    assert evidence["entrypoint_completed"] == 0
    assert evidence["observed_model_requests"] == 0
    assert evidence["rpc_delivered_without_inner"] == 1
    assert evidence["failed_or_denied_model_requests"] == 1
    assert not (capture / "stub_inner_calls.jsonl").exists()


def test_generated_sitecustomize_captures_real_af3_featurised_bundle(
    harness,
    tmp_path,
):
    completed, capture = _run_generated_sitecustomize_with_real_af3(
        harness,
        tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    evidence = harness._request_evidence(capture)
    assert evidence["rpc_delivered"] == 1
    assert evidence["request_reservations"] == 1
    assert evidence["inner_run_inference_invocations"] == 1
    assert evidence["inner_run_inference_completed"] == 1
    assert evidence["entrypoint_completed"] == 1
    assert evidence["observed_model_requests"] == 1
    completion = next(
        event
        for event in harness._read_jsonl(capture / "events.jsonl")
        if event["event"] == "model_entrypoint_completed"
    )
    assert completion["input_digest"]["object_array_count"] >= 6
    assert completion["input_digest"]["native_object_count"] >= 4
    assert completion["input_digest"]["numeric_array_count"] >= 50
    snapshot_path = Path(completion["input_snapshot_path"])
    assert snapshot_path.is_file()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    top = snapshot["tree"]["items"]

    cleaned_container = top["cleaned_struc"]
    assert cleaned_container["kind"] == "object_array"
    assert cleaned_container["dtype"] == "|O"
    assert cleaned_container["shape"] == []
    cleaned = cleaned_container["items"][0]
    assert cleaned["kind"] == "af3_structure"
    assert cleaned["type"] == "alphafold3.structure.structure.Structure"
    atoms = cleaned["fields"]["tables"]["atoms"]
    assert atoms["kind"] == "dataclass"
    assert atoms["fields"]["x"]["kind"] == "numeric_array"
    assert atoms["fields"]["x"]["dtype"] == "<f4"
    assert atoms["fields"]["x"]["shape"] == [20]
    assert atoms["fields"]["x"]["sha256"]
    assert atoms["fields"]["name"]["kind"] == "object_array"

    flat_layout = top["flat_output_layout"]["items"][0]
    assert flat_layout["kind"] == "dataclass"
    assert (
        flat_layout["type"]
        == "alphafold3.model.atom_layout.atom_layout.AtomLayout"
    )
    assert flat_layout["fields"]["atom_name"]["kind"] == "object_array"

    empty_structure = top["empty_output_struc"]["items"][0]
    assert empty_structure["kind"] == "af3_structure"
    assert (
        empty_structure["type"]
        == "alphafold3.structure.structure.Structure"
    )
    assert (
        top["polymer_ligand_bonds"]["items"][0]["kind"]
        == "scalar"
    )
    assert (
        top["ligand_ligand_bonds"]["items"][0]["kind"]
        == "scalar"
    )
    calls = (
        snapshot_path.parents[0] / "stub_inner_calls.jsonl"
    )
    assert calls.is_file()
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1


def test_generated_sitecustomize_raw_capture_failure_preserves_inner_count(
    harness,
    tmp_path,
):
    completed, capture = _run_generated_sitecustomize(
        harness,
        tmp_path,
        stub_mode="raw_object",
    )

    assert completed.returncode != 0
    assert (
        "object-dtype array is unsupported at result__object_result"
        in completed.stderr
    )
    evidence = harness._request_evidence(capture)
    assert evidence["rpc_delivered"] == 1
    assert evidence["request_reservations"] == 1
    assert evidence["inner_run_inference_invocations"] == 1
    assert evidence["inner_run_inference_completed"] == 1
    assert evidence["inner_run_inference_failed"] == 0
    assert evidence["entrypoint_completed"] == 0
    assert evidence["observed_model_requests"] == 1
    assert evidence["completed_model_requests"] == 1
    assert evidence["failed_or_denied_model_requests"] == 1
    assert len(
        (capture / "stub_inner_calls.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 1


def test_marker_only_target_is_rejected_before_model_stack(
    harness,
    tmp_path,
    monkeypatch,
):
    prepared = _prepare(harness, tmp_path)
    root = tmp_path / "acceptance"
    manifest_path, item = _manifest(harness, tmp_path)
    marker = _write_harmless_cache_marker(harness, tmp_path, item)
    marker_before = marker.read_bytes()
    _stub_receipt_assets(monkeypatch)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unproven refold output must not construct an evaluator")

    monkeypatch.setattr(refold, "construct_evaluators", forbidden)
    monkeypatch.setattr(
        refold,
        "_merge_backend_metrics_into_rows",
        lambda **_kwargs: None,
    )
    summary = refold.run_refold(
        manifest_jsonl=manifest_path,
        out_root=root / "caller" / "runs",
        evaluators=["af3"],
        config_yaml=root / prepared["inputs"]["config"],
        workers=1,
    )

    assert summary["processed"] == 0
    assert summary["skipped"] == 0
    assert summary["failed"] == 1
    assert summary["ran_af3"] == 0
    assert summary["evaluator_cache_misses"] == 0
    assert marker.read_bytes() == marker_before
    failure = json.loads(
        Path(summary["failure_report"]).read_text(encoding="utf-8").strip()
    )
    assert "no committed receipt" in failure["error"]


def test_core_rejects_changed_manifest_without_receipt_and_harness_detects_drift(
    harness,
    tmp_path,
    monkeypatch,
):
    prepared = _prepare(harness, tmp_path)
    root = tmp_path / "acceptance"
    manifest_path, item = _manifest(harness, tmp_path)
    _write_harmless_cache_marker(harness, tmp_path, item)
    _stub_receipt_assets(monkeypatch)

    monkeypatch.setattr(
        refold,
        "_merge_backend_metrics_into_rows",
        lambda **_kwargs: None,
    )
    item["seqs"]["A"] = "AAAAAAAAAAAAAAAAAAAA"
    item["inputs_hash"] = harness._manifest_inputs_hash(item)
    manifest_path.write_text(json.dumps(item, sort_keys=True) + "\n", encoding="utf-8")

    summary = refold.run_refold(
        manifest_jsonl=manifest_path,
        out_root=root / "caller" / "runs",
        evaluators=["af3"],
        config_yaml=root / prepared["inputs"]["config"],
        dry_run=True,
        workers=1,
    )
    assert summary["failed"] == 1
    assert summary["skipped"] == 0
    assert summary["ran_af3"] == 0

    with pytest.raises(harness.FrozenInputChanged, match="manifest changed"):
        harness.verify_frozen_inputs(root)


def test_cache_receipt_requires_zero_actual_model_callbacks(
    harness,
    tmp_path,
):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "evaluator_started",
                "stage": "cache_resume",
                "pid": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = harness._request_evidence(capture)
    assert evidence["observed_model_requests"] == 0
    assert evidence["completed_model_requests"] == 0
    assert evidence["evaluator_invocations"] == 1

