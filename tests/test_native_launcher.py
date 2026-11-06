"""Exercise installed entry routing, runtime isolation and real exec boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from astrolabe.launcher import command_words, runtime_command, runtime_profile


def profile(tmp_path):
    engine = tmp_path / "engine"
    engine.write_text("#!/bin/sh\nexit 0\n")
    engine.chmod(0o755)
    image = tmp_path / "runtime.sif"
    image.touch()
    return {"command": ["refine", "interface"], "engine": str(engine), "image": str(image),
            "env": {"PYTHONPATH": "/deps", "GH_CUDA_PREINITIALIZE": "0"},
            "forward_env": ["PROTEIN_BEAM_CPU_SET"]}


def test_runtime_is_command_scoped_and_never_recurses(tmp_path):
    config = tmp_path / "config/env/runtime_local.json"
    config.parent.mkdir(parents=True)
    p = profile(tmp_path)
    config.write_text(json.dumps({"version": 1, "profiles": [p]}))
    args = ["--config-dir", "/a space/refine", "--debug", "refine", "interface", "--help"]
    assert command_words(args) == ["refine", "interface", "--help"]
    assert runtime_profile(tmp_path, args, {}) == p
    assert runtime_profile(tmp_path, ["env", "doctor"], {}) is None
    assert runtime_profile(tmp_path, args, {"ASTROLABE_RUNTIME_ACTIVE": "1"}) is None
    with pytest.raises(ValueError, match="does not exist"):
        runtime_profile(tmp_path, args, {"ASTROLABE_RUNTIME_CONFIG": str(tmp_path / "missing")})


@pytest.mark.parametrize("mask", ["1", "GPU-example", "", None])
def test_runtime_preserves_resource_assignment_and_native_argv(tmp_path, mask):
    outside = tmp_path / "input with spaces"
    outside.mkdir()
    input_file = outside / "complex.pdb"
    input_file.touch()
    args = ["refine", "interface", "--input", str(input_file), "--target-homology=" + str(outside / "msa.json") + ":B", "--out", str(outside / "new/deep/results")]
    env = {"PROTEIN_BEAM_CPU_SET": "8-19", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
    if mask is not None:
        env["CUDA_VISIBLE_DEVICES"] = mask
    command, _ = runtime_command(tmp_path, args, profile(tmp_path), env, tmp_path)
    assert "--nv" in command
    assert command[-len(args):] == args
    assert f"{outside}:{outside}" in command
    assert "PYTHONPATH=/opt/Astrolabe:/deps" in command
    assert "PROTEIN_BEAM_CPU_SET=8-19" in command
    assert "CUDA_DEVICE_ORDER=FASTEST_FIRST" in command
    if mask is None:
        assert not any(a.startswith("CUDA_VISIBLE_DEVICES=") for a in command)
    else:
        assert "CUDA_VISIBLE_DEVICES=" + mask in command


@pytest.mark.parametrize("extra", [["--help"], ["--prepare-only"], ["--service-socket", "/a/b"], ["--service-socket=/a/b"]])
def test_preparation_help_and_reused_owner_hide_cuda(tmp_path, extra):
    p = profile(tmp_path)
    p["forward_env"].append("CUDA_VISIBLE_DEVICES")
    cmd, _ = runtime_command(tmp_path, ["refine", "interface", *extra], p, {"CUDA_VISIBLE_DEVICES": "1"}, tmp_path)
    assert "--nv" not in cmd
    assert "CUDA_VISIBLE_DEVICES=" in cmd
    assert "JAX_PLATFORMS=cpu" in cmd


def test_python_module_enters_configured_runtime_before_cli_dependencies(tmp_path):
    """A fresh interpreter must exec the engine, not import the heavy CLI first."""
    p = profile(tmp_path)
    capture = tmp_path / "args.json"
    Path(p["engine"]).write_text("#!" + sys.executable + "\nimport json,sys\nfrom pathlib import Path\nPath(" + repr(str(capture)) + ").write_text(json.dumps(sys.argv[1:]))\n")
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({"version": 1, "profiles": [p]}))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-S", "-m", "astrolabe", "refine", "interface", "--input", "one space.pdb"], cwd=root,
                            env=dict(os.environ, ASTROLABE_RUNTIME_ACTIVE="0", ASTROLABE_RUNTIME_CONFIG=str(config)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert args[-5:] == ["astrolabe", "refine", "interface", "--input", "one space.pdb"]


def test_checkout_keeps_other_command_interpreter_contract(tmp_path):
    p = profile(tmp_path)
    capture = tmp_path / "python_args.json"
    fake_python = tmp_path / "python"
    fake_python.write_text("#!" + sys.executable + "\nimport json,sys,os\nfrom pathlib import Path\nif 'sys.version_info' not in sys.argv[-1]: Path(" + repr(str(capture)) + ").write_text(json.dumps({'args':sys.argv[1:],'repo':os.environ['GH_HOST_REPO_ROOT'],'map':os.environ['GH_HOST_REPO_MAP_ROOT']}))\n")
    fake_python.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([str(root / "scripts/astrolabe"), "--version"], cwd=root,
                            env=dict(os.environ, ASTROLABE_PY=str(fake_python), ASTROLABE_RUNTIME_ACTIVE="1", ASTROLABE_HOST_REPO_ROOT="/custom-source", ASTROLABE_HOST_REPO_MAP_ROOT="/custom-map"), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result_env = json.loads(capture.read_text())
    args = result_env['args']
    assert args[-1] == "--version"
    assert str(root) in args
    assert result_env['repo'] == os.environ.get('GH_HOST_REPO_ROOT', '/custom-source')
    assert result_env['map'] == os.environ.get('GH_HOST_REPO_MAP_ROOT', '/custom-map')


def test_symlink_inputs_outside_cwd_keep_original_and_target_visible(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "complex.pdb").touch()
    aliases = tmp_path / "aliases"
    aliases.mkdir()
    link = aliases / "complex.pdb"
    link.symlink_to(actual / "complex.pdb")
    working = tmp_path / "working"
    working.mkdir()
    args = ["refine", "interface", "--input", str(link)]
    cmd, _ = runtime_command(tmp_path, args, profile(tmp_path), {}, working)
    assert f"{aliases}:{aliases}" in cmd
    assert f"{actual}:{actual}" in cmd
    assert cmd[-len(args):] == args


@pytest.mark.parametrize("assigned", [None, "", "1", "GPU-assigned"])
def test_native_interface_autoselects_only_without_a_resource_assignment(monkeypatch, assigned):
    from argparse import Namespace
    from astrolabe import cli, interface
    selected = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if assigned is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", assigned)
    monkeypatch.setattr(cli, "_maybe_autoselect_cuda_visible_devices", lambda **kwargs: selected.append(kwargs))
    monkeypatch.setattr(interface, "run_interface", lambda **kwargs: 0)
    ns = Namespace(prepare_only=False, service_socket=None, input="in.pdb", target_homology="msa.json", out_dir="out", target_chain=None, binder_chain=None, resume=False, retry_interrupted=False)
    assert cli._cmd_refine_interface(ns) == 0
    assert bool(selected) == (assigned is None)


def test_owner_and_cpu_proxy_share_real_af3_settings_without_loading_a_model(monkeypatch):
    """Catch release integrations that remove the owner's proxy dependency."""
    from omegaconf import OmegaConf
    from astrolabe.evaluators import af3_service
    from astrolabe.evaluators.wrappers import alphafold3 as af3
    cfg = OmegaConf.create(dict(model_dir="/weights", flash_attention_implementation="xla", num_samples=1,
                                num_recycles=3, save_embeddings=False, service_socket="/rpc/model.sock"))
    proxy = af3.RemoteModelRunner(cfg)
    assert proxy._settings == af3.service_model_settings(cfg)
    # Construct the real native owner's handler; importing its real dependency
    # used to fail after the GPU weights had already loaded.
    assert callable(af3_service.inference_handler(object(), cfg))
    monkeypatch.setattr(af3, "_configure_refstructure_env", lambda config: None)
    monkeypatch.setattr(af3, "_configure_msa_env", lambda config: None)
    monkeypatch.setattr(af3.jax, "local_devices", lambda *args, **kwargs: pytest.fail("CPU proxy initialized GPU"))
    assert isinstance(af3.alphafold3_setup(cfg), af3.RemoteModelRunner)
