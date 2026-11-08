"""Host paths survive the checkout CLI -> generated interpreter boundary."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from astrolabe import launcher


ROOT = Path(__file__).resolve().parents[1]


def _generated_runner(tmp_path, backend):
    spec = importlib.util.spec_from_file_location(
        "host_path_runner_builder", ROOT / "scripts/containers/apptainer_build.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    path = tmp_path / f"{backend}_runner.py"
    builder.write_python_runner(
        path, apptainer="/fake/apptainer", runner_backend=backend,
        docker_bin="/fake/docker", docker_image="fake-image", sif_path="/fake.sif",
        container_repo_root="/opt/Astrolabe", binds=[], env_defaults={},
        env_prepends=[], container_python="python", use_nv=False,
    )
    spec = importlib.util.spec_from_file_location(f"generated_{backend.replace('-', '_')}", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return path, runner


def _capture_runner(monkeypatch, runner, path, args):
    calls = []
    monkeypatch.setattr(runner.sys, "argv", [str(path), "-c", "pass", *args])
    monkeypatch.setattr(runner.subprocess, "check_call", lambda cmd, env=None: calls.append((cmd, env)))
    assert runner.main() == 0
    return calls[0]


def _inner_command(command, backend):
    return command if backend == "apptainer" else command[command.index("apptainer"):]


def test_checkout_passes_scoped_host_context(monkeypatch, tmp_path):
    cwd = tmp_path / "outside repo"
    cwd.mkdir()
    config = cwd / "case.yaml"
    config.write_text("steps: 1\n")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("ASTROLABE_RUNTIME_ACTIVE", "1")
    monkeypatch.setenv("ASTROLABE_PY", sys.executable)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *_args, **_kw: SimpleNamespace(returncode=0))
    captured = {}

    class CapturedExec(Exception):
        pass

    def capture(_binary, command, env):
        captured.update(command=command, env=env)
        raise CapturedExec

    monkeypatch.setattr(launcher.os, "execvpe", capture)
    args = ["run", "--standalone", "case.yaml", "--no-chdir"]
    with pytest.raises(CapturedExec):
        launcher.checkout_main(ROOT, args)
    assert captured["command"][-len(args):] == args
    assert captured["env"]["ASTROLABE_HOST_CWD"] == str(cwd)
    assert json.loads(captured["env"]["ASTROLABE_PATH_MOUNTS"]) == [str(cwd)]


@pytest.mark.parametrize("backend", ["apptainer", "docker-apptainer"])
@pytest.mark.parametrize("relative", [False, True])
def test_generated_runner_mounts_host_config_and_cwd(monkeypatch, tmp_path, backend, relative):
    path, runner = _generated_runner(tmp_path, backend)
    repo = tmp_path / "repo"
    repo.mkdir()
    cwd = tmp_path / "outside repo" / "working"
    cwd.mkdir(parents=True)
    configs = tmp_path / "external configs"
    configs.mkdir()
    config = configs / "case.yaml"
    config.write_text("steps: 1\n")
    output_parent = tmp_path / "external results"
    output_parent.mkdir()
    value = os.path.relpath(config, cwd) if relative else str(config)
    args = ["run", "--standalone", value, "--no-chdir",
            f"out_dir={output_parent / 'not-created' / 'run'}"]
    mounts = launcher._input_mounts(args, cwd)
    monkeypatch.setenv("GH_HOST_REPO_ROOT", str(repo))
    monkeypatch.setenv("ASTROLABE_HOST_CWD", str(cwd))
    monkeypatch.setenv("ASTROLABE_PATH_MOUNTS", json.dumps(mounts))
    command, env = _capture_runner(monkeypatch, runner, path, args)
    inner = _inner_command(command, backend)
    assert inner[inner.index("--pwd") + 1] == str(cwd)
    assert f"{repo}:/opt/Astrolabe" in inner
    assert f"{repo}:{repo}:ro" in inner
    assert f"{cwd}:{cwd}" in inner
    assert f"{configs}:{configs}" in inner
    assert f"{output_parent}:{output_parent}" in inner
    assert inner[-len(args):] == args
    assert "ASTROLABE_HOST_CWD" not in env
    assert "ASTROLABE_PATH_MOUNTS" not in env
    if backend == "docker-apptainer":
        assert f"{repo}:{repo}:rw" in command
        assert f"{cwd}:{cwd}:rw" in command
        assert f"{configs}:{configs}:rw" in command
        assert f"{output_parent}:{output_parent}:rw" in command


def test_input_mounts_resolve_missing_outputs_and_symlinks(monkeypatch, tmp_path):
    cwd = tmp_path / "working"
    cwd.mkdir()
    output_parent = tmp_path / "results"
    output_parent.mkdir()
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "case.yaml").write_text("steps: 1\n")
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    args = ["run", "--standalone", "../alias/case.yaml",
            "out_dir=../results/new/nested", "raw_model_results_dir=../results/raw/new"]
    monkeypatch.chdir(tmp_path)
    mounts = launcher._input_mounts(args, cwd)
    assert str(cwd) in mounts
    assert str(alias) in mounts
    assert str(actual) in mounts
    assert str(output_parent) in mounts
    assert "/" not in launcher._input_mounts(["run", "--standalone", "/absent-case.yaml"], cwd)


@pytest.mark.parametrize("backend", ["apptainer", "docker-apptainer"])
def test_generated_runner_without_context_keeps_repo_cwd(monkeypatch, tmp_path, backend):
    path, runner = _generated_runner(tmp_path, backend)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("GH_HOST_REPO_ROOT", str(repo))
    monkeypatch.delenv("ASTROLABE_HOST_CWD", raising=False)
    monkeypatch.delenv("ASTROLABE_PATH_MOUNTS", raising=False)
    command, _env = _capture_runner(monkeypatch, runner, path, ["--version"])
    inner = _inner_command(command, backend)
    assert inner[inner.index("--pwd") + 1] == "/opt/Astrolabe"
    assert f"{repo}:{repo}:ro" not in inner


@pytest.mark.parametrize("overlay_flag", ["--overlay", "-o"])
@pytest.mark.parametrize("chdir", [True, False])
def test_overlay_and_output_mounts_follow_standalone_chdir(tmp_path, overlay_flag, chdir):
    working = tmp_path / "work" / "launch"
    working.mkdir(parents=True)
    cases = tmp_path / "cases"
    cases.mkdir()
    config = cases / "base.yaml"
    config.write_text("steps: 1\n")
    overlays = tmp_path / "overlays"
    overlays.mkdir()
    (overlays / "extra.yaml").write_text("steps: 2\n")
    output = tmp_path / "results"
    output.mkdir()
    base = cases if chdir else working
    args = ["run", "--standalone", str(config), overlay_flag,
            os.path.relpath(overlays / "extra.yaml", base),
            "out_dir=" + os.path.relpath(output / "new", base)]
    if not chdir:
        args.insert(3, "--no-chdir")
    mounts = launcher._input_mounts(args, working)
    assert str(overlays) in mounts
    assert str(output) in mounts
    assert str(working) in mounts
