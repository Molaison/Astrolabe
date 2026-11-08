from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import astrolabe.dev_wrapper as dev_wrapper


def test_default_local_env_path_points_to_config_env_env_local_yaml(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    assert dev_wrapper.default_local_env_path(repo_root) == repo_root / "config" / "env" / "env_local.yaml"


def test_default_local_env_path_for_config_dir_points_to_env_env_local_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "custom_config"
    assert dev_wrapper.default_local_env_path_for_config_dir(config_dir) == config_dir / "env" / "env_local.yaml"


def test_resolve_default_gh_python_prefers_generated_hint(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    env_dir = repo_root / "config" / "env"
    env_dir.mkdir(parents=True)
    hinted_python = tmp_path / "shared" / "envs" / "gh" / "bin" / "python"
    hinted_python.parent.mkdir(parents=True)
    hinted_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hinted_python.chmod(0o755)

    other_python = tmp_path / "other" / "python"
    other_python.parent.mkdir(parents=True)
    other_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    other_python.chmod(0o755)

    (env_dir / ".gh_python_path").write_text(str(hinted_python) + "\n", encoding="utf-8")
    (env_dir / "env_local.yaml").write_text(
        f"# @package _global_\n\nsubprocess_envs:\n  gh:\n    python_bin: {other_python}\n",
        encoding="utf-8",
    )

    assert dev_wrapper.resolve_default_gh_python(repo_root) == str(hinted_python)


def test_resolve_default_gh_python_reads_env_local_overlay(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    env_dir = repo_root / "config" / "env"
    env_dir.mkdir(parents=True)
    env_python = tmp_path / "shared" / "envs" / "gh" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    env_python.chmod(0o755)

    (env_dir / "env_local.yaml").write_text(
        f"# @package _global_\n\nsubprocess_envs:\n  gh:\n    python_bin: {env_python}\n",
        encoding="utf-8",
    )

    assert dev_wrapper.resolve_default_gh_python(repo_root) == str(env_python)


def test_resolve_default_gh_python_skips_stale_hint_and_env_local(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    env_dir = repo_root / "config" / "env"
    env_dir.mkdir(parents=True)
    (env_dir / ".gh_python_path").write_text("/missing/hint/python\n", encoding="utf-8")
    (env_dir / "env_local.yaml").write_text(
        "# @package _global_\n\nsubprocess_envs:\n  gh:\n    python_bin: /missing/env/python\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    assert dev_wrapper.resolve_default_gh_python(repo_root) is None


def test_resolve_default_gh_python_uses_standard_gh_env_before_container_wrapper(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    env_dir = repo_root / "config" / "env"
    env_dir.mkdir(parents=True)
    (env_dir / ".gh_python_path").write_text("/missing/hint/python\n", encoding="utf-8")
    (env_dir / "env_local.yaml").write_text(
        "# @package _global_\n\nsubprocess_envs:\n  gh:\n    python_bin: /missing/env/python\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    gh_python = home / ".gh_envs" / "envs" / "gh" / "bin" / "python"
    gh_python.parent.mkdir(parents=True)
    gh_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    gh_python.chmod(0o755)
    wrapper = home / ".gh_envs" / "tools" / "gh_h100_container_python.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))

    assert dev_wrapper.resolve_default_gh_python(repo_root) == str(gh_python)


def test_resolve_default_gh_python_uses_executable_container_wrapper_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    env_dir = repo_root / "config" / "env"
    env_dir.mkdir(parents=True)
    (env_dir / ".gh_python_path").write_text("/missing/hint/python\n", encoding="utf-8")
    (env_dir / "env_local.yaml").write_text(
        "# @package _global_\n\nsubprocess_envs:\n  gh:\n    python_bin: /missing/env/python\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    wrapper = home / ".gh_envs" / "tools" / "gh_h100_container_python.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))

    assert dev_wrapper.resolve_default_gh_python(repo_root) == str(wrapper)


def test_write_gh_python_hint_for_config_dir_writes_under_config_env(tmp_path: Path) -> None:
    config_dir = tmp_path / "custom_config"
    python_bin = tmp_path / "python"

    hint_path = dev_wrapper.write_gh_python_hint_for_config_dir(config_dir, str(python_bin))

    assert hint_path == config_dir / "env" / ".gh_python_path"
    assert hint_path.read_text(encoding="utf-8") == f"{python_bin}\n"


def test_base_defaults_to_env_local() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base_text = (repo_root / "config" / "base.yaml").read_text(encoding="utf-8")
    assert "- env: env_local" in base_text


def test_gh_dev_reports_unavailable_configured_interpreter(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    unavailable = tmp_path / "missing-python"
    env = os.environ.copy()
    env.pop("GH_PY", None)
    env.update({
        "HOME": str(tmp_path),
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "ASTROLABE_PY": str(unavailable),
        "ASTROLABE_RUNTIME_ACTIVE": "1",
    })

    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "gh_dev"), "--help"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "interpreter must be an executable absolute path" in result.stderr
    assert str(unavailable) in result.stderr
    assert "Traceback" not in result.stderr


def test_gh_dev_auto_resolved_runner_binds_current_checkout(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    repo_root = tmp_path / "checkout"
    (repo_root / "scripts").mkdir(parents=True)
    for name in ("astrolabe", "gh_dev"):
        shutil.copy2(source_root / "scripts" / name, repo_root / "scripts" / name)
    (repo_root / "astrolabe").symlink_to(source_root / "astrolabe", target_is_directory=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    runner = tmp_path / "container_runner"
    runner.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in *sys.version_info*) exit 0;; esac\n"
        "printf 'host=%s\\n' \"${GH_HOST_REPO_ROOT-}\"\n"
        "printf 'map=%s\\n' \"${GH_HOST_REPO_MAP_ROOT-}\"\n"
        "printf 'args=%s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    (bin_dir / "python3").symlink_to(sys.executable)
    dev_wrapper.write_gh_python_hint_for_config_dir(repo_root / "config", str(runner))
    env = os.environ.copy()
    for key in (
        "GH_PY", "ASTROLABE_PY", "GH_HOST_REPO_ROOT", "GH_HOST_REPO_MAP_ROOT",
        "ASTROLABE_HOST_REPO_ROOT", "ASTROLABE_HOST_REPO_MAP_ROOT",
        "ASTROLABE_RUNTIME_CONFIG",
    ):
        env.pop(key, None)
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "ASTROLABE_RUNTIME_ACTIVE": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "gh_dev"), "run", "--task", "test"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert f"host={repo_root}" in result.stdout
    assert f"map={repo_root}" in result.stdout
    assert "run --task test" in result.stdout


def test_gh_dev_explicit_runner_preserves_frozen_source_opt_in(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = tmp_path / "container_runner"
    runner.write_text(
        "#!/bin/sh\n"
        "printf 'host=%s\\n' \"${GH_HOST_REPO_ROOT-}\"\n"
        "printf 'map=%s\\n' \"${GH_HOST_REPO_MAP_ROOT-}\"\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    env = os.environ.copy()
    env.update({"GH_PY": str(runner)})
    env.pop("GH_HOST_REPO_ROOT", None)
    env.pop("GH_HOST_REPO_MAP_ROOT", None)

    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "gh_dev"), "--help"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "host=\n" in result.stdout
    assert f"map={repo_root}" in result.stdout
