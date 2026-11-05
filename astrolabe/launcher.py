"""Installed CLI bootstrap; machine runtime configuration is separate from cases.

Only the standard library is imported before selecting the installed runtime.
Native Python installations need no profile. Container installations can put a
command-scoped profile in config/env/runtime_local.json (not in a case YAML).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def command_words(argv):
    """Skip supported global options without searching arbitrary option values."""
    i = 0
    while i < len(argv):
        word = argv[i]
        if word in ("--debug", "--autoselect-gpu") or word.startswith("--config-dir="):
            i += 1
        elif word == "--config-dir" and i + 1 < len(argv):
            i += 2
        else:
            return argv[i:]
    return []


def runtime_profile(root, argv, env):
    if env.get("ASTROLABE_RUNTIME_ACTIVE") == "1":
        return None
    path = Path(env.get("ASTROLABE_RUNTIME_CONFIG", str(root / "config/env/runtime_local.json"))).expanduser()
    if not path.exists():
        if "ASTROLABE_RUNTIME_CONFIG" in env:
            raise ValueError(f"Astrolabe runtime profile does not exist: {path}")
        return None
    settings = json.loads(path.read_text())
    if settings.get("version") != 1:
        raise ValueError(f"Unsupported Astrolabe runtime profile: {path}")
    words = command_words(argv)
    for profile in settings.get("profiles", []):
        prefix = profile["command"]
        if prefix and words[:len(prefix)] == prefix:
            return profile
    return None


def _existing_directory(value, *, resolve=True):
    path = Path(os.path.abspath(os.path.expanduser(value)))
    if resolve:
        path = path.resolve()
    if not path.is_dir():
        path = path.parent
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _input_mounts(argv, cwd):
    mounts = {str(cwd.resolve())}
    path_options = {"--input", "--structure", "--target-homology", "--out-dir", "--out", "--service-socket", "--config-dir"}
    for i, argument in enumerate(argv):
        key, equal, value = argument.partition("=")
        if key not in path_options:
            continue
        if not equal:
            if i + 1 >= len(argv):
                continue
            value = argv[i + 1]
        if key == "--target-homology" and not Path(value).expanduser().exists():
            # The native homology parser also accepts file.json:chain.
            value = value.rsplit(":", 1)[0]
        if value and not value.startswith("-"):
            mounts.add(str(_existing_directory(value)))
            # Preserve the path the user typed as well as a symlink target.
            mounts.add(str(_existing_directory(value, resolve=False)))
    return sorted(mounts)


def runtime_command(root, argv, profile, environment=None, cwd=None):
    """Build one direct engine exec with unchanged native CLI arguments."""
    env = dict(os.environ if environment is None else environment)
    cwd = Path.cwd() if cwd is None else Path(cwd)
    engine = str(Path(profile["engine"]).expanduser())
    image = str(Path(profile["image"]).expanduser())
    if not os.access(engine, os.X_OK) or not Path(image).is_file():
        raise ValueError("Configured Astrolabe container engine/image is unavailable")
    container_root = profile.get("repo_mount", "/opt/Astrolabe")
    command = [engine, "exec", "--containall", "--cleanenv", "--no-home", "--no-mount", "cwd", "--pwd", str(cwd)]
    cpu_only = any(a in ("--prepare-only", "--help", "-h", "--service-socket") or a.startswith("--service-socket=") for a in argv)
    values = dict(profile.get("env", {}))
    if cpu_only:
        values.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    else:
        command.append("--nv")
        values["CUDA_DEVICE_ORDER"] = env.get("CUDA_DEVICE_ORDER", values.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"))
        # An assigned device, UUID, empty restriction or scheduler mask wins.
        if "CUDA_VISIBLE_DEVICES" in env:
            values["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
    for key in profile.get("forward_env", []):
        if key in env:
            values[key] = env[key]
    if cpu_only:
        values.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    values["ASTROLABE_RUNTIME_ACTIVE"] = "1"
    values["PYTHONPATH"] = container_root + (":" + values["PYTHONPATH"] if values.get("PYTHONPATH") else "")
    binds = [f"{root}:{container_root}:ro", *profile.get("binds", [])]
    binds.extend(f"{p}:{p}" for p in _input_mounts(argv, cwd))
    # Preserve full host paths in native outputs and resume fingerprints.
    binds.append(f"{root}:{root}:ro")
    for bind in dict.fromkeys(binds):
        command.extend(["--bind", bind])
    container_home = values.pop("HOME", None)
    for key, value in values.items():
        command.extend(["--env", f"{key}={value}"])
    command.append(image)
    if container_home is not None:
        # Apptainer intentionally rejects HOME through --env.
        command.extend(["/usr/bin/env", f"HOME={container_home}"])
    command.extend([profile.get("python", "python"), "-m", "astrolabe", *argv])
    return command, env


def _dispatch_runtime(root, argv):
    profile = runtime_profile(root, argv, os.environ)
    if profile is not None:
        command, env = runtime_command(root, argv, profile)
        os.execvpe(command[0], command, env)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[1]
    try:
        _dispatch_runtime(root, argv)
    except (OSError, ValueError, KeyError) as error:
        print(f"astrolabe: runtime setup: {error}", file=sys.stderr)
        return 2
    from astrolabe.cli import main as cli_main
    return cli_main(argv)


def checkout_main(root, argv):
    """Keep the existing checkout interpreter contract for other commands."""
    root = Path(root).resolve()
    try:
        _dispatch_runtime(root, argv)
        from astrolabe.dev_wrapper import resolve_default_astrolabe_python
        python = os.getenv("ASTROLABE_PY") or os.getenv("GH_PY") or resolve_default_astrolabe_python(root) or sys.executable
        if not Path(python).is_absolute() or not os.access(python, os.X_OK):
            raise ValueError(f"Astrolabe interpreter must be an executable absolute path: {python}")
        result = subprocess.run([python, "-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"])
        if result.returncode:
            raise ValueError(f"Astrolabe requires Python 3.11 or newer: {python}")
        env = os.environ.copy()
        for key in ("ASTROLABE_HOST_REPO_ROOT", "ASTROLABE_HOST_REPO_MAP_ROOT"):
            env.setdefault(key, str(root))
        env.setdefault("GH_HOST_REPO_ROOT", env["ASTROLABE_HOST_REPO_ROOT"])
        env.setdefault("GH_HOST_REPO_MAP_ROOT", env["ASTROLABE_HOST_REPO_MAP_ROOT"])
        command = [python, "-c", "import sys; sys.path.insert(0, sys.argv.pop(1)); from astrolabe.launcher import main; raise SystemExit(main(sys.argv[1:]))", str(root), *argv]
        os.execvpe(python, command, env)
    except (OSError, ValueError, KeyError) as error:
        print(f"astrolabe: runtime setup: {error}", file=sys.stderr)
        return 2
