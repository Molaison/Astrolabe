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
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: expected a runtime configuration mapping")
    if settings.get("version") != 1:
        raise ValueError(f"Unsupported Astrolabe runtime profile: {path}")
    profiles = settings.get("profiles", [])
    if not isinstance(profiles, list):
        raise ValueError(f"{path}: profiles must be a list")
    words = command_words(argv)
    for index, profile in enumerate(profiles):
        field = f"{path}: profiles[{index}]"
        _validate_runtime_profile(profile, field)
        prefix = profile["command"]
        if prefix and words[:len(prefix)] == prefix:
            return profile
    return None


def _validate_runtime_profile(profile, field="runtime profile"):
    if not isinstance(profile, dict):
        raise ValueError(f"{field}: expected a mapping")
    for key in ("command", "binds", "forward_env"):
        value = profile.get(key, [] if key != "command" else None)
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"{field}.{key}: expected a list of nonempty strings")
    for key in ("engine", "image"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ValueError(f"{field}.{key}: expected a nonempty path")
    for key in ("python", "repo_mount"):
        if key in profile and (not isinstance(profile[key], str) or not profile[key].strip()):
            raise ValueError(f"{field}.{key}: expected a nonempty string")
    values = profile.get("env", {})
    if not isinstance(values, dict) or not all(
        isinstance(key, str) and key and isinstance(value, (str, int, float, bool))
        for key, value in values.items()
    ):
        raise ValueError(f"{field}.env: expected string keys and scalar values")


def _existing_directory(value, cwd, *, resolve=True):
    path = Path(os.path.expanduser(value))
    path = Path(os.path.abspath(path if path.is_absolute() else cwd / path))
    if resolve:
        path = path.resolve()
    if not path.is_dir():
        path = path.parent
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _input_mounts(argv, cwd):
    cwd = Path(cwd)
    mounts = {str(cwd.resolve())}
    # The run CLI resolves its standalone/config-dir against the caller, then
    # changes directory before reading overlays and resolving output overrides.
    run_cwd = cwd
    words = command_words(argv)
    if words and words[0] == "run":
        standalone = None
        chdir = True
        for i, argument in enumerate(argv):
            key, equal, value = argument.partition("=")
            if key == "--standalone":
                standalone = value if equal else (argv[i + 1] if i + 1 < len(argv) else None)
            elif argument in ("--chdir", "--no-chdir"):
                chdir = argument == "--chdir"
        if standalone and chdir:
            path = cwd / Path(standalone).expanduser()
            if path.is_file():
                run_cwd = path.resolve().parent
    path_options = {
        "--input", "--structure", "--target-homology", "--out-dir", "--out",
        "--service-socket", "--config-dir", "--standalone", "--overlay",
        "-o", "--overrides-file", "--mutable-positions", "--spheres", "--msa",
        "--ligand-conformer", "--manifest", "--config-yaml",
    }
    for i, argument in enumerate(argv):
        key, equal, value = argument.partition("=")
        if key not in path_options and key.lstrip("+") not in {"out_dir", "raw_model_results_dir"}:
            continue
        if not equal:
            if i + 1 >= len(argv):
                continue
            value = argv[i + 1]
        base = cwd if key in ("--standalone", "--config-dir") else run_cwd
        if key == "--target-homology" and not (base / Path(value).expanduser()).exists():
            # The native homology parser also accepts file.json:chain.
            value = value.rsplit(":", 1)[0]
        if value and not value.startswith("-"):
            for resolve in (True, False):
                directory = _existing_directory(value, base, resolve=resolve)
                # A missing path immediately under / must not bind the whole host.
                if directory != Path(directory.anchor):
                    mounts.add(str(directory))
    return sorted(mounts)


def runtime_command(root, argv, profile, environment=None, cwd=None):
    """Build one direct engine exec with unchanged native CLI arguments."""
    _validate_runtime_profile(profile)
    env = dict(os.environ if environment is None else environment)
    cwd = Path.cwd() if cwd is None else Path(cwd)
    engine = str(Path(profile["engine"]).expanduser())
    image = str(Path(profile["image"]).expanduser())
    if not os.access(engine, os.X_OK) or not Path(image).is_file():
        raise ValueError("Configured Astrolabe container engine/image is unavailable")
    container_root = profile.get("repo_mount", "/opt/Astrolabe")
    command = [engine, "exec", "--containall", "--cleanenv", "--no-home", "--no-mount", "cwd", "--pwd", str(cwd)]
    cpu_only = any(a in ("--prepare-only", "--dry-run", "--help", "-h", "--service-socket") or a.startswith("--service-socket=") for a in argv)
    values = dict(profile.get("env", {}))
    if cpu_only:
        values.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    else:
        command.append("--nv")
        values["CUDA_DEVICE_ORDER"] = env.get("CUDA_DEVICE_ORDER", values.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"))
        # An assigned device, UUID, empty restriction or scheduler mask wins.
        if "CUDA_VISIBLE_DEVICES" in env:
            values["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
    for key in (*profile.get("forward_env", []), "GH_DALPHABALL"):
        if key in env:
            values[key] = env[key]
    if cpu_only:
        values.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    values["ASTROLABE_RUNTIME_ACTIVE"] = "1"
    values["PYTHONPATH"] = container_root + (":" + values["PYTHONPATH"] if values.get("PYTHONPATH") else "")
    binds = [f"{root}:{container_root}:ro", *profile.get("binds", [])]
    binds.extend(f"{p}:{p}" for p in _input_mounts(argv, cwd))
    if env.get("GH_DALPHABALL"):
        binary = Path(os.path.expandvars(env["GH_DALPHABALL"])).expanduser()
        if binary.is_file():
            binary = binary.absolute()
            values["GH_DALPHABALL"] = str(binary)
            for parent in (binary.parent, binary.resolve().parent):
                binds.append(f"{parent}:{parent}:ro")
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
    try:
        from astrolabe.cli import main as cli_main
    except ModuleNotFoundError as error:
        if "--debug" in argv:
            raise
        print(f"astrolabe: missing CLI dependency {error.name!r} in {sys.executable}", file=sys.stderr)
        return 1
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
        cwd = Path.cwd()
        env["ASTROLABE_HOST_CWD"] = str(cwd)
        env["ASTROLABE_PATH_MOUNTS"] = json.dumps(_input_mounts(argv, cwd))
        command = [python, "-c", "import sys; sys.path.insert(0, sys.argv.pop(1)); from astrolabe.launcher import main; raise SystemExit(main(sys.argv[1:]))", str(root), *argv]
        os.execvpe(python, command, env)
    except (OSError, ValueError, KeyError) as error:
        print(f"astrolabe: runtime setup: {error}", file=sys.stderr)
        return 2
