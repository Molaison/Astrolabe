"""Test-only import/effect barriers and targeted startup stage observation.

No global tracing/profiling. Validators run unchanged; only forbidden runtime
boundaries are replaced by bombs. Flush every event so crashes cannot erase it.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time

LOG = os.environ.get("ASTROLABE_DAILY_PROBE")
ARMED = False
STACK = []
HEAVY = {"torch", "jax", "pyrosetta", "tensorflow", "alphafold", "alphafold3", "run"}
STAGES = {
    "astrolabe.startup_validation": {"validate_startup_config": "startup"},
    "astrolabe.task_startup": {"validate_task_config": "task"},
    "astrolabe.scoring_startup": {"validate_scoring_settings": "scoring"},
    "astrolabe.backend_startup": {"validate_backend_settings": "backend"},
    "astrolabe.startup_dependencies": {
        "check_startup_dependencies": "dependencies",
        "check_core_runtime_dependencies": "core_dependencies",
    },
}
BOMBS = {
    "astrolabe.evaluator_startup": ["initialize_plan"],
    "astrolabe.model_daemon": ["OwnedModelDaemon.__init__", "_load_factory"],
    # CLI software catalog reads declarative RNA metadata from these modules.
    "astrolabe.evaluators.base": ["Evaluator.__init__"],
    "astrolabe.evaluators.rna": ["RnaEvaluator.__init__", "_backend_kernel"],
}


def emit(event, **data):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event=event, pid=os.getpid(),
                                         at=time.monotonic(), **data)) + "\n")


def bomb(name):
    def blocked(*args, **kwargs):
        emit("forbidden_factory", target=name)
        raise RuntimeError("DAILY_CONFIG_BOUNDARY: " + name)
    return blocked


def observe(fn, stage):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        STACK.append(stage)
        emit("stage_enter", stage=stage)
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            emit("stage_error", stage=stage, error_type=type(exc).__name__,
                 reason=str(exc))
            raise
        else:
            if isinstance(result, list) and result:
                emit("stage_findings", stage=stage, reasons=result)
            emit("stage_exit", stage=stage)
            return result
        finally:
            STACK.pop()
    return wrapper


class Loader(importlib.abc.Loader):
    def __init__(self, original, name):
        self.original, self.name = original, name

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        for name, stage in STAGES.get(self.name, {}).items():
            setattr(module, name, observe(getattr(module, name), stage))
        for target in BOMBS.get(self.name, []):
            obj = module
            parts = target.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], bomb(self.name + "." + target))


class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        global ARMED
        if fullname == "astrolabe.cli":
            ARMED = True
            emit("cli_entered")
        if fullname.split(".")[0] in HEAVY:
            emit("forbidden_import", module=fullname)
            raise RuntimeError("DAILY_CONFIG_BOUNDARY: heavy import " + fullname)
        if fullname.startswith("astrolabe.evaluators.") and fullname not in BOMBS:
            emit("forbidden_evaluator_import", module=fullname)
            raise RuntimeError("DAILY_CONFIG_BOUNDARY: evaluator import " + fullname)
        if fullname in STAGES or fullname in BOMBS:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader, fullname)
            return spec
        return None


def audit(event, args):
    if ARMED and event == "open":
        filename, _mode, flags = args
        writes = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writes and filename != LOG:
            emit("forbidden_write", path=str(filename))
            raise RuntimeError("DAILY_CONFIG_BOUNDARY: file write " + str(filename))
        return
    if not ARMED or event not in {
        "subprocess.Popen", "os.fork", "os.posix_spawn", "socket.connect",
        "os.mkdir", "os.rename", "os.remove", "os.rmdir",
    }:
        return
    # A DAlphaBall protocol check is a dependency probe, not a model owner.
    # Permit only the explicitly supplied executable during that validator.
    if (event == "subprocess.Popen" and "dependencies" in STACK
            and os.environ.get("GH_DALPHABALL")
            and os.path.realpath(args[0]) == os.path.realpath(os.environ["GH_DALPHABALL"])):
        emit("dependency_process", executable=str(args[0]))
        return
    emit("forbidden_effect", operation=event)
    raise RuntimeError("DAILY_CONFIG_BOUNDARY: " + event)


if LOG:
    emit("probe_installed", python=sys.executable)
    sys.meta_path.insert(0, Boundary())
    sys.addaudithook(audit)
