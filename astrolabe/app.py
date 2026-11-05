from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING

from astrolabe import run_script_path

if TYPE_CHECKING:
    from astrolabe.design.composition import DesignRun
    from astrolabe.design.types import RunResult


@lru_cache(maxsize=1)
def _load_run_module():
    run_path = run_script_path()
    if not run_path.is_file():
        raise FileNotFoundError(f"Runner module not found: {run_path}")

    spec = importlib.util.spec_from_file_location("astrolabe_run_module", run_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load runner module spec from {run_path}")

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    return module


class Astrolabe:
    """Closed-loop sequence–structure design from a composed Hydra config.

    The application owns the existing repeat lifecycle. Evaluators, mutators,
    objectives and sequence synchronization remain configured by the input;
    constructing an application does not load a model or start a trajectory.
    """

    def __init__(self, config):
        self.config = config

    def run(self) -> RunResult | None:
        """Run a direct design composition or the configured repeat lifecycle."""
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        from astrolabe.design.composition import ControlRun, DesignRun

        if isinstance(self.config, (DesignRun, ControlRun)):
            return self.config.run()

        run = _load_run_module()

        def report_graceful_stop(completed_repeats: int, total_repeats: int) -> None:
            print(
                f"Graceful stop requested; exiting after repeat "
                f"{completed_repeats}/{total_repeats}.",
                file=sys.stderr,
                flush=True,
            )

        run.run_repeats(self.config, on_stop_requested=report_graceful_stop)
        return None


def run_from_cfg(cfg) -> None:
    """CLI entry into the same public Astrolabe application lifecycle."""
    Astrolabe(cfg).run()
