"""Mandatory runtime dependency failures precede model/CUDA/output setup."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/formal_config/startup/valid_minimal.yaml"


@pytest.mark.parametrize("entry", ["cli", "api"])
@pytest.mark.parametrize("missing", ["jsonschema", "rpds"])
def test_missing_core_dependency_precedes_runtime(tmp_path, entry, missing):
    # Use a fresh interpreter so cached modules cannot defeat the import fault.
    code = """
import importlib.abc, sys
class Fault(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == sys.argv[1]:
            raise ModuleNotFoundError('injected missing ' + fullname, name=fullname)
        if fullname.split('.')[0] in {'torch', 'jax', 'pyrosetta'}:
            raise AssertionError('scientific import before dependency error: ' + fullname)
sys.meta_path.insert(0, Fault())
if sys.argv[2] == 'cli':
    from astrolabe.cli import main
    raise SystemExit(main(['run', '--standalone', sys.argv[3], '--no-chdir',
                          'env=env', 'out_dir=' + sys.argv[4]]))
from omegaconf import OmegaConf
from astrolabe.app import Astrolabe
from astrolabe.startup_dependencies import StartupDependencyError
cfg = OmegaConf.load(sys.argv[3])
cfg.out_dir = sys.argv[4]
try:
    Astrolabe(cfg).run()
except StartupDependencyError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(2)
raise AssertionError('dependency error was missed')
"""
    output = tmp_path / "output"
    env = dict(os.environ, ASTROLABE_RUNTIME_ACTIVE="1", GH_CUDA_PREINITIALIZE="0",
               CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", code, missing, entry, str(FIXTURE), str(output)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Runtime dependency 'jsonschema'" in result.stderr
    assert "injected missing " + missing in result.stderr
    assert "scientific import before" not in result.stderr
    assert not output.exists()


def test_composition_does_not_require_runtime_packages(monkeypatch):
    from omegaconf import OmegaConf
    from astrolabe import startup_dependencies
    from astrolabe.startup_validation import validate_startup_config

    def forbidden():
        pytest.fail("composition/dry-run must not require full runtime packages")

    monkeypatch.setattr(startup_dependencies, "check_core_runtime_dependencies", forbidden)
    validate_startup_config(OmegaConf.load(FIXTURE), require_inputs=False)
