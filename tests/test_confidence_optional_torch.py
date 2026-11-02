"""The CPU package must register NumPy confidence losses without PyTorch."""
import os
from pathlib import Path
import subprocess
import sys


def test_numpy_confidence_registration_without_torch():
    code = '''
import sys
sys.modules["torch"] = None
import numpy as np
from astrolabe.evaluators.utils import StandardScores
from astrolabe.losses.confidence import plddt, _to_numpy_float_array
from astrolabe.losses.base import LossEngine
scores = StandardScores(names=["A"], plddt={"A": [0.8]}, complex_paths={})
assert plddt(scores) == {"A": 0.8}
assert np.array_equal(_to_numpy_float_array([[1, 2]]), np.array([[1., 2.]]))
engine = LossEngine({"loss_terms": {}})
engine.setup("plddt", None, None, None)
assert sys.modules["torch"] is None
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
