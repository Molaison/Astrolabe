"""Guard cold application imports without changing the available loss registry."""
from pathlib import Path
import subprocess
import sys


def test_cold_runner_registers_sequence_and_repeat_losses_without_spatial_geometry():
    root = Path(__file__).resolve().parents[1]
    script = """
import random
import sys
import numpy as np
from astrolabe.app import _load_run_module
_load_run_module()
from astrolabe.losses.base import loss_registry
required = {
    'sequence_residue_fraction': 'astrolabe.losses.sequence',
    'dihedral_pair_contact_count_raw': 'astrolabe.losses.repeat_symmetry',
    'clash_pairs': 'astrolabe.losses.design_screening',
}
for name, module in required.items():
    assert loss_registry.get_function(name).__module__ == module, name
assert 'scipy.spatial' not in sys.modules, 'cold sequence-only runner loaded spatial geometry'
random.seed(17)
np.random.seed(17)
python_state, numpy_state = random.getstate(), np.random.get_state()
torch = sys.modules.get('torch')
torch_state = torch.get_rng_state() if torch is not None else None
from astrolabe.losses.design_screening import _require_ckdtree
_require_ckdtree()
assert 'scipy.spatial' in sys.modules
assert random.getstate() == python_state
current = np.random.get_state()
assert current[0] == numpy_state[0] and np.array_equal(current[1], numpy_state[1])
assert current[2:] == numpy_state[2:]
if torch is not None:
    assert torch.equal(torch.get_rng_state(), torch_state)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root,
                            text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
