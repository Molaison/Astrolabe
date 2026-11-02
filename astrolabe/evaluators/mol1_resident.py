"""Resident-atomic Mol-1 using Molence's exact Python model API."""
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from astrolabe.evaluators.worker_identity import identity, probe_device
from astrolabe.evaluators.worker_session import WorkerArtifacts


@dataclass
class Mol1Runner:
    repo_path: str
    mol_root_dir: str
    load_checkpoint_path: str
    python_bin: str = sys.executable
    model_name: str = 'mol-1-base'
    num_cycles: int = 20
    num_steps: int = 200
    num_samples: int = 5
    dtype: str = 'bf16'
    use_msa: bool = False
    triangle_multiplicative: str = 'auto'
    triangle_attention: str = 'auto'
    enable_fusion: bool = True
    enable_tf32: bool = False
    need_atom_confidence: bool = False
    timeout_seconds: float | None = None

    def run(self, input_json, output_dir):
        if not hasattr(self, '_resident'):
            self._resident = Mol1Resident(self)
        return self._resident.run(input_json, output_dir)

    def close(self):
        if hasattr(self, '_resident'):
            self._resident.close(); del self._resident


class Mol1Resident:
    def __init__(self, runner):
        if os.environ.get('ASTROLABE_EXECUTION_ENDPOINT') or os.environ.get('ASTROLABE_EXECUTION_TOKEN'):
            raise ValueError('Mol-1 daemon currently requires trajectory-exclusive direct execution')
        if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(os.environ.get('LOCAL_RANK', '0')) != 0:
            raise ValueError('Mol-1 daemon supports one local device only')
        from astrolabe.runtime.client import DaemonClient
        python = shutil.which(runner.python_bin)
        if not python: raise FileNotFoundError('Mol-1 python executable not found')
        source = Path(runner.repo_path).resolve(strict=True)
        if not (source / 'mol' / '__init__.py').is_file() or not (source / 'mol' / 'runner' / 'inference.py').is_file():
            raise ValueError('invalid Molence Mol-1 source checkout')
        root_dir = Path(runner.mol_root_dir).resolve(strict=True)
        checkpoint = str(Path(runner.load_checkpoint_path).resolve(strict=True))
        env = {'PYTHONPATH': os.pathsep.join([str(source)] + [str(Path(p).resolve()) for p in sys.path if p]),
               'MOL_ROOT_DIR': str(root_dir)}
        device = probe_device(python, 'mol1', env=env)
        cmd = [python, '-m', 'astrolabe.evaluators.mol1_assets', runner.model_name, checkpoint, '0']
        try:
            from astrolabe.runtime.admission import run_probe
        except ImportError:
            probe_env = os.environ.copy(); probe_env.update(env)
            assets = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True, env=probe_env).stdout)['assets']
        else:
            assets = run_probe(cmd, env=env)['assets']
        options = asdict(runner)
        for key in ('python_bin', 'repo_path', 'timeout_seconds'):
            options.pop(key, None)
        self._artifacts = WorkerArtifacts()
        root = self._artifacts.root
        base = Path(__file__).resolve().parent
        code = [str(base / p) for p in ('mol1_assets.py', 'worker_identity.py', 'request_path.py', 'device_probe.py', 'wrappers/mol1_daemon.py')]
        code.append(str(source / 'mol'))
        semantics = {'options': options, 'trajectory': uuid.uuid4().hex}
        expected = identity(python, code, assets, device, semantics)
        spec = {'source': str(source), 'options': options, 'semantics': semantics,
                'code_paths': code, 'weight_paths': assets, 'device': device, 'bootstrap_dir': str(root / 'bootstrap')}
        path = self._artifacts.specification(expected, spec)
        self._client = DaemonClient([python, '-m', 'astrolabe.evaluators.wrappers.mol1_daemon', str(path)], expected, env=env, startup_timeout=600)
        self.timeout = runner.timeout_seconds

    def run(self, input_json, output_dir):
        data = json.loads(Path(input_json).read_text())
        if not isinstance(data, list) or len(data) != 1 or not data[0].get('modelSeeds'):
            raise ValueError('Mol-1 requires one job with explicit modelSeeds')
        name = data[0].get('name')
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid Mol-1 job name')
        with tempfile.TemporaryDirectory(prefix='request-', dir=self._artifacts.root) as tmp:
            request = Path(tmp)
            (request / 'input.json').write_text(json.dumps(data))
            reply = self._client.call({'request_dir': str(request)}, timeout=self.timeout)
            if reply != {'request': request.name}: raise ValueError('Mol-1 request mismatch')
            source = request / 'output' / name
            if not source.is_dir(): raise RuntimeError('Mol-1 output is missing')
            target = Path(output_dir).resolve() / ('request-' + uuid.uuid4().hex)
            target.mkdir(parents=True)
            shutil.copytree(source, target / name)
            return target

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._client.close(); self._client = None
        if getattr(self, '_artifacts', None) is not None:
            self._artifacts.close(); self._artifacts = None

    def __del__(self):
        try: self.close()
        except Exception: pass
