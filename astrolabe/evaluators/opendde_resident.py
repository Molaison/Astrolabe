"""Cross-interpreter, trajectory-exclusive OpenDDE resident-atomic runner."""
from dataclasses import asdict
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


class OpenDDEResident:
    def __init__(self, runner):
        if os.environ.get('ASTROLABE_EXECUTION_ENDPOINT') or os.environ.get('ASTROLABE_EXECUTION_TOKEN'):
            raise ValueError('OpenDDE daemon currently requires trajectory-exclusive direct execution')
        if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(os.environ.get('LOCAL_RANK', '0')) != 0:
            raise ValueError('OpenDDE daemon supports one local device only')
        if runner.foldcp_mode != 'single' or runner.foldcp_size_dp != 1 or runner.foldcp_size_cp != 1:
            raise ValueError('OpenDDE daemon supports single-device inference only')
        from astrolabe.runtime.client import DaemonClient
        python = shutil.which(runner.python_bin)
        if not python: raise FileNotFoundError('OpenDDE python executable not found')
        source = Path(runner.repo_path).resolve(strict=True)
        if not (source / 'opendde' / '__init__.py').is_file() or not (source / 'runner' / 'inference.py').is_file():
            raise ValueError('invalid OpenDDE source checkout')
        if not runner.load_checkpoint_path:
            raise ValueError('OpenDDE daemon requires an explicit local checkpoint')
        checkpoint = str(Path(runner.load_checkpoint_path).resolve(strict=True))
        env = {'PYTHONPATH': os.pathsep.join([str(source)] + [str(Path(p).resolve()) for p in sys.path if p])}
        if runner.opendde_root_dir is not None:
            env['OPENDDE_ROOT_DIR'] = str(runner.opendde_root_dir)
        device = probe_device(python, 'opendde', env=env)
        cmd = [python, '-m', 'astrolabe.evaluators.opendde_assets', runner.model_name, checkpoint, '1' if runner.use_template else '0']
        try:
            from astrolabe.runtime.admission import run_probe
        except ImportError:
            asset_env = os.environ.copy(); asset_env.update(env)
            assets = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True, env=asset_env).stdout)['assets']
        else:
            assets = run_probe(cmd, env=env)['assets']
        self._runner = runner
        self._runner_snapshot = asdict(runner)
        options = dict(self._runner_snapshot)
        options = {k: str(v) if isinstance(v, Path) else v for k, v in options.items()}
        for key in ('python_bin', 'repo_path', 'timeout_seconds', 'execution'):
            options.pop(key, None)
        self._artifacts = WorkerArtifacts()
        root = self._artifacts.root
        base = Path(__file__).resolve().parent
        code = [str(base / p) for p in ('opendde_assets.py', 'worker_identity.py', 'request_path.py', 'device_probe.py', 'wrappers/opendde_daemon.py')]
        code += [str(source / 'opendde'), str(source / 'runner')]
        semantics = {'options': options, 'trajectory': uuid.uuid4().hex}
        expected = identity(python, code, assets, device, semantics)
        spec = {'source': str(source), 'options': options, 'semantics': semantics,
                'code_paths': code, 'weight_paths': assets, 'device': device,
                'bootstrap_dir': str(root / 'bootstrap')}
        spec_path = self._artifacts.specification(expected, spec)
        self._client = DaemonClient([python, '-m', 'astrolabe.evaluators.wrappers.opendde_daemon', str(spec_path)], expected, env=env, startup_timeout=600)
        self.timeout = runner.timeout_seconds

    def run(self, input_json, output_dir):
        if asdict(self._runner) != self._runner_snapshot:
            raise ValueError('OpenDDE configuration changed after model initialization')
        data = json.loads(Path(input_json).read_text())
        if not isinstance(data, list) or len(data) != 1 or not data[0].get('modelSeeds'):
            raise ValueError('OpenDDE daemon requires one job with explicit modelSeeds')
        name = data[0].get('name')
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid OpenDDE job name')
        with tempfile.TemporaryDirectory(prefix='request-', dir=self._artifacts.root) as tmp:
            request = Path(tmp)
            (request / 'input.json').write_text(json.dumps(data))
            reply = self._client.call({'request_dir': str(request)}, timeout=self.timeout)
            if reply != {'request': request.name}: raise ValueError('OpenDDE request mismatch')
            source = request / 'output' / name
            if not source.is_dir(): raise RuntimeError('OpenDDE output is missing')
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
