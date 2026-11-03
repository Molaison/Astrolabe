"""One RF3 engine per exact configuration, using the common resource owner."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

from astrolabe.evaluators.lifecycle import request_runtime
from astrolabe.evaluators.worker_identity import identity, probe_device
from astrolabe.evaluators.worker_session import WorkerArtifacts, request_artifacts

BROKER_SHARED_ROOT = True
BROKER_ADMITTED_PROBE = True


class RF3Resident:
    def __init__(self, effective_config, *, exchange_owner_rng=False):
        from astrolabe.runtime.client import DaemonClient
        config = dict(effective_config)
        python = str(config.pop('python_bin', None) or sys.executable)
        python = shutil.which(python)
        if not python:
            raise FileNotFoundError('RF3 Python interpreter does not exist')
        source = Path(config.pop('repo_path')).expanduser().resolve(strict=True)
        checkpoint = Path(config['ckpt_path']).expanduser().resolve(strict=True)
        config['ckpt_path'] = str(checkpoint)
        if int(config.get('num_nodes', 1)) != 1 or int(config.get('devices_per_node', 1)) != 1:
            raise ValueError('RF3 common worker supports one allocated device per engine')
        base = Path(__file__).resolve().parent
        roots = [str(source / 'src'), str(source / 'models' / 'rf3' / 'src'), str(base.parents[1])]
        env = {'PYTHONPATH': os.pathsep.join(roots + [p for p in sys.path if p])}
        device = probe_device(python, 'rf3', env=env)
        self.device = device['name']
        self._artifacts = WorkerArtifacts()
        root = self._artifacts.root
        self._exchange_rng = exchange_owner_rng
        boot = Path(tempfile.mkdtemp(prefix='request-bootstrap-', dir=root))
        if exchange_owner_rng:
            from astrolabe.evaluators.torch_rng import capture, capture_settings
            from astrolabe.evaluators.tree_codec import dump_tree
            dump_tree(boot / 'initial', capture(self.device))
        code = [str(p) for p in (source / 'src', source / 'models' / 'rf3' / 'src') if p.exists()]
        if not code:
            raise ValueError('RF3 repo_path has no model implementation')
        code += [str(base / p) for p in ('rf3_resident.py', 'wrappers/rf3_resident_daemon.py', 'wrappers/rf3.py', 'tree_codec.py')]
        settings = {'config': config, 'exchange_owner_rng': exchange_owner_rng, 'trajectory': uuid.uuid4().hex,
                    'torch_settings': capture_settings() if exchange_owner_rng else None}
        expected = identity(python, code, [str(checkpoint)], device, settings, env=env)
        spec = {'source': str(source), 'settings': settings, 'code_paths': code, 'weight_paths': [str(checkpoint)],
                'device': device, 'bootstrap': str(boot)}
        spec_path = self._artifacts.specification(expected, spec)
        self._client = self._artifacts.start_client([python, '-m', 'astrolabe.evaluators.wrappers.rf3_resident_daemon', str(spec_path)],
                                    expected, env=env, startup_timeout=600)
        if exchange_owner_rng:
            from astrolabe.evaluators.torch_rng import restore
            from astrolabe.evaluators.tree_codec import load_tree
            restore(load_tree(boot / 'ready'), self.device)

    def run(self, request):
        with request_artifacts(self._artifacts.root) as tmp:
            directory = Path(tmp)
            data = json.loads(Path(request.prepared_input_path).read_text())
            (directory / 'input.json').write_text(json.dumps(data))
            payload = {'request_dir': str(directory), 'method': 'run', 'template_selection': list(request.template_selection),
                       'ground_truth_conformer_selection': list(request.ground_truth_conformer_selection),
                       'add_missing_atoms': bool(request.rf3_overrides.get('add_missing_atoms', True))}
            if self._exchange_rng:
                from astrolabe.evaluators.torch_rng import capture_settings
                payload['torch_settings'] = capture_settings()
            runtime = request_runtime(directory, 'managed', data, execution_settings=payload.get('torch_settings'))
            runtime['scientific_request_id'] = request.request_id
            result = self._client.call(payload, runtime=runtime)
            if result != {'request': directory.name}:
                raise ValueError('RF3 result request mismatch')
            if self._exchange_rng:
                from astrolabe.evaluators.torch_rng import restore
                from astrolabe.evaluators.tree_codec import load_tree
                restore(load_tree(directory / 'output_rng'), self.device)
            shutil.copytree(directory / 'output', request.output_dir, dirs_exist_ok=True)

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._artifacts.close_client(self._client, release=True); self._client = None
        if getattr(self, '_artifacts', None) is not None:
            self._artifacts.close(); self._artifacts = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
