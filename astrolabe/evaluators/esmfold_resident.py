"""Trajectory-exclusive ESMFold1 resident-atomic execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import torch

from astrolabe.evaluators.tree_codec import dump_tree, load_tree
from astrolabe.evaluators.torch_rng import capture, restore, capture_settings
from astrolabe.evaluators.lifecycle import request_runtime
from astrolabe.evaluators.worker_identity import identity, probe_device
from astrolabe.evaluators.worker_session import WorkerArtifacts, request_artifacts


BROKER_SHARED_ROOT = True
BROKER_ADMITTED_PROBE = True


class ESMFoldResident:
    def __init__(self, model_name, device):
        from astrolabe.runtime.client import DaemonClient
        checkpoint = Path(model_name).expanduser() if model_name.endswith('.pt') else Path(torch.hub.get_dir()) / 'checkpoints' / (model_name + '.pt')
        checkpoint = checkpoint.resolve(strict=True)
        if not checkpoint.is_file():
            raise ValueError('ESMFold1 checkpoint must be a local file')
        # Discover the exact model implementation in a separate interpreter so
        # imports cannot consume the trajectory's random state.
        env = {'PYTHONPATH': os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p)}
        child_env = os.environ.copy(); child_env.update(env)
        script = """import inspect,json,sys
from astrolabe.evaluators.esmfold import _load_esmfold_model_class
cls = _load_esmfold_model_class()
roots = []
for name in ('esm', 'esm2', 'openfold'):
    module = sys.modules.get(name)
    if module is not None:
        roots.extend(list(getattr(module, '__path__', ())))
print(json.dumps({'class_file': inspect.getfile(cls), 'roots': roots}))
"""
        discovered = json.loads(subprocess.run([sys.executable, '-c', script], env=child_env, check=True, capture_output=True, text=True).stdout)
        cls_file = Path(discovered['class_file']).resolve(strict=True)
        base = Path(__file__).resolve().parent
        code = [str(base / p) for p in ('esmfold.py', 'esmfold_resident.py', 'tree_codec.py', 'torch_rng.py', 'worker_identity.py', 'wrappers/esmfold_daemon.py')]
        code.append(str(cls_file.parent))
        code.extend(sorted({str(Path(p).resolve(strict=True)) for p in discovered['roots']}))
        # The ESM trunk may load a separate local hub checkpoint. Pin every
        # checkpoint in that directory and forbid network downloads in worker.
        hub_cache = Path(torch.hub.get_dir()) / 'checkpoints'
        weights = [str(checkpoint)]
        if hub_cache.is_dir(): weights.append(str(hub_cache.resolve()))
        device_info = probe_device(sys.executable, 'esmfold1', device, env=env)
        self._device = device_info['name']
        self._artifacts = WorkerArtifacts()
        root = self._artifacts.root
        boot = Path(tempfile.mkdtemp(prefix='request-bootstrap-', dir=root))
        dump_tree(boot / 'initial', capture(self._device))
        settings = {'model_name': str(checkpoint), 'trajectory': uuid.uuid4().hex,
                    'torch_settings': capture_settings()}
        expected = identity(sys.executable, code, weights, device_info, settings, env=env)
        spec = {'settings': settings, 'code_paths': code, 'weight_paths': weights,
                'device': device_info, 'class_file': str(cls_file), 'bootstrap': str(boot)}
        path = self._artifacts.specification(expected, spec)
        self._client = self._artifacts.start_client([sys.executable, '-m', 'astrolabe.evaluators.wrappers.esmfold_daemon', str(path)], expected, env=env, startup_timeout=600)
        restore(load_tree(boot / 'ready'), self._device)

    def run(self, input_feat):
        from astrolabe.evaluators.esmfold import ESMFoldEvaluator
        names = list(input_feat)
        with request_artifacts(self._artifacts.root) as tmp:
            request = Path(tmp)
            dump_tree(request / 'input', {'sequences': [input_feat[n] for n in names], 'rng': capture(self._device)})
            reply = self._client.call({'request_dir': str(request), 'method': 'infer'}, runtime=request_runtime(request, 'managed', input_feat, execution_settings=capture_settings()))
            if reply != {'request': request.name}: raise ValueError('ESMFold1 response request mismatch')
            result = load_tree(request / 'output')
            restore(result['rng'], self._device)
            return ESMFoldEvaluator._format_scores(names, result['output'], result['pdbs'])

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._artifacts.close_client(self._client, release=True); self._client = None
        if getattr(self, '_artifacts', None) is not None:
            self._artifacts.close(); self._artifacts = None

    def __del__(self):
        try: self.close()
        except Exception: pass
