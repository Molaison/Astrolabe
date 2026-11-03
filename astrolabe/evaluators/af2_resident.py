"""AF2 CPU feature processing with trajectory-exclusive resident forward."""
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

from astrolabe.evaluators.tree_codec import dump_tree, load_tree
from astrolabe.evaluators.worker_identity import identity, probe_device
from astrolabe.evaluators.worker_session import WorkerArtifacts


class AF2Resident:
    @classmethod
    def runners(cls, oligo_names, config):
        from astrolabe.evaluators.wrappers.alphafold import alphafold_setup
        cpu = alphafold_setup(oligo_names=oligo_names, prepare_only=True, **config)
        proxy = cls(next(iter(cpu.values())), config)
        return {name: proxy for name in oligo_names}

    def __init__(self, cpu_runner, config):
        if os.environ.get('ASTROLABE_EXECUTION_ENDPOINT') or os.environ.get('ASTROLABE_EXECUTION_TOKEN'):
            raise ValueError('AF2 daemon requires trajectory-exclusive direct execution')
        self._cpu = cpu_runner
        self.multimer_mode = cpu_runner.multimer_mode
        self._gh_initial_guess_cfg = cpu_runner._gh_initial_guess_cfg
        self._config = json.loads(json.dumps(config, allow_nan=False))
        self._client = None
        self._artifacts = None

    def process_features(self, raw_features, random_seed):
        return self._cpu.process_features(raw_features, random_seed=random_seed)

    def _start(self):
        from astrolabe.runtime.client import DaemonClient
        import alphafold
        data_dir = self._config.get('data_dir')
        if not data_dir: raise ValueError('AF2 daemon requires explicit data_dir')
        model_name = 'model_%s_%s' % (self._config.get('model_id', 3), self._config.get('suffix', 'multimer_v3'))
        weight = (Path(data_dir).expanduser() / 'params' / ('params_' + model_name + '.npz')).resolve(strict=True)
        source = Path(alphafold.__file__).resolve().parent
        base = Path(__file__).resolve().parent
        code = [str(base / p) for p in ('af2_resident.py', 'tree_codec.py', 'worker_identity.py', 'wrappers/alphafold.py', 'wrappers/af2_daemon.py')] + [str(source)]
        env = {'PYTHONPATH': os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p)}
        device = probe_device(sys.executable, 'af3', env=env)
        semantics = {'config': self._config, 'trajectory': uuid.uuid4().hex}
        expected = identity(sys.executable, code, [str(weight)], device, semantics)
        self._artifacts = WorkerArtifacts()
        spec = {'semantics': semantics, 'code_paths': code, 'weight_paths': [str(weight)], 'device': device}
        path = self._artifacts.specification(expected, spec)
        self._client = DaemonClient([sys.executable, '-m', 'astrolabe.evaluators.wrappers.af2_daemon', str(path)], expected, env=env, startup_timeout=600)

    def predict(self, processed_features, random_seed):
        if self._client is None: self._start()
        with tempfile.TemporaryDirectory(prefix='request-', dir=self._artifacts.root) as tmp:
            request = Path(tmp)
            dump_tree(request / 'input', {'features': processed_features, 'seed': int(random_seed)})
            answer = self._client.call({'request_dir': str(request)})
            if answer != {'request': request.name}: raise ValueError('AF2 response request mismatch')
            return load_tree(request / 'output')

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._client.close(); self._client = None
        if getattr(self, '_artifacts', None) is not None:
            self._artifacts.close(); self._artifacts = None

    def __del__(self):
        try: self.close()
        except Exception: pass
