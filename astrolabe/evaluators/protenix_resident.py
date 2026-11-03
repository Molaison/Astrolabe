"""Trajectory-exclusive resident Protenix runner with RNG state exchange."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

from astrolabe.evaluators.tree_codec import dump_tree, load_tree
from astrolabe.evaluators.lifecycle import request_runtime
from astrolabe.evaluators.worker_identity import identity, probe_device
from astrolabe.evaluators.worker_session import WorkerArtifacts, request_artifacts
from astrolabe.evaluators.torch_rng import capture, restore, capture_settings


BROKER_SHARED_ROOT = True
BROKER_ADMITTED_PROBE = True


class ProtenixResident:
    def __init__(self, config, *, source_dir, startup_timeout=600, timeout=None):
        # Broker reuse cannot preserve legacy random state without a coordinated
        # owner-side RNG lease. Direct mode owns this worker for one trajectory.
        if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(os.environ.get('LOCAL_RANK', '0')) != 0:
            raise ValueError('Protenix daemon supports one local device only')
        from astrolabe.runtime.client import DaemonClient
        from astrolabe.evaluators.wrappers.protenix import (
            DataDumper, _as_plain_mapping, _configdict_safe_value,
            _normalize_protenix_configs, configure_protenix_source,
        )
        source = Path(source_dir).expanduser().resolve(strict=True)
        configure_protenix_source(source)
        raw = _configdict_safe_value(_as_plain_mapping(config))
        raw = json.loads(json.dumps(raw, allow_nan=False))
        checkpoint = raw.get('load_checkpoint_path')
        if not checkpoint and raw.get('load_checkpoint_dir') and raw.get('model_name'):
            checkpoint = str(Path(raw['load_checkpoint_dir']) / (raw['model_name'] + '.pt'))
        if not checkpoint: raise ValueError('Protenix daemon requires an explicit checkpoint')
        checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        if not checkpoint.is_file(): raise ValueError('Protenix checkpoint must be a file')
        ptx_root = Path(os.environ.get('PROTENIX_ROOT_DIR', str(Path.home() / '.gh_envs/repos/Protenix')))
        ccd = Path(os.environ.get('GH_PROTENIX_CCD', str(ptx_root / 'release_data/components.v20240608.cif'))).expanduser().resolve(strict=True)
        ccd_rdkit = Path(os.environ.get('GH_PROTENIX_CCD_RDKIT', str(ptx_root / 'release_data/components.v20240608.cif.rdkit_mol.pkl'))).expanduser().resolve(strict=True)
        if not ccd.is_file() or not ccd_rdkit.is_file(): raise ValueError('Protenix CCD assets must be files')
        # Same normalization as ProtenixRunner, without constructing a model.
        self.configs = _normalize_protenix_configs(config)
        requested_model = raw.get('model_name')
        normalized_model = getattr(self.configs, 'model_name', None)
        if requested_model is not None and normalized_model != requested_model:
            raise ValueError(f'Protenix model identity changed: {requested_model!r} -> {normalized_model!r}')
        self.dumper = DataDumper(base_dir=self.configs.dump_dir,
                                 need_atom_confidence=self.configs.need_atom_confidence,
                                 sorted_by_ranking_score=self.configs.sorted_by_ranking_score)
        self._artifacts = WorkerArtifacts()
        root = self._artifacts.root
        env = {'PYTHONPATH': os.pathsep.join([str(source)] + [str(Path(p).resolve()) for p in sys.path if p])}
        device = probe_device(sys.executable, 'protenix', env=env)
        self._device = device['name']
        initial_rng = capture(self._device)
        bootstrap = Path(tempfile.mkdtemp(prefix='request-bootstrap-', dir=root))
        dump_tree(bootstrap / 'initial', initial_rng)
        base = Path(__file__).resolve().parent
        code = [str(base / name) for name in ('tree_codec.py', 'torch_rng.py', 'request_path.py', 'worker_identity.py', 'wrappers/protenix.py', 'wrappers/protenix_daemon.py')] + [str(source / 'protenix'), str(source / 'configs')]
        weights = [str(checkpoint), str(ccd), str(ccd_rdkit)]
        # Direct mode never shares a worker: the nonce also prevents future
        # broker implementations from merging independent trajectory streams.
        semantics = {'config': raw, 'effective_model': normalized_model, 'trajectory': uuid.uuid4().hex,
                     'torch_settings': capture_settings()}
        expected = identity(sys.executable, code, weights, device, semantics, env=env)
        spec = {'model_config': raw, 'semantics': semantics, 'source_dir': str(source),
                'code_paths': code, 'weight_paths': weights, 'device': device,
                'bootstrap': str(bootstrap)}
        path = self._artifacts.specification(expected, spec)
        self._client = self._artifacts.start_client([sys.executable, '-m', 'astrolabe.evaluators.wrappers.protenix_daemon', str(path)], expected, env=env, startup_timeout=startup_timeout)
        restore(load_tree(bootstrap / 'ready'), self._device)
        self.timeout = timeout
        self._pending_config = None

    def update_model_configs(self, new_configs):
        self._pending_config = new_configs

    def predict(self, data):
        if self._pending_config is None:
            raise RuntimeError('Protenix request did not update its model config')
        with request_artifacts(self._artifacts.root) as tmp:
            request = Path(tmp)
            settings = self._pending_config
            payload = {'data': data, 'rng': capture(self._device),
                       'n_token': int(data['N_token'].item()), 'seed': self.configs.seed,
                       'skip_amp': {'confidence_head': bool(settings.skip_amp.confidence_head),
                                    'sample_diffusion': bool(settings.skip_amp.sample_diffusion)}}
            dump_tree(request / 'input', payload)
            self._pending_config = None
            reply = self._client.call({'request_dir': str(request), 'method': 'predict'}, timeout=self.timeout, runtime=request_runtime(request, 'managed', data, execution_settings=capture_settings()))
            if reply != {'request': request.name}: raise ValueError('Protenix request mismatch')
            output = load_tree(request / 'output')
            restore(output['rng'], self._device)
            return output['prediction']

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._artifacts.close_client(self._client, release=True); self._client = None
        if getattr(self, '_artifacts', None) is not None:
            self._artifacts.close(); self._artifacts = None

    def __del__(self):
        try: self.close()
        except Exception: pass
