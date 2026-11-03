"""RhoFold weights and RNA-FM live in one trajectory-owned device worker."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

from omegaconf import OmegaConf
from astrolabe.evaluators.lifecycle import request_runtime
from astrolabe.evaluators.torch_rng import capture, restore, capture_settings
from astrolabe.evaluators.tree_codec import dump_tree, load_tree
from astrolabe.evaluators.worker_identity import identity, probe_device
from astrolabe.evaluators.worker_session import WorkerArtifacts, request_artifacts

BROKER_SHARED_ROOT = True
BROKER_ADMITTED_PROBE = True


class RhoFoldResident:
    def __init__(self, config):
        from astrolabe.runtime.client import DaemonClient
        options = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
        options.pop('execution', None)
        source_spec = importlib.util.find_spec('rhofold')
        if source_spec is None or not source_spec.submodule_search_locations:
            raise ValueError('RhoFold implementation is not installed')
        source = Path(next(iter(source_spec.submodule_search_locations))).resolve(strict=True)
        weight = Path(options['checkpoint']).expanduser().resolve(strict=True)
        import torch
        hub = Path(torch.hub.get_dir()) / 'checkpoints'
        # RNA-FM may load a separate hub checkpoint during construction. All
        # locally available inputs are pinned and downloads are forbidden.
        weights = [str(weight)] + ([str(hub.resolve())] if hub.is_dir() else [])
        env = {'PYTHONPATH': os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p)}
        device = probe_device(sys.executable, 'rhofold', options.get('device', 'cuda'), env=env)
        self.device = device['name']
        options['device'] = self.device
        self._initialization_settings = json.dumps(options, sort_keys=True)
        self._artifacts = WorkerArtifacts()
        root = self._artifacts.root
        if self._artifacts.shared and int(options.get('relax_steps') or 0) > 0:
            platform = options.get('openmm_platform')
            if platform not in ('CPU', 'Reference', 'CUDA', 'OpenCL'):
                raise ValueError('RhoFold shared relaxation requires explicit openmm_platform matching the legacy run')
            if platform == 'OpenCL' and not options.get('openmm_properties'):
                raise ValueError('OpenCL relaxation requires explicit PlatformIndex/DeviceIndex allocation binding')
            if platform in ('CUDA', 'OpenCL') and not self.device.startswith('cuda'):
                raise ValueError('RhoFold GPU relaxation requires a GPU-bound worker')
        boot = Path(tempfile.mkdtemp(prefix='request-bootstrap-', dir=root))
        dump_tree(boot / 'initial', capture(self.device))
        base = Path(__file__).resolve().parent
        code = [str(source)] + [str(base / p) for p in ('rhofold.py', 'rhofold_resident.py', 'openmm_binding.py', 'wrappers/rhofold_daemon.py', 'tree_codec.py')]
        settings = {'config': options, 'trajectory': uuid.uuid4().hex,
                    'torch_settings': capture_settings()}
        expected = identity(sys.executable, code, weights, device, settings, env=env)
        spec = {'settings': settings, 'device': device, 'source': str(source), 'code_paths': code,
                'weight_paths': weights, 'bootstrap': str(boot), 'shared': self._artifacts.shared}
        path = self._artifacts.specification(expected, spec)
        self._client = self._artifacts.start_client([sys.executable, '-m', 'astrolabe.evaluators.wrappers.rhofold_daemon', str(path)],
                                    expected, env=env, startup_timeout=600)
        restore(load_tree(boot / 'ready'), self.device)

    def validate_config(self, config):
        current = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
        current.pop('execution', None)
        if current.get('device') == 'cuda':
            current['device'] = 'cuda:0'
        if json.dumps(current, sort_keys=True) != self._initialization_settings:
            raise ValueError('RhoFold initialization or relaxation settings changed')

    def forward(self, features):
        with request_artifacts(self._artifacts.root) as tmp:
            path = Path(tmp)
            dump_tree(path / 'input', {'features': features, 'rng': capture(self.device)})
            reply = self._client.call({'request_dir': str(path), 'method': 'forward'},
                                      runtime=request_runtime(path, 'managed', features, execution_settings=capture_settings()))
            if reply != {'request': path.name}:
                raise ValueError('RhoFold request mismatch')
            result = load_tree(path / 'output')
            restore(result['rng'], self.device)
            return result['output']

    def relax(self, input_path, output_path):
        with request_artifacts(self._artifacts.root) as tmp:
            path = Path(tmp)
            shutil.copy2(input_path, path / 'input.pdb')
            dump_tree(path / 'rng', capture(self.device))
            reply = self._client.call({'request_dir': str(path), 'method': 'relax'},
                                      runtime=request_runtime(path, 'managed', {'pdb': (path / 'input.pdb').read_text()}, execution_settings=capture_settings()))
            if reply != {'request': path.name}:
                raise ValueError('RhoFold relaxation request mismatch')
            restore(load_tree(path / 'output')['rng'], self.device)
            shutil.copy2(path / 'relaxed.pdb', output_path)

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._artifacts.close_client(self._client, release=True)
            self._client = None
        if getattr(self, '_artifacts', None) is not None:
            self._artifacts.close()
            self._artifacts = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
