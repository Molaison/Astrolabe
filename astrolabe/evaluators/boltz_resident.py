"""Trajectory-exclusive Boltz1 resident prediction with owner-side preparation."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

from omegaconf import OmegaConf

from astrolabe.evaluators.tree_codec import dump_tree, load_tree
from astrolabe.evaluators.torch_rng import capture, restore, capture_settings
from astrolabe.evaluators.lifecycle import request_runtime
from astrolabe.evaluators.worker_identity import identity
from astrolabe.evaluators.worker_session import WorkerArtifacts
from astrolabe.runtime.admission import run_probe


BROKER_SHARED_ROOT = True
BROKER_ADMITTED_PROBE = True


class Boltz1Resident:
    resident_atomic = True

    def __init__(self, config):
        from astrolabe.runtime.client import DaemonClient
        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)
        self.config = dict(config)
        if self.config.get('devices') not in (1, [0]):
            raise ValueError('Boltz1 resident requires exactly one device')
        if self.config.get('accelerator') not in ('cpu', 'gpu', 'cuda', 'auto'):
            raise ValueError('unsupported Boltz1 accelerator')
        cache = Path(self.config['cache']).expanduser().resolve(strict=True)
        weights = [cache / 'ccd.pkl', cache / 'boltz1_conf.ckpt']
        if self.config.get('checkpoint'):
            weights.append(Path(self.config['checkpoint']).expanduser().resolve(strict=True))
        if any(not p.is_file() for p in weights):
            raise ValueError('Boltz1 resident requires local CCD and model checkpoints')
        upstream = importlib.util.find_spec('boltz')
        if upstream is None or not upstream.submodule_search_locations:
            raise ValueError('cannot locate Boltz1 implementation')
        package = Path(next(iter(upstream.submodule_search_locations))).resolve(strict=True)
        base = Path(__file__).resolve().parent
        code = [str(base / name) for name in ('boltz.py', 'boltz_resident.py', 'wrappers/boltz.py',
                'wrappers/boltz_daemon.py', 'tree_codec.py', 'torch_rng.py', 'worker_identity.py')]
        code.append(str(package))
        env = {'PYTHONPATH': os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p)}
        probe = """import json,os,sys,torch
acc=sys.argv[1]
if acc in ('gpu','cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
name='cpu' if acc=='cpu' or not torch.cuda.is_available() else 'cuda:0'
print(json.dumps({'name':name,'device_kind':torch.cuda.get_device_name(0) if name.startswith('cuda') else 'cpu','visible':os.environ.get('CUDA_VISIBLE_DEVICES')}))
"""
        device = run_probe([sys.executable, '-c', probe, self.config['accelerator']], env=env,
                           device='cpu' if self.config['accelerator'] == 'cpu' else None)
        self.device = device['name']
        self._artifacts = WorkerArtifacts()
        settings = {'config': self.config, 'trajectory': uuid.uuid4().hex,
                    'torch_settings': capture_settings()}
        expected = identity(sys.executable, code, [str(p) for p in weights], device, settings, env=env)
        spec = {'settings': settings, 'code_paths': code, 'weight_paths': [str(p) for p in weights], 'device': device}
        path = self._artifacts.specification(expected, spec)
        self._client = self._artifacts.start_client([sys.executable, '-m', 'astrolabe.evaluators.wrappers.boltz_daemon', str(path)],
                                    expected, env=env, startup_timeout=600)

    def prepare_request(self, yaml_path, destination):
        destination = Path(destination)
        if destination.exists():
            raise ValueError('Boltz1 resident output already exists')
        request = Path(tempfile.mkdtemp(prefix='request-', dir=self._artifacts.root))
        try:
            # Same owner-side preprocessing as Boltz1Runner.featurize_data,
            # with an explicit local-assets check in place of download().
            from astrolabe.evaluators.wrappers.boltz import check_inputs, process_inputs
            data = check_inputs(yaml_path, request, False)
            if data:
                cache = Path(self.config['cache']).expanduser()
                if not (cache / 'ccd.pkl').is_file() or not (cache / 'boltz1_conf.ckpt').is_file():
                    raise ValueError('Boltz1 local assets missing')
                process_inputs(data=data, out_dir=request, ccd_path=cache / 'ccd.pkl',
                               use_msa_server=self.config['use_msa_server'],
                               msa_server_url=self.config['msa_server_url'] or 'https://api.colabfold.com',
                               msa_pairing_strategy=self.config['msa_pairing_strategy'])
            return data, request
        except BaseException:
            shutil.rmtree(request, ignore_errors=True)
            raise

    def run_request(self, data, request, destination):
        from astrolabe.runtime.client import OutcomeUnknown
        request = Path(request)
        uncertain = False
        try:
            dump_tree(request / 'input', {'data': [str(p) for p in data], 'rng': capture(self.device)})
            response = self._client.call({'request_dir': str(request), 'method': 'predict'}, runtime=request_runtime(request, 'managed', data, execution_settings=capture_settings()))
            if response != {'request': request.name}:
                raise ValueError('Boltz1 response request mismatch')
            restore(load_tree(request / 'output')['rng'], self.device)
            # Preserve raw processed/prediction artifacts under the legacy path.
            destination = Path(destination)
            if destination.exists():
                raise ValueError('Boltz1 resident output already exists')
            shutil.copytree(request, destination,
                            ignore=lambda path, names: {'input', 'output'} & set(names) if Path(path) == request else set())
        except OutcomeUnknown as exc:
            uncertain = True
            exc.retained_request_dir = str(request)
            raise
        finally:
            if not uncertain:
                shutil.rmtree(request, ignore_errors=True)

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._artifacts.close_client(self._client, release=True); self._client = None
        if getattr(self, '_artifacts', None) is not None:
            self._artifacts.close(); self._artifacts = None

    def __del__(self):
        try: self.close()
        except Exception: pass
