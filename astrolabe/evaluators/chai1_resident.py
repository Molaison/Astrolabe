"""Trajectory-exclusive Chai1 resident-atomic adapter."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import uuid

from omegaconf import OmegaConf
from astrolabe.evaluators.tree_codec import dump_tree, load_tree
from astrolabe.evaluators.torch_rng import capture, restore, capture_settings
from astrolabe.evaluators.lifecycle import request_runtime
from astrolabe.evaluators.worker_identity import identity
from astrolabe.evaluators.worker_session import WorkerArtifacts, request_artifacts
from astrolabe.runtime.admission import run_probe
from astrolabe.runtime.client import DaemonClient, OutcomeUnknown


# These are the fixed reference and the installed vendor variant with the
# reviewed _component_cache / _component_moved_to CPU offload implementation.
CHAI1_SOURCES = {
    '962d8aaf33ecb4c8a2ec56439b01b632e87626879f9d889a84ade5fbdf322a63',
    'cd032c8eacb393b081ca14a8af764febf09aafe1727cebdccc085e7c6305b7a9',
}
COMPONENTS = ('feature_embedding.pt', 'bond_loss_input_proj.pt', 'token_embedder.pt',
              'trunk.pt', 'diffusion_module.pt', 'confidence_head.pt')


BROKER_SHARED_ROOT = True
BROKER_ADMITTED_PROBE = True


class Chai1Resident:
    def __init__(self, config):
        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)
        self.config = dict(config)
        component_residency = self.config.pop('component_residency', 'cpu')
        if component_residency not in ('cpu', 'device'):
            raise ValueError('Chai component_residency must be cpu or device')
        if self.config.get('use_msa_server', False) or self.config.get('use_templates_server', False):
            raise ValueError('Chai1 resident requires local MSA and templates')
        env = {'PYTHONPATH': os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p),
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        child_env = os.environ.copy(); child_env.update(env)
        script = """import hashlib,inspect,json,importlib
from pathlib import Path
from astrolabe.evaluators.chai1 import _load_chai1_module
m=_load_chai1_module()
paths=importlib.import_module(m.chai1_component.__module__)
p=Path(m.__file__).resolve()
print(json.dumps({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'downloads':str(paths.downloads_path.resolve()),'parameters':list(inspect.signature(m.run_inference).parameters)}))
"""
        discovered = json.loads(subprocess.run([sys.executable, '-c', script], env=child_env, check=True,
                                               capture_output=True, text=True).stdout)
        if discovered['sha256'] not in CHAI1_SOURCES:
            raise ValueError('unreviewed Chai1 vendor implementation')
        self.parameters = set(discovered['parameters'])
        downloads = Path(discovered['downloads']).resolve(strict=True)
        needed = [downloads / 'conformers_v1.apkl', *(downloads / 'models_v2' / c for c in COMPONENTS)]
        if self.config.get('use_esm_embeddings', True):
            needed.append(downloads / 'esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt')
        if any(not path.is_file() for path in needed):
            raise ValueError('Chai1 resident requires locally cached components and conformers')
        code_file = Path(discovered['file']).resolve(strict=True)
        base = Path(__file__).resolve().parent
        code = [str(base / p) for p in ('chai1.py', 'chai1_resident.py', 'x_masking.py',
                'chai_components.py', 'chai_stages.py', 'wrappers/chai1_daemon.py', 'tree_codec.py', 'torch_rng.py', 'worker_identity.py')]
        code.append(str(code_file.parent))
        probe = """import json,os,sys,torch
requested=sys.argv[1]
name=requested if requested else ('cuda:0' if torch.cuda.is_available() else 'cpu')
if name not in ('cpu','cuda:0'): raise ValueError('unsupported Chai1 device')
print(json.dumps({'name':name,'device_kind':torch.cuda.get_device_name(0) if name=='cuda:0' else 'cpu','visible':os.environ.get('CUDA_VISIBLE_DEVICES')}))
"""
        device = run_probe([sys.executable, '-c', probe, str(self.config.get('device') or '')], env=env,
                           device='cpu' if self.config.get('device') == 'cpu' else None)
        self.device = device['name']
        self.config['device'] = self.device
        unknown = set(self.config) - self.parameters
        if unknown:
            raise ValueError(f'unsupported Chai1 model settings: {sorted(unknown)}')
        self._artifacts = WorkerArtifacts()
        settings = {'config': self.config, 'trajectory': uuid.uuid4().hex,
                    'component_residency': component_residency, 'torch_settings': capture_settings()}
        expected = identity(sys.executable, code, [str(downloads)], device, settings, env=env)
        spec = {'settings': settings, 'code_paths': code, 'weight_paths': [str(downloads)],
                'device': device, 'source': str(code_file), 'sha256': discovered['sha256']}
        path = self._artifacts.specification(expected, spec)
        self._client = self._artifacts.start_client([sys.executable, '-m', 'astrolabe.evaluators.wrappers.chai1_daemon', str(path)],
                                    expected, env=env, startup_timeout=600)

    def run(self, input_feat):
        result = {}
        for name, original in input_feat.items():
            args = dict(original)
            for key in ('constraint_path', 'cyclic_subchains', 'cyclic_relative_patches'):
                if args.get(key) and key not in self.parameters:
                    raise ValueError(f'Chai1 implementation does not support {key}')
            for key in list(args):
                if key not in self.parameters:
                    if args[key] is not None:
                        raise ValueError(f'unsupported Chai1 request field {key}')
                    del args[key]
            with request_artifacts(self._artifacts.root) as tmp:
                request = Path(tmp)
                destination = Path(args['output_dir'])
                if destination.exists() and any(destination.iterdir()):
                    raise ValueError('Chai1 output directory is not empty')
                for key in ('fasta_file', 'constraint_path'):
                    if args.get(key) is not None:
                        target = request / key
                        shutil.copyfile(args[key], target)
                        args[key] = str(target)
                if args.get('msa_directory') is not None:
                    target = request / 'msa'
                    shutil.copytree(args['msa_directory'], target)
                    args['msa_directory'] = str(target)
                args['output_dir'] = str(request / 'predictions')
                dump_tree(request / 'input', {'args': args, 'rng': capture(self.device)})
                response = self._client.call({'request_dir': str(request), 'method': 'run'}, runtime=request_runtime(request, 'managed', args, execution_settings=capture_settings()))
                try:
                    if response != {'request': request.name}:
                        raise ValueError('Chai1 response request mismatch')
                    output = load_tree(request / 'output')
                    restore(output['rng'], self.device)
                    paths = []
                    for value in output['cif_paths']:
                        path = (request / 'predictions' / value).resolve(strict=True)
                        if not path.is_relative_to(request / 'predictions'):
                            raise ValueError('Chai1 result path outside request')
                        paths.append(destination / path.relative_to(request / 'predictions'))
                    ranking = [SimpleNamespace(aggregate_score=r['aggregate_score'],
                               ptm_scores=SimpleNamespace(**r['ptm_scores']),
                               plddt_scores=SimpleNamespace(**r['plddt_scores'])) for r in output['ranking']]
                    shutil.copytree(request / 'predictions', destination, dirs_exist_ok=True)
                    result[name] = SimpleNamespace(cif_paths=paths, ranking_data=ranking,
                                                  pae=output['pae'], pde=output['pde'], plddt=output['plddt'])
                except Exception as exc:
                    self._artifacts.close_client(self._client, release=True)
                    raise OutcomeUnknown('invalid Chai1 result; worker fenced') from exc
        return result

    def close(self):
        if getattr(self, '_client', None) is not None:
            self._artifacts.close_client(self._client, release=True); self._client = None
        if getattr(self, '_artifacts', None) is not None:
            self._artifacts.close(); self._artifacts = None

    def __del__(self):
        try: self.close()
        except Exception: pass
