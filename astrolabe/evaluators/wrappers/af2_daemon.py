"""Resident AlphaFold2 forward worker, with CPU features supplied by owner."""
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys


def main():
    fd = os.dup(1); os.dup2(2, 1); sys.stdout = os.fdopen(fd, 'w', buffering=1)
    spec = json.loads(Path(sys.argv[1]).read_text())
    from astrolabe.evaluators.worker_identity import identity, digest_path
    from astrolabe.evaluators.tree_codec import dump_tree, load_tree
    from astrolabe.evaluators.request_path import request_directory
    from astrolabe.runtime.server import serve
    from astrolabe.runtime.admission import operation
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], spec['semantics'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('AF2 visible device mismatch')
    with contextlib.redirect_stdout(sys.stderr):
        import alphafold
        import jax
        import numpy as np
        source = Path(alphafold.__file__).resolve().parent
        if str(source) not in spec['code_paths']:
            raise RuntimeError('AF2 source mismatch')
        if spec.get('broker_safe') and digest_path(source) != '334705a1a88ad96000acd5ddcf315696b4740c6fcc1f05e8108265fbb5f20d29':
            raise ValueError('AF2 broker requires the audited source revision')
        d = jax.local_devices(backend=spec['device']['platform'])[0]
        observed = {'platform': str(d.platform), 'id': int(d.id), 'device_kind': str(d.device_kind), 'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}
        if observed != spec['device']: raise RuntimeError('AF2 device mismatch')
        from astrolabe.evaluators.wrappers.alphafold import alphafold_setup
        cfg = spec['semantics']['config']
        model_name = 'model_%s_%s' % (cfg.get('model_id', 3), cfg.get('suffix', 'multimer_v3'))
        weight = (Path(cfg['data_dir']).expanduser() / 'params' / ('params_' + model_name + '.npz')).resolve(strict=True)
        if str(weight) != spec['weight_paths'][0]: raise RuntimeError('AF2 checkpoint mismatch')
        runner = alphafold_setup(oligo_names=['worker'], **cfg)['worker']
        if spec.get('broker_safe'):
            from alphafold.model.model import RunModel
            # Audited upstream predict consumes only fixed params, explicit
            # features and PRNGKey(seed); never permit lazy random parameters.
            audited = '9bd146358110bccf325c824e8d034b319d5c7592b901d035ab37555ae019d7fb'
            if (type(runner) is not RunModel or not runner.params or
                hashlib.sha256(inspect.getsource(RunModel.predict).encode()).hexdigest() != audited):
                raise ValueError('AF2 broker requires audited stateless RunModel.predict and loaded parameters')
        # Keep the loaded parameters on the bound device between requests.
        # No PRNG state is consumed and the original parameter dtypes remain.
        runner.params = jax.device_put(runner.params, d)
        jax.block_until_ready(runner.params)
    def handle(payload):
        request = request_directory(payload['request_dir'])
        batch = load_tree(request / 'input')
        with contextlib.redirect_stdout(sys.stderr), jax.default_device(d), operation(actual, phase='device'):
            prediction = runner.predict(batch['features'], random_seed=batch['seed'])
            prediction = jax.tree_util.tree_map(np.asarray, prediction)
        dump_tree(request / 'output', prediction)
        return {'request': request.name}
    from astrolabe.evaluators.lifecycle import model_lifecycle
    serve(handle, actual, load_count=1,
          lifecycle=lambda: model_lifecycle(runner.params, load_count=1, loaded=True,
                                           device=spec['device']),
          capabilities={'managed_device': True, 'managed_methods': ['forward']})


if __name__ == '__main__': main()
