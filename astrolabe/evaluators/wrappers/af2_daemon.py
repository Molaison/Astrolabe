"""Resident AlphaFold2 forward worker, with CPU features supplied by owner."""
import contextlib
import json
import os
from pathlib import Path
import sys


def main():
    fd = os.dup(1); os.dup2(2, 1); sys.stdout = os.fdopen(fd, 'w', buffering=1)
    spec = json.loads(Path(sys.argv[1]).read_text())
    from astrolabe.evaluators.worker_identity import identity
    from astrolabe.evaluators.tree_codec import dump_tree, load_tree
    from astrolabe.evaluators.request_path import request_directory
    from astrolabe.runtime.server import serve
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], spec['semantics'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('AF2 visible device mismatch')
    with contextlib.redirect_stdout(sys.stderr):
        import alphafold
        import jax
        import numpy as np
        if str(Path(alphafold.__file__).resolve().parent) not in spec['code_paths']:
            raise RuntimeError('AF2 source mismatch')
        d = jax.local_devices(backend=spec['device']['platform'])[0]
        observed = {'platform': str(d.platform), 'id': int(d.id), 'device_kind': str(d.device_kind), 'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}
        if observed != spec['device']: raise RuntimeError('AF2 device mismatch')
        from astrolabe.evaluators.wrappers.alphafold import alphafold_setup
        cfg = spec['semantics']['config']
        model_name = 'model_%s_%s' % (cfg.get('model_id', 3), cfg.get('suffix', 'multimer_v3'))
        weight = (Path(cfg['data_dir']).expanduser() / 'params' / ('params_' + model_name + '.npz')).resolve(strict=True)
        if str(weight) != spec['weight_paths'][0]: raise RuntimeError('AF2 checkpoint mismatch')
        runner = alphafold_setup(oligo_names=['worker'], **cfg)['worker']
    def handle(payload):
        request = request_directory(payload['request_dir'])
        batch = load_tree(request / 'input')
        with contextlib.redirect_stdout(sys.stderr), jax.default_device(d):
            prediction = runner.predict(batch['features'], random_seed=batch['seed'])
            prediction = jax.tree_util.tree_map(np.asarray, prediction)
        dump_tree(request / 'output', prediction)
        return {'request': request.name}
    serve(handle, actual)


if __name__ == '__main__': main()
