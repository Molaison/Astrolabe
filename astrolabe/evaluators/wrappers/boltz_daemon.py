"""Resident Boltz1 prediction worker; preparation stays in the owner."""
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
    from astrolabe.evaluators.torch_rng import capture, restore
    from astrolabe.evaluators.request_path import request_directory
    from astrolabe.runtime.server import serve
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], spec['settings'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('Boltz1 visible device mismatch')
    with contextlib.redirect_stdout(sys.stderr):
        from astrolabe.evaluators.wrappers import boltz
        def no_download(cache):
            if not (Path(cache) / 'ccd.pkl').is_file() or not (Path(cache) / 'boltz1_conf.ckpt').is_file():
                raise ValueError('Boltz1 local assets missing')
        boltz.download = no_download
        runner = boltz.Boltz1Runner(spec['settings']['config'])
    def handle(payload):
        request = request_directory(payload['request_dir'])
        batch = load_tree(request / 'input')
        with contextlib.redirect_stdout(sys.stderr):
            restore(batch['rng'], spec['device']['name'])
            runner.run_inference([Path(p) for p in batch['data']], request)
            state = capture(spec['device']['name'])
        dump_tree(request / 'output', {'rng': state})
        return {'request': request.name}
    serve(handle, actual)


if __name__ == '__main__': main()
