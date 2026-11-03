"""Resident Chai1 worker using the vendor component cache and offload policy."""
import contextlib
import hashlib
import importlib
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
        raise RuntimeError('Chai1 visible device mismatch')
    with contextlib.redirect_stdout(sys.stderr):
        from astrolabe.evaluators.chai1 import _load_chai1_module
        from astrolabe.evaluators.x_masking import patch_chai1_get_esm_embedding_context
        chai = _load_chai1_module()
        path = Path(chai.__file__).resolve(strict=True)
        if str(path) != spec['source'] or hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise RuntimeError('Chai1 implementation identity mismatch')
        paths = importlib.import_module(chai.chai1_component.__module__)
        def local_only(url, path):
            if not Path(path).is_file():
                raise FileNotFoundError(f'Chai1 asset is not cached: {path}')
        paths.download_if_not_exists = local_only
    def handle(payload):
        request = request_directory(payload['request_dir'])
        batch = load_tree(request / 'input')
        args = batch['args']
        for key in ('fasta_file', 'output_dir', 'constraint_path', 'msa_directory'):
            if args.get(key) is not None:
                args[key] = Path(args[key])
        with contextlib.redirect_stdout(sys.stderr):
            restore(batch['rng'], spec['device']['name'])
            with patch_chai1_get_esm_embedding_context(chai):
                output = chai.run_inference(**args, **spec['settings']['config'])
            ranking = []
            for score in output.ranking_data:
                ranking.append({'aggregate_score': score.aggregate_score,
                    'ptm_scores': {k: getattr(score.ptm_scores, k) for k in
                                   ('complex_ptm', 'interface_ptm', 'per_chain_ptm', 'per_chain_pair_iptm')},
                    'plddt_scores': {'per_chain_plddt': score.plddt_scores.per_chain_plddt}})
            root = (request / 'predictions').resolve(strict=True)
            cif_paths = []
            for path in output.cif_paths:
                path = Path(path).resolve(strict=True)
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError('Chai1 CIF outside request')
                cif_paths.append(str(path.relative_to(root)))
            result = {'cif_paths': cif_paths, 'ranking': ranking, 'pae': output.pae,
                      'pde': output.pde, 'plddt': output.plddt, 'rng': capture(spec['device']['name'])}
        dump_tree(request / 'output', result)
        return {'request': request.name}
    serve(handle, actual)


if __name__ == '__main__': main()
