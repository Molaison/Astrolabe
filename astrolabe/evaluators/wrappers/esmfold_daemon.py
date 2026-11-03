"""ESMFold1 resident-atomic worker using the standard daemon protocol."""
import contextlib
import inspect
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
    import torch
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], spec['settings'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('ESMFold1 visible device mismatch')
    boot = request_directory(spec['bootstrap'])
    with contextlib.redirect_stdout(sys.stderr):
        restore(load_tree(boot / 'initial'), spec['device']['name'])
        from astrolabe.evaluators.esmfold import ESMFoldEvaluator, _load_esmfold_model_class
        if str(Path(inspect.getfile(_load_esmfold_model_class())).resolve()) != spec['class_file']:
            raise RuntimeError('ESMFold1 model source mismatch')
        def no_download(*args, **kwargs):
            raise RuntimeError('ESMFold1 resident requires locally cached weights')
        torch.hub.download_url_to_file = no_download
        evaluator = ESMFoldEvaluator(model_name=spec['settings']['model_name'], device=spec['device']['name'])
        model = evaluator._ensure_model()
        dump_tree(boot / 'ready', capture(spec['device']['name']))
    def handle(payload):
        request = request_directory(payload['request_dir'])
        batch = load_tree(request / 'input')
        with contextlib.redirect_stdout(sys.stderr), torch.no_grad():
            restore(batch['rng'], spec['device']['name'])
            output = model.infer(batch['sequences'])
            pdbs = model.output_to_pdb(output)
            selected = {k: output[k] for k in ('chain_index', 'plddt', 'mean_plddt', 'ptm', 'predicted_aligned_error')}
            result = {'output': selected, 'pdbs': pdbs, 'rng': capture(spec['device']['name'])}
        dump_tree(request / 'output', result)
        return {'request': request.name}
    serve(handle, actual)


if __name__ == '__main__': main()
