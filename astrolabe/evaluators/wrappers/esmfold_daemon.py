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
    from astrolabe.runtime.admission import operation
    from astrolabe.evaluators.lifecycle import cpu_tree
    from astrolabe.evaluators.esmfold_output import SCORE_FIELDS, resolve_pdb_boundary
    import torch
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], spec['settings'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('ESMFold1 visible device mismatch')
    boot = request_directory(spec['bootstrap'])
    with contextlib.redirect_stdout(sys.stderr):
        restore(load_tree(boot / 'initial'), spec['device']['name'])
        from astrolabe.evaluators.esmfold import ESMFoldEvaluator, _load_esmfold_model_class
        from astrolabe.evaluators.loader_observer import LoaderObserver
        file_loads = LoaderObserver(torch, 'load')
        if str(Path(inspect.getfile(_load_esmfold_model_class())).resolve()) != spec['class_file']:
            raise RuntimeError('ESMFold1 model source mismatch')
        def no_download(*args, **kwargs):
            raise RuntimeError('ESMFold1 resident requires locally cached weights')
        torch.hub.download_url_to_file = no_download
        evaluator = ESMFoldEvaluator(model_name=spec['settings']['model_name'], device=spec['device']['name'])
        model = evaluator._ensure_model()
        pdb_boundary = resolve_pdb_boundary(model)
        dump_tree(boot / 'ready', capture(spec['device']['name']))
    def handle(payload):
        request = request_directory(payload['request_dir'])
        batch = load_tree(request / 'input')
        with contextlib.redirect_stdout(sys.stderr), torch.no_grad(), operation(actual, phase='device') as lease:
            restore(batch['rng'], spec['device']['name'])
            output = model.infer(batch['sequences'])
            # Unknown/custom helpers keep the native composite path. Missing
            # chain_index also keeps the original native error ordering.
            split_pdb = pdb_boundary is not None and 'chain_index' in output
            if split_pdb:
                host = pdb_boundary.prepare_output(output)
                selected = {key: host[key] for key in SCORE_FIELDS}
            else:
                pdbs = model.output_to_pdb(output)
                selected = cpu_tree({key: output[key] for key in SCORE_FIELDS})
            result = {'output': selected, 'rng': capture(spec['device']['name'])}
            # Drop device output references before yielding the device lease;
            # CPU formatting must not retain the unused pair/structure tensors.
            del output
            lease.update_lifecycle(**lifecycle())
        if split_pdb:
            pdbs = pdb_boundary.finalize_pdb(host)
        result['pdbs'] = pdbs
        dump_tree(request / 'output', result)
        return {'request': request.name}
    from astrolabe.evaluators.lifecycle import model_lifecycle
    def lifecycle():
        return model_lifecycle(model, load_count=evaluator._weight_load_count, loaded=evaluator._model is not None,
                               device=spec['device'], components={'checkpoint_deserializations': file_loads.count,
                               'checkpoint_read_failures': file_loads.failures, 'model_initializations': evaluator._weight_load_count,
                               'pdb_finalization': 'audited_host' if pdb_boundary is not None else 'native_composite'})
    serve(handle, actual, load_count=evaluator._weight_load_count, lifecycle=lifecycle,
          capabilities={'managed_device': True, 'managed_methods': ['infer']})


if __name__ == '__main__': main()
