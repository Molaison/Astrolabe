"""Molence Mol-1 resident-atomic worker using the pinned Python APIs."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    fd = os.dup(1); os.dup2(2, 1); sys.stdout = os.fdopen(fd, 'w', buffering=1)
    spec = json.loads(Path(sys.argv[1]).read_text())
    from astrolabe.evaluators.worker_identity import identity
    from astrolabe.evaluators.request_path import request_directory
    from astrolabe.runtime.server import serve
    from astrolabe.runtime.admission import operation, stage_event
    options = spec['options']
    if spec['semantics']['options'] != options or not spec['semantics']['trajectory']:
        raise ValueError('Mol-1 startup identity mismatch')
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], spec['semantics'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('Mol-1 visible device mismatch')
    source = Path(spec['source']).resolve(strict=True)
    if str(source / 'mol') not in spec['code_paths']:
        raise RuntimeError('Mol-1 source identity mismatch')
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import mol
        from mol.runner import batch_inference as batch
        from mol.runner.inference import infer_predict, InferenceRunner
        from astrolabe.evaluators.loader_observer import LoaderObserver
        checkpoint_loads = LoaderObserver(InferenceRunner, "load_checkpoint")
        if not Path(mol.__file__).resolve().is_relative_to(source / 'mol'):
            raise RuntimeError('Mol-1 imported unexpected source')
        if not Path(batch.__file__).resolve().is_relative_to(source / 'mol'):
            raise RuntimeError('Mol-1 imported unexpected runner')
        def verify_assets(cfg):
            if cfg.model_name != options['model_name'] or str(Path(cfg.load_checkpoint_path).expanduser().resolve()) != spec['weight_paths'][0]:
                raise RuntimeError('Mol-1 model identity changed')
            paths = [cfg.load_checkpoint_path, cfg.data.ccd_components_file, cfg.data.ccd_components_rdkit_mol_file,
                     cfg.data.pdb_cluster_file, cfg.data.obsolete_release_data_csv]
            paths = [str(Path(p).expanduser().resolve(strict=True)) for p in paths]
            if paths != spec['weight_paths'] or any(not Path(p).is_file() for p in paths):
                raise RuntimeError('Mol-1 model or reference assets differ')
        batch.download_inference_cache = verify_assets
        batch.inference_configs['dump_dir'] = spec['bootstrap_dir']
        runner = batch.get_default_runner(
            seeds=None, n_cycle=options['num_cycles'], n_step=options['num_steps'],
            n_sample=options['num_samples'], dtype=options['dtype'], device=spec['device']['name'],
            model_name=options['model_name'], load_checkpoint_path=options['load_checkpoint_path'],
            use_msa=options['use_msa'], trimul_kernel=options['triangle_multiplicative'],
            triatt_kernel=options['triangle_attention'], enable_fusion=options['enable_fusion'],
            enable_tf32=options['enable_tf32'], use_template=False, use_rna_msa=False,
            use_seeds_in_json=True, need_atom_confidence=options['need_atom_confidence'])
        if str(runner.device) != spec['device']['name']:
            raise RuntimeError('Mol-1 actual device mismatch')
        if str(runner.device).startswith('cuda') and torch.cuda.get_device_name(runner.device) != spec['device']['device_kind']:
            raise RuntimeError('Mol-1 actual GPU mismatch')
    def handle(payload):
        request = request_directory(payload['request_dir'])
        input_json = request / 'input.json'
        data = json.loads(input_json.read_text())
        if not isinstance(data, list) or len(data) != 1 or not data[0].get('modelSeeds'):
            raise ValueError('Mol-1 requires one explicit-seed job')
        name = data[0].get('name')
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid Mol-1 job name')
        out = request / 'output'; out.mkdir()
        runner.configs.dump_dir = str(out)
        runner.dump_dir = str(out); runner.error_dir = str(out / 'ERR')
        Path(runner.error_dir).mkdir()
        runner.dumper.base_dir = str(out)
        with contextlib.redirect_stdout(sys.stderr):
            prepare_start = time.perf_counter()
            runner.configs.input_json_path = batch.preprocess_input(
                str(input_json), out_dir=str(out), use_msa=options['use_msa'],
                use_template=False, use_rna_msa=False)
            stage_event('cpu_prepare', duration=time.perf_counter() - prepare_start, identity=actual)
            # Native infer_predict includes seeding, data loading, inference,
            # dumping and CUDA cache cleanup. Its admitted wall time is a
            # composite scope, never a GPU kernel-duration measurement.
            with operation(actual, phase='device') as lease:
                infer_predict(runner, runner.configs)
                lease.update_lifecycle(**lifecycle())
        for seed in data[0]['modelSeeds']:
            directory = out / name / f'seed_{int(seed)}' / 'predictions'
            summaries = sorted(directory.glob(f'{name}_summary_confidence_sample_*.json'))
            if len(summaries) != options['num_samples'] or any(not p.with_name(p.name.replace('_summary_confidence', '')).with_suffix('.cif').is_file() for p in summaries):
                raise RuntimeError(f'Mol-1 incomplete outputs for {name} seed {seed}')
            if options['need_atom_confidence'] and any(not p.with_name(p.name.replace('_summary_confidence', '_full_data')).is_file() for p in summaries):
                raise RuntimeError(f'Mol-1 missing full confidence for {name} seed {seed}')
        if (out / 'ERR').exists() and any((out / 'ERR').iterdir()):
            raise RuntimeError('Mol-1 reported inference errors')
        from astrolabe.evaluators.lifecycle import validate_output_directory
        validate_output_directory(out)
        return {'request': request.name}
    from astrolabe.evaluators.lifecycle import model_lifecycle
    def lifecycle():
        return model_lifecycle(runner.model, load_count=checkpoint_loads.count, loaded=checkpoint_loads.count > 0,
                               device=spec['device'], components={'checkpoint_load_failures': checkpoint_loads.failures})
    serve(handle, actual, load_count=checkpoint_loads.count, lifecycle=lifecycle,
          capabilities={'managed_device': True, 'managed_methods': ['run']})


if __name__ == '__main__': main()
