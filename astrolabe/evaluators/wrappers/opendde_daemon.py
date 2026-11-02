"""Resident OpenDDE handler using the pinned public Python runner API."""
import contextlib
import json
import os
from pathlib import Path
import sys


def main():
    fd = os.dup(1); os.dup2(2, 1); sys.stdout = os.fdopen(fd, 'w', buffering=1)
    spec = json.loads(Path(sys.argv[1]).read_text())
    from astrolabe.evaluators.worker_identity import identity
    from astrolabe.evaluators.request_path import request_directory
    from astrolabe.runtime.server import serve
    options = spec['options']
    if spec['semantics']['options'] != options or not spec['semantics']['trajectory']:
        raise ValueError('OpenDDE startup identity mismatch')
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], spec['semantics'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('OpenDDE visible device mismatch')
    source = Path(spec['source']).resolve(strict=True)
    if str(source / 'opendde') not in spec['code_paths'] or str(source / 'runner') not in spec['code_paths']:
        raise RuntimeError('OpenDDE source identity mismatch')
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import opendde
        from runner import batch_inference as batch
        from runner.inference import infer_predict
        if not Path(opendde.__file__).resolve().is_relative_to(source / 'opendde'):
            raise RuntimeError('OpenDDE imported unexpected source')
        if not Path(batch.__file__).resolve().is_relative_to(source / 'runner'):
            raise RuntimeError('OpenDDE imported unexpected runner')
        def verify_assets(cfg):
            if cfg.model_name != options['model_name'] or str(Path(cfg.load_checkpoint_path).expanduser().resolve()) != spec['weight_paths'][0]:
                raise RuntimeError('OpenDDE model identity changed')
            paths = [options['load_checkpoint_path'], cfg.data.ccd_components_file, cfg.data.ccd_components_rdkit_mol_file]
            if cfg.use_template:
                paths += [cfg.data.template.obsolete_pdbs_path, cfg.data.template.release_dates_path]
            paths = [str(Path(p).expanduser().resolve(strict=True)) for p in paths]
            if paths != spec['weight_paths'] or any(not Path(p).is_file() for p in paths):
                raise RuntimeError('OpenDDE model or CCD assets differ')
        # Replace only upstream auto-download with strict local verification.
        batch.download_inference_cache = verify_assets
        runner = batch.get_default_runner(
            seeds=None, dump_dir=spec['bootstrap_dir'], n_cycle=options['num_cycles'],
            n_step=options['num_steps'], n_sample=options['num_samples'], dtype=options['dtype'],
            model_name=options['model_name'], load_checkpoint_path=options['load_checkpoint_path'],
            use_msa=options['use_msa'], trimul_kernel=options['triangle_multiplicative'],
            triatt_kernel=options['triangle_attention'], enable_cache=options['enable_cache'],
            enable_fusion=options['enable_fusion'], enable_tf32=options['enable_tf32'],
            deterministic=options['deterministic'], use_template=options['use_template'],
            use_rna_msa=options['use_rna_msa'], need_atom_confidence=options['need_atom_confidence'],
            kalign_binary_path=options['kalign_binary_path'], use_tfg_guidance=options['use_tfg_guidance'],
            foldcp_mode=options['foldcp_mode'], foldcp_size_dp=options['foldcp_size_dp'],
            foldcp_size_cp=options['foldcp_size_cp'], foldcp_devices=options['foldcp_devices'],
            foldcp_metrics_jsonl=options['foldcp_metrics_jsonl'])
        if str(runner.device) != spec['device']['name']:
            raise RuntimeError('OpenDDE actual device mismatch')
        if str(runner.device).startswith('cuda') and torch.cuda.get_device_name(runner.device) != spec['device']['device_kind']:
            raise RuntimeError('OpenDDE actual GPU mismatch')
    def handle(payload):
        request = request_directory(payload['request_dir'])
        input_json = request / 'input.json'
        data = json.loads(input_json.read_text())
        if not isinstance(data, list) or len(data) != 1 or not data[0].get('modelSeeds'):
            raise ValueError('OpenDDE requires one explicit-seed job')
        name = data[0].get('name')
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid OpenDDE job name')
        out = request / 'output'; out.mkdir()
        runner.configs.dump_dir = str(out)
        runner.dump_dir = str(out); runner.error_dir = str(out / 'ERR')
        Path(runner.error_dir).mkdir()
        runner.dumper.base_dir = str(out)
        with contextlib.redirect_stdout(sys.stderr):
            runner.configs.input_json_path = batch.preprocess_input(
                str(input_json), out_dir=str(out), use_msa=options['use_msa'],
                use_template=options['use_template'], use_rna_msa=options['use_rna_msa'],
                hmmsearch_binary_path=options['hmmsearch_binary_path'],
                hmmbuild_binary_path=options['hmmbuild_binary_path'],
                seqres_database_path=options['seqres_database_path'],
                nhmmer_binary_path=options['nhmmer_binary_path'],
                hmmalign_binary_path=options['hmmalign_binary_path'],
                hmmbuild_rna_binary_path=options['hmmbuild_rna_binary_path'],
                ntrna_database_path=options['ntrna_database_path'],
                rfam_database_path=options['rfam_database_path'],
                rna_central_database_path=options['rna_central_database_path'],
                nhmmer_n_cpu=options['nhmmer_n_cpu'])
            infer_predict(runner, runner.configs)
        seeds = runner.configs.seeds or data[0]['modelSeeds']
        for seed in seeds:
            directory = out / name / f'seed_{int(seed)}' / 'predictions'
            summaries = sorted(directory.glob(f'{name}_summary_confidence_sample_*.json'))
            if not summaries or any(not p.with_name(p.name.replace('_summary_confidence', '')).with_suffix('.cif').is_file() for p in summaries):
                raise RuntimeError(f'OpenDDE missing outputs for {name} seed {seed}')
        if (out / 'ERR').exists() and any((out / 'ERR').iterdir()):
            raise RuntimeError('OpenDDE reported inference errors')
        return {'request': request.name}
    serve(handle, actual)


if __name__ == '__main__': main()
