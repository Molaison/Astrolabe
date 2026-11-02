"""Protenix trajectory worker: initialize once, predict with explicit RNG."""
import contextlib
import json
import os
from pathlib import Path
import sys


def main():
    fd = os.dup(1); os.dup2(2, 1); sys.stdout = os.fdopen(fd, 'w', buffering=1)
    spec = json.loads(Path(sys.argv[1]).read_text())
    from astrolabe.evaluators.worker_identity import identity
    from astrolabe.evaluators.tree_codec import load_tree, dump_tree
    from astrolabe.evaluators.torch_rng import capture, restore
    from astrolabe.evaluators.request_path import request_directory
    from astrolabe.runtime.server import serve
    if spec['semantics'].get('config') != spec['model_config'] or not spec['semantics'].get('trajectory'):
        raise ValueError('Protenix startup identity mismatch')
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], spec['semantics'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('Protenix visible device mismatch')
    source = Path(spec['source_dir']).resolve(strict=True)
    if str(source / 'protenix') not in spec['code_paths'] or str(source / 'configs') not in spec['code_paths']:
        raise RuntimeError('Protenix source identity mismatch')
    raw = spec['model_config']
    raw_checkpoint = raw.get('load_checkpoint_path')
    if not raw_checkpoint and raw.get('load_checkpoint_dir') and raw.get('model_name'):
        raw_checkpoint = str(Path(raw['load_checkpoint_dir']) / (raw['model_name'] + '.pt'))
    if not raw_checkpoint or str(Path(raw_checkpoint).expanduser().resolve(strict=True)) != spec['weight_paths'][0]:
        raise RuntimeError('Protenix requested checkpoint identity mismatch')
    boot = request_directory(spec['bootstrap'])
    with contextlib.redirect_stdout(sys.stderr):
        from omegaconf import OmegaConf
        from astrolabe.evaluators.wrappers.protenix import ProtenixRunner, configure_protenix_source
        configure_protenix_source(source)
        import protenix
        if not Path(protenix.__file__).resolve().is_relative_to(source / 'protenix'):
            raise RuntimeError('Protenix imported unexpected source')
        restore(load_tree(boot / 'initial'), spec['device']['name'])
        runner = ProtenixRunner(OmegaConf.create(spec['model_config']))
        if getattr(runner.configs, 'model_name', None) != spec['semantics']['effective_model']:
            raise RuntimeError('Protenix effective model identity mismatch')
        actual_checkpoint = getattr(runner.configs, 'load_checkpoint_path', None)
        if not actual_checkpoint and getattr(runner.configs, 'load_checkpoint_dir', None):
            actual_checkpoint = Path(runner.configs.load_checkpoint_dir) / (runner.configs.model_name + '.pt')
        if not actual_checkpoint or str(Path(actual_checkpoint).expanduser().resolve()) != spec['weight_paths'][0]:
            raise RuntimeError('Protenix effective checkpoint identity mismatch')
        if str(runner.device) != spec['device']['name']:
            raise RuntimeError('Protenix actual device mismatch')
        if str(runner.device).startswith('cuda'):
            import torch
            if torch.cuda.get_device_name(runner.device) != spec['device']['device_kind']:
                raise RuntimeError('Protenix actual GPU mismatch')
        dump_tree(boot / 'ready', capture(spec['device']['name']))
    def handle(payload):
        request = request_directory(payload['request_dir'])
        batch = load_tree(request / 'input')
        data = batch['data']
        if int(data['N_token'].item()) != batch['n_token'] or runner.configs.seed != batch['seed']:
            raise ValueError('Protenix request settings mismatch')
        with contextlib.redirect_stdout(sys.stderr):
            from astrolabe.evaluators.wrappers.protenix import update_inference_configs
            restore(batch['rng'], spec['device']['name'])
            updated = update_inference_configs(runner.configs, batch['n_token'])
            flags = {'confidence_head': bool(updated.skip_amp.confidence_head),
                     'sample_diffusion': bool(updated.skip_amp.sample_diffusion)}
            if flags != batch['skip_amp']: raise ValueError('Protenix dynamic configuration mismatch')
            runner.update_model_configs(updated)
            prediction = runner.predict(data)
            output = {'prediction': prediction, 'rng': capture(spec['device']['name'])}
        dump_tree(request / 'output', output)
        return {'request': request.name}
    serve(handle, actual)


if __name__ == '__main__': main()
