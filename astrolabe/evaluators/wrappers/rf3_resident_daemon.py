"""RF3 common daemon; preserve native post-initialization RNG replay."""
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    fd = os.dup(1); os.dup2(2, 1); sys.stdout = os.fdopen(fd, 'w', buffering=1)
    spec = json.loads(Path(sys.argv[1]).read_text())
    from astrolabe.evaluators.worker_identity import identity
    from astrolabe.evaluators.request_path import request_directory
    from astrolabe.evaluators.tree_codec import load_tree, dump_tree
    from astrolabe.evaluators.torch_rng import capture, restore, apply_settings
    from astrolabe.evaluators.lifecycle import model_lifecycle, validate_output_directory
    from astrolabe.evaluators.wrappers.rf3 import _load_rf3_inference_engine, _accepts_keyword
    from astrolabe.runtime.server import serve
    from astrolabe.runtime.admission import operation
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], spec['settings'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('RF3 visible device mismatch')
    device = spec['device']['name']
    boot = request_directory(spec['bootstrap'])
    loads = {'checkpoint_deserializations': 0, 'engine_initializations': 0}
    with contextlib.redirect_stdout(sys.stderr):
        if spec['settings']['exchange_owner_rng']:
            restore(load_tree(boot / 'initial'), device)
        engine_cls = _load_rf3_inference_engine(spec['source'])
        base = importlib.import_module('rf3._core.inference_engines.base')
        checkpoint_loader = base.load_checkpoint
        def load_checkpoint(*args, **kwargs):
            value = checkpoint_loader(*args, **kwargs)
            loads['checkpoint_deserializations'] += 1
            return value
        base.load_checkpoint = load_checkpoint
        engine = engine_cls(**spec['settings']['config'])
        engine.initialize()
        loads['engine_initializations'] += 1
        model = engine.trainer.state['model']
        if not engine.initialized_ or str(Path(engine.ckpt_path).resolve()) != spec['weight_paths'][0]:
            raise RuntimeError('RF3 initialization or checkpoint identity mismatch')
        state = model_lifecycle(model, load_count=loads['checkpoint_deserializations'], loaded=True, device=spec['device'])
        if state['devices'] and any(not name.startswith(device.split(':')[0]) for name in state['devices']):
            raise RuntimeError('RF3 model device differs from admitted device')
        post_init_rng = capture(device)
        # Native RF3 resets its RNG per request, but it does not reset process
        # precision/determinism settings to their post-construction values.
        replay_rng = {key: value for key, value in post_init_rng.items() if key != 'settings'}
        if spec['settings']['exchange_owner_rng']:
            dump_tree(boot / 'ready', post_init_rng)

    def handle(payload):
        directory = request_directory(payload['request_dir'])
        output = directory / 'output'; output.mkdir()
        kwargs = {'inputs': directory / 'input.json', 'out_dir': output,
                  'annotate_b_factor_with_plddt': True,
                  'template_selection': list(payload['template_selection']),
                  'ground_truth_conformer_selection': list(payload['ground_truth_conformer_selection'])}
        if _accepts_keyword(engine.run, 'add_missing_atoms'):
            kwargs['add_missing_atoms'] = bool(payload['add_missing_atoms'])
        with contextlib.redirect_stdout(sys.stderr):
            # Astrolabe's RF3 direct adapter already replays this snapshot for
            # every request. Advancing a trajectory RNG here would change it.
            with operation(actual, phase='device') as lease:
                restore(replay_rng, device)
                if spec['settings']['exchange_owner_rng']:
                    apply_settings(payload['torch_settings'])
                engine.run(**kwargs)
                final_rng = capture(device) if spec['settings']['exchange_owner_rng'] else None
                lease.update_lifecycle(**lifecycle())
            if spec['settings']['exchange_owner_rng']:
                dump_tree(directory / 'output_rng', final_rng)
        validate_output_directory(directory)
        return {'request': directory.name}

    def lifecycle():
        return model_lifecycle(engine.trainer.state['model'], load_count=loads['checkpoint_deserializations'],
                               loaded=bool(engine.initialized_), device=spec['device'], components=dict(loads))
    try:
        serve(handle, actual, load_count=loads['checkpoint_deserializations'], lifecycle=lifecycle,
              capabilities={'managed_device': True, 'managed_methods': ['run']})
    finally:
        engine.trainer = None
        engine.pipeline = None
        engine.initialized_ = False


if __name__ == '__main__':
    main()
