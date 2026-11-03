"""RhoFold model initialization, device inference and optional OpenMM owner."""
import contextlib
import json
import logging
import os
from pathlib import Path
import sys


def main():
    fd = os.dup(1); os.dup2(2, 1); sys.stdout = os.fdopen(fd, 'w', buffering=1)
    spec = json.loads(Path(sys.argv[1]).read_text())
    from astrolabe.evaluators.worker_identity import identity
    from astrolabe.evaluators.request_path import request_directory
    from astrolabe.evaluators.tree_codec import dump_tree, load_tree
    from astrolabe.evaluators.torch_rng import capture, restore
    from astrolabe.evaluators.lifecycle import model_lifecycle
    from astrolabe.runtime.server import serve
    from astrolabe.runtime.admission import operation
    actual = identity(sys.executable, spec['code_paths'], spec['weight_paths'], spec['device'], spec['settings'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['device']['visible']:
        raise RuntimeError('RhoFold visible device mismatch')
    config = spec['settings']['config']
    device = spec['device']['name']
    boot = request_directory(spec['bootstrap'])
    loads = {'rhofold_checkpoint': 0, 'hub_checkpoints': 0}
    relaxation_binding = {}
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import rhofold
        if str(Path(rhofold.__file__).resolve().parent) != spec['source']:
            raise RuntimeError('RhoFold imported unexpected source')
        from rhofold.config import rhofold_config
        from rhofold.rhofold import RhoFold
        restore(load_tree(boot / 'initial'), device)
        original_hub_loader = torch.hub.load_state_dict_from_url
        def no_download(*args, **kwargs):
            raise RuntimeError('RhoFold worker requires local RNA-FM weights')
        torch.hub.download_url_to_file = no_download
        def hub_load(*args, **kwargs):
            value = original_hub_loader(*args, **kwargs)
            loads['hub_checkpoints'] += 1
            return value
        torch.hub.load_state_dict_from_url = hub_load
        model = RhoFold(rhofold_config)
        state = torch.load(spec['weight_paths'][0], map_location=torch.device('cpu'))
        model.load_state_dict(state['model'])
        loads['rhofold_checkpoint'] += 1
        del state
        model.eval()
        model.to(device)
        dump_tree(boot / 'ready', capture(device))

    def handle(payload):
        request = request_directory(payload['request_dir'])
        method = payload['method']
        if method not in ('forward', 'relax'):
            raise ValueError('unsupported RhoFold method')
        batch = load_tree(request / ('input' if method == 'forward' else 'rng'))
        with contextlib.redirect_stdout(sys.stderr), operation(actual, phase='device') as lease:
            if method == 'forward':
                restore(batch['rng'], device)
                output = {}
                with torch.no_grad():
                    for name, feat in batch['features'].items():
                        value = model(tokens=feat['tokens'].to(device),
                                      rna_fm_tokens=feat['rna_fm_tokens'].to(device), seq=feat['seq'])[-1]
                        # Preserve the original location and order of sigmoid /
                        # softmax. Only already-computed values cross to CPU.
                        output[name] = {
                            'ss_prob_map': torch.sigmoid(value['ss'][0, 0]).detach().cpu().numpy(),
                            'dist_n': torch.softmax(value['n'].squeeze(0), dim=0).detach().cpu().numpy(),
                            'dist_p': torch.softmax(value['p'].squeeze(0), dim=0).detach().cpu().numpy(),
                            'dist_c': torch.softmax(value['c4_'].squeeze(0), dim=0).detach().cpu().numpy(),
                            'plddt_cpu': value['plddt'][0].detach().cpu().numpy(),
                            'cords_cpu': value['cord_tns_pred'][-1].squeeze(0).detach().cpu().numpy(),
                        }
                        del value
                result = {'output': output, 'rng': capture(device)}
            elif method == 'relax':
                from rhofold.relax import relax as relax_module
                restore(batch, device)
                relaxer = relax_module.AmberRelaxation(max_iterations=int(config['relax_steps']), logger=logging.getLogger('rhofold'))
                platform_name = config.get('openmm_platform')
                if platform_name:
                    try:
                        from openmm import Platform
                    except ImportError:
                        from simtk.openmm import Platform
                    platform = Platform.getPlatformByName(platform_name)
                    properties = dict(config.get('openmm_properties') or {})
                    if spec['shared'] and platform_name == 'CPU':
                        from astrolabe.runtime.admission import available_cpus
                        # This bound affects scheduling only; never change
                        # relaxation iterations, precision, forces or seeds.
                        maximum = min(available_cpus(), int(os.environ.get('OMP_NUM_THREADS', '1')))
                        threads = int(properties.get('Threads', maximum))
                        if threads < 1 or threads > maximum:
                            raise ValueError('OpenMM CPU threads exceed the admitted worker allocation')
                        properties['Threads'] = str(threads)
                    if platform_name == 'CUDA' and spec['shared']:
                        if properties.get('DeviceIndex', '0') != '0':
                            raise ValueError('CUDA relaxation must use the single visible allocated GPU')
                        properties['DeviceIndex'] = '0'
                    if platform_name == 'OpenCL' and spec['shared']:
                        from astrolabe.evaluators.openmm_binding import verify_opencl_binding
                        relaxation_binding.update(verify_opencl_binding(properties, spec['device']['visible']))
                    for key, value in properties.items():
                        platform.setPropertyDefaultValue(str(key), str(value))
                    relaxer.platform = platform
                # Modeller.addHydrogens otherwise selects an independent
                # fastest platform, even when Simulation is explicitly bound.
                hydrogen_method = relax_module.Modeller.addHydrogens
                if platform_name:
                    def bound_hydrogens(modeller, *args, **kwargs):
                        kwargs.setdefault('platform', relaxer.platform)
                        return hydrogen_method(modeller, *args, **kwargs)
                    relax_module.Modeller.addHydrogens = bound_hydrogens
                try:
                    relaxer.process(str(request / 'input.pdb'), str(request / 'relaxed.pdb'))
                finally:
                    relax_module.Modeller.addHydrogens = hydrogen_method
                result = {'rng': capture(device)}
            else:
                raise ValueError('unsupported RhoFold method')
            lease.update_lifecycle(**lifecycle())
        dump_tree(request / 'output', result)
        return {'request': request.name}

    def lifecycle():
        return model_lifecycle(model, load_count=sum(loads.values()), loaded=loads['rhofold_checkpoint'] > 0,
                               device=spec['device'], components={**loads, 'relaxation_binding': dict(relaxation_binding)})
    try:
        serve(handle, actual, load_count=sum(loads.values()), lifecycle=lifecycle,
              capabilities={'managed_device': True, 'managed_methods': ['forward', 'relax']})
    finally:
        # Framework allocators may retain pages; process exit is the release
        # boundary recorded by the broker, not this Python reference deletion.
        del model


if __name__ == '__main__':
    main()
