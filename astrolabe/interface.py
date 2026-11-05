"""Native, data-only input for bounded protein-interface beam optimization."""
from __future__ import annotations

import copy
from contextlib import nullcontext, contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace


FORMATS = {'.pdb', '.cif', '.mmcif'}

# Reserve native result paths, not the entire user-selected directory. Shells
# create redirected logs before the CLI starts, including for nohup runs.
_RUN_OUTPUT_PATHS = frozenset({
    'interface_plan.json', 'interface_results.json',
    'interface_plan.json.tmp', 'interface_results.json.tmp', 'prepared', 'cases', 'services',
    'beam_state.json', 'final_beam.json', 'pareto_archive.json', 'af3_calls.json',
    'scores.csv', 'timeline.jsonl', 'candidates', 'workers', 'raw_model_results',
    'owner.lock', 'root_calibration.json', 'root_parity.json',
})


def _sha(path):
    from astrolabe.protein_beam_contract import content_sha256
    return content_sha256(path)


def _beam_state(output):
    checkpoint = Path(output) / 'beam_state.json'
    return json.loads(checkpoint.read_text()) if checkpoint.is_file() else None


def _state_complete(state):
    return state is not None and state.get('status') == 'complete'


def _summary(case, state):
    return dict(case=case['name'], status=state['status'], evaluations=len(state['records']),
                budget_complete=state['budget_complete'], stop_reason=state.get('stop_reason'))


def _read_structure(path):
    import numpy as np
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1

    parser = MMCIFParser(QUIET=True) if path.suffix.lower() != '.pdb' else PDBParser(QUIET=True, PERMISSIVE=False)
    models = list(parser.get_structure('input', str(path)).get_models())
    if len(models) != 1:
        raise ValueError(f'{path.name}: supply exactly one structural model')
    sequences = {}
    for chain in models[0]:
        residues = list(chain)
        if not residues or any(not is_aa(r, standard=True) or r.id[0] != ' ' for r in residues):
            raise ValueError(f'{path.name}: only two standard protein chains are supported; unsupported residue in {chain.id}')
        for residue in residues:
            if residue.is_disordered() or any(a.is_disordered() for a in residue):
                raise ValueError(f'{path.name}: resolve alternate conformations before optimization')
            if any(a not in residue for a in ('N', 'CA', 'C', 'O')):
                raise ValueError(f'{path.name}: incomplete backbone in chain {chain.id}')
            if any(not np.isfinite(a.coord).all() for a in residue):
                raise ValueError(f'{path.name}: nonfinite atom coordinates')
        sequences[chain.id] = ''.join(seq1(r.resname) for r in residues)
    if len(sequences) != 2:
        raise ValueError(f'{path.name}: expected exactly two protein chains')
    return models[0], sequences


def _homology_payload(specification):
    value = str(specification)
    filename, separator, chain = value.rpartition(':')
    if not separator or not filename.lower().endswith('.json'):
        filename, chain = value, None
    path = Path(filename).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text())
    if payload.get('bondedAtomPairs') or payload.get('userCCD'):
        raise ValueError('Interface task cannot omit custom AF3 bonds or userCCD')
    proteins = []
    for entry in payload.get('sequences', []):
        raw = entry.get('protein')
        if raw is None:
            continue
        ids = raw.get('id', [])
        ids = [ids] if isinstance(ids, str) else ids
        for identifier in ids:
            if chain is None or identifier == chain:
                proteins.append((identifier, raw))
    if not proteins:
        raise ValueError('No selected protein in target homology JSON')
    return path, proteins


def inspect_case(path, homology_path, proteins, target_chain=None, binder_chain=None):
    """Validate identity and homology for every case before issuing any AF3 call."""
    import numpy as np
    from astrolabe.protein_beam_inputs import preserve_fixed_target_homology
    model, sequences = _read_structure(path)
    if target_chain is not None and target_chain not in sequences:
        raise ValueError(f'{path.name}: target chain {target_chain!r} is absent')
    if binder_chain is not None and binder_chain not in sequences:
        raise ValueError(f'{path.name}: binder chain {binder_chain!r} is absent')
    if target_chain is not None and target_chain == binder_chain:
        raise ValueError('Target and binder must be different chains')
    matches = [(c, identifier, raw) for c, seq in sequences.items()
               for identifier, raw in proteins if raw.get('sequence') == seq
               and (target_chain is None or c == target_chain)
               and (binder_chain is None or c != binder_chain)]
    # Prefer the longer chain only when this yields a unique homology match.
    if len(matches) > 1 and target_chain is None:
        longest = max(len(sequences[c]) for c in sequences)
        matches = [m for m in matches if len(sequences[m[0]]) == longest]
    if len(matches) != 1:
        raise ValueError(f'{path.name}: target/homology match is ambiguous or absent; specify --target-chain and JSON:chain')
    target, homology_chain, raw = matches[0]
    binder = next(c for c in sequences if c != target)
    if any(raw.get(k) for k in ('unpairedMsaPath', 'pairedMsaPath')):
        raise ValueError('Supply inline target MSA in AF3 JSON, rather than unresolved MSA paths')
    from scipy.spatial import cKDTree
    heavy = lambda c: [a.coord for r in model[c] for a in r if a.element not in {'H', 'D'}]
    if not any(cKDTree(heavy(binder)).query_ball_tree(cKDTree(heavy(target)), 10.0)):
        raise ValueError(f'{path.name}: no original protein interface within 10 Angstrom')
    specification = f'{homology_path}:{homology_chain}'
    protomer = SimpleNamespace(sequence=sequences[target], length=len(sequences[target]), mutable_mask=np.zeros(len(sequences[target]), dtype=bool))
    preserve_fixed_target_homology(SimpleNamespace(homologies={'B': specification}), protomer)
    return dict(input=str(path), input_sha256=_sha(path), binder_chain=binder, target_chain=target,
                binder_sequence=sequences[binder], target_sequence=sequences[target], homology=specification), model


def _normalized_text(model, binder, target):
    """Canonical A/B chains and contiguous residues, retaining parsed coordinates."""
    from Bio.PDB import MMCIFIO, Structure, Model, Chain
    structure = Structure.Structure('interface')
    canonical = Model.Model(0)
    structure.add(canonical)
    for original, name in ((binder, 'A'), (target, 'B')):
        chain = Chain.Chain(name)
        canonical.add(chain)
        for i, residue in enumerate(model[original], 1):
            residue = copy.deepcopy(residue, {id(residue.get_parent()): None})
            residue.detach_parent()
            residue.id = (' ', i, ' ')
            chain.add(residue)
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(io.StringIO())
    # Bio.PDB's default serializer rounds coordinates to three decimals. Keep
    # enough digits to round-trip its parsed float32 coordinates for CIF inputs.
    data = writer.dic
    atoms = list(structure.get_atoms())
    for i, axis in enumerate('xyz'):
        data[f'_atom_site.Cartn_{axis}'] = [format(float(a.coord[i]), '.9g') for a in atoms]
    writer.set_dict(data)
    handle = io.StringIO()
    writer.save(handle)
    return handle.getvalue()


def _prepare(input_path, target_homology, output, target_chain, binder_chain, budget=None):
    from omegaconf import OmegaConf, open_dict
    from astrolabe.protein_beam import load_config, validate_config
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError('Prediction budget must be a positive integer, including the root')
    path = Path(input_path).expanduser().resolve(strict=True)
    files = sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in FORMATS), key=lambda p: p.name) if path.is_dir() else [path]
    if not files or any(p.suffix.lower() not in FORMATS for p in files):
        raise ValueError('Input must be a PDB/CIF file or a directory containing PDB/CIF files')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', p.stem) for p in files):
        raise ValueError('Input filenames must have simple letters, digits, dots, dashes or underscores')
    if len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError('Input filenames have colliding case identifiers')
    homology_path, proteins = _homology_payload(target_homology)
    task_path = Path(__file__).resolve().parents[1] / 'config/interface.yaml'
    cases, artifacts = [], {}
    for file in files:
        info, model = inspect_case(file, homology_path, proteins, target_chain, binder_chain)
        name = file.stem
        relative = f'prepared/{name}'
        text = _normalized_text(model, info['binder_chain'], info['target_chain'])
        structure = output / relative / 'input.cif'
        cfg = load_config(task_path)
        with open_dict(cfg):
            cfg.task_name = name + '_interface'
            cfg.out_dir = str(output / 'cases' / name)
            cfg.seqs = {'A': info['binder_sequence'], 'B': '+' + info['target_sequence']}
            cfg.homologies = {'B': info['homology']}
            cfg.case_input.structure = str(structure)
            if budget is not None:
                cfg.protein_beam.max_evaluations = budget
                # An explicit budget drives the search beyond the default
                # single generation, retaining barriers and feasibility gates.
                cfg.protein_beam.max_generations = None
                cfg.protein_beam.af3_call_budget.max_total = budget
                cfg.protein_beam.af3_call_budget.max_per_case = budget
        validate_config(cfg)
        config = OmegaConf.to_yaml(cfg, resolve=False)
        artifacts[f'{relative}/input.cif'] = text
        artifacts[f'{relative}/config.yaml'] = config
        cases.append(dict(name=name, **info, config=f'{relative}/config.yaml', output=f'cases/{name}'))
    # Pin the entrypoint, all Python implementation and composed task defaults.
    # Hashes are cheap on these source files and do not read model weights.
    source = Path(__file__).resolve().parent
    implementation = {str(p.relative_to(source)): _sha(p) for p in sorted(source.rglob('*.py'))}
    implementation['../run.py'] = _sha(source.parent / 'run.py')
    plan = dict(schema=1, input=str(path), homology=str(homology_path), homology_sha256=_sha(homology_path),
                implementation=implementation, cases=cases,
                artifacts={k: hashlib.sha256(v.encode()).hexdigest() for k, v in artifacts.items()})
    return plan, artifacts


def _run_interface_cpu(*, input_path, target_homology, output_root, gpu_environment, socket_path=None,
                       target_chain=None, binder_chain=None, resume=False,
                       retry_interrupted=False, prepare_only=False, budget=None):
    """Prepare validated data inputs and run independent native beams in order."""
    from astrolabe.protein_beam import atomic_json, load_config, run_beam
    if retry_interrupted and not resume:
        raise ValueError('--retry-interrupted requires --resume')
    output = Path(output_root).expanduser().resolve()
    plan, artifacts = _prepare(input_path, target_homology, output, target_chain, binder_chain, budget)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'interface.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / 'interface_plan.json'
        if manifest.exists():
            if not resume:
                raise FileExistsError('Interface output exists; use --resume to preserve completed work')
            existing_plan = json.loads(manifest.read_text())
            if existing_plan != plan:
                # This preliminary plan comparison retains input/config artifact
                # checks. The completed fastpath below still enforces the exact
                # current scientific contract, including every registered source
                # hash, before returning without model work.
                complete = all(_state_complete(_beam_state(output / case['output']))
                               for case in existing_plan.get('cases', []))
                without_implementation = lambda value: {
                    key: item for key, item in value.items() if key != 'implementation'
                }
                if not complete or without_implementation(existing_plan) != without_implementation(plan):
                    raise ValueError('Interface inputs, implementation or task configuration changed on resume')
            for relative, digest in plan['artifacts'].items():
                if not (output / relative).is_file() or _sha(output / relative) != digest:
                    raise ValueError(f'Prepared input/config changed or missing: {relative}')
        else:
            if resume:
                raise FileNotFoundError('Cannot resume without an interface plan')
            collisions = sorted(p.name for p in output.iterdir() if p.name in _RUN_OUTPUT_PATHS)
            if collisions:
                raise FileExistsError(
                    f'Output directory contains existing Astrolabe run paths: {", ".join(collisions)}. '
                    'Use another --out-dir, or --resume with the original directory of an existing interface run. '
                    'Existing files were preserved.')
            for relative, text in artifacts.items():
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text)
            atomic_json(manifest, plan)
        if prepare_only:
            print(json.dumps(dict(status='prepared', cases=len(plan['cases']), plan=str(manifest), new_af3_calls=0)))
            return 0
        configurations = [(case, load_config(output / case['config'])) for case in plan['cases']]
        states = [(case, cfg, _beam_state(output / case['output'])) for case, cfg in configurations]
        if resume:
            from astrolabe.protein_beam_recovery import validate_completed_beam
            for case, cfg, state in states:
                if _state_complete(state):
                    validate_completed_beam(
                        cfg,
                        output / case['output'],
                        state,
                    )
        if resume and all(_state_complete(state) for _, _, state in states):
            summaries = [_summary(case, state) for case, _, state in states]
            print(json.dumps(dict(status='complete', cases=summaries)))
            return 0
        if resume:
            from astrolabe.evaluators.af3_lifecycle import recover_owned_services
            recover_owned_services(output)
            for case, _, state in states:
                if _state_complete(state):
                    continue
                _check_interrupted_attempts(output / case['output'], retry_interrupted)
        unfinished_configs = [(case, cfg) for case, cfg, state in states if not _state_complete(state)]
        selected_socket = socket_path or unfinished_configs[0][1].evaluator.alphafold3.get('service_socket')
        unfinished = bool(unfinished_configs)
        if unfinished and selected_socket and not Path(selected_socket).is_socket():
            raise ValueError('The explicitly selected AF3 service socket is unavailable')
        if unfinished and not selected_socket:
            from astrolabe.evaluators.af3_lifecycle import OwnedAF3
            service = OwnedAF3(unfinished_configs[0][1], output, gpu_environment)
        else:
            # Reused services belong to their caller. Completed recovery requires
            # neither a model load nor any service connection.
            service = nullcontext(str(selected_socket or ''))
        summaries_by_case = {
            case['name']: _summary(case, state)
            for case, _, state in states if _state_complete(state)
        }
        with service as active_socket:
            for case, cfg in unfinished_configs:
                destination = output / case['output']
                state = run_beam(cfg, destination, active_socket,
                                 workers=int(cfg.protein_beam.workers), in_flight=int(cfg.protein_beam.in_flight),
                                 resume=resume, retry_interrupted=retry_interrupted,
                                 on_abort=getattr(service, 'abort', None))
                summaries_by_case[case['name']] = _summary(case, state)
                summaries = [summaries_by_case[item['name']] for item in plan['cases']
                             if item['name'] in summaries_by_case]
                atomic_json(output / 'interface_results.json', summaries)
        summaries = [summaries_by_case[item['name']] for item in plan['cases']]
        print(json.dumps(dict(status='complete', cases=summaries)))
        return 0


@contextmanager
def cpu_coordinator_environment():
    original = os.environ.copy()
    keys = ('CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'JAX_PLATFORM_NAME', 'GH_CUDA_PREINITIALIZE')
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', GH_CUDA_PREINITIALIZE='0')
    os.environ.pop('JAX_PLATFORM_NAME', None)
    try:
        yield original
    finally:
        for key in keys:
            if key in original:
                os.environ[key] = original[key]
            else:
                os.environ.pop(key, None)


def run_interface(*, input_path, target_homology, output_root, socket_path=None,
                  target_chain=None, binder_chain=None, resume=False,
                  retry_interrupted=False, prepare_only=False, budget=None):
    with cpu_coordinator_environment() as gpu_environment:
        return _run_interface_cpu(input_path=input_path, target_homology=target_homology,
                                  output_root=output_root, gpu_environment=gpu_environment,
                                  socket_path=socket_path, target_chain=target_chain,
                                  binder_chain=binder_chain, resume=resume,
                                  retry_interrupted=retry_interrupted, prepare_only=prepare_only,
                                  budget=budget)


def _check_interrupted_attempts(output, retry):
    checkpoint = Path(output) / 'beam_state.json'
    if not checkpoint.exists():
        return
    state = json.loads(checkpoint.read_text())
    for record in state['records']:
        if record.get('status') != 'running':
            continue
        terminal = Path(output) / 'candidates' / f"{record['id']:06d}" / 'result.json'
        if not terminal.exists():
            if not retry:
                raise RuntimeError('Interrupted inference lacks a terminal result; inspect owner then use explicit --retry-interrupted')
            previous = state.get('service_socket')
            if previous and Path(previous).exists():
                raise RuntimeError('Previous AF3 owner socket still exists; drain/verify that owner before retrying')
