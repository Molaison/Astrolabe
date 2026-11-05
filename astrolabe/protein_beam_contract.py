"""Scientific identity and worker lifetime protections for a local protein beam."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def content_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def native_artifact_manifest(record):
    """Bind the native structure and proposal snapshot to their actual bytes."""
    result = {}
    for name in ("snapshot", "structure_path"):
        path = Path(record[name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Native {name} is unavailable: {path}")
        result[name] = dict(path=str(path), sha256=content_sha256(path))
    return result


def verify_native_artifacts(artifacts):
    for name, specification in artifacts.items():
        path = Path(specification["path"])
        if not path.is_file() or content_sha256(path) != specification["sha256"]:
            raise ValueError(f"Native {name} content changed after completion")


def scientific_contract(cfg):
    from omegaconf import OmegaConf
    source=Path(__file__).resolve().parent
    sources=[source.parent/'run.py',source/'interface.py',source/'protein_beam_calibration.py',source/'protein_beam.py',source/'protein_beam_inputs.py',source/'protein_beam_contract.py',
             source/'losses/interface_holes.py',source/'losses/interaction.py',
             source/'protein_beam_policy.py',source/'protein_beam_geometry.py',source/'protein_beam_seeds.py',
             source/'protein_beam_budget.py',
             source/'inference_pipeline.py',source/'inference_backends.py',
             source/'evaluators/af3_lifecycle.py',source/'evaluators/af3_pool.py',source/'evaluators/af3_pipeline.py',source/'evaluators/af3_service.py',
             source/'evaluators/inference_service.py',source/'evaluators/wrappers/alphafold3.py']
    sources += [source/'mutators/ligandmpnn.py',source/'mutators/utils.py',source/'mutators/base.py']
    sources += [source/'mutators/wrappers/ligandmpnn_run.py',source/'mutators/wrappers/ligandmpnn_utils.py',source/'protein_beam_root_parity.py']
    sources += [source/'app.py', source/'design/coordinator.py', source/'design/composition.py',
                source/'design/control_journal.py', source/'design/branch_types.py',
                source/'design/branching.py', source/'design/branch_execution.py',
                source/'design/branch_owner.py', source/'design/quotas.py', source/'design/io.py']
    data={}
    for spec in (cfg.get('homologies') or {}).values():
        if isinstance(spec,str):
            filename=spec.rsplit(':',1)[0] if '.json:' in spec else spec
            p=Path(filename)
            if p.is_file():data[str(p.resolve())]=content_sha256(p)
    raw=cfg.get('case_input') or {}
    if raw.get('structure'):
        p=Path(raw['structure']);data[str(p.resolve())]=content_sha256(p)
    if raw.get('root_structure'):
        p=Path(raw['root_structure']);data[str(p.resolve())]=content_sha256(p)
    parity=cfg.protein_beam.get('root_parity_reference')
    if parity:
        p=Path(str(parity));data[str(p.resolve())]=content_sha256(p)
        root=json.loads(p.read_text());structure=Path(root['structure_path']);label=structure.name.removesuffix('_model.cif')
        confidence=structure.parent.parent/'raw_model_results/alphafold3'/label/(label+'_confidences.json')
        for p in (structure,confidence):data[str(p.resolve())]=content_sha256(p)
    initial=cfg.protein_beam.get('initial_state')
    if initial:
        p=Path(str(initial));data[str(p.resolve())]=content_sha256(p)
        previous=json.loads(p.read_text())['records'][0]
        for key in ('snapshot','structure_path'):
            p=Path(previous[key]);data[str(p.resolve())]=content_sha256(p)
        audit=cfg.protein_beam.get('initial_state_audit')
        if audit:
            p=Path(str(audit));data[str(p.resolve())]=content_sha256(p)
    # Read-only weight assets are pinned by path/stat without rehashing GBs per beam.
    assets=[]
    paths=[Path(str(cfg.evaluator.alphafold3.model_dir))/'af3.bin']
    alias=cfg.stages.default.mutators[0]
    term=cfg.mutator.mutator_terms[alias]
    base=cfg.mutator.LigandMPNN
    model=str(term.get('model_type',base.model_type))
    weight_key={'ligand_mpnn':'checkpoint_ligand_mpnn','protein_mpnn':'checkpoint_protein_mpnn',
                'soluble_mpnn':'checkpoint_soluble_mpnn'}.get(model)
    if weight_key:
        paths.append(Path(str(term.get(weight_key,base.get(weight_key)))))
    for p in paths:
        if not p.is_file():raise FileNotFoundError(f'Model asset unavailable: {p}')
        info=p.stat()
        assets.append(dict(path=str(p.resolve()),size=info.st_size,mtime_ns=info.st_mtime_ns))
    contract=dict(config_yaml=OmegaConf.to_yaml(cfg,resolve=False),data_sha256=data,
                  source_sha256={p.name:content_sha256(p) for p in sources},
                  readonly_weight_asset_stat=assets)
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    return contract,digest


from astrolabe.process_lifetime import protect_worker_lifetime
