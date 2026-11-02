"""Inspect Mol-1 pinned runtime asset paths without downloading."""
import contextlib
import json
from pathlib import Path
import sys


def asset_paths(model_name, checkpoint, use_template):
    from mol.config.base import configs as base
    from mol.config.data import data_configs
    from mol.config.inference import inference_configs
    from mol.config.parser import parse_configs
    cfg = parse_configs(configs={**base, 'data': data_configs, **inference_configs, 'model_name': model_name}, fill_required_with_null=True)
    paths = [checkpoint, cfg.data.ccd_components_file, cfg.data.ccd_components_rdkit_mol_file,
             cfg.data.pdb_cluster_file, cfg.data.obsolete_release_data_csv]
    if use_template:
        paths += [cfg.data.template.obsolete_pdbs_path, cfg.data.template.release_dates_path]
    paths = [str(Path(p).expanduser().resolve(strict=True)) for p in paths]
    if any(not Path(p).is_file() for p in paths): raise ValueError('Mol-1 assets must be local files')
    return paths


if __name__ == '__main__':
    with contextlib.redirect_stdout(sys.stderr):
        assets = asset_paths(sys.argv[1], sys.argv[2], sys.argv[3] == '1')
    print(json.dumps({'assets': assets}))
