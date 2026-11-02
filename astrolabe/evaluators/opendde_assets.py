"""Discover OpenDDE's exact local inference assets without downloading."""
import contextlib
import json
from pathlib import Path
import sys


def asset_paths(model_name, checkpoint, use_template):
    from opendde.config.inference import build_inference_config
    cfg = build_inference_config(model_name=model_name, fill_required_with_null=True)
    assets = [checkpoint, cfg.data.ccd_components_file, cfg.data.ccd_components_rdkit_mol_file]
    if use_template:
        assets += [cfg.data.template.obsolete_pdbs_path, cfg.data.template.release_dates_path]
    result = [str(Path(p).expanduser().resolve(strict=True)) for p in assets]
    if any(not Path(p).is_file() for p in result):
        raise ValueError('OpenDDE local model and CCD assets must be files')
    return result


if __name__ == '__main__':
    with contextlib.redirect_stdout(sys.stderr):
        assets = asset_paths(sys.argv[1], sys.argv[2], sys.argv[3] == '1')
    print(json.dumps({'assets': assets}))
