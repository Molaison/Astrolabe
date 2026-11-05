"""Verify the shared native root before any MPNN proposal can use its context."""
import hashlib
import json
import math
import shlex
from pathlib import Path

def _atoms(path):
    columns=[];rows=[];active=False
    for line in Path(path).read_text().splitlines():
        if line.startswith('_atom_site.'):
            columns.append(line.strip().split('.',1)[1]);active=True;continue
        if not active:continue
        if not line.strip() or line.startswith('#'):
            if rows:break
            continue
        if line.startswith('_') or line=='loop_':break
        fields=shlex.split(line)
        if len(fields)!=len(columns):raise ValueError('Invalid native CIF atom table')
        atom=dict(zip(columns,fields));xyz=[float(atom[k]) for k in ['Cartn_x','Cartn_y','Cartn_z']]
        if not all(math.isfinite(x) for x in xyz):raise ValueError('Nonfinite root coordinate')
        rows.append([atom['label_asym_id'],int(atom['label_seq_id']),atom['label_comp_id'],atom['label_atom_id'],*xyz])
    if not rows:raise ValueError('Empty root atom table')
    return rows

def _confidence(record):
    path=Path(record['structure_path']);label=path.name.removesuffix('_model.cif')
    value=json.loads((path.parent.parent/'raw_model_results/alphafold3'/label/(label+'_confidences.json')).read_text())
    matrix=value['pae']
    residues = {(a[0], a[1]) for a in _atoms(path) if a[3] == 'CA'}
    expected = len(residues)
    if not expected or len(matrix)!=expected or any(len(row)!=expected or any(not math.isfinite(x) for x in row) for row in matrix):
        raise ValueError(f'Root requires full finite {expected}x{expected} PAE')
    return value

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def root_parity(record,reference_path):
    old=json.loads(Path(reference_path).read_text())
    fields=['sequence','seed','metrics','holes']
    different=[key for key in fields if record[key]!=old[key]]
    current_atoms=_digest(_atoms(record['structure_path']));reference_atoms=_digest(_atoms(old['structure_path']))
    current_conf=_digest(_confidence(record));reference_conf=_digest(_confidence(old))
    if current_atoms!=reference_atoms:different.append('all_atom_coordinates')
    if current_conf!=reference_conf:different.append('full_native_confidence')
    return dict(passed=not different,different=different,reference=str(reference_path),current_structure=record['structure_path'],
                current_atoms_sha256=current_atoms,reference_atoms_sha256=reference_atoms,
                current_confidence_sha256=current_conf,reference_confidence_sha256=reference_conf,
                boundary='Checked before first MPNN proposal and rechecked on resume; no numerical tolerance')
