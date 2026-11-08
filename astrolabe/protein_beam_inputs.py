"""Preserve precomputed homology for an entirely fixed target; no alignment edits."""
from __future__ import annotations
from pathlib import Path
import json


def preserve_fixed_target_homology(cfg, protomer):
    specification=cfg.homologies.get('B')
    if specification is None:
        return
    if protomer.mutable_mask.any():
        raise ValueError('Raw target homology preservation requires a fully fixed target')
    from astrolabe.core.folding_input import Template
    if isinstance(specification,str):
        filename,separator,chain=specification.rpartition(':')
        if not separator or not filename.endswith('.json'):
            raise ValueError('Use an AF3 JSON path:chain or an explicit homology mapping')
        from astrolabe.core.msa_utils import load_af3_chain_data
        payload=json.loads(Path(filename).read_text())
        if payload.get('bondedAtomPairs') or payload.get('userCCD'):
            raise ValueError('Standard-protein beam cannot silently omit custom AF3 bonds or userCCD')
        kind,raw=load_af3_chain_data(payload,chain)
        if kind!='protein':
            raise ValueError('Fixed target must be a protein')
        if raw['sequence']!=protomer.sequence:
            raise ValueError('Target sequence differs from the provided AF3 homology query')
    else:
        raw=specification
    if raw.get('modifications'):
        raise ValueError('Target PTMs require an explicitly supported sequence/input contract')
    templates=[]
    for t in raw.get('templates') or []:
        q=list(t['queryIndices']);indices=list(t['templateIndices'])
        if len(q)!=len(indices) or len(set(q))!=len(q) or any(not 0<=i<protomer.length for i in q):
            raise ValueError('Invalid fixed-target template mapping')
        templates.append(Template(mmcif=t['mmcif'],query_to_template_map=dict(zip(q,indices))))
    for key in ['unpairedMsa','pairedMsa']:
        msa=raw.get(key) or ''
        if msa:
            from astrolabe.core.msa_utils import parse_a3m_sequences
            if parse_a3m_sequences(msa)[0][1]!=protomer.sequence:
                raise ValueError('MSA query does not match fixed target sequence')
    protomer._homology={k:raw.get(k) for k in ['unpairedMsa','pairedMsa']}
    protomer._homology['templates']=templates
    # The public property otherwise remasks/reformats every new sequence value.
    protomer._homology_seq=protomer.sequence
