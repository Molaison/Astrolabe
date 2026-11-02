"""Mol-1 evaluator, using Molence's own model and confidence conventions."""
from collections import defaultdict
import json
from pathlib import Path

from astrolabe.core import global_config
from astrolabe.core.folding_input import ProteinChain
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.evaluators.wrappers.opendde import OpenDDEPreparedInput, collect_opendde_outputs


class Mol1Evaluator(Evaluator):
    def __init__(self, runner, output_dir):
        self.runner = runner
        self.output_dir = Path(output_dir)

    def prepare_input(self, inputs, runtime_config):
        output = {}
        step = int(getattr(global_config, 'step', 0) or 0)
        for name, fold in inputs.items():
            if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
                raise ValueError('invalid Mol-1 protomer name')
            if getattr(fold, 'bonded_atom_pairs', None) or getattr(fold, 'cyclic_chain_ids', None) or getattr(fold, 'cyclic_relative_patches', None):
                raise ValueError('Mol-1 adapter does not yet support topology constraints')
            if len(name) != len(fold.chains):
                raise ValueError('Mol-1 protomer mapping does not match chain count')
            chains, chain_order, lengths = [], [], {}
            for chain in fold.chains:
                if not isinstance(chain, ProteinChain) and type(chain).__name__ != "ProteinChain":
                    raise ValueError('Mol-1 adapter currently supports protein chains only')
                if chain.ptms or chain.templates:
                    raise ValueError('Mol-1 adapter does not yet support modifications or templates')
                cid, sequence = str(chain.id), str(chain.sequence)
                if not cid or Path(cid).name != cid or cid in ('.', '..') or cid in lengths or not sequence or any(x not in 'ACDEFGHIKLMNPQRSTVWYX' for x in sequence):
                    raise ValueError('invalid Mol-1 protein chain')
                entry = {'id': [cid], 'sequence': sequence, 'count': 1}
                for attr, key, filename in (('unpaired_msa', 'unpairedMsaPath', 'unpaired.a3m'), ('paired_msa', 'pairedMsaPath', 'paired.a3m')):
                    text = getattr(chain, attr, None)
                    if text:
                        msa = self.output_dir / 'input_msa' / f'step_{step:03d}_{name}' / cid / filename
                        msa.parent.mkdir(parents=True, exist_ok=True)
                        msa.write_text(text, encoding='utf-8')
                        entry[key] = str(msa.resolve())
                chains.append({'proteinChain': entry})
                chain_order.append({'id': cid, 'kind': 'protein'})
                lengths[cid] = len(sequence)
            if not chains: raise ValueError('Mol-1 requires at least one protein chain')
            seeds = [int(s) for s in (getattr(fold, 'rng_seeds', None) or [101])]
            if len(seeds) != len(set(seeds)):
                raise ValueError('Mol-1 requires unique seeds for distinct outputs')
            job = f'step_{step:03d}_{name}'
            directory = self.output_dir / 'inputs'; directory.mkdir(parents=True, exist_ok=True)
            path = directory / (job + '.json')
            path.write_text(json.dumps([{'name': job, 'modelSeeds': seeds, 'sequences': chains}]))
            output[name] = OpenDDEPreparedInput(path, self.output_dir, job, chain_order, lengths)
        return output

    def run_function(self, input_feat):
        from dataclasses import replace
        return {name: replace(prepared, output_dir=self.runner.run(prepared.json_path, prepared.output_dir))
                for name, prepared in input_feat.items()}

    def prepare_output(self, output, inputs, runtime_config):
        structures = Path(global_config.out_dir) / 'structures'
        plddt, ptm, iptm, pde, paths, chain_pairs = {}, {}, {}, {}, {}, {}
        per_proto = defaultdict(list)
        for name, prepared in output.items():
            scores = collect_opendde_outputs(prepared, structures_dir=structures)
            # Mol-1 documents whole-complex pLDDT on 0-100, chain scores on 0-1.
            plddt[name] = [x / 100.0 for x in scores['plddt']]
            if scores['ptm'] is not None: ptm[name] = scores['ptm']
            if scores['iptm'] is not None: iptm[name] = scores['iptm']
            if scores['pde'] is not None: pde[name] = scores['pde']
            paths[name] = scores['complex_path']
            per_chain = scores['per_proto_scores'] or {}
            for index, chain in enumerate(prepared.chain_order):
                per_proto[name[index]].extend(per_chain.get(chain['id'], []))
            if scores['chain_pair_scores']: chain_pairs[name] = scores['chain_pair_scores']
        return StandardScores(names=list(output), plddt=plddt, ptm=ptm or None, iptm=iptm or None,
                              pde=pde or None, pae=None, per_proto_scores=dict(per_proto) or None,
                              chain_pair_scores=chain_pairs or None, complex_paths=paths)
