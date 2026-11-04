"""CPU native output microbenchmark, NOT pretrained inference or GPU speedup.

The tensor shapes are compatible synthetic ESMFold outputs. The observed bytes
count requested host materializations (CPU tensors already reside on host); no
actual device-to-host transfer bandwidth is measured here.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from astrolabe.evaluators.lifecycle import cpu_tree

SCORES = ('chain_index','plddt','mean_plddt','ptm','predicted_aligned_error')


def native_module():
    path = ROOT / 'tests/fixtures/esmfold_output_native.py'
    spec = importlib.util.spec_from_file_location('native_esmfold_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_output(native, batch, length):
    generator = torch.Generator(device='cpu').manual_seed(9047)
    output = {'aatype': torch.arange(length).remainder(20).expand(batch, length).clone()}
    native.make_atom14_masks(output)
    chain = torch.div(torch.arange(length), max(1, length//2), rounding_mode='floor').clamp(max=1)
    output.update({
        'positions': torch.rand((2,batch,length,14,3), generator=generator),
        'residue_index': (torch.arange(length)+512*chain).expand(batch,length).clone(),
        'chain_index': chain.expand(batch,length).clone(),
        'plddt': torch.rand((batch,length,37), generator=generator)*100,
        'mean_plddt': torch.rand((batch,), generator=generator)*100,
        'ptm': torch.rand((batch,), generator=generator),
        'predicted_aligned_error': torch.rand((batch,length,length), generator=generator),
        # Real output names/shapes from the pinned ESM forward, not a real model run.
        's_z': torch.zeros((batch,length,length,128)),
        'distogram_logits': torch.zeros((batch,length,length,64)),
        'ptm_logits': torch.zeros((batch,length,length,64)),
    })
    return output


def materializations(fn, output):
    events = []
    names = {id(v): k for k,v in output.items()}
    original_to, original_cpu = torch.Tensor.to, torch.Tensor.cpu
    def record(tensor, method):
        events.append({'method': method, 'field': names.get(id(tensor),'derived_or_detached'),
                       'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                       'requested_bytes': tensor.numel()*tensor.element_size()})
    def observed_to(self, *args, **kwargs):
        device = kwargs.get('device', args[0] if args else None)
        if str(device) == 'cpu':
            record(self, 'to_cpu')
        return original_to(self, *args, **kwargs)
    def observed_cpu(self, *args, **kwargs):
        record(self, 'cpu')
        return original_cpu(self, *args, **kwargs)
    with patch.object(torch.Tensor,'to',observed_to), patch.object(torch.Tensor,'cpu',observed_cpu):
        fn(output)
    return {'events':events, 'requested_bytes':sum(e['requested_bytes'] for e in events),
            'calls':len(events), 'actual_gpu_transfer_bytes':None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('baseline','compare'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    native=native_module()
    boundary=None
    if args.mode == 'compare':
        from astrolabe.evaluators.esmfold_output import resolve_pdb_boundary
        boundary=resolve_pdb_boundary(native.NativeModel())
        assert boundary is not None, 'pinned native fixture must be recognized'
    def baseline(output):
        t0=time.perf_counter()
        pdbs=native.output_to_pdb(output)
        scores=cpu_tree({k:output[k] for k in SCORES})
        return pdbs,scores,time.perf_counter()-t0
    def candidate(output):
        t0=time.perf_counter()
        host=boundary.prepare_output(output)
        scores={k:host[k] for k in SCORES}
        inside=time.perf_counter()-t0
        pdbs=boundary.finalize_pdb(host)
        return pdbs,scores,inside
    report={'schema_version':1,'evidence_kind':'real torch CPU tensors and source-extracted native output routines; synthetic inputs',
        'pretrained':False,'gpu':False,'actual_d2h_measured':False,
        'torch':torch.__version__,'numpy':np.__version__,'torch_threads':torch.get_num_threads(),
        'head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'workloads':[],'error_count':0}
    for batch,length in ((1,24),(2,128)):
        output=make_output(native,batch,length)
        frozen={k:v.clone() for k,v in output.items()}
        methods={'baseline':baseline}
        if boundary is not None: methods['candidate']=candidate
        references=baseline(output)
        item={'batch':batch,'length':length,'repetitions':7,
              'input_tensor_shapes':{k:list(v.shape) for k,v in output.items()},
              'pdb_bytes':sum(len(p.encode()) for p in references[0]),
              'pdb_sha256':[hashlib.sha256(p.encode()).hexdigest() for p in references[0]],'methods':{}}
        # One untimed warm call per method, then alternating order to avoid fixed-order bias.
        for fn in methods.values(): fn(output)
        rows={name:[] for name in methods}
        for repeat in range(7):
            order=list(methods) if repeat%2 == 0 else list(reversed(methods))
            for name in order:
                rng=(random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
                t0=time.perf_counter(); pdbs,scores,inside=methods[name](output); total=time.perf_counter()-t0
                assert pdbs==references[0]
                for key in SCORES: assert torch.equal(scores[key],references[1][key])
                assert random.getstate()==rng[0]
                np_after=np.random.get_state()
                assert np_after[0]==rng[1][0] and np.array_equal(np_after[1],rng[1][1]) and np_after[2:]==rng[1][2:]
                assert torch.equal(torch.get_rng_state(),rng[2])
                rows[name].append({'total_seconds':total, 'candidate_device_block_cpu_seconds':inside,
                                   'pdb_count':len(pdbs),'error_count':0})
        for name,fn in methods.items():
            item['methods'][name]={'raw_trials':rows[name],
                'median_total_seconds':statistics.median(r['total_seconds'] for r in rows[name]),
                'median_candidate_device_block_cpu_seconds':statistics.median(r['candidate_device_block_cpu_seconds'] for r in rows[name]),
                'host_materializations':materializations(fn,output)}
        for k,v in output.items(): assert torch.equal(v,frozen[k])
        item['science_checks']={'pdb_byte_identical':True,'score_tensors_exact':True,'rng_exact':True,'inputs_unchanged':True}
        report['workloads'].append(item)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'error_count':report['error_count'],
        'workloads':[{k:v for k,v in w.items() if k in ('batch','length','repetitions','pdb_bytes')} for w in report['workloads']]}))

if __name__=='__main__': main()
