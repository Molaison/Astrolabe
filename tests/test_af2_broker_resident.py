"""Real broker and AF2 daemon with audited upstream RunModel and fake model apply."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_two_owners_share_af2_and_worker_death_fences(tmp_path, monkeypatch):
    from astrolabe.runtime.session import BrokerSession
    fake = tmp_path / 'fake_af2.py'
    fake.write_text('''\
import os
import jax.numpy as jnp
from alphafold.model.model import RunModel
import alphafold.model.model as model
class CPU:
    multimer_mode = False
    def __init__(self, initial): self._gh_initial_guess_cfg = initial
    def process_features(self, feat, random_seed): return feat

def setup(oligo_names, prepare_only=False, **cfg):
    if prepare_only:
        runner = CPU(cfg.get('initial_guess'))
    else:
        with open(os.environ['AF2_BROKER_LOADS'], 'a') as f: f.write('load\\n')
        runner = RunModel.__new__(RunModel)
        runner.params = {'loaded': 1}; runner.multimer_mode = False
        def apply(params, key, features):
            if int(features['value']) < 0: os._exit(19)
            return {'value': jnp.asarray(features['value']) + key[1], 'initial': jnp.asarray(features['initial_prev_pos'])}
        runner.apply = apply
        model.get_confidence_metrics = lambda result, multimer_mode: {}
    return {name: runner for name in oligo_names}
''')
    (tmp_path / 'sitecustomize.py').write_text('''\
import astrolabe.evaluators.wrappers.alphafold as wrapper
from fake_af2 import setup
wrapper.alphafold_setup = setup
''')
    (tmp_path / 'params').mkdir()
    (tmp_path / 'params' / 'params_model_3_ptm.npz').write_bytes(b'fake')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    monkeypatch.setenv('AF2_BROKER_LOADS', str(tmp_path / 'loads'))
    script = '''\
import json, os, sys
import numpy as np
from astrolabe.evaluators.af2_resident import AF2Resident
from astrolabe.runtime.client import OutcomeUnknown
owner=int(sys.argv[1]); root=sys.argv[2]
r=AF2Resident.runners(['A'], {'data_dir':root,'suffix':'ptm','initial_guess':{'path':'owner-'+str(owner)}})['A']
if owner == 9:
    try: r.predict({'value':np.asarray(-1),'initial_prev_pos':np.asarray([0])}, 7)
    except OutcomeUnknown: pass
    else: raise AssertionError('worker death was not uncertain')
    try: r.predict({'value':np.asarray(1),'initial_prev_pos':np.asarray([0])}, 7)
    except OutcomeUnknown as e: assert 'fenced' in str(e)
    else: raise AssertionError('device not fenced')
    print(json.dumps({'fenced':True}))
else:
    rows=[]
    for seed in (11,17):
        feat=r.process_features({'value':np.asarray(owner), 'initial_prev_pos':np.asarray([owner,seed])}, seed)
        out=r.predict(feat,seed)
        rows.append([int(out['value']),out['initial'].tolist()])
    print(json.dumps({'owner':owner,'pid':r._client.hello['pid'], 'guess':r._gh_initial_guess_cfg, 'rows':rows}))
r.close()
'''
    with BrokerSession() as session:
        processes = [subprocess.Popen([sys.executable, '-c', script, str(owner), str(tmp_path)], env=session.env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for owner in (1, 2)]
        results = []
        for proc in processes:
            out, err = proc.communicate(timeout=120)
            assert proc.returncode == 0, err
            results.append(json.loads(out.strip().splitlines()[-1]))
        assert results[0]['pid'] == results[1]['pid']
        for owner, result in zip((1, 2), results):
            assert result['guess'] == {'path': 'owner-' + str(owner)}
            assert result['rows'] == [[owner + seed, [owner, seed]] for seed in (11, 17)]
        assert (tmp_path / 'loads').read_text().splitlines() == ['load']
        failure = subprocess.run([sys.executable, '-c', script, '9', str(tmp_path)], env=session.env,
                                 capture_output=True, text=True, timeout=120)
        assert failure.returncode == 0, failure.stderr
        assert json.loads(failure.stdout.strip().splitlines()[-1]) == {'fenced': True}
