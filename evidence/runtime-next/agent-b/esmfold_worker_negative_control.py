"""Run the same real-tensor worker contract against the old or working source."""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
parser = argparse.ArgumentParser()
parser.add_argument('--baseline', action='store_true')
args = parser.parse_args()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

suite = load('esmfold_worker_contract', root/'tests/test_esmfold_output_boundary.py')
native = suite.native.__wrapped__()
with tempfile.TemporaryDirectory() as temporary:
    temp = Path(temporary)
    if args.baseline:
        source = subprocess.check_output(['git', '-C', str(root), 'show',
            'd66ee714f2e755948534f95d9d125287ce8dbd91:astrolabe/evaluators/wrappers/esmfold_daemon.py'], text=True)
        path = temp/'old_esmfold_daemon.py'
        path.write_text(source)
        suite.esmfold_daemon = load('old_esmfold_daemon_contract', path)
    try:
        suite.test_actual_worker_boundary_output_rng_and_unused_lifetime(native, temp, False)
    except AssertionError as exc:
        print(json.dumps({'baseline':args.baseline,'gate_pass':False,'error':str(exc),
                          'gpu':False, 'pretrained':False}))
        raise SystemExit(1)
    print(json.dumps({'baseline':args.baseline,'gate_pass':True,'gpu':False,'pretrained':False}))
