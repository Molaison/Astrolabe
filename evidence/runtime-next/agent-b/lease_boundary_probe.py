"""Read-only lease-boundary probe. Fake tensors/inference/admission; NOT GPU evidence.

Compiles the exact repository handle and pinned upstream output_to_pdb function.
The stub formatter records whether the real call graph invokes it inside the
device context. No Torch, model checkpoint, GPU, socket, or broker is touched.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess
import typing as T
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORKER = ROOT / "astrolabe/evaluators/wrappers/esmfold_daemon.py"
BASELINE = "d66ee714f2e755948534f95d9d125287ce8dbd91"
WORKER_SOURCE = subprocess.check_output(["git", "-C", str(ROOT), "show", BASELINE + ":astrolabe/evaluators/wrappers/esmfold_daemon.py"], text=True)
NATIVE = HERE / "esmfold1-native-misc.py"
events = []
active = 0

def event(kind, **extra):
    events.append({"event": kind, "device_lease_active": bool(active), **extra})

class Tensor:
    def __init__(self, value, name="tensor"):
        self.value = np.asarray(value)
        self.name = name
    def __getitem__(self, index):
        return Tensor(self.value[index], self.name)
    def to(self, device):
        event("tensor_to_host", name=self.name, bytes=self.value.nbytes)
        return self
    def cpu(self):
        return self.to("cpu")
    def numpy(self):
        return self.value

def atom14_to_atom37(value, output):
    event("atom14_to_atom37")
    return Tensor(np.pad(value.value, ((0,0),(0,0),(0,23),(0,0))), "atom37")

class OFProtein:
    def __init__(self, **kwargs):
        event("OFProtein")
        self.values = kwargs

def to_pdb(protein):
    event("native_host_to_pdb")
    return "PROBE_HOST_PDB"

@contextlib.contextmanager
def operation(*args, **kwargs):
    global active
    active += 1
    event("lease_enter")
    try:
        yield SimpleNamespace(update_lifecycle=lambda **kw:event("lifecycle"))
    finally:
        event("lease_exit")
        active -= 1

def get_function(path, name):
    tree = ast.parse(WORKER_SOURCE if path == WORKER else path.read_text())
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise RuntimeError(f"expected one {name}: {len(found)}")
    return found[0]

def compile_function(path, name, env):
    mod = ast.Module(body=[get_function(path,name)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), env)
    return env[name]

native = compile_function(NATIVE, "output_to_pdb", globals())
output = {
    "positions": Tensor(np.zeros((1,1,2,14,3),dtype=np.float32), "positions"),
    "aatype": Tensor(np.zeros((1,2),dtype=np.int64), "aatype"),
    "atom37_atom_exists": Tensor(np.ones((1,2,37),dtype=np.float32), "atom37_mask"),
    "residue_index": Tensor(np.arange(2)[None,:], "residue_index"),
    "chain_index": Tensor(np.zeros((1,2),dtype=np.int64), "chain_index"),
    "plddt": Tensor(np.ones((1,2,37),dtype=np.float32)*90, "plddt"),
    "mean_plddt": Tensor([90.0], "mean_plddt"),
    "ptm": Tensor([0.9], "ptm"),
    "predicted_aligned_error": Tensor(np.zeros((1,2,2)), "pae"),
    "unused_pair_probe": Tensor(np.zeros((1,2,2,16)), "unused_pair"),
}
def infer(sequences):
    event("fake_model_infer")
    return output
def cpu_tree(value):
    return {k:v.cpu() for k,v in value.items()}
env = dict(contextlib=contextlib,sys=sys,torch=SimpleNamespace(no_grad=contextlib.nullcontext),
    operation=operation,actual={},spec={"device":{"name":"fake-device"}},
    request_directory=Path,load_tree=lambda path:{"rng":{},"sequences":["AC"]},
    restore=lambda *a: event("rng_restore"),capture=lambda *a:{},
    model=SimpleNamespace(infer=infer,output_to_pdb=native),cpu_tree=cpu_tree,
    lifecycle=lambda:{},dump_tree=lambda *a:event("result_codec"))
handle = compile_function(WORKER,"handle",env)
handle({"request_dir": str(HERE / "probe_request_no_io")})
violations = [e for e in events if e["event"] in {"OFProtein","native_host_to_pdb"} and e["device_lease_active"]]
result = {
    "scope":"AST execution of actual worker+native helper with fake tensors, inference, formatter and device admission; code-contract evidence only",
    "real_gpu":False,"real_pretrained_model":False,"timing_claim":None,
    "baseline_head":BASELINE,
    "worker_sha256":hashlib.sha256(WORKER_SOURCE.encode()).hexdigest(),
    "native_sha256":hashlib.sha256(NATIVE.read_bytes()).hexdigest(),
    "host_formatter_inside_lease":bool(violations),
    "unused_output_copied":any(e.get("name") == "unused_pair" for e in events),
    "events":events,
}
parser=argparse.ArgumentParser()
parser.add_argument("--require-host-finalization",action="store_true")
args=parser.parse_args()
print(json.dumps(result,indent=2))
if args.require_host_finalization and violations:
    raise SystemExit(1)
