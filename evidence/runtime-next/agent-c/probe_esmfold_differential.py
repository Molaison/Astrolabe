"""Independent CPU/native-output differential; no weights/inference/GPU claims."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from astrolabe.evaluators.esmfold_output import SCORE_FIELDS, resolve_pdb_boundary
    torch.set_num_threads(1)
    fixture = repo / "tests/fixtures/esmfold_output_native.py"
    spec = importlib.util.spec_from_file_location("_c_native_output_probe", fixture)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    boundary = resolve_pdb_boundary(native.NativeModel())
    assert boundary is not None

    def rng_signature():
        np_state = np.random.get_state()
        data = (repr(random.getstate()).encode() + np_state[1].tobytes()
                + repr((np_state[0], *np_state[2:])).encode()
                + torch.get_rng_state().numpy().tobytes())
        return hashlib.sha256(data).hexdigest()

    rows = []
    for batch in (1, 3):
        for length in (2, 7, 31):
            for dtype in (torch.float16, torch.float32, torch.float64):
                for padding in (False, True):
                    generator = torch.Generator().manual_seed(271828)
                    output = {"aatype": torch.arange(length).remainder(21).repeat(batch, 1)}
                    native.make_atom14_masks(output)
                    chain = (torch.arange(length) // max(1, length // 3)).clamp(max=2)
                    output.update({
                        "positions": torch.rand((3, batch, length, 14, 6), generator=generator, dtype=dtype)[..., ::2],
                        "chain_index": chain.repeat(batch, 1),
                        "residue_index": (torch.arange(length) + chain * 200).repeat(batch, 1),
                        "plddt": torch.rand((batch, length, 37), generator=generator, dtype=dtype) * 100,
                        "mean_plddt": torch.rand((batch,), generator=generator, dtype=dtype) * 100,
                        "ptm": torch.rand((batch,), generator=generator, dtype=dtype),
                        "predicted_aligned_error": torch.rand((batch, length, length * 2), generator=generator, dtype=dtype)[..., ::2],
                        "unused_pair": torch.rand((batch, length, length, 8), generator=generator, dtype=dtype),
                    })
                    if padding:
                        output["chain_index"][:, -1] = -1
                        output["atom37_atom_exists"][:, -1] = 0
                    originals = {key: value.clone() for key, value in output.items()}
                    rng_before = rng_signature()
                    expected_pdb = native.NativeModel().output_to_pdb(output)
                    host = boundary.prepare_output(output)
                    actual_pdb = boundary.finalize_pdb(host)
                    assert actual_pdb == expected_pdb
                    for key in SCORE_FIELDS:
                        assert host[key].dtype == output[key].dtype
                        assert torch.equal(host[key].contiguous().view(torch.uint8),
                                           output[key].contiguous().view(torch.uint8))
                    assert all(torch.equal(value, originals[key]) for key, value in output.items())
                    assert rng_signature() == rng_before
                    rows.append({
                        "batch": batch, "residues": length, "dtype": str(dtype), "padding": padding,
                        "pdb_sha256": [hashlib.sha256(pdb.encode()).hexdigest() for pdb in actual_pdb],
                        "pdb_exact": True, "score_bits_exact": True, "input_unchanged": True, "rng_exact": True,
                    })

    rejected = []
    original_np = native.np
    try:
        native.np = object()
        assert resolve_pdb_boundary(native.NativeModel()) is None
        rejected.append("changed_numpy_binding")
    finally:
        native.np = original_np
    original_restypes = native.residue_constants.restypes
    try:
        native.residue_constants.restypes = list(reversed(original_restypes))
        assert resolve_pdb_boundary(native.NativeModel()) is None
        rejected.append("changed_residue_table")
    finally:
        native.residue_constants.restypes = original_restypes
    original_headers = native.get_pdb_headers
    try:
        native.get_pdb_headers = lambda *_args: ["altered"]
        assert resolve_pdb_boundary(native.NativeModel()) is None
        rejected.append("changed_formatter_dependency")
    finally:
        native.get_pdb_headers = original_headers
    class Custom(native.NativeModel):
        def output_to_pdb(self, output):
            return ["custom"]
    assert resolve_pdb_boundary(Custom()) is None
    rejected.append("custom_wrapper")
    assert resolve_pdb_boundary(native.NativeModel()) is not None

    report = {
        "head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "scope": "Independent real torch CPU/native-source output differential; synthetic outputs, no pretrained inference/GPU/broker",
        "torch": torch.__version__, "numpy": np.__version__,
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "cases": rows,
        "unknown_dependency_controls_rejected": rejected,
        "error_count": 0,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"head": report["head"], "cases": len(rows), "unknown_dependency_controls_rejected": rejected, "error_count": 0}))


if __name__ == "__main__":
    main()
