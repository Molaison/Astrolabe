"""Opt-in real CPU comparison; not evidence of Rosetta or GPU speedup.

ASTROLABE_METRICS_BENCHMARK=1 python -m pytest -q -s \
    tests/test_independent_structure_metrics_benchmark.py
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("ASTROLABE_METRICS_BENCHMARK") != "1",
                    reason="Opt-in full native CMS timing comparison")
def test_reference_and_chunked_cms_five_alternating_rounds():
    struc = pytest.importorskip("biotite.structure")
    library = pytest.importorskip("py_contact_ms")
    from astrolabe.losses.independent import contact_molecular_surface, load_structure

    path = Path(__file__).resolve().parents[1] / "examples/input/9TP_cid.cif"
    atoms = load_structure(path)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"

    def reference():
        ra = library.get_radii_from_names(atoms.res_name[a], atoms.atom_name[a])
        rb = library.get_radii_from_names(atoms.res_name[b], atoms.atom_name[b])
        value, per_atom, _ = library.calculate_contact_ms(atoms.coord[a], ra, atoms.coord[b], rb)
        return value, np.asarray(per_atom)

    def optimized():
        # The public Python reference integrates its target/B surface; native
        # Rosetta and Astrolabe's compatibility default integrate selector1/A.
        result = contact_molecular_surface(atoms, a, b, surface_side="b")
        return result.value, np.asarray(result.details["per_target_atom"])

    observations = []
    maximum_error = 0.0
    for round_index in range(5):
        output = {}
        order = [("reference", reference), ("optimized", optimized)]
        if round_index % 2:
            order.reverse()
        for method, function in order:
            started = time.perf_counter()
            output[method] = function()
            row = dict(round=round_index + 1, method=method,
                       seconds=time.perf_counter() - started, score=float(output[method][0]))
            observations.append(row)
            print(json.dumps(row), flush=True)
        value_error = abs(output["reference"][0] - output["optimized"][0])
        atom_error = float(np.max(np.abs(output["reference"][1] - output["optimized"][1])))
        maximum_error = max(maximum_error, value_error, atom_error)
        np.testing.assert_allclose(output["reference"][1], output["optimized"][1], rtol=1e-12, atol=1e-12)
        assert output["optimized"][0] == pytest.approx(output["reference"][0], rel=1e-12, abs=1e-12)
    medians = {method: statistics.median(r["seconds"] for r in observations if r["method"] == method)
               for method in ("reference", "optimized")}
    evidence = dict(
        baseline="py-contact-ms 0.1.1 public calculate_contact_ms",
        surface_side="b",
        input="examples/input/9TP_cid.cif; canonical amino acids; chains A and B",
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        atoms_a=int(a.sum()), atoms_b=int(b.sum()), completed_calls=len(observations), errors=0,
        versions={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "biotite", "py-contact-ms")},
        python=platform.python_version(), platform=platform.platform(),
        observations=observations, median_seconds=medians,
        speedup_over_python_reference=medians["reference"] / medians["optimized"],
        maximum_absolute_error=maximum_error,
        scope="CPU surface calculation only; no PyRosetta/GPU/trajectory throughput comparison",
    )
    destination = os.environ.get("ASTROLABE_METRICS_BENCHMARK_OUTPUT")
    if destination:
        Path(destination).write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)
