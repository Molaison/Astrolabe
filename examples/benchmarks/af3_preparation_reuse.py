"""Measure native AF3 preparation reuse across mutations, without model inference."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
import importlib.util
import hashlib
import dataclasses
import datetime
import subprocess
import json
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace

import yaml

import numpy as np


def _assert_equal(actual, expected, path):
    """Compare native metadata as well as numerical features."""
    if isinstance(expected, np.ndarray):
        if expected.dtype.kind != "O":
            np.testing.assert_array_equal(actual, expected, err_msg=path)
        else:
            if actual.shape != expected.shape:
                raise AssertionError(f"Shape differs for {path}")
            for i, (got, wanted) in enumerate(zip(actual.flat, expected.flat, strict=True)):
                _assert_equal(got, wanted, f"{path}[{i}]")
    elif dataclasses.is_dataclass(expected):
        for field in dataclasses.fields(expected):
            _assert_equal(getattr(actual, field.name), getattr(expected, field.name),
                          f"{path}.{field.name}")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"Keys differ for {path}")
        for key in expected:
            _assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        for i, (got, wanted) in enumerate(zip(actual, expected, strict=True)):
            _assert_equal(got, wanted, f"{path}[{i}]")
    elif hasattr(expected, "to_mmcif"):
        if actual.to_mmcif() != expected.to_mmcif():
            raise AssertionError(f"Structure differs for {path}")
    elif not bool(actual == expected):
        raise AssertionError(f"Value differs for {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--msa-depth", type=int, default=1024)
    parser.add_argument("--fixed-seed", action="store_true")
    parser.add_argument("--bare-native", action="store_true", help="Use caller environment instead of config/base.yaml preparation knobs.")
    parser.add_argument("--template-cif", type=Path, help="Optional real protein mmCIF template; its first chain becomes the fixed target.")
    parser.add_argument("--af3-source", type=Path, help="Optional AF3 src directory with native compiled extension already installed.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 5 or args.msa_depth < 1:
        parser.error("Use at least five runs and a positive MSA depth.")

    import alphafold3
    import alphafold3.cpp
    native_assets = Path(alphafold3.__file__).parent
    if args.af3_source:
        source = args.af3_source.resolve() / "alphafold3"
        if not (source / "model/features.py").is_file():
            parser.error("--af3-source must contain alphafold3/model/features.py")
        sys.meta_path = [finder for finder in sys.meta_path
                         if type(finder).__module__ != "_alphafold3_editable"]
        alphafold3.__path__ = [str(source)]
        from alphafold3.common import resources
        resources.ROOT = native_assets

    from alphafold3 import structure
    from alphafold3.common.folding_input import Input, ProteinChain, Template
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation
    from alphafold3.model import features

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "af3_preparation_benchmark", root / "astrolabe/af3_preparation.py")
    preparation = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = preparation
    spec.loader.exec_module(preparation)
    cache = preparation.AF3PreparationCache()
    settings_spec = importlib.util.spec_from_file_location(
        "af3_benchmark_settings", root / "astrolabe/af3_featurisation_settings.py")
    settings = importlib.util.module_from_spec(settings_spec)
    settings_spec.loader.exec_module(settings)
    base_config = yaml.safe_load((root / "config/base.yaml").read_text())["evaluator"]["alphafold3"]
    prep_config = {key: base_config[key] for key in (
        "refstructure_workers", "refstructure_cache", "refstructure_cache_max_entries",
        "msa_cache", "msa_cache_max_entries")}
    config = None if args.bare_native else SimpleNamespace(**prep_config)

    templates = []
    target_sequence = "ACDEFGHIKLMNPQRSTVWY" * 4
    if args.template_cif:
        native_structure = structure.from_mmcif(args.template_cif.read_text(), include_bonds=False)
        target_id = next(iter(native_structure.chain_single_letter_sequence()))
        target = native_structure.filter(chain_id=target_id)
        if target.release_date is None:
            # Predicted fixtures have no PDB release; native template parsing requires one.
            target = target.copy_and_update_globals(release_date=datetime.date(2000, 1, 1))
        target_sequence = target.chain_single_letter_sequence()[target_id]
        templates = [Template(mmcif=target.to_mmcif(),
                              query_to_template_map={i: i for i in range(len(target_sequence))})]
    rng = random.Random(781)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    base = "".join(rng.choices(alphabet, k=80))
    homologues = ["".join(rng.choices(alphabet, k=len(base))) for _ in range(args.msa_depth-1)]

    def fold(sequence, seed):
        a3m = ">query\n" + sequence + "\n" + "".join(
            f">homologue_{i}\n{row}\n" for i, row in enumerate(homologues))
        return Input(name="preparation_benchmark", rng_seeds=[seed], chains=[
            ProteinChain(id="A", sequence=target_sequence, ptms=[], unpaired_msa="",
                         paired_msa="", templates=templates),
            ProteinChain(id="B", sequence=sequence, ptms=[], unpaired_msa=a3m,
                         paired_msa="", templates=[])])

    ccd = chemical_components.Ccd(ccd_pickle_path=str(native_assets / "constants/converters/ccd.pickle"))
    old_native_caches = {}
    side_caches = {"baseline": {}, "cached": {}}
    for name in ("_MSA_FEATURES_CACHE", "_REFSTRUCTURE_REFERENCE_CACHE"):
        if hasattr(features, name):
            old_native_caches[name] = getattr(features, name)
            for side in side_caches:
                side_caches[side][name] = OrderedDict()

    def prepare(side, item):
        for name, value in side_caches[side].items():
            setattr(features, name, value)
        scope = preparation.preparation_cache_scope(cache) if side == "cached" else nullcontext()
        with settings.featurisation_environment(config), scope:
            return featurisation.featurise_input(fold_input=item, ccd=ccd, buckets=None, verbose=False)[0]

    source_file = Path(features.__file__).resolve()
    native_cpp = Path(alphafold3.cpp.__file__).resolve()
    revision = subprocess.run(["git", "-C", str(source_file.parents[3]), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip()
    result = {"preparation_config": None if args.bare_native else prep_config,
              "template_cif": str(args.template_cif) if args.template_cif else None,
              "features_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
              "native_revision": revision, "native_cpp": str(native_cpp),
              "existing_native_caches": list(old_native_caches),
              "classification": "native_CPU_preparation_no_model_inference",
              "target_length": len(target_sequence), "design_length": len(base),
              "msa_depth": args.msa_depth, "runs_per_side": args.runs,
              "fixed_seed": args.fixed_seed, "errors": 0,
              "baseline_s": [], "cached_s": [],
              "native_source": str(Path(features.__file__).resolve())}
    numeric = 0
    try:
        for side in ("baseline", "cached"):
            prepare(side, fold(base, 17))
        for iteration in range(args.runs):
            offset = iteration % (len(base)-3)
            sequence = base[:offset] + "CKE" + base[offset+3:]
            item = fold(sequence, 17 if args.fixed_seed else 19 + iteration)
            values = {}
            sides = ("baseline", "cached") if iteration % 2 == 0 else ("cached", "baseline")
            for side in sides:
                start = time.perf_counter()
                values[side] = prepare(side, item)
                result[side + "_s"].append(time.perf_counter() - start)
            if values["baseline"].keys() != values["cached"].keys():
                raise AssertionError("Feature keys differ.")
            for key, expected in values["baseline"].items():
                actual = values["cached"][key]
                if isinstance(expected, np.ndarray) and expected.dtype.kind != "O":
                    np.testing.assert_array_equal(actual, expected, err_msg=key)
                    numeric += 1
                else:
                    _assert_equal(actual, expected, key)
        result["numeric_feature_arrays_checked"] = numeric
        result["cache_statistics"] = cache.statistics
        for side in ("baseline", "cached"):
            result[side + "_median_s"] = statistics.median(result[side + "_s"])
        result["speedup"] = result["baseline_median_s"] / result["cached_median_s"]
    except BaseException:
        result["errors"] += 1
        raise
    finally:
        for name, value in old_native_caches.items():
            setattr(features, name, value)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
