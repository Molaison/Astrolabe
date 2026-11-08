"""Bounded native Proton parity probe. Heavy imports occur only in main()."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime, timezone


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot preserve metadata type {type(value)!r}")


def _require_finite_energy(value, label):
    """Validate raw scientific values before any audit serialization."""
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_energy(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_energy(child, f"{label}[{index}]")
    elif hasattr(value, "detach"):
        _require_finite_energy(value.detach().cpu().tolist(), label)
    elif hasattr(value, "tolist"):
        _require_finite_energy(value.tolist(), label)
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError(f"Nonfinite scientific energy at {label}: {value!r}")


def _check_scientific_energies(value, label="native"):
    if dataclasses.is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{label}.{key}"
            if "energy" in key or "energies" in key or key.endswith("_dH"):
                _require_finite_energy(child, path)
            else:
                _check_scientific_energies(child, path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_scientific_energies(child, f"{label}[{index}]")


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _criteria(manifest, item):
    result = {}
    for name in ("deterministic", "stochastic"):
        params = dict(manifest["criteria"]["shared"])
        params.update(manifest["criteria"][name])
        bounded = dict(explicit_centers=item["centers"], center_count=2,
                       center_protonation_types=["HIS-S"], samples_per_site=1,
                       max_plans_per_seed=1)
        if any(k in params and params[k] != v for k, v in bounded.items()):
            raise ValueError("NEED_DECISION: manifest conflicts with one-output explicit-center bound")
        params.update(bounded)
        result[name] = params
    return result


def _validate_manifest(manifest):
    if manifest.get("schema") != "proton_native_parity_v1":
        raise ValueError("NEED_DECISION: expected proton_native_parity_v1 input schema")
    if manifest["contexts"] != ["full_binder_mask", "original_restricted_mask"]:
        raise ValueError("NEED_DECISION: unexpected context recipe")
    if not 1 <= len(manifest["items"]) <= 2:
        raise ValueError("NEED_DECISION: bounded harness accepts one or two parents")
    ids = [item["item_id"] for item in manifest["items"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate item_id")
    for item in manifest["items"]:
        sequence = item["binder_sequence"]
        mutable, fixed = set(item["mutable_positions"]), set(item["fixed_positions"])
        if mutable & fixed or mutable | fixed != set(range(len(sequence))):
            raise ValueError("Mutable/fixed positions must partition the binder (zero based)")
        hs = item["histidine_positions"]
        if len(hs) != 2 or len(set(hs)) != 2 or not set(hs) <= mutable:
            raise ValueError("Expected two distinct mutable His centers")
        if any(sequence[p] != "H" for p in hs):
            raise ValueError("His position does not match binder_sequence")
        if len(item["centers"]) != 2 or any(c["protonation_type"] != "HIS-S" for c in item["centers"]):
            raise ValueError("NEED_DECISION: center target must be HIS-S")
        for name, params in _criteria(manifest, item).items():
            expected = dict(combined_lambda=.3, max_mutations=3, neighbour_k=16,
                            seed_source="inverse", selective_source="potts", selective=True,
                            dep_map={"HIS-S": ["HIS-P"]})
            expected.update(method="block_descent" if name == "deterministic" else "autoregressive",
                            backend="potts" if name == "deterministic" else "mpnn",
                            temperature=0.0 if name == "deterministic" else .2)
            if name == "deterministic":
                expected["block_size"] = 2
            if any(params.get(k) != v for k, v in expected.items()):
                raise ValueError("NEED_DECISION: manifest differs from authorized parity recipe")


# These are upstream Git blob IDs, not hashes of the patched release files.
# Pinned against SOURCE_PROVENANCE.json and the verified upstream Git objects.
_UPSTREAM_COMMIT = "09682abfa7d20e0abcdeea0490b7a4b1c190aee3"
_UPSTREAM_BLOBS = {
    "foundry/models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py":
        "b6684182892aa7d16936ffb51e26c805992d0134",
    "foundry/models/mpnn/src/mpnn/utils/inference.py":
        "a4fc50bc1f0f814978332066a367b0f8543e94d8",
}


def _validate_source(root, arm):
    """Verify critical source bytes before imports; never alter the source tree."""
    root = Path(root).resolve()
    upstream = arm in {"pristine", "clean"}
    result = {"root": str(root), "kind": "pinned_upstream" if upstream else "legacy_snapshot",
              "upstream_commit": _UPSTREAM_COMMIT if upstream else None,
              "scope": "two critical source files; not whole-tree qualification", "files": {}}
    mismatches = []
    for relative, expected in _UPSTREAM_BLOBS.items():
        path = root / relative
        if not path.is_file():
            mismatches.append(f"{relative}: missing")
            continue
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()
        result["files"][relative] = {"path": str(path.resolve()), "git_blob": blob,
                                     "sha256": hashlib.sha256(content).hexdigest()}
        if upstream and blob != expected:
            mismatches.append(f"{relative}: Git blob {blob}, expected upstream {expected}")
    if mismatches:
        if upstream:
            raise ValueError(
                f"{arm} requires exact upstream source at {_UPSTREAM_COMMIT}; "
                + "; ".join(mismatches)
                + ". Explicitly prepare a separate original upstream source tree and pass "
                  "--source-root PATH. The release vendor is patched; do not reverse patches "
                  "or modify the shared tree. clean applies only its process-local mask adapter."
            )
        raise ValueError("Independent legacy source tree is incomplete: " + "; ".join(mismatches))
    provenance = root / "SOURCE_PROVENANCE.json"
    if provenance.is_file():
        result["manifest"] = {"path": str(provenance), "sha256": _sha(provenance)}
    return result


def _source_root(repo, arm):
    return (repo / "vendor" / ("proton_pottsmpnn_legacy" if arm == "legacy" else "proton_pottsmpnn")).resolve()


def _module_provenance(root):
    result = {}
    for name, module in sorted(sys.modules.items()):
        if name.split(".")[0] not in ("mpnn", "foundry"):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            paths = [Path(p).resolve() for p in getattr(module, "__path__", [])]
            if any(not p.is_relative_to(root) for p in paths):
                raise RuntimeError(f"Mixed source namespace: {name}: {paths}")
            result[name] = {"namespace_paths": [str(p) for p in paths]}
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"Mixed source module: {name}: {path}")
        result[name] = {"path": str(path), "sha256": _sha(path)}
    return result


def compare_reports(reports):
    """Read-only exact comparison of completed arm reports (no numerical tolerance invented)."""
    reports = list(reports)
    if len({r["arm"] for r in reports}) != len(reports):
        raise ValueError("Duplicate parity arm")
    if len({(r["manifest_sha256"], r["checkpoint"]["sha256"], r["hbplus"]["sha256"])
            for r in reports}) != 1:
        raise ValueError("Cannot compare different manifests, checkpoints, or HBPLUS binaries")
    comparisons = []
    for i, left in enumerate(reports):
        for right in reports[i + 1:]:
            other = {(c["item_id"], c["mode"]): c for c in right["cases"]}
            for a in left["cases"]:
                key = (a["item_id"], a["mode"])
                b = other.get(key)
                row = {"arms": [left["arm"], right["arm"]], "item_id": key[0], "mode": key[1]}
                if b is None or "states" not in a or "states" not in b:
                    row["status"] = "incomplete"
                else:
                    row["context_exact"] = {k: a["context_arrays"][k] == b["context_arrays"][k]
                                             for k in ("S_native", "E_idx", "etab")}
                    row["raw_annotation_hash_exact"] = a["context_tokens_sha256"] == b["context_tokens_sha256"]
                    # Annotation categories are a set of names. Their iteration
                    # order varies with PYTHONHASHSEED across fresh processes;
                    # compare every named column exactly, retaining raw hashes.
                    ax, bx = a.get("context_token_annotation"), b.get("context_token_annotation")
                    if ax is not None and bx is not None:
                        row["tokens_exact"] = (ax["tokens"] == bx["tokens"]
                            and set(ax["annotation_categories"]) == set(bx["annotation_categories"])
                            and ax["atom_count"] == bx["atom_count"]
                            and ax["token_count"] == bx["token_count"])
                        row["changed_annotation_columns"] = sorted(k for k in set(ax["tokens"]) | set(bx["tokens"])
                            if ax["tokens"].get(k) != bx["tokens"].get(k))
                    else:
                        row["tokens_exact"] = row["raw_annotation_hash_exact"]
                    row["vocabulary_exact"] = a["token_to_idx"] == b["token_to_idx"]
                    row["state_energies_exact"] = a["states"] == b["states"]
                    row["single_center_exact"] = a["single_center_S_minus_P"] == b["single_center_S_minus_P"]
                    row["native_outputs_exact"] = {}
                    for name in ("deterministic", "stochastic"):
                        ar, br = a["native_runs"].get(name, {}), b["native_runs"].get(name, {})
                        def sequences(run):
                            return [(o["native_fields"]["canonical_sequence"],
                                     o["native_fields"]["extended_tokens"]) for o in run.get("outputs", [])]
                        row["native_outputs_exact"][name] = (sequences(ar) == sequences(br)
                                                              if ar.get("outputs") and br.get("outputs") else None)
                    requires_design = key[1] == "full_binder_mask" or "pristine" not in row["arms"]
                    complete = a["status"] == b["status"] == "complete" if requires_design else True
                    matches = (all(row["context_exact"].values()) and row["tokens_exact"]
                               and row["vocabulary_exact"] and row["state_energies_exact"]
                               and row["single_center_exact"])
                    if requires_design:
                        matches = matches and all(v is True for v in row["native_outputs_exact"].values())
                    if not requires_design:
                        row["status"] = "expected_pristine_mask_difference"
                        row["unsafe_pristine_design_skipped"] = True
                    else:
                        row["status"] = "match" if matches and complete else "difference"
                comparisons.append(row)
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repo-root", "checkpoint", "hbplus", "inputs", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--arm", choices=("pristine", "clean", "legacy"), required=True)
    parser.add_argument("--source-root", type=Path,
                        help="Separate original upstream tree for pristine/clean; default vendor is patched.")
    args = parser.parse_args(argv)
    if args.arm == "legacy" and args.source_root is not None:
        raise ValueError("legacy uses the independent repo/vendor/proton_pottsmpnn_legacy tree; --source-root is only for pristine/clean")
    # One process is one arm. Do not purge/reuse a previously imported engine.
    if any(n.split(".")[0] in ("mpnn", "foundry") for n in sys.modules):
        raise RuntimeError("Start a fresh process per arm; mpnn/foundry already imported")
    manifest = json.loads(args.inputs.read_text())
    _validate_manifest(manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed/partial evidence: {report_path}")
    started = time.monotonic()
    report = {"schema": "proton_native_parity_result_v1", "arm": args.arm,
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "manifest": str(args.inputs.resolve()), "manifest_sha256": _sha(args.inputs),
              "manifest_data": manifest, "cases": [], "budgets": {"engines": 0, "contexts": 0,
              "native_calls": 0, "native_outputs": 0, "context_cache_hits": 0},
              "semantics": "Minimize native target S minus off P selective energy; no sign flip. "
                           "Potts energies are not probabilities or pKa."}
    _write(report_path, report)
    try:
        repo = args.repo_root.resolve()
        source = args.source_root.resolve() if args.source_root is not None else _source_root(repo, args.arm)
        report["source_root"] = str(source)
        report["source_provenance"] = _validate_source(source, args.arm)
        report["checkpoint"] = {"path": str(args.checkpoint.resolve()), "sha256": _sha(args.checkpoint)}
        report["hbplus"] = {"path": str(args.hbplus.resolve()), "sha256": _sha(args.hbplus)}
        os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                          OPENBLAS_NUM_THREADS="2", HBPLUS_PATH=str(args.hbplus.resolve()))
        sys.path[:0] = [str(source / "foundry/models/mpnn/src"), str(source / "foundry/src")]
        import numpy as np
        import torch
        from atomworks.ml.utils.token import get_token_starts
        from mpnn.inference_engines import potts_mpnn_ph as native
        from mpnn.utils import inference

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True)
        report["runtime"] = {"python": sys.version, "torch": torch.__version__,
                             "numpy": np.__version__, "device": "cpu", "threads": 2, "vocab": "v6"}
        wrappers = repo / "astrolabe/mutators/wrappers"
        if args.arm == "clean":
            path = wrappers / "proton_design_mask_adapter.py"
            adapter = _load_file(path, "_parity_mask_adapter")
            report["adapter"] = {"installed": bool(adapter.install_design_mask_adapter()),
                                 "path": str(path), "sha256": _sha(path)}
        else:
            report["adapter"] = {"installed": False}
        io = _load_file(wrappers / "proton_pottsmpnn_daemon.py", "_parity_io")
        report["io_helper"] = {"path": str(wrappers / "proton_pottsmpnn_daemon.py"),
                               "sha256": _sha(wrappers / "proton_pottsmpnn_daemon.py")}

        def seed_all(seed):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)

        def snapshot(array):
            tokens = array[get_token_starts(array)]
            return {"annotation_categories": array.get_annotation_categories(),
                    "tokens": {name: getattr(tokens, name).tolist()
                               for name in tokens.get_annotation_categories()},
                    "atom_count": len(array), "token_count": len(tokens)}

        # Observe the actual parser, including the shim's returned annotation. No stub data.
        original_parser = inference.parse_atom_array
        active_case = None

        def observed_parse(array, *positional, **kwargs):
            event = {"before": snapshot(array)}
            active_case["parser_events"].append(event)
            try:
                parsed = original_parser(array, *positional, **kwargs)
                event["returned"] = snapshot(parsed["assemblies"]["1"][0])
                return parsed
            except Exception:
                event["traceback"] = traceback.format_exc()
                raise

        inference.parse_atom_array = observed_parse
        seed_all(0)
        report["budgets"]["engines"] += 1
        engine = native.PottsMPNNPHEngine(checkpoint_path=str(args.checkpoint.resolve()),
                    extended_vocab="v6", device="cpu", out_directory=str(output),
                    write_fasta=False, write_structures=False)
        for item_index, item in enumerate(manifest["items"]):
            for mode in manifest["contexts"]:
                case = {"item_id": item["item_id"], "mode": mode, "seed": item["seed"],
                        "status": "running", "parser_events": [], "native_runs": {}}
                active_case = case
                report["cases"].append(case)
                case_path = output / f"{item_index}_{mode}.json"
                try:
                    pdb = Path(item["pdb_path"])
                    if _sha(pdb) != item["input_sha256"]:
                        raise ValueError("Input structure hash differs from manifest")
                    atom_array = io._load_atom_array(pdb)
                    _, observed = io._token_chain_sequences(atom_array)
                    case["observed_chain_sequences"] = observed
                    if observed.get(item["binder_chain"]) != item["binder_sequence"]:
                        raise ValueError("Structure binder differs from original pair sequence")
                    fixed = [] if mode == "full_binder_mask" else item["fixed_positions"]
                    io._apply_design_mask(atom_array, item["binder_chain"], fixed)
                    case["caller_annotation"] = snapshot(atom_array)
                    seed_all(item["seed"])
                    report["budgets"]["contexts"] += 1
                    with torch.no_grad():
                        ctx = engine._build_context(atom_array, item["binder_chain"], base_seed=item["seed"])
                    if str(ctx.device) != "cpu":
                        raise RuntimeError("Context is not CPU")
                    binder = ctx.chainA_all_idx.tolist()
                    if len(binder) != len(item["binder_sequence"]):
                        raise ValueError("Context binder length differs from pair sequence")
                    centers = [binder[p] for p in item["histidine_positions"]]
                    if [int(ctx.token_aa.res_id[p]) for p in centers] != [c["res_id"] for c in item["centers"]]:
                        raise ValueError("NEED_DECISION: native center residue mapping differs")
                    case["context_token_annotation"] = snapshot(ctx.token_aa)
                    case["context_tokens_sha256"] = hashlib.sha256(json.dumps(
                        _jsonable(case["context_token_annotation"]), sort_keys=True,
                        allow_nan=False).encode()).hexdigest()
                    if not case["parser_events"]:
                        raise RuntimeError("No real native parse_atom_array event observed")
                    requested_local = list(range(len(binder))) if not fixed else item["mutable_positions"]
                    requested = sorted(binder[p] for p in requested_local)
                    actual = sorted(ctx.chA_free_idx.tolist())
                    target_free = (ctx.free_mask & ~ctx.chainA_t).nonzero().flatten().tolist()
                    case["mask"] = {"requested_binder_local_idx": requested_local,
                                    "requested_context_idx": requested, "chA_free_idx": actual,
                                    "actual_binder_local_idx": [i for i, p in enumerate(binder) if p in actual],
                                    "target_free_idx": target_free,
                                    "exact": actual == requested and not target_free}
                    arrays = dict(S_native=ctx.S_native, E_idx=ctx.scorer.E_idx,
                                  etab=ctx.scorer.etab_out, free_mask=ctx.free_mask,
                                  chainA_all_idx=ctx.chainA_all_idx, chA_free_idx=ctx.chA_free_idx)
                    arrays = {k: v.detach().cpu().numpy() for k, v in arrays.items()}
                    npz = output / f"{item_index}_{mode}_context.npz"
                    np.savez(npz, **arrays)
                    case["context_arrays"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                                  "sha256": hashlib.sha256(v.tobytes(order="C")).hexdigest()}
                                              for k, v in arrays.items()}
                    case["context_npz"] = str(npz)
                    case["token_to_idx"] = dict(ctx.encoding.token_to_idx)
                    case["context_center_idx"] = centers
                    initial = ctx.encode_initial_sequence(item["binder_sequence"])
                    neutral = initial.clone()
                    for p in centers:
                        neutral[p] = ctx.encoding.token_to_idx["HIS-S"]
                    energies = {}
                    case["states"] = energies
                    for pattern in ("SS", "SP", "PS", "PP"):
                        sequence = neutral.clone()
                        for p, state in zip(centers, pattern):
                            sequence[p] = ctx.encoding.token_to_idx["HIS-" + state]
                        energies[pattern] = {"energy": float(ctx.scorer.H_of(sequence)),
                                             "encoded_sequence": sequence.tolist()}
                        _require_finite_energy(energies[pattern]["energy"], f"states.{pattern}.energy")
                    si, pi = (ctx.encoding.token_to_idx[t] for t in ("HIS-S", "HIS-P"))
                    gaps = [native._selective_energy_of(ctx, neutral, p, si, [pi]) for p in centers]
                    case["single_center_S_minus_P"] = gaps
                    _require_finite_energy(gaps, "single_center_S_minus_P")
                    case["single_center_S_minus_P_sum"] = sum(gaps)
                    _require_finite_energy(case["single_center_S_minus_P_sum"], "single_center_S_minus_P_sum")
                    pins = [engine._pin(ctx, native.PHDesignCriteria.from_params(
                        _criteria(manifest, item)["deterministic"]), p, "HIS-S") for p in centers]
                    case["native_locked_S_minus_P_sum"] = native._selective_energy_sum(ctx, neutral, pins)
                    _require_finite_energy(case["native_locked_S_minus_P_sum"], "native_locked_S_minus_P_sum")
                    case["joint_PP_minus_SS"] = energies["PP"]["energy"] - energies["SS"]["energy"]
                    _require_finite_energy(case["joint_PP_minus_SS"], "joint_PP_minus_SS")
                    if not case["mask"]["exact"]:
                        case["status"] = "unsafe_mask_design_skipped"
                        if args.arm != "pristine" or mode != "original_restricted_mask":
                            case["failure"] = "Required exact mask failed"
                    else:
                        # Keep the author's direct public API, reusing only this exact built context.
                        build_context = engine._build_context

                        def cached_context(array, chain, *, base_seed=0):
                            if array is not atom_array or chain != item["binder_chain"] or base_seed != item["seed"]:
                                raise RuntimeError("Context cache identity mismatch")
                            report["budgets"]["context_cache_hits"] += 1
                            return ctx

                        engine._build_context = cached_context
                        try:
                            for name, params in _criteria(manifest, item).items():
                                run = case["native_runs"][name] = {"params": params}
                                criteria = native.PHDesignCriteria.from_params(params)
                                run["effective_criteria"] = _jsonable(criteria)
                                seed_all(item["seed"])
                                report["budgets"]["native_calls"] += 1
                                with torch.no_grad():
                                    designs = engine.run_ph_redesign(atom_array=atom_array,
                                        binder_chain=item["binder_chain"], criteria_list=[criteria],
                                        seed=item["seed"], initial_sequences=[item["binder_sequence"]], n_jobs=1)
                                run["seed_energies"] = getattr(designs, "seed_energies", None)
                                run["outputs"] = [{"native_fields": _jsonable(d),
                                                   "metadata": io._design_metadata(d)} for d in designs]
                                report["budgets"]["native_outputs"] += len(designs)
                                _write(case_path, case)
                                _check_scientific_energies(run["seed_energies"], "seed_energies")
                                _check_scientific_energies(list(designs), "native_outputs")
                                _check_scientific_energies([o["metadata"] for o in run["outputs"]], "metadata")
                                if len(designs) != 1:
                                    raise RuntimeError("NEED_DECISION: expected exactly one native output per method")
                                changed = [p for p, (a, b) in enumerate(zip(item["binder_sequence"], designs[0].canonical_sequence)) if a != b]
                                run["changed_binder_local_idx"] = changed
                                if len(designs[0].canonical_sequence) != len(binder) or len(changed) > 3 or not set(changed) <= set(requested_local):
                                    raise RuntimeError("Native output violated sequence/mask/mutation contract")
                        finally:
                            engine._build_context = build_context
                        case["status"] = "complete"
                except Exception:
                    case["status"] = "failed"
                    case["traceback"] = traceback.format_exc()
                finally:
                    _write(case_path, case)
                    report["modules"] = _module_provenance(source)
                    _write(report_path, report)
        report["status"] = "failed" if any(c["status"] == "failed" or c.get("failure") for c in report["cases"]) else "complete"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.monotonic() - started
        _write(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "budgets": report["budgets"]}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
