"""Protein interface beam search with isolated CPU workers and one AF3 GPU owner.

Run ``python -m astrolabe.protein_beam --config case.yaml --out output``.
Configs may inherit repository task defaults through Hydra's ``defaults`` list;
existing standalone configs remain supported.
Only the owner sees CUDA. Workers use the native evaluator, mutator and LossEngine.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import uuid

METRICS = ("interface_ipae", "interface_ipsae", "interface_sc", "interface_holes", "interface_buns")
_RUNTIME = None
PRESERVE_FIXED_TARGET_HOMOLOGY = True
PROTEIN_BEAM_SEARCH_ALGORITHM = "branching"
PROTEIN_BEAM_SEARCH_OWNER = "isolated_native"
_EXECUTION_FIELDS = frozenset(
    {
        "socket_path",
        "workers",
        "in_flight",
        "resume",
        "retry_interrupted",
    }
)
_SEARCH_FIELDS = frozenset(
    {
        "algorithm",
        "owner",
        "execution",
        "summary_format",
    }
)


def validate_pae_matrices(matrices, expected_residues):
    import numpy as np
    array = np.asarray(matrices)
    if (array.ndim < 2 or array.shape[-1] != array.shape[-2] or
        array.shape[-1] < expected_residues or not array.size or not np.isfinite(array).all()):
        raise ValueError("AF3 PAE must be finite square matrices covering every AB residue")


from astrolabe.design.io import atomic_json
from astrolabe.configuration import (
    load_composed_config as load_config,
    selected_search_target,
)


def event(path, kind, **fields):
    data = json.dumps(dict(event=kind, ts=time.time(), pid=os.getpid(), **fields), allow_nan=False) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data.encode())
    finally:
        os.close(fd)


def select_beam(candidates, width):
    unique = {}
    for candidate in candidates:
        if candidate.get("status") != "done" or not candidate.get("eligible", True):
            continue
        score = candidate["score"]
        if not math.isfinite(score) or any(not math.isfinite(candidate["metrics"][m]) for m in METRICS):
            raise ValueError("Beam selection requires complete finite metrics")
        sequence = candidate["sequence"]
        previous = unique.get(sequence)
        if previous is None or (-score, candidate["id"]) < (-previous["score"], previous["id"]):
            unique[sequence] = candidate
    return sorted(unique.values(), key=lambda c: (-c["score"], c["id"]))[:width]


def validate_config(cfg):
    from omegaconf import OmegaConf
    _validate_branching_selection(cfg)
    if set(cfg.seqs) != {"A", "B"} or list(cfg.stages) != ["default"]:
        raise ValueError("Protein beam currently requires exactly binder A, fixed target B and one default stage")
    # Native sequence statements may contain fixed segments; disallow unknown/random starts.
    for name, statement in cfg.seqs.items():
        tokens = str(statement).split()
        if not tokens or any(set(t.lstrip("+")) - set("ACDEFGHIKLMNPQRSTVWY") for t in tokens):
            raise ValueError("Supply explicit standard protein sequences (including any fixed '+' segments)")
    if not all(t.startswith("+") for t in str(cfg.seqs.B).split()):
        raise ValueError("All target B segments must be fixed with '+'")
    if cfg.structure_start is not None or cfg.get("post_eval_hooks"):
        # Do not silently bypass native reference gates or introduce unmatched repacking.
        if cfg.structure_start is not None or any(bool(v.get("enabled", False)) for v in cfg.post_eval_hooks.values() if hasattr(v, "get")):
            raise ValueError("Reference gates/post-eval hooks are not supported by this executor")
    stage = cfg.stages.default
    if list(stage.oligo) != ["AB"] or int(stage.mutation_num) <= 0:
        raise ValueError("Use oligo [AB] and an explicit positive integer mutation_num")
    if stage.acceptance_gate or stage.acceptance_threshold or stage.early_stopping:
        raise ValueError("Beam uses its configured objective; native acceptance gates are not silently ignored")
    if set(cfg.evaluator.evaluator_terms) != {str(stage.evaluator)} or str(cfg.evaluator.evaluator_terms[stage.evaluator]._type).lower() != "alphafold3":
        raise ValueError("Exactly one enabled AF3 evaluator is required")
    if len(stage.mutators) != 1 or set(cfg.mutator.mutator_terms) != set(stage.mutators):
        raise ValueError("Exactly one LigandMPNN proposal route is required")
    term = cfg.mutator.mutator_terms[stage.mutators[0]]
    if str(term._type).lower() != "ligandmpnn" or not term.position_selector or bool(term.get("fallback_to_all_mutable_if_selector_empty", True)):
        raise ValueError("Use LigandMPNN with an explicit selector and no fallback to all mutable positions")
    if not bool(term.get("enforce_mutation_num", False)):
        raise ValueError("LigandMPNN must enforce the explicit mutation budget")
    expected = dict(zip(METRICS, ("ipae", "ipsae", "protomer_shape_complementarity", "interface_holes", "interface_buried_unsatisfied_hbonds")))
    for name, native in expected.items():
        if name not in cfg.loss.loss_terms or cfg.loss.loss_terms[name]._type != native:
            raise ValueError(f"Missing native metric definition: {name} ({native})")
    if not bool(cfg.loss.loss_terms.interface_sc.get("strict", False)):
        raise ValueError("Protein beam requires strict shape-complementarity scoring")
    objective = str(stage.weighted_score_expression)
    if not objective or "???" in objective:
        raise ValueError("Explicit objective is required")
    settings=cfg.protein_beam
    if str(settings.get('selection','weighted')) not in {'weighted','pareto'}:
        raise ValueError('Unknown protein beam selection policy')
    if str(settings.get('holes_ranking','raw_sum')) not in {'raw_sum','per_atom'}:
        raise ValueError('Unknown holes ranking definition')
    from astrolabe.protein_beam_calibration import calibration_enabled
    automatic_root = calibration_enabled(cfg)
    if automatic_root:
        policy = settings.root_calibration
        if (cfg.case_input.get('root_structure') or settings.get('root_parity_reference') or settings.get('initial_state')):
            raise ValueError('Automatic root calibration requires a fresh root, without historical root imports')
        for key in ('original_ca_delta', 'original_interface_bb_delta'):
            if not math.isfinite(float(policy[key])) or float(policy[key]) <= 0:
                raise ValueError('Root calibration deltas must be positive finite limits')
    gates=settings.get('gates') or {}
    known_geometry={'target_ca_rmsd','binder_pose_ca_rmsd','binder_self_ca_rmsd'}
    known_geometry.update({'interface_bb_rmsd','root_binder_pose_ca_rmsd','root_interface_bb_rmsd','root_complex_bb_rmsd','root_interface_binder_bb_rmsd'})
    if set(gates)-{'geometry_max','root_metric_max_ratio','root_metric_min_ratio','root_metric_max_delta'}:
        raise ValueError('Unknown protein beam gate')
    if set(gates.get('geometry_max') or {})-known_geometry:
        raise ValueError('Unknown geometry gate')
    if gates.get('geometry_max') and not (cfg.get('case_input') or {}).get('structure'):
        raise ValueError('Geometry gate requires the supplied case structure')
    if set(gates.get('geometry_max') or {}) & (known_geometry-{'target_ca_rmsd','binder_pose_ca_rmsd','binder_self_ca_rmsd'}) and not (cfg.get('case_input') or {}).get('root_structure') and not automatic_root:
        raise ValueError('Extended geometry gates require a fixed native root structure')
    for group in gates.values():
        if any(not math.isfinite(float(v)) or float(v)<=0 for v in group.values()):
            raise ValueError('Beam gates must be positive finite exploration limits')
    for key in ('root_metric_max_ratio','root_metric_min_ratio','root_metric_max_delta'):
        if set(gates.get(key) or {})-set(METRICS):
            raise ValueError('Unknown metric gate')
    limit=settings.get('max_total_mutations')
    if limit is not None and (int(limit)!=float(limit) or int(limit)<1):
        raise ValueError('max_total_mutations must be a positive integer')
    generations=settings.get('max_generations')
    if generations is not None and (int(generations)!=float(generations) or int(generations)<1):
        raise ValueError('max_generations must be a positive integer')
    proposal_batch=settings.get('proposal_batch_size')
    if proposal_batch is not None and (int(proposal_batch)!=float(proposal_batch) or int(proposal_batch)<1):
        raise ValueError('proposal_batch_size must be a positive integer')
    return objective


def load_worker_config(config_yaml):
    """Freeze the coordinator's quota path before worker-local output changes."""
    from omegaconf import OmegaConf, open_dict
    cfg = OmegaConf.create(config_yaml)
    cfg._set_flag("allow_objects", True)
    policy = cfg.protein_beam.get("af3_call_budget")
    if policy and policy.get("ledger"):
        with open_dict(policy):
            policy.ledger = str(Path(str(policy.ledger)).expanduser().resolve())
    return cfg


def _execution_int(value, *, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"search.execution.{name} must be a positive integer")
    return value


def _execution_bool(value, *, name):
    if type(value) is not bool:
        raise ValueError(f"search.execution.{name} must be a boolean")
    return value


def _search_summary_format(search):
    summary_format = search.get("summary_format")
    if summary_format is None:
        return None
    normalized = str(summary_format).strip().lower()
    if normalized != "json":
        raise ValueError(
            "search.summary_format must be null or 'json', "
            f"got {summary_format!r}"
        )
    return normalized


def _validate_branching_selection(cfg):
    """Validate search routing without requiring an execution socket."""
    search = cfg.get("search") if cfg is not None and hasattr(cfg, "get") else None
    if search is None:
        return None
    if not isinstance(search, Mapping):
        raise TypeError("search configuration must be a mapping")
    unknown_search_fields = set(search.keys()) - _SEARCH_FIELDS
    if unknown_search_fields:
        raise ValueError(
            "invalid search configuration: unknown="
            f"{sorted(unknown_search_fields)!r}"
        )
    _search_summary_format(search)

    selection = selected_search_target(cfg)
    if selection is None:
        raise ValueError("configured protein beam requires an explicit search selection")
    algorithm, owner = selection
    if algorithm != PROTEIN_BEAM_SEARCH_ALGORITHM:
        raise ValueError(
            "configured protein beam requires "
            f"search.algorithm={PROTEIN_BEAM_SEARCH_ALGORITHM!r}"
        )
    if owner != PROTEIN_BEAM_SEARCH_OWNER:
        raise ValueError(
            "configured protein beam requires "
            f"search.owner={PROTEIN_BEAM_SEARCH_OWNER!r}"
        )

    beam_search = cfg.get("beam_search")
    if beam_search is not None:
        if not isinstance(beam_search, Mapping):
            raise TypeError("beam_search must be a mapping")
        if beam_search:
            raise ValueError(
                "Legacy beam_search cannot be combined with explicit "
                "search.algorithm and search.owner"
            )
    return search


def _configured_branching_execution(cfg, *, socket_path=None):
    """Return normalized, explicit execution controls for configured replay."""
    search = _validate_branching_selection(cfg)
    if search is None:
        raise ValueError("configured protein beam requires cfg.search")

    execution = search.get("execution")
    if execution is None:
        raise ValueError("configured protein beam requires cfg.search.execution")
    if not hasattr(execution, "get") or not hasattr(execution, "keys"):
        raise TypeError("cfg.search.execution must be a mapping")
    unknown = set(execution.keys()) - _EXECUTION_FIELDS
    missing = _EXECUTION_FIELDS - set(execution.keys())
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)!r}")
        if unknown:
            details.append(f"unknown={sorted(unknown)!r}")
        raise ValueError("invalid cfg.search.execution: " + ", ".join(details))

    raw_socket = execution.get("socket_path")
    if socket_path is not None:
        raw_socket = socket_path
    if raw_socket is None or not str(raw_socket).strip():
        raise ValueError("cfg.search.execution.socket_path must be explicitly set")
    socket_path = str(Path(str(raw_socket)).expanduser().resolve())
    workers = _execution_int(execution.get("workers"), name="workers")
    in_flight = _execution_int(execution.get("in_flight"), name="in_flight")
    if in_flight > workers:
        raise ValueError("search.execution.in_flight must be <= workers")
    resume = _execution_bool(execution.get("resume"), name="resume")
    retry_interrupted = _execution_bool(
        execution.get("retry_interrupted"),
        name="retry_interrupted",
    )
    if retry_interrupted and not resume:
        raise ValueError("search.execution.retry_interrupted requires resume=true")
    return dict(
        socket_path=socket_path,
        workers=workers,
        in_flight=in_flight,
        resume=resume,
        retry_interrupted=retry_interrupted,
    )


def _canonical_replay_config(
    cfg,
    *,
    output,
    seed,
    socket_path,
    workers,
    in_flight,
    resume,
    retry_interrupted,
):
    """Build a standalone config that ordinary Astrolabe can replay safely."""
    from omegaconf import OmegaConf, open_dict

    replay = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    replay._set_flag("allow_objects", True)
    output = Path(output).expanduser().resolve()
    current_search = replay.get("search")
    summary_format = (
        _search_summary_format(current_search)
        if isinstance(current_search, Mapping)
        else None
    )
    with open_dict(replay):
        replay.repeats = 1
        replay.seed = int(seed)
        replay.out_dir = str(output)
        if "repeat_plan" in replay:
            del replay.repeat_plan
        canonical_search = {
            "algorithm": PROTEIN_BEAM_SEARCH_ALGORITHM,
            "owner": PROTEIN_BEAM_SEARCH_OWNER,
            "execution": {
                "socket_path": str(Path(str(socket_path)).expanduser().resolve()),
                "workers": int(workers),
                "in_flight": int(in_flight),
                "resume": bool(resume),
                "retry_interrupted": bool(retry_interrupted),
            },
        }
        if summary_format is not None:
            canonical_search["summary_format"] = summary_format
        replay.search = canonical_search
    return replay


def _scientific_contract_config(cfg):
    """Remove replay controls from the native scientific identity."""
    from omegaconf import OmegaConf, open_dict

    contract_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    contract_cfg._set_flag("allow_objects", True)
    with open_dict(contract_cfg):
        if contract_cfg.get("search") is not None:
            del contract_cfg.search
        if contract_cfg.get("repeat_plan") is not None:
            del contract_cfg.repeat_plan
    return contract_cfg


def _saved_protein_beam_selection(cfg):
    search = cfg.get("search") if cfg is not None and hasattr(cfg, "get") else None
    if search is None or not hasattr(search, "get"):
        return None
    algorithm = str(search.get("algorithm", "")).strip()
    owner = str(search.get("owner", "")).strip()
    return algorithm, owner


def _validate_historical_output_resume(output, *, resume):
    """Allow a dedicated caller to validate legacy output only via explicit resume."""
    output = Path(output)
    if not output.exists():
        return
    config_path = output / "config.yaml"
    if config_path.is_file():
        from omegaconf import OmegaConf

        saved = OmegaConf.load(config_path)
        selection = _saved_protein_beam_selection(saved)
        if selection is not None:
            if selection != (
                PROTEIN_BEAM_SEARCH_ALGORITHM,
                PROTEIN_BEAM_SEARCH_OWNER,
            ):
                raise ValueError(
                    "saved protein beam output selects another search owner"
                )
            return
    entries = {path.name for path in output.iterdir()}
    if entries - {"owner.lock"} and not resume:
        raise FileExistsError(
            "Historical protein beam output has no canonical replay marker; "
            "resume it only through run_configured_branching with "
            "search.execution.resume=true"
        )
    if entries and not resume:
        raise FileExistsError(
            "Protein beam output exists; set search.execution.resume=true "
            "to validate and resume it"
        )


def run_configured_branching(cfg, repeat):
    """Configured protein-beam owner called by ordinary Astrolabe search dispatch."""
    from astrolabe.design.repeats import RepeatRun

    if not isinstance(repeat, RepeatRun):
        raise TypeError("repeat must be a design.repeats.RepeatRun")
    if repeat.output is None:
        raise ValueError("repeat.output is required for configured protein beam")
    if repeat.start is not None or repeat.initial_sequence is not None:
        raise ValueError(
            "configured protein beam does not support repeat starts or generated "
            "initial sequences"
        )
    execution = _configured_branching_execution(cfg)
    output = Path(repeat.output).expanduser().resolve()
    _validate_historical_output_resume(output, resume=execution["resume"])

    from omegaconf import OmegaConf, open_dict

    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(run_cfg):
        run_cfg.seed = repeat.seed
        run_cfg.repeats = 1
        run_cfg.out_dir = str(output)
    return run_beam(
        run_cfg,
        output,
        execution["socket_path"],
        workers=execution["workers"],
        in_flight=execution["in_flight"],
        resume=execution["resume"],
        retry_interrupted=execution["retry_interrupted"],
    )


def worker_init(config_yaml, output, socket_path):
    global _RUNTIME
    from astrolabe.protein_beam_contract import protect_worker_lifetime
    protect_worker_lifetime()
    # Set before importing native modules, JAX or torch. The owner keeps the GPU env.
    os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", GH_CUDA_PREINITIALIZE="0",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    worker_cpus = os.environ.get("PROTEIN_BEAM_CPU_SET")
    if worker_cpus:
        cpus = [int(c) for c in worker_cpus.split(",")]
        index = int(mp.current_process().name.rsplit("-", 1)[-1])-1
        os.sched_setaffinity(0, {cpus[index % len(cpus)]})
    os.environ.pop('JAX_PLATFORM_NAME', None)
    from omegaconf import OmegaConf, open_dict
    import torch
    torch.set_num_threads(1)
    from astrolabe.core import global_config as gc
    from run import initialize_global_config, setup_protomers, _init_pyrosetta_runtime
    from astrolabe.evaluators.utils import construct_evaluators
    from astrolabe.mutators.base import build_mutators
    from astrolabe.losses.base import LossEngine
    cfg = load_worker_config(config_yaml)
    directory = Path(output)/"workers"/(str(os.getpid())+"_"+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.out_dir = directory
        cfg.raw_model_results_dir = directory/"raw_model_results"
        cfg.raw_model_results_dir.mkdir()
        cfg.evaluator.alphafold3.service_socket = socket_path
        cfg.evaluator.enabled_aliases = [cfg.stages.default.evaluator]
        cfg.mutator.enabled_aliases = list(cfg.stages.default.mutators)
    gc.reset()
    initialize_global_config(cfg)
    target_spec = cfg.homologies.get("B")
    with open_dict(cfg):
        cfg.homologies.B = None
    setup_protomers(cfg)
    with open_dict(cfg):
        cfg.homologies.B = target_spec
    from astrolabe.protein_beam_inputs import preserve_fixed_target_homology
    preserve_fixed_target_homology(cfg, gc.protomers["B"])
    gc.init_const("num_tokens", {"AB": sum(p.length for p in gc.protomers.values())})
    gc.init_const("structure_start", None)
    gc.set_variable("ignored_complexes", [])
    gc.set_variable("negative_design_complexes", [])
    gc.set_variable("mutation_num", int(cfg.stages.default.mutation_num))
    if gc.protomers["B"].mutable_mask.any():
        raise ValueError("Target mask is mutable")
    _init_pyrosetta_runtime()
    evaluator = construct_evaluators(cfg.evaluator, cfg.raw_model_results_dir, ["AB"],
                                     enabled_aliases=[cfg.stages.default.evaluator])[cfg.stages.default.evaluator]
    mutator = build_mutators(cfg.mutator, cfg.raw_model_results_dir)[cfg.stages.default.mutators[0]]
    mutator.mutation_weight = 1.0
    if str(mutator.runner.device) != "cpu":
        raise RuntimeError("LigandMPNN must be isolated on CPU")
    loss = LossEngine(cfg.loss)
    loss.setup(str(cfg.stages.default.weighted_score_expression), None, None)
    _RUNTIME = (cfg, evaluator, mutator, loss, Path(output))
    event(Path(output)/"timeline.jsonl", "worker_ready", cuda_visible=os.environ["CUDA_VISIBLE_DEVICES"],
          jax_platform="cpu", ligandmpnn_device=str(mutator.runner.device))


def worker_task(task):
    import numpy as np
    from astrolabe.core import global_config as gc
    from astrolabe.argparser import set_random_seeds
    from astrolabe.design.legacy import restore_native_sequence as _restore_sequence
    from astrolabe.design.legacy import native_sequence_key as _sequence_key
    from astrolabe.utils.base import construct_standard_input
    from astrolabe.evaluators.utils import ComplexPoses
    cfg, evaluator, mutator, loss, output = _RUNTIME
    timeline = output/"timeline.jsonl"
    start = time.time()
    event(timeline, task["kind"]+"_start", task_id=task["id"])
    _restore_sequence(task["sequence"], gc.protomers["A"], gc.segments)
    set_random_seeds(int(task["seed"]))
    gc.set_variable("step", task["id"])
    os.environ["PH_TRAJECTORY_ID"] = f"{cfg.get('task_name', 'protein_beam')}:candidate_{task['id']}"
    if task["kind"] == "proposal":
        if task.get("parent_artifacts"):
            from astrolabe.protein_beam_contract import verify_native_artifacts
            verify_native_artifacts(task["parent_artifacts"])
        with open(task["snapshot"], "rb") as handle:
            parent_scores = pickle.load(handle)
        before = gc.protomers["A"].sequence
        target = gc.protomers["B"].sequence
        mutator.runner.seed = int(task["seed"])
        mutation_context = {}
        mutator.mutate(gc.protomers, parent_scores, mutation_context)
        sequence = _sequence_key(gc.protomers["A"], gc.segments, mutated=True)
        if gc.protomers["B"].mutated_sequence != target:
            raise ValueError("Proposal changed the fixed target")
        changed = [i for i, (a, b) in enumerate(zip(before, sequence)) if a != b]
        if len(sequence) != len(before) or any(not gc.protomers["A"].mutable_mask[i] for i in changed):
            raise ValueError("Proposal violated the binder mutable mask")
        contexts = mutation_context.get("mutator_selector_score_contexts") or []
        selected = set(contexts[-1]["positions"].get("A", [])) if contexts else set()
        if not selected or not set(changed) <= selected or len(changed) > int(cfg.stages.default.mutation_num):
            raise ValueError("Proposal violated its interface selector or mutation budget")
        result = dict(task, sequence=sequence, changed_positions=changed, selected_positions=sorted(selected),
                      proposal_seed=int(task["seed"]), proposal_s=time.time()-start)
        from astrolabe.protein_beam_policy import sequence_rejections
        root_sequence=''.join(str(cfg.seqs.A).replace('+','').split())
        reasons=sequence_rejections(sequence,before,root_sequence,list(cfg.exclude_AA),
                                    cfg.protein_beam.get('max_total_mutations'))
        result.update(valid=not reasons,proposal_rejections=reasons,
                      total_mutations=sum(a!=b for a,b in zip(root_sequence,sequence)))
        proposal_dir = output/"proposals"/f"{task['id']:06d}"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(proposal_dir/"result.json", result)
    else:
        if task['kind']=='rescore':
            with open(task['snapshot'],'rb') as handle:
                scores=pickle.load(handle)
        else:
            if cfg.protein_beam.get('af3_call_budget'):
                from astrolabe.protein_beam_budget import claim_call
                claim_call(cfg.protein_beam.af3_call_budget,task)
            scores = evaluator.evaluate(construct_standard_input(["AB"], int(task["seed"])), {})
        prediction_end = time.time()
        # Missing/invalid inputs must not become valid zero-valued ipSAE or Sc.
        from astrolabe.losses.confidence import _resolve_residue_level_pae_matrices
        # AF3's StandardScores.pae may hold a chain-level vector. Resolve the
        # complete residue PAE from the native confidence sidecar, as native losses do.
        matrices = _resolve_residue_level_pae_matrices(scores, oligo_name="AB")
        validate_pae_matrices(matrices, sum(p.length for p in gc.protomers.values()))
        scores.pae["AB"] = matrices
        from astrolabe.protein_beam_geometry import verify_prediction_sequences
        verify_prediction_sequences(scores.complex_paths['AB'],task['sequence'],gc.protomers['B'].sequence)
        scores.complex_poses = ComplexPoses.construct_complex_poses(scores)
        scores.collect_interface_holes_details=str(cfg.protein_beam.get('holes_ranking','raw_sum'))=='per_atom'
        value, env = loss.compute_loss(scores, {})
        metrics = {name: float(env[name]) for name in METRICS}
        if not math.isfinite(float(value)) or not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("Incomplete/nonfinite candidate metrics")
        directory = output/"candidates"/f"{task['id']:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        # Never pickle PyRosetta poses across processes. Preserve PAE/confidence and model path.
        scores.complex_poses = None
        snapshot = directory/"scores.pkl"
        temporary = snapshot.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(scores, handle, protocol=5)
        os.replace(temporary, snapshot)
        result = dict(task, status="done", score=float(value), metrics=metrics,
                      snapshot=str(snapshot), structure_path=str(scores.complex_paths["AB"]),
                      prediction_s=prediction_end-start, scoring_s=time.time()-prediction_end,
                      wall_s=time.time()-start)
        result['evaluation_origin']='reused_native_cpu_rescored' if task['kind']=='rescore' else 'fresh_af3'
        from astrolabe.protein_beam_contract import native_artifact_manifest
        result['native_artifacts'] = native_artifact_manifest(result)
        if scores.collect_interface_holes_details:
            result['holes']=scores._interface_holes_details['AB']
        if (cfg.protein_beam.get('gates') or {}).get('geometry_max'):
            from astrolabe.protein_beam_geometry import pose_geometry
            from astrolabe.protein_beam_calibration import geometry_root
            root_path = geometry_root(cfg, output, task, result['structure_path'])
            result['geometry']=pose_geometry(str(cfg.case_input.structure),result['structure_path'],root_path)
        atomic_json(directory/"result.json", result)
    event(timeline, task["kind"]+"_end", task_id=task["id"], duration_s=time.time()-start)
    return result


def execute_batch(pool, tasks, width):
    """Compatibility projection onto the shared bounded transport."""
    from astrolabe.design.branch_execution import gather_ordered
    return gather_ordered(pool, tasks, width, worker_task, identity=lambda task: task["id"])


@contextmanager
def controlled_pool(on_abort=None, **kwargs):
    """Abort owned GPU work before waiting for CPU workers stuck in its RPC."""
    with ProcessPoolExecutor(**kwargs) as pool:
        try:
            yield pool
        except BaseException as error:
            from astrolabe.evaluators.af3_lifecycle import deferred_termination_signals
            with deferred_termination_signals():
                if on_abort is not None:
                    try:
                        on_abort()
                    except BaseException as cleanup:
                        error.add_note(f'AF3 abort: {cleanup!r}')
                # Process objects belong to this executor; never target arbitrary
                # PIDs or a shared GPU owner. Completed native files are retained.
                processes = list((getattr(pool, '_processes', None) or {}).values())
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=2)
            raise


def run_beam(cfg, output, socket_path, *, workers=3, in_flight=3, resume=False, retry_interrupted=False, on_abort=None):
    from omegaconf import OmegaConf
    validate_config(cfg)
    settings = cfg.protein_beam
    width, per_parent, budget = (int(settings[k]) for k in ("width", "proposals_per_parent", "max_evaluations"))
    proposal_batch_size=int(settings.get('proposal_batch_size',workers))
    if min(width, per_parent, budget, workers, in_flight) < 1 or in_flight > workers:
        raise ValueError("Beam/budget/workers must be positive; in_flight <= workers")
    seed = int(cfg.seed)
    if seed < 1:
        raise ValueError("Positive explicit seed required")
    if socket_path is None or not str(socket_path).strip():
        raise ValueError("Protein beam requires an explicit AF3 service socket")
    socket_path = str(Path(str(socket_path)).expanduser().resolve())
    if cfg.get("search") is not None:
        _configured_branching_execution(cfg, socket_path=socket_path)
    from astrolabe.protein_beam_policy import pareto_select, candidate_rejections
    holes_ranking=str(settings.get('holes_ranking','raw_sum'))
    selection=(lambda candidates,width:pareto_select(candidates,width,holes_ranking)) if str(settings.get('selection','weighted'))=='pareto' else select_beam
    gates=settings.get('gates') or {}
    def mark_candidates(state):
        root=next((r for r in state['records'] if r['id']==0 and r['status']=='done'),None)
        if root is None:
            return
        for record in state['records']:
            if record['status']=='done':
                reasons=candidate_rejections(record,root['metrics'],state.get('calibrated_gates',gates))
                record.update(eligible=not reasons,rejection_reasons=reasons)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/"owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker_config_yaml = OmegaConf.to_yaml(cfg, resolve=False)
        replay_config = _canonical_replay_config(
            cfg,
            output=output,
            seed=seed,
            socket_path=socket_path,
            workers=workers,
            in_flight=in_flight,
            resume=resume,
            retry_interrupted=retry_interrupted,
        )
        replay_config_yaml = OmegaConf.to_yaml(replay_config, resolve=False)
        from astrolabe.protein_beam_contract import scientific_contract
        contract_cfg = _scientific_contract_config(cfg)
        contract, digest = scientific_contract(contract_cfg)
        checkpoint = output/"beam_state.json"
        root_verified=False
        def verify_root_context(state):
            nonlocal root_verified
            reference=settings.get('root_parity_reference')
            root=next((r for r in state['records'] if r['id']==0 and r['status']=='done'),None)
            if root_verified:return
            from astrolabe.protein_beam_calibration import calibration_enabled, calibrate_root, verify_calibrated_state
            if calibration_enabled(cfg):
                root = verify_calibrated_state(cfg, output, state)
                if root is None:return
                try:
                    calibrate_root(cfg, output, state, root)
                except Exception as exc:
                    state.update(status='interrupted', error=str(exc))
                    atomic_json(checkpoint, state)
                    raise
                root_verified=True
                return
            if not reference or root is None:return
            from astrolabe.protein_beam_root_parity import root_parity
            evidence=root_parity(root,str(reference))
            atomic_json(output/'root_parity.json',evidence)
            if not evidence['passed']:
                state.update(status='interrupted',error='Native root context parity failed before proposal')
                atomic_json(checkpoint,state)
                raise ValueError(state['error'])
            root_verified=True
        if checkpoint.exists():
            if not resume:
                raise FileExistsError("Beam state exists; use --resume without changing its scientific config")
            state = json.loads(checkpoint.read_text())
            previous_service_socket = state.get("service_socket")
            if state["status"] == "complete":
                from astrolabe.protein_beam_recovery import validate_completed_beam
                state = validate_completed_beam(
                    cfg,
                    output,
                    state,
                    require_exports=False,
                )
                atomic_json(checkpoint, state)
                export_results(state, output, settings, holes_ranking)
                return state
            if state["config_hash"] != digest:
                raise ValueError("Scientific config changed on resume")
            # Reconcile completed native results; explicit retry requires the old owner to have ended.
            for record in state["records"]:
                if record["status"] == "running":
                    result_path = output/"candidates"/f"{record['id']:06d}"/"result.json"
                    if not result_path.exists():
                        if not retry_interrupted:
                            raise RuntimeError("Interrupted inference lacks a terminal result; inspect owner then use explicit --retry-interrupted")
                        previous_socket = state.get("service_socket")
                        if previous_socket and Path(previous_socket).exists():
                            raise RuntimeError("Previous AF3 owner socket still exists; drain/verify that owner before retrying")
                        record.setdefault("attempt_history", []).append(dict(attempt=record.get("attempt",1),status="interrupted",cost="retained in prior owner/worker logs; not assumed zero"))
                        record["attempt"] = record.get("attempt",1)+1
                        record["status"] = "pending"
                        record.pop('af3_call_token',None)
                        for task in state["pending"]:
                            if task["id"] == record["id"]:
                                task.pop('af3_call_token',None)
                                task.update(record)
                    else:
                        native = json.loads(result_path.read_text())
                        from astrolabe.protein_beam_calibration import calibration_enabled, validate_native_result
                        if calibration_enabled(cfg):
                            validate_native_result(record, native, output)
                        record.update(native)
            state["service_socket"] = socket_path
        else:
            if any(output.iterdir()) and set(p.name for p in output.iterdir()) != {"owner.lock"}:
                raise FileExistsError("Refusing to mix beam output with an existing directory")
            sequence = "".join(str(cfg.seqs.A).replace("+", "").split())
            state = dict(config_hash=digest, scientific_contract=contract, service_socket=socket_path,
                         status="running", generation=0, proposals=0,
                         records=[], beam=[], pending=[], seen=[], started_at=time.time())
            state["pending"] = [dict(id=0, kind="evaluate", generation=0, parent_id=None,
                                     sequence=sequence, seed=seed, status="pending")]
            if settings.get('initial_state'):
                from astrolabe.protein_beam_seeds import load_root
                root=load_root(cfg,str(settings.initial_state),contract)
                root.update(kind='rescore',status='pending')
                root['total_mutations']=0
                state.update(records=[],pending=[root])
            (output/"config.yaml").write_text(replay_config_yaml)
            atomic_json(checkpoint, state)
        previous_service_socket = locals().get("previous_service_socket", state.get("service_socket"))
        verify_root_context(state)
        from astrolabe.protein_beam_calibration import calibration_enabled
        if calibration_enabled(cfg) and root_verified and not state.get('generation_work') and not state['pending']:
            mark_candidates(state)
            state['beam'] = selection(state['records'], width)
            atomic_json(checkpoint, state)
        # Both isolated and serial native refinements use the same branching
        # policy and the same coordinator drive. Only scientific owner callbacks
        # and the selected execution transport differ.
        from dataclasses import replace
        from astrolabe import Astrolabe
        from astrolabe.design import Runtime
        from astrolabe.design.branch_owner import NativeBranchTrial
        from astrolabe.design.branch_types import (
            BranchAction, BranchEvent, BranchJob, BranchMember, BranchProposal,
            BranchingState, branch_codec,
        )
        from astrolabe.design.branching import BranchingPolicy
        from astrolabe.design.composition import ControlRun
        from astrolabe.design.store import JSONCodec
        from astrolabe.design.types import Key, key_for
        record_codec = JSONCodec("astrolabe.native-record/v1")
        atomic_json(checkpoint, state)
        with Runtime(root=output/".framework", scientific_contract=Key(digest)) as runtime:
            data = runtime.data
            data.register(record_codec)
            context = key_for(("native-comparison/v1", digest))

            def member(raw):
                return BranchMember(int(raw["id"]), str(raw["sequence"]),
                    data.put(raw, record_codec), float(raw["score"]),
                    bool(raw.get("eligible", True)), context)

            def select_members(members, capacity):
                values = [data.get(item.value, record_codec) for item in members]
                selected = selection(values, capacity)
                by_id = {item.id: item for item in members}
                return tuple(by_id[int(value["id"])] for value in selected)

            def native_task(job):
                if job.phase == "propose":
                    parent = data.get(job.parent.value, record_codec)
                    task = dict(id=job.id, kind="proposal", sequence=parent["sequence"],
                        snapshot=parent["snapshot"], parent_id=job.parent.id,
                        generation=job.generation, seed=job.seed)
                    if parent.get("native_artifacts"):
                        from astrolabe.protein_beam_contract import verify_native_artifacts
                        verify_native_artifacts(parent["native_artifacts"])
                        task["parent_artifacts"] = parent["native_artifacts"]
                    return task
                if job.proposal is not None:
                    proposal = data.get(job.proposal.value, record_codec)
                    return dict(proposal, id=job.id, proposal_id=job.proposal.id,
                        kind="evaluate", generation=job.generation, seed=job.seed, status="pending")
                if job.source is None:
                    raise ValueError("Root job requires an immutable native input")
                return dict(data.get(job.source, record_codec))

            def native_result(job, raw):
                if job.phase == "propose":
                    return BranchProposal(job.id, job.parent.id, str(raw["sequence"]),
                        data.put(raw, record_codec), bool(raw.get("valid", True)),
                        tuple(raw.get("proposal_rejections", ())))
                if raw.get("status") != "done":
                    raise ValueError("Native evaluator returned a non-terminal record")
                if raw.get("native_artifacts"):
                    from astrolabe.protein_beam_contract import verify_native_artifacts
                    verify_native_artifacts(raw["native_artifacts"])
                state["records"] = [r for r in state["records"] if r["id"] != raw["id"]] + [raw]
                state["records"].sort(key=lambda record: record["id"])
                verify_root_context(state)
                mark_candidates(state)
                result = next(r for r in state["records"] if r["id"] == raw["id"])
                return member(result)

            def native_probe(job, task):
                directory = "proposals" if job.phase == "propose" else "candidates"
                result_file = output/directory/f"{job.id:06d}"/"result.json"
                if not result_file.is_file():
                    return None
                raw = json.loads(result_file.read_text())
                if job.phase == "evaluate" and calibration_enabled(cfg):
                    from astrolabe.protein_beam_calibration import validate_native_result
                    validate_native_result(task, raw, output)
                return raw

            def reserve_native(job, task):
                policy = settings.get("af3_call_budget")
                if job.phase != "evaluate" or not policy:
                    return task
                from astrolabe.protein_beam_budget import reserve_calls
                admitted, refused = reserve_calls(policy, [task])
                return admitted[0] if admitted else None

            def retry_native(job, task, receipt):
                terminal_cpu = job.phase == "propose" and receipt.status == "failed"
                if not terminal_cpu and not retry_interrupted:
                    return None
                previous_socket = previous_service_socket
                if not terminal_cpu and previous_socket and Path(previous_socket).exists():
                    raise RuntimeError("Previous AF3 owner socket still exists; drain/verify that owner before retrying")
                previous_attempt = int(task.get("attempt", 1))
                updated = next((r for r in state["records"] if r["id"] == job.id), {})
                task.setdefault("attempt_history", []).append(dict(attempt=previous_attempt,
                    status=receipt.status, cost="retained; not assumed zero"))
                task["attempt"] = max(previous_attempt + 1, int(updated.get("attempt", 1)))
                task.pop("af3_call_token", None)
                return task

            def report_control(current, *, request):
                records = [data.get(item.value, record_codec) for item in current.records]
                pending = []
                for job in current.queued + current.pending:
                    if job.phase != "evaluate":
                        continue
                    admitted_task = trial.task_record(job)
                    task = native_task(job) if admitted_task is None else admitted_task
                    task = dict(task, status="running" if job in current.pending else "pending")
                    pending.append(task)
                    if job in current.pending:
                        records.append(task.copy())
                state.update(records=sorted(records, key=lambda value: value["id"]),
                    pending=pending, beam=[data.get(item.value, record_codec) for item in current.frontier],
                    generation=current.generation, proposals=current.proposals,
                    rejected_proposals=current.rejected,
                    duplicate_proposals=current.duplicates,
                    execution_profile="shared_branching", control_schema=1)
                if current.finished and str(settings.get("selection", "weighted")) == "pareto":
                    # Ranking annotations are a derived output view. They do
                    # not mutate immutable native observations or reselect parents.
                    annotated = selection(state["records"], width)
                    if [row["id"] for row in annotated] != [item.id for item in current.frontier]:
                        raise ValueError("Exported Pareto view differs from the committed frontier")
                    state["beam"] = annotated
                if current.parents and not current.finished:
                    before = sum(int(r.get("generation", 0)) < current.generation for r in records)
                    state["generation_work"] = dict(
                        parents=[data.get(item.value, record_codec) for item in current.parents],
                        counts={str(parent): count for parent, count in current.counts},
                        goal=min(budget-before, len(current.parents)*per_parent),
                        tasks=[native_task(job) for job in current.batch])
                else:
                    state.pop("generation_work", None)
                if current.refused:
                    state["unevaluated"] = [dict(native_task(job), status="af3_budget_exhausted")
                                            for job in current.refused]
                    state["af3_budget_exhausted"] = True
                if current.stop_reason:
                    reason_names = {"native_call_budget": "af3_call_budget",
                                    "proposal_cap": "distinct_sequence_proposal_cap"}
                    state["stop_reason"] = reason_names.get(current.stop_reason, current.stop_reason)
                atomic_json(checkpoint, state)

            policy = BranchingPolicy(width=width, proposals_per_parent=per_parent,
                max_evaluations=budget, proposal_cap=budget*10, seed=seed,
                evaluation_seed=seed, quota_mode="admitted", expansion="balanced",
                proposal_batch_size=proposal_batch_size, evaluation_width=in_flight,
                pipeline=bool(settings.get("pipeline_proposals", False)),
                selection=select_members, selection_identity=digest + "/selection",
                archive_selection=str(settings.get("selection", "weighted")) == "pareto",
                max_generations=settings.get("max_generations"))
            root_task = next(iter(state["pending"]), None)
            if root_task is None and state["records"]:
                root_task = dict(state["records"][0], status="pending")
            if root_task is None:
                raise ValueError("Native branching requires a root input")
            root_job = BranchJob("evaluate", 0, 0, seed, source=data.put(root_task, record_codec))
            initial = policy.start(root_job)
            with controlled_pool(on_abort, max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=worker_init, initargs=(worker_config_yaml, str(output), socket_path)) as pool:
                trial = NativeBranchTrial(pool, runtime, identity=key_for(("isolated-native/v1", digest)),
                    worker=worker_task, task_factory=native_task, result_factory=native_result,
                    probe=native_probe, reserve=reserve_native, retry=retry_native, report=report_control,
                    charge_identity=lambda task: task.get("af3_call_token"))
                try:
                    final = Astrolabe(ControlRun(policy, trial, initial, runtime,
                        branch_codec(BranchingState), branch_codec(BranchAction),
                        branch_codec(BranchEvent))).run()
                except BaseException as exc:
                    state.update(status="interrupted", error=repr(exc))
                    atomic_json(checkpoint, state)
                    raise
            report_control(final, request=None)
        state.update(status="complete", completed_at=time.time(), budget_complete=len(state["records"]) == budget)
        state["wall_s"] = state["completed_at"]-state["started_at"]
        atomic_json(checkpoint, state)
        export_results(state, output, settings, holes_ranking)
        return state


def export_results(state, output, settings, holes_ranking):
    """Idempotent CPU exports, including recovery after a complete checkpoint."""
    from astrolabe.protein_beam_policy import pareto_select
    output = Path(output)
    atomic_json(output/"final_beam.json", {k: state[k] for k in ("beam", "status", "budget_complete", "wall_s", "config_hash")})
    if str(settings.get('selection','weighted'))=='pareto':
        archive=pareto_select(copy.deepcopy(state['records']),len(state['records']),holes_ranking)
        atomic_json(output/'pareto_archive.json',[r for r in archive if r['selection']['pareto_front']==0])
    temporary = output/'scores.csv.tmp'
    with temporary.open("w") as handle:
        fields = ["id", "parent_id", "generation", "sequence", "score", *METRICS, "structure_path"]
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for r in state["records"]:
            writer.writerow({**{k:r.get(k) for k in fields if k not in METRICS}, **r["metrics"]})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output/'scores.csv')



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--service-socket", help="Existing AF3 owner socket; defaults to evaluator.alphafold3.service_socket")
    parser.add_argument("--workers", type=int, help="Override protein_beam.workers")
    parser.add_argument("--in-flight", type=int, help="Override protein_beam.in_flight")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-interrupted", action="store_true", help="Explicitly retry missing terminal candidates only after previous owner has ended")
    ns = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", GH_CUDA_PREINITIALIZE="0")
    os.environ.pop('JAX_PLATFORM_NAME', None)
    cfg = load_config(ns.config)
    from omegaconf import open_dict
    output = str(Path(ns.out).expanduser().resolve())
    with open_dict(cfg):
        # The quota's default ${out_dir}/af3_calls.json now follows the actual
        # CLI output, so a new run does not require a second path edit.
        cfg.out_dir = output
    settings = cfg.protein_beam
    configured_execution = None
    if cfg.get("search") is not None:
        evaluator_socket = cfg.evaluator.alphafold3.get("service_socket")
        execution = cfg.search.get("execution")
        configured_socket = (
            execution.get("socket_path")
            if execution is not None and hasattr(execution, "get")
            else None
        )
        fallback_socket = None
        if ns.service_socket is None and (
            configured_socket is None or not str(configured_socket).strip()
        ):
            fallback_socket = evaluator_socket
        try:
            configured_execution = _configured_branching_execution(
                cfg,
                socket_path=ns.service_socket or fallback_socket,
            )
        except ValueError as exc:
            if "socket_path" in str(exc):
                parser.error(str(exc))
            raise
    workers = (
        ns.workers
        if ns.workers is not None
        else (
            configured_execution["workers"]
            if configured_execution is not None
            else int(settings.get("workers", 3))
        )
    )
    in_flight = (
        ns.in_flight
        if ns.in_flight is not None
        else (
            configured_execution["in_flight"]
            if configured_execution is not None
            else int(settings.get("in_flight", 3))
        )
    )
    socket_path = (
        ns.service_socket
        or (
            configured_execution["socket_path"]
            if configured_execution is not None
            else None
        )
        or cfg.evaluator.alphafold3.get("service_socket")
    )
    if not socket_path:
        parser.error("Set evaluator.alphafold3.service_socket or pass --service-socket")
    resume = bool(ns.resume or (configured_execution and configured_execution["resume"]))
    retry_interrupted = bool(
        ns.retry_interrupted
        or (configured_execution and configured_execution["retry_interrupted"])
    )
    if retry_interrupted and not resume:
        parser.error("--retry-interrupted requires --resume")
    result = run_beam(cfg, output, str(socket_path), workers=workers, in_flight=in_flight,
                      resume=resume, retry_interrupted=retry_interrupted)
    print(json.dumps(dict(status=result["status"], evaluations=len(result["records"]),
                         budget_complete=result["budget_complete"], wall_s=result["wall_s"])))
    return 0 if result["budget_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
