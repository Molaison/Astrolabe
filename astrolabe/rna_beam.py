"""Scientific adapters for archived NA-MPNN/AlphaFold 3 RNA refinement.

Search control is the generic branching coordinator. This module validates
the archived RNA protocol, supplies native proposal/evaluation/measurement
callbacks, and projects completed branch records into the historical output
files. It contains no search loop or second budget/frontier owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
import copy
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from astrolabe.design.legacy import (
    capture_legacy_candidate,
    finalize_pending_mutations,
    native_sequence_key as _sequence_key,
    restore_native_sequence as _restore_sequence,
)
from astrolabe.design.repeats import RepeatRun
from astrolabe.design.owner import scientific_session
from astrolabe.design.store import DataclassCodec
from astrolabe.design.types import Blob, Key, key_for


@dataclass(frozen=True)
class _RNAPreparedCheckpoint:
    protocol: Key
    candidate: Blob
    rng_before: Blob
    rng_after: Blob


@dataclass(frozen=True)
class _BivalentRawOutput:
    identifier: str
    model_path: Path
    confidence_path: Path


_RNA_PREPARED_CODEC = DataclassCodec(
    _RNAPreparedCheckpoint,
    "astrolabe.rna-beam.prepared/v1",
    nested_types=(Blob, Key),
)


def _single_component(terms, expected_type: str) -> str:
    active = [(str(alias), term) for alias, term in terms.items() if term is not None]
    if len(active) != 1 or str(active[0][1].get("_type", "")).lower() != expected_type:
        raise ValueError(f"RNA beam search requires exactly one {expected_type} component")
    return active[0][0]


def _validate_sequence_statements(seqs) -> None:
    # The native RNA random-segment parser does not set initialized_from_random.
    for statement in seqs.values():
        for token in str(statement).split():
            segment = token.split("@", 1)[0].lstrip("+")
            if segment and set(segment) <= set("0123456789-+"):
                raise ValueError("RNA beam refinement requires a supplied candidate sequence, not random-length segments")


def _validate_shared_rna(protomers, segments, oligo: str | None = None) -> str:
    from astrolabe.core.protomer_config_parser import ProtomerTypes

    oligo = oligo or "".join(protomers)
    if oligo not in ("AAAA", "AAAAA", "ABCD", "ABCDE") or set(oligo) != set(protomers):
        raise ValueError("RNA oligomers require oligo: [AAAA]/[AAAAA], or shared protomers ABCD/ABCDE")
    first = protomers["A"]
    component_ids = [id(segments[name]) for name in first.components]
    if len(set(component_ids)) != len(component_ids):
        raise ValueError("This oligomer workflow requires interchain sharing, without intrachain repeats")
    for proto in protomers.values():
        if proto.type != ProtomerTypes.RNA or len(proto.sequence) != 87 or set(proto.sequence) - set("ACGU"):
            raise ValueError("Every oligomer chain must be an explicit, unmodified 87-nucleotide A/C/G/U RNA sequence")
        if [id(segments[name]) for name in proto.components] != component_ids:
            raise ValueError("All RNA chains must reference the same ordered segments (e.g. A: '<sequence>@shared'; B: shared)")
        if proto.ptm_dict or not proto.mutable_mask.all():
            raise ValueError("RNA oligomer refinement requires all 87 standard nucleotides to be mutable")
    return oligo


def _expand_repeated_protomer(oligo: str) -> str:
    """Map AF3's distinct chain IDs to native shared-segment protomer views.

    NAMPNN addresses physical chains by protomer name; repeating A directly
    would address only chain A of AF3's A/B/C/D[/E] structure.
    """
    if set(oligo) != {"A"}:
        return oligo
    from astrolabe.core import global_config
    from astrolabe.core.protomer_config_parser import Protomer

    physical_oligo = "ABCDE"[:len(oligo)]
    statement = " ".join(global_config.protomers["A"].components)
    for name in physical_oligo[1:]:
        proto = Protomer(name, statement, "rna")
        global_config.protomers[name] = proto
        global_config.mutable_protomers[name] = proto
    return physical_oligo


def _validate_bivalent_rna(protomers, segments, oligo: str) -> str:
    from astrolabe.core.protomer_config_parser import ProtomerTypes

    if oligo != "ABC" or list(protomers) != list("ABC"):
        raise ValueError("Bivalent RNA requires ordered RNA A and U1A proteins B/C, oligo: [ABC]")
    rna = protomers["A"]
    if rna.type != ProtomerTypes.RNA or len(rna.components) != 2:
        raise ValueError("RNA A must contain two references to one shared 20nt segment")
    first, second = (segments[name] for name in rna.components)
    if first is not second or len(first.sequence) != 20 or set(first.sequence) - set("ACGU"):
        raise ValueError("RNA A must contain two identical shared 20nt repeats, e.g. '20@repeat repeat'")
    if rna.ptm_dict or not rna.mutable_mask.all():
        raise ValueError("Both RNA repeats must be unmodified and mutable")
    for chain in ("B", "C"):
        proto = protomers[chain]
        if proto.type != ProtomerTypes.PROTEIN or proto.length != 96 or proto.mutable_mask.any():
            raise ValueError("Bivalent RNA requires fixed 96-residue U1A protein chains B/C")
    if protomers["B"].sequence != protomers["C"].sequence:
        raise ValueError("The two fixed U1A proteins must have identical sequences")
    return oligo


def _normalize_bivalent_sequence_statement(statement: str) -> str:
    """The native RNA random initializer expects a range, including fixed lengths."""
    tokens = statement.split()
    for index, token in enumerate(tokens):
        definition, *aliases = token.split("@")
        if definition == "20":
            tokens[index] = "@".join(["20-20", *aliases])
    return " ".join(tokens)


def validate_bivalent_objective_weights(objective_weights) -> dict[str, float]:
    keys = {"weak_interface", "repeat_contacts", "repeat_plddt", "clashes"}
    if objective_weights is None or set(objective_weights) != keys:
        raise ValueError("objective_weights must explicitly provide weak_interface, repeat_contacts, repeat_plddt and clashes")
    weights = {key: float(objective_weights[key]) for key in sorted(keys)}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("All objective weights must be finite and nonnegative; clashes is subtracted")
    return weights


def score_bivalent_rna(metrics: dict, objective_weights) -> float:
    """Archived contact/PAE objective; clash penalty saturates at one atom pair.

    Weight keys weak_interface/repeat_contacts/repeat_plddt multiply archived
    dual_interface/repeat_cohesion/weak_arm_plddt, respectively.
    """
    weights = validate_bivalent_objective_weights(objective_weights)
    values = {key: float(metrics[key]) for key in ("dual_interface", "repeat_cohesion", "weak_arm_plddt", "interchain_clashes")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Bivalent objective metrics must be finite")
    return (weights["weak_interface"] * values["dual_interface"]
            + weights["repeat_contacts"] * values["repeat_cohesion"]
            + weights["repeat_plddt"] * values["weak_arm_plddt"]
            - weights["clashes"] * min(values["interchain_clashes"], 1))


def proposal_seed(run_seed: int, proposal_step: int) -> int:
    return (run_seed * 100003 + proposal_step) % 2147483647 or 1


def _same_structure_representation(left: Path, right: Path) -> bool:
    """Compare native structures except for canonical empty dummy metadata."""

    left_bytes = left.read_bytes()
    right_bytes = right.read_bytes()
    suffixes = {left.suffix.lower(), right.suffix.lower()}
    if suffixes & {".cif", ".mmcif"}:
        if suffixes != {".cif", ".mmcif"} and (
            left.suffix.lower() not in {".cif", ".mmcif"}
            or right.suffix.lower() not in {".cif", ".mmcif"}
        ):
            return False
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict

        left_fields = dict(MMCIF2Dict(str(left)))
        right_fields = dict(MMCIF2Dict(str(right)))
        for fields, path in ((left_fields, left), (right_fields, right)):
            marker = fields.pop("_dummy.title", None)
            if marker not in (None, [""]):
                raise ValueError(
                    f"non-empty _dummy.title serialization marker in {path}: {marker!r}"
                )
        return left_bytes == right_bytes or left_fields == right_fields
    return left_bytes == right_bytes


def _resolve_bivalent_raw_output(
    model_path: Path,
    assembly: str,
    raw_model_results_dir: Path,
) -> _BivalentRawOutput:
    """Resolve a restored structure to one atom-identical AF3 raw run."""

    model_name = model_path.stem
    if model_name.endswith("_model"):
        model_name = model_name[: -len("_model")]
    if not model_name:
        raise ValueError("bivalent model path does not contain an output identifier")

    candidates = [model_name]
    if model_path.parent.name.startswith("measurement-input_"):
        safe_assembly = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            assembly,
        ).strip("._") or "assembly"
        transport_prefix = f"{safe_assembly}-"
        if model_name.startswith(transport_prefix):
            stripped = model_name[len(transport_prefix) :]
            if stripped and stripped not in candidates:
                candidates.append(stripped)

    matches = []
    refreshed = model_path
    raw_root = Path(raw_model_results_dir) / "alphafold3"
    for identifier in candidates:
        raw_model = raw_root / identifier / f"{identifier}_model.cif"
        raw_confidence = (
            raw_root / identifier / f"{identifier}_confidences.json"
        )
        if (
            raw_model.is_file()
            and raw_confidence.is_file()
            and _same_structure_representation(raw_model, refreshed)
        ):
            matches.append(
                _BivalentRawOutput(
                    identifier=identifier,
                    model_path=raw_model,
                    confidence_path=raw_confidence,
                )
            )

    if not matches:
        raise ValueError(
            "no content-matching AlphaFold 3 raw output for restored "
            f"{model_path.name!r}; candidates={candidates!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            "ambiguous AlphaFold 3 raw output for restored "
            f"{model_path.name!r}; candidates={[match.identifier for match in matches]!r}"
        )
    return matches[0]


def _measure_native_scores(
    *,
    scores,
    candidate_label: str,
    oligo: str,
    raw_model_results_dir: Path,
    objective_weights,
    bivalent: bool,
):
    """Run the complete RNA measurement body over restored native scores."""

    import shutil

    import numpy as np

    from astrolabe.analysis.rna_geometry import measure_bivalent_model
    from astrolabe.design.native_refinement import NativeMeasurement

    all_plddt = np.asarray(scores.plddt[oligo], dtype=float)
    metrics = {
        "mean_plddt": float(all_plddt.mean()),
        "rna_sequence": candidate_label,
    }
    score = metrics["mean_plddt"]
    if bivalent:
        model_path = Path(scores.complex_paths[oligo])
        raw_output = _resolve_bivalent_raw_output(
            model_path,
            oligo,
            raw_model_results_dir,
        )
        metrics.update(
            measure_bivalent_model(
                raw_output.model_path,
                raw_output.confidence_path,
            )
        )
        sidecar = model_path.with_name(
            f"{model_path.stem.removesuffix('_model')}_confidences.json"
        )
        shutil.copy2(raw_output.confidence_path, sidecar)
        metrics["confidence_path"] = str(sidecar)
        score = score_bivalent_rna(metrics, objective_weights)
    return NativeMeasurement(
        score=score,
        metrics=metrics,
        structure_path=str(scores.complex_paths[oligo]),
    )


def _propose_native_candidate(
    mutator,
    candidate,
    scores,
    *,
    seed: int,
    step: int,
    generation: int,
    parent_id: int,
    proposal_index: int,
    assemblies: tuple[str, ...],
    state: str,
    data,
    label,
    diagnostics: dict | None = None,
):
    """Run one complete native mutation proposal and capture its candidate."""

    from astrolabe.argparser import set_random_seeds
    from astrolabe.core import global_config
    from astrolabe.design.native_refinement import NativeProposal

    set_random_seeds(seed)
    global_config.set_variable("step", step)
    mutator.runner.extra_args["seed"] = seed
    mutator.runner.job_name = (
        f"generation_{generation:04d}_parent_{parent_id:04d}"
        f"_proposal_{proposal_index:02d}"
    )
    if diagnostics is not None:
        diagnostics["before_sequences"] = {
            name: segment.sequence
            for name, segment in global_config.segments.items()
        }
    mutator.mutate(global_config.protomers, scores, {})
    if diagnostics is not None:
        diagnostics["pending_mutations"] = {
            name: list(segment.mutations or ())
            for name, segment in global_config.segments.items()
            if segment.mutations
        }
    finalize_pending_mutations()
    if diagnostics is not None:
        diagnostics["after_sequences"] = {
            name: segment.sequence
            for name, segment in global_config.segments.items()
        }
    child = capture_legacy_candidate(
        assemblies=assemblies,
        state=state,
        data=data,
    ).candidate
    return NativeProposal(child, label(child))


def beam_expansion_order(parents, proposals_per_parent):
    """Archived round-robin order keeps a partial final generation balanced."""
    return [(parent, index) for index in range(proposals_per_parent) for parent in parents]


def _rna_scientific_config(
    args,
    *,
    seed: int,
    beam_width: int,
    proposals_per_parent: int,
    max_evaluations: int,
    objective_weights,
    initialization: str,
) -> dict:
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(args, resolve=False)
    if not isinstance(raw, dict):
        raise TypeError("RNA execution config must resolve to a mapping")
    for name in (
        "beam_search",
        "debug",
        "logging",
        "out_dir",
        "raw_model_results_dir",
        "repeat_index",
        "repeat_plan",
        "repeats",
        "resume",
        "search",
        "summary_format",
        "task_name",
    ):
        raw.pop(name, None)
    raw["seed"] = seed
    configured_search = args.get("search") or args.get("beam_search")
    proposal_request_budget = (
        configured_search.get("max_proposal_requests")
        if configured_search is not None
        else None
    )
    raw["search"] = {
        "algorithm": "branching",
        "owner": "archived_refinement",
        "width": beam_width,
        "proposals_per_parent": proposals_per_parent,
        "max_evaluations": max_evaluations,
        "max_proposal_requests": proposal_request_budget,
        "objective_weights": objective_weights,
        "initialization": initialization,
    }
    return raw


def _configured_path(value) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _selected_mapping(value) -> dict:
    from omegaconf import OmegaConf

    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=False)
    return dict(value) if isinstance(value, Mapping) else {}


def _rna_selection_projection(
    scientific_config,
    *,
    evaluator_alias: str,
    mutator_alias: str,
):
    """Project the real RNA config into the selected-component helper schema.

    This copy exists only while computing scientific and asset identities. It
    never replaces or writes the configuration used to construct native models.
    """
    from omegaconf import OmegaConf, open_dict

    projection = OmegaConf.create(copy.deepcopy(scientific_config))
    projection._set_flag("allow_objects", True)
    stages = _selected_mapping(projection.get("stages"))
    default = _selected_mapping(stages.get("default"))
    default["mutators"] = [mutator_alias]
    default["evaluator"] = evaluator_alias
    stages["default"] = default

    mutator = _selected_mapping(projection.get("mutator"))
    nampnn = _selected_mapping(mutator.get("NAMPNN"))
    if nampnn.get("checkpoint") in (None, "", "null", "None"):
        repo_path = _configured_path(nampnn.get("repo_path"))
        if repo_path is not None:
            # Materialize the runner's actual default checkpoint in the
            # identity-only projection. The real mutator config is unchanged.
            nampnn["checkpoint"] = str(
                repo_path / "models" / "design_model" / "s_19137.pt"
            )
    if nampnn:
        mutator["NAMPNN"] = nampnn

    with open_dict(projection):
        projection.stages = stages
        if mutator:
            projection.mutator = mutator
    return projection


def _rna_selected_scientific_identity(
    scientific_config,
    *,
    evaluator_alias: str,
    mutator_alias: str,
) -> Key:
    from astrolabe.design.configured_refinement import _scientific_config_payload

    payload = _scientific_config_payload(
        _rna_selection_projection(
            scientific_config,
            evaluator_alias=evaluator_alias,
            mutator_alias=mutator_alias,
        )
    )
    return key_for(("astrolabe.rna-beam.selected-config/v1", payload))


def _rna_resource_content(
    scientific_config,
    *,
    evaluator_alias: str,
    mutator_alias: str,
) -> Key:
    from astrolabe.design.configured_refinement import (
        _asset_manifest_identity,
        _selected_asset_manifest,
    )
    from astrolabe.design.portable_native_builder import content_manifest

    projection = _rna_selection_projection(
        scientific_config,
        evaluator_alias=evaluator_alias,
        mutator_alias=mutator_alias,
    )
    manifest = _selected_asset_manifest(projection)
    if not manifest.complete:
        details = [
            f"{item.role}: {item.detail}"
            for item in manifest.components
            if item.state == "unsupported"
        ]
        raise ValueError(
            "RNA model assets cannot be content-verified: "
            + "; ".join(details)
        )

    supplemental_paths: dict[str, Path] = {}
    named_files = _selected_mapping(scientific_config.get("named_files"))
    for role, value in named_files.items():
        path = _configured_path(value)
        if path is not None:
            supplemental_paths[f"inputs/named-files/{role}"] = path
    homologies = _selected_mapping(scientific_config.get("homologies"))
    for role, value in homologies.items():
        # Inline homology specifications are already covered by the complete
        # scientific-config identity. Only external path-like values add file
        # content to the resource identity.
        if isinstance(value, Mapping):
            continue
        path = _configured_path(value)
        if path is not None:
            supplemental_paths[f"inputs/homologies/{role}"] = path
    supplemental = content_manifest(supplemental_paths, prefix="asset")
    return key_for(
        (
            "astrolabe.rna-beam.resources/v2",
            _asset_manifest_identity(manifest),
            key_for(("astrolabe.rna-beam.inputs/v1", supplemental)),
        )
    )


def _rna_source_content():
    import importlib

    from astrolabe.design.portable_native_builder import source_manifest

    root = Path(__file__).parent
    design = root / "design"
    return source_manifest(
        importlib.import_module("run"),
        {
            "source/extra/rna_beam": Path(__file__),
            "source/extra/model_execution": root / "model_execution.py",
            "source/extra/runtime": design / "runtime.py",
            "source/extra/control_journal": design / "control_journal.py",
            "source/extra/branching": design / "branching.py",
            "source/extra/store": design / "store.py",
            "source/extra/legacy": design / "legacy.py",
            "source/extra/operations": design / "operations.py",
            "source/extra/native_support": design / "native_support.py",
            "source/extra/native_scores": design / "native_scores.py",
            "source/extra/native_refinement": design / "native_refinement.py",
            "source/extra/native_refinement_runtime": (
                design / "native_refinement_runtime.py"
            ),
            "source/extra/configured_refinement": (
                design / "configured_refinement.py"
            ),
            "source/components/mutator_registry": root / "mutators" / "base.py",
            "source/components/nampnn": root / "mutators" / "na_mpnn.py",
            "source/components/nampnn_runner": (
                root / "mutators" / "wrappers" / "na_mpnn_runner.py"
            ),
            "source/components/nampnn_daemon": (
                root / "mutators" / "wrappers" / "na_mpnn_daemon.py"
            ),
            "source/components/evaluator_registry": root / "evaluators" / "utils.py",
            "source/components/alphafold3": root / "evaluators" / "alphafold3.py",
            "source/components/alphafold3_wrapper": (
                root / "evaluators" / "wrappers" / "alphafold3.py"
            ),
            "source/core/protomer_config_parser": (
                root / "core" / "protomer_config_parser.py"
            ),
            "source/core/global_config": root / "core" / "__init__.py",
            "source/utils/base": root / "utils" / "base.py",
            "source/argparser": root / "argparser.py",
        },
    )


def _rna_protocol(
    args,
    *,
    seed: int,
    beam_width: int,
    proposals_per_parent: int,
    max_evaluations: int,
    objective_weights,
    initialization: str,
    evaluator_alias: str,
    mutator_alias: str,
    requested_oligo: str,
    oligo: str,
) -> Key:
    scientific_config = _rna_scientific_config(
        args,
        seed=seed,
        beam_width=beam_width,
        proposals_per_parent=proposals_per_parent,
        max_evaluations=max_evaluations,
        objective_weights=objective_weights,
        initialization=initialization,
    )
    config_identity = key_for(
        (
            "astrolabe.rna-beam.scientific-config/v2",
            scientific_config,
            _rna_selected_scientific_identity(
                scientific_config,
                evaluator_alias=evaluator_alias,
                mutator_alias=mutator_alias,
            ),
        )
    )
    source_identity = key_for(
        ("astrolabe.rna-beam.source/v1", _rna_source_content())
    )
    resource_identity = key_for(
        (
            "astrolabe.rna-beam.resources/v2",
            _rna_resource_content(
                scientific_config,
                evaluator_alias=evaluator_alias,
                mutator_alias=mutator_alias,
            ),
        )
    )
    return key_for(
        (
            "astrolabe.rna-beam/v2",
            config_identity,
            source_identity,
            resource_identity,
            seed,
            beam_width,
            proposals_per_parent,
            max_evaluations,
            objective_weights,
            initialization,
            requested_oligo,
            oligo,
        )
    )


def _write_table(path: Path, records: list[dict]) -> None:
    metric_names = sorted({key for row in records for key in row["metrics"]})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "evaluation_id", "proposal_step", "sequence", "generation", "parent_id", "status",
            "score", "structure_path", "entered_beam", "in_final_beam",
        ] + metric_names)
        writer.writeheader()
        for row in records:
            writer.writerow({**{key: value for key, value in row.items() if key != "metrics"}, **row["metrics"]})


def _validate_archived_search(cfg):
    """Validate an explicitly selected scientific adapter on every entry."""
    search = cfg.get("search")
    if search is None:
        return
    from astrolabe.configuration import selected_search_target

    if selected_search_target(cfg) != ("branching", "archived_refinement"):
        raise ValueError("archived refinement requires search.owner=archived_refinement")
    unknown = set(search) - {
        "algorithm", "owner", "policy", "width", "proposals_per_parent",
        "max_evaluations", "measurement", "initialization", "start",
        "archived_starts", "archived_start", "objective_weights", "summary_format",
        "resume", "max_proposal_requests",
    }
    if unknown:
        raise ValueError(f"unsupported archived search fields: {sorted(unknown)!r}")
    if cfg.get("beam_search"):
        raise ValueError("legacy beam_search conflicts with explicit search")
    summary_format = search.get("summary_format")
    if summary_format is not None and str(summary_format).strip().lower() != "json":
        raise ValueError("search.summary_format must be null or 'json'")
    resume = search.get("resume", False)
    if type(resume) is not bool:
        raise ValueError("search.resume must be boolean")
    proposal_request_budget = search.get("max_proposal_requests")
    if proposal_request_budget is not None and (
        type(proposal_request_budget) is not int
        or proposal_request_budget < 1
    ):
        raise ValueError(
            "search.max_proposal_requests must be a positive integer"
        )
    policy = search.get("policy")
    if policy is None:
        policy = {}
    if not hasattr(policy, "get"):
        raise TypeError("cfg.search.policy must be a mapping")
    controls = {"width", "proposals_per_parent", "max_evaluations"}
    if set(policy) - controls:
        raise ValueError("cfg.search.policy contains unsupported controls")
    for name in controls.intersection(search).intersection(policy):
        if search[name] != policy[name]:
            raise ValueError(f"conflicting cfg.search.{name} and policy.{name}")


def run_rna_beam(
    cfg, *, out_dir, seed: int, beam_width: int = 8,
    proposals_per_parent: int = 4, max_evaluations: int = 1000,
) -> dict:
    """Run the paper's 4/5-chain, 87-nt RNA oligomer refinement.

    ``cfg`` is a composed Astrolabe DictConfig with ``seqs``, ``types``,
    ``homologies`` and exactly one enabled NA-MPNN/AF3 term in their respective
    component maps. ``oligo: [AAAA]``/``[AAAAA]`` repeats candidate protomer A;
    alternatively, A defines a shared segment alias and B--D/E reuse it.
    The archived protocol uses five diffusion samples per evaluation: its
    pLDDT objective averages samples/residues while the retained structure is
    selected by AF3 ranking_score. These distinct aggregation rules are
    recorded in the summary. No random initializer, cached initial score or
    structure-start acceptance rule is used. ``out_dir`` is the exact new run
    directory, not its parent; it must not already exist.

    One AF3 model and one mutator wrapper are reused. The existing NA-MPNN
    wrapper launches its native subprocess for each proposal. Seeds for both
    proposals/native sampling use (seed*100003+proposal_step)%2147483647 or 1;
    AF3 uses the unchanged run seed for every sequence.

    Returns the JSON-serializable final beam and budget summary. Artifacts are
    01_evaluations.jsonl (append-only events, including failed calls),
    02_evaluations.csv (scores in [0,1], entry/final selection flags),
    03_final_beam.json, 04_config.yaml and native structures/raw_model_results.
    Predictor and proposal failures propagate unchanged, without replacement.
    """
    return _run_rna_scientific_composition(
        cfg,
        out_dir=out_dir,
        seed=seed,
        beam_width=beam_width,
        proposals_per_parent=proposals_per_parent,
        max_evaluations=max_evaluations,
        objective_weights=None,
    )


def run_bivalent_rna_beam(
    cfg, *, out_dir, seed: int, beam_width: int, proposals_per_parent: int,
    objective_weights, max_evaluations: int = 200,
) -> dict:
    """Optimize RNA A='20@repeat repeat' with two fixed 96-aa U1A chains B/C.

    The shared 20nt unit is initialized once and is the sequence
    deduplication key; every prediction contains both copies. Explicit starting
    repeat sequences are also supported. ``beam_search.initialization`` selects
    ``archived`` (archived_starts[archived_start]), ``random`` (seeded 20nt), or
    ``configured`` (use seqs.A). The original 40nt run used three contact-guided
    shortened candidates; random is a distinct manuscript-described protocol.
    The initial AF3 prediction counts
    toward the 200-evaluation budget. Independent starts belong to the caller.

    Archived width/proposals are 8/4 and weights are 2/1/0.5/0.5. The first
    three terms are PAE-based weaker-interface quality, repeat cohesion and
    lower repeat pLDDT; the clash term is min(interchain_clashes,1).

    Native configuration/lifecycle match ``run_rna_beam``. With the archived
    five samples, the objective uses the ranking-selected CIF and its own PAE
    confidence JSON, not the five-sample mean. The sampled ensemble mean is
    retained as a separately named diagnostic.
    The output contains raw metrics, weights, score and separate geometry
    review flags; geometry flags do not silently reject evaluated candidates.
    """
    weights = validate_bivalent_objective_weights(objective_weights)
    return _run_rna_scientific_composition(
        cfg,
        out_dir=out_dir,
        seed=seed,
        beam_width=beam_width,
        proposals_per_parent=proposals_per_parent,
        max_evaluations=max_evaluations,
        objective_weights=weights,
    )


@scientific_session
def _run_rna_scientific_composition(
    cfg, *, out_dir, seed, beam_width, proposals_per_parent, max_evaluations,
    objective_weights, persisted_config=None, resume: bool = False,
) -> dict:
    if type(resume) is not bool:
        raise TypeError("resume must be boolean")
    bivalent = objective_weights is not None
    if type(proposals_per_parent) is not int or proposals_per_parent < 1:
        raise ValueError("proposals_per_parent must be a positive integer")
    if type(seed) is not int or not 1 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in 1..2**32-1")

    _validate_archived_search(cfg)
    if cfg.get("search") is not None and persisted_config is None:
        from omegaconf import open_dict

        selected = copy.deepcopy(cfg)
        expected = "bivalent_geometry" if bivalent else "confidence"
        if selected.search.get("measurement") != expected:
            raise ValueError(f"selected measurement must be {expected!r}")
        # Explicit public API controls override the corresponding config values;
        # initialization and scientific options use the same configured adapter.
        with open_dict(selected):
            if "policy" in selected.search:
                del selected.search.policy
            selected.search.width = beam_width
            selected.search.proposals_per_parent = proposals_per_parent
            selected.search.max_evaluations = max_evaluations
            if bivalent:
                selected.search.objective_weights = objective_weights
        return run_configured_refinement(
            selected,
            RepeatRun(index=0, total=1, seed=seed, output=Path(out_dir)),
        )

    import numpy as np
    from omegaconf import OmegaConf, open_dict
    from astrolabe.argparser import set_random_seeds
    from astrolabe.core import global_config
    from astrolabe.design.branching import BranchingPolicy
    from astrolabe.design.legacy import LegacyContext, capture_legacy_candidate
    from astrolabe.design.native_refinement import (
        NativeMeasurement,
        NativeProposal,
        NativeRefinementSpec,
        run_native_refinement,
    )
    from astrolabe.design.runtime import ContractConflict, Runtime
    from astrolabe.design.store import DataStore, design_codec
    from astrolabe.evaluators.utils import construct_evaluators, _merge_conf_with_defaults
    from astrolabe.mutators.base import build_mutators
    from astrolabe.utils.base import construct_standard_input
    from run import initialize_global_config, setup_protomers

    args = copy.deepcopy(cfg)
    initialization = "configured"
    if bivalent:
        initialization = str(args.beam_search.get("initialization", "configured"))
        if initialization == "archived":
            start = args.beam_search.archived_starts[int(args.beam_search.get("archived_start", 0))]
            args.seqs.A = f"{start['sequence']}@repeat repeat"
        elif initialization == "random":
            args.seqs.A = "20@repeat repeat"
        elif initialization != "configured":
            raise ValueError("Bivalent initialization must be archived, random, or configured")
    _validate_sequence_statements({name: statement for name, statement in args.seqs.items()
                                   if not bivalent or name != "A"})
    evaluator_alias = _single_component(args.evaluator.evaluator_terms, "alphafold3")
    mutator_alias = _single_component(args.mutator.mutator_terms, "nampnn")
    af3_conf = _merge_conf_with_defaults(args.evaluator, "alphafold3",
                                       args.evaluator.evaluator_terms[evaluator_alias], alias=evaluator_alias)
    if int(af3_conf.num_seeds) != 1:
        raise ValueError("Archived RNA beam search uses exactly one fixed AF3 seed")
    if args.get("structure_start") is not None:
        raise ValueError("RNA beam search re-predicts its starting sequence; remove structure_start")

    output = Path(out_dir).expanduser().resolve()
    args._set_flag("allow_objects", True)
    with open_dict(args):
        args.out_dir = output
        args.raw_model_results_dir = output / "raw_model_results"
        args.seed = seed
        args.mutator.enabled_aliases = [mutator_alias]
        args.evaluator.enabled_aliases = [evaluator_alias]
        if bivalent:
            args.seqs.A = _normalize_bivalent_sequence_statement(str(args.seqs.A))
    oligos = args.get("oligo") or ["".join(str(name) for name in args.seqs)]
    if len(oligos) != 1:
        raise ValueError("RNA beam search refines exactly one configured oligomer")
    requested_oligo = str(oligos[0])
    oligo = (
        "ABCDE"[: len(requested_oligo)]
        if set(requested_oligo) == {"A"}
        else requested_oligo
    )
    protocol = _rna_protocol(
        args,
        seed=seed,
        beam_width=beam_width,
        proposals_per_parent=proposals_per_parent,
        max_evaluations=max_evaluations,
        objective_weights=objective_weights,
        initialization=initialization,
        evaluator_alias=evaluator_alias,
        mutator_alias=mutator_alias,
        requested_oligo=requested_oligo,
        oligo=oligo,
    )

    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"RNA resume root does not exist: {output}")
        required = (
            output / "01_evaluations.jsonl",
            output / "04_config.yaml",
            output / ".framework" / "journal.sqlite",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "RNA resume requires prior durable run files: "
                + ", ".join(str(path) for path in missing)
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    args.raw_model_results_dir.mkdir(exist_ok=resume)
    framework = output / ".framework"
    data = DataStore(framework / "data")
    data.register(_RNA_PREPARED_CODEC)
    runtime = Runtime(
        root=framework,
        data=data,
        scientific_contract=protocol,
    )

    if persisted_config is None:
        persisted_config = copy.deepcopy(args)
        legacy_search = args.get("beam_search", {})
        with open_dict(persisted_config):
            persisted_config.search = {
                "algorithm": "branching",
                "owner": "archived_refinement",
                "width": beam_width,
                "proposals_per_parent": proposals_per_parent,
                "max_evaluations": max_evaluations,
                "measurement": "bivalent_geometry" if bivalent else "confidence",
                "initialization": initialization,
                "resume": resume,
            }
            if bivalent:
                persisted_config.search.objective_weights = objective_weights
                if initialization == "archived":
                    persisted_config.search.start = legacy_search.archived_starts[
                        int(legacy_search.get("archived_start", 0))
                    ]
            for name in ("beam_search", "repeat_plan"):
                if name in persisted_config:
                    del persisted_config[name]
            persisted_config.repeats = 1
            persisted_config.out_dir = str(output)
            persisted_config.raw_model_results_dir = str(args.raw_model_results_dir)
    if not resume:
        OmegaConf.save(persisted_config, output / "04_config.yaml")

    prepared = runtime.load("rna/prepared", _RNA_PREPARED_CODEC)
    if resume and prepared is None:
        raise ContractConflict(
            "RNA durable resume has no prepared checkpoint"
        )
    if prepared is not None and prepared.protocol != protocol:
        raise ContractConflict(
            "RNA scientific config, source or model assets changed on resume"
        )

    global_config.reset()
    initialize_global_config(args)
    set_random_seeds(seed)
    rng_before = runtime._capture_rng()
    if prepared is not None:
        runtime._restore_rng(prepared.rng_before)
    setup_protomers(args)
    if bivalent:
        validated_requested = _validate_bivalent_rna(
            global_config.protomers,
            global_config.segments,
            requested_oligo,
        )
        validated_oligo = validated_requested
    else:
        validated_requested = _validate_shared_rna(
            global_config.protomers,
            global_config.segments,
            requested_oligo,
        )
        validated_oligo = _expand_repeated_protomer(validated_requested)
    if requested_oligo != validated_requested or oligo != validated_oligo:
        raise ValueError("RNA structure-chain mapping changed during preparation")
    protomer = global_config.protomers["A"]
    initial_sequence = _sequence_key(protomer, global_config.segments)
    global_config.init_const("num_tokens", {oligo: sum(global_config.protomers[name].length for name in oligo)})
    global_config.init_const("structure_start", None)
    global_config.set_variable("ignored_complexes", [])
    global_config.set_variable("negative_design_complexes", [])
    global_config.set_variable("mutation_num", sum(proto.mutable_length for proto in global_config.protomers.values()))
    context = LegacyContext()
    translation = context.capture_candidate(assemblies=(oligo,), data=data)
    if prepared is not None:
        saved_candidate = data.get(prepared.candidate, design_codec)
        if saved_candidate != translation.candidate:
            raise ContractConflict(
                "RNA prepared candidate differs from deterministic replay"
            )
        initial_candidate = saved_candidate
        runtime._restore_rng(prepared.rng_after)
    else:
        initial_candidate = translation.candidate
        candidate_ref = data.put(initial_candidate, design_codec)
        runtime.save(
            "rna/prepared",
            _RNAPreparedCheckpoint(
                protocol,
                candidate_ref,
                rng_before,
                runtime._capture_rng(),
            ),
            _RNA_PREPARED_CODEC,
        )
    aliases = {
        alias: variable
        for variable in translation.variables
        for alias in variable.aliases
    }

    def label_candidate(candidate) -> str:
        assignments = {
            assignment.variable: assignment.value
            for assignment in candidate.assignments
        }
        seen = set()
        tokens = []
        for component in protomer.components:
            variable = aliases[str(component)]
            if variable.identity in seen:
                continue
            seen.add(variable.identity)
            value = assignments[variable.identity]
            polymer_tokens = getattr(value, "tokens", None)
            if polymer_tokens is None:
                raise ValueError("RNA beam candidate contains a non-polymer variable")
            tokens.extend(polymer_tokens)
        return "".join(tokens)

    def load_mutator():
        value = build_mutators(args.mutator, args.raw_model_results_dir)[mutator_alias]
        if value.position_selector not in (None, "None") or value.max_mutations_per_proto is not None:
            raise ValueError("RNA proposals require all mutable positions; remove selector and mutation limits")
        value.runner.batch_size = 1
        value.runner.number_of_batches = 1
        return value

    def load_evaluator():
        return construct_evaluators(
            args.evaluator,
            args.raw_model_results_dir,
            [oligo],
            enabled_aliases=[evaluator_alias],
        )[evaluator_alias]

    def propose(
        mutator,
        candidate,
        scores,
        *,
        seed,
        step,
        generation,
        parent_id,
        proposal_index,
        output,
    ):
        return _propose_native_candidate(
            mutator,
            candidate,
            scores,
            seed=seed,
            step=step,
            generation=generation,
            parent_id=parent_id,
            proposal_index=proposal_index,
            assemblies=(oligo,),
            state=translation.state,
            data=data,
            label=label_candidate,
        )

    def evaluate(
        evaluator,
        candidate,
        *,
        seed,
        step,
        assemblies,
        output,
        proposal_id,
    ):
        global_config.set_variable("step", step)
        return evaluator.evaluate(
            construct_standard_input(list(assemblies), seed),
            {},
        )

    def measure(candidate, scores, job):
        return _measure_native_scores(
            scores=scores,
            candidate_label=label_candidate(candidate),
            oligo=oligo,
            raw_model_results_dir=args.raw_model_results_dir,
            objective_weights=objective_weights,
            bivalent=bivalent,
        )

    proposal_request_budget = args.get("beam_search", {}).get(
        "max_proposal_requests"
    )
    if proposal_request_budget is not None and (
        type(proposal_request_budget) is not int
        or proposal_request_budget < 1
    ):
        raise ValueError(
            "beam_search.max_proposal_requests must be a positive integer"
        )
    policy = BranchingPolicy(
        width=beam_width,
        proposals_per_parent=proposals_per_parent,
        max_evaluations=max_evaluations,
        proposal_cap=10 * max_evaluations,
        proposal_request_budget=proposal_request_budget,
        seed=seed,
        evaluation_seed=seed,
        proposal_seed=proposal_seed,
        quota_mode="attempts",
        expansion="round_robin",
        proposal_batch_size=1,
        evaluation_width=1,
        pipeline=False,
        immediate_evaluation=True,
        cap_error=True,
    )

    records: list[dict] = []
    journal_path = output / "01_evaluations.jsonl"
    prior_events = []
    if resume:
        prior_events = [
            json.loads(line)
            for line in journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    prior_event_names = {str(item.get("event")) for item in prior_events}
    with ExitStack() as stack:
        stack.callback(runtime.close)
        journal = stack.enter_context(
            journal_path.open("a" if resume else "x", encoding="utf-8")
        )
        def event(
            kind: str,
            fields: dict | None = None,
            **data,
        ) -> None:
            payload = {**(fields or {}), **data}
            journal.write(json.dumps({"event": kind, **payload}, allow_nan=False) + "\n")
            journal.flush()

        owner = None
        try:
            if "search_started" not in prior_event_names:
                event("search_started", seed=seed, beam_width=beam_width,
                      proposals_per_parent=proposals_per_parent, max_evaluations=max_evaluations,
                      oligo=requested_oligo, structure_chains=oligo, initial_sequence=initial_sequence,
                      workflow="bivalent_rna" if bivalent else "rna_oligomer", objective_weights=objective_weights)
            spec = NativeRefinementSpec(
                output=output,
                seed=seed,
                width=beam_width,
                proposals_per_parent=proposals_per_parent,
                max_evaluations=max_evaluations,
                evaluator_alias=evaluator_alias,
                mutator_alias=mutator_alias,
                assemblies=(oligo,),
                initial_candidate=initial_candidate,
                context=context,
                translation=translation,
                data=data,
                load_mutator=load_mutator,
                load_evaluator=load_evaluator,
                propose=propose,
                evaluate=evaluate,
                measure=measure,
                label=label_candidate,
                proposal_seed=proposal_seed,
                on_event=event,
            )
            run = run_native_refinement(
                spec,
                policy=policy,
                protocol=protocol,
                runtime=runtime,
            )
            owner = run.owner
            selected_ids = {member.id for member in run.selected}
            member_rows = []
            for evaluation_id, member in run.member_items():
                metrics = run.metrics(member)
                member_rows.append(
                    {
                        "evaluation_id": evaluation_id,
                        "proposal_step": member.proposal_step,
                        "sequence": member.label,
                        "generation": member.generation,
                        "parent_id": member.parent_id,
                        "status": "completed",
                        "score": member.score,
                        "structure_path": member.structure_path,
                        "entered_beam": evaluation_id in selected_ids,
                        "in_final_beam": evaluation_id in selected_ids,
                        "metrics": metrics,
                    }
                )
            records = member_rows
            _write_table(output / "02_evaluations.csv", records)
            state = run.state
            duplicates = int(getattr(state, "duplicates", 0))
            stop_reason = getattr(state, "stop_reason", None) or "budget_exhausted"
            selected = [
                {
                    "evaluation_id": member.id,
                    "sequence": record.label,
                    "score": member.score,
                    "structure_path": record.structure_path,
                    "generation": record.generation,
                    "parent_id": record.parent_id,
                    "metrics": run.metrics(record),
                }
                for member, record in zip(
                    run.selected,
                    run.members(),
                    strict=True,
                )
            ]
            if "search_completed" not in prior_event_names:
                event(
                    "search_completed",
                    evaluations=len(records),
                    proposals=run.proposals,
                    duplicate_proposals=duplicates,
                    budget_complete=len(records) >= max_evaluations,
                )
            result = {"out_dir": str(output), "seed": seed, "oligo": requested_oligo,
                      "structure_chains": oligo,
                      "beam_width": beam_width, "proposals_per_parent": proposals_per_parent,
                      "max_evaluations": max_evaluations, "evaluations": len(records),
                      "proposals": run.proposals, "duplicate_proposals": duplicates,
                      "stop_reason": stop_reason, "budget_complete": len(records) >= max_evaluations,
                      "score": "weighted_bivalent_score" if bivalent else "mean_plddt",
                      "score_scale": None if bivalent else [0, 1], "objective_weights": objective_weights,
                      "sequence_key_length": len(initial_sequence),
                      "initialization": initialization,
                      "af3_samples_per_evaluation": int(af3_conf.num_samples),
                      "score_aggregation": "ranking-selected sample contact/PAE objective" if bivalent else "mean pLDDT over all diffusion samples and residues",
                      "structure_selection": "AF3 ranking_score",
                      "proposal_order": "round_robin", "proposal_cap": 10 * max_evaluations,
                      "proposal_request_budget": proposal_request_budget,
                      "beam": selected}
            (output / "03_final_beam.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            if not selected:
                raise RuntimeError("generic branching search returned no selected members")
            _restore_sequence(selected[0]["sequence"], protomer, global_config.segments)
            return result
        except BaseException as exc:
            event(
                "search_failed",
                evaluations=len(records),
                budget_complete=False,
                error=repr(exc),
            )
            raise
        finally:
            _write_table(output / "02_evaluations.csv", records)
            if owner is not None:
                owner.shutdown()


def run_configured_refinement(cfg, repeat: RepeatRun) -> dict:
    """Configured assembly adapter from canonical search fields to RNA science.

    Routing/configuration owns selection of this factory. This function only
    translates the fixed canonical RNA refinement fields into the existing
    scientific adapter and invokes the same generic ``NativeRefinementSpec`` +
    ``BranchingPolicy`` path as the legacy public entry points.
    """

    if not isinstance(repeat, RepeatRun):
        raise TypeError("repeat must be a design.repeats.RepeatRun")
    if cfg is None or not hasattr(cfg, "get"):
        raise TypeError("cfg must be a composed configuration")
    search = cfg.get("search")
    if search is None:
        raise ValueError("configured refinement requires cfg.search")
    _validate_archived_search(cfg)

    policy = search.get("policy", {})
    if policy is None:
        policy = {}
    width = search.get("width", policy.get("width"))
    proposals_per_parent = search.get(
        "proposals_per_parent",
        policy.get("proposals_per_parent"),
    )
    max_evaluations = search.get(
        "max_evaluations",
        policy.get("max_evaluations"),
    )
    proposal_request_budget = search.get("max_proposal_requests")
    for name, value in (
        ("width", width),
        ("proposals_per_parent", proposals_per_parent),
        ("max_evaluations", max_evaluations),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"cfg.search {name} must be a positive integer")

    measurement = str(search.get("measurement", "")).strip().lower()
    initialization = str(search.get("initialization", "configured")).strip().lower()
    if initialization not in {"configured", "random", "archived"}:
        raise ValueError(
            "cfg.search.initialization must be configured, random, or archived"
        )

    translated = copy.deepcopy(cfg)
    legacy_search = {"initialization": initialization}
    if proposal_request_budget is not None:
        legacy_search["max_proposal_requests"] = proposal_request_budget
    start = repeat.start if repeat.start is not None else search.get("start")
    archived_starts = search.get("archived_starts")
    archived_start = search.get("archived_start", 0)
    if start is not None and initialization != "archived":
        raise ValueError("cfg.search.start is valid only for archived initialization")
    if start is not None:
        if isinstance(start, str):
            archived_starts = [{"sequence": start}]
        elif hasattr(start, "items"):
            record = dict(start)
            if "sequence" not in record and "seq" in record:
                record["sequence"] = record["seq"]
            if not isinstance(record.get("sequence"), str) or not record["sequence"]:
                raise ValueError("cfg.search.start mapping requires a sequence")
            archived_starts = [record]
        else:
            raise TypeError("cfg.search.start must be a sequence string or mapping")
        archived_start = 0
    if archived_starts is not None:
        if isinstance(archived_starts, str):
            archived_starts = [{"sequence": archived_starts}]
            archived_start = 0
        try:
            items = list(archived_starts)
        except TypeError as error:
            raise TypeError("cfg.search.archived_starts must be a sequence") from error
        if not items:
            raise ValueError("cfg.search.archived_starts must not be empty")
        archived_starts = []
        for item in items:
            if not hasattr(item, "items"):
                raise TypeError(
                    "cfg.search.archived_starts entries must be mappings"
                )
            record = dict(item)
            if "sequence" not in record and "seq" in record:
                record["sequence"] = record["seq"]
            if not isinstance(record.get("sequence"), str) or not record["sequence"]:
                raise ValueError(
                    "cfg.search.archived_starts entries require a sequence"
                )
            archived_starts.append(record)
        legacy_search["archived_starts"] = archived_starts
        legacy_search["archived_start"] = archived_start

    from omegaconf import OmegaConf, open_dict

    with open_dict(translated):
        if "search" in translated:
            del translated.search
        translated.beam_search = OmegaConf.create(legacy_search)
        if repeat.initial_sequence is not None:
            target = str(repeat.initial_sequence.get("target", ""))
            value = repeat.initial_sequence.get("value")
            if not target or not isinstance(value, str):
                raise ValueError(
                    "RepeatRun.initial_sequence requires target and string value"
                )
            if translated.get("seqs") is None:
                translated.seqs = {}
            translated.seqs[target] = value

    if measurement == "confidence":
        if initialization != "configured":
            raise ValueError(
                "RNA confidence refinement supports configured initialization only"
            )
        objective_weights = None
    elif measurement == "bivalent_geometry":
        if initialization == "archived" and archived_starts is None:
            raise ValueError(
                "archived bivalent_geometry refinement requires start or archived_starts"
            )
        objective_weights = validate_bivalent_objective_weights(
            search.get("objective_weights")
        )
    else:
        raise ValueError(
            "cfg.search.measurement must be confidence or bivalent_geometry"
        )

    output = repeat.output
    if output is None:
        raise ValueError("RepeatRun.output is required for configured refinement")
    resume = search.get("resume", False)
    if type(resume) is not bool:
        raise ValueError("search.resume must be boolean")
    if resume and (repeat.total != 1 or repeat.index != 0):
        raise ValueError("search.resume=true requires exactly one repeat")

    persisted = copy.deepcopy(cfg)
    selected_start = None
    if initialization == "archived":
        if start is not None:
            selected_start = dict(archived_starts[0])
        elif archived_starts is not None:
            selected_start = dict(archived_starts[int(archived_start)])
    with open_dict(persisted):
        persisted.seed = repeat.seed
        persisted.repeats = 1
        persisted.repeat_index = repeat.index
        persisted.out_dir = str(output)
        if "repeat_plan" in persisted:
            del persisted.repeat_plan
        persisted.search.algorithm = str(
            persisted.search.get("algorithm", "branching")
        )
        persisted.search.owner = str(
            persisted.search.get("owner", "archived_refinement")
        )
        if selected_start is not None:
            persisted.search.start = selected_start
            persisted.search.archived_start = 0
            persisted.search.archived_starts = [selected_start]
        if repeat.initial_sequence is not None:
            target = str(repeat.initial_sequence.get("target", ""))
            value = repeat.initial_sequence.get("value")
            if not target or not isinstance(value, str):
                raise ValueError(
                    "RepeatRun.initial_sequence requires target and string value"
                )
            if persisted.get("seqs") is None:
                persisted.seqs = {}
            persisted.seqs[target] = value
    return _run_rna_scientific_composition(
        translated,
        out_dir=output,
        seed=repeat.seed,
        beam_width=width,
        proposals_per_parent=proposals_per_parent,
        max_evaluations=max_evaluations,
        objective_weights=objective_weights,
        persisted_config=persisted,
        resume=resume,
    )
