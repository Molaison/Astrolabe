"""Binder-internal histidine network terms for neutral and acid-release design.

This module deliberately keeps PyRosetta imports lazy.  It reuses the pose,
hydrogen-bond, and chain-selector conventions from :mod:`ph_sensitive`, but only
evaluates binder-internal networks.

Neutral-histidine protocol
--------------------------
AF3 histidines are read from the current complex structure.  Standard ``HIS``
and ``HIS_D`` residues are treated as neutral and are not repacked or sampled
between tautomers.  A charged ``HIS_P`` residue is converted to neutral ``HIS``
without refolding; common heavy-atom coordinates are restored after the type
conversion.  Cys residues and covalent connectivity are never rewritten by this
module.

Release-state scoring delegates the two protonation states to the shared
Proton-PottsMPNN runner.  This module never imports the isolated Proton runtime.
"""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from astrolabe.core.chain_resolution import resolve_selector_chain_ids
from astrolabe.losses.base import loss_registry
from astrolabe.losses.ph_sensitive import (
    _chain_residues,
    _hbonds,
    _load_pose,
    _pyrosetta,
)


_METRICS = {"neutral", "release", "unpaired"}
_HIS_TYPE_NAMES = {"HIS", "HIS_D"}
_CATION_NAMES = {"HIS", "LYS", "ARG"}
_RING_ATOMS = {"ND1", "NE2"}
_AUDIT_ATTR = "ph_core_network_audits"
_NETWORK_CACHE: dict[
    tuple[str, str, str, tuple[float, float, float, float]],
    dict[str, Any],
] = {}


@dataclass(frozen=True)
class _NetworkParameters:
    hbond_scale: float
    depth_scale: float
    probe_radius: float
    resolution: float

    @property
    def cache_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.hbond_scale,
            self.depth_scale,
            self.probe_radius,
            self.resolution,
        )


@dataclass(frozen=True)
class _ReleaseParameters:
    gap_scale: float
    gap_reference: float
    runner_key: str
    runner_identity: str

    @property
    def cache_tuple(self) -> tuple[float, float, str, str]:
        return (
            self.gap_scale,
            self.gap_reference,
            self.runner_key,
            self.runner_identity,
        )


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _network_parameters(
    *,
    hbond_scale: float,
    depth_scale: float,
    probe_radius: float,
    resolution: float,
) -> _NetworkParameters:
    return _NetworkParameters(
        hbond_scale=_finite_positive("hbond_scale", hbond_scale),
        depth_scale=_finite_positive("depth_scale", depth_scale),
        probe_radius=_finite_positive("probe_radius", probe_radius),
        resolution=_finite_positive("resolution", resolution),
    )


def _plain_config(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    if hasattr(config, "_metadata") and hasattr(config, "items"):
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(config):
                return dict(OmegaConf.to_container(config, resolve=True))
        except ImportError:
            pass
    if hasattr(config, "items") and not isinstance(config, Mapping):
        try:
            return {str(key): config[key] for key in config}
        except Exception:
            pass
    if isinstance(config, Mapping):
        return dict(config)
    raise ValueError("runner_config must be a mapping or null.")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def _runtime_identity(runner_key: str, runner_config: Any) -> tuple[str, str]:
    config = _plain_config(runner_config)
    digest = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    return f"{runner_key}:{digest}", digest


def _release_parameters(
    *,
    gap_scale: float,
    gap_reference: float,
    runner_key: str,
    runner_config: Any,
) -> _ReleaseParameters:
    gap_scale = _finite_positive("gap_scale", gap_scale)
    gap_reference = _finite("gap_reference", gap_reference)
    runner_key = str(runner_key or "").strip()
    if not runner_key:
        raise ValueError("runner_key must be nonempty.")
    runtime_identity, _digest = _runtime_identity(runner_key, runner_config)
    return _ReleaseParameters(
        gap_scale=gap_scale,
        gap_reference=gap_reference,
        runner_key=runner_key,
        runner_identity=runtime_identity,
    )


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _neutralize_charged_histidines(pose):
    """Return a coordinate-preserving neutral-HIS clone.

    Only charged histidine residue types are changed.  Standard neutral HIS
    tautomer types are used exactly as loaded from AF3, without rotamer or
    tautomer sampling.
    """

    pr = _pyrosetta()
    neutral = pose.clone()
    for index in range(1, neutral.total_residue() + 1):
        residue = neutral.residue(index)
        if not str(residue.name3()).startswith("HIS"):
            continue
        if residue.type().base_name() in _HIS_TYPE_NAMES:
            continue
        heavy_coordinates = {
            residue.atom_name(atom).strip(): tuple(residue.xyz(atom))
            for atom in range(1, residue.nheavyatoms() + 1)
        }
        pr.rosetta.protocols.simple_moves.MutateResidue(index, "HIS").apply(neutral)
        converted = neutral.residue(index)
        for atom in range(1, converted.nheavyatoms() + 1):
            name = converted.atom_name(atom).strip()
            if name in heavy_coordinates:
                neutral.set_xyz(
                    pr.rosetta.core.id.AtomID(atom, index),
                    pr.rosetta.numeric.xyzVector_double_t(*heavy_coordinates[name]),
                )
    return neutral


def _residue_mapping(original_pose, binder_indices: list[int]) -> list[dict[str, Any]]:
    original_info = original_pose.pdb_info()
    mapping = []
    for apo_index, original_index in enumerate(binder_indices, start=1):
        mapping.append(
            {
                "apo_index": apo_index,
                "original_pose_index": int(original_index),
                "original_chain": str(original_info.chain(original_index)),
                "original_number": int(original_info.number(original_index)),
                    "original_insertion_code": str(
                        original_info.icode(original_index)
                    ).strip(),
            }
        )
    return mapping


def _binder_apo_pose(original_pose, binder_indices: list[int]):
    """Slice the same AF3 binder coordinates into a normalized chain-A pose."""

    if not binder_indices:
        raise ValueError("binder_chain selected no protein residues from the AF3 complex.")
    pr = _pyrosetta()
    neutral = _neutralize_charged_histidines(original_pose)
    selected = pr.rosetta.utility.vector1_unsigned_long()
    for index in binder_indices:
        selected.append(int(index))
    apo = pr.Pose()
    pr.rosetta.core.pose.pdbslice(apo, neutral, selected)
    info = pr.rosetta.core.pose.PDBInfo(apo)
    for index in range(1, apo.total_residue() + 1):
        info.chain(index, "A")
        info.number(index, index)
        info.icode(index, " ")
    apo.pdb_info(info)
    return apo


def _write_normalized_apo(
    apo,
    *,
    source_sha256: str,
    binder_chain: str,
    oligo_name: str,
) -> Path:
    digest = hashlib.sha256(
        f"{source_sha256}|{binder_chain}|{oligo_name}".encode("utf-8")
    ).hexdigest()[:24]
    root = Path(tempfile.gettempdir()) / "astrolabe_ph_core"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{digest}.pdb"
    tmp = root / f".{digest}.{os.getpid()}.tmp"
    apo.dump_pdb(str(tmp))
    os.replace(tmp, path)
    return path


def _heavy_endpoint_depths(pose, hbond, depth_calculator) -> tuple[float, float]:
    donor = pose.residue(hbond.don_res())
    acceptor = pose.residue(hbond.acc_res())
    donor_atom = int(donor.atom_base(int(hbond.don_hatm())))
    acceptor_atom = int(hbond.acc_atm())
    if donor_atom <= 0 or acceptor_atom <= 0:
        raise ValueError("Hydrogen bond does not resolve to heavy donor/acceptor atoms.")
    depths = (
        float(depth_calculator.calcdepth(donor.atom(donor_atom), donor.atom_type_set())),
        float(depth_calculator.calcdepth(acceptor.atom(acceptor_atom), acceptor.atom_type_set())),
    )
    if not all(math.isfinite(value) for value in depths):
        raise ValueError("AtomicDepth returned a non-finite heavy-atom depth.")
    return depths


def _ring_is_donor_or_acceptor(pose, hbond) -> set[int]:
    result = set()
    acceptor = pose.residue(hbond.acc_res())
    if acceptor.name3() == "HIS":
        atom_name = acceptor.atom_name(hbond.acc_atm()).strip()
        if atom_name in _RING_ATOMS:
            result.add(hbond.acc_res())
    donor = pose.residue(hbond.don_res())
    if donor.name3() == "HIS":
        donor_atom = int(donor.atom_base(int(hbond.don_hatm())))
        if donor_atom > 0 and donor.atom_name(donor_atom).strip() in _RING_ATOMS:
            result.add(hbond.don_res())
    return result


def _count_unpaired_deep_histidines(
    pose,
    histidines: set[int],
    depth_calculator,
) -> tuple[int, list[dict[str, Any]]]:
    satisfied: set[int] = set()
    for hbond in _hbonds(pose):
        energy = float(hbond.energy())
        if not math.isfinite(energy):
            raise ValueError("PyRosetta returned a non-finite hydrogen-bond energy.")
        if energy >= -0.5:
            continue
        if hbond.don_res() == hbond.acc_res():
            continue
        satisfied.update(_ring_is_donor_or_acceptor(pose, hbond) & histidines)

    unpaired = 0
    records = []
    for index in sorted(histidines):
        residue = pose.residue(index)
        depths = []
        for atom_name in ("ND1", "NE2"):
            try:
                atom = int(residue.atom_index(atom_name))
            except Exception as exc:
                raise ValueError(
                    f"Histidine apo residue {index} is missing ring atom {atom_name}."
                ) from exc
            value = float(
                depth_calculator.calcdepth(
                    residue.atom(atom),
                    residue.atom_type_set(),
                )
            )
            if not math.isfinite(value):
                raise ValueError("AtomicDepth returned a non-finite ring depth.")
            depths.append(value)
        mean_depth = float(np.mean(depths))
        deep = mean_depth >= 4.0
        bonded = index in satisfied
        if deep and not bonded:
            unpaired += 1
        records.append(
            {
                "apo_histidine": int(index),
                "ring_mean_depth": mean_depth,
                "deep": bool(deep),
                "has_negative_internal_hbond": bool(bonded),
                "unpaired_deep": bool(deep and not bonded),
            }
        )
    return unpaired, records


def _component_candidates(
    pose,
    histidines: set[int],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    graph_nodes = set(histidines)
    graph_nodes.update(int(edge["partner_residue"]) for edge in edges)
    parent = {index: index for index in graph_nodes}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for edge in edges:
        union(int(edge["histidine_residue"]), int(edge["partner_residue"]))

    by_root: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        by_root[find(edge["histidine_residue"])].append(edge)

    components = []
    for root, component_edges in by_root.items():
        component_nodes = sorted(index for index in graph_nodes if find(index) == root)
        center_nodes = sorted(index for index in component_nodes if index in histidines)
        partner_nodes = sorted(index for index in component_nodes if index not in histidines)
        component_edges.sort(
            key=lambda edge: (
                float(edge["energy"]),
                int(edge["histidine_residue"]),
                int(edge["partner_residue"]),
            )
        )
        quality = min(float(edge["quality"]) for edge in component_edges)
        components.append(
            {
                "quality": quality,
                "n_centers": len(center_nodes),
                "histidine_positions": center_nodes,
                "partner_positions": partner_nodes,
                "component_positions": component_nodes,
                "min_energy": min(float(edge["energy"]) for edge in component_edges),
                "edges": component_edges,
            }
        )
    components.sort(
        key=lambda component: (
            -float(component["quality"]),
            float(component["min_energy"]),
            tuple(component["component_positions"]),
            tuple(
                (
                    edge["histidine_residue"],
                    edge["partner_residue"],
                )
                for edge in component["edges"]
            ),
        )
    )
    return components


def _build_network_evidence(
    pose,
    *,
    source_path: Path,
    source_sha256: str,
    oligo_name: str,
    binder_chain: str,
    parameter_tuple: tuple[float, float, float, float],
) -> dict[str, Any]:
    hbond_scale, depth_scale, probe_radius, resolution = parameter_tuple
    chains = resolve_selector_chain_ids(oligo_name, binder_chain)
    if len(chains) != 1:
        raise ValueError(
            "ph_core_network requires binder_chain to resolve to exactly one chain; "
            f"got {chains!r}."
        )
    binder_indices = sorted(_chain_residues(pose, oligo_name, binder_chain))
    original_mapping = _residue_mapping(pose, binder_indices)
    apo = _binder_apo_pose(pose, binder_indices)
    apo_path = _write_normalized_apo(
        apo,
        source_sha256=source_sha256,
        binder_chain=binder_chain,
        oligo_name=oligo_name,
    )
    binder_sequence = "".join(str(apo.residue(index).name1()) for index in range(1, apo.total_residue() + 1))
    if len(binder_sequence) != len(original_mapping):
        raise ValueError("Normalized binder sequence length does not match residue mapping.")

    histidines = {
        index
        for index in range(1, apo.total_residue() + 1)
        if apo.residue(index).name3() == "HIS"
    }
    base = {
        "schema": "ph_core_network_v1",
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "normalized_apo_path": str(apo_path),
        "binder_chain": binder_chain,
        "binder_sequence": binder_sequence,
        "binder_length": len(binder_sequence),
        "histidine_positions_apo": sorted(histidines),
        "histidine_positions_zero_based": [index - 1 for index in sorted(histidines)],
        "original_residue_mapping": original_mapping,
        "parameters": {
            "hbond_scale": hbond_scale,
            "depth_scale": depth_scale,
            "probe_radius": probe_radius,
            "resolution": resolution,
        },
        "neutral_histidine_protocol": (
            "HIS/HIS_D as loaded from AF3; charged HIS_P converted to neutral HIS "
            "with common heavy-atom coordinates preserved; no repack or tautomer sampling"
        ),
    }
    if not histidines:
        return {
            **base,
            "status": "no_network",
            "neutral": 0.0,
            "unpaired": 0,
            "selected_component": None,
            "all_components": [],
            "unpaired_histidines": [],
            "edge_count": 0,
        }

    pr = _pyrosetta()
    depth_calculator = pr.rosetta.core.scoring.atomic_depth.AtomicDepth(
        apo,
        probe_radius,
        False,
        resolution,
    )
    unpaired_count, unpaired_records = _count_unpaired_deep_histidines(
        apo,
        histidines,
        depth_calculator,
    )

    best_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for hbond in _hbonds(apo):
        acceptor_index = int(hbond.acc_res())
        donor_index = int(hbond.don_res())
        if acceptor_index not in histidines or donor_index not in histidines | {
            index
            for index in range(1, apo.total_residue() + 1)
            if apo.residue(index).name3() in {"LYS", "ARG"}
        }:
            continue
        energy = float(hbond.energy())
        if not math.isfinite(energy):
            raise ValueError("PyRosetta returned a non-finite hydrogen-bond energy.")
        if energy >= 0.0:
            continue
        acceptor = apo.residue(acceptor_index)
        donor = apo.residue(donor_index)
        if donor.name3() not in _CATION_NAMES:
            continue
        if acceptor.atom_name(hbond.acc_atm()).strip() not in _RING_ATOMS:
            continue
        donor_heavy = int(donor.atom_base(int(hbond.don_hatm())))
        if donor_heavy <= 0 or donor.atom_is_backbone(donor_heavy):
            continue
        try:
            depths = _heavy_endpoint_depths(apo, hbond, depth_calculator)
        except Exception as exc:
            raise ValueError(
                f"Could not measure H-bond endpoints for apo residues "
                f"{acceptor_index}-{donor_index}: {exc}"
            ) from exc
        mean_depth = float(np.mean(depths))
        energy_quality = float(np.clip(-energy / hbond_scale, 0.0, 1.0))
        depth_quality = float(np.clip(mean_depth / depth_scale, 0.0, 1.0))
        record = {
            "histidine_residue": acceptor_index,
            "partner_residue": donor_index,
            "partner_type": str(donor.name3()),
            "histidine_ring_atom": str(acceptor.atom_name(hbond.acc_atm())).strip(),
            "partner_heavy_atom": str(donor.atom_name(donor_heavy)).strip(),
            "energy": energy,
            "mean_atomic_depth": mean_depth,
            "energy_quality": energy_quality,
            "depth_quality": depth_quality,
            "quality": min(energy_quality, depth_quality),
        }
        pair = (acceptor_index, donor_index)
        previous = best_by_pair.get(pair)
        if previous is None or record["energy"] < previous["energy"]:
            best_by_pair[pair] = record

    edges = sorted(
        best_by_pair.values(),
        key=lambda edge: (
            int(edge["histidine_residue"]),
            int(edge["partner_residue"]),
        ),
    )
    if not edges:
        return {
            **base,
            "status": "no_network",
            "neutral": 0.0,
            "unpaired": unpaired_count,
            "selected_component": None,
            "all_components": [],
            "unpaired_histidines": unpaired_records,
            "edge_count": 0,
        }

    components = _component_candidates(apo, histidines, edges)
    selected = components[0]
    for component in components:
        for edge in component["edges"]:
            edge["histidine_original_index"] = original_mapping[
                int(edge["histidine_residue"]) - 1
            ]["original_pose_index"]
            edge["partner_original_index"] = original_mapping[
                int(edge["partner_residue"]) - 1
            ]["original_pose_index"]
    selected["histidine_original_indices"] = [
        original_mapping[position - 1]["original_pose_index"]
        for position in selected["histidine_positions"]
    ]
    return {
        **base,
        "status": "network",
        "neutral": float(selected["quality"]),
        "unpaired": unpaired_count,
        "selected_component": selected,
        "all_components": components,
        "unpaired_histidines": unpaired_records,
        "edge_count": len(edges),
    }


def _clear_network_cache() -> None:
    _NETWORK_CACHE.clear()


def _network_cached(
    source_path_string: str,
    source_sha256: str,
    source_mtime_ns: int,
    source_size: int,
    oligo_name: str,
    binder_chain: str,
    parameter_tuple: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Content-addressed wrapper; mtime and size are audit-only arguments."""

    key = (source_sha256, oligo_name, binder_chain, parameter_tuple)
    cached = _NETWORK_CACHE.get(key)
    if cached is None:
        pose = _load_pose(source_path_string)
        cached = _build_network_evidence(
            pose,
            source_path=Path(source_path_string),
            source_sha256=source_sha256,
            oligo_name=oligo_name,
            binder_chain=binder_chain,
            parameter_tuple=parameter_tuple,
        )
        _NETWORK_CACHE[key] = copy.deepcopy(cached)
    result = copy.deepcopy(cached)
    result["source_path"] = str(Path(source_path_string).resolve())
    result["source_mtime_ns"] = int(source_mtime_ns)
    result["source_size"] = int(source_size)
    return result


def _get_shared_proton_runner(*, config: Any, key: str):
    from astrolabe.mutators.wrappers.proton_pottsmpnn_runner import (
        get_shared_proton_runner,
    )

    return get_shared_proton_runner(config=config, key=key)


def _extract_gap(result: Any) -> float:
    if not isinstance(result, Mapping):
        raise RuntimeError("Proton score_states must return a mapping.")
    if "gap" not in result:
        raise RuntimeError("Proton score_states did not return a 'gap' value.")
    gap = float(result["gap"])
    if not math.isfinite(gap):
        raise ValueError("Proton score_states returned a non-finite gap.")
    return gap


def _extract_state_energy(result: Mapping[str, Any], field: str) -> float:
    if field not in result:
        raise RuntimeError(f"Proton score_states did not return '{field}'.")
    energy = float(result[field])
    if not math.isfinite(energy):
        raise ValueError(f"Proton score_states returned a non-finite {field}.")
    return energy


@lru_cache(maxsize=64)
def _release_cached(
    network_identity: str,
    apo_path: str,
    binder_sequence: str,
    histidine_positions: tuple[int, ...],
    gap_scale: float,
    gap_reference: float,
    runner_key: str,
    runner_identity: str,
    runner_config_json: str,
) -> dict[str, Any]:
    del network_identity
    runner_config = json.loads(runner_config_json) if runner_config_json else None
    runner = _get_shared_proton_runner(config=runner_config, key=runner_key)
    raw = runner.score_states(
        pdb_path=apo_path,
        binder_chain="A",
        binder_sequence=binder_sequence,
        chain_sequences={"A": binder_sequence},
        histidine_positions=list(histidine_positions),
    )
    gap = _extract_gap(raw)
    neutral_energy = _extract_state_energy(raw, "neutral_energy")
    protonated_energy = _extract_state_energy(raw, "protonated_energy")
    return {
        "gap": gap,
        "state_energies": {
            "neutral": neutral_energy,
            "protonated": protonated_energy,
        },
        "proton_result": copy.deepcopy(dict(raw)),
        "gap_scale": gap_scale,
        "gap_reference": gap_reference,
        "runner_key": runner_key,
        "runner_identity": runner_identity,
    }


def _network_identity(evidence: Mapping[str, Any]) -> str:
    selected = evidence.get("selected_component")
    return _canonical_json(
        {
            "source_sha256": evidence["source_sha256"],
            "binder_sequence": evidence["binder_sequence"],
            "normalized_apo_path": evidence["normalized_apo_path"],
            "status": evidence["status"],
            "selected": selected,
            "unpaired": evidence["unpaired"],
        }
    )


def _audit_for_oligo(
    eval_result,
    *,
    name: str,
    oligo_name: str,
    binder_chain: str,
    metric: str,
    network_parameters: _NetworkParameters,
    release_parameters: _ReleaseParameters,
    runner_config: Any,
) -> dict[str, Any]:
    source = Path(eval_result.complex_paths[name]).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"AF3 complex structure does not exist: {source}")
    stat = source.stat()
    source_sha256 = _source_sha256(source)
    evidence = copy.deepcopy(
        _network_cached(
            str(source),
            source_sha256,
            stat.st_mtime_ns,
            stat.st_size,
            oligo_name,
            binder_chain,
            network_parameters.cache_tuple,
        )
    )
    audit = {
        **evidence,
        "metric": metric,
        "original_G": None,
        "gap": None,
        "state_energies": None,
        "proton_result": None,
        "runner_key": release_parameters.runner_key,
        "runner_identity": release_parameters.runner_identity,
    }
    if metric != "release" or evidence["status"] == "no_network":
        return audit

    selected = evidence["selected_component"]
    config_json = _canonical_json(_plain_config(runner_config))
    release = _release_cached(
        _network_identity(evidence),
        evidence["normalized_apo_path"],
        evidence["binder_sequence"],
        tuple(int(index) - 1 for index in selected["histidine_positions"]),
        release_parameters.gap_scale,
        release_parameters.gap_reference,
        release_parameters.runner_key,
        release_parameters.runner_identity,
        config_json,
    )
    gap = release["gap"]
    n_centers = int(selected["n_centers"])
    release_score = float(evidence["neutral"]) * math.tanh(
        (
            gap / n_centers
            - release_parameters.gap_reference
        )
        / release_parameters.gap_scale
    )
    if not math.isfinite(release_score):
        raise ValueError("Computed ph_core_network release score is non-finite.")
    audit.update(
        {
            "original_G": gap,
            "gap": gap,
            "release": release_score,
            "release_gap_per_center": gap / n_centers,
            **copy.deepcopy(release),
        }
    )
    return audit


def _record_audit(eval_result, oligo_name: str, audit: Mapping[str, Any]) -> None:
    payload = json.loads(json.dumps(audit, default=_json_default))
    records = getattr(eval_result, _AUDIT_ATTR, None)
    if records is None:
        records = {}
        setattr(eval_result, _AUDIT_ATTR, records)
    oligo_record = records.setdefault(str(oligo_name), {})
    metrics = oligo_record.setdefault("metrics", {})
    metrics[str(payload["metric"])] = payload
    # Keep the latest payload at the top level for backward compatibility with
    # early consumers, while ``metrics`` preserves every metric invocation.
    oligo_record.update(payload)
    logging.info("ph_core_network_audit %s", json.dumps(payload, sort_keys=True))


@loss_registry.register(
    name="ph_core_network",
    description=(
        "Binder-internal neutral histidine network quality, Proton acid-release "
        "response, or deep-unpaired-histidine count"
    ),
    required_params=["binder_chain"],
    optional_params=[
        "metric",
        "hbond_scale",
        "depth_scale",
        "probe_radius",
        "resolution",
        "gap_scale",
        "gap_reference",
        "runner_key",
        "runner_config",
    ],
    supports_oligomer_specific=True,
)
def ph_core_network(
    eval_result,
    binder_chain: str,
    metric: str = "neutral",
    hbond_scale: float = 0.5,
    depth_scale: float = 4.0,
    probe_radius: float = 1.4,
    resolution: float = 0.25,
    gap_scale: float = 1.0,
    gap_reference: float = 0.0,
    runner_key: str = "default",
    runner_config: Any = None,
):
    """Return one scalar per AF3 oligomer; full JSON audit is attached separately."""

    metric = str(metric or "").strip().lower()
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {sorted(_METRICS)}; got {metric!r}.")
    network_parameters = _network_parameters(
        hbond_scale=hbond_scale,
        depth_scale=depth_scale,
        probe_radius=probe_radius,
        resolution=resolution,
    )
    release_parameters = _release_parameters(
        gap_scale=gap_scale,
        gap_reference=gap_reference,
        runner_key=runner_key,
        runner_config=runner_config,
    )

    scores = {}
    for name in eval_result.names:
        audit = _audit_for_oligo(
            eval_result,
            name=name,
            oligo_name=str(name),
            binder_chain=binder_chain,
            metric=metric,
            network_parameters=network_parameters,
            release_parameters=release_parameters,
            runner_config=runner_config,
        )
        if metric == "unpaired":
            value = float(audit["unpaired"])
        elif audit["status"] == "no_network":
            value = 0.0
        elif metric == "neutral":
            value = float(audit["neutral"])
        else:
            value = float(audit["release"])
        if not math.isfinite(value):
            raise ValueError("ph_core_network produced a non-finite score.")
        scores[name] = value
        _record_audit(eval_result, name, audit)
    return scores
