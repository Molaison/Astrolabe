from __future__ import annotations

import logging
import json
from collections.abc import Mapping
from typing import Any, Dict

import numpy as np

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.base import Mutator
from astrolabe.mutators.capabilities import PROTEIN_ONLY, MutatorCapabilities
from astrolabe.mutators.utils import prepare_exclude_sites
from astrolabe.utils.mmcif_ids import int_id_to_str_id

logger = logging.getLogger(__name__)

CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


V6_DEP_MAP = {
    "HIS-P": ["HIS-S"],
    "ASP-P": ["ASP-D"],
    "GLU-P": ["GLU-D"],
}
V3_V4_NEUTRAL_HIS_TOKENS = frozenset({"HID", "HIE", "HIS-D"})

NATIVE_DEFAULT_FORBIDDEN_TOKENS = ("HIS-A", "ASP-A", "GLU-A", "HIS-D")
DEFAULT_CENTER_PROTONATION_TYPES = ("HIS-P", "ASP-P", "GLU-P")
STANDARD_AA_TO_TOKEN = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}
# H/D/E must cover every microstate the v6 checkpoint can emit.  The legacy
# HID/HIE/HIS-D names are accepted as forbidden inputs but are ignored by v6.
AA_TO_V6_TOKENS = {
    **{aa: (token,) for aa, token in STANDARD_AA_TO_TOKEN.items()},
    "H": ("HIS-P", "HIS-S", "HIS-A", "HIS-D", "HID", "HIE"),
    "D": ("ASP-P", "ASP-D", "ASP-A"),
    "E": ("GLU-P", "GLU-D", "GLU-A"),
}
KNOWN_TOKEN_NAMES = frozenset(
    set(STANDARD_AA_TO_TOKEN.values())
    | {
        "UNK",
        "HID",
        "HIE",
        "HIS-P",
        "HIS-S",
        "HIS-A",
        "HIS-D",
        "ASP-P",
        "ASP-D",
        "ASP-A",
        "GLU-P",
        "GLU-D",
        "GLU-A",
    }
)
STANDARD_TOKEN_TO_AA = {token: aa for aa, token in STANDARD_AA_TO_TOKEN.items()}
TOKEN_TO_CANONICAL_AA = {
    token: aa for aa, tokens in AA_TO_V6_TOKENS.items() for token in tokens
}


def _token_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        value = [value.decode("utf-8")]
    elif isinstance(value, str):
        value = [value]
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(
            f"Proton-PottsMPNN {label} must be a list of tokens."
        ) from exc
    tokens: list[str] = []
    for item in items:
        token = str(item).strip()
        if token:
            tokens.append(token)
    return tokens


def _merge_forbidden_tokens(*groups: Any) -> list[str]:
    """Keep the native microstate defaults and append every caller token once."""
    merged = list(NATIVE_DEFAULT_FORBIDDEN_TOKENS)
    for group in groups:
        for token in _token_list(group, "forbidden_tokens"):
            if token not in merged:
                merged.append(token)
    return merged


def tokens_for_excluded_aas(exclude_aa: Any) -> list[str]:
    """Translate Astrolabe one-letter/three-letter exclusions into v6 tokens."""
    tokens: list[str] = []
    for entry in _token_list(exclude_aa, "exclude_AA"):
        upper = entry.upper()
        if len(upper) == 1 and upper in AA_TO_V6_TOKENS:
            candidates = AA_TO_V6_TOKENS[upper]
        elif upper in STANDARD_TOKEN_TO_AA:
            candidates = AA_TO_V6_TOKENS[STANDARD_TOKEN_TO_AA[upper]]
        elif upper in KNOWN_TOKEN_NAMES:
            candidates = (upper,)
        else:
            raise ValueError(
                "Proton-PottsMPNN exclude_AA accepts one-letter amino acids, "
                "three-letter names, or v6 tokens; got "
                f"{entry!r}. Examples: 'C' -> CYS, 'H' -> HIS-P/HIS-S/HIS-A."
            )
        for token in candidates:
            if token not in tokens:
                tokens.append(token)
    return tokens


def _apply_excluded_amino_acids(
    criteria: Mapping[str, Any], exclude_aa: Any
) -> dict[str, Any]:
    """Add hard Astrolabe AA exclusions to the native valid-token mask.

    ``PHDesignCriteria.forbidden_tokens`` is enforced inside the optimizer's
    allowed-token mask.  We therefore translate before the engine runs and do
    not repair an already-selected joint sequence afterwards.
    """
    criteria = dict(criteria)
    excluded_tokens = tokens_for_excluded_aas(exclude_aa)
    criteria["forbidden_tokens"] = _merge_forbidden_tokens(
        criteria.get("forbidden_tokens"), excluded_tokens
    )
    forbidden = set(criteria["forbidden_tokens"])

    explicit_centers = criteria.get("explicit_centers") or []
    if isinstance(explicit_centers, Mapping):
        explicit_centers = [explicit_centers]
    explicit_conflicts: list[str] = []
    for center in explicit_centers:
        if not isinstance(center, Mapping):
            continue
        protonation_type = str(center.get("protonation_type") or "").strip()
        if protonation_type and protonation_type in forbidden:
            explicit_conflicts.append(protonation_type)
    if explicit_conflicts:
        raise ValueError(
            "Proton-PottsMPNN exclude_AA/forbidden_tokens forbids explicit centre "
            f"protonation type(s) {sorted(set(explicit_conflicts))}. Remove those "
            "centres or stop excluding the corresponding amino acid."
        )

    center_types = _token_list(criteria.get("center_types"), "center_types")
    center_type_conflicts = [token for token in center_types if token in forbidden]
    if center_type_conflicts:
        raise ValueError(
            "Proton-PottsMPNN exclude_AA/forbidden_tokens conflicts with the exact "
            f"center_types {center_type_conflicts}. Use allowed protonation types "
            "or remove the exclusion."
        )

    configured_pool = criteria.get("center_protonation_types")
    pool = (
        list(DEFAULT_CENTER_PROTONATION_TYPES)
        if configured_pool is None
        else _token_list(configured_pool, "center_protonation_types")
    )
    allowed_pool = [token for token in pool if token not in forbidden]
    method = str(criteria.get("method") or "converged_mcmc").strip().lower()
    if (
        not allowed_pool
        and method not in WHOLE_CHAIN_METHODS
        and not _is_centre_free(criteria)
    ):
        raise ValueError(
            "Proton-PottsMPNN exclude_AA/forbidden_tokens removes every configured "
            f"centre protonation type {pool}; allow H, D, or E, or use a "
            "centre-free/whole-chain method."
        )
    criteria["center_protonation_types"] = allowed_pool

    extra_forbidden = [
        token
        for token in criteria["forbidden_tokens"]
        if token not in NATIVE_DEFAULT_FORBIDDEN_TOKENS
    ]
    if method == "mpnn_sample" and extra_forbidden:
        raise ValueError(
            "Proton-PottsMPNN method='mpnn_sample' has no valid-AA mask upstream "
            "and cannot enforce exclude_AA/forbidden_tokens "
            f"({extra_forbidden}). Use method='gibbs' or a placement method."
        )
    return criteria


def _assert_no_forbidden_tokens(
    selected: Mapping[str, Any], features: Mapping[str, Any]
) -> None:
    """Hard contract check: no forbidden token may survive at a mutable site.

    This rejects an invalid joint proposal; it never edits the selected
    sequence.  Fixed positions are exempt so a pre-existing fixed Cys is kept.
    """
    extended_tokens = selected.get("extended_tokens")
    if extended_tokens is None:
        raise ValueError(
            "Proton-PottsMPNN selected design is missing extended_tokens; cannot "
            "verify the forbidden-token contract."
        )
    extended_tokens = list(extended_tokens)
    binder_length = len(str(features.get("binder_sequence") or ""))
    if len(extended_tokens) != binder_length:
        raise ValueError(
            "Proton-PottsMPNN selected extended_tokens length "
            f"{len(extended_tokens)} does not match binder length {binder_length}."
        )
    criteria = features.get("criteria") or {}
    forbidden = {str(token) for token in criteria.get("forbidden_tokens") or []}
    if not forbidden:
        return
    fixed = {int(position) for position in features.get("fixed_positions") or []}
    violations = [
        (position, str(token))
        for position, token in enumerate(extended_tokens)
        if position not in fixed and str(token) in forbidden
    ]
    if violations:
        raise ValueError(
            "Proton-PottsMPNN selected joint design contains forbidden token(s) "
            f"{violations[:5]} at mutable positions; refusing to queue it."
        )


def _normalize_criteria(criteria: Any) -> dict[str, Any]:
    if criteria is None:
        criteria = {}
    if not isinstance(criteria, Mapping):
        raise ValueError("Proton-PottsMPNN criteria must be a mapping.")
    criteria = dict(criteria)
    seed_source = str(criteria.get("seed_source") or "inverse").strip().lower()
    if seed_source != "inverse" and not (seed_source == "native" and _is_centre_free(criteria)):
        raise ValueError(
            "Proton-PottsMPNN requires criteria.seed_source='inverse', or 'native' "
            "for centre-free greedy_energy_block; the current sequence must be the optimization start."
        )
    criteria["seed_source"] = seed_source
    configured_max = criteria.get("max_mutations", None)
    if configured_max not in (None, "", "None", "null"):
        configured_max = int(configured_max)
        if configured_max == 0:
            raise ValueError(
                "Proton-PottsMPNN criteria.max_mutations=0 means 'no native cap' "
                "upstream, which is incompatible with a finite GH mutation budget. "
                "Use null to derive the designable cap from the GH budget."
            )
        if configured_max < 0:
            raise ValueError(
                "Proton-PottsMPNN criteria.max_mutations must be positive."
            )
        criteria["max_mutations"] = configured_max

    dep_map = criteria.get("dep_map")
    if dep_map is None:
        dep_map = V6_DEP_MAP
    elif not isinstance(dep_map, Mapping):
        raise ValueError("Proton-PottsMPNN criteria.dep_map must be a mapping.")
    normalized_dep_map = {
        str(key): list(values)
        for key, values in V6_DEP_MAP.items()
    }
    for key, values in dep_map.items():
        if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
            raise ValueError(
                "Proton-PottsMPNN criteria.dep_map values must be lists of tokens."
            )
        normalized_dep_map[str(key)] = [str(token) for token in values]
    invalid_his_tokens = sorted(
        {
            token
            for values in normalized_dep_map.values()
            for token in values
            if token in V3_V4_NEUTRAL_HIS_TOKENS
        }
    )
    if invalid_his_tokens:
        raise ValueError(
            "Proton-PottsMPNN uses the v6 vocabulary; neutral histidine is HIS-S, "
            f"not {invalid_his_tokens}. Remove v3/v4 HID/HIE mappings."
        )
    criteria["dep_map"] = normalized_dep_map
    criteria["forbidden_tokens"] = _merge_forbidden_tokens(
        criteria.get("forbidden_tokens")
    )
    return criteria


WHOLE_CHAIN_METHODS = frozenset({"gibbs", "mpnn_sample"})


def _is_centre_free(criteria: Mapping[str, Any]) -> bool:
    """Whether the native criteria select the documented centre-free mode."""
    method = str(criteria.get("method") or "converged_mcmc").strip().lower()
    if method != "greedy_energy_block":
        return False
    if criteria.get("explicit_centers") or criteria.get("center_types"):
        return False
    if str(criteria.get("infill_scope") or "neighbourhood").strip().lower() != "chain":
        return False
    return int(criteria.get("center_count", 1) or 0) == 0


def estimate_pin_count(criteria: Mapping[str, Any]) -> int:
    """Upper bound on engine-pinned centre positions for one design.

    Upstream ``max_mutations`` caps only designable positions; centre pins are
    additional canonical edits. Reserve this count from the GH budget.
    """
    method = str(criteria.get("method") or "converged_mcmc").strip().lower()
    if method in WHOLE_CHAIN_METHODS or _is_centre_free(criteria):
        return 0
    explicit = criteria.get("explicit_centers") or []
    if explicit:
        return len(list(explicit))
    center_types = criteria.get("center_types") or []
    if center_types:
        return len(list(center_types))
    return max(0, int(criteria.get("center_count", 1) or 0))


def resolve_designable_cap(
    criteria: Mapping[str, Any], budget: int, *, binder_length: int
) -> tuple[int, int]:
    """Return (designable_cap, pin_budget) for the native engine.

    Native ``max_mutations=0`` means unbounded, never pass it when a GH budget
    exists. Pins are reserved first so the selected joint design can be applied
    without truncation. The documented centre-free mode
    (``greedy_energy_block`` + ``infill_scope=chain`` + ``center_count=0``)
    has no pin reserve; its native designable cap is the GH budget itself.
    """
    budget = int(budget)
    binder_length = int(binder_length)
    if budget <= 0:
        raise ValueError(
            "Proton-PottsMPNN requires a positive GH mutation budget "
            "(mutation_num * mutation_weight)."
        )
    method = str(criteria.get("method") or "converged_mcmc").strip().lower()
    if method in WHOLE_CHAIN_METHODS:
        if budget < binder_length:
            raise ValueError(
                f"Proton-PottsMPNN method {method!r} redesigns the whole binder and "
                f"is not bounded by max_mutations; GH budget {budget} < binder length "
                f"{binder_length}. Use block_descent or raise mutation_num."
            )
        return budget, 0

    if _is_centre_free(criteria):
        configured = criteria.get("max_mutations", None)
        if configured in (None, "", "None", "null"):
            return budget, 0
        configured = int(configured)
        if configured <= 0:
            raise ValueError(
                "Proton-PottsMPNN criteria.max_mutations must be positive."
            )
        if configured > budget:
            raise ValueError(
                "Proton-PottsMPNN criteria.max_mutations "
                f"({configured}) exceeds the GH mutation budget ({budget}); "
                "increase mutation_num or lower the native cap."
            )
        return configured, 0

    pin_budget = estimate_pin_count(criteria)
    if pin_budget <= 0:
        raise ValueError(
            "Proton-PottsMPNN placement methods require at least one centre pin; "
            "the centre-free form is supported only by method='greedy_energy_block' "
            "with infill_scope='chain', center_count=0, and no "
            "center_types/explicit_centers."
        )
    if budget <= pin_budget:
        raise ValueError(
            f"Proton-PottsMPNN budget {budget} must exceed the requested centre pin "
            f"budget {pin_budget}; increase mutation_num or reduce center_count/"
            "center_types."
        )
    configured = criteria.get("max_mutations", None)
    if configured in (None, "", "None", "null"):
        return budget - pin_budget, pin_budget
    configured = int(configured)
    if configured <= 0:
        raise ValueError("Proton-PottsMPNN criteria.max_mutations must be positive.")
    if configured + pin_budget > budget:
        raise ValueError(
            "Proton-PottsMPNN criteria.max_mutations "
            f"({configured}) + centre pins ({pin_budget}) exceeds the GH mutation "
            f"budget ({budget}); increase mutation_num or lower the native cap."
        )
    return configured, pin_budget

def _mutable_positions(protomer, excluded_positions) -> np.ndarray:
    mask = np.asarray(protomer.mutable_mask, dtype=bool).copy()
    for position in excluded_positions or []:
        position = int(position)
        if position < 0 or position >= len(mask):
            raise ValueError(
                f"Excluded position {position} is out of range for protomer "
                f"{protomer.name!r} length {len(mask)}."
            )
        mask[position] = False
    return mask


def _mutable_segments(protomer, mutable_mask) -> set[str]:
    mask = np.asarray(mutable_mask, dtype=bool)
    segments: set[str] = set()
    for index, component_slice in enumerate(protomer.components_slices):
        if mask[component_slice].any():
            segments.add(str(protomer.components[index]))
    return segments


def _assert_no_tied_positions(protomer) -> None:
    tied = [tuple(t) for t in protomer.tied_positions if len(tuple(t)) > 1]
    if tied:
        raise ValueError(
            "Proton-PottsMPNN does not support tied/symmetric positions; "
            f"protomer {protomer.name!r} has {len(tied)} tied position group(s)."
        )


def select_binder_chain(
    chain_protos: list[tuple[str, str]],
    mutable_by_proto: Mapping[str, np.ndarray],
    configured_binder_chain: str | None,
) -> str:
    mutable_chains = [
        chain
        for chain, proto in chain_protos
        if bool(np.asarray(mutable_by_proto[proto], dtype=bool).any())
    ]
    configured = (
        None
        if configured_binder_chain in (None, "", "None", "null")
        else str(configured_binder_chain).strip()
    )
    if configured is not None:
        chains = [chain for chain, _ in chain_protos]
        if configured not in chains:
            raise ValueError(
                f"Proton-PottsMPNN binder_chain {configured!r} is not one of "
                f"the oligomer chains {chains}."
            )
        if configured not in mutable_chains:
            raise ValueError(
                f"Proton-PottsMPNN binder_chain {configured!r} has no mutable positions."
            )
        if len(mutable_chains) != 1:
            raise ValueError(
                "Proton-PottsMPNN is a single-binder joint optimizer, but multiple "
                f"protein chains have mutable positions: {mutable_chains}. "
                "Fix the selector/mutable masks or set binder_chain and fix the others."
            )
        return configured
    if not mutable_chains:
        raise ValueError(
            "Proton-PottsMPNN found no mutable protein chain in the oligomer."
        )
    if len(mutable_chains) > 1:
        raise ValueError(
            "Proton-PottsMPNN is a single-binder joint optimizer, but multiple "
            f"protein chains have mutable positions: {mutable_chains}. "
            "Apply a position selector or fix residues so exactly one chain is mutable."
        )
    return mutable_chains[0]


def canonical_edits(
    current_sequence: str,
    designed_sequence: str,
    *,
    fixed_positions: list[int] | tuple[int, ...] = (),
    label: str = "Proton-PottsMPNN",
) -> list[tuple[int, str, str]]:
    current = str(current_sequence).strip().upper()
    designed = str(designed_sequence).strip().upper()
    if len(current) != len(designed):
        raise ValueError(
            f"{label} designed sequence length {len(designed)} does not match "
            f"current sequence length {len(current)}."
        )
    fixed = {int(p) for p in fixed_positions}
    edits: list[tuple[int, str, str]] = []
    for position, (from_aa, to_aa) in enumerate(zip(current, designed)):
        if to_aa not in CANONICAL_AA:
            raise ValueError(
                f"{label} produced non-canonical residue {to_aa!r} at position "
                f"{position}; GH supports only the 20 canonical amino acids."
            )
        if from_aa == to_aa:
            continue
        if position in fixed:
            raise ValueError(
                f"{label} changed fixed position {position} ({from_aa}->{to_aa}); "
                "the engine design mask was not respected."
            )
        if from_aa not in CANONICAL_AA:
            raise ValueError(
                f"{label} cannot queue an edit at position {position}: current "
                f"residue {from_aa!r} is not a canonical amino acid."
            )
        edits.append((position, from_aa, to_aa))
    return edits


def _segment_site(protomer, position: int) -> tuple[str, int]:
    for index, component_slice in enumerate(protomer.components_slices):
        if component_slice.start <= position < component_slice.stop:
            return protomer.components[index], position - int(component_slice.start)
    raise ValueError(
        f"Position {position} does not map to a segment of protomer {protomer.name!r}."
    )


class ProtonPottsMPNNMutator(Mutator):
    """Joint Potts redsigner from christian-creator/ProtonPottsMPNN."""

    name = "ProtonPottsMPNN"
    capabilities = MutatorCapabilities(
        supported_types=PROTEIN_ONLY,
        needs_structure=True,
        outputs_probabilities=False,
        outputs_scores=False,
        supports_transition_blending=False,
        supports_shared_selection_policy=False,
        supports_initialization_from_structure=False,
    )

    def __init__(
        self,
        sequence_model_runner,
        position_selector=None,
        binder_chain=None,
        criteria=None,
        n_jobs=1,
        preserve_disulfides=False,
        protect_gly_pro=False,
        reference_sequence=None,
        max_total_mutations=None,
        network_proposal=None,
        network_only=False,
    ):
        self.runner = sequence_model_runner
        if not isinstance(network_only, bool):
            raise ValueError("network_only must be boolean")
        self.network_only = network_only
        self.position_selector = position_selector
        self.binder_chain = (
            None
            if binder_chain in (None, "", "None", "null")
            else str(binder_chain).strip()
        )
        criteria = dict(criteria or {})
        if network_only:
            # One dynamic neutral-His pair; no ordinary redesign or His-count reward.
            criteria.update(method="greedy_energy_block", backend="potts",
                            center_count=0, center_types=[], explicit_centers=[],
                            infill_scope="chain", seed_source="native",
                            selective=False, combined_lambda=0.0, global_weight=0.0,
                            placement_region=["all"], block_size=2,
                            aa_bias={}, aa_bias_caps={}, network_pair_only=True)
            criteria.update(network_pair_bias=0.0, network_pair_max_steps=1,
                            temperature=0.1, repetitive_window_weight=0.0)
            criteria["forbidden_tokens"] = _merge_forbidden_tokens(
                criteria.get("forbidden_tokens"), ["HIS", "HIS-P", "HIS-A", "HIS-D"]
            )
            network_proposal = {**dict(network_proposal or {}), "enabled": True}
        self.criteria = _normalize_criteria(criteria)
        from astrolabe.mutators.proton_network_geometry import NetworkProposalParameters
        self.network_proposal = NetworkProposalParameters.from_mapping(network_proposal)
        if self.network_proposal.enabled and not _is_centre_free(self.criteria):
            raise ValueError("network_proposal requires centre-free greedy_energy_block.")
        if self.network_proposal.enabled and not network_only and float(self.criteria.get("network_pair_bias", 0.0)) <= 0:
            raise ValueError("network_proposal requires a positive network_pair_bias.")
        self.preserve_disulfides = bool(preserve_disulfides)
        self.protect_gly_pro = bool(protect_gly_pro)
        self.reference_sequence = (
            None if reference_sequence is None else str(reference_sequence).strip().upper()
        )
        self.max_total_mutations = max_total_mutations
        if max_total_mutations is not None:
            if (isinstance(max_total_mutations, bool)
                    or int(max_total_mutations) != max_total_mutations
                    or int(max_total_mutations) < 0
                    or not self.reference_sequence
                    or any(aa not in CANONICAL_AA for aa in self.reference_sequence)):
                raise ValueError("max_total_mutations requires a nonnegative integer and explicit canonical reference_sequence.")
            if not _is_centre_free(self.criteria):
                raise ValueError("max_total_mutations requires centre-free greedy_energy_block.")
        self.n_jobs = int(n_jobs or 1)
        if self.n_jobs < 1:
            raise ValueError("Proton-PottsMPNN n_jobs must be at least 1.")
        self.last_mutation_telemetry: dict[str, Any] = {
            "mutator": self.name,
            "oligos": {},
        }

    def shutdown(self) -> None:
        self.runner.close()

    def _mutation_budget(self) -> int:
        return int(self.n_mutations())

    def _active_scores(
        self, evaluation_scores: StandardScores, runtime_config: Dict[str, Any]
    ) -> StandardScores:
        excluded_complexes = set()
        if runtime_config is not None and hasattr(runtime_config, "get"):
            excluded_complexes.update(runtime_config.get("ignored_complexes") or [])
            excluded_complexes.update(
                runtime_config.get("negative_design_complexes") or []
            )
        active_oligos = [
            oligo_name
            for oligo_name in evaluation_scores.names
            if oligo_name not in excluded_complexes
        ]
        if active_oligos and active_oligos != list(evaluation_scores.names):
            return evaluation_scores.filter_by_oligomer(active_oligos)
        return evaluation_scores

    def prepare_mutation_input(
        self,
        protomers,
        evaluation_scores: StandardScores,
        runtime_config: Dict[str, Any],
    ):
        # This reservation belongs to one proposal step, never a permanent His lock.
        if runtime_config is not None:
            runtime_config.pop("network_pair_roles", None)
        active_scores = self._active_scores(evaluation_scores, runtime_config)
        active_names = list(active_scores.names)
        if len(active_names) != 1:
            raise ValueError(
                "Proton-PottsMPNN supports exactly one active complex after "
                f"excluding ignored/negative designs; got {len(active_names)} "
                f"({active_names!r}). Multiple active contexts would each require "
                "the full GH budget and are not a supported joint model."
            )
        excluded_positions = prepare_exclude_sites(
            self.position_selector,
            active_scores,
            protomers,
            runtime_config=runtime_config,
            mutator_alias=getattr(self, "alias", self.name),
            fallback_to_all_mutable_if_selector_empty=False,
        )
        budget = self._mutation_budget()
        if budget <= 0:
            raise ValueError(
                "Proton-PottsMPNN requires a positive GH mutation budget "
                "(mutation_num * mutation_weight)."
            )

        prepared: dict[str, dict[str, Any]] = {}
        owner_by_binder_proto: dict[str, str] = {}
        owner_by_mutable_segment: dict[str, str] = {}
        for oligo_name in active_scores.names:
            chain_protos = [
                (int_id_to_str_id(chain_index + 1), proto)
                for chain_index, proto in enumerate(oligo_name)
                if proto in protomers
            ]
            if not chain_protos:
                raise ValueError(
                    f"Proton-PottsMPNN found no configured protomers for oligomer "
                    f"{oligo_name!r}."
                )
            non_protein = [
                proto
                for _, proto in chain_protos
                if protomers[proto].type != ProtomerTypes.PROTEIN
            ]
            if non_protein:
                raise ValueError(
                    "Proton-PottsMPNN supports protein components only; "
                    f"oligomer {oligo_name!r} contains {non_protein}."
                )
            for _, proto in chain_protos:
                _assert_no_tied_positions(protomers[proto])

            mutable_by_proto = {
                proto: _mutable_positions(
                    protomers[proto], excluded_positions.get(proto, [])
                )
                for _, proto in chain_protos
            }
            binder_chain = select_binder_chain(
                chain_protos, mutable_by_proto, self.binder_chain
            )
            binder_proto_name = dict(chain_protos)[binder_chain]
            binder_proto = protomers[binder_proto_name]
            binder_mask = mutable_by_proto[binder_proto_name]
            if set(binder_proto.ptm_dict) & set(np.where(binder_mask)[0].tolist()):
                raise ValueError(
                    "Proton-PottsMPNN does not support PTMs at mutable positions; "
                    f"protomer {binder_proto_name!r} has a PTM in the design region."
                )
            fixed_positions = sorted(
                int(p) for p in np.where(~binder_mask)[0].tolist()
            )
            oligo_label = str(oligo_name)
            prior_owner = owner_by_binder_proto.get(binder_proto_name)
            if prior_owner is not None:
                raise ValueError(
                    "Proton-PottsMPNN does not support coupling multiple active "
                    f"oligomers through the same mutable protomer {binder_proto_name!r} "
                    f"({prior_owner!r} and {oligo_label!r}); this is not a joint "
                    "multistate context. Fix one of the copies or use a single "
                    "coherent oligomer."
                )
            owner_by_binder_proto[binder_proto_name] = oligo_label
            for segment_name in sorted(_mutable_segments(binder_proto, binder_mask)):
                prior_owner = owner_by_mutable_segment.get(segment_name)
                if prior_owner is not None:
                    raise ValueError(
                        "Proton-PottsMPNN does not support multiple active oligomers "
                        f"writing the same mutable segment {segment_name!r} "
                        f"({prior_owner!r} and {oligo_label!r}); refusing a partial "
                        "or implicitly shared joint proposal."
                    )
                owner_by_mutable_segment[segment_name] = oligo_label
            binder_sequence = str(binder_proto.sequence).upper()
            excluded_tokens = tokens_for_excluded_aas(global_config.exclude_AA)
            excluded_aas = {
                TOKEN_TO_CANONICAL_AA[token]
                for token in excluded_tokens
                if token in TOKEN_TO_CANONICAL_AA
            }
            method = str(self.criteria.get("method") or "converged_mcmc").strip().lower()
            if method == "greedy_energy_block" and excluded_aas:
                mutable_excluded = [
                    int(position)
                    for position in np.where(binder_mask)[0]
                    if binder_sequence[int(position)] in excluded_aas
                ]
                if mutable_excluded:
                    raise ValueError(
                        "Proton-PottsMPNN method='greedy_energy_block' only moves "
                        "positions whose energy improves, so it cannot guarantee "
                        "removal of excluded residues already present at mutable "
                        f"positions {mutable_excluded}. Use block_descent/"
                        "autoregressive or change those seed residues."
                    )
            criteria = _apply_excluded_amino_acids(
                self.criteria, global_config.exclude_AA
            )
            designable_cap, pin_budget = resolve_designable_cap(
                criteria, budget, binder_length=len(binder_sequence)
            )
            criteria["max_mutations"] = designable_cap
            pdb_path = active_scores.complex_paths.get(oligo_name)
            if pdb_path in (None, ""):
                raise FileNotFoundError(
                    f"Proton-PottsMPNN has no structure for oligomer {oligo_name!r}."
                )
            suffix = str(pdb_path).lower().rsplit(".", 1)[-1]
            if suffix not in {"pdb", "cif", "mmcif"}:
                raise ValueError(
                    "Proton-PottsMPNN requires PDB or mmCIF evaluator structures; "
                    f"got {pdb_path!r}."
                )
            if self.preserve_disulfides or self.protect_gly_pro:
                from astrolabe.losses.ph_sensitive import _load_pose

                pose = _load_pose(pdb_path)
                indices = [i for i in range(1, pose.total_residue() + 1)
                           if pose.pdb_info().chain(i) == binder_chain]
                if "".join(pose.residue(i).name1() for i in indices) != binder_sequence:
                    raise ValueError("Protected-residue pose sequence does not match the binder.")
                for position, index in enumerate(indices):
                    residue = pose.residue(index)
                    if ((self.preserve_disulfides and residue.type().is_disulfide_bonded())
                            or (self.protect_gly_pro and residue.name1() in {"G", "P"})):
                        binder_mask[position] = False
                fixed_positions = sorted(int(p) for p in np.where(~binder_mask)[0].tolist())
            if self.max_total_mutations is not None:
                if len(self.reference_sequence) != len(binder_sequence):
                    raise ValueError("reference_sequence length does not match the binder.")
                criteria["mutation_reference_sequence"] = self.reference_sequence
                criteria["max_total_mutations"] = int(self.max_total_mutations)
            if self.network_proposal.enabled:
                from astrolabe.mutators.proton_network_geometry import build_network_pair_preferences
                preferences, geometry = build_network_pair_preferences(
                    pdb_path, binder_chain, binder_sequence,
                    np.where(binder_mask)[0].tolist(), self.network_proposal,
                    fallback_to_best=self.network_only,
                )
                criteria["network_pair_preferences"] = preferences
                geometry["source_path"] = str(pdb_path)
                geometry["network_pair_bias"] = float(criteria["network_pair_bias"])
                logger.info("ph_network_proposal_audit %s", json.dumps(geometry, sort_keys=True))
            prepared[oligo_name] = {
                "pdb_path": str(pdb_path),
                "binder_chain": binder_chain,
                "binder_protomer": binder_proto_name,
                "binder_sequence": binder_sequence,
                "chain_sequences": {
                    chain: str(protomers[proto].sequence).upper()
                    for chain, proto in chain_protos
                },
                "fixed_positions": fixed_positions,
                "criteria": criteria,
                "max_mutations": budget,
                "pin_budget": pin_budget,
                "n_jobs": self.n_jobs,
            }
        return prepared

    def compute_mutation(self, mutation_input_features: Dict[str, Any]):
        mutation_output: dict[str, dict[str, Any]] = {}
        for oligo_name, input_feat in mutation_input_features.items():
            output = self.runner.design_sequences(**input_feat)
            output.update({"name": oligo_name, "feature_dict": input_feat})
            mutation_output[oligo_name] = output
        return mutation_output

    def prepare_mutation_output(
        self, mutation_output, protomers, evaluation_scores, runtime_config
    ):
        budget = self._mutation_budget()
        pending_mutations: list[
            tuple[str, tuple[int, str, str, float, float]]
        ] = []
        pending_telemetry: dict[str, dict[str, Any]] = {}
        network_pair_roles: dict[str, dict[int, str]] = {}
        claimed_sites: set[tuple[str, int]] = set()
        for oligo_name, output in (mutation_output or {}).items():
            features = output["feature_dict"]
            selected = output["selected"]
            binder_proto_name = features["binder_protomer"]
            protomer = global_config.protomers[binder_proto_name]
            current = str(features["binder_sequence"])
            _assert_no_forbidden_tokens(selected, features)
            designed = str(selected["canonical_sequence"])
            edits = canonical_edits(
                current,
                designed,
                fixed_positions=features["fixed_positions"],
                label=f"Proton-PottsMPNN[{oligo_name}]",
            )
            if self.network_only and edits:
                changed = {int(edit[0]) for edit in edits}
                roles = {"HIS": "H", "LYS": "K", "ARG": "R"}
                matches = [p for p in features["criteria"].get("network_pair_preferences", [])
                           if changed <= {int(p["i"]), int(p["j"])}
                           and designed[int(p["i"])] == roles[p["aa_i"]]
                           and designed[int(p["j"])] == roles[p["aa_j"]]]
                if len(changed) > 2 or not matches:
                    raise ValueError("network_only produced edits outside a supported His-H/K/R pair.")
                # The native engine returns a genotype, not a unique pair id.
                # Declare one deterministic supported match and keep BOTH roles,
                # including a partner that required no canonical substitution.
                declared_pair = sorted(matches, key=lambda p: (
                    -float(p.get("quality", 0.0)), int(p["i"]), int(p["j"]),
                    str(p["aa_i"]), str(p["aa_j"]))) [0]
                network_pair_roles[binder_proto_name] = {
                    int(declared_pair["i"]): roles[declared_pair["aa_i"]],
                    int(declared_pair["j"]): roles[declared_pair["aa_j"]],
                }
            if len(edits) > budget:
                raise ValueError(
                    f"Proton-PottsMPNN[{oligo_name}] produced {len(edits)} edits, "
                    f"exceeding the GH mutation budget {budget}. The joint proposal "
                    "must not be silently truncated."
                )
            if self.max_total_mutations is not None:
                total = sum(a != b for a, b in zip(self.reference_sequence, designed))
                if total > self.max_total_mutations:
                    raise ValueError("Proton-PottsMPNN joint proposal exceeds the original-parent mutation budget.")

            edit_records: list[dict[str, Any]] = []
            for position, from_aa, to_aa in edits:
                segment_name, segment_site = _segment_site(protomer, position)
                segment = global_config.segments[segment_name]
                site_key = (segment_name, segment_site)
                if site_key in claimed_sites:
                    raise ValueError(
                        f"Proton-PottsMPNN[{oligo_name}] duplicate proposed site "
                        f"{segment_name}:{segment_site}; refusing an inconsistent "
                        "joint proposal."
                    )
                queued = segment.mutations or []
                if any(int(m[0]) == segment_site for m in queued):
                    raise ValueError(
                        f"Proton-PottsMPNN[{oligo_name}] site {segment_name}:"
                        f"{segment_site} already has a queued mutation from another "
                        "mutator; refusing to overwrite a joint proposal."
                    )
                claimed_sites.add(site_key)
                pending_mutations.append(
                    (
                        segment_name,
                        (
                            segment_site,
                            from_aa,
                            to_aa,
                            0.0,
                            float(selected.get("final_potts_energy") or 0.0),
                        ),
                    )
                )
                edit_records.append(
                    {
                        "position": position,
                        "segment": segment_name,
                        "segment_site": segment_site,
                        "from": from_aa,
                        "to": to_aa,
                    }
                )

            pending_telemetry[oligo_name] = {
                "binder_chain": features["binder_chain"],
                "binder_protomer": binder_proto_name,
                "budget": budget,
                "n_edits": len(edits),
                "edits": edit_records,
                "selected": dict(selected),
                "designs_considered": output.get("designs_considered"),
                "raw_output_path": output.get("raw_output_path"),
                "seed_energies": output.get("seed_energies"),
                "declared_network_pair_roles": network_pair_roles.get(binder_proto_name, {}),
                "pair_identity_source": "deterministic supported canonical match; not an engine pair-id claim",
            }

        if pending_mutations:
            global_config.segment_manager.add_mutations(
                pending_mutations, len(pending_mutations)
            )
        self.last_mutation_telemetry["oligos"] = pending_telemetry
        self.last_mutation_telemetry["network_pair_roles"] = network_pair_roles
        if runtime_config is not None:
            runtime_config["network_pair_roles"] = network_pair_roles
        for oligo_name, telemetry in pending_telemetry.items():
            logger.info(
                "Proton-PottsMPNN[%s] applied %d joint-design edit(s); raw=%s",
                oligo_name,
                telemetry["n_edits"],
                telemetry.get("raw_output_path"),
            )
        return {"proton_pottsmpnn": self.last_mutation_telemetry}
