"""Import-cheap confidence compatibility for structure-only evaluators."""
from __future__ import annotations

import ast
from collections.abc import Mapping


_CONFIDENCE_LOSSES = frozenset({
    "pae", "pde", "ptm", "plddt", "iptm", "binder_iptm", "binder_ptm",
    "chain_pair_iptm", "target_hotspot_plddt", "ipsae", "pae_min", "scaled_pae_min",
    "intra_contact_lis", "intra_iptm", "intra_lis", "intra_pae_min",
    "cross_contact_lis", "cross_iptm", "cross_lis", "cross_pae_min",
})


def validate_structure_only_scores(*, active_evaluators, active_mutators,
                                   active_losses, per_residue_scores=None,
                                   stages=None, expression_terms=None):
    """Reject confidence uses that could otherwise silently score missing data.

    The startup coordinator supplies only selected, merged terms. Structural
    scores/selectors remain supported. Physical input restrictions are checked
    by prepare_input, before any BioEmu2 subprocess can start.
    """
    structure_aliases = {str(alias) for alias, kind, _factory, _options in active_evaluators
                         if kind.lower() in {"bioemu2", "generated_structure"}}
    if not structure_aliases:
        return []
    errors = []
    expression_terms = expression_terms if isinstance(expression_terms, Mapping) else {}

    def variables(expression):
        try:
            return {node.id for node in ast.walk(ast.parse(str(expression), mode="eval"))
                    if isinstance(node, ast.Name)}
        except SyntaxError:
            return set()  # The coordinator owns syntax diagnostics.

    default = (stages or {}).get("default") or {}
    selected = stages or {"default": {}}
    for name, stage in selected.items():
        if not isinstance(stage, Mapping):
            continue
        current = {**default, **{key: value for key, value in stage.items() if value is not None}}
        evaluator = current.get("evaluator")
        if stages and evaluator not in structure_aliases:
            continue
        explanation = f"stages.{name}: structure-only evaluator provides no confidence metrics"
        loss_names = current.get("active_loss_terms")
        if not isinstance(loss_names, (list, tuple)):
            loss_names = None
        losses = active_losses if loss_names is None else {
            alias: options for alias, options in active_losses.items() if alias in loss_names
        }
        referenced = set(loss_names or ())
        for field in ("weighted_score_expression", "acceptance_threshold", "acceptance_gate",
                      "early_stopping", "prefix_early_stopping"):
            referenced.update(variables(current.get(field)))
        pending = list(referenced)
        while pending:
            alias = pending.pop()
            for dependency in variables(expression_terms.get(alias)):
                if dependency not in referenced:
                    referenced.add(dependency)
                    pending.append(dependency)
        if referenced & _CONFIDENCE_LOSSES:
            errors.append(f"{explanation}: {sorted(referenced & _CONFIDENCE_LOSSES)}")
        for alias, options in losses.items():
            if str(options.get("_type", "")).lower() in _CONFIDENCE_LOSSES:
                errors.append(f"loss.loss_terms.{alias}: {explanation}")
        mutators = current.get("mutators")
        if not isinstance(mutators, (list, tuple)):
            mutators = None
        for alias, _kind, options in active_mutators:
            if stages and mutators is not None and alias not in mutators:
                continue
            if variables(options.get("position_selector")) & {
                "bfactor", "plddt", "pae", "pde", "ptm", "iptm",
            }:
                errors.append(f"mutator.mutator_terms.{alias}.position_selector: {explanation}")

    explanation = f"structure-only evaluator(s) {sorted(structure_aliases)} provide no confidence metrics"
    config = per_residue_scores or {}
    if isinstance(config, Mapping):
        for key in ("terms", "mutator_score_terms"):
            terms = config.get(key) or {}
            if isinstance(terms, Mapping) and any(
                str(name).lower() in _CONFIDENCE_LOSSES and value is not None and value != 0
                for name, value in terms.items()
            ):
                errors.append(f"per_residue_scores.{key}: {explanation}")
    return errors
