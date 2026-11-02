"""Read assessment identities from the existing decision owner.

The returned dictionary is a call-local view.  It never advances accepted state.
"""
from astrolabe.core import global_config


def _decisions():
    try:
        return global_config.decisions
    except RuntimeError:
        return None


def _accepted(decisions):
    if decisions is None:
        return None
    return decisions.evaluation_results[decisions.last_accepted]


def score_runtime_view(runtime_config, evaluation_result, *, decisions=None, initial=False, selector=False):
    """Bind an actual input and the owner's accepted result to a local runtime view.

    Non-StandardScores callers retain their existing explicit runtime contract.
    """
    identity = getattr(evaluation_result, 'assessment_id', None)
    if not identity:
        return runtime_config
    if initial:
        accepted = evaluation_result
    else:
        if decisions is None:
            decisions = _decisions()
        accepted = _accepted(decisions)
        if accepted is None:
            return runtime_config
    accepted_id = getattr(accepted, 'assessment_id', None)
    if not accepted_id:
        raise ValueError('accepted assessment has no score identity')
    view = dict(runtime_config or {})
    snapshots = dict(view.get('score_snapshots') or {})
    snapshots['current_proposal'] = str(identity)
    snapshots['last_accepted'] = str(accepted_id)
    view['score_capture_snapshot'] = str(identity)
    assessments = [evaluation_result]
    if accepted_id != identity:
        assessments.append(accepted)
    snapshots.pop('reference', None)
    try:
        manager = global_config.stage_manager
    except RuntimeError:
        manager = None
    active = getattr(manager, 'transition_consistency_gate_context', None)
    if active is not None:
        references = (getattr(manager, 'current_transition_input_scores', None),
                      getattr(active, 'reference_scores', None))
        if selector:
            if any(getattr(reference, 'assessment_id', None) == identity for reference in references if reference is not None):
                snapshots['reference'] = str(identity)
        else:
            reference = getattr(active, 'reference_scores', None)
            reference_id = getattr(reference, 'assessment_id', None)
            if reference_id:
                snapshots['reference'] = str(reference_id)
                if all(getattr(item, 'assessment_id', None) != reference_id for item in assessments):
                    assessments.append(reference)
    view['score_snapshots'] = snapshots
    view['_score_evidence_assessments'] = tuple(assessments)
    if decisions is not None:
        view['_score_owned_records'] = getattr(decisions, 'score_records_by_assessment', {})
    return view
