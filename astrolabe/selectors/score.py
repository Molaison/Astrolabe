"""Position selectors over explicitly scoped immutable score evidence."""
from collections import defaultdict
from collections.abc import Mapping
import numpy as np

from astrolabe.score_evidence import ScoreQuery, query_score
from astrolabe.selectors.base import ExplicitPositionSelector


def _records(evaluation_result, protomers, runtime_config):
    records = list(getattr(evaluation_result, 'score_records', None) or [])
    records.extend(runtime_config.get('score_records', []) or [])
    capture = runtime_config.get('score_capture_snapshot')
    if capture:
        from astrolabe.score_evidence import records_from_standard_scores
        records.extend(records_from_standard_scores(evaluation_result, capture, protomers=protomers))
    catalog = runtime_config.get('score_catalog', []) or []
    if catalog:
        from astrolabe.score_evidence import records_from_contributions
        from astrolabe.score_contributions import score_contributions_from_runtime_config
        if not capture:
            raise ValueError('score_capture_snapshot is required for catalog capture')
        contributions = list(getattr(evaluation_result, 'score_contributions', None) or [])
        if '_score_owned_records' not in runtime_config:
            contributions.extend(score_contributions_from_runtime_config(runtime_config))
        records.extend(records_from_contributions(contributions, catalog, capture))
        records.extend((runtime_config.get('_score_owned_records') or {}).get(str(capture), ()))
    snapshots = runtime_config.get('score_snapshots') or getattr(evaluation_result, 'score_snapshots', None) or {}
    return records, snapshots


def _observation(config, records, snapshots, protomers, *, allow_equal_weights=False, selector_member_snapshot=None):
    query = ScoreQuery(**dict(config['query']))
    broadcast = config.get('broadcast', False)
    if (broadcast and query.axes) or (not broadcast and len(query.axes) != 1):
        raise ValueError('score selector needs one position axis or an explicit scalar broadcast')
    values = query_score(records, query, snapshots=snapshots, generating_selector=True,
                         selector_member_snapshot=selector_member_snapshot)
    target = snapshots.get(query.snapshot, query.snapshot)
    record = next(r for r in records if r.spec.name == query.name and r.spec.source == query.source
                  and r.snapshot == target and r.subject == query.subject and r.relation == query.relation
                  and (query.entity is None or r.spec.entity == query.entity)
                  and (query.entity_scope is None or r.spec.entity_scope == query.entity_scope))
    mapping = config['mapping']
    if broadcast:
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError('scalar broadcast requires an explicit nonempty position mapping')
        labels = tuple(mapping)
        values = np.full(len(labels), values)
    else:
        axis = record.spec.axes.index(query.axes[0])
        labels = tuple((query.select or {}).get(query.axes[0], record.coordinates[axis]))
    if mapping == 'auto':
        mapping = {}
        for label in labels:
            parts = str(label).split(':')
            if query.axes[0] == 'design_position' and len(parts) == 2 and parts[1].isdigit():
                mapping[label] = (parts[0], int(parts[1]) - 1)
            elif len(parts) == 4 and parts[2].isdigit():
                mapping[label] = (parts[3], int(parts[2]) - 1)
            else:
                raise ValueError('automatic mapping requires unambiguous residue or design-position labels')
    if not isinstance(mapping, Mapping) or set(mapping) != set(labels):
        raise ValueError('position mapping must cover every selected physical coordinate')
    configured_weights = config.get('position_weights')
    if configured_weights == 'equal' and allow_equal_weights:
        weights = {label: 1.0 for label in labels}
    elif configured_weights is None:
        weights = {label: 1.0 for label in labels}
    elif isinstance(configured_weights, Mapping) and set(configured_weights) == set(labels):
        weights = configured_weights
    else:
        raise ValueError('position weights must cover every coordinate')
    entries = []
    for label, value in zip(labels, values):
        proto, index = mapping[label]
        if proto not in protomers or not 0 <= int(index) < int(protomers[proto].length):
            raise ValueError('physical coordinate maps outside design positions')
        weight = float(weights[label])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('position weights must be finite and nonnegative')
        entries.append((str(proto), int(index), float(value), weight))
    return record, entries, configured_weights is not None


def _canonical_positions(protomers):
    """Project protomer offsets onto actual shared SegmentManager variables."""
    from astrolabe.core import global_config
    try:
        segments, manager = global_config.segments, global_config.segment_manager
    except RuntimeError:
        segments, manager = {}, None
    positions = {}
    for name, protomer in protomers.items():
        components = getattr(protomer, 'components', None)
        slices = getattr(protomer, 'components_slices', None)
        if components and slices and manager is not None:
            for component, component_slice in zip(components, slices, strict=True):
                segment = segments[component]
                canonical = manager.primary_names[id(segment)]
                for offset, index in enumerate(range(component_slice.start, component_slice.stop)):
                    positions[(str(name), index)] = ('segment', canonical, offset)
            if sum(1 for proto, _ in positions if proto == str(name)) != int(protomer.length):
                raise ValueError('incomplete segment-to-protomer position mapping')
        else:
            for index in range(int(protomer.length)):
                positions[(str(name), index)] = ('protomer', str(name), index)
    return positions


def _multi_context(config, records, snapshots, protomers, evaluation_result):
    observations = config['observations']
    if not isinstance(observations, (list, tuple)) and not hasattr(observations, '__iter__'):
        raise ValueError('observations must be a nonempty sequence')
    observations = list(observations)
    if len(observations) < 2:
        raise ValueError('multi-context aggregation requires at least two observations')
    aggregation = dict(config.get('aggregation') or {})
    if set(aggregation) != {'reduction', 'missing', 'count'}:
        raise ValueError('aggregation must explicitly specify reduction, missing and count')
    reduction, missing, count = (aggregation['reduction'], aggregation['missing'], aggregation['count'])
    if reduction not in {'mean', 'sum'} or missing not in {'error', 'ignore'} or count not in {'context', 'physical'}:
        raise ValueError('invalid multi-context aggregation policy')
    canonical = _canonical_positions(protomers)
    groups = []
    seen_contexts = set()
    for item in observations:
        item = dict(item)
        context = str(item.get('context') or '')
        if not context or context in seen_contexts or 'weight' not in item:
            raise ValueError('each observation requires a unique context and explicit weight')
        seen_contexts.add(context)
        context_weight = float(item['weight'])
        if not np.isfinite(context_weight) or context_weight < 0:
            raise ValueError('context weights must be finite and nonnegative')
        query_config = dict(item['query'])
        observation_snapshots = snapshots
        member_snapshot = None
        if query_config.get('snapshot') == 'member':
            if getattr(evaluation_result, 'assessment_id', None) != snapshots.get('last_accepted'):
                raise ValueError('member evidence requires the actual last accepted assessment')
            member_snapshot = (getattr(evaluation_result, 'member_assessment_ids', None) or {}).get(context)
            if not member_snapshot:
                raise ValueError(f'accepted assessment has no member identity for {context!r}')
            observation_snapshots = dict(snapshots)
            observation_snapshots['member'] = member_snapshot
        record, entries, explicit_position_weights = _observation(
            item, records, observation_snapshots, protomers, allow_equal_weights=True,
            selector_member_snapshot=member_snapshot,
        )
        if record.spec.entity_scope == 'oligo' and record.spec.entity != context:
            raise ValueError('observation context differs from score entity')
        selected_axes = query_config.get('select') or {}
        if 'context' in record.spec.axes:
            axis = record.spec.axes.index('context')
            context_labels = tuple(selected_axes.get('context', record.coordinates[axis]))
            if context_labels != (context,):
                raise ValueError('observation must select exactly its declared context')
        elif record.spec.entity_scope != 'oligo' and record.spec.entity != context:
            axis_name = tuple(query_config.get('axes') or ())
            axis = record.spec.axes.index(axis_name[0])
            labels = tuple(selected_axes.get(axis_name[0], record.coordinates[axis]))
            if not labels or not all(len(str(label).split(':')) >= 3 and str(label).split(':')[0] == context for label in labels):
                raise ValueError('observation lacks evidence for its declared context')
        by_position = defaultdict(list)
        for proto, index, value, weight in entries:
            by_position[canonical[(proto, index)]].append((value, weight))
        if not explicit_position_weights and any(len(values) > 1 for values in by_position.values()):
            raise ValueError('repeated physical positions require explicit position_weights')
        if any(sum(weight for _, weight in values) <= 0 for values in by_position.values()):
            raise ValueError('position weights must sum to a positive value')
        groups.append((context_weight, by_position))
    active = [(weight, positions) for weight, positions in groups if weight > 0]
    if not active:
        raise ValueError('at least one context weight must be positive')
    keys = set().union(*(set(positions) for _, positions in active))
    if missing == 'error' and any(set(positions) != keys for _, positions in active):
        raise ValueError('a canonical position is missing from an active context')
    masks = {name: np.zeros(proto.length, dtype=bool) for name, proto in protomers.items()}
    threshold = float(config['threshold'])
    if not np.isfinite(threshold):
        raise ValueError('invalid score threshold')
    operation = config.get('comparison', 'ge')
    if operation not in {'ge', 'gt', 'le', 'lt'}:
        raise ValueError('unknown score comparison')
    compare = getattr(np, {'ge':'greater_equal', 'gt':'greater', 'le':'less_equal', 'lt':'less'}[operation])
    selected = {}
    for key in keys:
        numerator = denominator = 0.0
        for context_weight, positions in active:
            if key not in positions:
                continue
            values = positions[key]
            local_weight = sum(weight for _, weight in values)
            local_sum = sum(value * weight for value, weight in values)
            if count == 'context':
                numerator += context_weight * local_sum / local_weight
                denominator += context_weight
            else:
                numerator += context_weight * local_sum
                denominator += context_weight * local_weight
        if denominator <= 0:
            raise ValueError('no positive aggregation weight for a position')
        score = numerator / denominator if reduction == 'mean' else numerator
        selected[key] = bool(compare(score, threshold))
    for destination, key in canonical.items():
        if key in selected:
            masks[destination[0]][destination[1]] = selected[key]
    return ExplicitPositionSelector(masks)


def score_position_selector(evaluation_result, protomers, name, *, runtime_config=None):
    runtime_config = runtime_config or {}
    configs = runtime_config.get('score_selector_queries') or getattr(evaluation_result, 'score_selector_queries', None) or {}
    if name not in configs:
        raise ValueError(f'unknown score selector query {name!r}')
    config = dict(configs[name])
    records, snapshots = _records(evaluation_result, protomers, runtime_config)
    if 'observations' in config:
        if 'query' in config:
            raise ValueError('choose either a single query or multiple observations')
        return _multi_context(config, records, snapshots, protomers, evaluation_result)
    _, entries, explicit_weights = _observation(config, records, snapshots, protomers)
    destinations = [(proto, index) for proto, index, _, _ in entries]
    if len(set(destinations)) != len(destinations) and not explicit_weights:
        raise ValueError('repeated physical positions require explicit weights')
    masks = {key: np.zeros(proto.length, dtype=bool) for key, proto in protomers.items()}
    sums, weight_sums = {}, {}
    for proto, index, value, weight in entries:
        key = (proto, index)
        sums[key] = sums.get(key, 0.0) + weight * value
        weight_sums[key] = weight_sums.get(key, 0.0) + weight
    threshold = float(config['threshold'])
    if not np.isfinite(threshold) or any(weight <= 0 for weight in weight_sums.values()):
        raise ValueError('invalid threshold or position weights')
    operation = config.get('comparison', 'ge')
    if operation not in {'ge', 'gt', 'le', 'lt'}:
        raise ValueError('unknown score comparison')
    for (proto, index), total in sums.items():
        masks[proto][index] = getattr(np, {'ge':'greater_equal', 'gt':'greater', 'le':'less_equal', 'lt':'less'}[operation])(total / weight_sums[(proto,index)], threshold)
    return ExplicitPositionSelector(masks)
