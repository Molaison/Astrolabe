"""Explicit, dimension-independent selection and pre-inference sequence limits."""
from __future__ import annotations
import math

DIRECTIONS = dict(interface_ipae=-1, interface_ipsae=1, interface_sc=1,
                  interface_holes=-1, interface_buns=-1)


def sequence_rejections(sequence, parent, root, excluded, max_total_mutations=None):
    if len(sequence) != len(parent) or len(sequence) != len(root):
        raise ValueError('Candidate/parent/root sequence lengths differ')
    reasons = []
    forbidden = set(excluded)
    if any(a != b and b in forbidden for a, b in zip(root, sequence)):
        reasons.append('excluded_amino_acid')
    if max_total_mutations is not None and sum(a != b for a,b in zip(root,sequence)) > max_total_mutations:
        reasons.append('total_mutation_limit')
    return reasons


def candidate_rejections(candidate, root_metrics, gates):
    """Configured exploration limits, not empirical claims about binding quality."""
    reasons = []
    for name, limit in (gates.get('geometry_max') or {}).items():
        value = float(candidate['geometry'][name])
        if not math.isfinite(value):
            raise ValueError('Nonfinite geometry')
        if value > float(limit):
            reasons.append(name)
    for name, factor in (gates.get('root_metric_max_ratio') or {}).items():
        if candidate['metrics'][name] > root_metrics[name] * float(factor):
            reasons.append(name)
    for name, factor in (gates.get('root_metric_min_ratio') or {}).items():
        if candidate['metrics'][name] < root_metrics[name] * float(factor):
            reasons.append(name)
    for name, delta in (gates.get('root_metric_max_delta') or {}).items():
        if candidate['metrics'][name] > root_metrics[name] + float(delta):
            reasons.append(name)
    return reasons


def pareto_select(candidates, width, holes_ranking='raw_sum'):
    """Nondominated fronts, then range-normalized crowding, then stable ID.

    Each metric's monotone positive affine rescaling preserves this order.
    Holes can use the explicitly requested native per-atom mean; raw sums stay recorded.
    """
    unique = {}
    for candidate in candidates:
        if candidate.get('status') != 'done' or not candidate.get('eligible', True):
            continue
        if not all(math.isfinite(float(candidate['metrics'][m])) for m in DIRECTIONS):
            raise ValueError('Pareto selection requires complete finite metrics')
        sequence = candidate['sequence']
        if sequence not in unique or candidate['id'] < unique[sequence]['id']:
            unique[sequence] = candidate
    remaining = sorted(unique.values(), key=lambda c:c['id'])
    if holes_ranking not in {'raw_sum','per_atom'}:
        raise ValueError('Unknown holes ranking definition')
    def metric(candidate,name):
        if name=='interface_holes' and holes_ranking=='per_atom':
            details=candidate['holes']
            if details['atom_count']<1 or not math.isfinite(details['mean']):
                raise ValueError('Invalid native holes normalization')
            return details['mean']
        return candidate['metrics'][name]
    values = {c['id']:tuple(metric(c,m)*d for m,d in DIRECTIONS.items()) for c in remaining}
    def dominates(a,b):
        return all(x>=y for x,y in zip(a,b)) and any(x>y for x,y in zip(a,b))
    result = []
    rank = 0
    while remaining and len(result)<width:
        front = [c for c in remaining if not any(dominates(values[o['id']],values[c['id']])
                    for o in remaining if o['id']!=c['id'])]
        crowding = {c['id']:0.0 for c in front}
        for name in DIRECTIONS:
            ordered = sorted(front,key=lambda c:(metric(c,name),c['id']))
            low,high=metric(ordered[0],name),metric(ordered[-1],name)
            if high == low:
                continue
            crowding[ordered[0]['id']] += (metric(ordered[1],name)-low)/(high-low)
            crowding[ordered[-1]['id']] += (high-metric(ordered[-2],name))/(high-low)
            for i in range(1,len(ordered)-1):
                crowding[ordered[i]['id']] += (metric(ordered[i+1],name)-metric(ordered[i-1],name))/(high-low)
        for c in front:
            value = crowding[c['id']]
            c['selection'] = dict(pareto_front=rank,crowding_distance=value,holes_ranking=holes_ranking)
        result.extend(sorted(front,key=lambda c:(-crowding[c['id']],c.get('total_mutations',0),c['id']))[:width-len(result)])
        ids={c['id'] for c in front}
        remaining=[c for c in remaining if c['id'] not in ids]
        rank += 1
    return result
