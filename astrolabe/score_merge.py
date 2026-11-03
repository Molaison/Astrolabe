"""Merge independent StandardScores without guessing candidate correspondence."""
from pathlib import Path
import re

from astrolabe.evaluators.utils import StandardScores, ComplexPoses
from astrolabe.score_contributions import normalize_score_contributions


def _candidate_key(candidate):
    if candidate.candidate_key is not None:
        return candidate.candidate_key
    # Existing AF3/Protenix adapters encode scientific sample identity in
    # their candidate artifact names. Never infer correspondence by list index.
    keys = set()
    for name in candidate.names:
        path = candidate.complex_paths.get(name)
        if path is None:
            raise ValueError('candidate is missing its physical structure path')
        basename = Path(path).name
        match = re.search(r'_candidate_seed(\d+)_sample(\d+)_model\.(?:cif|pdb)$', basename)
        if match:
            keys.add(('af3', int(match.group(1)), int(match.group(2))))
            continue
        match = re.search(r'_candidate_(\d+)_model\.(?:cif|pdb)$', basename)
        if match:
            keys.add(('protenix', int(match.group(1))))
            continue
        raise ValueError('candidate merge requires producer-supplied candidate_key or a recognized seed/sample artifact name')
    if len(keys) != 1:
        raise ValueError('candidate contains inconsistent seed/sample identities')
    return next(iter(keys))


def merge_standard_scores(results):
    if not results:
        raise ValueError('Cannot merge empty evaluation results.')
    if len(results) == 1:
        return results[0]
    names = [name for result in results for name in result.names]
    if len(set(names)) != len(names):
        raise ValueError('cannot merge duplicate oligomer names')
    fields = set(StandardScores.__dataclass_fields__) - {
        'assessment_id', 'names', 'per_proto_scores', 'per_proto_score_terms', 'complex_poses',
        'score_contributions', 'candidate_scores', 'candidate_results', 'candidate_key',
        'member_assessment_ids', 'pre_eval_hook_reports', 'post_eval_hook_reports', 'score_records',
    }
    merged = {'names': names}
    for field in fields:
        combined = None
        for result in results:
            value = getattr(result, field, None)
            if isinstance(value, dict):
                if combined is None:
                    combined = {}
                if not isinstance(combined, dict) or set(combined).intersection(value):
                    raise ValueError(f'duplicate or incompatible {field} score scope')
                combined.update(value)
            elif value is not None and combined is None:
                combined = value
        merged[field] = combined
    per_proto, terms, contributions, records, poses, members = {}, {}, [], [], {}, {}
    pre_reports, post_reports = {}, {}
    for result in results:
        for proto, value in (result.per_proto_scores or {}).items():
            # Retain existing legacy first-protomer aggregation semantics.
            per_proto.setdefault(proto, value)
        for proto, metric_map in (result.per_proto_score_terms or {}).items():
            terms.setdefault(proto, {}).update(metric_map)
        contributions.extend(normalize_score_contributions(result.score_contributions))
        records.extend(result.score_records or ())
        result_poses = getattr(result.complex_poses, 'poses', None)
        if isinstance(result_poses, dict):
            poses.update(result_poses)
        for name in result.names:
            members[name] = (result.member_assessment_ids or {}).get(name, result.assessment_id)
        for target, source in ((pre_reports, result.pre_eval_hook_reports), (post_reports, result.post_eval_hook_reports)):
            if source:
                if set(target).intersection(source):
                    raise ValueError('duplicate hook report scope')
                target.update(source)
    merged.update(per_proto_scores=per_proto or None, per_proto_score_terms=terms or None,
                  score_contributions=contributions or None, score_records=tuple(records) if records else None,
                  member_assessment_ids=members, pre_eval_hook_reports=pre_reports or None,
                  post_eval_hook_reports=post_reports or None,
                  complex_poses=ComplexPoses(poses) if poses else None)
    keys = [result.candidate_key for result in results]
    if any(key is not None for key in keys):
        if any(key != keys[0] for key in keys):
            raise ValueError('cannot merge different candidate keys')
        merged['candidate_key'] = keys[0]
    for field in ('candidate_scores', 'candidate_results'):
        collections = [getattr(result, field) for result in results]
        if not any(collections):
            continue
        if not all(collections):
            raise ValueError(f'cannot merge incomplete {field} collections')
        by_result = []
        for collection in collections:
            lookup = {}
            for candidate in collection:
                key = _candidate_key(candidate)
                try:
                    if key in lookup:
                        raise ValueError(f'duplicate {field} candidate_key')
                    lookup[key] = candidate
                except TypeError as exc:
                    raise ValueError('candidate_key must be hashable') from exc
            by_result.append(lookup)
        if any(set(lookup) != set(by_result[0]) for lookup in by_result[1:]):
            raise ValueError(f'{field} candidate_key sets differ across contexts')
        # Preserve the first producer's order; correspondence is always by key.
        merged_candidates = []
        for key in by_result[0]:
            candidate = merge_standard_scores([lookup[key] for lookup in by_result])
            candidate.candidate_key = key
            merged_candidates.append(candidate)
        merged[field] = merged_candidates
    return StandardScores(**merged)
