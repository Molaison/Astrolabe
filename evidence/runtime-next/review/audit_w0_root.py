"""Lead's independent audit of exported W0 files; never starts a workload."""
import argparse
import collections
import csv
import hashlib
import json
import pathlib
import statistics

p = argparse.ArgumentParser()
p.add_argument('artifact', type=pathlib.Path)
p.add_argument('output', type=pathlib.Path)
a = p.parse_args()
root = a.artifact / 'comparison'
report = json.loads((root / 'report.json').read_text())
reference = {}
totals = collections.Counter()
groups = collections.defaultdict(list)
audit = []
for run in report['runs']:
    directory = root / pathlib.Path(run['directory']).name
    assert run['exit_code'] == 0 and run['valid_workload']
    assert run['process_release']['release_confirmed']
    assert not run['process_release']['remaining_identities']
    identities = []
    counts = collections.Counter()
    for provenance in sorted(directory.rglob('provenance.json')):
        pd = json.loads(provenance.read_text())
        assert pd['state'] == 'completed' and pd['terminal']
        identity = (pd['run']['repeat_index'], pd['run']['seed'])
        identities.append(identity)
        with provenance.with_name('scores.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert [row['step'] for row in rows] == ['0', '1', '2']
        assert rows == reference.setdefault(identity, rows)
        for row in rows:
            initial = row['step'] == '0'
            accepted = row['accepted'] == 'True'
            counts['initializations' if initial else 'mutation_attempts'] += 1
            counts['accepted_initializations' if initial else 'accepted_mutations'] += int(accepted)
        counts['trajectories'] += 1
    assert sorted(identities) == [(0, 17), (1, 18), (2, 19)]
    events = [json.loads(line) for line in (directory / 'runtime.jsonl').read_text().splitlines() if line.strip()]
    phases = collections.Counter(e['phase'] for e in events if e['event'] == 'stage')
    assert all(phases[phase] == 9 for phase in ('cpu_prepare', 'cpu_execute', 'cpu_finalize'))
    counts['evaluations'] = phases['cpu_execute']
    totals.update(counts)
    groups[(run['variant'], run['mode'])].append(run)
    audit.append({'directory': directory.name, 'repetition': run['repetition'],
                  'seconds': run['seconds'], 'counts': dict(counts), 'phases': dict(phases),
                  'raw_csv_exact': True, 'process_release_confirmed': True})
assert len(audit) == 30 and totals['trajectories'] == 90 and totals['evaluations'] == 270
assert totals['accepted_mutations'] == 0 and totals['mutation_attempts'] == 180
comparisons = []
for mode in ('direct', 'resident_serial', 'resident_adaptive'):
    b = sorted(groups['baseline', mode], key=lambda r: r['repetition'])
    c = sorted(groups['candidate', mode], key=lambda r: r['repetition'])
    assert len(b) == len(c) == 5
    bs = [r['seconds'] for r in b]
    cs = [r['seconds'] for r in c]
    ratios = [x/y for x,y in zip(bs,cs)]
    comparisons.append({'mode': mode, 'baseline_seconds': bs, 'candidate_seconds': cs,
        'baseline_median': statistics.median(bs), 'candidate_median': statistics.median(cs),
        'median_wall_reduction_percent': 100*(1-statistics.median(cs)/statistics.median(bs)),
        'aggregate_evaluations_per_second_before': 45/sum(bs),
        'aggregate_evaluations_per_second_after': 45/sum(cs),
        'aggregate_throughput_change_percent': 100*(sum(bs)/sum(cs)-1),
        'paired_speed_ratios': ratios, 'all_five_faster': all(r>1 for r in ratios),
        'strict_range_separation': max(cs)<min(bs),
        'paired_range_is_uncertainty_not_confidence_interval': [min(ratios),max(ratios)]})
result = {'status':'DONE', 'artifact': str(a.artifact),
          'report_sha256': hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),
          'totals':dict(totals), 'raw_csv_exact_for_each_identity':True,
          'comparisons':comparisons, 'trials':audit,
          'limits':['CPU Dummy/MCMC only','no accepted mutation progress in W0',
                    'elapsed/process-release records are the original frozen observer measurements',
                    'not a GPU, coordinate, checkpoint or pretrained certificate']}
a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:result[k] for k in ('status','totals','comparisons')},indent=2))
