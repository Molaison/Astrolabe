"""Lead's direct raw-file audit, independent of C's benchmark reader."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import argparse
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET

ROOT = Path('/workspace/scratch/1a0b1639644d/cloud-20261104')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw', type=Path, default=ROOT / 'control/final-head-c')
parser.add_argument('--output', type=Path, default=ROOT / 'control/final-head-root')
args = parser.parse_args()
RAW = args.raw.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
errors = []
inputs = {}

def check(value, label):
    if not value:
        errors.append(label)

def read_json(path):
    content = path.read_bytes()
    inputs[str(path.relative_to(RAW))] = hashlib.sha256(content).hexdigest()
    return json.loads(content)

junit = {}
for kind, expected in [('cpu', 378), ('torch', 181)]:
    path = RAW / (kind + '-artifact/regressions/junit.xml')
    inputs[str(path.relative_to(RAW))] = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = ET.parse(path).getroot().findall('.//testcase')
    counts = {'cases': len(cases), 'failure': sum(len(x.findall('failure')) for x in cases),
              'error': sum(len(x.findall('error')) for x in cases),
              'skipped': sum(len(x.findall('skipped')) for x in cases)}
    check(counts == {'cases': expected, 'failure': 0, 'error': 0, 'skipped': 0}, kind + ' JUnit counts')
    junit[kind] = counts

comp = RAW / 'cpu-artifact/comparison'
report = read_json(comp / 'report.json')
check(report['candidate_commit'] == '1bdaf3a53227c9197b930dc00a01b752acd90067', 'final HEAD')
check(report['baseline_commit'] == 'd66ee714f2e755948534f95d9d125287ce8dbd91', 'baseline HEAD')
check(report['protocol_sha256'] == 'f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5', 'W0 protocol')
check(len(report['runs']) == 30, '30 trials')
reference = {}
counts = Counter()
timings = defaultdict(list)
for run in report['runs']:
    trial = comp / Path(run['directory']).name
    check(trial.is_dir(), str(trial) + ' exists')
    check(hashlib.sha256((trial / 'config.yaml').read_bytes()).hexdigest() == run['config_sha256'], trial.name + ' config')
    csvs = sorted(trial.rglob('scores.csv'))
    check(len(csvs) == 3, trial.name + ' three trajectories')
    for path in csvs:
        content = path.read_bytes()
        inputs[str(path.relative_to(RAW))] = hashlib.sha256(content).hexdigest()
        rows = list(csv.DictReader(content.decode().splitlines()))
        identity = path.parent.name
        reference.setdefault(identity, rows)
        check(rows == reference[identity], trial.name + '/' + identity + ' scientific CSV exact')
        check(len(rows) == 3, trial.name + '/' + identity + ' three rows')
        counts['trajectories'] += 1
        counts['rows'] += len(rows)
        counts['accepted_initializations'] += sum(x['accepted'] == 'True' and x['step'] == '0' for x in rows)
        counts['accepted_mutations'] += sum(x['accepted'] == 'True' and x['step'] != '0' for x in rows)
        counts['mutation_attempts'] += sum(x['step'] != '0' for x in rows)
    events = [json.loads(x) for x in (trial / 'runtime.jsonl').read_text().splitlines() if x]
    stages = Counter(x['phase'] for x in events if x.get('event') == 'stage')
    check(stages == {'cpu_prepare': 9, 'cpu_execute': 9, 'cpu_finalize': 9}, trial.name + ' actual stages')
    counts.update(stages)
    release = run['process_release']
    check(release['release_confirmed'] and release['observed_tree_empty'] and not release['remaining_identities']
          and not release['observation_errors'] and not release['unverified_process_groups'], trial.name + ' observed release')
    check(run['exit_code'] == 0 and run['valid_workload'] and not run['failures'], trial.name + ' validity')
    check(run['allocated_gpu_seconds'] == 0, trial.name + ' CPU only')
    if run['mode'] != 'direct':
        check(run['runtime_diagnostics']['all_registered_residency_released'] is True, trial.name + ' residency released')
        counts['resident_releases'] += 1
    timings[(run['variant'], run['mode'])].append((run['repetition'], run['seconds']))

performance = {}
for mode in ('direct', 'resident_serial', 'resident_adaptive'):
    base = [x[1] for x in sorted(timings[('baseline', mode)])]
    cand = [x[1] for x in sorted(timings[('candidate', mode)])]
    check(len(base) == len(cand) == 5, mode + ' five pairs')
    performance[mode] = {'baseline_seconds': base, 'candidate_seconds': cand,
                         'baseline_median': statistics.median(base), 'candidate_median': statistics.median(cand),
                         'baseline_eval_per_s': 45 / sum(base), 'candidate_eval_per_s': 45 / sum(cand),
                         'aggregate_gain_percent': (sum(base) / sum(cand) - 1) * 100,
                         'paired_speed_ratios': [b / c for b, c in zip(base, cand)],
                         'faster_pairs': sum(c < b for b, c in zip(base, cand)),
                         'ranges_separated': max(cand) < min(base)}
check(dict(counts) == {'trajectories': 90, 'rows': 270, 'accepted_initializations': 90,
                       'accepted_mutations': 0, 'mutation_attempts': 180, 'cpu_prepare': 270,
                       'cpu_execute': 270, 'cpu_finalize': 270, 'resident_releases': 20}, 'complete independent counts')

positive = RAW / 'scientific-positive-01'
positive_rows = {}
states = {}
for side in ('baseline', 'candidate'):
    path = next((positive / side).rglob('scores.csv'))
    positive_rows[side] = list(csv.DictReader(path.read_text().splitlines()))
    inputs[str(path.relative_to(RAW))] = hashlib.sha256(path.read_bytes()).hexdigest()
    states[side] = read_json(positive / side / 'terminal-scientific-state.json')
check(positive_rows['baseline'] == positive_rows['candidate'], 'positive scientific CSV exact')
check(states['baseline'] == states['candidate'], 'positive complete scientific state and CPU RNG exact')
rows = positive_rows['candidate']
check(len(rows) == 5, 'positive five rows')
check(sum(x['accepted'] == 'True' and x['acceptance_gate_passed'] != '' for x in rows) == 1,
      'positive one accepted mutation')
check(sum(x['note'] == 'enter stage' for x in rows) == 1, 'positive one stage transition')
check(sum(x['note'] == 're-evaluate due to stage change' for x in rows) == 1, 'positive one score recomputation')

result = {'status': 'PASS' if not errors else 'FAIL', 'errors': errors, 'source_head': report['candidate_commit'],
          'junit': junit, 'w0_counts': dict(counts), 'performance': performance,
          'positive_control': {'rows_exact': positive_rows['baseline'] == positive_rows['candidate'],
                               'state_exact': states['baseline'] == states['candidate'], 'scientific_rows': len(rows)},
          'scope': 'Lead independently reads original JUnit, every raw CSV/stage stream, config digest, observed release, raw timing, and positive terminal state. No new inference or benchmark.'}
(OUT / 'raw-independent-audit.json').write_text(json.dumps(result, indent=2) + '\n')
(OUT / 'input-digests.json').write_text(json.dumps(inputs, indent=2) + '\n')
print(json.dumps(result, indent=2))
raise SystemExit(bool(errors))
