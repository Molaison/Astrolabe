"""Read archived measurements; no benchmark execution, no GPU or scientific calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=Path('evidence/runtime-20261103/cpu-three-mode-02'))
    args = parser.parse_args()
    report = json.loads((args.evidence / 'report.json').read_text())
    rows = []
    for run in report['runs']:
        trial = Path(run['directory']).name
        directory = args.evidence / trial
        trace_file = directory / 'runtime.jsonl'
        row = {k: run[k] for k in ('mode', 'seconds', 'completed', 'valid_workload')}
        row['trial'] = trial
        if trace_file.exists():
            trace = [json.loads(line) for line in trace_file.read_text().splitlines() if line]
            samples = [json.loads(line) for line in (directory / 'telemetry.jsonl').read_text().splitlines() if line]
            launcher_start = samples[0]['monotonic_ns'] / 1e9 - samples[0]['elapsed_seconds']
            broker_start = next(x['time'] for x in trace if x['event'] == 'broker_started')
            trajectories = [x for x in trace if x['event'] == 'trajectory_complete']
            durations = [x['duration'] for x in trajectories]
            stage_durations = [x['duration'] for x in trace if x['event'] == 'stage']
            row.update(
                event_counts=dict(Counter(x['event'] for x in trace)),
                pre_broker_seconds=broker_start - launcher_start,
                trajectory_seconds=durations,
                trajectory_sum_seconds=sum(durations),
                measured_stage_sum_seconds=sum(stage_durations),
                stage_fraction_of_trajectory_time=sum(stage_durations) / sum(durations),
                other_seconds=run['seconds'] - (broker_start-launcher_start) - sum(durations),
                note='process duration includes cold startup and shutdown; stage timings omit surrounding Python control and IO',
            )
        rows.append(row)
    result = {
        'kind': 'derived_archived_evidence_not_new_benchmark',
        'source_protocol_sha256': report['protocol_sha256'],
        'source': str(args.evidence),
        'rows': rows,
        'summary': {mode: {
            'median_end_to_end_seconds': statistics.median(x['seconds'] for x in rows if x['mode'] == mode),
            'median_pre_broker_seconds': statistics.median(x['pre_broker_seconds'] for x in rows if x['mode'] == mode),
            'median_trajectory_seconds': statistics.median(d for x in rows if x['mode'] == mode for d in x['trajectory_seconds']),
            'measured_stage_seconds_all_runs': sum(x['measured_stage_sum_seconds'] for x in rows if x['mode'] == mode),
        } for mode in ('resident_serial', 'resident_adaptive')},
    }
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
