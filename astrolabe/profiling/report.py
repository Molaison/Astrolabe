"""Read-only, standard-library profiling reports; all timings are diagnostic.

Durations are nanoseconds unless a field states otherwise. Inclusive span sums
are not elapsed time or CPU work. GPU samples describe a device, not ownership.
"""

from __future__ import annotations

from collections import defaultdict
import html
import json
import math
import os
from pathlib import Path
import tempfile


_LIMITATIONS = [
    "Profiling can perturb the measured run; these are observational diagnostics.",
    "Host spans (including asynchronous GPU dispatch) are not device kernel durations or GPU concurrency.",
    "Event/span counts are instrumentation counts, not scientific sample counts.",
    "Device utilization and compute-process snapshots do not prove GPU isolation or exclusive ownership.",
    "Different hosts' wall clocks are not assumed synchronized; cross-host unions/concurrency are unknown.",
    "Missing or failed resource observations are unknown, never zero; sampled peaks are not continuous peaks.",
]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _ns(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _dict(value):
    return value if isinstance(value, dict) else {}


def _json(text):
    def reject(value):
        raise ValueError("non-finite JSON number: " + value)
    return json.loads(text, parse_constant=reject)


def _load(directory):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Session directory does not exist: {root}")
    warnings, records = [], []
    try:
        session = _json((root / "session.json").read_text(encoding="utf-8"))
        if not isinstance(session, dict):
            raise ValueError("session.json must contain an object")
    except (OSError, UnicodeError, ValueError) as exc:
        session = {}
        warnings.append(f"session.json unavailable or invalid: {exc}")
    files = sorted(root.glob("events-*.jsonl"))
    sidecars = sorted(root.glob("process-*.json"))
    processes = []
    for path in sidecars:
        try:
            process = _json(path.read_text(encoding="utf-8"))
            if not isinstance(process, dict):
                raise ValueError("process manifest must be an object")
            processes.append(process)
            for key in ("dropped", "dropped_limit", "dropped_io", "errors"):
                if process.get(key):
                    warnings.append(f"{path.name}: {key}={process[key]}")
            if process.get("status") != "closed":
                warnings.append(f"{path.name}: recorder is not closed; buffered/missing records are possible.")
        except (OSError, UnicodeError, ValueError) as exc:
            warnings.append(f"{path.name}: unavailable/invalid process manifest: {exc}")
    if not sidecars:
        warnings.append("No process manifests: dropped/unwritten event accounting is unknown.")
    if session.get("schema_version") != 1:
        warnings.append(f"Session schema_version={session.get('schema_version')!r} is not the supported version 1; interpretation is best effort.")
    if not files:
        warnings.append("No events-*.jsonl files found; absence is not evidence of zero work.")
    for path in files:
        parsed = 0
        try:
            with path.open("rb") as handle:
                for lineno, raw in enumerate(handle, 1):
                    where = f"{path.name}:{lineno}"
                    if not raw.endswith(b"\n"):
                        warnings.append(f"{where}: unterminated JSONL tail; file may be truncated or still being written.")
                    try:
                        row = _json(raw.decode("utf-8"))
                        if not isinstance(row, dict):
                            raise ValueError("record must be an object")
                        if row.get("kind") not in ("span", "event", "resource"):
                            raise ValueError("unknown/missing record kind")
                        if not _ns(row.get("ts_ns")):
                            raise ValueError("invalid/missing ts_ns")
                        if not isinstance(row.get("name"), str):
                            raise ValueError("invalid/missing name")
                        if not row.get("hostname") or row.get("pid") is None or row.get("tid") is None:
                            warnings.append(f"{where}: incomplete host/process/thread identity.")
                        records.append(row)
                        parsed += 1
                    except (UnicodeError, ValueError) as exc:
                        warnings.append(f"{where}: skipped invalid record: {exc}")
        except OSError as exc:
            warnings.append(f"Cannot read {path.name}: {exc}")
        owners = [p for p in processes if p.get("event_file") == path.name]
        if not owners:
            warnings.append(f"{path.name}: no matching process manifest; event loss is unknown.")
        for process in owners:
            if process.get("written_events") != parsed:
                warnings.append(f"{path.name}: parsed {parsed} records but recorder declares written_events={process.get('written_events')!r}.")
    for process in processes:
        if process.get("written_events") and process.get("event_file") not in {path.name for path in files}:
            warnings.append(f"Recorder declares written records in missing file {process.get('event_file')!r}.")
    return root, session, records, warnings, [root / "session.json", *files, *sidecars], processes


def _union(intervals):
    total, end = 0, None
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def _window(intervals):
    if not intervals:
        return {"start_ns": None, "end_ns": None, "duration_ns": None}
    start, end = min(a for a, _ in intervals), max(b for _, b in intervals)
    return {"start_ns": start, "end_ns": end, "duration_ns": end - start}


def _simultaneous(intervals):
    marks = sorted((point, delta) for start, end in intervals if end > start
                   for point, delta in ((start, 1), (end, -1)))
    active = peak = 0
    for _, delta in marks:
        active += delta
        peak = max(peak, active)
    return peak


def _host(row):
    return str(row.get("hostname") or "unknown")


def _identity(row):
    return (_host(row), str(row.get("pid", "unknown")), str(row.get("tid", "unknown")))


def _span_stats(rows):
    durations = sorted(row["duration_ns"] for row in rows)
    by_host = defaultdict(list)
    for row in rows:
        by_host[_host(row)].append((row["ts_ns"], row["ts_ns"] + row["duration_ns"]))
    cpu = [row.get("thread_cpu_ns", _dict(row.get("attrs")).get("thread_cpu_ns")) for row in rows]
    cpu_valid = [value for value in cpu if _ns(value)]
    n = len(rows)
    per_host = {host: dict(window=_window(intervals), union_active_wall_ns=_union(intervals),
                           max_simultaneous_host_spans=_simultaneous(intervals))
                for host, intervals in sorted(by_host.items())}
    single = next(iter(per_host.values())) if len(per_host) == 1 else {}
    return {
        "count": n,
        "errors": sum(bool(row.get("error") or _dict(row.get("attrs")).get("error")) for row in rows),
        "total_inclusive_wall_ns": sum(durations) if n else None,
        "union_active_wall_ns": single.get("union_active_wall_ns"),
        "mean_ns": sum(durations) / n if n else None,
        "p50_ns": durations[math.ceil(n * .50) - 1] if n else None,
        "p95_ns": durations[math.ceil(n * .95) - 1] if n else None,
        "max_ns": max(durations) if n else None,
        "thread_cpu_ns": sum(cpu_valid) if n and len(cpu_valid) == n else None,
        "observed_inclusive_thread_cpu_ns": sum(cpu_valid) if cpu_valid else None,
        "thread_cpu_observed_count": len(cpu_valid),
        "max_simultaneous_host_spans": single.get("max_simultaneous_host_spans"),
        "by_hostname": per_host,
    }


def _full_window(session, warnings):
    start = session.get("start_time_ns", session.get("start_wall_time_ns"))
    end = session.get("end_time_ns", session.get("end_wall_time_ns"))
    mono_start, mono_end = session.get("monotonic_origin_ns"), session.get("end_monotonic_ns")
    wall_duration = end - start if _ns(start) and _ns(end) else None
    mono_duration = mono_end - mono_start if _ns(mono_start) and _ns(mono_end) and mono_end >= mono_start else None
    discrepancy = wall_duration - mono_duration if wall_duration is not None and mono_duration is not None else None
    duration = mono_duration if mono_duration is not None else wall_duration if wall_duration is not None and wall_duration >= 0 else None
    if duration is None:
        warnings.append("Full session window is unknown/incomplete; no observed-span window is substituted.")
    elif mono_duration is None:
        warnings.append("Monotonic session elapsed time is unavailable; duration uses an explicit wall-clock fallback, which is vulnerable to clock adjustments.")
    if discrepancy:
        warnings.append(f"Wall elapsed minus monotonic elapsed is {discrepancy} ns (clock adjustment or boundary-capture skew); performance elapsed uses monotonic time. Wall-aligned resource weighting/coverage may be affected.")
    if wall_duration is None or wall_duration < 0:
        warnings.append("Wall alignment window is missing or reversed; wall-aligned resource coverage is unknown.")
    return {"start_ns": start if _ns(start) else None, "end_ns": end if _ns(end) else None,
            "duration_ns": duration,
            "duration_source": "monotonic" if mono_duration is not None else "wall_clock_fallback" if duration is not None else "unknown",
            "monotonic_duration_ns": mono_duration, "wall_duration_ns": wall_duration,
            "wall_minus_monotonic_ns": discrepancy}


def _gap_bound(session):
    config = _dict(session.get("config"))
    interval = config.get("sample_interval_s", config.get("resource_interval_s"))
    if _number(interval) and interval > 0:
        return int(interval * 2 * 1_000_000_000)
    return None


def _resource_stats(records, session, full, warnings):
    """Integrate only adjacent successful observations, bounded by cadence.

    A left-hold rectangle covers [sample_i, sample_i+1). Never extrapolate the
    final sample, bridge a failed sample, or bridge more than two configured
    sampling intervals. The 2x bound is an integration policy, not a performance
    acceptance threshold. Without a configured cadence, weighted means unknown.
    """
    gpu, cpu, tree_rss = defaultdict(list), defaultdict(list), defaultdict(list)
    gap = _gap_bound(session)
    resources = [row for row in records if row["kind"] == "resource"]
    stream_positions = defaultdict(int)
    failures = 0
    for row in sorted(resources, key=lambda item: item["ts_ns"]):
        attrs = _dict(row.get("attrs"))
        stream = (_host(row), row["name"])
        position = stream_positions[stream]
        stream_positions[stream] += 1
        failed = bool(row.get("error") or attrs.get("error") or attrs.get("status") not in (None, "ok"))
        if failed:
            failures += 1
            warnings.append(f"Resource sampling failure/partial observation {row['name']} at {row['ts_ns']}: {row.get('error') or attrs.get('error') or attrs.get('errors') or attrs.get('status')}")
            continue
        # GPU payload is the existing runtime.telemetry sampler schema.
        raw = attrs.get("devices", attrs.get("gpus"))
        if isinstance(raw, dict):
            raw = list(raw.values())
        if isinstance(raw, list):
            for device in raw:
                if isinstance(device, dict) and isinstance(device.get("uuid"), str):
                    gpu[(_host(row), device["uuid"])].append((row, device, position))
        cpu_data = attrs if row["name"] == "resource.cpu" else _dict(attrs.get("cpu"))
        processes = cpu_data.get("processes")
        if isinstance(processes, list) and processes and all(isinstance(p, dict) and _ns(p.get("rss_bytes")) for p in processes):
            tree_rss[_host(row)].append((row["ts_ns"], sum(p["rss_bytes"] for p in processes)))
        for process in processes if isinstance(processes, list) else []:
            if not isinstance(process, dict):
                continue
            epoch_kind = "start_ticks" if "start_ticks" in process else "create_time_s"
            epoch = process.get(epoch_kind)
            if process.get("pid") is None or not _number(epoch):
                warnings.append(f"CPU process identity/epoch unknown at {row['ts_ns']}; CPU work excluded.")
                continue
            cpu[(_host(row), str(process["pid"]), epoch_kind, epoch)].append((row, process, position))

    def acceptable(left, right):
        delta = right[0]["ts_ns"] - left[0]["ts_ns"]
        return (gap is not None and 0 < delta <= gap and right[2] == left[2] + 1
                and left[0]["name"] == right[0]["name"])

    def coverage(intervals, host):
        observed = _union(intervals)
        # Sampling timestamps align on wall time, so coverage uses that window,
        # not the separately reported monotonic performance duration.
        wall_duration = full["wall_duration_ns"]
        denominator = wall_duration if host == session.get("hostname") and wall_duration is not None and wall_duration >= 0 else None
        within = _union([(max(a, full["start_ns"]), min(b, full["end_ns"]))
                         for a, b in intervals if denominator and a < full["end_ns"] and b > full["start_ns"]])
        return dict(covered_wall_ns=observed,
                    full_session_denominator_ns=denominator,
                    coverage_denominator_clock="wall_alignment_not_performance_elapsed",
                    covered_within_full_session_ns=within if denominator is not None else None,
                    full_session_coverage_fraction=within / denominator if denominator else None)

    gpu_summary = []
    for (host, uuid), samples in sorted(gpu.items()):
        samples.sort(key=lambda pair: pair[0]["ts_ns"])
        intervals, weighted, peaks, snapshots = [], 0.0, [], []
        memory_intervals = []
        for row, device, _ in samples:
            total, free = device.get("total_bytes"), device.get("free_bytes")
            if _number(total) and _number(free) and 0 <= free <= total:
                peaks.append(total - free)
                memory_intervals.append((row["ts_ns"], row["ts_ns"]))
            snapshots.append(dict(ts_ns=row["ts_ns"], processes=device.get("processes"),
                                  process_identities=device.get("process_identities")))
        for left_sample, right_sample in zip(samples, samples[1:]):
            left, device, _ = left_sample
            right, next_device, _ = right_sample
            a, b = left["ts_ns"], right["ts_ns"]
            value, next_value = device.get("utilization"), next_device.get("utilization")
            if (acceptable(left_sample, right_sample) and _number(value) and 0 <= value <= 100
                    and _number(next_value) and 0 <= next_value <= 100):
                intervals.append((a, b))
                weighted += value * (b - a)
        covered = _union(intervals)
        gpu_summary.append(dict(hostname=host, uuid=uuid, sample_count=len(samples),
                                sample_window=_window([(r["ts_ns"], r["ts_ns"]) for r, _, _ in samples]),
                                utilization_time_weighted_mean_percent=weighted / covered if covered else None,
                                sampled_device_memory_peak_bytes=max(peaks) if peaks else None,
                                memory_observed_sample_count=len(peaks), memory_sample_window=_window(memory_intervals),
                                compute_process_observations=snapshots, isolation="unknown", **coverage(intervals, host)))
    cpu_summary = []
    all_cpu_intervals = defaultdict(list)
    for (host, pid, epoch_kind, epoch), samples in sorted(cpu.items()):
        samples.sort(key=lambda pair: pair[0]["ts_ns"])
        intervals, work = [], 0
        for left_sample, right_sample in zip(samples, samples[1:]):
            left, data, _ = left_sample
            right, next_data, _ = right_sample
            a, b = left["ts_ns"], right["ts_ns"]
            first, last = data.get("cpu_time_s"), next_data.get("cpu_time_s")
            if acceptable(left_sample, right_sample) and _number(first) and _number(last) and last >= first >= 0:
                intervals.append((a, b))
                work += round((last - first) * 1_000_000_000)
        covered = _union(intervals)
        all_cpu_intervals[host].extend(intervals)
        rss = [(row["ts_ns"], data["rss_bytes"]) for row, data, _ in samples if _ns(data.get("rss_bytes"))]
        cpu_summary.append(dict(hostname=host, pid=pid, process_epoch={epoch_kind: epoch}, sample_count=len(samples),
                                observed_process_cpu_work_ns=work if covered else None,
                                mean_busy_cores=work / covered if covered else None,
                                busy_cores_denominator_wall_ns=covered if covered else None,
                                sample_window=_window([(r["ts_ns"], r["ts_ns"]) for r, _, _ in samples]),
                                sampled_rss_peak_bytes=max((value for _, value in rss), default=None),
                                rss_observed_sample_count=len(rss), rss_sample_window=_window([(ts, ts) for ts, _ in rss]),
                                **coverage(intervals, host)))
    if not resources:
        warnings.append("No resource samples: CPU work, GPU utilization and memory peaks are unknown.")
    if resources and gap is None:
        warnings.append("Sampling cadence is unknown: no time-weighted resource means or sampled CPU work computed.")
    cpu_totals = []
    for host in sorted(set(all_cpu_intervals) | set(tree_rss)):
        intervals = all_cpu_intervals[host]
        known = [p["observed_process_cpu_work_ns"] for p in cpu_summary if p["hostname"] == host and p["observed_process_cpu_work_ns"] is not None]
        work = sum(known) if known else None
        covered = _union(intervals)
        rss = tree_rss[host]
        cpu_totals.append(dict(hostname=host, observed_total_process_cpu_work_ns=work,
                               observed_mean_busy_cores=work / covered if work is not None and covered else None,
                               busy_cores_denominator_wall_ns=covered if covered else None,
                               process_coverage_wall_sum_ns=sum(p["covered_wall_ns"] for p in cpu_summary if p["hostname"] == host),
                               sampled_tree_rss_sum_peak_bytes=max((value for _, value in rss), default=None),
                               rss_observed_sample_count=len(rss), rss_sample_window=_window([(ts, ts) for ts, _ in rss]),
                               **coverage(intervals, host)))
    return dict(sample_count=len(resources), failed_or_partial_sample_count=failures,
                cpu=cpu_summary or None, cpu_totals_by_hostname=cpu_totals or None, gpu=gpu_summary or None,
                max_integrated_gap_ns=gap,
                integration_policy="Adjacent successful samples; left hold; <=2x configured interval; no final extrapolation.",
                rss_scope="Sampled resident bytes only, not continuous peaks. Tree peak is the maximum same-snapshot process RSS sum, not the sum of per-process peaks; shared pages can be double-counted. Failed/partial or incomplete-RSS tree snapshots are excluded, not zero.",
                cpu_work_scope="Sum of matched process-epoch CPU deltas across accepted intervals. Partial observation only: exits, starts, unsampled/short-lived children and failed intervals are excluded, not zero. Coverage is any matched process, not proof of complete tree accounting.")


def _summary(root, session, records, warnings, processes):
    full = _full_window(session, warnings)
    if session.get("status") not in ("completed", "complete", "finished", "success", "closed"):
        warnings.append(f"Session status is {session.get('status', 'unknown')!r}; failed or unfinished sessions may have missing spans.")
    for key in ("warnings", "errors", "dropped", "dropped_limit", "dropped_io", "dropped_events", "dropped_records", "write_errors"):
        if session.get(key):
            warnings.append(f"Session {key}: {session[key]}")
    spans = []
    for row in records:
        if row["kind"] == "span":
            if not _ns(row.get("duration_ns")):
                warnings.append(f"Span {row['name']!r} at {row['ts_ns']} has unknown/invalid duration; excluded from timing statistics.")
            else:
                spans.append(row)
        if row.get("kind") == "event" and any(word in row["name"].lower() for word in ("drop", "error", "fail", "truncat")):
            warnings.append(f"Diagnostic event {row['name']}: {row.get('attrs', {})}")
    groups, lanes = defaultdict(list), defaultdict(list)
    for row in spans:
        groups[row["name"]].append(row)
        lanes[(*_identity(row), row["name"])].append(row)
    host_intervals = defaultdict(list)
    for row in spans:
        host_intervals[_host(row)].append((row["ts_ns"], row["ts_ns"] + row["duration_ns"]))
    # Only an explicitly named completion boundary can qualify here. Generic
    # dispatch spans must never be silently treated as completed inference.
    completed = [row for row in records if row["name"] == "evaluator.infer_completed"
                 and _ns(row.get("duration_ns")) and (row["kind"] == "span" or row["duration_ns"] > 0)
                 and not (row.get("error") or _dict(row.get("attrs")).get("error"))]
    completion_hosts = defaultdict(list)
    for row in completed:
        completion_hosts[_host(row)].append((row["ts_ns"], row["ts_ns"] + row["duration_ns"]))
    if not completed:
        warnings.append("No timed evaluator.infer_completed records: inference host-completion timing is unknown.")
    resources = _resource_stats(records, session, full, warnings)
    return dict(
        report_schema_version=1, classification="observational_diagnostic", directory=str(root),
        session=session, process_manifests=processes, full_run_window=full,
        observed_span_windows_by_hostname={host: dict(window=_window(intervals), union_active_wall_ns=_union(intervals))
                                           for host, intervals in sorted(host_intervals.items())},
        inference_host_completion={"meaning": "Explicit completed host boundary only; not kernel time, throughput, or device concurrency.",
                                   "record_count": len(completed),
                                   "by_hostname": {host: dict(window=_window(intervals), union_active_wall_ns=_union(intervals))
                                                   for host, intervals in sorted(completion_hosts.items())}},
        modules={name: _span_stats(rows) for name, rows in sorted(groups.items())},
        thread_modules=[dict(hostname=host, pid=pid, tid=tid, name=name, **_span_stats(rows))
                        for (host, pid, tid, name), rows in sorted(lanes.items())],
        resources=resources,
        statistics_policy={"percentiles": "nearest rank", "intervals": "half-open [start,end)",
                           "inclusive_cpu": "Thread CPU sums can double-count nested spans; not total CPU work.",
                           "wall_alignment": "Wall-clock starts plus monotonic durations; wall-clock adjustments are not corrected.",
                           "integrity": "Unknown unless producer explicitly accounts for dropped/unwritten records; missing terminal spans cannot be reconstructed."},
        warnings=list(dict.fromkeys(warnings)), limitations=_LIMITATIONS,
    )


def _trace(records):
    """Numeric synthetic lanes prevent equal OS IDs on different hosts colliding."""
    processes, threads, events = {}, {}, []
    origin = min((row["ts_ns"] for row in records), default=0)
    for row in sorted(records, key=lambda item: item["ts_ns"]):
        host, pid, tid = _identity(row)
        process_key, thread_key = (host, pid), (host, pid, tid)
        if process_key not in processes:
            processes[process_key] = len(processes) + 1
            events.append(dict(ph="M", name="process_name", pid=processes[process_key], tid=0,
                               args={"name": f"{host} pid={pid}"}))
        if thread_key not in threads:
            threads[thread_key] = len(threads) + 1
            events.append(dict(ph="M", name="thread_name", pid=processes[process_key], tid=threads[thread_key],
                               args={"name": f"host thread {tid}"}))
        args = dict(_dict(row.get("attrs")), hostname=host, original_pid=pid, original_tid=tid,
                    wall_ts_ns=row["ts_ns"])
        if "error" in row:
            args["error"] = row["error"]
        event = dict(name=row["name"], cat="host." + row["kind"], ts=(row["ts_ns"] - origin) / 1000,
                     pid=processes[process_key], tid=threads[thread_key], args=args)
        if row["kind"] == "span" and _ns(row.get("duration_ns")):
            event.update(ph="X", dur=row["duration_ns"] / 1000)
        else:
            event.update(ph="i", s="t")
        events.append(event)
        if row["kind"] == "resource":
            numeric = {}

            def flatten(value, prefix=""):
                if isinstance(value, dict):
                    for key, item in value.items():
                        flatten(item, f"{prefix}.{key}" if prefix else str(key))
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        flatten(item, f"{prefix}.{index}")
                elif _number(value):
                    numeric[prefix] = value
            flatten(_dict(row.get("attrs")))
            if numeric:
                events.append(dict(event, ph="C", name="raw." + row["name"], args=numeric))
    return dict(traceEvents=events, displayTimeUnit="ms", wall_time_origin_ns=origin,
                metadata={"scope": "Host spans and sampled counters; no GPU kernel-duration/concurrency claim.",
                          "cross_host_clock_alignment": "unknown"})


def _escaped(value):
    if value is None:
        return "unknown"
    return html.escape(str(value), quote=True)


def _table(headers, rows):
    return ("<table><thead><tr>" + "".join("<th>" + _escaped(h) + "</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + _escaped(v) + "</td>" for v in row)
                                                    + "</tr>" for row in rows) + "</tbody></table>")


def _ms(value):
    return f"{value / 1_000_000:.6g}" if _number(value) else None


def _duration_bars(modules):
    maximum = max((data["total_inclusive_wall_ns"] or 0 for _, data in modules), default=0)
    parts = [f'<div class="chart"><svg viewBox="0 0 1160 {max(36, len(modules) * 28)}" role="img" aria-label="Module inclusive duration in milliseconds">']
    for index, (name, data) in enumerate(modules):
        value = data["total_inclusive_wall_ns"]
        width = 680 * value / maximum if maximum and value is not None else 0
        y = index * 28
        parts.append(f'<g><title>{_escaped(name)}: {_escaped(_ms(value))} ms inclusive</title>'
                     f'<text x="0" y="{y + 17}">{_escaped(name[:45])}</text>'
                     f'<rect x="350" y="{y + 3}" width="{width:.6f}" height="18" fill="#4078b0"/>'
                     f'<text x="1040" y="{y + 17}">{_escaped(_ms(value))} ms</text></g>')
    return "".join(parts) + "</svg></div>"


def _thread_timeline(records):
    hosts = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["kind"] == "span" and _ns(row.get("duration_ns")):
            hosts[_host(row)][(*_identity(row)[1:], row["name"])].append(row)
    parts = ["<h2>Per-thread host span timeline</h2><p>Each host has its own wall-aligned axis, in ms from its first observed span. Bars show actual span positions/durations, not GPU kernel time. Overlapping spans can overlap visually.</p>"]
    for host, lanes in sorted(hosts.items()):
        rows = [row for lane in lanes.values() for row in lane]
        start = min(row["ts_ns"] for row in rows)
        end = max(row["ts_ns"] + row["duration_ns"] for row in rows)
        extent = end - start
        names = {name: index for index, name in enumerate(sorted({row["name"] for row in rows}))}
        colors = ("#4078b0", "#bc662b", "#398358", "#8b5cad", "#b04463")
        parts.append("<h3>" + _escaped(host) + "</h3>"
                     f'<div class="chart"><svg viewBox="0 0 1160 {len(lanes) * 28 + 30}" role="img" aria-label="Host thread span timeline in milliseconds">'
                     f'<text x="350" y="16">0 ms</text><text x="1000" y="16">{_escaped(_ms(extent))} ms</text>')
        for index, ((pid, tid, name), spans) in enumerate(sorted(lanes.items())):
            label = f"pid={pid} tid={tid} {name}"
            y = index * 28 + 28
            parts.append(f'<g><title>{_escaped(label)}</title><text x="0" y="{y + 17}">{_escaped(label[:45])}</text>')
            for row in sorted(spans, key=lambda item: item["ts_ns"]):
                x = 350 + (680 * (row["ts_ns"] - start) / extent if extent else 0)
                width = 680 * row["duration_ns"] / extent if extent else 0
                color = colors[names[name] % len(colors)]
                parts.append(f'<rect x="{x:.6f}" y="{y + 3}" width="{width:.6f}" height="18" fill="{color}">'
                             f'<title>{_escaped(label)}; start {_escaped(_ms(row["ts_ns"] - start))} ms; duration {_escaped(_ms(row["duration_ns"]))} ms</title></rect>')
            parts.append("</g>")
        parts.append("</svg></div>")
    return "".join(parts)


def _html(summary, title="Astrolabe profiling report", records=None):
    sections = ["<h1>" + _escaped(title) + "</h1><p>Diagnostic observations only. Duration tables and charts use milliseconds (ms); raw complete data retains nanoseconds. Unknown is not zero.</p>"]
    if "full_run_window" in summary:
        keys = ("count", "errors", "total_inclusive_wall_ns", "union_active_wall_ns", "mean_ns", "p50_ns", "p95_ns", "max_ns", "thread_cpu_ns", "max_simultaneous_host_spans")
        headers = ("module", "count", "errors", "inclusive total (ms)", "union active (ms)", "mean (ms)", "p50 (ms)", "p95 (ms)", "max (ms)", "thread CPU (ms)", "max simultaneous host spans")
        modules = sorted(summary["modules"].items(), key=lambda item: (-(item[1]["total_inclusive_wall_ns"] or 0), item[0]))
        full = summary["full_run_window"]
        sections += ["<h2>Full run elapsed</h2>",
                     _table(("elapsed (ms)", "duration source", "wall alignment extent (ms)", "wall minus monotonic (ms)"),
                            [(_ms(full["duration_ns"]), full["duration_source"], _ms(full["wall_duration_ns"]), _ms(full["wall_minus_monotonic_ns"]))]),
                     "<h2>Host modules</h2>",
                     _table(headers, [(name, *(_ms(data[k]) if k.endswith("_ns") else data[k] for k in keys)) for name, data in modules]),
                     "<h2>Module inclusive duration</h2><p>Sorted by inclusive total; bars share one duration scale. Nested spans can double-count time, so these totals are not elapsed run time.</p>",
                     _duration_bars(modules)]
        if records is not None:
            sections.append(_thread_timeline(records))
        sections.append("<h2>Observed host windows</h2>" + _table(("host", "window extent (ms)", "union active (ms)"),
                        [(host, _ms(data["window"]["duration_ns"]), _ms(data["union_active_wall_ns"]))
                         for host, data in summary["observed_span_windows_by_hostname"].items()]))
    else:
        sections.append("<h2>Same-module diagnostic comparison</h2>")
        ordered = sorted(summary["modules"].items(), key=lambda item: (-(item[1]["total_inclusive_wall_ns"]["candidate"] or item[1]["total_inclusive_wall_ns"]["baseline"] or 0), item[0]))
        sections.append(_table(("module", "metric / unit", "baseline", "candidate", "candidate-baseline", "candidate/baseline (unitless)"),
                               [(name, metric[:-3] + " (ms)" if metric.endswith("_ns") else metric,
                                 *(_ms(data[k]) if metric.endswith("_ns") else data[k] for k in ("baseline", "candidate", "delta")), data["ratio"])
                                for name, metrics in ordered for metric, data in metrics.items()]))
    sections.append("<h2>Warnings and limitations</h2><ul>" + "".join("<li>" + _escaped(item) + "</li>"
                                                                   for item in summary.get("warnings", []) + summary.get("limitations", [])) + "</ul>")
    # Never embed raw JSON in a script element: names and metadata are untrusted.
    sections.append("<h2>Complete data (null = unknown)</h2><pre>" + _escaped(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)) + "</pre>")
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
            "<title>" + _escaped(title) + "</title><style>body{font:14px system-ui;margin:2em;color:#17212b}"
            "table{border-collapse:collapse;display:block;overflow:auto}th,td{border:1px solid #bcc6d0;padding:.4em;text-align:left}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1em}.chart{overflow-x:auto}svg{min-width:1000px;width:100%;font:12px system-ui}</style><body>"
            + "".join(sections) + "</body></html>")


def _write(output_dir, documents, protected):
    output = Path(output_dir).expanduser().resolve()
    # Protect sources even when a requested destination is a symlink to one.
    protected = {path.resolve() for path in protected}
    for name in documents:
        if (output / name).resolve() in protected:
            raise ValueError(f"Refusing to overwrite profiling source: {output / name}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, content in documents.items():
        fd, temporary = tempfile.mkstemp(prefix="." + name + ".", dir=output)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            os.replace(temporary, output / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        paths[name] = str(output / name)
    return paths


def report_session(directory, output_dir=None) -> dict:
    """Return ``{'paths': {filename: absolute_path}, 'summary': {...}}``.

    Writes summary.json, report.html and trace.json; defaults to the session
    directory. Raw session.json and events-*.jsonl are never written.
    """
    root, session, records, warnings, protected, processes = _load(directory)
    summary = _summary(root, session, records, warnings, processes)
    paths = _write(root if output_dir is None else output_dir,
                   {"summary.json": summary, "report.html": _html(summary, records=records), "trace.json": _trace(records)}, protected)
    return {"paths": paths, "summary": summary}


def _difference(baseline, candidate):
    known = _number(baseline) and _number(candidate)
    return dict(baseline=baseline, candidate=candidate,
                delta=candidate - baseline if known else None,
                ratio=candidate / baseline if known and baseline != 0 else None)


def _metadata_diff(baseline, candidate, prefix=""):
    differences = []
    for key in sorted(set(baseline) | set(candidate)):
        path = f"{prefix}.{key}" if prefix else key
        left, right = baseline.get(key), candidate.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            differences.extend(_metadata_diff(left, right, path))
        elif key not in baseline or key not in candidate or left != right:
            differences.append(dict(field=path, baseline=left, candidate=right,
                                    baseline_present=key in baseline, candidate_present=key in candidate))
    return differences


def _provenance(summary):
    session = summary["session"]
    metadata = _dict(session.get("metadata"))
    return dict(source=session.get("source", metadata.get("source")),
                hostname=session.get("hostname"),
                observed_hostnames=sorted({row["hostname"] for row in summary["thread_modules"]}),
                hardware=session.get("hardware", metadata.get("hardware")),
                observed_gpu_uuids=[dict(hostname=row["hostname"], uuid=row["uuid"]) for row in summary["resources"]["gpu"] or []],
                python=session.get("python"), cpu_affinity=session.get("cpu_affinity"))


def compare_sessions(baseline, candidate, output_dir=None) -> dict:
    """Diagnostic comparison only; without output_dir this writes no files.

    Explicit output writes comparison.json and comparison.html, not session.json.
    Ratios are candidate/baseline, not certified speedups or equivalence proofs.
    """
    left_root, left_session, left_records, left_warnings, left_protected, left_processes = _load(baseline)
    right_root, right_session, right_records, right_warnings, right_protected, right_processes = _load(candidate)
    left = _summary(left_root, left_session, left_records, left_warnings, left_processes)
    right = _summary(right_root, right_session, right_records, right_warnings, right_processes)
    metrics = ("count", "errors", "total_inclusive_wall_ns", "union_active_wall_ns", "mean_ns", "p50_ns", "p95_ns", "max_ns", "thread_cpu_ns", "max_simultaneous_host_spans")
    modules = {}
    for name in sorted(set(left["modules"]) | set(right["modules"])):
        modules[name] = {metric: _difference(left["modules"].get(name, {}).get(metric), right["modules"].get(name, {}).get(metric)) for metric in metrics}
    summary = dict(
        report_schema_version=1, classification="diagnostic_comparison", modules=modules,
        full_run_duration_ns=_difference(left["full_run_window"]["duration_ns"], right["full_run_window"]["duration_ns"]),
        baseline=left, candidate=right,
        provenance={"baseline": _provenance(left), "candidate": _provenance(right)},
        configuration_differences=_metadata_diff(_dict(left_session.get("config")), _dict(right_session.get("config"))),
        metadata_differences=_metadata_diff(_dict(left_session.get("metadata")), _dict(right_session.get("metadata"))),
        warnings=[*("baseline: " + item for item in left["warnings"]), *("candidate: " + item for item in right["warnings"])],
        limitations=[*_LIMITATIONS,
                     "Inputs, seeds, model/weights, resources and scientific outputs have not been independently matched or verified.",
                     "Metadata/config equality does not establish scientific equivalence; differences and ratios are not certified speedups.",
                     "Instrumentation coverage, run completeness, sampling coverage and profiling perturbation may differ."],
    )
    paths = {}
    if output_dir is not None:
        paths = _write(output_dir, {"comparison.json": summary, "comparison.html": _html(summary, "Astrolabe diagnostic comparison")},
                       left_protected + right_protected)
    return {"paths": paths, "summary": summary}
