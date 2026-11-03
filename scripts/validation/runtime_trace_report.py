#!/usr/bin/env python3
"""Summarize execution evidence without turning GPU activity into useful work.

The workflow's terminal scientific records remain the throughput denominator.
These stage observations explain cold starts, queueing and resource pressure;
they are neither independent timing nor backend accuracy certification.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


def _add_stat(target, duration):
    if not isinstance(duration, (float, int)) or duration < 0:
        return
    target["count"] = target.get("count", 0) + 1
    target["total_seconds"] = target.get("total_seconds", 0) + duration
    target["min_seconds"] = min(target.get("min_seconds", duration), duration)
    target["max_seconds"] = max(target.get("max_seconds", duration), duration)
    target["mean_seconds"] = target["total_seconds"] / target["count"]


def summarize(path, *, maximum_groups=4096):
    path = Path(path)
    if not path.is_file():
        return {"available": False, "reason": "runtime trace not produced", "path": str(path)}
    counters, controllers, cohorts, phases = Counter(), Counter(), {}, {}
    workers, active_workers, last_resource, idle = {}, {}, None, Counter()
    current_reason = "no_controller_observation"
    rss_peak = artifact_peak = vram_peak = None
    busy_sum, busy_duration = 0.0, 0.0
    malformed = capped = 0
    stopped = None

    def bounded_group(mapping, key):
        nonlocal capped
        if key not in mapping and len(mapping) >= maximum_groups:
            capped += 1
            return None
        return mapping.setdefault(key, {})

    with path.open() as stream:
        for line in stream:
            if len(line) > 2 * 1024**2:
                malformed += 1
                continue
            try:
                event = json.loads(line)
                kind = event["event"]
            except (ValueError, KeyError, TypeError):
                malformed += 1
                continue
            counters[kind] += 1
            backend = event.get("backend", "unattributed")
            if kind == "request_complete":
                key = (str(event.get("cohort", backend)), event.get("shape_key", "unknown"),
                       event.get("phase", "unknown"), "cold" if event.get("cold_start") else "steady")
                group = bounded_group(cohorts, key)
                if group is not None:
                    _add_stat(group, event.get("duration"))
                    group["failures"] = group.get("failures", 0) + (event.get("outcome") != "completed")
            if kind in ("compute_end", "device_end", "stage"):
                phase = "device" if kind == "device_end" else event.get("phase", "unknown")
                # A composite native call includes host work. Never label its
                # wall time as kernel/device execution.
                key = (str(backend), phase, "nested" if event.get("nested") else kind)
                group = bounded_group(phases, key)
                if group is not None:
                    _add_stat(group, event.get("duration"))
            lifecycle = event.get("lifecycle")
            if kind == "model_loaded" or isinstance(lifecycle, dict):
                lifecycle = lifecycle or {}
                key = (str(backend), lifecycle.get("worker_epoch", event.get("pid")))
                worker = bounded_group(workers, key)
                if worker is not None:
                    worker["backend"] = str(backend)
                    worker["worker_epoch"] = lifecycle.get("worker_epoch")
                    worker["device"] = event.get("device", worker.get("device"))
                    worker["lifecycle"] = lifecycle
                    count = lifecycle.get("load_count")
                    if type(count) is int:
                        worker["maximum_observed_load_count"] = max(count, worker.get("maximum_observed_load_count") or 0)
                    else:
                        worker.setdefault("maximum_observed_load_count", None)
                    if kind == "model_loaded":
                        worker["worker_start_observed"] = True
                        worker["identity"] = event.get("identity")
                        active_workers[str(backend)] = key
                    if kind == "model_unloaded":
                        worker["release_observed"] = True
            if kind in ("model_unloaded", "resident_released"):
                name = str(event.get("resident", backend))
                key = active_workers.get(name)
                if key in workers:
                    if kind == "model_unloaded" or event.get("process_exit") is True:
                        workers[key]["release_observed"] = True
                        workers[key]["release_evidence"] = kind
                if kind == "model_unloaded":
                    active_workers.pop(name, None)
            if kind == "controller":
                current_reason = event.get("reason", "unknown")
                controllers[current_reason] += 1
            if kind == "resource_sample":
                stamp = event.get("time")
                for field, value in (("rss", event.get("rss_bytes")), ("artifact", event.get("artifact_bytes"))):
                    if isinstance(value, (int, float)):
                        if field == "rss":
                            rss_peak = max(rss_peak or 0, value)
                        else:
                            artifact_peak = max(artifact_peak or 0, value)
                devices = list((event.get("devices") or {}).values())
                activities = [d["gpu_activity"] for d in devices if d.get("available") and d.get("gpu_activity") is not None]
                for device in devices:
                    value = device.get("used_bytes")
                    if device.get("available") and isinstance(value, (int, float)):
                        vram_peak = max(vram_peak or 0, value)
                busy = sum(activities) / len(activities) if activities and len(activities) == len(devices) else None
                if last_resource is not None and isinstance(stamp, (int, float)):
                    previous_stamp, previous_busy, reason = last_resource
                    elapsed = stamp - previous_stamp
                    if elapsed > 0 and previous_busy is not None:
                        busy_sum += previous_busy * elapsed
                        busy_duration += elapsed
                        if previous_busy < 10:
                            idle[reason] += elapsed
                if isinstance(stamp, (int, float)):
                    last_resource = (stamp, busy, current_reason)
            if kind == "broker_stopped":
                stopped = event.get("resources")
    worker_rows = list(workers.values())
    known_loads = [w["maximum_observed_load_count"] for w in worker_rows
                   if w.get("maximum_observed_load_count") is not None]
    return {
        "available": True, "path": str(path), "event_counts": dict(counters),
        "malformed_lines": malformed, "group_limit_drops": capped,
        "request_observations": [{"backend_cohort": b, "shape_key": s, "phase": p,
                                  "temperature": c, **v} for (b, s, p, c), v in sorted(cohorts.items())],
        "phase_observations": [{"backend": b, "phase": p, "source": source, **v}
                               for (b, p, source), v in sorted(phases.items())],
        "cold_definition": "first observed shape/phase for a worker, or adapter-supplied cold flag; CLI starts are independently cold",
        "worker_instances": worker_rows,
        "known_model_load_count": sum(known_loads) if known_loads else None,
        "workers_without_load_count_evidence": sum(w.get("maximum_observed_load_count") is None for w in worker_rows),
        "controller_decisions": dict(controllers),
        "sampled_low_activity_seconds_by_control_context": dict(idle),
        "idle_attribution_limit": "sampled activity below 10% grouped by the contemporaneous controller observation; context is diagnostic evidence, not proof of exclusive causation",
        "time_weighted_sampled_gpu_activity_percent": busy_sum / busy_duration if busy_duration else None,
        "gpu_observed_seconds": busy_duration,
        "peak_process_tree_rss_bytes": rss_peak, "peak_artifact_bytes": artifact_peak,
        "peak_device_vram_bytes": vram_peak, "terminal_ledger": stopped,
        "all_registered_residency_released": None if stopped is None else not bool(stopped.get("resident")),
        "throughput_note": "stage completions are not scientific task completions; use terminal workflow provenance and score/decision parity",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = json.dumps(summarize(args.trace), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")
