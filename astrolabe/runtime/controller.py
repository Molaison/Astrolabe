"""Windowed control of independent in-flight trajectories.

The objective is completed, valid work per elapsed time. Device activity only
explains headroom/starvation. Same-card kernels remain serial without a backend
certification; this controller does not invent candidates or change samples.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ControlPolicy:
    window_seconds: float = 5.0
    min_completions: int = 4
    cooldown_seconds: float = 10.0
    gain_threshold: float = .05
    regression_threshold: float = .10
    memory_high: float = .90
    gpu_busy_target: float = 90.0
    poll_seconds: float = 1.0

    @classmethod
    def parse(cls, value=None):
        value = dict(value or {})
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'unknown execution control options: {sorted(unknown)}')
        result = cls(**value)
        if type(result.min_completions) is not int or result.min_completions < 2:
            raise ValueError('min_completions must be an integer >= 2')
        for name in ('window_seconds', 'cooldown_seconds', 'poll_seconds'):
            v = getattr(result, name)
            if not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('gain_threshold', 'regression_threshold', 'memory_high'):
            v = getattr(result, name)
            if not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 < v < 1:
                raise ValueError(f'{name} must be in (0, 1)')
        if not 0 < result.gpu_busy_target <= 100:
            raise ValueError('gpu_busy_target must be in (0, 100]')
        return result


class AdaptiveController:
    def __init__(self, maximum, *, policy=None, started=0.0, events=None):
        if type(maximum) is not int or maximum < 1:
            raise ValueError('maximum in-flight count must be positive')
        self.maximum = maximum
        self.limit = 1
        self.policy = ControlPolicy.parse(policy)
        self.window_start = started
        self.last_event_time = started
        self.cooldown_until = started
        self.events = events
        self.window = []
        self.baselines = {}
        self.experiment = None
        self.previous_failures = 0
        self.last_reason = 'conservative_start'
        self.previous_cpu = None
        self.effective_window_start = started
        self.effective_previous = 0
        self.effective_completed = 0
        self.effective_throughput = None

    def _decision(self, limit, reason, now, **evidence):
        before = self.limit
        self.limit = max(1, min(self.maximum, limit))
        self.last_reason = reason
        if self.events:
            self.events.write('controller', previous_limit=before, limit=self.limit,
                              reason=reason, objective='completed_scientific_trajectories',
                              control_proxy='warm_backend_shape_completions',
                              effective_completed=self.effective_completed,
                              effective_throughput=self.effective_throughput, evidence=evidence)
        return self.limit

    def _shrink(self, reason, now, **evidence):
        self.experiment = None
        self.window.clear()
        self.window_start = now
        self.last_event_time = max(self.last_event_time, now)
        self.cooldown_until = now + self.policy.cooldown_seconds
        return self._decision(max(1, self.limit // 2), reason, now, **evidence)

    def observe(self, metrics, *, now, independent_work=True, active=1):
        if not isinstance(metrics, dict):
            return self._shrink('telemetry_unavailable', now)
        self.effective_completed = int(metrics.get('effective_completed', self.effective_completed))
        if now - self.effective_window_start >= self.policy.window_seconds:
            self.effective_throughput = ((self.effective_completed - self.effective_previous) /
                                         (now - self.effective_window_start))
            self.effective_window_start = now
            self.effective_previous = self.effective_completed
        observation = metrics.get('observation') or {}
        ledger = metrics.get('ledger') or {}
        limits = ledger.get('limits') or {}
        failures = int(metrics.get('failures', 0))
        if metrics.get('trajectory_failure'):
            self.previous_failures = failures
            return self._shrink('execution_failure', now, failure_source='trajectory',
                                failures=failures, oom=metrics.get('oom', 0))
        if failures > self.previous_failures:
            self.previous_failures = failures
            return self._shrink('execution_failure', now, failures=failures, oom=metrics.get('oom', 0))
        devices = list((observation.get('devices') or {}).values())
        vram_pressure = max((d['used_bytes'] / d['total_bytes'] for d in devices
                             if d.get('available') and d.get('total_bytes') and d.get('used_bytes') is not None), default=0)
        rss = observation.get('rss_bytes', 0)
        ram_pressure = rss / limits['ram_bytes'] if limits.get('ram_bytes') else 0
        cgroup_pressure = (observation.get('memory') or {}).get('pressure') or 0
        if max(vram_pressure, ram_pressure, cgroup_pressure) >= self.policy.memory_high:
            return self._shrink('memory_pressure', now, vram=vram_pressure, ram=ram_pressure,
                                cgroup=cgroup_pressure)
        if metrics.get('stopping'):
            return self._shrink('resource_or_owner_stop', now)
        # GPU activity is averaged over driver sampling windows. Never replace
        # an unavailable counter by a zero-busy observation.
        activities = [d['gpu_activity'] for d in devices
                      if d.get('available') and d.get('gpu_activity') is not None]
        busy = sum(activities) / len(activities) if activities else None
        gpu_expected = bool(ledger.get('devices'))
        if gpu_expected and (not devices or len(activities) != len(devices)):
            return self._decision(self.limit, 'gpu_telemetry_unavailable', now)
        cpu_busy = None
        cpu = (observation.get('time'), observation.get('cpu_ticks'))
        if self.previous_cpu and None not in cpu and cpu[0] > self.previous_cpu[0]:
            cpu_busy = (max(0, cpu[1] - self.previous_cpu[1]) /
                        (cpu[0] - self.previous_cpu[0]) / observation.get('clock_ticks', 100) /
                        max(1, limits.get('cpu_slots', 1)))
        if None not in cpu:
            self.previous_cpu = cpu
        fresh = [event for event in metrics.get('recent', []) if event['time'] > self.last_event_time]
        if fresh:
            self.last_event_time = max(event['time'] for event in fresh)
        control_phases = ('device', 'composite', 'managed', 'trajectory')
        cold_finished = [event['time'] for event in fresh if event.get('cold_start') and
                         event.get('outcome') == 'completed' and event.get('phase') in control_phases]
        if cold_finished:
            self.window_start = max(self.window_start, max(cold_finished))
            self.window = [event for event in self.window if event['time'] > self.window_start]
        self.window.extend(event for event in fresh if event['time'] > self.window_start and
                           event.get('outcome') == 'completed' and not event.get('cold_start') and
                           event.get('phase') in control_phases)
        if not independent_work:
            reason = 'causal_dependency_or_no_independent_supply' if active else 'workload_complete'
            return self._decision(self.limit, reason, now, gpu_activity=busy)
        if now < self.cooldown_until:
            return self._decision(self.limit, 'cooldown', now)
        if cold_finished:
            return self._decision(self.limit, 'cold_start_observed', now,
                                  steady_window_start=self.window_start, completions=len(self.window))
        elapsed = now - self.window_start
        if elapsed < self.policy.window_seconds or len(self.window) < self.policy.min_completions:
            return self._decision(self.limit, 'observing_steady_work', now, completions=len(self.window), elapsed=elapsed)
        groups = {}
        for event in self.window:
            key = (event['backend'], event.get('shape_key', 'unknown'))
            groups[key] = groups.get(key, 0) + 1
        # A real pipeline naturally alternates mutator and evaluator work.
        # Keep a rate for EACH backend/shape; a faster cheap stage must not
        # conceal regression of the slow model stage.
        if not groups or any(key[1] == 'unknown' for key in groups):
            self.window.clear()
            self.window_start = now
            return self._decision(self.limit, 'unknown_workload_shape', now)
        key = tuple(sorted(groups))
        rates = {cohort: count / elapsed for cohort, count in groups.items()}
        throughput = sum(groups.values()) / elapsed
        self.window.clear()
        self.window_start = now
        if self.experiment is not None:
            experiment = self.experiment
            self.experiment = None
            if experiment['key'] != key:
                self.cooldown_until = now + self.policy.cooldown_seconds
                return self._decision(experiment['limit'], 'workload_changed_during_trial', now)
            gain = min(rates[cohort] / experiment['rates'][cohort] - 1 for cohort in key)
            if gain < self.policy.gain_threshold:
                self.cooldown_until = now + self.policy.cooldown_seconds
                reason = 'throughput_regression' if gain <= -self.policy.regression_threshold else 'insufficient_throughput_gain'
                return self._decision(experiment['limit'], reason, now, gain=gain,
                                      throughput=throughput, baseline=experiment['throughput'])
            self.baselines[key] = (self.limit, throughput, rates)
            self.cooldown_until = now + self.policy.cooldown_seconds
            return self._decision(self.limit, 'throughput_gain_retained', now, gain=gain, throughput=throughput)
        previous = self.baselines.get(key)
        if previous and previous[0] == self.limit and any(
                rates[cohort] < previous[2][cohort] * (1 - self.policy.regression_threshold) for cohort in key):
            return self._shrink('steady_throughput_regression', now, throughput=throughput, baseline=previous[1])
        self.baselines[key] = (self.limit, throughput, rates)
        waiting = metrics.get('waiting', 0)
        if cpu_busy is not None and cpu_busy >= .9:
            return self._decision(self.limit, 'cpu_supply_saturated', now, cpu_busy=cpu_busy)
        if waiting > 0 or (busy is not None and busy >= self.policy.gpu_busy_target):
            return self._decision(self.limit, 'device_supplied', now, ready=waiting, gpu_activity=busy)
        if self.limit >= self.maximum:
            return self._decision(self.limit, 'allocation_limit', now)
        self.experiment = {'key': key, 'limit': self.limit, 'throughput': throughput, 'rates': rates}
        return self._decision(self.limit + 1, 'independent_supply_trial', now,
                              throughput=throughput, gpu_activity=busy, ready=waiting)
