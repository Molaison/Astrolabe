"""Opt-in admission controller, used by AF3 for complete evaluator RPCs.

Broker-backed request-local owners are not yet qualified and are rejected,
never silently serialized or certified. Memory defaults are trial bounds, not
measured optima.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import os
import threading
import time


def require_private_evaluator_owner(request_local_rng, *, broker=None, environment=None):
    """Reject the unqualified broker combination, without changing its policy.

    A broker can keep inference at one and fence a failed GPU lease while other
    requests wait indefinitely. Neither concurrent admission nor failure/drain
    is qualified for this path. Do not remove the socket or invent a profile.
    """
    environment = os.environ if environment is None else environment
    if request_local_rng and (broker is not None or environment.get("ASTROLABE_RESOURCE_SOCKET")):
        raise RuntimeError(
            "Adaptive/request-local evaluator with ASTROLABE_RESOURCE_SOCKET/resource broker "
            "is not qualified: concurrent admission and failure/drain are unsupported; "
            "refusing owner startup before inference")


@dataclass(frozen=True)
class AdaptiveEvaluationSettings:
    enabled: bool = False
    max_concurrency: int = 1
    target_utilization: float = 90.0
    observation_window_s: float = 10.0
    sample_interval_s: float = 1.0
    memory_per_request_bytes: int = 6 * 1024**3
    memory_reserve_bytes: int = 1024**3
    throughput_drop_fraction: float = .15


def adaptive_settings(raw=None):
    if isinstance(raw, AdaptiveEvaluationSettings):
        raw = asdict(raw)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("adaptive settings must be a mapping")
    unknown = set(raw) - set(AdaptiveEvaluationSettings.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown adaptive settings: {', '.join(sorted(map(str, unknown)))}")
    settings = AdaptiveEvaluationSettings(**raw)
    if type(settings.enabled) is not bool:
        raise ValueError("adaptive.enabled must be boolean")
    for key in ("max_concurrency", "memory_per_request_bytes", "memory_reserve_bytes"):
        value = getattr(settings, key)
        if type(value) is not int or value < (0 if key == "memory_reserve_bytes" else 1):
            raise ValueError(f"adaptive.{key} must be a valid nonnegative/positive integer")
    for key in ("target_utilization", "observation_window_s", "sample_interval_s",
                "throughput_drop_fraction"):
        value = getattr(settings, key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"adaptive.{key} must be finite and positive")
    if settings.target_utilization > 100 or settings.throughput_drop_fraction >= 1:
        raise ValueError("adaptive utilization/drop fraction is outside its range")
    if settings.sample_interval_s > settings.observation_window_s:
        raise ValueError("adaptive sample interval must not exceed observation window")
    return settings


class EvaluatorInferController:
    """One monitor and admission gate; its owner defines the admitted operation.

    The AF3 process pool admits complete evaluator RPCs, not pure model infer.
    Only native child gpu_start/gpu_end events measure the latter.
    """

    def __init__(self, settings, *, device="cuda:0", sampler=None, emit=None,
                 clock=time.monotonic):
        from astrolabe.runtime.telemetry import NvidiaSMISampler

        self.settings = adaptive_settings(settings)
        self.device = device
        self._sampler = sampler if sampler is not None else NvidiaSMISampler(
            timeout=min(2.0, self.settings.sample_interval_s / 2))
        self._emit = emit if emit is not None else lambda event, **fields: None
        self._clock = clock
        self._cv = threading.Condition(threading.RLock())
        self._stop = threading.Event()
        self._thread = None
        self.enabled = self.settings.enabled
        self.limit = 1
        self.ready = self.active = self.completed = 0
        self.warming = 0
        self.quarantined = False
        self._window_at = clock()
        self._window_completed = 0
        self._samples = []
        self._previous_rate = None
        self._previous_limit = 1
        self._last_decision = None
        self._window_loaded = True

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._monitor, name="af3-infer-adaptive", daemon=True)
            self._thread.start()

    def close(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join()  # Sampler subprocess queries have bounded timeouts.

    def snapshot(self):
        with self._cv:
            return dict(enabled=self.enabled, limit=self.limit, ready=self.ready,
                        active=self.active, completed=self.completed, warming=self.warming,
                        quarantined=self.quarantined, settings=asdict(self.settings),
                        last_decision=None if self._last_decision is None else dict(self._last_decision))

    def control(self, *, enabled, reset=False):
        if type(enabled) is not bool or type(reset) is not bool:
            raise ValueError("adaptive_control enabled/reset must be boolean")
        with self._cv:
            if self._stop.is_set():
                raise RuntimeError("stopped owner cannot be reset; rebuild its owner")
            if self.active or self.ready or self.warming:
                raise RuntimeError("adaptive_control requires a completely drained owner")
            if self.quarantined:
                raise RuntimeError("quarantined owner cannot be reset; rebuild its owner")
            self.enabled = enabled
            if reset or not enabled:
                self.limit = 1
            if reset:
                self.completed = 0
            self._window_at = self._clock()
            self._window_completed = self.completed
            self._samples.clear()
            self._previous_rate = None
            self._previous_limit = self.limit
            self._window_loaded = True
            self._emit("adaptive_control", enabled=enabled, reset=reset, **self._counts())
            return self.snapshot()

    def quarantine(self):
        with self._cv:
            self.quarantined = True
            self.limit = 1
            self._cv.notify_all()

    @contextmanager
    def inference_slot(self):
        with self._cv:
            waiters = getattr(self, "_inference_waiters", None)
            if waiters is None:
                self._inference_waiters = waiters = []
            if not self.ready and not self.active:
                self._reset_idle_window()
            ticket = object()
            waiters.append(ticket)
            self.ready += 1
            self._cv.notify_all()
            try:
                # A notification does not reserve a slot: the oldest prepared
                # request keeps priority even if a later thread wakes first.
                while ((waiters[0] is not ticket or self.active >= self.limit)
                       and not self.quarantined and not self._stop.is_set()):
                    self._cv.wait()
                if self.quarantined or self._stop.is_set():
                    raise RuntimeError("Evaluator owner is quarantined or stopping")
                self.active += 1
            finally:
                waiters.remove(ticket)
                self.ready -= 1
                if not self.ready and not self.active:
                    self._reset_idle_window()
                # Admit the next head into any remaining capacity, or release
                # it after this waiter was cancelled by close/quarantine.
                self._cv.notify_all()
        ok = False
        try:
            yield
            ok = True
        finally:
            with self._cv:
                self.active -= 1
                if ok:
                    self.completed += 1
                    # Exclude initial compilation from the first throughput window.
                    if self.completed == 1:
                        self._window_at = self._clock()
                        self._window_completed = 1
                        self._samples.clear()
                if not self.ready and not self.active:
                    self._reset_idle_window()
                self._cv.notify_all()

    def _reset_idle_window(self):
        self._window_at = self._clock()
        self._window_completed = self.completed
        self._samples.clear()
        self._window_loaded = True
        self._previous_rate = None
        self._previous_limit = self.limit

    def _counts(self):
        return dict(ready=self.ready, active=self.active, completed=self.completed, limit=self.limit)

    def _device_row(self, raw):
        from astrolabe.runtime.telemetry import validate_sample

        rows = list(validate_sample(raw).values())
        ordinal = int(self.device.split(":", 1)[1])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        selected = visible.split(",")[ordinal].strip() if visible is not None else str(ordinal)
        matches = [row for row in rows if (str(row["index"]) == selected
                   or selected.startswith("GPU-") and row["uuid"].startswith(selected))]
        if len(matches) != 1:
            raise ValueError("Adaptive telemetry cannot identify the owner's physical GPU")
        return matches[0]

    def observe(self, raw, *, sampled_at):
        """Consume one bounded sample; also used by deterministic CPU checks."""
        now = self._clock()
        try:
            row = self._device_row(raw)
            if not 0 <= now - sampled_at <= 2 * self.settings.sample_interval_s:
                row = None
        except (ValueError, TypeError, KeyError, IndexError):
            row = None
        with self._cv:
            self._window_loaded = self._window_loaded and bool(self.ready) and not self.warming
            if row is None:
                self._samples.clear()
            else:
                self._samples.append((sampled_at, row["utilization"], row["free_bytes"]))
                self._emit("adaptive_sample", gpu_uuid=row["uuid"],
                           utilization=row["utilization"], free_bytes=row["free_bytes"],
                           process_ids=sorted(row["processes"]), sampled_at=sampled_at)
            elapsed = now - self._window_at
            if elapsed < self.settings.observation_window_s:
                return
            samples = [s for s in self._samples if s[0] >= self._window_at]
            fresh = bool(row is not None and samples)
            utilization = sum(s[1] for s in samples) / len(samples) if fresh else None
            free = row["free_bytes"] if fresh else None
            rate = (self.completed - self._window_completed) / elapsed
            before = self.limit
            reason = "hold"
            if self.quarantined:
                reason = "quarantined"
            elif not self.enabled:
                reason = "fixed1"
            elif not fresh:
                reason = "telemetry_unavailable"
            elif free < self.settings.memory_per_request_bytes + self.settings.memory_reserve_bytes:
                self.limit = max(1, self.limit - 1)
                reason = "memory_pressure"
            elif not self.ready:
                reason = "starved"
            elif not self.completed or self.warming:
                reason = "warming_up"
            elif utilization < self.settings.target_utilization and not self._window_loaded:
                reason = "supply_limited_window"
            elif utilization < self.settings.target_utilization and self.limit < self.settings.max_concurrency:
                self.limit += 1
                reason = "increase"
            elif (utilization >= self.settings.target_utilization and self._window_loaded
                  and self.completed - self._window_completed >= 2 * before
                  and self._previous_rate is not None and self._previous_limit == before
                  and rate < self._previous_rate * (1 - self.settings.throughput_drop_fraction)):
                self.limit = max(1, self.limit - 1)
                reason = "throughput_drop"
            decision = dict(reason=reason, enabled=self.enabled, previous_limit=before,
                            utilization=utilization, free_bytes=free, telemetry_fresh=fresh,
                            throughput_per_s=rate, window_s=elapsed, starved=not bool(self.ready),
                            window_loaded=self._window_loaded,
                            window_completed=self.completed - self._window_completed,
                            **self._counts())
            self._last_decision = decision
            self._emit("adaptive_window_decision", **decision)
            # A rate belongs to the capacity that produced it, not a new limit.
            # Require two completed waves and continuous observed demand so a
            # single completion or a startup/tail window cannot drive backoff.
            self._previous_rate = rate if (
                fresh and self._window_loaded and not self.warming
                and self.completed - self._window_completed >= 2 * before
                and self.limit == before) else None
            self._previous_limit = before
            self._window_loaded = True
            self._window_at = now
            self._window_completed = self.completed
            self._samples.clear()
            self._cv.notify_all()

    def _monitor(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._stop.is_set() or self.active or self.ready)
                if self._stop.is_set():
                    return
            sampled_at = self._clock()
            try:
                raw = self._sampler()
            except Exception:
                raw = None
            self.observe(raw, sampled_at=sampled_at)
            self._stop.wait(self.settings.sample_interval_s)
