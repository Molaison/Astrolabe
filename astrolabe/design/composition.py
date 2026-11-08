"""Public composition payload for one generic design run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from .types import Candidate, RunResult, Runtime, SearchPolicy, System, Trial


@dataclass(frozen=True)
class DesignRun:
    """One caller-owned System, Trial, Policy and Runtime composition."""

    system: System
    initial: Candidate
    trial: Trial
    policy: SearchPolicy
    runtime: Runtime
    capacity: int = 1

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or self.capacity < 1:
            raise ValueError("capacity must be a positive integer")

    def run(self) -> RunResult:
        """Execute through the sole generic coordinator."""

        from . import coordinator

        return coordinator.run(
            self.system,
            self.initial,
            self.trial,
            self.policy,
            self.runtime,
            capacity=self.capacity,
        )


S = TypeVar("S")


@dataclass(frozen=True)
class ControlRun(Generic[S]):
    """Typed control composition for isolated native jobs or collections.

    Molecular work uses DesignRun's domain validation. Higher level process
    jobs retain their actual job/result types rather than inventing molecular
    Candidates. Both execute the same coordinator drive.
    """

    policy: object
    trial: object
    initial_state: S
    runtime: object | None = None
    state_codec: object | None = None
    request_codec: object | None = None
    result_codec: object | None = None

    def run(self):
        from . import coordinator
        from astrolabe.model_execution import model_run_scope
        with model_run_scope():
            if self.runtime is None:
                return coordinator.drive(self.policy, self.trial, self.initial_state)
            from .control_journal import run_recorded_control
            return run_recorded_control(self)
