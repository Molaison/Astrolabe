"""Real SQLite/process-death checks for the bounded RuntimeRecovery model."""
from dataclasses import dataclass
import os
from pathlib import Path

import pytest

from astrolabe.design.operations import INVOKE, operation
from astrolabe.design.runtime import BudgetExhausted, OutcomeUnknown, Runtime
from astrolabe.design.types import History, Requirement, key_for


@dataclass(frozen=True)
class Value:
    number: int


CUTS = (
    "reserved:before",
    "reserved:after",
    "issued:before",
    "issued:after",
    "invoke",
    "completed:before",
    "completed:after",
    "none",
)
COMPLETE_ON_RESUME = {
    "reserved:before", "reserved:after", "issued:before", "completed:after", "none",
}


def component_at(root, boundary, cut="none"):
    def invoke(request):
        with (root / "effects").open("ab", buffering=0) as stream:
            stream.write(b"invoke\n")
            os.fsync(stream.fileno())
        if cut == "invoke":
            os._exit(73)
        return Value(request.number + 1)

    return operation("work", Value, Value, invoke, version="recovery-contract-1",
                     paid_boundary=boundary)


def call(runtime, component, work="first"):
    runtime.preflight((Requirement("work", INVOKE, "required"),))
    with runtime.bind(work, History(key_for(()), ())):
        return runtime.call("round", component, Value(1))


def effects(root):
    path = root / "effects"
    return path.read_bytes().splitlines() if path.exists() else []


def crash_child(root, boundary, cut):
    class CutRuntime(Runtime):
        def _write(self, kind, name, value):
            label = value.get("status") if kind == "attempt" else None
            if cut == f"{label}:before":
                os._exit(73)
            super()._write(kind, name, value)
            if cut == f"{label}:after":
                os._exit(73)

    pid = os.fork()
    if pid == 0:
        try:
            component = component_at(root, boundary, cut)
            with CutRuntime((component,), root=root, budgets={boundary: 1}) as runtime:
                call(runtime, component)
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(74)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == (0 if cut == "none" else 73)


@pytest.mark.parametrize("boundary", ("predict", "score"))
@pytest.mark.parametrize("cut", CUTS)
def test_crash_resume_never_replays(tmp_path, boundary, cut):
    expected_source = os.environ.get("ASTROLABE_RECOVERY_SOURCE")
    if expected_source:
        import astrolabe.design.runtime as module
        assert Path(module.__file__).resolve() == Path(expected_source).resolve()
    crash_child(tmp_path, boundary, cut)
    component = component_at(tmp_path, boundary)
    for _ in range(2):
        with Runtime((component,), root=tmp_path, budgets={boundary: 1}) as runtime:
            if cut in COMPLETE_ON_RESUME:
                assert call(runtime, component) == Value(2)
            else:
                with pytest.raises(OutcomeUnknown):
                    call(runtime, component)
            assert len(runtime.attempts()) == 1
    assert len(effects(tmp_path)) == (0 if cut == "issued:after" else 1)


@pytest.mark.parametrize("boundary", ("predict", "score"))
def test_budget_survives_unknown(tmp_path, boundary):
    crash_child(tmp_path, boundary, "invoke")
    component = component_at(tmp_path, boundary)
    with Runtime((component,), root=tmp_path, budgets={boundary: 1}) as runtime:
        with pytest.raises(OutcomeUnknown):
            call(runtime, component)
        with pytest.raises(BudgetExhausted):
            call(runtime, component, work="second")
    assert effects(tmp_path) == [b"invoke"]
