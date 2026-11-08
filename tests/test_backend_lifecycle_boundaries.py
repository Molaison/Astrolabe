from __future__ import annotations

from astrolabe.evaluators.rf3 import RoseTTAFold3Evaluator
from astrolabe.evaluators.wrappers.rf3 import RF3RunnerService
from astrolabe.runtime_support import ResourceOwner
from astrolabe.structure_processors.peptide_docker import RAPiDockPeptideDockingProcessor
from astrolabe.structure_processors.wrappers.rapidock_runner import RAPiDockRunner


class FakeIdempotentRunner:
    def __init__(self):
        self.calls = 0
        self.releases = 0
        self.closed = False

    def shutdown(self):
        self.calls += 1
        if not self.closed:
            self.releases += 1
            self.closed = True


def test_real_outer_classes_delegate_to_existing_runner_and_allow_repeated_shutdown(tmp_path):
    runner = FakeIdempotentRunner()
    evaluator = RoseTTAFold3Evaluator(runner, tmp_path)
    predocker = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)
    predocker.runner = runner
    evaluator.shutdown()
    evaluator.shutdown()
    predocker.shutdown()
    predocker.shutdown()
    assert runner.calls == 4
    assert runner.releases == 1


def test_resource_owner_deduplicates_outer_aliases_but_not_distinct_outers_sharing_runner(tmp_path):
    runner = FakeIdempotentRunner()
    evaluator = RoseTTAFold3Evaluator(runner, tmp_path)
    predocker = RAPiDockPeptideDockingProcessor.__new__(RAPiDockPeptideDockingProcessor)
    predocker.runner = runner
    owner = ResourceOwner()
    owner.add(evaluator, "evaluator")
    owner.add(evaluator, "evaluator alias")
    owner.add(predocker, "predocker")
    owner.close()
    owner.close()
    # Identity deduplication applies to outer objects. Both distinct outers
    # delegate; their shared runner's own idempotence is what prevents re-release.
    assert runner.calls == 2
    assert runner.releases == 1


def test_existing_runner_shutdown_contract_is_repeatable_without_process():
    from threading import RLock
    for cls in (RF3RunnerService, RAPiDockRunner):
        runner = cls.__new__(cls)
        runner._lock = RLock()
        runner._proc = None
        runner.shutdown()
        runner.shutdown()
        assert runner._proc is None
