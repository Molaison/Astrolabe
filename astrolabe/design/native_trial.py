"""Reusable proposal/evaluation composition over native or custom operations.

Scientific tasks supply their operations and sample protocol. Frontier loops,
admission, deduplication and stopping belong to a SearchPolicy.
"""
from __future__ import annotations

import inspect

from .operations import Components
from .types import (
    AssessRequest, EvidenceUse, MeasureRequest, PredictRequest, ProposeRequest,
    Scored, TrialOutcome, key_for,
)


class RefinementTrial:
    fail_fast = True

    def __init__(self, system, *, predict, propose, assess, sampling, assemblies,
                 root_sample, measure=None, sources=(), initialization_order=None):
        self.system, self.predict, self.propose, self.assess = system, predict, propose, assess
        self.measure, self.sampling = measure, sampling
        self.assemblies, self.sources = tuple(assemblies), tuple(sources)
        self.root_sample = root_sample
        operations = (propose, predict, measure, assess)
        selected = {op.descriptor.alias: op for op in operations if op is not None}
        if len(selected) != sum(op is not None for op in operations):
            raise ValueError("Refinement operation aliases must be distinct")
        self.parts = Components(selected)
        self.requirements = self.parts.requirements(tuple(selected))
        self.initialization_order = tuple(initialization_order or (propose, predict))
        for component in self.initialization_order:
            if component not in operations:
                raise ValueError("Initialization owner is not part of this Trial")
        self.identity = key_for(("refinement-trial/v1", inspect.getsource(type(self)), system.identity,
            tuple((name, op.descriptor.identity) for name, op in selected.items()),
            sampling, self.assemblies, root_sample, self.sources,
            tuple(op.descriptor.alias for op in self.initialization_order)))

    def bootstrap(self, runtime):
        for component in self.initialization_order:
            initialize = getattr(component, "initialization", None)
            if initialize is not None:
                runtime.call("initialize/" + component.descriptor.alias,
                             initialize, component.initialization_request)

    def _evaluate(self, label, candidate, sample, runtime):
        evidence = runtime.call(label + "/predict", self.predict,
            PredictRequest(self.system, candidate, self.assemblies, (), self.sources, (sample,)))
        if self.measure is not None:
            evidence = runtime.call(label + "/measure", self.measure,
                MeasureRequest(self.system, candidate,
                    (EvidenceUse(evidence, "current", candidate.identity),), ()))
        assessment = runtime.call(label + "/assess", self.assess,
            AssessRequest(candidate, evidence, runtime.history, self.assess.descriptor.identity))
        return Scored(candidate, evidence, assessment)

    def root(self, candidate, runtime):
        return self._evaluate("root", candidate, self.root_sample, runtime)

    def perform(self, work, runtime):
        if work.stage in ("propose", "search"):
            if not work.parents:
                raise ValueError("Proposal work requires declared parents")
            if len(work.parents) != 1 and work.candidate is None:
                raise ValueError("Multi-parent proposal requires an explicit working candidate")
            working = work.candidate or work.parents[0].candidate
            proposed = runtime.call("propose", self.propose,
                ProposeRequest(self.system, tuple(parent.candidate for parent in work.parents),
                    working, tuple(EvidenceUse(parent.evidence, "base_guidance", working.identity)
                                   for parent in work.parents),
                    (), None, self.sampling, work.sample))
            children = ()
            if work.stage == "search":
                children = tuple(self._evaluate(f"child/{index}", proposal.candidate,
                                                work.sample, runtime)
                                 for index, proposal in enumerate(proposed.items))
            return TrialOutcome(work.identity, "completed", proposed.items, children, ())
        if work.stage == "evaluate":
            if work.candidate is None:
                raise ValueError("Evaluation work requires an admitted candidate")
            scored = self._evaluate("evaluate", work.candidate, work.sample, runtime)
            return TrialOutcome(work.identity, "completed", (), (scored,), ())
        raise ValueError(f"Unsupported refinement stage {work.stage!r}")
