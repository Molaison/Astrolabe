"""CPU tutorial for shared variables and interchangeable search policies.

The objective is the fraction of A tokens in a short sequence. This exercises
composition and recovery; it performs no structure prediction or protein-quality
assessment. Run with --output DIR; repeating the command resumes the same run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from astrolabe import Astrolabe
from astrolabe.design import (
    ArrayCodec, Assignment, Axis, Components, DataStore, Dependency, Domain,
    Evidence, EvidenceUse, JSONCodec, Measurement, Occurrence, Polymer, Reduction,
    Runtime, Scored, Subject, TrialOutcome, key_for, make_candidate, make_system,
    manifest, operation,
)
from astrolabe.design.composition import DesignRun
from astrolabe.design.measurements import measurement_view
from astrolabe.design.sampling import independent_distribution, sample_independent
from astrolabe.design.search import (
    BeamSearchPolicy, PopulationSearchPolicy, TrajectorySearchPolicy,
)
from astrolabe.design.types import (
    Assessment, AssessRequest, MeasureRequest, Proposals, ProposeRequest,
    SampleIndependentRequest, SamplingPolicy,
)


class SequenceTrial:
    def __init__(self, system, data):
        self.system, self.data = system, data
        self.protocol = key_for("tutorial/uniform-independent/v1")
        self.sampling = SamplingPolicy(self.protocol, data.put({}, JSONCodec()))
        self.propose = operation("propose", ProposeRequest, Proposals,
            self._propose, version="tutorial/v1")
        self.measure = operation("measure", MeasureRequest, Measurement,
            self._measure, version="tutorial/v1")
        self.assess = operation("assess", AssessRequest, Assessment,
            self._assess, version="tutorial/v1")
        self.parts = Components({op.descriptor.alias: op for op in
                                 (self.propose, self.measure, self.assess)})
        self.requirements = self.parts.requirements(("propose", "measure", "assess"))
        self.identity = key_for(("tutorial/trial/v1", system.identity, self.protocol))

    def _propose(self, request):
        distribution = independent_distribution(self.data, alphabet=("A", "G"),
            sites={"sites": [{"variable": "shared", "position": i, "label": f"shared:{i}"}
                             for i in (1, 2, 3)]},
            samples=(request.sample,), values=np.zeros((1, 3, 2)),
            semantics="logits", conditioning=manifest((Dependency("base", request.working.identity),)))
        return sample_independent(SampleIndependentRequest(distribution, request, 1),
                                  self.data, np.random.default_rng(request.sample.seed))

    def _measure(self, request):
        shared = next(a.value for a in request.candidate.assignments if a.variable == "shared")
        values = self.data.put(np.asarray([token == "A" for token in shared.tokens], dtype=float), ArrayCodec())
        site_map = self.data.put({"labels": [f"shared:{i}" for i in range(len(shared.tokens))]}, JSONCodec())
        definition, producer = key_for("indicator-A/v1"), key_for("tutorial/composition/v1")
        subject = Subject(request.candidate.identity, None, None, (), None, None)
        inputs = manifest((Dependency("candidate", request.candidate.identity), Dependency("definition", definition)))
        return Measurement(key_for((definition, values, subject)), "composition.A", definition,
            "fraction", producer, (subject,), (Axis("site", "variable_token", site_map),),
            values, "present", None, inputs)

    def _assess(self, request):
        metric, = request.evidence.measurements
        reduced = measurement_view(metric, self.data).reduce(
            Reduction(("site",), "mean", self.data.put({}, JSONCodec()), "fail"))
        value = float(self.data.get(reduced.values, ArrayCodec()))
        inputs = manifest((Dependency("metric", reduced.identity),
                           Dependency("history", request.history.revision)))
        return Assessment(request.evidence.identity, request.objective_revision,
                          request.history.revision, (value,), (), inputs)

    def _evaluate(self, label, candidate, runtime):
        measurement = runtime.call(label + "/measure", self.measure,
                                  MeasureRequest(self.system, candidate, (), ()))
        evidence = Evidence(key_for("empty"), candidate.identity, (), (), manifest(())).with_measurements((measurement,))
        assessment = runtime.call(label + "/assess", self.assess,
            AssessRequest(candidate, evidence, runtime.history, key_for("fraction-A/v1")))
        return Scored(candidate, evidence, assessment)

    def root(self, candidate, runtime):
        return self._evaluate("root", candidate, runtime)

    def perform(self, work, runtime):
        parent, = work.parents
        proposed = runtime.call("propose", self.propose, ProposeRequest(self.system,
            tuple(p.candidate for p in work.parents), parent.candidate,
            (EvidenceUse(parent.evidence, "base_guidance", parent.candidate.identity),),
            (), None, self.sampling, work.sample))
        proposal, = proposed.items
        child = self._evaluate("child", proposal.candidate, runtime)
        return TrialOutcome(work.identity, "completed", proposed.items, (child,), ())


def run_example(output: Path, mode="beam"):
    data = DataStore(output / "data")
    mapping = data.put({"labels": [f"shared:{i}" for i in range(4)]}, JSONCodec())
    occurrences = tuple(Occurrence(key_for(("occurrence", state)), state, state,
        chain, "shared", "shared", mapping) for state, chain in (("bound", "A"), ("free", "B")))
    system = make_system(("shared", "fixed_tail"), occurrences,
        data.put({"bonds": []}, JSONCodec()),
        (Domain("shared", (0,), (), None), Domain("fixed_tail", (0, 1), (), None)))
    initial = make_candidate(system, (Assignment("shared", Polymer("protein", tuple("GGGG"), ())),
                                     Assignment("fixed_tail", Polymer("protein", tuple("GA"), ()))))
    trial = SequenceTrial(system, data)
    policies = {"trajectory": TrajectorySearchPolicy, "beam": BeamSearchPolicy,
                "population": PopulationSearchPolicy}
    options = dict(rounds=3, offspring=2, max_work=9, seed=17, directions=("max",))
    if mode != "trajectory":
        options["width"] = 2
    policy = policies[mode](**options)
    with Runtime(trial.parts, root=output, data=data) as runtime:
        return Astrolabe(DesignRun(system, initial, trial, policy, runtime)).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search", choices=("trajectory", "beam", "population"), default="beam")
    args = parser.parse_args()
    result = run_example(args.output, args.search)
    for selected in result.selected:
        print({a.variable: "".join(a.value.tokens) for a in selected.candidate.assignments},
              selected.assessment.values)
    print(result.manifest.selected_summary)
