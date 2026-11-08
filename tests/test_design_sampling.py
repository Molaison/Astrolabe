from __future__ import annotations

import numpy as np
import pytest

from astrolabe.design.sampling import (
    BiasDistributionRequest,
    SamplingOperationRegistry,
    bias_distribution,
    distribution_probabilities,
    fuse_log_distributions,
    ignore_distribution,
    independent_distribution,
    mix_distributions,
    sample_independent,
    temperature_distribution,
    transform_distribution,
)
from astrolabe.design.store import ArrayCodec, DataStore, JSONCodec
from astrolabe.design.types import (
    Assignment,
    Domain,
    FuseRequest,
    Occurrence,
    Polymer,
    ProposeRequest,
    Sample,
    SampleIndependentRequest,
    SamplingPolicy,
    key_for,
    make_candidate,
    make_system,
    manifest,
)


ARRAYS = ArrayCodec()
JSON = JSONCodec()


def _sample(index: str = "0", seed: int = 23) -> Sample:
    return Sample(key_for("protocol/draw"), "0", seed, index)


def _system_candidate(data: DataStore):
    topology = data.put({"cross_bonds": []}, JSON)
    occurrence = Occurrence(
        identity=key_for("occurrence/A"),
        state="state",
        assembly="assembly",
        chain="A",
        component="A_1",
        variable="A",
        index_map=data.put({"labels": ["A", "B", "C"]}, JSON),
    )
    allowed = data.put(
        {"allowed_tokens": {"1": ["A", "C"]}},
        JSON,
    )
    system = make_system(
        ("A",),
        (occurrence,),
        topology,
        (Domain("A", (0,), (), allowed),),
    )
    candidate = make_candidate(
        system,
        (
            Assignment(
                "A",
                Polymer("protein", ("A", "A", "C"), ((2, "PTM2"),)),
            ),
        ),
    )
    return system, candidate


def _distribution(
    data: DataStore,
    *,
    values=None,
    semantics: str = "sampling_probs",
    alphabet: tuple[str, ...] = ("A", "C", "G"),
):
    values = (
        np.asarray([[[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]], dtype=float)
        if values is None
        else np.asarray(values, dtype=float)
    )
    return independent_distribution(
        data,
        alphabet=alphabet,
        sites={
            "variable": "A",
            "positions": [1, 2],
            "labels": ["A:1", "A:2"],
        },
        samples=(_sample(),),
        values=values,
        semantics=semantics,
        conditioning=manifest(()),
    )


def _context(data: DataStore):
    system, candidate = _system_candidate(data)
    return ProposeRequest(
        system=system,
        parents=(candidate,),
        working=candidate,
        evidence=(),
        selection=(),
        packed=None,
        sampling=SamplingPolicy(
            key_for("sampling-policy"),
            data.put({"temperature": 1.0, "mutation_count": 1}, JSON),
        ),
        sample=_sample("proposal"),
    )


def test_logits_and_sampling_probs_have_explicit_decode_semantics():
    data = DataStore()
    raw = np.asarray([[[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]]], dtype=float)
    logits = _distribution(data, values=raw, semantics="logits")
    probabilities = _distribution(
        data,
        values=np.asarray(
            [[[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]], dtype=float
        ),
        semantics="sampling_probs",
    )

    expected = np.exp(raw) / np.exp(raw).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(distribution_probabilities(logits, data), expected)
    np.testing.assert_array_equal(
        distribution_probabilities(probabilities, data),
        np.asarray([[[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]]),
    )
    with pytest.raises(ValueError, match="nonnegative|sum to one"):
        _distribution(data, values=raw, semantics="empirical")


def test_bias_ignore_temperature_record_one_boundary_each_and_reject_duplicates():
    data = DataStore()
    distribution = _distribution(
        data,
        values=np.asarray([[[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]]], dtype=float),
        semantics="sampling_probs",
    )
    biased = bias_distribution(
        distribution,
        np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        data,
    )
    assert biased.semantics == "log_probs"
    assert [item.operation for item in biased.applied] == ["bias"]
    assert biased.applied[0].decoding_boundary.startswith("score-domain")
    np.testing.assert_allclose(
        np.exp(data.get(biased.values, ARRAYS)).sum(axis=-1),
        np.ones((1, 2)),
    )
    with pytest.raises(ValueError, match="already records"):
        bias_distribution(biased, np.zeros((1, 2, 3)), data)

    ignored = ignore_distribution(
        biased,
        np.asarray([[[True, True, False], [False, True, True]]]),
        data,
    )
    assert [item.operation for item in ignored.applied] == ["bias", "ignore"]
    cooled = temperature_distribution(ignored, 0.5, data)
    assert [item.operation for item in cooled.applied] == [
        "bias",
        "ignore",
        "temperature",
    ]
    assert cooled.semantics == "log_probs"

    with pytest.raises(ValueError, match="retain at least one"):
        ignore_distribution(
            _distribution(data),
            np.zeros((1, 2, 3), dtype=bool),
            data,
        )
    with pytest.raises(ValueError, match="autoregressive/joint"):
        transform_distribution(
            distribution,
            operation_name="autoregressive",
            parameter_reference=data.put({"order": [0, 1]}, JSON),
            data=data,
        )


def test_probability_mixture_and_log_fusion_are_distinct_typed_results():
    data = DataStore()
    left = _distribution(
        data,
        values=np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        semantics="logits",
    )
    right = _distribution(
        data,
        values=np.asarray([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]),
        semantics="logits",
    )
    request = FuseRequest((left, right), (0.25, 0.75))

    mixture = mix_distributions(request, data)
    log_fusion = fuse_log_distributions(request, data)
    assert mixture.semantics == "sampling_probs"
    assert log_fusion.semantics == "logits"
    assert mixture.applied[-1].operation == "probability_mixture"
    assert log_fusion.applied[-1].operation == "log_fusion"
    assert mixture.values != log_fusion.values
    np.testing.assert_allclose(
        data.get(mixture.values, ARRAYS).sum(axis=-1),
        np.ones((1, 2)),
    )

    with pytest.raises(ValueError, match="separate operation"):
        fuse_log_distributions(FuseRequest((mixture, right), (0.5, 0.5)), data)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        mix_distributions(FuseRequest((left, right), (1.0, -1.0)), data)


def test_log_fusion_skips_zero_mask_weights_and_rejects_empty_row_intersection():
    data = DataStore()
    sites = {
        "sites": [{"variable": "A", "position": 1, "label": "A:1"}]
    }
    left = independent_distribution(
        data,
        alphabet=("A", "C", "G"),
        sites=sites,
        samples=(_sample(),),
        values=np.asarray([[[-np.inf, 0.0, 0.0]]]),
        semantics="logits",
        conditioning=manifest(()),
    )
    right = independent_distribution(
        data,
        alphabet=("A", "C", "G"),
        sites=sites,
        samples=(_sample(),),
        values=np.asarray([[[0.0, 0.0, -np.inf]]]),
        semantics="logits",
        conditioning=manifest(()),
    )
    fused = fuse_log_distributions(FuseRequest((left, right), (0.0, 1.0)), data)
    np.testing.assert_array_equal(
        data.get(fused.values, ARRAYS), np.asarray([[[0.0, 0.0, -np.inf]]])
    )
    assert any(
        dependency.role == "fuse-input"
        and dependency.identity == left.identity
        for dependency in fused.conditioning.dependencies
    )

    second = independent_distribution(
        data,
        alphabet=("A", "C", "G"),
        sites=sites,
        samples=(_sample(),),
        values=np.asarray([[[0.0, -np.inf, -np.inf]]]),
        semantics="logits",
        conditioning=manifest(()),
    )
    third = independent_distribution(
        data,
        alphabet=("A", "C", "G"),
        sites=sites,
        samples=(_sample(),),
        values=np.asarray([[[-np.inf, -np.inf, 0.0]]]),
        semantics="logits",
        conditioning=manifest(()),
    )
    conflicting = independent_distribution(
        data,
        alphabet=("A", "C", "G"),
        sites=sites,
        samples=(_sample(),),
        values=np.asarray([[[-np.inf, 0.0, -np.inf]]]),
        semantics="logits",
        conditioning=manifest(()),
    )
    with pytest.raises(ValueError, match="excludes every alphabet entry"):
        fuse_log_distributions(
            FuseRequest(
                (second, third, conflicting),
                (1.0, 1.0, 1.0),
            ),
            data,
        )


def test_empirical_semantics_are_finite_probabilities_not_logits():
    data = DataStore()
    empirical = _distribution(
        data,
        values=np.asarray([[[0.25, 0.25, 0.5], [0.1, 0.8, 0.1]]]),
        semantics="empirical",
    )
    np.testing.assert_array_equal(
        distribution_probabilities(empirical, data),
        np.asarray([[[0.25, 0.25, 0.5], [0.1, 0.8, 0.1]]]),
    )
    with pytest.raises(ValueError, match="empirical distributions require"):
        temperature_distribution(empirical, 0.5, data)


def test_independent_sampling_is_seeded_and_preserves_fixed_token_and_ptm():
    data = DataStore()
    distribution = _distribution(
        data,
        values=np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
    )
    request = SampleIndependentRequest(
        distribution=distribution,
        context=_context(data),
        count=2,
    )

    first = sample_independent(request, data, np.random.default_rng(101))
    second = sample_independent(request, data, np.random.default_rng(101))
    assert first == second
    assert len(first.items) == 2
    assert first.items[0].identity != first.items[1].identity
    for proposal in first.items:
        polymer = proposal.candidate.assignments[0].value
        assert polymer.tokens[0] == "A"
        assert polymer.modifications == ((2, "PTM2"),)
        assert polymer.tokens[1] in {"A", "C"}

    forbidden = _distribution(
        data,
        values=np.asarray([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
    )
    with pytest.raises(ValueError, match="is not allowed"):
        sample_independent(
            SampleIndependentRequest(forbidden, request.context, 1),
            data,
            np.random.default_rng(1),
        )
    with pytest.raises(TypeError, match="explicit"):
        sample_independent(request, data, np.random.RandomState(1))

    fixed_sites = independent_distribution(
        data,
        alphabet=("A", "C", "G"),
        sites={
            "variable": "A",
            "positions": [0, 1, 2],
            "labels": ["A:0", "A:1", "A:2"],
        },
        samples=(_sample(),),
        values=np.asarray(
            [
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ]
        ),
        semantics="sampling_probs",
        conditioning=manifest(()),
    )
    with pytest.raises(ValueError, match="fixed token"):
        sample_independent(
            SampleIndependentRequest(fixed_sites, request.context, 1),
            data,
            np.random.default_rng(1),
        )


def test_typed_sampling_operations_execute_transforms_fusion_and_draws():
    data = DataStore()
    distribution = _distribution(data)
    bias = data.put(np.zeros((1, 2, 3)), ARRAYS)
    registry = SamplingOperationRegistry

    bias_operation = registry.build_operation(
        "bias",
        alias="bias",
        options={},
        context=data,
    )
    transformed = bias_operation.invoke(
        bias_operation.prepare(BiasDistributionRequest(distribution, bias))
    )
    assert transformed.applied[0].operation == "bias"

    mix_operation = registry.build_operation(
        "probability_mixture",
        alias="mixture",
        options={},
        context=data,
    )
    mixed = mix_operation.invoke(
        mix_operation.prepare(FuseRequest((distribution, transformed), (0.5, 0.5)))
    )
    assert mixed.semantics == "sampling_probs"

    draw_operation = registry.build_operation(
        "sample_independent",
        alias="draw",
        options={},
        context=data,
    )
    drawn = draw_operation.invoke(
        draw_operation.prepare(
            SampleIndependentRequest(distribution, _context(data), 1)
        )
    )
    assert len(drawn.items) == 1

    resumed_operation = registry.build_operation(
        "sample_independent",
        alias="draw",
        options={},
        context=data,
    )
    resumed = resumed_operation.invoke(
        resumed_operation.prepare(
            SampleIndependentRequest(distribution, _context(data), 1)
        )
    )
    assert resumed == drawn


def test_canonical_sites_apply_multisegment_replacements_atomically():
    data = DataStore()
    topology = data.put({"cross_bonds": []}, JSON)
    allowed = data.put(
        {"allowed_tokens": {"0": ["A", "C"], "1": ["A", "C"]}},
        JSON,
    )
    system = make_system(
        ("A", "B"),
        (),
        topology,
        (
            Domain("A", (), (), allowed),
            Domain("B", (), (), allowed),
        ),
    )
    candidate = make_candidate(
        system,
        (
            Assignment(
                "A",
                Polymer("protein", ("A", "C"), ((1, "PTM_A1"),)),
            ),
            Assignment("B", Polymer("rna", ("A", "A"), ())),
        ),
    )
    context = ProposeRequest(
        system=system,
        parents=(candidate,),
        working=candidate,
        evidence=(),
        selection=(),
        packed=None,
        sampling=SamplingPolicy(
            key_for("sampling-policy"),
            data.put({"temperature": 1.0, "mutation_count": 1}, JSON),
        ),
        sample=_sample("proposal"),
    )
    distribution = independent_distribution(
        data,
        alphabet=("A", "C"),
        sites={
            "sites": [
                {"variable": "A", "position": 0, "label": "A:0"},
                {"variable": "B", "position": 1, "label": "B:1"},
            ]
        },
        samples=(_sample(),),
        values=np.asarray([[[0.0, 1.0], [0.0, 1.0]]]),
        semantics="sampling_probs",
        conditioning=manifest(()),
    )

    proposals = sample_independent(
        SampleIndependentRequest(distribution, context, 1),
        data,
        np.random.default_rng(9),
    )
    assignments = {
        assignment.variable: assignment.value
        for assignment in proposals.items[0].candidate.assignments
    }
    assert assignments["A"].tokens == ("C", "C")
    assert assignments["A"].modifications == ((1, "PTM_A1"),)
    assert assignments["B"].tokens == ("A", "C")

    with pytest.raises(ValueError, match="duplicate mapped site"):
        independent_distribution(
            data,
            alphabet=("A", "C"),
            sites={
                "sites": [
                    {"variable": "A", "position": 0, "label": "A:0-a"},
                    {"variable": "A", "position": 0, "label": "A:0-b"},
                ]
            },
            samples=(_sample(),),
            values=np.asarray([[[1.0, 0.0], [1.0, 0.0]]]),
            semantics="sampling_probs",
            conditioning=manifest(()),
        )
