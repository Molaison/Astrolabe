from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random

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
        data=data,
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
    expected_probs = np.asarray([[[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]])
    expected_probs /= expected_probs.sum(axis=-1, keepdims=True)
    np.testing.assert_array_equal(distribution_probabilities(probabilities, data), expected_probs)
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
        data=data,
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


def _one_site(data, values, semantics="sampling_probs", samples=None):
    return independent_distribution(
        data,
        alphabet=("A", "C"),
        sites={"sites": [{"variable": "A", "position": 1, "label": "A:1"}]},
        samples=(_sample(),) if samples is None else samples,
        values=np.asarray(values),
        semantics=semantics,
        conditioning=manifest(()),
    )


@pytest.mark.parametrize("bias", [[0.0, 0.0], [2.0, -1.0], [1000.0, 1000.0]])
def test_probability_bias_matches_weighted_probability_formula(bias):
    data = DataStore()
    probabilities = np.array([0.9, 0.1])
    source = _one_site(data, [[probabilities]])
    result = bias_distribution(source, bias, data)
    expected = probabilities * np.exp(np.array(bias) - max(bias))
    expected /= expected.sum()
    np.testing.assert_allclose(distribution_probabilities(result, data)[0, 0], expected)


def test_bias_cannot_revive_zero_probability_or_masked_entries():
    data = DataStore()
    source = _one_site(data, [[[1.0, 0.0]]])
    biased = bias_distribution(source, [0.0, 1000.0], data)
    np.testing.assert_array_equal(distribution_probabilities(biased, data), [[[1, 0]]])
    masked = ignore_distribution(_one_site(data, [[[0.2, 0.8]]]), [True, False], data)
    biased = bias_distribution(masked, [0.0, 1000.0], data)
    np.testing.assert_array_equal(distribution_probabilities(biased, data), [[[1, 0]]])


@pytest.mark.parametrize("values", [[[0.0, 0.0]], [[-2.0, -2.0]]])
def test_log_probs_reject_unnormalized_rows(values):
    data = DataStore()
    with pytest.raises(ValueError, match="normalized"):
        _one_site(data, [values], "log_probs")


def test_log_probs_mask_renormalizes_and_remains_sampleable():
    data = DataStore()
    source = _one_site(data, np.log([[[0.2, 0.8]]]), "log_probs")
    masked = ignore_distribution(source, [True, False], data)
    np.testing.assert_array_equal(data.get(masked.values, ARRAYS), [[[0.0, -np.inf]]])
    np.testing.assert_array_equal(distribution_probabilities(masked, data), [[[1.0, 0.0]]])
    proposals = sample_independent(
        SampleIndependentRequest(masked, _context(data), 1), data, np.random.default_rng(1)
    )
    assert proposals.items[0].candidate.assignments[0].value.tokens == ("A", "A", "C")


@pytest.mark.parametrize("semantics", ["sampling_probs", "empirical", "logits", "log_probs"])
def test_mask_rejects_zero_effective_mass_at_transform_boundary(semantics):
    data = DataStore()
    values = [[[0.0, -np.inf]]] if semantics in {"logits", "log_probs"} else [[[1.0, 0.0]]]
    source = _one_site(data, values, semantics)
    with pytest.raises(ValueError, match="zero effective mass"):
        ignore_distribution(source, [False, True], data)


@pytest.mark.parametrize("weights", [(0.25, 0.75), (1e308, 1e308), (1e-308, 1e-308)])
def test_mixture_and_log_fusion_match_independent_numerical_references(weights):
    data = DataStore()
    left_p = np.array([0.9, 0.1])
    right_p = np.array([0.2, 0.8])
    scaled = np.array(weights) / max(weights)
    alpha = scaled / scaled.sum()
    left = _one_site(data, [[left_p]])
    right = _one_site(data, [[right_p]])
    mixed = mix_distributions(FuseRequest((left, right), weights), data)
    assert data.get(mixed.values, ARRAYS).shape == (1, 1, 2)
    np.testing.assert_allclose(
        distribution_probabilities(mixed, data)[0, 0], alpha[0] * left_p + alpha[1] * right_p
    )
    expected = left_p ** alpha[0] * right_p ** alpha[1]
    expected /= expected.sum()
    for semantics in ("logits", "log_probs"):
        left = _one_site(data, np.log([[left_p]]), semantics)
        right = _one_site(data, np.log([[right_p]]), semantics)
        fused = fuse_log_distributions(FuseRequest((left, right), weights), data)
        assert data.get(fused.values, ARRAYS).shape == (1, 1, 2)
        np.testing.assert_allclose(distribution_probabilities(fused, data)[0, 0], expected)


@pytest.mark.parametrize("fusion", [mix_distributions, fuse_log_distributions])
def test_weight_underflow_cannot_silently_drop_a_source(fusion):
    data = DataStore()
    left = _one_site(data, [[[0.0, -np.inf]]], "logits")
    right = _one_site(data, [[[-np.inf, 0.0]]], "logits")
    with pytest.raises(ValueError, match="weight normalization underflow"):
        fusion(FuseRequest((left, right), (1e308, 5e-324)), data)


@pytest.mark.parametrize("semantics", ["logits", "log_probs"])
def test_fusion_retains_mask_support_with_huge_and_zero_weights(semantics):
    data = DataStore()
    left = _one_site(data, [[[0.0, -np.inf]]], semantics)
    right = _one_site(data, [[[-np.inf, 0.0]]], semantics)
    request = FuseRequest((left, right), (1e308, 1e308))
    mixed = mix_distributions(request, data)
    np.testing.assert_array_equal(distribution_probabilities(mixed, data), [[[0.5, 0.5]]])
    with pytest.raises(ValueError, match="excluded|excludes every"):
        fuse_log_distributions(request, data)
    for fusion in (mix_distributions, fuse_log_distributions):
        result = fusion(FuseRequest((left, right), (0.0, 1e308)), data)
        assert data.get(result.values, ARRAYS).shape == (1, 1, 2)
        np.testing.assert_array_equal(distribution_probabilities(result, data), [[[0.0, 1.0]]])


def test_distribution_shape_and_numeric_types_fail_closed():
    data = DataStore()
    for values in ([[1.0, 0.0]], [[[1.0 + 0j, 0j]]], [[[True, False]]]):
        with pytest.raises((ValueError, TypeError)):
            _one_site(data, values)
    source = _one_site(data, [[[1.0, 0.0]]])
    malformed = replace(source, values=data.put(np.array([[1.0, 0.0]]), ARRAYS))
    with pytest.raises(ValueError, match="shape"):
        mix_distributions(FuseRequest((source, malformed), (0.5, 0.5)), data)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int64, np.float32, np.float64])
def test_logit_decode_casts_before_subtraction_and_exponentiation(dtype):
    data = DataStore()
    source = _one_site(data, np.array([[[127, -128]]], dtype=dtype), "logits")
    expected = np.array([1.0, np.exp(-255.0)])
    expected /= expected.sum()
    actual = distribution_probabilities(source, data)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual[0, 0], expected, rtol=1e-12, atol=0)


def test_duplicate_sample_identities_rejected_before_row_selection():
    data = DataStore()
    with pytest.raises(ValueError, match="duplicate sample"):
        _one_site(data, [[[1.0, 0.0]], [[0.0, 1.0]]], samples=(_sample(), _sample()))
    with pytest.raises(ValueError, match="non-empty"):
        _one_site(data, np.zeros((0, 1, 2)), samples=())
    first, second = _sample("first"), _sample("second")
    source = _one_site(data, [[[1.0, 0.0]], [[0.0, 1.0]]], samples=(first, second))
    context = replace(_context(data), sample=second)
    proposals = sample_independent(
        SampleIndependentRequest(source, context, 1), data, np.random.default_rng(1)
    )
    assert proposals.items[0].candidate.assignments[0].value.tokens[1] == "C"
    with pytest.raises(ValueError, match="multi-sample"):
        sample_independent(
            SampleIndependentRequest(source, _context(data), 1), data, np.random.default_rng(1)
        )


@pytest.mark.parametrize("semantics", ["sampling_probs", "empirical", "logits", "log_probs"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_distribution_rejects_nonfinite_input(semantics, invalid):
    with pytest.raises(ValueError):
        _one_site(DataStore(), [[[invalid, 0.0]]], semantics)


def test_nonfinite_transform_and_arithmetic_fail_closed():
    data = DataStore()
    source = _one_site(data, [[[1e308, 0.0]]], "logits")
    for bias in ([np.nan, 0], [np.inf, 0], [-np.inf, 0], [1e308, 0]):
        with pytest.raises(ValueError):
            bias_distribution(source, bias, data)
    for temperature in (np.nan, np.inf, 0.0, -1.0, 1e-308):
        with pytest.raises(ValueError):
            temperature_distribution(source, temperature, data)
    with pytest.raises(ValueError, match="arithmetic"):
        distribution_probabilities(_one_site(data, [[[1e308, -1e308]]], "logits"), data)
    with pytest.raises(ValueError, match="underflow"):
        distribution_probabilities(_one_site(data, [[[0.0, -1000.0]]], "logits"), data)
    for weights in ((0.0, 0.0), (np.nan, 1.0), (np.inf, 1.0), (-1.0, 2.0)):
        with pytest.raises(ValueError):
            mix_distributions(FuseRequest((source, source), weights), data)


@pytest.mark.parametrize("semantics", ["sampling_probs", "empirical", "logits", "log_probs"])
def test_negative_infinity_is_only_allowed_as_partial_log_support(semantics):
    data = DataStore()
    with pytest.raises(ValueError):
        _one_site(data, [[[-np.inf, -np.inf]]], semantics)
    if semantics in {"sampling_probs", "empirical"}:
        with pytest.raises(ValueError, match="finite"):
            _one_site(data, [[[1.0, -np.inf]]], semantics)
    else:
        source = _one_site(data, [[[0.0, -np.inf]]], semantics)
        np.testing.assert_array_equal(distribution_probabilities(source, data), [[[1.0, 0.0]]])


@pytest.mark.skipif(np.finfo(np.longdouble).max <= np.finfo(float).max,
                    reason="longdouble has no wider exponent range on this platform")
def test_float_conversion_rejects_overflow_and_support_underflow():
    data = DataStore()
    too_large = np.array([[[np.longdouble("1e400"), 0]]], dtype=np.longdouble)
    with pytest.raises(ValueError, match="arithmetic"):
        _one_site(data, too_large, "logits")
    too_small = np.array([[[np.longdouble("1e-400"), 1]]], dtype=np.longdouble)
    with pytest.raises(ValueError, match="arithmetic"):
        _one_site(data, too_small)


def test_rng_consumption_global_state_and_single_sample_fallback_are_preserved():
    data = DataStore()
    source = _distribution(data, values=[[[0.3, 0.7, 0.0], [0.0, 1.0, 0.0]]])
    context = _context(data)
    assert context.sample not in source.samples  # Existing singleton fallback.
    request = SampleIndependentRequest(source, context, 4)
    rng = np.random.default_rng(101)
    reference = np.random.default_rng(101)
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    result = sample_independent(request, data, rng)
    for proposal in result.items:
        expected = tuple(source.alphabet[int(reference.choice(3, p=row))]
                         for row in distribution_probabilities(source, data)[0])
        assert proposal.candidate.assignments[0].value.tokens[1:] == expected
    assert rng.bit_generator.state == reference.bit_generator.state
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]
    assert random.getstate() == python_state
    first = SamplingOperationRegistry.build_operation("sample_independent", alias="draw", options={}, context=data)
    second = SamplingOperationRegistry.build_operation("sample_independent", alias="draw", options={}, context=data)
    assert first.invoke(first.prepare(request)) == second.invoke(second.prepare(request))


def test_sampling_still_rejects_changes_to_modified_tokens():
    data = DataStore()
    source = _distribution(data, values=[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    with pytest.raises(ValueError, match="modified token"):
        sample_independent(
            SampleIndependentRequest(source, _context(data), 1), data, np.random.default_rng(1)
        )


@pytest.mark.parametrize("dependency", ["sampling.py", "types.py", "store.py", "operations.py", "numpy"])
def test_operation_identity_binds_direct_dependency_content_and_numpy_version(monkeypatch, dependency):
    data = DataStore()
    def build():
        return SamplingOperationRegistry.build_operation("sample_independent", alias="draw", options={}, context=data)
    original = build().descriptor.identity
    if dependency == "numpy":
        monkeypatch.setattr(np, "__version__", np.__version__ + "+identity-test")
    else:
        read_bytes = Path.read_bytes
        def changed_bytes(path):
            content = read_bytes(path)
            return content + b"\n# identity-test" if path.name == dependency else content
        monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    assert build().descriptor.identity != original


@pytest.mark.parametrize("semantics", ["sampling_probs", "empirical"])
def test_float32_probability_decode_normalizes_before_sampling(semantics):
    data = DataStore()
    raw = np.array([[[0.1, 0.9]]], dtype=np.float32)
    source = _one_site(data, raw, semantics)
    actual = distribution_probabilities(source, data)
    expected = raw.astype(float) / raw.astype(float).sum(axis=-1, keepdims=True)
    np.testing.assert_array_equal(actual, expected)
    assert actual.sum() == 1.0
    rng, reference = np.random.default_rng(71), np.random.default_rng(71)
    request = SampleIndependentRequest(source, _context(data), 12)
    proposals = sample_independent(request, data, rng)
    for proposal in proposals.items:
        index = int(reference.choice(2, p=expected[0, 0]))
        assert proposal.candidate.assignments[0].value.tokens[1] == source.alphabet[index]
    assert rng.bit_generator.state == reference.bit_generator.state
    masked = _one_site(data, np.array([[[0.0, 0.99999994]]], dtype=np.float32), semantics)
    np.testing.assert_array_equal(distribution_probabilities(masked, data), [[[0.0, 1.0]]])


@pytest.mark.parametrize("offset", [1e16, -1e16, 1e308, -1e308])
def test_probability_bias_is_invariant_to_large_common_offset(offset):
    data = DataStore()
    source = _one_site(data, [[[0.1, 0.9]]])
    result = bias_distribution(source, [offset, offset], data)
    np.testing.assert_allclose(distribution_probabilities(result, data), [[[0.1, 0.9]]], rtol=1e-14)


def test_nonconstant_bias_offset_and_excluded_bias_reference():
    data = DataStore()
    source = _one_site(data, [[[0.1, 0.9]]])
    plain = bias_distribution(source, [0.0, 2.0], data)
    shifted = bias_distribution(source, [1e16, 1e16 + 2.0], data)
    np.testing.assert_allclose(distribution_probabilities(plain, data),
                               distribution_probabilities(shifted, data), rtol=1e-14)
    source = _distribution(data, values=[[[0.1, 0.9, 0.0], [0.2, 0.8, 0.0]]])
    result = bias_distribution(source, [[0.0, 0.0, 1e308], [1e16, 1e16, 0.0]], data)
    np.testing.assert_allclose(distribution_probabilities(result, data),
                               distribution_probabilities(source, data), rtol=1e-14)
    np.testing.assert_array_equal(distribution_probabilities(result, data)[..., 2], 0.0)


@pytest.mark.parametrize("offset", [1e16, -1e16, 1e308, -1e308])
def test_log_fusion_removes_each_sources_common_offset_before_combining(offset):
    data = DataStore()
    informative = _one_site(data, [[[0.0, 2.0]]], "logits")
    flat = _one_site(data, [[[0.0, 0.0]]], "logits")
    shifted_flat = _one_site(data, [[[offset, offset]]], "logits")
    plain = fuse_log_distributions(FuseRequest((informative, flat), (0.5, 0.5)), data)
    shifted = fuse_log_distributions(FuseRequest((informative, shifted_flat), (0.5, 0.5)), data)
    expected = np.exp([-1.0, 0.0]); expected /= expected.sum()
    np.testing.assert_allclose(distribution_probabilities(shifted, data)[0, 0], expected, rtol=1e-14)
    np.testing.assert_array_equal(data.get(plain.values, ARRAYS), data.get(shifted.values, ARRAYS))


def test_log_fusion_removes_nonconstant_source_offset_and_fails_unrepresentable_range():
    data = DataStore()
    left = _one_site(data, [[[0.0, 1.0]]], "logits")
    right = _one_site(data, [[[0.0, 2.0]]], "logits")
    shifted = _one_site(data, [[[1e16, 1e16 + 2.0]]], "logits")
    plain = fuse_log_distributions(FuseRequest((left, right), (0.5, 0.5)), data)
    actual = fuse_log_distributions(FuseRequest((left, shifted), (0.5, 0.5)), data)
    np.testing.assert_array_equal(data.get(plain.values, ARRAYS), data.get(actual.values, ARRAYS))
    extreme = _one_site(data, [[[1e308, -1e308]]], "logits")
    with pytest.raises(ValueError, match="arithmetic"):
        fuse_log_distributions(FuseRequest((extreme, left), (0.5, 0.5)), data)
    with pytest.raises(ValueError, match="arithmetic"):
        bias_distribution(_one_site(data, [[[0.1, 0.9]]]), [1e308, -1e308], data)


@pytest.mark.parametrize("dtype,values", [
    (np.int64, [2**53, 2**53 + 1]),
    (np.int64, [2**60, 2**60 + 1]),
    (np.int64, [2**63 - 2, 2**63 - 1]),
    (np.uint64, [2**64 - 2, 2**64 - 1]),
])
def test_inexact_integer_logits_and_bias_fail_closed(dtype, values):
    data = DataStore()
    with pytest.raises(ValueError, match="integer-to-float.*lose exact"):
        _one_site(data, np.array([[values]], dtype=dtype), "logits")
    with pytest.raises(ValueError, match="integer-to-float.*lose exact"):
        bias_distribution(_one_site(data, [[[0.1, 0.9]]]), np.array(values, dtype=dtype), data)


def test_exact_large_integer_logits_retain_representable_differences():
    data = DataStore()
    source = _one_site(data, np.array([[[2**53, 2**53 + 2]]], dtype=np.int64), "logits")
    expected = np.exp([-2.0, 0.0]); expected /= expected.sum()
    np.testing.assert_allclose(distribution_probabilities(source, data)[0, 0], expected, rtol=1e-14)


def test_log_fusion_centers_on_joint_support_before_combining():
    data = DataStore()
    left = _distribution(data, values=[[[-1e308, -1e308, 0.0]] * 2], semantics="logits")
    right = _distribution(data, values=[[[0.0, 2.0, -np.inf]] * 2], semantics="logits")
    fused = fuse_log_distributions(FuseRequest((left, right), (0.5, 0.5)), data)
    expected = np.exp([-1.0, 0.0, -np.inf]); expected /= expected.sum()
    np.testing.assert_allclose(distribution_probabilities(fused, data), [[expected, expected]], rtol=1e-14)
