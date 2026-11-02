"""Typed independent-distribution transforms, fusion, and sampling.

The module works in two explicit domains. ``logits``/``log_probs`` are score
domains; ``sampling_probs``/``empirical`` are probability domains. A transform
that crosses that boundary converts only where its definition requires it and
records the boundary in ``Applied.decoding_boundary``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .operations import OperationOwner, operation
from .store import ArrayCodec, DataStore, JSONCodec
from .types import (
    Applied,
    Assignment,
    Candidate,
    Dependency,
    FuseRequest,
    IndependentDistribution,
    Polymer,
    Proposal,
    Proposals,
    Sample,
    SampleIndependentRequest,
    SamplingPolicy,
    key_for,
    manifest,
)


_ARRAYS = ArrayCodec()
_JSON = JSONCodec()
_LOG_DOMAINS = frozenset({"logits", "log_probs"})
_PROBABILITY_DOMAINS = frozenset({"sampling_probs", "empirical"})
_SEMANTICS = _LOG_DOMAINS | _PROBABILITY_DOMAINS


@contextmanager
def _checked_arithmetic():
    """Reject unrepresentable arithmetic instead of persisting invalid results."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            yield
    except FloatingPointError as exc:
        raise ValueError(f"non-finite or unrepresentable sampling arithmetic: {exc}") from exc


def _float_array(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be real numeric")
    with _checked_arithmetic():
        converted = array.astype(float, copy=False)
    if np.issubdtype(array.dtype, np.integer) and any(
        int(original) != int(floating)
        for original, floating in zip(array.flat, converted.flat, strict=True)
    ):
        raise ValueError(f"{name}: integer-to-float conversion would lose exact values")
    return converted


@dataclass(frozen=True)
class BiasDistributionRequest:
    distribution: IndependentDistribution
    bias: object


@dataclass(frozen=True)
class IgnoreDistributionRequest:
    distribution: IndependentDistribution
    mask: object


@dataclass(frozen=True)
class TemperatureDistributionRequest:
    distribution: IndependentDistribution
    temperature: object


def _store_data(value) -> DataStore:
    if isinstance(value, DataStore):
        return value
    data = getattr(value, "data", None)
    if isinstance(data, DataStore):
        return data
    raise TypeError("context must be a DataStore or expose a DataStore as .data")


def _array(data: DataStore, reference) -> np.ndarray:
    value = data.get(reference, _ARRAYS)
    if not isinstance(value, np.ndarray):
        raise TypeError("distribution parameters must decode to numpy arrays")
    return value


def _sites(data: DataStore, distribution: IndependentDistribution) -> tuple[dict, ...]:
    payload = data.get(distribution.sites, _JSON)
    if not isinstance(payload, Mapping):
        raise ValueError("sites must be a JSON object")
    if set(payload) == {"sites"}:
        raw_sites = payload["sites"]
        if not isinstance(raw_sites, list) or not raw_sites:
            raise ValueError("sites must be a non-empty list")
        sites = []
        for item in raw_sites:
            if not isinstance(item, Mapping) or set(item) != {
                "variable",
                "position",
                "label",
            }:
                raise ValueError(
                    "each site must contain exactly variable, position, and label"
                )
            sites.append(dict(item))
    elif set(payload) == {"variable", "positions", "labels"}:
        variable = payload["variable"]
        positions = payload["positions"]
        labels = payload["labels"]
        if not isinstance(positions, list) or not isinstance(labels, list):
            raise ValueError("legacy positions and labels must be lists")
        if len(positions) != len(labels):
            raise ValueError("legacy site labels must match positions")
        sites = [
            {"variable": variable, "position": position, "label": label}
            for position, label in zip(positions, labels, strict=True)
        ]
    else:
        raise ValueError(
            "sites must use canonical sites[] or legacy variable/positions/labels"
        )

    coordinates: set[tuple[str, int]] = set()
    labels: set[str] = set()
    for site in sites:
        variable = site["variable"]
        position = site["position"]
        label = site["label"]
        if not isinstance(variable, str) or not variable:
            raise ValueError("site variable must be a non-empty string")
        if type(position) is not int or position < 0:
            raise ValueError("site position must be a non-negative integer")
        if not isinstance(label, str) or not label:
            raise ValueError("site label must be a non-empty string")
        coordinate = (variable, position)
        if coordinate in coordinates:
            raise ValueError(f"duplicate mapped site {coordinate!r}")
        if label in labels:
            raise ValueError(f"duplicate site label {label!r}")
        coordinates.add(coordinate)
        labels.add(label)
    return tuple(sites)


def _distribution_values(
    data: DataStore, distribution: IndependentDistribution
) -> np.ndarray:
    if not isinstance(distribution, IndependentDistribution):
        raise TypeError("distribution must be an IndependentDistribution")
    if distribution.semantics not in _SEMANTICS:
        raise ValueError(f"unsupported distribution semantics {distribution.semantics!r}")
    if (
        not isinstance(distribution.alphabet, tuple)
        or not distribution.alphabet
        or any(not isinstance(token, str) or not token for token in distribution.alphabet)
    ):
        raise ValueError("distribution alphabet must be a non-empty tuple of tokens")
    if len(set(distribution.alphabet)) != len(distribution.alphabet):
        raise ValueError("distribution alphabet tokens must be unique")
    if not isinstance(distribution.samples, tuple) or not distribution.samples:
        raise ValueError("distribution samples must be a non-empty tuple")
    if any(not isinstance(sample, Sample) for sample in distribution.samples):
        raise TypeError("distribution samples must contain Sample identities")
    if len(set(distribution.samples)) != len(distribution.samples):
        raise ValueError("duplicate sample identities are not allowed")
    site_map = _sites(data, distribution)
    values = _array(data, distribution.values)
    expected = (len(distribution.samples), len(site_map), len(distribution.alphabet))
    if values.shape != expected:
        raise ValueError(
            f"distribution values shape {values.shape} does not match "
            f"(samples, sites, alphabet) {expected}"
        )
    values = _float_array(values, "distribution values")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("distribution values may not contain NaN or positive infinity")
    if distribution.semantics in _PROBABILITY_DOMAINS and not np.all(
        np.isfinite(values)
    ):
        raise ValueError("probability values must be finite")
    if distribution.semantics in _LOG_DOMAINS and np.any(
        np.all(np.isneginf(values), axis=-1)
    ):
        raise ValueError("a log distribution row cannot exclude every alphabet entry")
    if distribution.semantics in _PROBABILITY_DOMAINS:
        if np.any(values < 0):
            raise ValueError("probability values must be nonnegative")
        with _checked_arithmetic():
            sums = values.sum(axis=-1, dtype=float)
        if not np.allclose(sums, 1.0, rtol=1e-7, atol=1e-9):
            raise ValueError(
                f"{distribution.semantics} rows must sum to one; got {sums.tolist()}"
            )
    if distribution.semantics == "log_probs":
        with _checked_arithmetic(), np.errstate(under="ignore"):
            sums = np.exp(values).sum(axis=-1)
        if not np.allclose(sums, 1.0, rtol=1e-7, atol=1e-9):
            raise ValueError("log_probs rows must be normalized to sum to one")
    return values


def _identity(
    distribution: IndependentDistribution,
    semantics: str,
    values,
    applied: tuple[Applied, ...],
    data: DataStore,
) -> IndependentDistribution:
    identity = key_for(
        (
            "independent-distribution/v1",
            distribution.alphabet,
            distribution.sites,
            distribution.samples,
            values.identity,
            semantics,
            distribution.conditioning,
            applied,
            distribution.ancillary_evidence,
        )
    )
    result = IndependentDistribution(
        identity=identity,
        alphabet=distribution.alphabet,
        sites=distribution.sites,
        samples=distribution.samples,
        values=values,
        semantics=semantics,
        conditioning=distribution.conditioning,
        applied=applied,
        ancillary_evidence=distribution.ancillary_evidence,
    )
    _distribution_values(data, result)
    return result


def independent_distribution(
    data: DataStore,
    *,
    alphabet: tuple[str, ...],
    sites: Mapping,
    samples: tuple,
    values,
    semantics: str,
    conditioning,
    applied: tuple[Applied, ...] = (),
    ancillary_evidence: tuple = (),
) -> IndependentDistribution:
    """Store and validate a distribution without converting its semantics."""

    if semantics not in _SEMANTICS:
        raise ValueError(f"unsupported distribution semantics {semantics!r}")
    values_reference = data.put(np.asarray(values), _ARRAYS)
    sites_reference = data.put(dict(sites), _JSON)
    provisional = IndependentDistribution(
        identity=key_for(("unvalidated-distribution", values_reference)),
        alphabet=tuple(alphabet),
        sites=sites_reference,
        samples=tuple(samples),
        values=values_reference,
        semantics=semantics,
        conditioning=conditioning,
        applied=tuple(applied),
        ancillary_evidence=tuple(ancillary_evidence),
    )
    _distribution_values(data, provisional)
    return _identity(provisional, semantics, values_reference, tuple(applied), data)


def _check_unique_transform(distribution: IndependentDistribution, operation_name: str):
    if any(item.operation == operation_name for item in distribution.applied):
        raise ValueError(
            f"distribution already records a {operation_name!r} transform; "
            "compose repeated intent into one explicit request"
        )


def _transform_parameters(
    data: DataStore,
    distribution: IndependentDistribution,
    operation_name: str,
    parameter_reference,
) -> tuple[IndependentDistribution, Applied]:
    if not isinstance(parameter_reference, type(distribution.values)):
        raise TypeError("transform parameters must be a Blob")
    parameter_key = parameter_reference.identity
    transform_id = key_for(
        ("distribution-transform/v2", operation_name, distribution.identity, parameter_key)
    )
    applied = Applied(
        request=transform_id,
        operation=operation_name,
        parameters=parameter_key,
        decoding_boundary=_boundary_text(operation_name),
    )
    dependencies = (
        Dependency("source-distribution", distribution.identity),
        Dependency("transform-parameters", parameter_key),
        Dependency("transform-request", transform_id),
    )
    conditioning = manifest(distribution.conditioning.dependencies + dependencies)
    return (
        IndependentDistribution(
            identity=distribution.identity,
            alphabet=distribution.alphabet,
            sites=distribution.sites,
            samples=distribution.samples,
            values=distribution.values,
            semantics=distribution.semantics,
            conditioning=conditioning,
            applied=distribution.applied,
            ancillary_evidence=distribution.ancillary_evidence,
        ),
        applied,
    )


def _boundary_text(operation_name: str) -> str:
    if operation_name == "bias":
        return "score-domain bias; probability inputs cross via log once"
    if operation_name == "ignore":
        return "mask before decoding; excluded log entries become -inf and probabilities zero"
    if operation_name == "temperature":
        return "score-domain divide-by-temperature; probability inputs cross via log once"
    return "unspecified"


def transform_distribution(
    distribution: IndependentDistribution,
    *,
    operation_name: str,
    parameter_reference,
    data: DataStore,
) -> IndependentDistribution:
    """Apply one audited transform to one independent distribution.

    ``parameter_reference`` is an ArrayCodec Blob for bias/ignore and a
    JSONCodec Blob containing a positive finite scalar for temperature.
    """

    data = _store_data(data)
    values = _distribution_values(data, distribution)
    name = str(operation_name).strip().lower()
    if name not in {"bias", "ignore", "temperature"}:
        raise ValueError(
            f"unsupported independent transform {operation_name!r}; "
            "autoregressive/joint transforms are not independent-likelihood operations"
        )
    _check_unique_transform(distribution, name)

    staged, applied = _transform_parameters(
        data, distribution, name, parameter_reference
    )
    if name == "ignore":
        mask = _array(data, parameter_reference)
        if mask.dtype != np.bool_:
            raise TypeError("ignore mask must be boolean")
        try:
            mask = np.broadcast_to(mask, values.shape).copy()
        except ValueError as exc:
            raise ValueError(
                f"ignore mask shape {mask.shape} does not broadcast to {values.shape}"
            ) from exc
        if np.any(~mask.any(axis=-1)):
            raise ValueError("ignore mask must retain at least one alphabet entry per site")
        if distribution.semantics in _LOG_DOMAINS:
            transformed = np.where(mask, values, -np.inf)
            if np.any(np.all(np.isneginf(transformed), axis=-1)):
                raise ValueError("ignore mask leaves a row with zero effective mass")
            if distribution.semantics == "log_probs":
                transformed = _normalize_log_array(transformed)
        else:
            transformed = np.where(mask, values, 0.0)
            with _checked_arithmetic():
                sums = transformed.sum(axis=-1, keepdims=True)
                if np.any(sums <= 0):
                    raise ValueError("ignore mask leaves a row with zero effective mass")
                transformed = transformed / sums
        semantics = distribution.semantics
    else:
        if distribution.semantics == "empirical":
            raise ValueError(
                "empirical distributions require an explicit conversion before "
                "bias or temperature; their denominator semantics are not inferred"
            )
        if name == "bias":
            raw = _array(data, parameter_reference)
            raw = _float_array(raw, "bias")
            try:
                raw = np.broadcast_to(raw, values.shape)
            except ValueError as exc:
                raise ValueError(
                    f"bias shape {raw.shape} does not broadcast to {values.shape}"
                ) from exc
            if not np.all(np.isfinite(raw)):
                raise ValueError("bias must contain only finite values")
            if distribution.semantics in _PROBABILITY_DOMAINS:
                with np.errstate(divide="ignore"):
                    scores = np.log(values)
            else:
                scores = values
            # Remove each operand's arbitrary row constant before addition.
            # Excluded entries must not set the bias reference level.
            raw = _center_log_array(np.where(np.isfinite(scores), raw, -np.inf))
            scores = _center_log_array(scores)
            with _checked_arithmetic():
                transformed = scores + raw
        else:
            payload = data.get(parameter_reference, _JSON)
            if (
                not isinstance(payload, Mapping)
                or set(payload) != {"temperature"}
                or isinstance(payload["temperature"], bool)
                or not isinstance(payload["temperature"], (int, float))
            ):
                raise ValueError(
                    "temperature parameters must be JSON with exactly one numeric "
                    "temperature field"
                )
            temperature = float(payload["temperature"])
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and greater than zero")
            if distribution.semantics in _PROBABILITY_DOMAINS:
                with np.errstate(divide="ignore"):
                    scores = np.log(values)
            else:
                scores = values
            with _checked_arithmetic():
                transformed = scores / temperature

        if distribution.semantics == "logits":
            semantics = "logits"
        else:
            semantics = "log_probs"
            transformed = _normalize_log_array(transformed)

    reference = data.put(transformed, _ARRAYS)
    return _identity(
        staged,
        semantics,
        reference,
        distribution.applied + (applied,),
        data,
    )


def bias_distribution(
    distribution: IndependentDistribution,
    bias,
    data: DataStore,
) -> IndependentDistribution:
    if not isinstance(bias, np.ndarray):
        bias = np.asarray(bias)
    return transform_distribution(
        distribution,
        operation_name="bias",
        parameter_reference=_store_data(data).put(bias, _ARRAYS),
        data=data,
    )


def ignore_distribution(
    distribution: IndependentDistribution,
    mask,
    data: DataStore,
) -> IndependentDistribution:
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        raise TypeError("ignore mask must be boolean")
    return transform_distribution(
        distribution,
        operation_name="ignore",
        parameter_reference=_store_data(data).put(mask, _ARRAYS),
        data=data,
    )


def temperature_distribution(
    distribution: IndependentDistribution,
    temperature: float,
    data: DataStore,
) -> IndependentDistribution:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be numeric")
    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    reference = _store_data(data).put({"temperature": value}, _JSON)
    return transform_distribution(
        distribution,
        operation_name="temperature",
        parameter_reference=reference,
        data=data,
    )


def _center_log_array(values: np.ndarray) -> np.ndarray:
    """Remove row constants before combining scores, retaining log support."""
    array = _float_array(values, "log distribution")
    if np.any(np.isnan(array)) or np.any(np.isposinf(array)) or np.any(
        np.all(np.isneginf(array), axis=-1)
    ):
        raise ValueError("log distribution contains invalid or fully excluded rows")
    with _checked_arithmetic():
        maximum = np.max(array, axis=-1, keepdims=True)
        shifted = array - maximum
    if np.any(np.isfinite(array) & ~np.isfinite(shifted)):
        raise ValueError("log centering changed finite support")
    return shifted


def _normalize_log_array(values: np.ndarray) -> np.ndarray:
    shifted = _center_log_array(values)
    with _checked_arithmetic():
        # Underflow here only omits negligible partition-function terms; the
        # returned log values retain every finite input's support.
        with np.errstate(under="ignore"):
            mass = np.exp(shifted).sum(axis=-1, keepdims=True)
        result = shifted - np.log(mass)
    if np.any(np.isfinite(shifted) & ~np.isfinite(result)):
        raise ValueError("log normalization changed finite support")
    return result


def distribution_probabilities(
    distribution: IndependentDistribution, data: DataStore
) -> np.ndarray:
    """Decode one distribution to probabilities at an explicit boundary."""

    values = _distribution_values(_store_data(data), distribution)
    if distribution.semantics in _PROBABILITY_DOMAINS:
        with _checked_arithmetic():
            probabilities = values / values.sum(axis=-1, keepdims=True)
        if np.any((values > 0) != (probabilities > 0)):
            raise ValueError("probability normalization changed finite support")
        return probabilities
    normalized = _normalize_log_array(values)
    with _checked_arithmetic(), np.errstate(under="ignore"):
        probabilities = np.exp(normalized)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
    if np.any(np.isfinite(values) & (probabilities == 0)):
        raise ValueError("probability decoding underflow changed finite support")
    return probabilities


def _validate_fuse(
    request: FuseRequest, data: DataStore, *, probability: bool
) -> tuple[tuple[np.ndarray, ...], tuple]:
    if not isinstance(request, FuseRequest):
        raise TypeError("request must be a FuseRequest")
    if not request.distributions:
        raise ValueError("fusion requires at least one distribution")
    if len(request.weights) != len(request.distributions):
        raise ValueError("fusion requires exactly one weight per distribution")
    weights = tuple(request.weights)
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) < 0
        for weight in weights
    ):
        raise ValueError("fusion weights must be finite and nonnegative")
    largest = max(map(float, weights))
    if largest <= 0:
        raise ValueError("fusion weights must have positive sum")
    scaled = tuple(float(weight) / largest for weight in weights)
    total = math.fsum(scaled)
    normalized_weights = tuple(weight / total for weight in scaled)
    if any(
        weight > 0 and normalized == 0
        for weight, normalized in zip(weights, normalized_weights, strict=True)
    ):
        raise ValueError("fusion weight normalization underflow changed source support")

    first = request.distributions[0]
    values = []
    for distribution in request.distributions:
        if (
            distribution.alphabet != first.alphabet
            or distribution.sites != first.sites
            or distribution.samples != first.samples
        ):
            raise ValueError(
                "fused distributions must share alphabet, sites, and sample identities"
            )
        if probability:
            if distribution.semantics not in _SEMANTICS:
                raise ValueError("unsupported fusion input semantics")
            values.append(distribution_probabilities(distribution, data))
        else:
            if distribution.semantics not in _LOG_DOMAINS:
                raise ValueError(
                    "log fusion requires logits or log_probs inputs; "
                    "probability fusion is a separate operation"
                )
            values.append(
                np.asarray(_distribution_values(data, distribution), dtype=float)
            )
    if not probability and any(
        distribution.semantics != first.semantics
        for distribution in request.distributions
    ):
        raise ValueError("log fusion inputs must use the same log semantics")
    return tuple(values), normalized_weights


def _fused_distribution(
    data: DataStore,
    distributions: tuple[IndependentDistribution, ...],
    values: np.ndarray,
    semantics: str,
    weights: tuple[float, ...],
    operation_name: str,
) -> IndependentDistribution:
    reference = data.put(values, _ARRAYS)
    parameters = data.put(
        {
            "weights": list(weights),
            "source_count": len(distributions),
        },
        _JSON,
    )
    dependencies = [
        Dependency("fuse-input", distribution.identity)
        for distribution in distributions
    ]
    dependencies.extend(
        Dependency("fuse-conditioning", dependency.identity)
        for distribution in distributions
        for dependency in distribution.conditioning.dependencies
    )
    dependencies.append(Dependency("fuse-parameters", parameters.identity))
    conditioning = manifest(tuple(dependencies))
    request_key = key_for(
        (
            operation_name,
            tuple(distribution.identity for distribution in distributions),
            weights,
        )
    )
    applied = Applied(
        request=request_key,
        operation=operation_name,
        parameters=parameters.identity,
        decoding_boundary=(
            "weighted probability mixture after explicit per-input decoding"
            if operation_name == "probability_mixture"
            else "weighted score fusion before any sampling decode"
        ),
    )
    source_applied = tuple(
        item
        for distribution in distributions
        for item in distribution.applied
    )
    first = distributions[0]
    result = IndependentDistribution(
        identity=key_for(
            (
                "fused-independent-distribution/v1",
                tuple(distribution.identity for distribution in distributions),
                weights,
                reference,
                semantics,
                conditioning,
                source_applied,
                applied,
                first.ancillary_evidence,
            )
        ),
        alphabet=first.alphabet,
        sites=first.sites,
        samples=first.samples,
        values=reference,
        semantics=semantics,
        conditioning=conditioning,
        applied=source_applied + (applied,),
        ancillary_evidence=first.ancillary_evidence,
    )
    _distribution_values(data, result)
    return result


def mix_distributions(
    request: FuseRequest, data: DataStore
) -> IndependentDistribution:
    """Form a weighted sampling-probability mixture."""

    data = _store_data(data)
    decoded, weights = _validate_fuse(request, data, probability=True)
    with _checked_arithmetic():
        mixed = sum(
            weight * probabilities
            for weight, probabilities in zip(weights, decoded, strict=True)
            if weight > 0
        )
        mixed = np.asarray(mixed, dtype=float)
        sums = mixed.sum(axis=-1, keepdims=True)
        if np.any(sums <= 0):
            raise ValueError("probability mixture has a row with zero effective mass")
        mixed /= sums
    return _fused_distribution(
        data,
        tuple(request.distributions),
        mixed,
        "sampling_probs",
        weights,
        "probability_mixture",
    )


def fuse_log_distributions(
    request: FuseRequest, data: DataStore
) -> IndependentDistribution:
    """Fuse logits or log-probabilities without treating them as probabilities."""

    data = _store_data(data)
    decoded, weights = _validate_fuse(request, data, probability=False)
    support = np.logical_and.reduce([
        np.isfinite(values)
        for weight, values in zip(weights, decoded, strict=True)
        if weight > 0
    ])
    if np.any(~support.any(axis=-1)):
        raise ValueError("log fusion excludes every alphabet entry for at least one site")
    with _checked_arithmetic():
        fused = sum(
            weight * _center_log_array(np.where(support, values, -np.inf))
            for weight, values in zip(weights, decoded, strict=True)
            if weight > 0
        )
    semantics = request.distributions[0].semantics
    if semantics == "log_probs":
        fused = _normalize_log_array(np.asarray(fused, dtype=float))
    else:
        fused = np.asarray(fused, dtype=float)
        if np.any(np.isnan(fused)):
            raise ValueError("log fusion produced NaN")
        if np.any(np.all(np.isneginf(fused), axis=-1)):
            raise ValueError(
                "log fusion excludes every alphabet entry for at least one site"
            )
    return _fused_distribution(
        data,
        tuple(request.distributions),
        fused,
        semantics,
        weights,
        "log_fusion",
    )


def _assignments_for_row(
    context, sites: tuple[dict, ...], row
) -> tuple[Assignment, ...]:
    assignments = {
        assignment.variable: assignment for assignment in context.working.assignments
    }
    tokens_by_variable: dict[str, list[str]] = {}
    for site, token in zip(sites, row, strict=True):
        variable = site["variable"]
        assignment = assignments.get(variable)
        if assignment is None or not isinstance(assignment.value, Polymer):
            raise ValueError(
                f"sampling variable {variable!r} must be a polymer assignment"
            )
        tokens = tokens_by_variable.setdefault(
            variable, list(assignment.value.tokens)
        )
        position = site["position"]
        if position >= len(tokens):
            raise ValueError(
                f"sampling position {position} is outside variable {variable!r}"
            )
        tokens[position] = str(token)
    return tuple(
        Assignment(
            variable,
            Polymer(
                kind=assignments[variable].value.kind,
                tokens=tuple(tokens),
                modifications=assignments[variable].value.modifications,
            ),
        )
        for variable, tokens in tokens_by_variable.items()
    )


def sample_independent(
    request: SampleIndependentRequest,
    data: DataStore,
    rng: np.random.Generator,
) -> Proposals:
    """Draw complete independent proposals using a caller-owned RNG.

    The request's system domains are enforced by ``System.edit``. Fixed tokens
    and polymer modifications therefore retain their canonical validation.
    """

    if not isinstance(request, SampleIndependentRequest):
        raise TypeError("request must be a SampleIndependentRequest")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    if type(request.count) is not int or request.count <= 0:
        raise ValueError("sample count must be a positive integer")
    data = _store_data(data)
    context = request.context
    if not isinstance(context.sampling, SamplingPolicy):
        raise TypeError("context.sampling must be a SamplingPolicy")
    if context.working.system != context.system.identity:
        raise ValueError("working candidate belongs to a different system")
    if any(parent.system != context.system.identity for parent in context.parents):
        raise ValueError("all parents must belong to the sampled system")
    sites = _sites(data, request.distribution)
    probabilities = distribution_probabilities(request.distribution, data)
    if request.context.sample in request.distribution.samples:
        sample_index = request.distribution.samples.index(request.context.sample)
    elif len(request.distribution.samples) == 1:
        sample_index = 0
    else:
        raise ValueError(
            "multi-sample distributions require context.sample to identify one "
            "distribution sample explicitly"
        )
    probabilities = probabilities[sample_index]
    items = []
    for _ in range(request.count):
        rows = []
        for probability_row in probabilities:
            sums = probability_row.sum(axis=-1)
            if np.any(probability_row < 0) or not np.allclose(
                sums, 1.0, rtol=1e-7, atol=1e-9
            ):
                raise ValueError("sampling probabilities must be nonnegative and sum to one")
            index = int(rng.choice(len(request.distribution.alphabet), p=probability_row))
            rows.append(request.distribution.alphabet[index])
        replacements = _assignments_for_row(context, sites, tuple(rows))
        candidate: Candidate = context.system.edit(
            context.working,
            replacements,
            data=data,
        )
        used_evidence = tuple(item.evidence.identity for item in context.evidence)
        proposal_key = key_for(
            (
                "independent-proposal/v1",
                request.distribution.identity,
                context.system.identity,
                context.working.identity,
                tuple(parent.identity for parent in context.parents),
                candidate.identity,
                used_evidence,
                context.sampling.identity,
                context.sample,
                len(items),
            )
        )
        items.append(
            Proposal(
                identity=proposal_key,
                parents=tuple(parent.identity for parent in context.parents),
                base=context.working.identity,
                candidate=candidate,
                used_evidence=used_evidence,
                protocol=context.sampling.identity,
            )
        )
    return Proposals(tuple(items))


class SamplingOperationRegistry(OperationOwner):
    """Typed operation owner for transforms, independent draws, and fusion."""

    @classmethod
    def build_operation(
        cls, name: str, *, alias: str, options: dict, context
    ):
        data = _store_data(context)
        normalized = str(name).strip().lower()
        # Bind shipped direct dependencies, not arbitrary dynamic replacements
        # of imported functions or classes in an already-running interpreter.
        directory = Path(__file__).parent
        implementation_identity = key_for(
            (
                "sampling-operation/v2",
                tuple(
                    (name, (directory / name).read_bytes())
                    for name in ("sampling.py", "types.py", "store.py", "operations.py")
                ),
                ("numpy", np.__version__),
            )
        )
        if normalized == "sample_independent":
            return operation(
                alias,
                SampleIndependentRequest,
                Proposals,
                lambda request: sample_independent(
                    request,
                    data,
                    _rng_from_request(request),
                ),
                version="sampling/independent/v2",
                options=options,
                model_identity=implementation_identity,
            )
        if normalized in {"mix", "probability_mixture"}:
            return operation(
                alias,
                FuseRequest,
                IndependentDistribution,
                lambda request: mix_distributions(request, data),
                version="sampling/probability-mixture/v1",
                options=options,
                model_identity=implementation_identity,
            )
        if normalized in {"fuse", "log_fusion"}:
            return operation(
                alias,
                FuseRequest,
                IndependentDistribution,
                lambda request: fuse_log_distributions(request, data),
                version="sampling/log-fusion/v1",
                options=options,
                model_identity=implementation_identity,
            )
        transforms = {
            "bias": (
                BiasDistributionRequest,
                lambda request: transform_distribution(
                    request.distribution,
                    operation_name="bias",
                    parameter_reference=request.bias,
                    data=data,
                ),
                "sampling/bias/v1",
            ),
            "ignore": (
                IgnoreDistributionRequest,
                lambda request: transform_distribution(
                    request.distribution,
                    operation_name="ignore",
                    parameter_reference=request.mask,
                    data=data,
                ),
                "sampling/ignore/v1",
            ),
            "temperature": (
                TemperatureDistributionRequest,
                lambda request: transform_distribution(
                    request.distribution,
                    operation_name="temperature",
                    parameter_reference=request.temperature,
                    data=data,
                ),
                "sampling/temperature/v1",
            ),
        }
        if normalized in transforms:
            request_type, invoke, version = transforms[normalized]
            return operation(
                alias,
                request_type,
                IndependentDistribution,
                invoke,
                version=version,
                options=options,
                model_identity=implementation_identity,
            )
        raise ValueError(f"SamplingOperationRegistry: unknown operation {name!r}")


def _rng_from_request(request: SampleIndependentRequest) -> np.random.Generator:
    seed = request.context.sample.seed
    if type(seed) is not int or seed < 0:
        raise ValueError("context.sample.seed must be a non-negative integer")
    return np.random.default_rng(seed)


__all__ = [
    "BiasDistributionRequest",
    "IgnoreDistributionRequest",
    "SamplingOperationRegistry",
    "TemperatureDistributionRequest",
    "bias_distribution",
    "distribution_probabilities",
    "fuse_log_distributions",
    "ignore_distribution",
    "independent_distribution",
    "mix_distributions",
    "sample_independent",
    "temperature_distribution",
    "transform_distribution",
]
