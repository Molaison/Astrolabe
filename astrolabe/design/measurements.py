"""Immutable measurement selection and reduction over content-addressed arrays."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .operations import OperationOwner, operation
from .store import ArrayCodec, DataStore, JSONCodec
from .types import (
    Axis,
    Dependency,
    Measurement,
    Reduction,
    Selection,
    Subject,
    key_for,
    manifest,
)


_ARRAYS = ArrayCodec()
_JSON = JSONCodec()
_REDUCERS = frozenset({"mean", "sum", "min", "max", "weighted_mean"})


@dataclass(frozen=True)
class MeasurementProjectionRequest:
    measurement: Measurement
    selections: tuple[Selection, ...] = ()


@dataclass(frozen=True)
class MeasurementReductionRequest:
    measurement: Measurement
    selections: tuple[Selection, ...]
    reduction: Reduction


def _store_data(value: Any) -> DataStore:
    if isinstance(value, DataStore):
        return value
    data = getattr(value, "data", None)
    if isinstance(data, DataStore):
        return data
    raise TypeError("context must be a DataStore or expose a DataStore as .data")


def _array(data: DataStore, reference):
    value = data.get(reference, _ARRAYS)
    if not isinstance(value, np.ndarray):
        raise TypeError("array references must decode to numpy arrays")
    return value


def _json(data: DataStore, reference):
    return data.get(reference, _JSON)


def _axis_labels(index_map: Any, *, axis_name: str) -> tuple[str, ...]:
    labels = index_map.get("labels") if isinstance(index_map, dict) else index_map
    if not isinstance(labels, (list, tuple)):
        raise ValueError(
            f"axis {axis_name!r} index_map must contain an ordered labels list"
        )
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError(f"axis {axis_name!r} labels must be non-empty strings")
    labels = tuple(labels)
    if len(set(labels)) != len(labels):
        raise ValueError(f"axis {axis_name!r} labels must be unique")
    return labels


def _subset_axis_map(
    data: DataStore,
    axis: Axis,
    mask: np.ndarray,
) -> Axis:
    source_map = _json(data, axis.index_map)
    labels = _axis_labels(source_map, axis_name=axis.name)
    selected = tuple(labels[index] for index in np.flatnonzero(mask))
    subset = {
        key: value
        for key, value in source_map.items()
        if key not in {"labels", "subject_keys", "source_index_map"}
    }
    subset["labels"] = list(selected)
    if "subject_keys" in source_map:
        subject_keys = source_map["subject_keys"]
        subset["subject_keys"] = [
            subject_keys[index] for index in np.flatnonzero(mask)
        ]
    subset["source_index_map"] = {
        "codec": axis.index_map.codec,
        "digest": axis.index_map.identity.digest,
    }
    reference = data.put(subset, _JSON)
    return Axis(axis.name, axis.kind, reference)


def _provenance(
    source: Measurement,
    *,
    role: str,
    derived_key: Any,
    extra: tuple[Dependency, ...] = (),
):
    dependencies = [
        Dependency("source-measurement", source.identity),
        Dependency(role, derived_key),
    ]
    dependencies.extend(
        Dependency(f"upstream:{dependency.role}", dependency.identity)
        for dependency in source.inputs.dependencies
    )
    dependencies.extend(extra)
    return manifest(tuple(dependencies))


class ImmutableMeasurementView:
    """A validated view whose only state is a subject and axis selections."""

    def __init__(
        self,
        measurement: Measurement,
        data: DataStore,
        *,
        selections: tuple[Selection, ...] = (),
    ) -> None:
        if not isinstance(measurement, Measurement):
            raise TypeError("measurement must be a Measurement")
        if not isinstance(data, DataStore):
            raise TypeError("data must be a DataStore")
        self._measurement = measurement
        self._data = data
        self._selections = tuple(selections)
        self._validate_base()
        for selection in self._selections:
            self._validate_selection(selection)

    @property
    def measurement(self) -> Measurement:
        return self._measurement

    @property
    def selections(self) -> tuple[Selection, ...]:
        return self._selections

    def select(self, selection: Selection) -> ImmutableMeasurementView:
        self._validate_selection(selection)
        if any(item.axis_name == selection.axis_name for item in self._selections):
            raise ValueError(
                f"axis {selection.axis_name!r} has already been selected"
            )
        return ImmutableMeasurementView(
            self._measurement,
            self._data,
            selections=self._selections + (selection,),
        )

    def between(
        self,
        left: Selection,
        right: Selection,
        *,
        left_axis: str,
        right_axis: str,
    ) -> ImmutableMeasurementView:
        if left.axis_name != left_axis or right.axis_name != right_axis:
            raise ValueError("left/right selection names must match their declared axes")
        if left_axis == right_axis:
            raise ValueError("between requires two distinct named axes")
        left_definition = self._axis(left_axis)
        right_definition = self._axis(right_axis)
        if left_definition.kind != right_definition.kind:
            raise ValueError(
                "between selections must use axes of the same semantic kind"
            )
        view = self.select(left)
        return view.select(right)

    def materialize(self) -> Measurement:
        if not self._selections:
            return self._measurement

        source = self._measurement
        values = (
            None if source.values is None else _array(self._data, source.values)
        )
        valid = (
            _array(self._data, source.valid_mask)
            if source.valid_mask is not None
            else None
        )
        axes = list(source.axes)

        for selection in self._selections:
            axis_index = self._axis_index(selection.axis_name)
            mask = self._selection_mask(selection)
            if values is not None:
                values = np.compress(mask, values, axis=axis_index)
            if valid is not None:
                valid = np.compress(mask, valid, axis=axis_index)
            axes[axis_index] = _subset_axis_map(
                self._data, axes[axis_index], mask
            )

        missing = source.missing
        if values is not None and missing != "failed":
            all_valid = values.size > 0 and (valid is None or bool(np.all(valid)))
            missing = "present" if all_valid else "partial"
            if valid is None and not all_valid:
                valid = np.ones(values.shape, dtype=bool)
        values_reference = (
            None if values is None else self._data.put(values, _ARRAYS)
        )
        valid_reference = (
            None
            if valid is None
            else self._data.put(valid.astype(bool, copy=False), _ARRAYS)
        )
        extra = tuple(
            Dependency("selection-mask", selection.mask.identity)
            for selection in self._selections
        )
        inputs = _provenance(
            source,
            role="materialized-values" if values_reference is not None else "selection",
            derived_key=(
                values_reference.identity
                if values_reference is not None
                else key_for(self._selections)
            ),
            extra=extra,
        )
        identity = key_for(
            (
                "measurement-materialize/v2",
                source.identity,
                source.subjects,
                self._selections,
                None if values_reference is None else values_reference.identity,
                None if valid_reference is None else valid_reference.identity,
                missing,
            )
        )
        return Measurement(
            identity=identity,
            quantity=source.quantity,
            definition=source.definition,
            units=source.units,
            producer=source.producer,
            subjects=source.subjects,
            axes=tuple(axes),
            values=values_reference,
            missing=missing,
            valid_mask=valid_reference,
            inputs=inputs,
        )

    def reduce(self, reduction: Reduction) -> Measurement:
        if not isinstance(reduction, Reduction):
            raise TypeError("reduction must be a Reduction")
        if reduction.method not in _REDUCERS:
            raise ValueError(
                f"unsupported reduction {reduction.method!r}; expected one of "
                f"{sorted(_REDUCERS)}"
            )
        if reduction.missing not in {"fail", "omit", "propagate"}:
            raise ValueError(
                f"unsupported missing policy {reduction.missing!r}; expected "
                "'fail', 'omit', or 'propagate'"
            )
        if len(set(reduction.axes)) != len(reduction.axes):
            raise ValueError("reduction axes must be unique")
        if any(name not in {axis.name for axis in self._measurement.axes} for name in reduction.axes):
            raise ValueError("reduction references an unknown axis")

        selected = self.materialize()
        # Failed payloads may be retained for diagnostics, never as valid data.
        if selected.missing == "failed":
            if reduction.missing == "fail":
                raise ValueError("cannot reduce failed measurement with missing='fail'")
            return self._unavailable_reduction(selected, reduction)
        if not reduction.axes:
            if selected.values is None:
                if reduction.missing == "fail":
                    raise ValueError(
                        f"cannot reduce {selected.quantity!r}: values are "
                        f"{selected.missing}"
                    )
                return selected
            values = _array(self._data, selected.values)
            valid = (
                _array(self._data, selected.valid_mask)
                if selected.valid_mask is not None
                else np.broadcast_to(np.asarray(True), values.shape)
            )
            if reduction.missing == "fail" and (
                valid.size == 0 or not bool(np.all(valid))
            ):
                raise ValueError(
                    "reduction encountered empty or invalid values with missing='fail'"
                )
            return selected
        if selected.values is None:
            if reduction.missing == "fail":
                raise ValueError(
                    f"cannot reduce {selected.quantity!r}: values are {selected.missing}"
                )
            return self._unavailable_reduction(selected, reduction)

        values = _array(self._data, selected.values)
        valid = (
            _array(self._data, selected.valid_mask)
            if selected.valid_mask is not None
            else np.broadcast_to(np.asarray(True), values.shape)
        )
        values = np.asarray(values)
        valid = np.asarray(valid)
        if values.ndim != len(selected.axes):
            raise ValueError("measurement values shape does not match its axes")
        if valid.shape != values.shape:
            raise ValueError("valid_mask shape must exactly match values")

        axis_names = tuple(axis.name for axis in selected.axes)
        reduce_indices = tuple(axis_names.index(name) for name in reduction.axes)
        other_indices = tuple(
            index for index in range(len(axis_names)) if index not in reduce_indices
        )
        permutation = other_indices + reduce_indices
        transposed_values = np.transpose(values, permutation)
        transposed_valid = np.transpose(valid, permutation)
        other_shape = tuple(values.shape[index] for index in other_indices)
        reduce_shape = tuple(values.shape[index] for index in reduce_indices)
        weights = self._reduction_weights(reduction, reduce_shape)
        result, result_valid = self._reduce_values(
            transposed_values,
            transposed_valid,
            weights,
            method=reduction.method,
            missing=reduction.missing,
            reduced_ndim=len(reduce_indices),
        )
        result = result.reshape(other_shape)
        result_valid = result_valid.reshape(other_shape)
        remaining_axes = tuple(selected.axes[index] for index in other_indices)
        result_values = self._data.put(result, _ARRAYS)
        all_valid = result_valid.size > 0 and bool(np.all(result_valid))
        mask_reference = (
            None
            if all_valid
            else self._data.put(result_valid, _ARRAYS)
        )
        parameters_key = reduction.parameters.identity
        definition = key_for(
            (
                "measurement-reduction-definition/v1",
                selected.definition,
                reduction.axes,
                reduction.method,
                parameters_key,
                reduction.missing,
            )
        )
        inputs = _provenance(
            selected,
            role="reduction-parameters",
            derived_key=parameters_key,
            extra=(
                Dependency("reduced-values", result_values.identity),
                Dependency("reduction", definition),
            ),
        )
        identity = key_for(
            (
                "measurement-reduction/v1",
                selected.identity,
                reduction,
                result_values.identity,
                None if mask_reference is None else mask_reference.identity,
            )
        )
        return Measurement(
            identity=identity,
            quantity=selected.quantity,
            definition=definition,
            units=selected.units,
            producer=selected.producer,
            subjects=selected.subjects,
            axes=remaining_axes,
            values=result_values,
            missing="present" if all_valid else "partial",
            valid_mask=mask_reference,
            inputs=inputs,
        )

    def _unavailable_reduction(
        self, selected: Measurement, reduction: Reduction
    ) -> Measurement:
        definition = key_for(
            (
                "measurement-reduction-definition/v1",
                selected.definition,
                reduction.axes,
                reduction.method,
                reduction.parameters.identity,
                reduction.missing,
            )
        )
        inputs = _provenance(
            selected,
            role="reduction-parameters",
            derived_key=reduction.parameters.identity,
            extra=(Dependency("reduction", definition),),
        )
        return Measurement(
            identity=key_for(
                ("measurement-unavailable-reduction/v1", selected.identity, reduction)
            ),
            quantity=selected.quantity,
            definition=definition,
            units=selected.units,
            producer=selected.producer,
            subjects=selected.subjects,
            axes=tuple(
                axis for axis in selected.axes if axis.name not in reduction.axes
            ),
            values=None,
            missing=selected.missing,
            valid_mask=None,
            inputs=inputs,
        )

    def _reduction_weights(
        self,
        reduction: Reduction,
        reduce_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        if reduction.method != "weighted_mean":
            return None
        parameters = _json(self._data, reduction.parameters)
        if not isinstance(parameters, Mapping) or "weights" not in parameters:
            raise ValueError("weighted_mean requires a weights parameter")
        weights = np.asarray(parameters["weights"], dtype=float)
        # Validate the original parameter before broadcasting, in NumPy's
        # bounded iterator buffers rather than a full-size boolean temporary.
        for chunk in np.nditer(
            weights, flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"],
        ):
            if not np.isfinite(chunk).all() or np.any(chunk < 0):
                raise ValueError("weighted_mean weights must be finite and nonnegative")
        try:
            weights = np.broadcast_to(weights, reduce_shape)
        except ValueError as exc:
            raise ValueError(
                f"weight shape {weights.shape} does not match reduced shape {reduce_shape}"
            ) from exc
        return weights

    @staticmethod
    def _exact_reduction(
        values: np.ndarray, valid: np.ndarray, weights: np.ndarray | None,
        *, method: str,
    ) -> tuple[int, int]:
        """Stream an exact numerator/denominator without per-element storage.

        Finite binary floats are dyadic rationals. Aligning their power-of-two
        denominators permits exact integer accumulation of values, products,
        and weights. Only two running integers, their exponents, and the last
        weight ratio are retained. Integer width depends on input precision and
        log(count), not a list of broadcast elements. Array flat iterators keep
        arbitrary axis strides and zero-stride broadcasts without copying.
        """
        total = 0
        exponent = 0
        denominator = 1 if method == "sum" else 0
        denominator_exponent = 0
        weighted = weights is not None
        integer_values = np.issubdtype(values.dtype, np.integer)
        coefficients = repeat(1) if weights is None else weights.flat
        last_weight = None
        weight_numerator, weight_exponent = 1, 0
        for value, included, weight in zip(values.flat, valid.flat, coefficients):
            if not included:
                continue
            if weighted:
                if weight == 0:
                    continue
                if weight != last_weight:
                    weight_numerator, weight_denominator = weight.as_integer_ratio()
                    weight_exponent = weight_denominator.bit_length() - 1
                    last_weight = weight
                if weight_exponent > denominator_exponent:
                    denominator <<= weight_exponent - denominator_exponent
                    denominator_exponent = weight_exponent
                denominator += weight_numerator << (
                    denominator_exponent - weight_exponent
                )
            elif method == "mean":
                denominator += 1
            if not value:
                continue
            if integer_values:
                numerator, value_denominator = int(value), 1
            else:
                numerator, value_denominator = value.as_integer_ratio()
            value_exponent = value_denominator.bit_length() - 1 + weight_exponent
            if value_exponent > exponent:
                total <<= value_exponent - exponent
                exponent = value_exponent
            total += (numerator * weight_numerator) << (exponent - value_exponent)
        if exponent >= denominator_exponent:
            return total, denominator << (exponent - denominator_exponent)
        return total << (denominator_exponent - exponent), denominator

    @staticmethod
    def _rounded_ratio(numerator: int, denominator: int) -> float:
        # Python integer true division rounds the exact ratio to binary64;
        # neither the sum nor an intermediate product needs to fit a float.
        try:
            result = numerator / denominator
        except OverflowError as exc:
            raise ValueError(
                "reduction result cannot be represented as finite float64"
            ) from exc
        if not np.isfinite(result):
            raise ValueError("reduction result cannot be represented as finite float64")
        if numerator and result == 0:
            raise ValueError("reduction result underflows float64")
        return result

    @staticmethod
    def _reduce_values(
        values: np.ndarray,
        valid: np.ndarray,
        weights: np.ndarray | None,
        *,
        method: str,
        missing: str,
        reduced_ndim: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if missing == "fail" and (valid.size == 0 or not np.all(valid)):
            raise ValueError(
                "reduction encountered empty or invalid values with missing='fail'"
            )
        reduce_axes = tuple(range(values.ndim - reduced_ndim, values.ndim))
        other_shape = values.shape[:values.ndim - reduced_ndim]
        group_valid = np.asarray(np.any(valid, axis=reduce_axes))
        if missing == "propagate":
            group_valid &= np.all(valid, axis=reduce_axes)

        if method in {"sum", "mean", "weighted_mean"}:
            if method == "weighted_mean" and weights is None:
                raise AssertionError("weighted_mean requires weights")
            result = np.full(group_valid.shape, np.nan, dtype=float)
            for index in np.ndindex(other_shape):
                if not group_valid[index]:
                    continue
                numerator, denominator = ImmutableMeasurementView._exact_reduction(
                    values[index], valid[index],
                    weights if method == "weighted_mean" else None, method=method,
                )
                if denominator == 0:
                    group_valid[index] = False
                    continue
                result[index] = ImmutableMeasurementView._rounded_ratio(
                    numerator, denominator
                )
            if missing == "fail" and not np.all(group_valid):
                raise ValueError("reduction has no positive weight with missing='fail'")
            return result, group_valid

        integer_values = np.issubdtype(values.dtype, np.integer)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            if method == "min":
                initial = np.iinfo(values.dtype).max if integer_values else np.inf
                result = np.min(
                    values, axis=reduce_axes, where=valid, initial=initial
                )
            elif method == "max":
                initial = np.iinfo(values.dtype).min if integer_values else -np.inf
                result = np.max(
                    values, axis=reduce_axes, where=valid, initial=initial
                )
            else:
                raise AssertionError(method)

            result = np.where(group_valid, np.asarray(result, dtype=np.longdouble), np.nan)
            unrepresentable = ~np.isfinite(result) | (
                np.abs(result) > np.finfo(float).max
            )
            if np.any(group_valid & unrepresentable):
                raise ValueError("reduction result cannot be represented as finite float64")
            converted = np.asarray(result, dtype=float)
            if np.any(group_valid & (result != 0) & (converted == 0)):
                raise ValueError("reduction result underflows float64")
        return converted, group_valid

    def _validate_base(self) -> None:
        measurement = self._measurement
        if not measurement.subjects:
            raise ValueError("measurement must contain at least one subject")
        if len(set(measurement.subjects)) != len(measurement.subjects):
            raise ValueError("measurement subjects must be unique")
        axis_names = [axis.name for axis in measurement.axes]
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("measurement axis names must be unique")
        if measurement.missing not in {"present", "partial", "unavailable", "failed"}:
            raise ValueError("invalid measurement missing status")
        if measurement.values is None and measurement.valid_mask is not None:
            raise ValueError("a measurement without values cannot have a valid mask")
        if measurement.values is None and measurement.missing == "present":
            raise ValueError("present measurements require values")
        if measurement.values is None and measurement.missing == "partial":
            raise ValueError("partial measurements require values")
        if measurement.values is not None and measurement.missing == "unavailable":
            raise ValueError("unavailable measurements cannot contain values")
        values = (
            None if measurement.values is None else _array(self._data, measurement.values)
        )
        if values is not None:
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise TypeError("measurement values must be real numeric")
            if values.ndim != len(measurement.axes):
                raise ValueError(
                    "measurement values must have exactly one dimension per named axis"
                )
        for index, axis in enumerate(measurement.axes):
            index_map = _json(self._data, axis.index_map)
            labels = _axis_labels(index_map, axis_name=axis.name)
            if values is not None and values.shape[index] != len(labels):
                raise ValueError(
                    f"axis {axis.name!r} map has {len(labels)} labels but array "
                    f"dimension is {values.shape[index]}"
                )
            self._validate_subject_binding(axis, index_map)
        if values is None:
            return
        if measurement.missing == "partial" and measurement.valid_mask is None:
            raise ValueError("partial measurements require a valid_mask")
        if measurement.valid_mask is not None:
            valid = _array(self._data, measurement.valid_mask)
            if valid.shape != values.shape or valid.dtype != np.bool_:
                raise ValueError(
                    "valid_mask must be a boolean array with the same shape as values"
                )
            if measurement.missing == "partial" and valid.size and bool(np.all(valid)):
                raise ValueError("partial measurements require at least one invalid value")
        else:
            valid = np.broadcast_to(np.asarray(True), values.shape)
        for chunk, mask_chunk in np.nditer(
            (values, valid), flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=[["readonly"], ["readonly"]],
        ):
            if np.any(~np.isfinite(chunk) & mask_chunk):
                raise ValueError("non-finite measurement values must be marked invalid")
        if measurement.missing == "present" and not bool(np.all(valid)):
            raise ValueError("present measurements cannot contain invalid values")
        if measurement.missing == "present" and values.size == 0:
            raise ValueError("present measurements require at least one value")

    def _axis(self, name: str) -> Axis:
        for axis in self._measurement.axes:
            if axis.name == name:
                return axis
        raise ValueError(f"unknown measurement axis {name!r}")

    def _axis_index(self, name: str) -> int:
        return next(
            index
            for index, axis in enumerate(self._measurement.axes)
            if axis.name == name
        )

    def _validate_selection(self, selection: Selection) -> None:
        if not isinstance(selection, Selection):
            raise TypeError("selection must be a Selection")
        if selection.subject not in self._measurement.subjects:
            raise ValueError("selection subject is not present in the measurement")
        axis = self._axis(selection.axis_name)
        if selection.index_map != axis.index_map:
            raise ValueError(
                f"selection for axis {axis.name!r} must carry that axis's exact index_map"
            )
        mask = _array(self._data, selection.mask)
        if mask.dtype != np.bool_ or mask.ndim != 1:
            raise ValueError("selection mask must be a one-dimensional boolean array")
        labels = _axis_labels(
            _json(self._data, axis.index_map), axis_name=axis.name
        )
        if mask.shape[0] != len(labels):
            raise ValueError(
                f"selection mask length {mask.shape[0]} does not match axis "
                f"label count {len(labels)}"
            )
        index_map = _json(self._data, axis.index_map)
        subject_keys = index_map.get("subject_keys")
        subject_key = index_map.get("subject_key")
        selected_subject = key_for(selection.subject).digest
        selected_indices = np.flatnonzero(mask)
        if subject_keys is not None:
            mismatched = [
                index
                for index in selected_indices
                if subject_keys[index] != selected_subject
            ]
            if mismatched:
                raise ValueError(
                    f"selection for {axis.name!r} includes labels bound to another subject"
                )
        elif subject_key is not None:
            if selected_indices.size and subject_key != selected_subject:
                raise ValueError(
                    f"selection for {axis.name!r} targets another subject"
                )
        elif len(self._measurement.subjects) > 1:
            raise ValueError(
                f"axis {axis.name!r} has no explicit subject binding"
            )

    def _selection_mask(self, selection: Selection) -> np.ndarray:
        return _array(self._data, selection.mask)

    def _validate_subject_binding(self, axis: Axis, index_map: Mapping) -> None:
        known = {key_for(subject).digest for subject in self._measurement.subjects}
        subject_keys = index_map.get("subject_keys")
        subject_key = index_map.get("subject_key")
        if subject_keys is not None:
            if (
                not isinstance(subject_keys, list)
                or len(subject_keys) != len(index_map["labels"])
                or any(not isinstance(item, str) for item in subject_keys)
            ):
                raise ValueError(
                    f"axis {axis.name!r} subject_keys must list one digest per label"
                )
            unknown = set(subject_keys) - known
            if unknown:
                raise ValueError(
                    f"axis {axis.name!r} references unknown subject digests"
                )
        if subject_key is not None:
            if not isinstance(subject_key, str) or subject_key not in known:
                raise ValueError(
                    f"axis {axis.name!r} subject_key is not a listed subject"
                )
            if axis.kind != "subject":
                raise ValueError(
                    f"axis {axis.name!r} may use subject_key only with kind='subject'"
                )
            if subject_keys is not None and any(
                item != subject_key for item in subject_keys
            ):
                raise ValueError(
                    f"axis {axis.name!r} subject_key and subject_keys must agree"
                )
        if len(self._measurement.subjects) > 1 and (
            subject_keys is None and subject_key is None
        ):
            raise ValueError(
                f"multi-subject measurement axis {axis.name!r} requires explicit "
                "subject_keys or subject_key"
            )


def measurement_view(
    measurement: Measurement, data: DataStore
) -> ImmutableMeasurementView:
    """Create a validated immutable measurement view."""

    return ImmutableMeasurementView(measurement, data)


def materialize_measurement(
    measurement: Measurement,
    data: DataStore,
    selections: tuple[Selection, ...] = (),
) -> Measurement:
    view = measurement_view(measurement, data)
    for selection in tuple(selections):
        view = view.select(selection)
    return view.materialize()


def reduce_measurement(
    measurement: Measurement,
    data: DataStore,
    selections: tuple[Selection, ...],
    reduction: Reduction,
) -> Measurement:
    view = measurement_view(measurement, data)
    for selection in tuple(selections):
        view = view.select(selection)
    return view.reduce(reduction)


class MeasurementOperationRegistry(OperationOwner):
    """Typed operations owned by the measurement module."""

    @classmethod
    def build_operation(
        cls, name: str, *, alias: str, options: dict, context
    ):
        data = _store_data(context)
        normalized = str(name).strip().lower()
        # The closure captures the store; content-addressed input blobs supply
        # its scientific data. Bind the actual implementation source as well.
        implementation_identity = key_for(("measurement-operation/v1", Path(__file__).read_bytes()))
        if normalized in {"measurement", "materialize"}:
            return operation(
                alias,
                MeasurementProjectionRequest,
                Measurement,
                lambda request: materialize_measurement(
                    request.measurement, data, request.selections
                ),
                version="measurements/materialize/v1",
                options=options,
                model_identity=implementation_identity,
            )
        if normalized in {"reduction", "reduce"}:
            return operation(
                alias,
                MeasurementReductionRequest,
                Measurement,
                lambda request: reduce_measurement(
                    request.measurement,
                    data,
                    request.selections,
                    request.reduction,
                ),
                version="measurements/reduce/v1",
                options=options,
                model_identity=implementation_identity,
            )
        raise ValueError(f"MeasurementOperationRegistry: unknown operation {name!r}")


__all__ = [
    "ImmutableMeasurementView",
    "MeasurementOperationRegistry",
    "MeasurementProjectionRequest",
    "MeasurementReductionRequest",
    "materialize_measurement",
    "measurement_view",
    "reduce_measurement",
]
