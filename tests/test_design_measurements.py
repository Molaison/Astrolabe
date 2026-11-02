from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from astrolabe.design.measurements import (
    MeasurementOperationRegistry,
    MeasurementProjectionRequest,
    MeasurementReductionRequest,
    materialize_measurement,
    measurement_view,
    reduce_measurement,
)
from astrolabe.design.store import ArrayCodec, DataStore, JSONCodec
from astrolabe.design.types import (
    Axis,
    Dependency,
    Measurement,
    Reduction,
    Sample,
    Selection,
    Subject,
    key_for,
    manifest,
)


ARRAYS = ArrayCodec()
JSON = JSONCodec()


def _subject(name: str = "candidate-1") -> Subject:
    return Subject(
        candidate=key_for(name),
        state="state-a",
        assembly="assembly-a",
        occurrences=(key_for(f"occurrence/{name}"),),
        artifact=key_for(f"artifact/{name}"),
        sample=Sample(key_for("protocol"), "0", 17, "sample-0"),
    )


def _axis_map(
    data: DataStore,
    labels: tuple[str, ...],
    subject_keys: tuple[str, ...] | None = None,
) -> object:
    payload = {"labels": list(labels)}
    if subject_keys is not None:
        payload["subject_keys"] = list(subject_keys)
    return data.put(payload, JSON)


def _selection(
    data: DataStore,
    subject: Subject,
    axis: Axis,
    mask: tuple[bool, ...],
) -> Selection:
    return Selection(
        subject=subject,
        axis_name=axis.name,
        index_map=axis.index_map,
        mask=data.put(np.asarray(mask, dtype=bool), ARRAYS),
    )


def _measurement(
    data: DataStore,
    values,
    *,
    subjects: tuple[Subject, ...] | None = None,
    valid_mask=None,
    missing: str = "present",
) -> tuple[Measurement, Axis, Axis]:
    subjects = subjects or (_subject(),)
    rows = Axis("row_site", "site", _axis_map(data, ("R0", "R1", "R2")))
    columns = Axis("column_site", "site", _axis_map(data, ("C0", "C1")))
    source = key_for("source/pae")
    measurement = Measurement(
        identity=key_for(("measurement", subjects, values.tolist())),
        quantity="test/pae",
        definition=key_for("definition/pae"),
        units="angstrom",
        producer=key_for("producer/pae"),
        subjects=tuple(subjects),
        axes=(rows, columns),
        values=data.put(np.asarray(values, dtype=float), ARRAYS),
        missing=missing,
        valid_mask=(
            None
            if valid_mask is None
            else data.put(np.asarray(valid_mask, dtype=bool), ARRAYS)
        ),
        inputs=manifest((Dependency("source", source),)),
    )
    return measurement, rows, columns


def _parameters(data: DataStore, payload) -> object:
    return data.put(payload, JSON)


def test_select_validates_subject_exact_axis_map_and_preserves_subset_order():
    data = DataStore()
    measurement, rows, _ = _measurement(data, np.arange(6, dtype=float).reshape(3, 2))
    subject = measurement.subjects[0]

    wrong_map = Axis("row_site", "site", _axis_map(data, ("R2", "R1", "R0")))
    with pytest.raises(ValueError, match="exact index_map"):
        measurement_view(measurement, data).select(
            _selection(data, subject, wrong_map, (True, False, True))
        )
    with pytest.raises(ValueError, match="subject"):
        measurement_view(measurement, data).select(
            _selection(
                data,
                _subject("candidate-2"),
                rows,
                (True, False, True),
            )
        )

    selected = measurement_view(measurement, data).select(
        _selection(data, subject, rows, (False, True, True))
    )
    materialized = selected.materialize()

    assert materialized.subjects == (subject,)
    assert materialized.axes[0].kind == "site"
    subset_map = data.get(materialized.axes[0].index_map, JSON)
    assert subset_map["labels"] == ["R1", "R2"]
    np.testing.assert_array_equal(
        data.get(materialized.values, ARRAYS),
        np.asarray([[2, 3], [4, 5]], dtype=float),
    )
    np.testing.assert_array_equal(
        data.get(measurement.values, ARRAYS),
        np.arange(6, dtype=float).reshape(3, 2),
    )
    assert any(
        dependency.role == "selection-mask"
        for dependency in materialized.inputs.dependencies
    )


def test_between_uses_two_distinct_named_ordered_axes_and_reduces_mean():
    data = DataStore()
    measurement, rows, columns = _measurement(
        data, np.arange(6, dtype=float).reshape(3, 2)
    )
    subject = measurement.subjects[0]
    view = measurement_view(measurement, data).between(
        _selection(data, subject, rows, (True, False, True)),
        _selection(data, subject, columns, (True, True)),
        left_axis="row_site",
        right_axis="column_site",
    )
    reduction = Reduction(
        axes=("row_site",),
        method="mean",
        parameters=_parameters(data, {}),
        missing="fail",
    )

    result = view.reduce(reduction)

    np.testing.assert_array_equal(
        data.get(result.values, ARRAYS),
        np.asarray([2.0, 3.0]),
    )
    np.testing.assert_array_equal(
        data.get(measurement.values, ARRAYS),
        np.arange(6, dtype=float).reshape(3, 2),
    )
    assert [axis.name for axis in result.axes] == ["column_site"]
    assert any(
        dependency.role == "source-measurement"
        for dependency in result.inputs.dependencies
    )
    assert any(
        dependency.role == "reduction" for dependency in result.inputs.dependencies
    )

    with pytest.raises(ValueError, match="distinct"):
        measurement_view(measurement, data).between(
            _selection(data, subject, rows, (True, True, False)),
            _selection(data, subject, rows, (False, True, True)),
            left_axis="row_site",
            right_axis="row_site",
        )


def test_reductions_apply_all_methods_and_explicit_missing_policies():
    data = DataStore()
    values = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 100.0],
            [5.0, 6.0],
        ]
    )
    mask = np.ones(values.shape, dtype=bool)
    mask[1, 1] = False
    measurement, rows, _ = _measurement(
        data, values, valid_mask=mask, missing="partial"
    )
    subject = measurement.subjects[0]
    selection = _selection(data, subject, rows, (False, True, True))
    parameters = _parameters(data, {})

    expected = {
        "mean": [4.0, 6.0],
        "sum": [8.0, 6.0],
        "min": [3.0, 6.0],
        "max": [5.0, 6.0],
    }
    for method, expected_values in expected.items():
        result = reduce_measurement(
            measurement,
            data,
            (selection,),
            Reduction(("row_site",), method, parameters, "omit"),
        )
        np.testing.assert_array_equal(
            data.get(result.values, ARRAYS),
            np.asarray(expected_values, dtype=float),
        )

    propagated = reduce_measurement(
        measurement,
        data,
        (selection,),
        Reduction(("row_site",), "mean", parameters, "propagate"),
    )
    propagated_values = data.get(propagated.values, ARRAYS)
    propagated_mask = data.get(propagated.valid_mask, ARRAYS)
    assert np.isnan(propagated_values[1])
    assert not propagated_mask[1]

    with pytest.raises(ValueError, match="missing='fail'"):
        reduce_measurement(
            measurement,
            data,
            (selection,),
            Reduction(("row_site",), "mean", parameters, "fail"),
        )


def test_weighted_mean_uses_parameters_and_unavailable_stays_absent():
    data = DataStore()
    measurement, rows, _ = _measurement(
        data,
        np.asarray(
            [
                [1.0, 2.0],
                [5.0, 6.0],
                [9.0, 10.0],
            ]
        ),
    )
    subject = measurement.subjects[0]
    weights = data.put({"weights": [1.0, 3.0]}, JSON)
    result = reduce_measurement(
        measurement,
        data,
        (_selection(data, subject, rows, (True, True, False)),),
        Reduction(("row_site",), "weighted_mean", weights, "fail"),
    )
    np.testing.assert_array_equal(
        data.get(result.values, ARRAYS), np.asarray([4.0, 5.0])
    )

    unavailable = Measurement(
        identity=key_for("unavailable"),
        quantity="test/unavailable",
        definition=key_for("definition/unavailable"),
        units="count",
        producer=key_for("producer/unavailable"),
        subjects=(subject,),
        axes=(rows,),
        values=None,
        missing="unavailable",
        valid_mask=None,
        inputs=manifest(()),
    )
    absent = measurement_view(unavailable, data).reduce(
        Reduction(("row_site",), "sum", _parameters(data, {}), "omit")
    )
    assert absent.values is None
    assert absent.missing == "unavailable"
    with pytest.raises(ValueError, match="values are unavailable"):
        measurement_view(unavailable, data).reduce(
            Reduction(("row_site",), "sum", _parameters(data, {}), "fail")
        )


def test_typed_measurement_operations_are_real_functions():
    data = DataStore()
    measurement, rows, _ = _measurement(data, np.arange(6, dtype=float).reshape(3, 2))
    subject = measurement.subjects[0]
    selection = _selection(data, subject, rows, (False, True, True))
    registry = MeasurementOperationRegistry

    projection_operation = registry.build_operation(
        "materialize",
        alias="projection",
        options={},
        context=data,
    )
    request = MeasurementProjectionRequest(measurement, (selection,))
    projected = projection_operation.invoke(projection_operation.prepare(request))
    assert projected == materialize_measurement(measurement, data, (selection,))

    reduction_operation = registry.build_operation(
        "reduce",
        alias="reduction",
        options={},
        context=data,
    )
    reduction_request = MeasurementReductionRequest(
        measurement,
        (selection,),
        Reduction(("row_site",), "sum", _parameters(data, {}), "fail"),
    )
    reduced = reduction_operation.invoke(
        reduction_operation.prepare(reduction_request)
    )
    assert reduced == reduce_measurement(
        measurement,
        data,
        reduction_request.selections,
        reduction_request.reduction,
    )


def test_missing_status_and_nonfinite_values_must_match_array_and_mask_state():
    data = DataStore()
    subject = _subject()
    rows = Axis("site", "site", _axis_map(data, ("R0", "R1")))
    columns = Axis("pair", "site", _axis_map(data, ("C0", "C1")))
    base = Measurement(
        identity=key_for("status-check"),
        quantity="test/status",
        definition=key_for("definition/status"),
        units="count",
        producer=key_for("producer/status"),
        subjects=(subject,),
        axes=(rows, columns),
        values=data.put(np.asarray([[1.0, 2.0], [3.0, 4.0]]), ARRAYS),
        missing="present",
        valid_mask=None,
        inputs=manifest(()),
    )
    with pytest.raises(ValueError, match="present measurements require values"):
        measurement_view(
            Measurement(
                identity=key_for("none-present"),
                quantity=base.quantity,
                definition=base.definition,
                units=base.units,
                producer=base.producer,
                subjects=base.subjects,
                axes=base.axes,
                values=None,
                missing="present",
                valid_mask=None,
                inputs=base.inputs,
            ),
            data,
        )
    with pytest.raises(ValueError, match="partial measurements require values"):
        measurement_view(
            Measurement(
                identity=key_for("none-partial"),
                quantity=base.quantity,
                definition=base.definition,
                units=base.units,
                producer=base.producer,
                subjects=base.subjects,
                axes=base.axes,
                values=None,
                missing="partial",
                valid_mask=None,
                inputs=base.inputs,
            ),
            data,
        )
    with pytest.raises(ValueError, match="cannot contain values"):
        measurement_view(
            Measurement(
                identity=key_for("values-unavailable"),
                quantity=base.quantity,
                definition=base.definition,
                units=base.units,
                producer=base.producer,
                subjects=base.subjects,
                axes=base.axes,
                values=base.values,
                missing="unavailable",
                valid_mask=None,
                inputs=base.inputs,
            ),
            data,
        )

    nan_values = np.asarray([[1.0, np.nan], [3.0, 4.0]])
    with pytest.raises(ValueError, match="non-finite"):
        measurement_view(
            Measurement(
                identity=key_for("present-nan"),
                quantity=base.quantity,
                definition=base.definition,
                units=base.units,
                producer=base.producer,
                subjects=base.subjects,
                axes=base.axes,
                values=data.put(nan_values, ARRAYS),
                missing="present",
                valid_mask=None,
                inputs=base.inputs,
            ),
            data,
        )
    partial = Measurement(
        identity=key_for("partial-nan"),
        quantity=base.quantity,
        definition=base.definition,
        units=base.units,
        producer=base.producer,
        subjects=base.subjects,
        axes=base.axes,
        values=data.put(nan_values, ARRAYS),
        missing="partial",
        valid_mask=data.put(
            np.asarray([[True, False], [True, True]], dtype=bool),
            ARRAYS,
        ),
        inputs=base.inputs,
    )
    measurement_view(partial, data)

    hidden_nan = Measurement(
        identity=key_for("hidden-nan"),
        quantity=partial.quantity,
        definition=partial.definition,
        units=partial.units,
        producer=partial.producer,
        subjects=partial.subjects,
        axes=partial.axes,
        values=partial.values,
        missing="partial",
        valid_mask=data.put(np.ones((2, 2), dtype=bool), ARRAYS),
        inputs=partial.inputs,
    )
    with pytest.raises(
        ValueError, match="marked invalid|at least one invalid"
    ):
        measurement_view(hidden_nan, data)


def test_empty_axis_reduction_honors_missing_for_scalar_nan():
    data = DataStore()
    subject = _subject()
    values = data.put(np.asarray(np.nan, dtype=np.float32), ARRAYS)
    mask = data.put(np.asarray(False), ARRAYS)
    measurement = Measurement(
        identity=key_for("scalar-nan"),
        quantity="test/scalar-nan",
        definition=key_for("definition/scalar-nan"),
        units="unitless",
        producer=key_for("producer/scalar-nan"),
        subjects=(subject,),
        axes=(),
        values=values,
        missing="partial",
        valid_mask=mask,
        inputs=manifest(()),
    )
    view = measurement_view(measurement, data)
    parameters = _parameters(data, {})

    with pytest.raises(ValueError, match="missing='fail'"):
        view.reduce(Reduction((), "mean", parameters, "fail"))

    for policy in ("omit", "propagate"):
        selected = view.reduce(Reduction((), "mean", parameters, policy))
        selected_values = data.get(selected.values, ARRAYS)
        selected_mask = data.get(selected.valid_mask, ARRAYS)
        assert selected.missing == "partial"
        assert selected_values.shape == ()
        assert np.isnan(selected_values.item())
        assert selected_mask.shape == ()
        assert bool(selected_mask.item()) is False


def test_empty_axis_reduction_keeps_all_invalid_arrays_and_partial_mask():
    data = DataStore()
    values = np.asarray(
        [
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    mask = np.zeros(values.shape, dtype=bool)
    subject = _subject()
    rows = Axis(
        "row_site",
        "site",
        _axis_map(data, ("R0", "R1", "R2")),
    )
    columns = Axis(
        "column_site",
        "site",
        _axis_map(data, ("C0", "C1")),
    )
    measurement = Measurement(
        identity=key_for("all-invalid-array"),
        quantity="test/all-invalid",
        definition=key_for("definition/all-invalid"),
        units="angstrom",
        producer=key_for("producer/all-invalid"),
        subjects=(subject,),
        axes=(rows, columns),
        values=data.put(values, ARRAYS),
        missing="partial",
        valid_mask=data.put(mask, ARRAYS),
        inputs=manifest(()),
    )
    view = measurement_view(measurement, data)
    parameters = _parameters(data, {})

    with pytest.raises(ValueError, match="missing='fail'"):
        view.reduce(Reduction((), "mean", parameters, "fail"))

    for policy in ("omit", "propagate"):
        selected = view.reduce(Reduction((), "mean", parameters, policy))
        selected_values = data.get(selected.values, ARRAYS)
        selected_mask = data.get(selected.valid_mask, ARRAYS)
        assert selected.missing == "partial"
        assert selected_values.shape == values.shape
        assert np.isnan(selected_values).all()
        np.testing.assert_array_equal(selected_mask, mask)


def test_cross_subject_ordered_pair_uses_explicit_axis_bindings():
    data = DataStore()
    first = _subject("first")
    second = _subject("second")
    first_key = key_for(first).digest
    second_key = key_for(second).digest
    rows = Axis(
        "row_site",
        "site",
        _axis_map(data, ("R0", "R1", "R2"), (first_key,) * 3),
    )
    columns = Axis(
        "column_site",
        "site",
        _axis_map(data, ("C0", "C1"), (second_key,) * 2),
    )
    values = np.arange(6, dtype=float).reshape(3, 2)
    measurement = Measurement(
        identity=key_for("cross-subject-pae"),
        quantity="test/pae",
        definition=key_for("definition/cross-subject-pae"),
        units="angstrom",
        producer=key_for("producer/pae"),
        subjects=(first, second),
        axes=(rows, columns),
        values=data.put(values, ARRAYS),
        missing="present",
        valid_mask=None,
        inputs=manifest((Dependency("source", key_for("pair-source")),)),
    )

    selected = measurement_view(measurement, data).between(
        _selection(data, first, rows, (True, False, True)),
        _selection(data, second, columns, (False, True)),
        left_axis="row_site",
        right_axis="column_site",
    )
    result = selected.reduce(
        Reduction(
            ("row_site",),
            "mean",
            _parameters(data, {}),
            "fail",
        )
    )

    assert result.subjects == (first, second)
    np.testing.assert_array_equal(
        data.get(result.values, ARRAYS), np.asarray([3.0])
    )
    resulting_map = data.get(result.axes[0].index_map, JSON)
    assert resulting_map["labels"] == ["C1"]
    assert resulting_map["subject_keys"] == [second_key]

    with pytest.raises(ValueError, match="another subject"):
        measurement_view(measurement, data).select(
            _selection(data, first, columns, (True, False))
        )

    unbound = Measurement(
        identity=key_for("unbound-cross-subject"),
        quantity="test/pae",
        definition=measurement.definition,
        units=measurement.units,
        producer=measurement.producer,
        subjects=measurement.subjects,
        axes=(
            rows,
            Axis("column_site", "site", _axis_map(data, ("C0", "C1"))),
        ),
        values=measurement.values,
        missing="present",
        valid_mask=None,
        inputs=measurement.inputs,
    )
    with pytest.raises(ValueError, match="requires explicit"):
        measurement_view(unbound, data)


@pytest.mark.parametrize("method", ("sum", "mean", "min", "max", "weighted_mean"))
@pytest.mark.parametrize("policy", ("fail", "omit", "propagate"))
def test_empty_selection_reduction_is_missing_never_valid_zero(method, policy):
    data = DataStore()
    measurement, rows, _ = _measurement(data, np.arange(6.0).reshape(3, 2))
    selection = _selection(data, measurement.subjects[0], rows, (False,) * 3)
    empty = materialize_measurement(measurement, data, (selection,))
    assert empty.missing == "partial"
    assert data.get(empty.values, ARRAYS).shape == (0, 2)
    assert data.get(empty.axes[0].index_map, JSON)["labels"] == []
    measurement_view(empty, data)
    parameters = _parameters(data, {"weights": 1.0} if method == "weighted_mean" else {})
    reduction = Reduction((rows.name,), method, parameters, policy)

    if policy == "fail":
        with pytest.raises(ValueError, match="empty.*missing='fail'"):
            reduce_measurement(measurement, data, (selection,), reduction)
        return

    result = reduce_measurement(measurement, data, (selection,), reduction)
    assert result.missing == "partial"
    assert np.isnan(data.get(result.values, ARRAYS)).all()
    np.testing.assert_array_equal(data.get(result.valid_mask, ARRAYS), [False, False])
    # A missing result must itself remain a valid input to another reduction.
    again = measurement_view(result, data).reduce(
        Reduction(("column_site",), method, parameters, policy)
    )
    assert np.isnan(data.get(again.values, ARRAYS).item())
    assert not data.get(again.valid_mask, ARRAYS).item()


@pytest.mark.parametrize("policy", ("fail", "omit", "propagate"))
@pytest.mark.parametrize("axes", ((), ("row_site",), ("row_site", "column_site")))
@pytest.mark.parametrize("with_payload", (False, True))
def test_failed_measurements_never_become_valid(policy, axes, with_payload):
    data = DataStore()
    measurement, rows, _ = _measurement(data, np.arange(6.0).reshape(3, 2))
    failed = replace(
        measurement,
        identity=key_for(("failed", with_payload)),
        missing="failed",
        values=measurement.values if with_payload else None,
    )
    selection = _selection(data, failed.subjects[0], rows, (True, False, True))
    reduction = Reduction(axes, "mean", _parameters(data, {}), policy)
    if policy == "fail":
        with pytest.raises(ValueError, match="failed.*missing='fail'"):
            reduce_measurement(failed, data, (selection,), reduction)
        return

    result = reduce_measurement(failed, data, (selection,), reduction)
    assert result.missing == "failed"
    assert result.values is None
    assert result.valid_mask is None
    assert tuple(axis.name for axis in result.axes) == tuple(
        axis.name for axis in failed.axes if axis.name not in axes
    )
    measurement_view(result, data)
    assert any(dep.identity == failed.identity for dep in result.inputs.dependencies)


@pytest.mark.parametrize(
    "mask, expected_status",
    [((True, False, True), "present"), ((False, True, False), "partial")],
)
def test_projection_relabels_partial_from_selected_mask_and_preserves_origin(
    mask, expected_status
):
    data = DataStore()
    measurement, rows, _ = _measurement(
        data,
        np.arange(6.0).reshape(3, 2),
        valid_mask=[[True, True], [False, False], [True, True]],
        missing="partial",
    )
    selection = _selection(data, measurement.subjects[0], rows, mask)
    result = materialize_measurement(measurement, data, (selection,))
    assert result.missing == expected_status
    assert result.quantity == measurement.quantity
    assert result.definition == measurement.definition
    assert result.units == measurement.units
    assert result.producer == measurement.producer
    assert result.subjects == measurement.subjects
    assert Dependency("source-measurement", measurement.identity) in result.inputs.dependencies
    assert Dependency("selection-mask", selection.mask.identity) in result.inputs.dependencies
    for upstream in measurement.inputs.dependencies:
        assert Dependency(f"upstream:{upstream.role}", upstream.identity) in result.inputs.dependencies
    assert measurement.missing == "partial"
    view = measurement_view(result, data)
    reduced = view.reduce(
        Reduction((rows.name,), "mean", _parameters(data, {}), "omit")
    )
    measurement_view(reduced, data)
    if expected_status == "present":
        np.testing.assert_array_equal(data.get(reduced.values, ARRAYS), [2.0, 3.0])
    else:
        assert np.isnan(data.get(reduced.values, ARRAYS)).all()
        assert not data.get(reduced.valid_mask, ARRAYS).any()


@pytest.mark.parametrize("empty", (False, True))
def test_unavailable_measurement_can_be_selected_and_reduced(empty):
    data = DataStore()
    measurement, rows, _ = _measurement(data, np.arange(6.0).reshape(3, 2))
    absent = replace(
        measurement, identity=key_for("unavailable-selection"),
        values=None, missing="unavailable",
    )
    mask = (False,) * 3 if empty else (True, False, True)
    selection = _selection(data, absent.subjects[0], rows, mask)
    projected = materialize_measurement(absent, data, (selection,))
    assert projected.values is None
    assert projected.valid_mask is None
    assert projected.missing == "unavailable"
    assert data.get(projected.axes[0].index_map, JSON)["labels"] == (
        [] if empty else ["R0", "R2"]
    )
    assert Dependency("source-measurement", absent.identity) in projected.inputs.dependencies
    assert projected.subjects == absent.subjects
    measurement_view(projected, data)
    for policy in ("omit", "propagate"):
        result = reduce_measurement(
            absent, data, (selection,),
            Reduction((rows.name,), "sum", _parameters(data, {}), policy),
        )
        assert result.missing == "unavailable"
        assert result.values is None
        assert result.valid_mask is None
        assert [axis.name for axis in result.axes] == ["column_site"]
        measurement_view(result, data)
    with pytest.raises(ValueError, match="values are unavailable"):
        reduce_measurement(
            absent, data, (selection,),
            Reduction((rows.name,), "sum", _parameters(data, {}), "fail"),
        )


def test_weight_shapes_follow_declared_reduction_axes_and_broadcast_only():
    data = DataStore()
    values = np.arange(6.0).reshape(3, 2)
    measurement, rows, columns = _measurement(data, values)
    for bad_weights in ([[1, 2, 3], [4, 5, 6]], [1, 2, 3, 4, 5, 6]):
        with pytest.raises(ValueError, match="weight shape.*reduced shape"):
            reduce_measurement(
                measurement, data, (),
                Reduction(
                    (rows.name, columns.name), "weighted_mean",
                    _parameters(data, {"weights": bad_weights}), "fail",
                ),
            )
    # Each case has an independently specified element-to-weight assignment.
    for weights, expanded in (
        (2, np.full((3, 2), 2)),
        ([1, 3], np.array([[1, 3], [1, 3], [1, 3]])),
        ([[1], [2], [4]], np.array([[1, 1], [2, 2], [4, 4]])),
        ([[1, 2], [3, 4], [5, 6]], np.array([[1, 2], [3, 4], [5, 6]])),
    ):
        expected = (values * expanded).sum() / expanded.sum()
        result = reduce_measurement(
            measurement, data, (),
            Reduction(
                (rows.name, columns.name), "weighted_mean",
                _parameters(data, {"weights": weights}), "fail",
            ),
        )
        assert data.get(result.values, ARRAYS).item() == pytest.approx(expected)
        # Reversing the explicit axes requires reversing the weight dimensions.
        reverse = reduce_measurement(
            measurement, data, (),
            Reduction(
                (columns.name, rows.name), "weighted_mean",
                _parameters(data, {"weights": expanded.T.tolist()}), "fail",
            ),
        )
        assert data.get(reverse.values, ARRAYS).item() == pytest.approx(expected)


def test_stable_averages_and_unrepresentable_results_are_never_valid_inf():
    data = DataStore()
    maximum = np.finfo(float).max
    tiny = np.nextafter(0.0, 1.0)
    for value in (1e308, maximum, tiny):
        measurement, rows, _ = _measurement(data, np.full((3, 2), value))
        for method, parameters in (
            ("mean", {}),
            ("weighted_mean", {"weights": [maximum] * 3}),
        ):
            result = reduce_measurement(
                measurement, data, (),
                Reduction((rows.name,), method, _parameters(data, parameters), "fail"),
            )
            np.testing.assert_array_equal(data.get(result.values, ARRAYS), [value, value])
            assert result.missing == "present"
            measurement_view(result, data)
    overflowing, rows, _ = _measurement(data, np.full((3, 2), 1e308))
    with pytest.raises(ValueError, match="cannot be represented as finite float64"):
        reduce_measurement(
            overflowing, data, (),
            Reduction((rows.name,), "sum", _parameters(data, {}), "fail"),
        )


@pytest.mark.parametrize("policy", ("fail", "omit", "propagate"))
def test_zero_weight_groups_are_missing_or_raise(policy):
    data = DataStore()
    measurement, rows, _ = _measurement(data, np.arange(6.0).reshape(3, 2))
    reduction = Reduction(
        (rows.name,), "weighted_mean", _parameters(data, {"weights": [0, 0, 0]}), policy
    )
    if policy == "fail":
        with pytest.raises(ValueError, match="no positive weight.*missing='fail'"):
            measurement_view(measurement, data).reduce(reduction)
    else:
        result = measurement_view(measurement, data).reduce(reduction)
        assert np.isnan(data.get(result.values, ARRAYS)).all()
        assert not data.get(result.valid_mask, ARRAYS).any()
        measurement_view(result, data)


def test_stable_reduction_preserves_cancellation_and_extreme_weight_products():
    data = DataStore()
    measurement, rows, _ = _measurement(
        data, np.array([[1e308, 1e308], [1.0, 1e-308], [-1e308, -1e308]])
    )
    for method, divisor in (("sum", 1), ("mean", 3), ("weighted_mean", 3)):
        parameters = {"weights": [1, 1, 1]} if method == "weighted_mean" else {}
        result = reduce_measurement(
            measurement, data, (),
            Reduction((rows.name,), method, _parameters(data, parameters), "fail"),
        )
        np.testing.assert_allclose(
            data.get(result.values, ARRAYS), np.array([1.0, 1e-308]) / divisor,
            rtol=1e-14, atol=0,
        )
    extreme, rows, _ = _measurement(data, np.array([[1e308, 1e308], [0, 0], [0, 0]]))
    result = reduce_measurement(
        extreme, data, (), Reduction(
            (rows.name,), "weighted_mean",
            _parameters(data, {"weights": [1e-308, 1e308, 0]}), "fail",
        ),
    )
    np.testing.assert_allclose(data.get(result.values, ARRAYS), [1e-308] * 2, rtol=1e-14, atol=0)
    underflow, rows, _ = _measurement(
        data, np.array([[np.nextafter(0.0, 1.0)] * 2, [0, 0], [0, 0]])
    )
    with pytest.raises(ValueError, match="underflows float64"):
        reduce_measurement(
            underflow, data, (),
            Reduction((rows.name,), "mean", _parameters(data, {}), "fail"),
        )


@pytest.mark.parametrize("policy", ("omit", "propagate"))
def test_masked_nonfinite_payloads_and_retained_empty_axes_remain_composable(policy):
    data = DataStore()
    measurement, rows, columns = _measurement(
        data, np.array([[1, 2], [0, 0], [5, 6]]),
        valid_mask=[[True, True], [False, False], [True, True]], missing="partial",
    )
    measurement = replace(
        measurement, identity=key_for("masked-nonfinite-payloads"),
        values=data.put(np.array([[1, 2], [np.inf, np.nan], [5, 6]]), ARRAYS),
    )
    result = reduce_measurement(
        measurement, data, (),
        Reduction((rows.name,), "weighted_mean", _parameters(data, {"weights": [1, 0, 1]}), policy),
    )
    if policy == "omit":
        np.testing.assert_array_equal(data.get(result.values, ARRAYS), [3, 4])
    else:
        assert np.isnan(data.get(result.values, ARRAYS)).all()
    measurement_view(result, data)
    empty = _selection(data, measurement.subjects[0], rows, (False,) * 3)
    retained = reduce_measurement(
        measurement, data, (empty,),
        Reduction((columns.name,), "min", _parameters(data, {}), policy),
    )
    assert data.get(retained.values, ARRAYS).shape == (0,)
    assert retained.missing == "partial"
    measurement_view(retained, data)
    again = measurement_view(retained, data).reduce(
        Reduction((rows.name,), "max", _parameters(data, {}), policy)
    )
    assert np.isnan(data.get(again.values, ARRAYS).item())


def test_typed_operations_share_content_identity_across_stores_and_close_results():
    records = []
    for data in (DataStore(), DataStore()):
        measurement, rows, _ = _measurement(data, np.full((3, 2), 1e308))
        projection = MeasurementOperationRegistry.build_operation(
            "materialize", alias="project", options={}, context=data
        )
        empty = _selection(data, measurement.subjects[0], rows, (False,) * 3)
        request = MeasurementProjectionRequest(measurement, (empty,))
        projected = projection.invoke(projection.prepare(request))
        decoded = projection.descriptor.result_codec.decode(
            projection.descriptor.result_codec.encode(projected)
        )
        measurement_view(decoded, data)
        assert projection.descriptor.request_codec.references(request)
        assert projection.descriptor.result_codec.references(decoded)
        operation = MeasurementOperationRegistry.build_operation(
            "reduce", alias="reduce", options={}, context=data
        )
        reduction = Reduction((rows.name,), "mean", _parameters(data, {}), "fail")
        prepared = operation.prepare(MeasurementReductionRequest(measurement, (), reduction))
        result = operation.invoke(prepared)
        np.testing.assert_array_equal(data.get(result.values, ARRAYS), [1e308, 1e308])
        measurement_view(result, data)
        records.append((projection.descriptor.identity, decoded, operation.descriptor.identity, prepared.inputs, result))
    assert records[0] == records[1]


@pytest.mark.parametrize("axes", (("row_site", "column_site"), ("column_site", "row_site")))
@pytest.mark.parametrize("method", ("sum", "mean", "weighted_mean"))
def test_multilevel_cancellation_keeps_unit_residual_in_both_axis_orders(axes, method):
    data = DataStore()
    measurement, _, _ = _measurement(
        data, np.array([[1e40, 1e20], [1, -1e40], [-1e20, 0]])
    )
    result = reduce_measurement(
        measurement, data, (), Reduction(
            axes, method,
            _parameters(data, {"weights": 1} if method == "weighted_mean" else {}),
            "fail",
        ),
    )
    assert data.get(result.values, ARRAYS).item() == (1 if method == "sum" else 1 / 6)
    assert result.missing == "present"
    assert result.valid_mask is None
    measurement_view(result, data)


@pytest.mark.parametrize("axes", (
    ("row_site",), ("column_site",),
    ("row_site", "column_site"), ("column_site", "row_site"),
))
@pytest.mark.parametrize("method", ("sum", "mean", "weighted_mean"))
def test_accurate_reductions_match_fraction_oracle_with_nonuniform_weights(axes, method):
    data = DataStore()
    values = np.array([[1e40, 1e20], [1, -1e40], [-1e20, 0]])
    weights = np.array([[3, 7], [5, 3], [7, 11]])
    measurement, rows, columns = _measurement(data, values)
    indices = tuple((rows.name, columns.name).index(name) for name in axes)
    if indices == (0,):
        weight_parameter = [3, 5, 7]
        expanded_weights = np.array([[3, 3], [5, 5], [7, 7]])
    elif indices == (1,):
        weight_parameter = [3, 7]
        expanded_weights = np.array([[3, 7], [3, 7], [3, 7]])
    else:
        weight_parameter = weights.tolist() if indices == (0, 1) else weights.T.tolist()
        expanded_weights = weights
    # The oracle uses Fraction, independently of the integer-accumulator code.
    expected = []
    remaining = tuple(index for index in range(2) if index not in indices)
    shape = tuple(values.shape[index] for index in remaining)
    for retained in np.ndindex(shape):
        selector = [slice(None), slice(None)]
        for index, position in zip(remaining, retained):
            selector[index] = position
        group = values[tuple(selector)].ravel().tolist()
        coefficients = expanded_weights[tuple(selector)].ravel().tolist()
        if method == "weighted_mean":
            exact = sum(
                (Fraction(value) * weight for value, weight in zip(group, coefficients)),
                Fraction(),
            ) / sum(coefficients)
        else:
            exact = sum(map(Fraction, group), Fraction())
            if method == "mean":
                exact /= len(group)
        expected.append(float(exact))
    result = reduce_measurement(
        measurement, data, (), Reduction(
            axes, method,
            _parameters(
                data, {"weights": weight_parameter} if method == "weighted_mean" else {}
            ), "fail",
        ),
    )
    np.testing.assert_array_equal(
        data.get(result.values, ARRAYS), np.array(expected).reshape(shape)
    )
    assert result.missing == "present"


@pytest.mark.parametrize("policy", ("fail", "omit", "propagate"))
@pytest.mark.parametrize("method", ("sum", "mean", "weighted_mean"))
def test_accurate_cancellation_preserves_explicit_missing_policy(policy, method):
    data = DataStore()
    measurement, _, _ = _measurement(
        data, np.array([[1e40, 1e20], [1, -1e40], [-1e20, 0]]),
        valid_mask=[[True, True], [True, True], [True, False]], missing="partial",
    )
    reduction = Reduction(
        ("row_site", "column_site"), method,
        _parameters(data, {"weights": 1} if method == "weighted_mean" else {}), policy,
    )
    if policy == "fail":
        with pytest.raises(ValueError, match="invalid.*missing='fail'"):
            measurement_view(measurement, data).reduce(reduction)
        return
    result = measurement_view(measurement, data).reduce(reduction)
    if policy == "omit":
        assert data.get(result.values, ARRAYS).item() == (1 if method == "sum" else 1 / 5)
        assert result.missing == "present"
        assert result.valid_mask is None
    else:
        assert np.isnan(data.get(result.values, ARRAYS).item())
        assert result.missing == "partial"
        assert not data.get(result.valid_mask, ARRAYS).item()
    assert measurement.missing == "partial"
    measurement_view(result, data)


@pytest.mark.parametrize("policy", ("fail", "omit", "propagate"))
def test_exact_arithmetic_rejects_output_overflow_without_changing_missing(policy):
    data = DataStore()
    maximum = np.finfo(float).max
    source, rows, _ = _measurement(
        data, np.array([[maximum, maximum], [maximum, maximum], [-maximum, -maximum]])
    )
    # A float intermediate would overflow; the actual output is representable.
    result = reduce_measurement(
        source, data, (), Reduction((rows.name,), "sum", _parameters(data, {}), policy)
    )
    np.testing.assert_array_equal(data.get(result.values, ARRAYS), [maximum, maximum])
    overflow = replace(
        source, identity=key_for("exact-output-overflow"),
        values=data.put(np.full((3, 2), maximum), ARRAYS),
    )
    with pytest.raises(ValueError, match="cannot be represented as finite float64"):
        reduce_measurement(
            overflow, data, (), Reduction((rows.name,), "sum", _parameters(data, {}), policy)
        )
    assert source.missing == overflow.missing == "present"
    # Mean and products exceeding float range still yield the correct finite mean.
    for method, parameters in (("mean", {}), ("weighted_mean", {"weights": [maximum] * 3})):
        average = reduce_measurement(
            overflow, data, (), Reduction((rows.name,), method, _parameters(data, parameters), policy)
        )
        np.testing.assert_array_equal(data.get(average.values, ARRAYS), [maximum, maximum])
        assert average.missing == "present"


@pytest.mark.parametrize("method", ("sum", "mean", "weighted_mean"))
def test_accurate_arithmetic_uses_binary_input_precision_and_keeps_true_zero(method):
    data = DataStore()
    for values in (
        np.array([[0.1, 0.2], [-0.3, 0], [0, 0]]),
        np.array([[0.1, 0.2], [-0.3, 0], [0, 0]], dtype=np.float32),
        np.array([[2**60, 1], [-2**60, 0], [0, 0]], dtype=np.int64),
        np.array([[1e40, -1e40], [0, 0], [0, 0]]),
        # Normal binary64 rounding near its upper limit remains permitted.
        np.array([[np.finfo(float).max, 2.0**969], [0, 0], [0, 0]]),
    ):
        source, _, _ = _measurement(data, np.zeros((3, 2)))
        reference = data.put(values, ARRAYS)
        source = replace(source, identity=key_for(("input-precision", reference)), values=reference)
        parameters = {"weights": 1} if method == "weighted_mean" else {}
        result = reduce_measurement(
            source, data, (), Reduction(
                ("row_site", "column_site"), method, _parameters(data, parameters), "fail"
            ),
        )
        exact = sum((Fraction(value) for value in values.ravel().tolist()), Fraction())
        if method != "sum":
            exact /= values.size
        assert data.get(result.values, ARRAYS).item() == float(exact)
        assert data.get(result.values, ARRAYS).dtype == np.float64
        assert result.missing == "present"
        assert result.valid_mask is None


def test_weighted_mean_preserves_residual_lost_by_rounded_products():
    data = DataStore()
    first = 1 + 2**-27
    weight = 1 - 2**-27
    assert first * weight == 1.0  # A rounded product would erase the residual.
    values = np.array([[first, -1], [0, 0], [0, 0]])
    weights = [[weight, 1], [0, 0], [0, 0]]
    measurement, _, _ = _measurement(data, values)
    expected = float((Fraction(first) * Fraction(weight) - 1) / (Fraction(weight) + 1))
    assert expected != 0
    result = reduce_measurement(
        measurement, data, (), Reduction(
            ("row_site", "column_site"), "weighted_mean",
            _parameters(data, {"weights": weights}), "fail",
        ),
    )
    assert data.get(result.values, ARRAYS).item() == expected
    assert result.missing == "present"


@pytest.mark.parametrize("method", ("sum", "mean", "weighted_mean"))
def test_declared_missing_and_numerical_overflow_are_distinct(method):
    data = DataStore()
    maximum = np.finfo(float).max
    measurement, rows, _ = _measurement(
        data, np.array([[maximum, 1], [maximum, 2], [0, 3]]),
        valid_mask=[[True, True], [True, True], [False, True]], missing="partial",
    )
    parameters = _parameters(data, {"weights": [1, 1, 1]} if method == "weighted_mean" else {})
    with pytest.raises(ValueError, match="invalid.*missing='fail'"):
        reduce_measurement(measurement, data, (), Reduction((rows.name,), method, parameters, "fail"))
    propagated = reduce_measurement(
        measurement, data, (), Reduction((rows.name,), method, parameters, "propagate")
    )
    # The declared invalid group propagates missing without doing arithmetic on
    # its unreportable partial sum. The other group remains valid and unchanged.
    output = data.get(propagated.values, ARRAYS)
    assert np.isnan(output[0])
    assert output[1] == (6 if method == "sum" else 2)
    np.testing.assert_array_equal(data.get(propagated.valid_mask, ARRAYS), [False, True])
    assert propagated.missing == "partial"
    if method == "sum":
        with pytest.raises(ValueError, match="cannot be represented as finite float64"):
            reduce_measurement(measurement, data, (), Reduction((rows.name,), method, parameters, "omit"))
    else:
        omitted = reduce_measurement(
            measurement, data, (), Reduction((rows.name,), method, parameters, "omit")
        )
        np.testing.assert_array_equal(data.get(omitted.values, ARRAYS), [maximum, 2])
        assert omitted.missing == "present"
    assert measurement.missing == "partial"


@pytest.mark.parametrize("method", ("sum", "mean", "weighted_mean"))
def test_exact_accumulation_preserves_wide_inputs_and_tiny_weight_products(method):
    data = DataStore()
    source, rows, _ = _measurement(data, np.zeros((3, 2)))
    epsilon = np.finfo(np.longdouble).eps
    wide = np.array([[1 + np.longdouble(epsilon), 1], [-1, -1], [0, 0]], dtype=np.longdouble)
    reference = data.put(wide, ARRAYS)
    source = replace(source, identity=key_for(("wide-inputs", reference)), values=reference)
    result = reduce_measurement(
        source, data, (), Reduction(
            (rows.name,), method,
            _parameters(data, {"weights": [1, 1, 1]} if method == "weighted_mean" else {}), "fail",
        ),
    )
    expected = float(epsilon) if method == "sum" else float(epsilon / 3)
    np.testing.assert_array_equal(data.get(result.values, ARRAYS), [expected, 0])
    assert result.missing == "present"
    if method == "weighted_mean":
        tiny = np.nextafter(0.0, 1.0)
        source, rows, _ = _measurement(data, np.full((3, 2), tiny))
        tiny_result = reduce_measurement(
            source, data, (), Reduction(
                (rows.name,), method, _parameters(data, {"weights": [tiny] * 3}), "fail"
            ),
        )
        np.testing.assert_array_equal(data.get(tiny_result.values, ARRAYS), [tiny, tiny])
        assert tiny_result.missing == "present"


@pytest.mark.parametrize("status", ("present", "unavailable", "failed"))
@pytest.mark.parametrize("mixed", (False, True))
def test_conflicting_axis_subject_bindings_are_rejected_before_selection(status, mixed):
    data = DataStore()
    first, second = _subject("first"), _subject("second")
    first_key, second_key = key_for(first).digest, key_for(second).digest
    measurement, rows, columns = _measurement(data, np.arange(6.0).reshape(3, 2))
    conflicting = Axis(rows.name, "subject", data.put({
        "labels": ["R0", "R1", "R2"], "subject_key": first_key,
        "subject_keys": [first_key, first_key, second_key] if mixed else [second_key] * 3,
    }, JSON))
    bound_columns = replace(columns, index_map=_axis_map(data, ("C0", "C1"), (second_key,) * 2))
    measurement = replace(
        measurement, subjects=(first, second), axes=(conflicting, bound_columns),
        missing=status, values=measurement.values if status == "present" else None,
    )
    # Even a future mask restricted to agreeing entries must not hide a
    # contradictory axis definition; unavailable/failed inputs are checked too.
    with pytest.raises(ValueError, match="subject_key and subject_keys must agree"):
        measurement_view(measurement, data)


@pytest.mark.parametrize("form", ("subject_key", "subject_keys", "both"))
@pytest.mark.parametrize("empty", (False, True))
def test_consistent_axis_subject_bindings_preserve_selection_behavior(form, empty):
    data = DataStore()
    first, second = _subject("first"), _subject("second")
    first_key, second_key = key_for(first).digest, key_for(second).digest
    measurement, rows, columns = _measurement(data, np.arange(6.0).reshape(3, 2))
    payload = {"labels": ["R0", "R1", "R2"]}
    if form in {"subject_key", "both"}:
        payload["subject_key"] = first_key
    if form in {"subject_keys", "both"}:
        payload["subject_keys"] = [first_key] * 3
    bound_rows = Axis(rows.name, "subject", data.put(payload, JSON))
    bound_columns = replace(columns, index_map=_axis_map(data, ("C0", "C1"), (second_key,) * 2))
    measurement = replace(measurement, subjects=(first, second), axes=(bound_rows, bound_columns))
    view = measurement_view(measurement, data)
    mask = (False,) * 3 if empty else (True, False, True)
    projected = view.select(_selection(data, first, bound_rows, mask)).materialize()
    measurement_view(projected, data)
    assert projected.subjects == (first, second)
    np.testing.assert_array_equal(
        data.get(projected.values, ARRAYS), data.get(measurement.values, ARRAYS)[list(mask)]
    )
    if not empty:
        with pytest.raises(ValueError, match="another subject"):
            view.select(_selection(data, second, bound_rows, mask))


@pytest.mark.parametrize("axes", (
    ("row", "column"), ("column", "row"), ("model", "column"),
))
@pytest.mark.parametrize("method, weight_kind", (
    ("sum", None), ("mean", None), ("min", None), ("max", None),
    ("weighted_mean", "scalar"), ("weighted_mean", "leading"),
    ("weighted_mean", "trailing"), ("weighted_mean", "dense"),
))
@pytest.mark.parametrize("policy", ("fail", "omit", "propagate"))
def test_streamed_strided_groups_and_extreme_broadcast_weights_match_exact_oracle(
    axes, method, weight_kind, policy
):
    data = DataStore()
    values = np.array([
        [[1e40, 1e20, 1, -1e40], [-1e20, 0, 0.1, 0.2], [-0.3, 2, -2, 1]],
        [[-1e40, 1, 1e40, 0], [1e20, -1e20, 3, 0], [1, -1, 2, -2]],
    ])
    valid = np.ones(values.shape, dtype=bool)
    if policy != "fail":
        valid[0, 0, 0] = False
        values[0, 0, 0] = np.nan
    if policy == "omit":
        valid[1] = False
        values[1] = np.inf
    names = ("model", "row", "column")
    definitions = tuple(
        Axis(name, "site", _axis_map(data, tuple(str(i) for i in range(length))))
        for name, length in zip(names, values.shape)
    )
    reference = data.put(values, ARRAYS)
    source, _, _ = _measurement(data, np.zeros((3, 2)))
    source = replace(
        source, identity=key_for(("streamed-input", reference, policy)),
        axes=definitions, values=reference,
        valid_mask=data.put(valid, ARRAYS) if not valid.all() else None,
        missing="present" if valid.all() else "partial",
    )
    indices = tuple(names.index(name) for name in axes)
    reduced_shape = tuple(values.shape[index] for index in indices)
    remaining = tuple(index for index in range(values.ndim) if index not in indices)
    output_shape = tuple(values.shape[index] for index in remaining)
    palette = np.array([
        np.finfo(float).max, np.nextafter(0.0, 1.0), 0, 1e-200, 1e200, 2.0**-40,
    ])
    if weight_kind == "leading":
        weights = palette[np.arange(reduced_shape[0]) % palette.size, None]
    elif weight_kind == "trailing":
        weights = palette[np.arange(reduced_shape[1]) % palette.size]
    elif weight_kind == "dense":
        weights = palette[np.arange(np.prod(reduced_shape)).reshape(reduced_shape) % palette.size]
    else:
        weights = np.asarray(np.finfo(float).max if weight_kind == "scalar" else 1)
    expanded = np.broadcast_to(weights, reduced_shape)
    parameters = _parameters(data, {"weights": weights.tolist()} if weight_kind else {})
    reduction = Reduction(axes, method, parameters, policy)
    expected = np.full(output_shape, np.nan)
    expected_valid = np.zeros(output_shape, dtype=bool)
    unrepresentable = False
    for retained in np.ndindex(output_shape):
        numerator, denominator = Fraction(), Fraction()
        present = []
        all_valid = True
        for reduced in np.ndindex(reduced_shape):
            coordinate = [0] * values.ndim
            for index, position in zip(remaining, retained):
                coordinate[index] = position
            for index, position in zip(indices, reduced):
                coordinate[index] = position
            coordinate = tuple(coordinate)
            if not valid[coordinate]:
                all_valid = False
                continue
            value = Fraction(float(values[coordinate]))
            present.append(value)
            weight = Fraction(float(expanded[reduced])) if weight_kind else Fraction(1)
            numerator += value * weight
            denominator += weight
        if not present or (policy == "propagate" and not all_valid):
            continue
        if method == "weighted_mean" and denominator == 0:
            continue
        if method == "sum":
            exact = numerator
        elif method in {"mean", "weighted_mean"}:
            exact = numerator / denominator
        elif method == "min":
            exact = min(present)
        else:
            exact = max(present)
        rounded = float(exact)
        unrepresentable |= bool(exact and rounded == 0)
        expected[retained] = rounded
        expected_valid[retained] = True
    if unrepresentable:
        with pytest.raises(ValueError, match="underflows float64"):
            measurement_view(source, data).reduce(reduction)
        return
    result = measurement_view(source, data).reduce(reduction)
    np.testing.assert_array_equal(data.get(result.values, ARRAYS), expected)
    actual_valid = (
        np.ones(output_shape, dtype=bool) if result.valid_mask is None
        else data.get(result.valid_mask, ARRAYS)
    )
    np.testing.assert_array_equal(actual_valid, expected_valid)
    assert result.missing == ("present" if expected_valid.all() else "partial")
    assert tuple(axis.name for axis in result.axes) == tuple(names[index] for index in remaining)
    measurement_view(result, data)


@pytest.mark.parametrize("dtype", (np.int64, np.uint64))
@pytest.mark.parametrize("method", ("min", "max"))
def test_streamed_integer_extrema_keep_native_precision_and_mask(dtype, method):
    data = DataStore()
    limits = np.iinfo(dtype)
    values = np.array([[limits.max, limits.min], [1, 2], [3, 4]], dtype=dtype)
    valid = np.array([[False, True], [True, False], [True, True]])
    source, rows, _ = _measurement(data, np.zeros((3, 2)))
    source = replace(
        source, values=data.put(values, ARRAYS), valid_mask=data.put(valid, ARRAYS),
        missing="partial",
    )
    result = reduce_measurement(
        source, data, (), Reduction((rows.name,), method, _parameters(data, {}), "omit")
    )
    expected = [
        float((min if method == "min" else max)(
            int(values[row, column]) for row in range(3) if valid[row, column]
        )) for column in range(2)
    ]
    np.testing.assert_array_equal(data.get(result.values, ARRAYS), expected)
    assert result.missing == "present"


@pytest.mark.parametrize("case", ("scalar_extreme", "complex_broadcast"))
def test_fresh_process_peak_memory_has_no_broadcast_element_object_growth(case):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    environment = dict(os.environ)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "PYTHONPATH": os.pathsep.join(sys.path),
    })
    completed = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--measure-memory",
         "--size", "1000", "--case", case],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, check=True, timeout=120,
    )
    measurement = json.loads(completed.stdout)
    assert measurement["value"] == 1 and measurement["missing"] == "present"
    # Allow four dense input buffers for ArrayCodec load/copy and allocator
    # overlap. A ratio or large integer per broadcast element exceeds this
    # buffer-derived budget; the former implementation used over 50 buffers.
    assert measurement["increment_kib"] * 1024 <= 4 * measurement["values_bytes"]


def _measure_reduction_memory(size: int, case: str) -> dict:
    """Run in a fresh Linux process; report full public-API peak RSS and time.

    Reproduce with this file's --measure-memory CLI. Inputs, codecs, validation,
    and reduction are identical before/after an implementation change.
    """
    import resource
    import time

    data = DataStore()
    labels = tuple(str(index) for index in range(size))
    rows = Axis("row", "site", _axis_map(data, labels))
    columns = Axis("column", "site", _axis_map(data, labels))
    values = np.ones((size, size), dtype=np.float64)
    maximum = np.finfo(float).max
    if case == "scalar_extreme":
        weights = maximum
    elif case == "complex_broadcast":
        palette = (maximum, np.nextafter(0.0, 1.0), 0, 1e-200, 1e200, 2.0**-40)
        weights = [[palette[index % len(palette)]] for index in range(size)]
    else:
        raise ValueError(f"unknown memory case {case!r}")
    source = Measurement(
        identity=key_for(("memory-probe", size, case)), quantity="test/memory",
        definition=key_for("definition/memory"), units="unitless",
        producer=key_for("producer/memory"), subjects=(_subject(),),
        axes=(rows, columns), values=data.put(values, ARRAYS), missing="present",
        valid_mask=None, inputs=manifest(()),
    )
    reduction = Reduction(
        (rows.name, columns.name), "weighted_mean", _parameters(data, {"weights": weights}), "fail"
    )
    before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    result = measurement_view(source, data).reduce(reduction)
    elapsed = time.perf_counter() - started
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    value = data.get(result.values, ARRAYS).item()
    assert value == 1.0 and result.missing == "present" and result.valid_mask is None
    return {
        "size": size, "case": case, "values_bytes": values.nbytes,
        "before_peak_kib": before_kib, "peak_rss_kib": peak_kib,
        "increment_kib": peak_kib - before_kib, "seconds": elapsed,
        "value": value, "missing": result.missing,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fresh-process measurement memory probe")
    parser.add_argument("--measure-memory", action="store_true", required=True)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--case", choices=("scalar_extreme", "complex_broadcast"), required=True)
    arguments = parser.parse_args()
    print(json.dumps(_measure_reduction_memory(arguments.size, arguments.case), sort_keys=True))
