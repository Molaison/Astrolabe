from __future__ import annotations

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
