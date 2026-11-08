from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.losses.shape_complementarity import _median_shape, _trim


def _dot(x, y, buried):
    return SimpleNamespace(
        coor=SimpleNamespace(x=x, y=y, z=0.0),
        outnml=SimpleNamespace(x=1.0, y=0.0, z=0.0),
        buried=buried,
    )


def test_trimming_includes_exact_band_boundary():
    outside = np.nextafter(np.float32(1.5), np.float32(2))
    coords, _, indices = _trim([_dot(0, 0, False), _dot(1.5, 0, True),
                               _dot(outside, 0, True)])
    assert indices.tolist() == [2]
    assert coords[0, 0] == outside


def test_trimming_checks_float_boundary_beyond_double_radius():
    # The second component vanishes in float squared-distance addition.
    with pytest.raises(RuntimeError, match="No molecular surface dots remain"):
        _trim([_dot(0, 0, False), _dot(1.5, 0.0001, True)])


def test_buried_surface_without_accessible_points_is_retained():
    _, _, indices = _trim([_dot(0, 0, True), _dot(1, 0, True)])
    assert indices.tolist() == [0, 1]


def test_empty_surface_fails():
    with pytest.raises(RuntimeError, match="No molecular surface dots"):
        _trim([])


@pytest.mark.parametrize("farther_y", [0.0, 0.0001])
def test_nearest_neighbor_keeps_last_native_float_tie(farther_y):
    coords = np.array([[0, 0, 0]], dtype=np.float32)
    normals = np.array([[1, 0, 0]], dtype=np.float32)
    other = np.array([[1, 0, 0], [1, farther_y, 0]], dtype=np.float32)
    opposing = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)
    score, selected, _ = _median_shape(coords, normals, other, opposing)
    assert selected.tolist() == [1]
    assert 0.60 < score < 0.62
    reverse_score, reverse_selected, _ = _median_shape(coords, normals, other[::-1], opposing[::-1])
    assert reverse_selected.tolist() == [1]
    assert -0.62 < reverse_score < -0.60


def test_nearest_neighbor_keeps_float_underflow_tie():
    coords = np.zeros((1, 3), dtype=np.float32)
    normals = np.array([[1, 0, 0]], dtype=np.float32)
    other = np.array([[0, 0, 0], [1e-23, 0, 0]], dtype=np.float32)
    opposing = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)
    score, selected, _ = _median_shape(coords, normals, other, opposing)
    assert selected.tolist() == [1]
    assert score > 0.98


def test_single_opposing_dot_has_no_second_neighbor():
    coords = np.zeros((1, 3), dtype=np.float32)
    normals = np.array([[1, 0, 0]], dtype=np.float32)
    score, selected, _ = _median_shape(coords, normals, np.array([[1, 0, 0]], dtype=np.float32),
                                       -normals)
    assert selected.tolist() == [0]
    assert score > 0.60
