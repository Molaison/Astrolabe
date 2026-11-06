from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "unified-gpu-acceptance"
    / "typed_v1_continuation.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "typed_v1_continuation",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cif(*, timestamp: str, coordinate: str = "1.000") -> str:
    return "\n".join(
        (
            "data_test",
            "_atom_site.Cartn_x 2.000",
            f"_atom_site.Cartn_y {coordinate}",
            "#",
            (
                '_ma_model_list.model_group_name '
                f'"AlphaFold-beta-20231127 (3.0.1 @ {timestamp})"'
            ),
            "#",
            "",
        )
    )


def test_timestamp_only_cif_difference_is_accepted(tmp_path: Path) -> None:
    module = _module()
    left = tmp_path / "left.cif"
    right = tmp_path / "right.cif"
    left.write_text(
        _cif(timestamp="2026-11-01 13:27:48"),
        encoding="utf-8",
    )
    right.write_text(
        _cif(timestamp="2026-11-01 13:28:50"),
        encoding="utf-8",
    )

    result = module._compare_cif_pair(left, right)

    assert result["all_other_cif_data_exact"] is True
    assert result["atom_ids_and_coordinates_exact"] is True
    assert result["left_sha256"] != result["right_sha256"]
    assert result["timestamp_differences"] == [
        {
            "line": 5,
            "left_timestamp": "2026-11-01 13:27:48",
            "right_timestamp": "2026-11-01 13:28:50",
        }
    ]


def test_non_timestamp_cif_difference_is_rejected(tmp_path: Path) -> None:
    module = _module()
    left = tmp_path / "left.cif"
    right = tmp_path / "right.cif"
    left.write_text(
        _cif(timestamp="2026-11-01 13:27:48", coordinate="1.000"),
        encoding="utf-8",
    )
    right.write_text(
        _cif(timestamp="2026-11-01 13:28:50", coordinate="1.001"),
        encoding="utf-8",
    )

    with pytest.raises(module.ValidationError, match="non-timestamp"):
        module._compare_cif_pair(left, right)


def test_model_name_without_timestamp_difference_is_rejected(
    tmp_path: Path,
) -> None:
    module = _module()
    left = tmp_path / "left.cif"
    right = tmp_path / "right.cif"
    left.write_text(
        _cif(timestamp="2026-11-01 13:27:48"),
        encoding="utf-8",
    )
    right.write_text(
        _cif(timestamp="2026-11-01 13:28:50").replace(
            "3.0.1",
            "3.0.2",
        ),
        encoding="utf-8",
    )

    with pytest.raises(module.ValidationError, match="outside the timestamp"):
        module._compare_cif_pair(left, right)
