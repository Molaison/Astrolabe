from __future__ import annotations

import pytest

from scripts.validation.public_cli_recovery_root_step_validator import (
    is_allowed_root_step_difference,
)


ROOT_STEP = {
    "path": "$[0].step",
    "kind": "value",
    "left": None,
    "right": 0,
}


def test_exact_root_step_null_to_zero_is_allowed():
    assert is_allowed_root_step_difference([ROOT_STEP]) is True


@pytest.mark.parametrize(
    "differences",
    [
        [],
        [{**ROOT_STEP, "path": "$[1].step"}],
        [{**ROOT_STEP, "left": 1}],
        [{**ROOT_STEP, "right": False}],
        [{**ROOT_STEP, "right": 1}],
        [{**ROOT_STEP, "kind": "right-only"}],
        [
            ROOT_STEP,
            {
                "path": "$[0].score",
                "kind": "value",
                "left": 1.0,
                "right": 2.0,
            },
        ],
    ],
)
def test_any_other_metadata_difference_is_rejected(differences):
    assert is_allowed_root_step_difference(differences) is False
