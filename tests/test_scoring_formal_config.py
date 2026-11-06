"""Direct value contract; CLI integration is exercised after the parent hook lands."""

import json
from pathlib import Path

import pytest
import yaml

from astrolabe.scoring_startup import validate_scoring_settings


ROOT = Path(__file__).parent / "fixtures/formal_config/scoring"
CASES = json.loads((ROOT / "manifest.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_explicit_scoring_config(case):
    config = yaml.safe_load((ROOT / case["file"]).read_text())
    family = "loss" if "probe" in config["loss"]["loss_terms"] else "predocker"
    section = config[family][family + "_terms"]
    assert list(section) == ["probe"]
    options = section["probe"]
    path = family + "." + family + "_terms.probe"
    errors = validate_scoring_settings(family, options["_type"], options,
                                        path=path, check_paths=True)
    if case["expected_error"] is None:
        assert errors == []
    else:
        assert len(errors) == 1, errors
        assert errors[0].startswith(case["expected_error"] + ":")
    for dotted in case["parameters"]:
        assert dotted.startswith(path + ".")


def test_unrelated_family_and_kind_are_not_given_scientific_limits():
    assert validate_scoring_settings("evaluator", "sequence", {"num_samples": 0},
                                     path="evaluator.check", check_paths=True) == []
    assert validate_scoring_settings("loss", "custom_loss", {"cutoff": -1},
                                     path="loss.custom", check_paths=True) == []


def test_enormous_integer_is_a_field_error_not_an_overflow():
    huge = 10 ** 1000
    assert validate_scoring_settings("loss", "interface_holes",
                                     {"interface_distance": huge},
                                     path="loss.probe", check_paths=False) == [
                                         "loss.probe.interface_distance: expected a finite number"]
    assert validate_scoring_settings("predocker", "haddock3_nucleic",
                                     {"restraint_distance": [2, 4, huge]},
                                     path="predocker.probe", check_paths=False) == [
                                         "predocker.probe.restraint_distance: expected three ordered positive distances"]
