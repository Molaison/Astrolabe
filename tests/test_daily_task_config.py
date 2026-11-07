"""Compatibility checks for daily-task references; CLI corpus is separate."""
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from astrolabe.startup_validation import StartupValidationError, validate_startup_config
from astrolabe.core.binding_site import parse_binding_site
from astrolabe.core.sequence_ranges import parse_ranges


FIXTURES = Path(__file__).parent / "fixtures/daily_task_config"


def config():
    return OmegaConf.load(FIXTURES / "valid_protein.yaml")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_binding_distance_must_be_finite_number(value):
    with pytest.raises(ValueError, match="distance"):
        parse_binding_site({"binder": "A", "target": "B:1", "max_distance": value})


def test_partial_site_map_and_chain_first_homomer_are_valid():
    cfg = config()
    cfg.seqs = {"A": "ACDE", "B": "ACDE", "C": "ACDE"}
    cfg.stages.default.oligo = ["AA", "AC"]
    cfg.binding_site = {"AA": {"binder": "A", "target": "chain:B:4"}}
    validate_startup_config(cfg, check_dependencies=False)


def test_random_length_does_not_get_sampled_or_assumed_fixed():
    cfg = config()
    cfg.seqs = {"A": "3-12", "B": "ACDE"}
    cfg.stages.default.oligo = ["AB"]
    cfg.binding_site = {"binder": "A:12", "target": "B"}
    validate_startup_config(cfg, check_dependencies=False)
    cfg.binding_site.binder = "A:13"
    with pytest.raises(StartupValidationError, match="binding_site.binder"):
        validate_startup_config(cfg, check_dependencies=False)


def test_pdb_numbering_is_not_sequence_position_numbering(tmp_path):
    path = tmp_path / "input.pdb"
    path.write_text("ATOM      1  CA  ALA A   0       0.000   0.000   0.000  1.00 10.00           C\n")
    cfg = config()
    cfg.named_files = {"input": str(path)}
    cfg.seqs = {"A": "input:A0"}
    cfg.stages.default.oligo = ["A"]
    validate_startup_config(cfg, check_dependencies=False)
    cfg.seqs.A = "input:A1"
    with pytest.raises(StartupValidationError, match="residues"):
        validate_startup_config(cfg, check_dependencies=False)


def test_reversed_structure_range_cannot_silently_select_whole_chain():
    assert parse_ranges("A0-2") == {"A": [0, 1, 2]}
    with pytest.raises(ValueError, match="end < start"):
        parse_ranges("A10-2")


def test_unselected_loss_reference_is_ignored():
    cfg = config()
    cfg.loss.loss_terms = {
        "inactive": {"_type": "hotspot_dsasa", "select_expression": "missing:1", "_for": "ZZ"}}
    cfg.stages.default.active_loss_terms = []
    validate_startup_config(cfg, check_dependencies=False)


def test_segment_alias_is_not_an_implicit_protomer_name():
    cfg = config()
    cfg.seqs = {"A": "ACDE@region"}
    cfg.stages.default.oligo = ["A"]
    cfg.loss.loss_terms = {"hotspot": {"_type": "hotspot_dsasa", "select_expression": "region:2"}}
    validate_startup_config(cfg, check_dependencies=False)
    cfg.loss.loss_terms.hotspot.select_expression = "A:2"
    with pytest.raises(StartupValidationError, match="unknown segment alias"):
        validate_startup_config(cfg, check_dependencies=False)


def test_file_segment_hotspot_uses_parsed_length_without_reopening(monkeypatch):
    from astrolabe.core.structure_parser import PDBParser
    cfg = OmegaConf.load(FIXTURES / "semantic.file_segment_hotspot_range.yaml")
    calls = []
    parse = PDBParser.parse
    def counted(self):
        calls.append(str(self.file_path))
        return parse(self)
    monkeypatch.setattr(PDBParser, "parse", counted)
    with pytest.raises(StartupValidationError, match="outside segment"):
        validate_startup_config(cfg, check_dependencies=False)
    assert len(calls) == 1
