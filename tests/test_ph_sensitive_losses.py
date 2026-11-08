from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from astrolabe.losses import ph_sensitive as ph
from astrolabe.losses.base import LossEngine, loss_registry


PUBLISHED_WEIGHTS = [1, 3, 2, 6, 3, 9, 6, 18, 1, 1, 3, 2, 2, 6, 3, 3, 9, 6, 6, 18]


def test_example_overlay_adds_bounded_ph_objective_and_limits_metrics_to_complex():
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[1]
    task = OmegaConf.load(root / "config/benchmark_binder_v1_4.yaml")
    overlay = OmegaConf.load(root / "examples/01_ph_sensitive_design.overlay.yaml")
    merged = OmegaConf.merge(task, overlay)
    assert merged.stages.default == task.stages.default
    expression = merged.stages.ligand.weighted_score_expression
    assert "iptm - 0.3 * sigmoid((binder_rog - globin_rog)/5.0)" in expression
    assert "min(binder_loop_frac,1) * 0.6" in expression
    assert "0.2 * (2 * sigmoid(ph_release / 5.0) - 1)" in expression
    assert merged.stages.ligand.acceptance_gate is None
    assert list(merged.stages.ligand.mutators) == ["ph_design"]
    for term in overlay.loss.loss_terms.values():
        assert list(term.oligomer_specific) == ["AB"]
    assert merged.mutator.mutator_terms.ph_design.criteria.dep_map["HIS-P"] == ["HIS-S"]
    assert merged.mutator.mutator_terms.ph_design.criteria.center_types == ["HIS-S"]
    assert merged.mutator.mutator_terms.ph_design.criteria.dep_map["HIS-S"] == ["HIS-P"]
    engine = LossEngine(OmegaConf.to_container(merged.loss), validate_config=False)
    values = {"iptm": 0.7, "binder_rog": 12.0, "binder_loop_frac": 0.2,
              "A_length": 85, "ph_release": 0.0}
    neutral = engine.weighted_score(values, expression)
    positive = engine.weighted_score({**values, "ph_release": 20.0}, expression)
    negative = engine.weighted_score({**values, "ph_release": -20.0}, expression)
    assert neutral < positive < neutral + 0.2
    assert neutral - 0.2 < negative < neutral


def test_published_twenty_term_formula():
    assert list(ph.PH_SCORE_WEIGHTS.values()) == PUBLISHED_WEIGHTS
    for name, weight in ph.PH_SCORE_WEIGHTS.items():
        counts = dict.fromkeys(ph.PH_SCORE_WEIGHTS, 0)
        counts[name] = 2
        assert ph.ph_score_from_counts(counts) == 2 * weight
    assert ph.ph_score_from_counts(dict.fromkeys(ph.PH_SCORE_WEIGHTS, 1)) == 108.0


@pytest.mark.parametrize("bad", [-1, 0.5, float("nan"), float("inf")])
def test_ph_score_rejects_invalid_counts(bad):
    counts = dict.fromkeys(ph.PH_SCORE_WEIGHTS, 0)
    counts["low_pH_surf_D"] = bad
    with pytest.raises(ValueError, match="Invalid histidine count"):
        ph.ph_score_from_counts(counts)


def test_ph_score_requires_all_terms():
    with pytest.raises(ValueError, match="20 published"):
        ph.ph_score_from_counts({"low_pH_surf_D": 1})


def test_every_ph_loss_is_registered_and_infers_pyrosetta():
    from astrolabe.runtime_requirements import _pyrosetta_loss_types_from_source

    names = {
        "ph_score", "ph_dddg_elec", "ph_histidine_hbond_energy",
        "ph_histidine_hbond_depth", "ph_histidine_pair_energy", "ph_histidine_cation_contacts",
        "ph_histidine_cation_acceptors",
        "ph_positive_charge_count",
    }
    assert names <= set(loss_registry.valid_losses)
    _pyrosetta_loss_types_from_source.cache_clear()
    assert names <= _pyrosetta_loss_types_from_source()
    engine = LossEngine({"loss_terms": {
        "acid_release": {"_type": "ph_dddg_elec", "binder_chain": "A", "target_chain": "B"},
    }}, validate_config=False)
    engine.setup("acid_release", None, None, validate_config=False)
    assert engine.loss_specs["acid_release"][0] == "ph_dddg_elec"


@pytest.fixture(scope="module")
def rosetta():
    pytest.importorskip("pyrosetta")
    return ph._pyrosetta()


@pytest.fixture
def network_pose(rosetta):
    """Construct a real Lys donor directed at the accepting His ring nitrogen."""
    pr = rosetta
    pose = pr.pose_from_sequence("AHA")
    pr.rosetta.core.pose.append_pose_to_pose(pose, pr.pose_from_sequence("AKA"), True)
    info = pr.rosetta.core.pose.PDBInfo(pose)
    for index in range(1, 7):
        info.chain(index, "A" if index <= 3 else "B")
        info.number(index, (index - 1) % 3 + 1)
    pose.pdb_info(info)
    his, lys = pose.residue(2), pose.residue(5)
    acceptor = np.array(his.xyz("ND1"))
    outward = acceptor - (np.array(his.xyz("CG")) + np.array(his.xyz("CE1"))) / 2
    outward /= np.linalg.norm(outward)
    nz = lys.atom_index("NZ")
    hydrogen = next(i for i in range(lys.nheavyatoms() + 1, lys.natoms() + 1)
                    if lys.atom_base(i) == nz)
    origin = np.array(lys.xyz(nz))
    direction = np.array(lys.xyz(hydrogen)) - origin
    direction /= np.linalg.norm(direction)
    target = -outward
    cross = np.cross(direction, target)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    rotation = np.eye(3) + skew + (skew @ skew) / (1 + np.dot(direction, target))
    destination = acceptor + 2.8 * outward
    for index in range(4, 7):
        for atom in range(1, pose.residue(index).natoms() + 1):
            xyz = rotation @ (np.array(pose.residue(index).xyz(atom)) - origin) + destination
            pose.set_xyz(pr.rosetta.core.id.AtomID(atom, index),
                         pr.rosetta.numeric.xyzVector_double_t(*xyz))
    return pose


@pytest.fixture
def network_scores(network_pose, tmp_path):
    path = tmp_path / "network.pdb"
    network_pose.dump_pdb(str(path))
    return SimpleNamespace(names=["AB"], complex_paths={"AB": path})


def test_real_network_hbond_energy_and_depth(network_scores):
    energy = ph.ph_histidine_hbond_energy(network_scores, 2, 5)["AB"]
    assert energy < -0.5
    depth = ph.ph_histidine_hbond_depth(network_scores, 2, 5)["AB"]
    assert np.isfinite(depth) and depth > 0


def test_real_glycine_double_mutant_cycle(network_scores, rosetta):
    pose = ph._load_pose(network_scores.complex_paths["AB"])
    sf = rosetta.create_score_function("ref2015")
    expected = 0.0
    for mutated, sign in [((), 1), ((2,), -1), ((5,), -1), ((2, 5), 1)]:
        clone = pose.clone()
        for index in mutated:
            rosetta.rosetta.protocols.simple_moves.MutateResidue(index, "GLY").apply(clone)
        expected += sign * sf(clone)
    observed = ph.ph_histidine_pair_energy(network_scores, 2, 5)["AB"]
    assert observed == pytest.approx(expected)


def test_real_states_preserve_protonation_and_global_options(network_scores, rosetta):
    opts = rosetta.rosetta.basic.options
    before = (opts.get_boolean_option("pH:pH_mode"), opts.get_real_option("pH:value_pH"))
    low, high = ph._ph_states(network_scores.complex_paths["AB"])
    assert low.residue(2).type().base_name() == "HIS_P"
    assert high.residue(2).type().base_name() in {"HIS", "HIS_D"}
    assert (opts.get_boolean_option("pH:pH_mode"), opts.get_real_option("pH:value_pH")) == before
    # Consumers cannot mutate a cached pose shared by other losses.
    rosetta.rosetta.protocols.simple_moves.MutateResidue(2, "GLY").apply(low)
    assert ph._ph_states(network_scores.complex_paths["AB"])[0].residue(2).name3() == "HIS"


def test_real_ph_formula_and_electrostatic_binding_difference(network_scores):
    assert np.isfinite(ph.ph_score(network_scores, "A", "B")["AB"])
    low, high = ph._ph_states(network_scores.complex_paths["AB"])
    binder, target = ph._interface_residues(high, "AB", "A", "B")
    delta_low = ph._electrostatic_binding_energy(low, binder, target)
    delta_high = ph._electrostatic_binding_energy(high, binder, target)
    assert ph.ph_dddg_elec(network_scores, "A", "B")["AB"] == pytest.approx(delta_low - delta_high)
    assert abs(delta_low - delta_high) > 1e-3
    assert np.isfinite(ph.ph_dddg_elec(network_scores, "A", "B", pack_separated=True)["AB"])
    assert ph.ph_histidine_cation_acceptors(network_scores, "A", "B")["AB"] in {0.0, 1.0}


def test_real_histidine_pattern_counts_once_per_residue(network_pose):
    counts = ph._histidine_pattern_counts(network_pose, {1, 2, 3}, {4, 5, 6}, "high_pH", set(), 0.0)
    assert counts["high_pH_surf_A"] == 1
    assert sum(counts.values()) == 1


def test_real_cation_contacts_enumerates_library_rotamers(network_scores):
    assert ph.ph_histidine_cation_contacts(network_scores, "A", "B", 4.0)["AB"] == 1.0
    assert ph.ph_histidine_cation_contacts(network_scores, "A", "B", 4.0, atom_mode="sidechain_heavy")["AB"] == 1.0


def test_positive_charge_count_distinguishes_histidine_microstates(network_scores):
    assert ph.ph_positive_charge_count(network_scores, "A", phase="high")["AB"] == 0.0
    assert ph.ph_positive_charge_count(network_scores, "A", phase="low")["AB"] == 1.0
    assert ph.ph_positive_charge_count(network_scores, "B", phase="high")["AB"] == 1.0


def test_missing_bond_is_zero_and_invalid_pair_is_not_silently_scored(network_scores, rosetta, tmp_path):
    with pytest.raises(ValueError, match="histidine"):
        ph.ph_histidine_hbond_energy(network_scores, 1, 5)
    with pytest.raises(ValueError, match="distinct"):
        ph.ph_histidine_pair_energy(network_scores, 2, 2)
    with pytest.raises(ValueError, match="disjoint"):
        ph.ph_score(network_scores, "A", "A")
    with pytest.raises(ValueError, match="does not match"):
        ph.ph_score(network_scores, "C", "B")
    with pytest.raises(ValueError, match="positive and finite"):
        ph.ph_histidine_cation_contacts(network_scores, "A", "B", 0)
    pose = ph._load_pose(network_scores.complex_paths["AB"])
    shift = rosetta.rosetta.numeric.xyzVector_double_t(50, 0, 0)
    for index in range(4, 7):
        for atom in range(1, pose.residue(index).natoms() + 1):
            pose.set_xyz(rosetta.rosetta.core.id.AtomID(atom, index), pose.residue(index).xyz(atom) + shift)
    path = tmp_path / "no_bond.pdb"
    pose.dump_pdb(str(path))
    scores = SimpleNamespace(names=["AB"], complex_paths={"AB": path})
    assert ph.ph_histidine_hbond_energy(scores, 2, 5)["AB"] == 0.0
    assert ph.ph_histidine_hbond_depth(scores, 2, 5)["AB"] == 0.0


def test_loss_engine_executes_named_ph_terms_and_gate(network_scores):
    engine = LossEngine({"loss_terms": {
        "network": {"_type": "ph_histidine_hbond_energy", "histidine_residue": 2, "partner_residue": 5},
        "release": {"_type": "ph_dddg_elec", "binder_chain": "A", "target_chain": "B"},
    }}, validate_config=False)
    engine.setup("release - network", None, "network < -0.5", validate_config=False)
    per_oligo, terms = engine._extract_scores_from_specs(network_scores, engine.loss_specs)
    assert per_oligo["AB"]["network"] < -0.5
    assert np.isfinite(terms["release"])
    assert engine.eval_condition(dict(terms), "network < -0.5") is True


def test_interface_backbone_rmsd_is_global_rigid_body_invariant(network_scores, network_pose, rosetta, tmp_path):
    from astrolabe.losses.structure import interface_backbone_rmsd

    reference = str(network_scores.complex_paths["AB"])
    assert interface_backbone_rmsd(network_scores, reference, "A", "B")["AB"] < 1e-6
    shifted = network_pose.clone()
    shift = rosetta.rosetta.numeric.xyzVector_double_t(10, -7, 12)
    for index in range(1, shifted.total_residue() + 1):
        # Numbering is irrelevant: the correspondence is chain-wise sequence position.
        shifted.pdb_info().number(index, 100 + index)
        for atom in range(1, shifted.residue(index).natoms() + 1):
            shifted.set_xyz(rosetta.rosetta.core.id.AtomID(atom, index), shifted.residue(index).xyz(atom) + shift)
    path = tmp_path / "shifted.pdb"
    shifted.dump_pdb(str(path))
    scores = SimpleNamespace(names=["AB"], complex_paths={"AB": path})
    assert interface_backbone_rmsd(scores, reference, "A", "B")["AB"] < 0.002
    # A displacement of the target relative to the binder is not a global transform.
    for index in range(4, 7):
        for atom in range(1, shifted.residue(index).natoms() + 1):
            shifted.set_xyz(rosetta.rosetta.core.id.AtomID(atom, index), shifted.residue(index).xyz(atom) + shift)
    shifted.dump_pdb(str(path))
    assert interface_backbone_rmsd(scores, reference, "A", "B")["AB"] > 0.1


def test_interface_backbone_rmsd_rejects_missing_interface_and_coverage(network_scores, tmp_path):
    from Bio.PDB import PDBIO
    from astrolabe.losses.design_screening import _load_biopython_structure
    from astrolabe.losses.structure import interface_backbone_rmsd

    reference = str(network_scores.complex_paths["AB"])
    with pytest.raises(ValueError, match="No reference-interface"):
        interface_backbone_rmsd(network_scores, reference, "A", "B", 0.001)
    structure = _load_biopython_structure(network_scores.complex_paths["AB"])
    chain = next(structure.get_models())["B"]
    chain.detach_child(next(iter(chain)).id)
    writer = PDBIO()
    writer.set_structure(structure)
    path = tmp_path / "missing.pdb"
    writer.save(str(path))
    scores = SimpleNamespace(names=["AB"], complex_paths={"AB": path})
    with pytest.raises(ValueError, match="unequal residue coverage"):
        interface_backbone_rmsd(scores, reference, "A", "B")
