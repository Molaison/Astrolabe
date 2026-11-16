"""Custom CCD isolation at real Astrolabe call sites, with a native-loader double."""
from collections.abc import Mapping
from contextlib import nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from astrolabe.core import global_config, protomer_config_parser, protomer_utils


CUSTOM = '{"ZZZ": {"_chem_comp.one_letter_code": ["G"]}}'


@pytest.fixture
def native_ccd(monkeypatch):
    """Model the pinned AF3 constructor's shared dictionary, not its chemistry."""
    shared = {"ZZZ": {"_chem_comp.one_letter_code": ("A",)}}

    def module(name, **values):
        result = ModuleType(name)
        result.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, result)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            if parent in sys.modules:
                monkeypatch.setattr(sys.modules[parent], child, result, raising=False)
        return result

    class Ccd(Mapping):
        def __init__(self, ccd_pickle_path=None, user_ccd=None):
            self._ccd_pickle_path = ccd_pickle_path
            self._dict = shared
            if user_ccd is not None:
                if not user_ccd:
                    raise ValueError("User CCD cannot be an empty string.")
                self._dict.update({name: {key: tuple(value) for key, value in entry.items()}
                                   for name, entry in json.loads(user_ccd).items()})

        def __getitem__(self, key): return self._dict[key]
        def __iter__(self): return iter(self._dict)
        def __len__(self): return len(self._dict)

    module("alphafold3")
    module("alphafold3.constants")
    components = module("alphafold3.constants.chemical_components", Ccd=Ccd,
                        cif_dict=SimpleNamespace(parse_multi_data_cif=json.loads))
    protomer_utils.cached_ccd.cache_clear()
    global_config.reset()
    try:
        yield SimpleNamespace(components=components, shared=shared, module=module)
    finally:
        protomer_utils.cached_ccd.cache_clear()
        global_config.reset()


@pytest.mark.parametrize("factory", [protomer_utils.cached_ccd, protomer_config_parser.cached_ccd])
def test_ptm_and_mutator_ccd_factories_preserve_default_components(native_ccd, factory):
    original = factory()
    custom = factory(user_ccd=CUSTOM)
    assert custom["ZZZ"]["_chem_comp.one_letter_code"] == ("G",)
    assert original["ZZZ"]["_chem_comp.one_letter_code"] == ("A",)
    assert factory()["ZZZ"]["_chem_comp.one_letter_code"] == ("A",)


def test_isolated_ccd_preserves_pickle_path_and_rejects_bad_overlay_without_mutation(native_ccd):
    from astrolabe.utils.af3_ccd import create_isolated_ccd

    base = create_isolated_ccd(native_ccd.components, "chosen.pickle")
    overlay = protomer_config_parser.cached_ccd("chosen.pickle", CUSTOM)
    assert base._ccd_pickle_path == overlay._ccd_pickle_path == "chosen.pickle"
    assert base._dict is native_ccd.shared and overlay._dict is not native_ccd.shared
    for invalid in ("", "malformed CIF"):
        with pytest.raises(ValueError):
            create_isolated_ccd(native_ccd.components, user_ccd=invalid)
    assert base["ZZZ"]["_chem_comp.one_letter_code"] == ("A",)


def test_custom_ptm_parsing_does_not_change_later_default_sequence(native_ccd):
    global_config.init_const("user_ccd", CUSTOM)
    global_config.init_const("nonstandard_aliases", {})
    assert protomer_config_parser.parse_ptm_sequence("(ZZZ)")[0] == "G"
    global_config.reset()
    global_config.init_const("user_ccd", None)
    global_config.init_const("nonstandard_aliases", {})
    assert protomer_config_parser.parse_ptm_sequence("(ZZZ)")[0] == "A"


def test_startup_token_count_keeps_each_inputs_ccd_isolated(native_ccd, monkeypatch):
    from run import calculate_token_numbers
    from astrolabe.utils import af3_compat

    monkeypatch.setattr(af3_compat, "to_af3_input", lambda value: value)
    monkeypatch.setattr(af3_compat, "allow_all_x_protein_chains_during_featurisation",
                        lambda _: nullcontext())
    module = native_ccd.module
    module("alphafold3.model")
    seen = []

    def tokenizer(layout, *, ccd, **kwargs):
        seen.append(ccd["ZZZ"]["_chem_comp.one_letter_code"][0])
        return SimpleNamespace(atom_name=["CA"]), None, None

    module("alphafold3.model.features", tokenizer=tokenizer)
    module("alphafold3.model.pipeline")
    module("alphafold3.model.pipeline.inter_chain_bonds",
           get_polymer_ligand_and_ligand_ligand_bonds=lambda *args, **kwargs: (None, None))
    module("alphafold3.model.pipeline.pipeline", WholePdbPipeline=SimpleNamespace(
        Config=lambda: SimpleNamespace(filter_clashes=False, filter_crystal_aids=False,
                                       drop_ligand_leaving_atoms=False,
                                       remove_nonsymmetric_bonds=False,
                                       max_atoms_per_token=24,
                                       flatten_non_standard_residues=True)))
    module("alphafold3.model.pipeline.structure_cleaning",
           clean_structure=lambda struct, **kwargs: (struct, {}),
           create_empty_output_struc_and_layout=lambda **kwargs: (None, None))
    inputs = {name: SimpleNamespace(name=name, user_ccd=custom,
                                   to_structure=lambda **kwargs: SimpleNamespace(num_chains=1))
              for name, custom in [("custom", CUSTOM), ("default", None)]}
    assert calculate_token_numbers(inputs) == {"custom": 1, "default": 1}
    assert seen == ["G", "A"]


def test_extra_residue_ccd_loading_does_not_modify_native_base(native_ccd, monkeypatch, tmp_path):
    from astrolabe.utils import extra_residues

    global_config.init_const("protomers", {})
    global_config.init_const("out_dir", tmp_path)
    global_config.init_const("user_ccd", CUSTOM)
    global_config.init_const("nonstandard_aliases", {
        "Z1": {"ccd_id": "ZZZ", "parent_aa": "A", "kind": "ptm"},
    })
    observed = []

    def generate(**kwargs):
        observed.append(kwargs["ccd_entry"]["_chem_comp.one_letter_code"])
        return tmp_path / "ZZZ.params"

    monkeypatch.setattr(extra_residues, "_get_runtime_rosetta_builtin_residue_names", lambda: set())
    monkeypatch.setattr(extra_residues, "_infer_nonstandard_alias_rosetta_properties",
                        lambda *args: ["PROTEIN"])
    monkeypatch.setattr(extra_residues, "generate_rosetta_polymer_params_for_ccd", generate)
    extra_residues.setup_extra_residues(extra_residue_params=[])
    assert observed == [("G",)]
    assert native_ccd.shared["ZZZ"]["_chem_comp.one_letter_code"] == ("A",)
