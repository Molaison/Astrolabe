import os
import pickle

import pytest
from rdkit import Chem

from astrolabe.evaluators.boltz_ccd import BoltzCcdCache


def save(path, smiles):
    path.write_bytes(pickle.dumps({"X": Chem.MolFromSmiles(smiles)}))


def test_repeated_preparation_loads_once_and_isolates_molecule_edits(tmp_path, monkeypatch):
    path = tmp_path / "ccd.pkl"
    save(path, "CCO")
    calls = []
    original = pickle.load
    def load(stream):
        calls.append(stream.name)
        return original(stream)
    monkeypatch.setattr(pickle, "load", load)
    cache = BoltzCcdCache()
    first = cache.for_request(path)
    first["X"].GetAtomWithIdx(0).SetFormalCharge(1)
    second = cache.for_request(path)
    assert first["X"].GetAtomWithIdx(0).GetFormalCharge() == 1
    assert second["X"].GetAtomWithIdx(0).GetFormalCharge() == 0
    assert calls == [str(path)]
    assert second.get("missing") is None


def test_replaced_asset_invalidates_even_when_mtime_is_preserved(tmp_path):
    path = tmp_path / "ccd.pkl"
    save(path, "C")
    cache = BoltzCcdCache()
    previous = cache.for_request(path)
    stat = path.stat()
    replacement = tmp_path / "replacement.pkl"
    save(replacement, "N")
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(path)
    assert cache.for_request(path)["X"].GetAtomWithIdx(0).GetAtomicNum() == 7
    assert previous["X"].GetAtomWithIdx(0).GetAtomicNum() == 6


def test_failed_load_does_not_publish_a_cache_entry(tmp_path):
    path = tmp_path / "ccd.pkl"
    path.write_bytes(pickle.dumps([]))
    cache = BoltzCcdCache()
    with pytest.raises(ValueError, match="component mapping"):
        cache.for_request(path)
    save(path, "CO")
    assert cache.for_request(path)["X"].GetNumAtoms() == 2


def test_clear_releases_owner_but_keeps_existing_requests_usable(tmp_path):
    path = tmp_path / "ccd.pkl"
    save(path, "CCO")
    cache = BoltzCcdCache()
    request = cache.for_request(path)
    cache.clear()
    path.unlink()
    assert request["X"].GetNumAtoms() == 3
    with pytest.raises(FileNotFoundError):
        cache.for_request(path)

def test_native_boltz_parser_matches_fresh_ccd_and_preserves_conformers(tmp_path):
    from dataclasses import fields
    import numpy as np
    parser = pytest.importorskip("boltz.data.parse.schema")
    molecule = Chem.MolFromSmiles("CCO")
    conformer = Chem.Conformer(3)
    for index, (atom, coords) in enumerate(zip(
        molecule.GetAtoms(), ((0., 0., 0.), (1.4, 0., 0.), (2., 1.2, 0.))
    )):
        atom.SetProp("name", f"{atom.GetSymbol()}{index + 1}")
        atom.SetProp("leaving_atom", "0")
        conformer.SetAtomPosition(index, coords)
    conformer.SetProp("name", "Ideal")
    molecule.AddConformer(conformer)
    path = tmp_path / "native_ccd.pkl"
    old_properties = Chem.GetDefaultPickleProperties()
    try:
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        path.write_bytes(pickle.dumps({"X": molecule}))
    finally:
        Chem.SetDefaultPickleProperties(old_properties)

    schema = {"version": 1, "sequences": [{"ligand": {"id": "L", "ccd": "X"}}]}
    fresh = parser.parse_boltz_schema("native", schema, pickle.loads(path.read_bytes()))
    cache = BoltzCcdCache()
    first = cache.for_request(path)
    cached = parser.parse_boltz_schema("native", schema, first)
    assert len(cached.structure.atoms) == 3
    assert cached.record == fresh.record
    assert cached.sequences == fresh.sequences
    for field in fields(fresh.structure):
        np.testing.assert_equal(getattr(cached.structure, field.name),
                                getattr(fresh.structure, field.name))
    first["X"].GetConformer().SetAtomPosition(0, (99., 99., 99.))
    another = parser.parse_boltz_schema("native", schema, cache.for_request(path))
    for field in fields(fresh.structure):
        np.testing.assert_equal(getattr(another.structure, field.name),
                                getattr(fresh.structure, field.name))


def test_runner_shutdown_dispatch_releases_preparation_asset(tmp_path, monkeypatch):
    """Exercise the actual runner class without importing model-only packages."""
    import ast
    import __future__
    from pathlib import Path
    from astrolabe.evaluators.utils import shutdown_model_handles

    path = Path(__file__).parents[1] / "astrolabe/evaluators/wrappers/boltz.py"
    source = path.read_text()
    definition = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.ClassDef) and node.name == "Boltz1Runner")
    namespace = {}
    exec(compile(ast.get_source_segment(source, definition), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    runner = object.__new__(namespace["Boltz1Runner"])
    runner._ccd_cache = BoltzCcdCache()
    asset = tmp_path / "ccd.pkl"
    save(asset, "CCO")
    calls = []
    load = pickle.load
    def counting_load(stream):
        calls.append(stream.name)
        return load(stream)
    monkeypatch.setattr(pickle, "load", counting_load)
    request = runner._ccd_cache.for_request(asset)
    runner._ccd_cache.for_request(asset)
    shutdown_model_handles(runner)
    runner._ccd_cache.for_request(asset)
    assert calls == [str(asset), str(asset)]
    assert request["X"].GetNumAtoms() == 3
