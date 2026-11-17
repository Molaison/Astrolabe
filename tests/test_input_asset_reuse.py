from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from astrolabe.core.folding_input import Input, ProteinChain


def _input(sequence_b="ACDEFA", seed=17):
    return Input(name="AB", rng_seeds=[seed], chains=[
        ProteinChain(id=chain, sequence=sequence, ptms=[], templates=[],
                     unpaired_msa=f">query\n{sequence}\n>homologue\nAC-EFG\n", paired_msa="")
        for chain, sequence in (("A", "ACDEFG"), ("B", sequence_b))],
        bonded_atom_pairs=None, user_ccd=None)


def test_protenix_only_rewrites_changed_chain_assets(tmp_path, monkeypatch):
    from astrolabe.evaluators.wrappers import protenix
    from astrolabe.evaluators import input_assets
    monkeypatch.setattr(protenix, "global_config", SimpleNamespace(out_dir=tmp_path))
    first = protenix.from_af3_to_protenix(_input(), "AB", use_constraints=False)
    paths = [Path(row["proteinChain"][field])
             for row in first["sequences"] for field in ("unpairedMsaPath", "pairedMsaPath")]
    for path in paths:
        os.utime(path, ns=(1234567890000000000, 1234567890000000000))
    replaced = []
    replace = input_assets.os.replace
    def record(source, destination):
        replaced.append(Path(destination))
        return replace(source, destination)
    monkeypatch.setattr(input_assets.os, "replace", record)

    second = protenix.from_af3_to_protenix(_input("ACDKFA"), "AB", use_constraints=False)
    assert replaced == paths[2:]
    assert all(path.stat().st_mtime_ns == 1234567890000000000 for path in paths[:2])
    assert second["sequences"][1]["proteinChain"]["sequence"] == "ACDKFA"
    assert "ACDKFA" in paths[2].read_text()
    assert "ACDKFA" in paths[3].read_text()


def test_mol1_reuses_immutable_msa_without_stale_jobs(tmp_path, monkeypatch):
    from astrolabe.evaluators import mol1
    state = SimpleNamespace(step=0, binding_site=None, dynamic_binding_site=None)
    monkeypatch.setattr(mol1, "global_config", state)
    evaluator = object.__new__(mol1.Mol1Evaluator)
    evaluator.output_dir = tmp_path
    evaluator.model_runner = SimpleNamespace(settings={"use_msa": True, "constraints": {}})

    first = evaluator.prepare_input({"AB": _input()}, None)["AB"]
    old_paths = [Path(row["proteinChain"]["unpairedMsaPath"]) for row in first.job["sequences"]]
    for path in old_paths:
        os.utime(path, ns=(1234567890000000000, 1234567890000000000))
    old_contents = [path.read_text() for path in old_paths]
    state.step = 1
    second = evaluator.prepare_input({"AB": _input("ACDKFA", seed=18)}, None)["AB"]
    new_paths = [Path(row["proteinChain"]["unpairedMsaPath"]) for row in second.job["sequences"]]

    assert old_paths[0] == new_paths[0]
    assert old_paths[1] != new_paths[1]
    assert old_paths[0].stat().st_mtime_ns == 1234567890000000000
    assert [path.read_text() for path in old_paths] == old_contents
    assert second.job["modelSeeds"] == [18]
    assert second.job["sequences"][1]["proteinChain"]["sequence"] == "ACDKFA"
    assert json.loads(first.json_path.read_text())[0] == first.job
    assert json.loads(second.json_path.read_text())[0] == second.job
    assert len(list((tmp_path / "inputs/msa").glob("*.a3m"))) == 3
