from scripts.validation.runtime_structure_parity import compare


def test_observed_direct_coordinate_error_does_not_excuse_wrong_atom_or_larger_change(tmp_path):
    def trial(index, mode, x, atom="CA"):
        path = tmp_path / f"{index}.pdb"
        path.write_text(f"ATOM      1 {atom:^4s} ALA A   1    {x:8.3f}{0.:8.3f}{0.:8.3f}  1.00 20.00           C\n")
        return {"run": index, "mode": mode, "valid_workload": True,
                "structure_artifacts": {"(0,17):structures/step1.pdb": str(path)}}
    direct = [trial(1, "direct", 1.), trial(2, "direct", 1.002)]
    assert compare(direct + [trial(3, "resident_serial", 1.001)])["certified"]
    result = compare(direct + [trial(4, "resident_adaptive", 1.004)])
    assert not result["certified"] and result["differences"][0]["kind"] == "coordinates"
    result = compare(direct + [trial(5, "resident_adaptive", 1.001, "CB")])
    assert not result["certified"] and result["differences"][0]["kind"] == "atom_identity"
