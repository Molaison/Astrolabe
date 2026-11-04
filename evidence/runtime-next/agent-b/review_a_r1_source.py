"""B's narrow independent source-equivalence guard for exact R1 commits.

Undo only enumerated, asserted import/dependency wiring changes in the candidate,
then compare complete module ASTs (including all math, decorators and defaults).
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


BEFORE = "cf62f856cf4dfde861a7b0d2589aa843ea53f79e"
AFTER = "605bad02f5c71083a95142f03bebc0db24a27d84"
LOCAL = {
    "interaction": {
        "_ca_contact_pairs": "cKDTree", "_sc_contact_pairs": "cKDTree",
        "_interface_residues_from_structure_model": "cKDTree",
        "_alpha_shape_kept_simplex_indices": "Delaunay, QhullError",
        "_build_ligand_environment_pose": "cKDTree",
        "find_interface_residues_by_ca": "cKDTree",
        "get_hotspot_interaction_number": "cKDTree",
    },
    "repeat_symmetry": {"_residue_pair_contact_count": "cKDTree",
                        "_minimum_interchain_atom_distance": "cKDTree"},
}


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = {"before": BEFORE, "after": AFTER, "modules": {}}
    for module in ["interaction", "repeat_symmetry", "design_screening"]:
        path = f"astrolabe/losses/{module}.py"
        source = {label: subprocess.check_output(["git", "show", f"{commit}:{path}"],
                  cwd=args.repo, text=True) for label, commit in [("before", BEFORE), ("after", AFTER)]}
        before, after = ast.parse(source["before"]), ast.parse(source["after"])
        original_functions = {n.name: n for n in before.body if isinstance(n, ast.FunctionDef)}
        functions = {n.name: n for n in after.body if isinstance(n, ast.FunctionDef)}
        for name, imported in LOCAL.get(module, {}).items():
            expected = ast.parse(f"from scipy.spatial import {imported}").body[0]
            assert dump(functions[name].body[0]) == dump(expected), (path, name)
            del functions[name].body[0]
        if module == "interaction":
            typing_node = next(n for n in after.body if isinstance(n, ast.ImportFrom) and n.module == "typing")
            assert typing_node.names[0].name == "TYPE_CHECKING"
            del typing_node.names[0]
            type_block = next(n for n in after.body if isinstance(n, ast.If)
                              and isinstance(n.test, ast.Name) and n.test.id == "TYPE_CHECKING")
            assert dump(type_block) == dump(ast.parse("if TYPE_CHECKING:\n    from scipy.spatial import Delaunay").body[0])
            index = after.body.index(type_block)
            after.body[index] = next(n for n in before.body if isinstance(n, ast.ImportFrom) and n.module == "scipy.spatial")
            function = functions["_alpha_shape_kept_simplex_indices"]
            assert isinstance(function.returns, ast.Constant)
            assert function.returns.value == "tuple[Delaunay, set[int], float]"
            function.returns = ast.parse(function.returns.value, mode="eval").body
        elif module == "repeat_symmetry":
            original_import = next(n for n in before.body if isinstance(n, ast.ImportFrom) and n.module == "scipy.spatial")
            after.body.insert(before.body.index(original_import), original_import)
        else:
            expected_helper = ast.parse('''def _require_ckdtree():
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise ImportError("scipy is required for BindCraft-aligned clash/interface-residue losses.") from exc
    return cKDTree
''').body[0]
            assert dump(functions["_require_ckdtree"]) == dump(expected_helper)
            helper_index = after.body.index(functions["_require_ckdtree"])
            original_index = before.body.index(original_functions["_require_ckdtree"])
            original_try = before.body[original_index - 1]
            assert isinstance(original_try, ast.Try)
            assert isinstance(original_try.body[0], ast.ImportFrom)
            assert original_try.body[0].module == "scipy.spatial"
            after.body[helper_index:helper_index + 1] = [original_try, original_functions["_require_ckdtree"]]
            for name in ["halludesign_key_res_plddt", "_calculate_bindcraft_clash_score", "_bindcraft_hotspot_residues"]:
                function = functions[name]
                position = 1 if isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) else 0
                expected = ast.parse("cKDTree = _require_ckdtree()").body[0]
                assert dump(function.body[position]) == dump(expected), name
                function.body[position] = ast.Expr(value=function.body[position].value)
        assert dump(before) == dump(after), f"non-import/non-enumerated AST difference: {path}"
        data["modules"][path] = {"equivalent_after_exact_import_changes_reversed": True,
            "source_sha256": {label: hashlib.sha256(text.encode()).hexdigest() for label, text in source.items()},
            "normalized_ast_sha256": hashlib.sha256(dump(before).encode()).hexdigest()}
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
