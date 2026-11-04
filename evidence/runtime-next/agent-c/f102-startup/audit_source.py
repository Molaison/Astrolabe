#!/usr/bin/env python3
"""Read immutable git objects, never a moving integration worktree."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--git-repo", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    p = json.loads(a.protocol.read_text())
    heads = {"baseline": p["baseline_commit"], "candidate": p["candidate_commit"]}

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(a.git_repo), *argv])

    def source(head, path):
        return git("show", head + ":" + path)

    modules = {}
    for role, head in heads.items():
        files = git("ls-tree", "-r", "--name-only", head, "astrolabe/losses").decode().splitlines()
        modules[role] = {f: source(head, f) for f in files if f.endswith(".py")}
    assert set(modules["baseline"]) == set(modules["candidate"])
    bodies = []
    for role in heads:
        tree = ast.parse(modules[role]["astrolabe/losses/interaction.py"])
        bodies.append({n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)})
    bf, cf = bodies
    assert set(bf) == set(cf)
    unchanged, guards, other = [], [], []
    for name, bn in bf.items():
        cn = cf[name]
        if ast.dump(bn) == ast.dump(cn):
            unchanged.append(name)
            continue
        first = cn.body[0]
        if name != "_require_pyrosetta" and len(cn.body) == len(bn.body) + 1 and isinstance(first, ast.Expr) and isinstance(first.value, ast.Call) and isinstance(first.value.func, ast.Name) and first.value.func.id == "_require_pyrosetta" and not first.value.args and not first.value.keywords:
            cn.body = cn.body[1:]
            equal = ast.dump(bn) == ast.dump(cn)
            cn.body.insert(0, first)
            (guards if equal else other).append(name)
        else:
            other.append(name)
    registrations = []

    def registered(data):
        return {n.name: {"signature": ast.dump(n.args), "decorators": [ast.dump(d) for d in n.decorator_list]}
            for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and any(
                isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and isinstance(d.func.value, ast.Name)
                and d.func.value.id == "loss_registry" and d.func.attr == "register" for d in n.decorator_list)}

    for path in modules["baseline"]:
        before, after = [registered(modules[role][path]) for role in heads]
        assert before == after, path
        registrations.extend({"file": path, "function": name, **value} for name, value in before.items())
    assert other == ["_require_pyrosetta"] and len(guards) == 6
    report = {"status": "PASS", "scope": "immutable-source-only; no native execution or scientific workload",
        "heads": heads, "interaction_unchanged_functions": unchanged, "interaction_guard_only_functions": guards,
        "interaction_other_changed_functions": other, "all_direct_registered_signatures_decorators_exact": True,
        "direct_registered_definitions": registrations,
        "registry_limitation": "Static decorator definitions do not count dynamically registered aliases; installed registry must still be compared at runtime.",
        "loss_module_hashes": {role: {path: hashlib.sha256(data).hexdigest() for path, data in fs.items()} for role, fs in modules.items()},
        "protocol_sha256": hashlib.sha256(a.protocol.read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "heads": heads, "unchanged_functions": len(unchanged),
        "guard_only_functions": guards, "direct_registered_definitions": len(registrations)}, sort_keys=True))


if __name__ == "__main__":
    main()
