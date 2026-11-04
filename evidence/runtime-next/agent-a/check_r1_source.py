"""Narrow source guard: imports/helper wiring may move; numerical bodies may not."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE = "cf62f856cf4dfde861a7b0d2589aa843ea53f79e"
FILES = ["astrolabe/losses/interaction.py", "astrolabe/losses/repeat_symmetry.py",
         "astrolabe/losses/design_screening.py"]


class NormalizeImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module in ("scipy.spatial", "typing") else node

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "_require_ckdtree":
            return None
        if node.name == "_alpha_shape_kept_simplex_indices" and isinstance(node.returns, ast.Constant):
            node.returns = ast.parse(node.returns.value, mode="eval").body
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "cKDTree" and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "_require_ckdtree"):
            return ast.Expr(value=node.value)
        return self.generic_visit(node)

    def visit_Try(self, node):
        if (len(node.body) == 1 and isinstance(node.body[0], ast.ImportFrom)
                and node.body[0].module == "scipy.spatial"):
            return None
        return self.generic_visit(node)


checks, sources = {}, {}
for name in FILES:
    before = subprocess.check_output(["git", "show", BASE + ":" + name], text=True)
    after = Path(name).read_text()
    checks[name] = ast.dump(NormalizeImports().visit(ast.parse(before))) == ast.dump(
        NormalizeImports().visit(ast.parse(after)))
    sources[name] = {"before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                     "after_sha256": hashlib.sha256(after.encode()).hexdigest()}
assert all(checks.values()), checks
print(json.dumps({"baseline": BASE, "unchanged_math_and_registration": checks, "sources": sources,
                  "scope": "import placement, TYPE_CHECKING and quoted private annotation, and private dependency-helper wiring are normalized; helper error behavior is covered separately by tests"}, indent=2))
