import ast
from pathlib import Path

from astrolabe.backend_catalog import BACKENDS, get_backend
from astrolabe.evaluators.utils import EvaluatorRegistry
from astrolabe.software_catalog import SOFTWARE


ROOT = Path(__file__).resolve().parents[1]


def registered_names(path, decorator):
    tree = ast.parse((ROOT / path).read_text())
    return {node.args[0].value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == decorator and node.args
            and isinstance(node.args[0], ast.Constant)}


def test_every_registered_model_has_an_execution_disposition():
    for component, path, decorator in (
        ("evaluator", "astrolabe/evaluators/utils.py", "register_evaluator_factory"),
        ("mutator", "astrolabe/mutators/base.py", "register_mutator_factory"),
    ):
        observed = registered_names(path, decorator)
        if component == "evaluator":
            observed = set(EvaluatorRegistry.list_available())
        declared = {alias for record in BACKENDS if record.component == component
                    for alias in record.aliases}
        assert declared == observed
    tree = ast.parse((ROOT / "astrolabe/structure_processors/base.py").read_text())
    mapping = next(node.value for node in tree.body if isinstance(node, ast.AnnAssign)
                   and isinstance(node.target, ast.Name)
                   and node.target.id == "_PREDOCKER_TYPE_TO_MODULE")
    observed = {key.value for key in mapping.keys}
    declared = {alias for record in BACKENDS if record.component == "processor"
                for alias in record.aliases}
    assert declared == observed
    assert {item for record in BACKENDS for item in record.software} <= set(SOFTWARE)


def test_model_aliases_preserve_their_family_and_execution_claim():
    assert get_backend("dplm").component == "mutator"
    assert get_backend("dplm", component="initializer").name == "dplm_initializer"
    assert get_backend("ProteinMPNN").name == "ligandmpnn"
    assert get_backend("boltz").execution == "daemon"
    assert "Boltz 1" in get_backend("boltz").detail
    assert "CPU" in get_backend("chai1").detail
    assert get_backend("RAPiDock").name != get_backend("symmetry_canonicalizer").name
