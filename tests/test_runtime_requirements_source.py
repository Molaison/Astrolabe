"""Source scanning must preserve dependency inference without importing models."""
import ast
from pathlib import Path

import pytest

from astrolabe.runtime_requirements import _source_segment


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("body", [
    "def café(): return 'pyrosetta.😀'  # not part of node",
    "def café():\n    text = 'a\fb\vc\u2028d'\n    return text",
    "def café(\n    x='😀',\n):\n    return ('pyrosetta.', x)",
])
def test_source_segment_matches_ast_utf8_and_line_boundaries(newline, body):
    source = (body + "\n").replace("\n", newline)
    node = ast.parse(source).body[0]
    assert _source_segment(source.encode().splitlines(keepends=True), node) == (
        ast.get_source_segment(source, node)
    )


def test_source_segment_matches_ast_for_registered_loss_corpus():
    """Exercise real multiline functions; dependency heuristics see identical text."""
    losses = Path(__file__).resolve().parents[1] / "astrolabe" / "losses"
    checked = 0
    for path in sorted(losses.glob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        if "pyrosetta" not in source and ".rosetta." not in source:
            continue
        lines = source.encode().splitlines(keepends=True)
        for node in ast.parse(source).body:
            if not isinstance(node, ast.FunctionDef) or not node.decorator_list:
                continue
            assert _source_segment(lines, node) == ast.get_source_segment(source, node)
            checked += 1
    assert checked > 0
