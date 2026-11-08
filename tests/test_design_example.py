"""The published CPU composition example is executable and resumable."""
import json
from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize("mode", ["trajectory", "beam", "population"])
def test_sequence_example_policies_keep_fixed_state_and_resume(tmp_path, mode):
    example = Path(__file__).resolve().parents[1] / "examples/02_composable_sequence_search.py"
    run_example = runpy.run_path(str(example))["run_example"]
    first = run_example(tmp_path, mode)
    assert first.selected
    for selected in first.selected:
        values = {item.variable: item.value for item in selected.candidate.assignments}
        assert values["shared"].tokens[0] == "G"
        assert values["fixed_tail"].tokens == tuple("GA")
        assert selected.evidence.measurements[0].quantity == "composition.A"
    repeated = run_example(tmp_path, mode)
    assert repeated.selected == first.selected
    rows = json.loads(first.manifest.selected_summary.read_text())
    assert len(rows) == len(first.selected)
    assert set(rows[0]) == {"candidate", "objective", "gates", "evidence"}
