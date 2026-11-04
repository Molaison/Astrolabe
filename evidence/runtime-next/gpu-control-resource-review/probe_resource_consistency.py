"""CPU-only independent raw-resource consistency probe; no model/GPU execution."""
import copy
import importlib.util
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location("author_gpu_contract_fixtures", repo / "tests/test_gpu_mutator_control.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
control = module.control
protocol = json.loads(control.DEFAULT_PROTOCOL.read_text())
frozen, execution = module.synthetic_run.__wrapped__(output, protocol)
# Materialize the production measurement shape, then round-trip JSON so the
# duplicated observation dictionaries have the same independence as saved raw.
for row in execution["trials"]:
    row["measurement"]["process_release"] = copy.deepcopy(row["observation"])
baseline = control.aggregate(frozen, copy.deepcopy(execution))
assert baseline["status"] == "PASS", baseline["status"]
corrupted = json.loads(json.dumps(execution))
corrupted["trials"][0]["measurement"]["process_release"].update(
    release_confirmed=False, observed_tree_empty=False, remaining_identities=[[12345, 67890]])
actual = control.aggregate(frozen, corrupted)
result = {
    "scope": "Synthetic CPU outputs and raw metadata only; no GPU/worker/job was launched",
    "baseline_status": baseline["status"],
    "corrupt_measurement_release": corrupted["trials"][0]["measurement"]["process_release"],
    "duplicated_row_observation": corrupted["trials"][0]["observation"],
    "corrupt_status": actual["status"],
    "certified_throughput_present": "effective_design_requests_per_second" in actual["per_mode"]["legacy"],
    "expected": "INVALID when the reused measurement's own release record contradicts the duplicated row"
}
(output / "resource-consistency-result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
assert actual["status"] == "INVALID", "contradictory measurement release must not certify throughput"
