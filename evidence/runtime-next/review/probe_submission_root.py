"""Independent submission fault probe: actual pool, controlled worker, no GPU."""
import json
from pathlib import Path
import subprocess
import sys

repository, output = map(Path, sys.argv[1:3])
sys.path.insert(0, str(repository.resolve()))
from astrolabe.experimental.candidate_batch_policy import (
    CandidatePoolConfig, CandidatePoolScheduler, CandidateTask,
)

def case(batches):
    calls, errors, selected = [], [], []
    def worker(task):
        calls.append([task.generation, task.candidate_id])
        return task.payload
    config = CandidatePoolConfig(num_candidates=3, min_completed=3, max_wait_s=1)
    with CandidatePoolScheduler(config, worker) as pool:
        for generation, tasks in batches:
            try:
                pool.submit_generation(generation, tasks)
            except ValueError as exc:
                errors.append(str(exc))
            for collection in pool.drain():
                if collection.selected:
                    selected.append([collection.selected.task.generation,
                                     collection.selected.candidate_id,
                                     collection.selected.score])
    return {"calls": calls, "errors": errors, "selected": selected}

task = lambda identity, generation=1, score=.5: CandidateTask(identity, generation, score)
result = {
    "source": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
    "scope": "real ThreadPool scheduler; controlled inexpensive workers; no pretrained model or GPU",
    "valid": case([(1, [task("a", score=.2), task("b", score=.8)])]),
    "wrong_generation": case([(1, [task("valid"), task("foreign", 2)])]),
    "duplicate_in_batch": case([(1, [task("dup"), task("dup")])]),
    "duplicate_after_completion": case([(1, [task("once")]), (1, [task("once")])]),
    "id_reused_in_new_generation": case([(1, [task("valid")]), (2, [task("valid", 2)])]),
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
