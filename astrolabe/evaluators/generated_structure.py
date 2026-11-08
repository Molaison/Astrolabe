"""Score the exact current generated coordinates, without model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from astrolabe.core.folding_input import Ligand, ProteinChain
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores
from astrolabe.mutators.rfd4 import structure_sequences
from astrolabe.mutators.rfd4_types import ChainSpec


class GeneratedStructureEvaluator(Evaluator):
    preserves_generated_coordinates = True

    def __init__(self, *, initial_structures: Mapping[str, str | Path]):
        self.initial_structures = {
            str(assembly): Path(path).expanduser().resolve()
            for assembly, path in initial_structures.items()
        }
        if not self.initial_structures:
            raise ValueError("generated_structure.initial_structures must map assembly to path")

    def evaluate(self, inputs=None, runtime_config=None):
        runtime_config = runtime_config or {}
        step = int(runtime_config.get("step", 0))
        if step < 0:
            raise ValueError("generated_structure requires a nonnegative step")
        paths = {}
        for assembly, folding_input in inputs.items():
            chains = []
            for chain in folding_input.chains:
                if isinstance(chain, ProteinChain) and not chain.ptms:
                    chains.append(ChainSpec(chain.id, chain.id, chain.sequence))
                elif (isinstance(chain, Ligand) and not chain.smiles
                      and chain.ccd_ids and len(chain.ccd_ids) == 1):
                    chains.append(ChainSpec(chain.id, chain.id, chain.ccd_ids[0], kind="ligand"))
                else:
                    raise ValueError("generated_structure requires canonical proteins or single-CCD ligand chains")
            expected = {chain.chain: chain.sequence for chain in chains}
            if not chains or len(expected) != len(chains) or not any(c.kind == "protein" for c in chains):
                raise ValueError("generated_structure requires unique chains and at least one protein")
            if step == 0:
                if assembly not in self.initial_structures:
                    raise ValueError(f"Missing initial generated structure for {assembly}")
                path = self.initial_structures[assembly]
            else:
                record = runtime_config.get("generated_structure_records", {}).get(assembly)
                if record is None:
                    raise ValueError(f"Missing current-step generated structure for {assembly} at step {step}")
                if record["step"] != step:
                    raise ValueError(f"Stale generated structure for {assembly} at step {step}")
                index_path = Path(record["index_path"])
                index = json.loads(index_path.read_text())
                if (index["assembly"] != assembly
                        or index["attempt"] != record["attempt"]
                        or index["attempt"].partition(":")[0] != str(step)
                        or index["validation"] != "generated_unrefolded"):
                    raise ValueError("Generated structure step/assembly ownership mismatch")
                for key in ("request_digest", "structure_digest"):
                    if index[key] != record[key]:
                        raise ValueError(f"Generated structure {key} mismatch")
                sequences = dict(index["sequences"])
                if len(sequences) != len(index["sequences"]) or sequences != expected:
                    raise ValueError("Generated structure sequence differs from evaluation input")
                path = index_path.parent / "rfd4.generated.cif"
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                file_record = index["files"][path.name]
                if (digest != record["structure_digest"]
                        or digest != file_record["sha256"]
                        or len(content) != file_record["size"]):
                    raise ValueError("Generated structure digest mismatch")
            if structure_sequences(path, tuple(chains)) != expected:
                raise ValueError(f"Generated structure sequence differs from evaluation input for {assembly}")
            paths[assembly] = path
        return StandardScores(names=list(inputs), plddt={}, complex_paths=paths)
