from __future__ import annotations

from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores


class SequenceEvaluator(Evaluator):
    """Expose input assemblies for sequence losses without predicting structures."""

    def debug_cache_identity(self):
        return {"sequence_evaluator": "v1"}

    def evaluate(self, inputs=None, runtime_config=None):
        return StandardScores(
            names=list(inputs or {}),
            plddt={},
            complex_paths={},
            per_proto_scores={},
            per_proto_score_terms={},
            ptm={},
            pae={},
            pde={},
            iptm={},
            chain_pair_scores={},
        )
