"""Native RFD4 structure-only reward, without an independent prediction."""
from __future__ import annotations

from astrolabe.losses.base import loss_registry


@loss_registry.register(
    name="rfd4_native_ligand_reward",
    required_params=["python_bin", "source_dir"],
    description="Native small-molecule search terminal reward: min(PLIP HB pairs, 6) + 0.5 * hydrophobic pairs",
    supports_oligomer_specific=True,
)
def rfd4_native_ligand_reward(eval_result, python_bin, source_dir):
    from astrolabe.mutators.rfd4_backend import native_interaction_metrics

    paths = getattr(eval_result, "complex_paths", None)
    if not paths:
        raise ValueError("Native RFD4 ligand reward requires exact generated complex paths")
    scores = {}
    for assembly, path in paths.items():
        metrics = native_interaction_metrics(path, python_bin=python_bin, source_dir=source_dir)
        scores[assembly] = min(metrics["hbond_atom_pair_count_coi"], 6.0) + (
            0.5 * metrics["hydrophobic_atom_pair_count_coi"]
        )
    return scores
