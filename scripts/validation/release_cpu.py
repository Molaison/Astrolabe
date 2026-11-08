#!/usr/bin/env python3
"""Run the release or execution CPU suite; forward remaining pytest arguments."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_CPU_TESTS = (
    "tests/test_distribution.py",
    "tests/test_release_cli.py",
    "tests/test_release_validation.py",
    "tests/test_cli_completion.py",
    "tests/test_cli_lightweight_paths.py",
    "tests/test_cli_run_dry_configs.py",
    "tests/test_env_local_defaults.py",
    "tests/test_cli_vendor_paths.py",
    "tests/test_container_behavior_alignment.py",
    "tests/test_apptainer_build.py",
    "tests/test_install.py",
    "tests/test_env_setup_selection.py::test_env_setup_writes_canonical_runner_hint",
    "tests/test_env_setup_selection.py::test_snapshot_source_identity_does_not_require_model_weights",
    "tests/test_cpu_release_example.py",
    "tests/test_run_provenance.py",
    "tests/test_evaluator_stages.py",
    "tests/test_evaluator_runtime_stages.py",
    "tests/test_resource_admission.py",
    "tests/test_argparser_enabled_aliases.py",
    "tests/test_paper_cli.py",
    "tests/test_paper_scores.py",
    "tests/test_paper_statistics.py",
    "tests/test_paper_agreement.py",
    "tests/test_paper_baselines.py",
    "tests/test_paper_twostate.py",
    "tests/test_paper_geometry.py",
    "tests/test_paper_assemblies.py",
    "tests/test_paper_cyclic.py",
    "tests/test_rna_beam.py",
    "tests/test_rna_geometry.py",
    "tests/test_posthoc_collect_required_metrics.py",
    "tests/test_posthoc_filtering.py",
    "tests/test_posthoc_report.py",
    "tests/test_posthoc_target_chain_metrics.py",
    "tests/test_constraint_compliance_analysis.py",
    "tests/test_standard_scores_normalization.py",
    "tests/test_shape_complementarity.py",
    "tests/test_shape_complementarity_loss.py",
    "tests/test_design_data.py",
    "tests/test_design_runtime.py",
    "tests/test_backend_startup.py",
    "tests/test_startup_validation.py",
    "tests/test_pyrosetta_lazy_import.py",
    "tests/test_protein_beam_metric_guards.py",
    "tests/test_runtime_context.py",
    "tests/test_runtime_support.py",
    "tests/test_backend_lifecycle_boundaries.py",
    "tests/test_confidence_optional_torch.py",
    "tests/test_decision_assessments.py",
    "tests/test_evidence_boundaries.py",
    "tests/test_factory_lifecycle.py",
    "tests/test_run_execution_evidence.py",
    "tests/test_stage_evidence_regressions.py",
    "tests/test_design_native_scores.py",
    "tests/test_design_legacy.py",
    "tests/test_design_variable_length.py",
    "tests/test_native_flow_parity.py",
    "tests/test_portable_decision_assessments.py",
    "tests/test_portable_native_flow.py",
    "tests/test_design_native_registry.py",
    "tests/test_cli_proton_states.py",
    "tests/test_ph_automatic_integration.py",
    "tests/test_ph_core_losses.py",
    "tests/test_ph_sensitive_losses.py",
    "tests/test_posthoc_network_scores.py",
    "tests/test_proton_automatic_bias.py",
    "tests/test_proton_native_parity_contract.py",
    "tests/test_proton_network_fallback.py",
    "tests/test_proton_network_only.py",
    "tests/test_proton_network_pair_bias.py",
    "tests/test_proton_network_roles.py",
    "tests/test_proton_pottsmpnn_mutator.py",
    "tests/test_proton_state_patterns.py",
    "tests/test_proton_state_scoring.py",
    "tests/test_interface_motif_rmsd.py",
)


EXECUTION_CPU_TESTS = RELEASE_CPU_TESTS + (
    "tests/test_kernel_stages.py",
    "tests/test_refinement_worker.py",
    "tests/test_native_refinement_recovery.py",
    "tests/test_configured_refinement.py",
    "tests/test_model_execution.py",
    "tests/test_model_daemon.py",
    "tests/test_native_evaluator_daemons.py",
    "tests/test_inference_pipeline.py",
    "tests/test_af3_pipeline.py",
    "tests/test_af3_execution_boundary.py",
    "tests/test_alphafold3_prepare_output_scores.py",
    "tests/test_alphafold3_scores_chain_ptm.py",
    "tests/test_af3_bucket_policy.py",
    "tests/test_af3_refstructure_env_wiring.py",
    "tests/test_parallel_refinement.py",
    "tests/test_persistent_scientific_worker.py",
    "tests/test_design_branch_execution.py",
    "tests/test_design_branch_owner.py",
    "tests/test_design_branching.py",
    "tests/test_resource_session.py",
    "tests/test_local_model_pool.py",
    "tests/test_model_daemon_concurrency.py",
    "tests/test_model_pipeline.py",
    "tests/test_mutator_runtime_stages.py",
    "tests/test_esmfold_loading_stages.py",
    "tests/test_staged_loading_admission.py",
    "tests/test_shared_model_clients.py",
    "tests/test_shared_model_spawn.py",
    "tests/test_telemetry_readiness_recovery.py",
    "tests/test_evaluator_startup.py",
    "tests/test_daily_task_config.py",
    "tests/test_rna_evaluator_adapters.py",
    "tests/test_integrated_cli_workflow.py",
    "tests/test_af3_lifecycle.py",
    "tests/test_af3_pool.py",
    "tests/test_af3_resident.py",
    "tests/test_auxiliary_resource_admission.py",
    "tests/test_boltz_runtime_contract.py",
    "tests/test_runtime_review_regressions.py",
    "tests/test_trajectory_model_owner.py",
    "tests/test_bioemu2.py",
    "tests/test_design_variable_length.py",
    "tests/test_design_native_prediction.py",
    "tests/test_protein_beam.py",
    "tests/test_protein_beam_routing.py",
    "tests/test_interface_budget.py",
    "tests/test_rfd4_backend.py",
    "tests/test_rfd4_feedback.py",
    "tests/test_rfd4_mutator.py",
    "tests/test_rfd4_reconstruction.py",
)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--suite", choices=("release", "execution"), default="release")
    options, arguments = parser.parse_known_args(arguments)
    selected = EXECUTION_CPU_TESTS if options.suite == "execution" else RELEASE_CPU_TESTS
    return subprocess.run(
        [sys.executable, "-m", "pytest", *selected, *arguments],
        cwd=REPO_ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
