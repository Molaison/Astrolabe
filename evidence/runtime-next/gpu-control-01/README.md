# Exact runner-control supplement

[runner-certification-supplement.json](runner-certification-supplement.json) updates the readiness/admission state of the pre-run matrix's `proteinmpnn_4gyt_ab_gpu_control_v1` scope. It does not update registry certification: 28 registry keys remain BLOCKED and 6 PARTIAL.

Outcome: **BLOCKED_ADMISSION / GPU_NOT_RUN**. Native pretrained CPU constructor/input freeze and launcher preparation completed. The sole Slurm request, job 33334, never obtained an allocation; actual GPU usage was 0 GPU-hours. There are no GPU outputs, RNG comparisons or performance measurements.

Complete original bytes and independent review artifacts are under [the final-source Agent B evidence directory](../final-head-1bdaf3a/agent-b/README.md). The human report is [runtime_gpu_mutator_certification.md](../../../docs/runtime_gpu_mutator_certification.md); independent terminal-accounting evidence is [Agent A's proof](../final-head-1bdaf3a/agent-a/accounting-proof.json).
