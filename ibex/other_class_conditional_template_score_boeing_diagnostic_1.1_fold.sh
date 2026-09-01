#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMClassBoeing
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-class-conditional-boeing
#SBATCH -o /home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --constraint=rome
#SBATCH --account=pi-hadwigm

set -euo pipefail
source ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_common.sh
readonly VERIFY_FIRST_FOLD_JOB_ID=${VERIFY_FIRST_FOLD_JOB_ID:?VERIFY_FIRST_FOLD_JOB_ID is required}
readonly VERIFY_FIRST_FOLD_AUTH_DIR=${VERIFY_FIRST_FOLD_AUTH_DIR:?VERIFY_FIRST_FOLD_AUTH_DIR is required}
readonly VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256=${VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256:?VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256 is required}
readonly VERIFY_RESOURCE_SMOKE_PASS=${VERIFY_RESOURCE_SMOKE_PASS:?VERIFY_RESOURCE_SMOKE_PASS is required}
readonly VERIFY_RESOURCE_SMOKE_PASS_SHA256=${VERIFY_RESOURCE_SMOKE_PASS_SHA256:?VERIFY_RESOURCE_SMOKE_PASS_SHA256 is required}

ptm_boeing_diag_stage_gate
ptm_boeing_diag_activate_runtime fold "${SLURM_CPUS_PER_TASK:-32}"
ptm_boeing_diag_require_slurm_resources
ptm_boeing_diag_targeted_preflight
ptm_boeing_diag_require_input_evidence
ptm_boeing_diag_require_parent_releases \
  "$VERIFY_FIRST_FOLD_JOB_ID" \
  "$VERIFY_FIRST_FOLD_AUTH_DIR" \
  "$VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256" \
  "$VERIFY_RESOURCE_SMOKE_PASS" \
  "$VERIFY_RESOURCE_SMOKE_PASS_SHA256"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly RUN_DIR="$BOEING_DIAG_EXPERIMENT_ROOT/runs/slurm_${JOB_ID}_0_${SHORT_COMMIT}_outer_boeing_747"
[[ ! -e "$RUN_DIR" ]] || \
  ptm_boeing_diag_die "immutable Boeing diagnostic fold output already exists: $RUN_DIR"

echo "experiment=Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1"
echo "stage=boeing_only_outer_fold_after_authenticated_parent_stop_and_resource_release"
echo "outer_family=boeing_747"
echo "evidence_scope=exposed_post_stop_visualization_diagnostic"
echo "run_dir=$RUN_DIR"
hostname
lscpu

/usr/bin/time -v python "$BOEING_DIAG_RUNNER" \
  --config "$BOEING_DIAG_CONFIG" \
  --expected-config-sha256 "$BOEING_DIAG_CONFIG_SHA256" \
  --outer-family boeing_747 \
  --output-dir "$RUN_DIR" \
  --device cpu \
  --kinematic-input-manifest "$BOEING_DIAG_INPUT_MANIFEST" \
  --kinematic-input-manifest-sha256 "$BOEING_DIAG_INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$BOEING_DIAG_SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$BOEING_DIAG_SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$BOEING_DIAG_SIDECAR_ROOT" \
  --sidecar-population-manifest "$BOEING_DIAG_POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$BOEING_DIAG_POPULATION_MANIFEST_SHA256"

ptm_boeing_diag_stage_unchanged
echo "boeing_diagnostic_fold_job_id=$JOB_ID"
echo "boeing_diagnostic_fold_dir=$RUN_DIR"
