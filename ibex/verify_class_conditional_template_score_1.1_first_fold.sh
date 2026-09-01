#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMClassFirst
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-class-conditional-score
#SBATCH -o /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --constraint=rome
#SBATCH --account=deepvortex

set -euo pipefail
source ibex/verify_class_conditional_template_score_1.1_common.sh
readonly WRAPPER=ibex/verify_class_conditional_template_score_1.1_first_fold.sh
readonly SMOKE_PASS=${RESOURCE_SMOKE_PASS:?RESOURCE_SMOKE_PASS is required}
readonly SMOKE_PASS_SHA256=${RESOURCE_SMOKE_PASS_SHA256:?RESOURCE_SMOKE_PASS_SHA256 is required}

ptm_class_stage_gate "$WRAPPER"
ptm_class_activate_runtime first_fold "${SLURM_CPUS_PER_TASK:-32}"
ptm_class_require_slurm_resources
ptm_class_targeted_preflight
ptm_class_require_evidence
ptm_class_require_resource_smoke "$SMOKE_PASS" "$SMOKE_PASS_SHA256"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly RUN_DIR="$CLASS_EXPERIMENT_ROOT/runs/slurm_${JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
[[ ! -e "$RUN_DIR" ]] || ptm_class_die "immutable first-fold output already exists: $RUN_DIR"

echo "experiment=Verify_ClassConditionalTemplateScore_1.1"
echo "stage=first_outer_fold_after_authenticated_resource_smoke"
echo "outer_family=half_cylinder"
echo "run_dir=$RUN_DIR"
hostname
lscpu

/usr/bin/time -v python "$CLASS_RUNNER" \
  --config "$CLASS_CONFIG" --expected-config-sha256 "$CLASS_CONFIG_SHA256" \
  --outer-family half_cylinder --output-dir "$RUN_DIR" --device cpu \
  --kinematic-input-manifest "$CLASS_INPUT_MANIFEST" \
  --kinematic-input-manifest-sha256 "$CLASS_INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$CLASS_SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$CLASS_SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$CLASS_SIDECAR_ROOT" \
  --sidecar-population-manifest "$CLASS_POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$CLASS_POPULATION_MANIFEST_SHA256"

ptm_class_stage_unchanged "$WRAPPER"
echo "first_fold_job_id=$JOB_ID"
echo "first_fold_run_dir=$RUN_DIR"
