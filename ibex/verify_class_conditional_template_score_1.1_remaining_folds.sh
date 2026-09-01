#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMClassRemain
#SBATCH --array=1-4%2
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-class-conditional-score
#SBATCH -o /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --constraint=rome
#SBATCH --account=deepvortex

set -euo pipefail
source ibex/verify_class_conditional_template_score_1.1_common.sh
readonly WRAPPER=ibex/verify_class_conditional_template_score_1.1_remaining_folds.sh
readonly FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required}
readonly FIRST_AUTH_DIR=${FIRST_FOLD_AUTH_DIR:?FIRST_FOLD_AUTH_DIR is required}
readonly FIRST_AUTH_COMPLETE_SHA256=${FIRST_FOLD_AUTH_COMPLETE_SHA256:?FIRST_FOLD_AUTH_COMPLETE_SHA256 is required}
readonly SMOKE_PASS=${RESOURCE_SMOKE_PASS:?RESOURCE_SMOKE_PASS is required}
readonly SMOKE_PASS_SHA256=${RESOURCE_SMOKE_PASS_SHA256:?RESOURCE_SMOKE_PASS_SHA256 is required}
readonly TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
readonly -a OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)
[[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || ptm_class_die "FIRST_FOLD_JOB_ID must be numeric"
[[ "$TASK_ID" =~ ^[1-4]$ ]] || ptm_class_die "remaining-fold task must be 1..4"
readonly OUTER_FAMILY=${OUTER_FAMILIES[$TASK_ID]}

ptm_class_stage_gate "$WRAPPER"
ptm_class_activate_runtime "remaining_${TASK_ID}" "${SLURM_CPUS_PER_TASK:-32}"
ptm_class_require_slurm_resources
ptm_class_targeted_preflight
ptm_class_require_evidence
ptm_class_require_resource_smoke "$SMOKE_PASS" "$SMOKE_PASS_SHA256"
ptm_class_require_file_sha256 "$FIRST_AUTH_DIR/AGGREGATE_COMPLETE.json" "$FIRST_AUTH_COMPLETE_SHA256" "first-fold authentication completion"

readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly FOLD_DIR="$CLASS_EXPERIMENT_ROOT/runs/slurm_${FIRST_FOLD_JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
STOP_VERSION=$(python - "$FIRST_AUTH_DIR" "$FIRST_AUTH_COMPLETE_SHA256" "$EXPECTED_GIT_COMMIT" "$CLASS_CONFIG_SHA256" "$FOLD_DIR" <<'PY'
from pathlib import Path
import sys
from scripts.aggregate_verify_class_conditional_template_score_1_1 import (
    authenticate_single_fold_release,
)

release = authenticate_single_fold_release(
    Path(sys.argv[1]),
    expected_completion_sha256=sys.argv[2],
    expected_fold_commit=sys.argv[3],
    expected_config_sha256=sys.argv[4],
    expected_fold_directory=Path(sys.argv[5]),
)
print(str(bool(release["stop_version"])).lower())
PY
)
readonly STOP_VERSION
[[ "$STOP_VERSION" == false ]] || ptm_class_die "authenticated first-fold certificate stops this version"

readonly ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
readonly RUN_DIR="$CLASS_EXPERIMENT_ROOT/runs/slurm_${ARRAY_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${OUTER_FAMILY}"
[[ ! -e "$RUN_DIR" ]] || ptm_class_die "immutable fold output already exists: $RUN_DIR"

echo "experiment=Verify_ClassConditionalTemplateScore_1.1"
echo "stage=remaining_fold_after_authenticated_no_stop"
echo "outer_family=$OUTER_FAMILY"
echo "run_dir=$RUN_DIR"
hostname
lscpu

/usr/bin/time -v python "$CLASS_RUNNER" \
  --config "$CLASS_CONFIG" --expected-config-sha256 "$CLASS_CONFIG_SHA256" \
  --outer-family "$OUTER_FAMILY" --output-dir "$RUN_DIR" --device cpu \
  --kinematic-input-manifest "$CLASS_INPUT_MANIFEST" \
  --kinematic-input-manifest-sha256 "$CLASS_INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$CLASS_SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$CLASS_SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$CLASS_SIDECAR_ROOT" \
  --sidecar-population-manifest "$CLASS_POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$CLASS_POPULATION_MANIFEST_SHA256"

ptm_class_stage_unchanged "$WRAPPER"
echo "remaining_fold_status=completed_with_fresh_prediction_before_outer_label_gate"
