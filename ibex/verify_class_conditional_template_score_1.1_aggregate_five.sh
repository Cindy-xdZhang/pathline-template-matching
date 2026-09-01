#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMClassAgg5
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-class-conditional-score
#SBATCH -o /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --constraint=rome
#SBATCH --account=pi-hadwigm

set -euo pipefail
source ibex/verify_class_conditional_template_score_1.1_common.sh
readonly WRAPPER=ibex/verify_class_conditional_template_score_1.1_aggregate_five.sh
readonly FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required}
readonly REMAINING_FOLD_ARRAY_JOB_ID=${REMAINING_FOLD_ARRAY_JOB_ID:?REMAINING_FOLD_ARRAY_JOB_ID is required}
readonly FIRST_AUTH_DIR=${FIRST_FOLD_AUTH_DIR:?FIRST_FOLD_AUTH_DIR is required}
readonly FIRST_AUTH_COMPLETE_SHA256=${FIRST_FOLD_AUTH_COMPLETE_SHA256:?FIRST_FOLD_AUTH_COMPLETE_SHA256 is required}
readonly SMOKE_PASS=${RESOURCE_SMOKE_PASS:?RESOURCE_SMOKE_PASS is required}
readonly SMOKE_PASS_SHA256=${RESOURCE_SMOKE_PASS_SHA256:?RESOURCE_SMOKE_PASS_SHA256 is required}
[[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || ptm_class_die "FIRST_FOLD_JOB_ID must be numeric"
[[ "$REMAINING_FOLD_ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || ptm_class_die "REMAINING_FOLD_ARRAY_JOB_ID must be numeric"

ptm_class_stage_gate "$WRAPPER"
ptm_class_activate_runtime aggregate_five "${SLURM_CPUS_PER_TASK:-32}"
ptm_class_require_slurm_resources
ptm_class_targeted_preflight
ptm_class_require_evidence
ptm_class_require_resource_smoke "$SMOKE_PASS" "$SMOKE_PASS_SHA256"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly FIRST_FOLD_DIR="$CLASS_EXPERIMENT_ROOT/runs/slurm_${FIRST_FOLD_JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
ptm_class_require_file_sha256 "$FIRST_AUTH_DIR/AGGREGATE_COMPLETE.json" "$FIRST_AUTH_COMPLETE_SHA256" "first-fold authentication completion"
STOP_VERSION=$(python - "$FIRST_AUTH_DIR" "$FIRST_AUTH_COMPLETE_SHA256" "$EXPECTED_GIT_COMMIT" "$CLASS_CONFIG_SHA256" "$FIRST_FOLD_DIR" <<'PY'
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
[[ "$STOP_VERSION" == false ]] || ptm_class_die "authenticated first-fold certificate stops five-fold aggregation"

readonly -a OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)
RUN_DIRS=()
for TASK_ID in 0 1 2 3 4; do
  FAMILY=${OUTER_FAMILIES[$TASK_ID]}
  if [[ "$TASK_ID" == 0 ]]; then
    RUN_DIR="$FIRST_FOLD_DIR"
  else
    SOURCE_JOB_ID=$REMAINING_FOLD_ARRAY_JOB_ID
    RUN_DIR="$CLASS_EXPERIMENT_ROOT/runs/slurm_${SOURCE_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${FAMILY}"
  fi
  [[ -d "$RUN_DIR" ]] || ptm_class_die "required fold directory is missing: $RUN_DIR"
  RUN_DIRS+=("$RUN_DIR")
done
readonly OUTPUT_DIR="$CLASS_EXPERIMENT_ROOT/aggregate/slurm_${JOB_ID}_${SHORT_COMMIT}"
[[ ! -e "$OUTPUT_DIR" ]] || ptm_class_die "immutable aggregate output already exists: $OUTPUT_DIR"

/usr/bin/time -v python "$CLASS_AGGREGATOR" \
  --config "$CLASS_CONFIG" --expected-config-sha256 "$CLASS_CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_GIT_COMMIT" \
  --run-dir "${RUN_DIRS[0]}" --run-dir "${RUN_DIRS[1]}" \
  --run-dir "${RUN_DIRS[2]}" --run-dir "${RUN_DIRS[3]}" \
  --run-dir "${RUN_DIRS[4]}" --output-dir "$OUTPUT_DIR" \
  --mode complete-five-fold --device cpu \
  --kinematic-input-manifest "$CLASS_INPUT_MANIFEST" \
  --kinematic-input-manifest-sha256 "$CLASS_INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$CLASS_SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$CLASS_SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$CLASS_SIDECAR_ROOT" \
  --sidecar-population-manifest "$CLASS_POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$CLASS_POPULATION_MANIFEST_SHA256"

ptm_class_stage_unchanged "$WRAPPER"
echo "aggregate_status=complete_five_fold_artifacts_freshly_authenticated"
echo "aggregate_output_dir=$OUTPUT_DIR"
