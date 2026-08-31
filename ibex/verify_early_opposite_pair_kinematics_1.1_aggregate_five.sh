#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMEarlyAgg5
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-early-kinematics
#SBATCH -o /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_early_opposite_pair_kinematics_1.1_common.sh
readonly WRAPPER=ibex/verify_early_opposite_pair_kinematics_1.1_aggregate_five.sh
readonly INPUT_MANIFEST=${KINEMATIC_INPUT_MANIFEST:?KINEMATIC_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${KINEMATIC_INPUT_MANIFEST_SHA256:?KINEMATIC_INPUT_MANIFEST_SHA256 is required}
readonly SYNTHETIC_PASS=${SYNTHETIC_PASS:?SYNTHETIC_PASS is required}
readonly SYNTHETIC_PASS_SHA256=${SYNTHETIC_PASS_SHA256:?SYNTHETIC_PASS_SHA256 is required}
readonly POPULATION_MANIFEST=${SIDECAR_POPULATION_MANIFEST:?SIDECAR_POPULATION_MANIFEST is required}
readonly POPULATION_MANIFEST_SHA256=${SIDECAR_POPULATION_MANIFEST_SHA256:?SIDECAR_POPULATION_MANIFEST_SHA256 is required}
readonly FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required}
readonly REMAINING_FOLD_ARRAY_JOB_ID=${REMAINING_FOLD_ARRAY_JOB_ID:?REMAINING_FOLD_ARRAY_JOB_ID is required}
[[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || ptm_die "FIRST_FOLD_JOB_ID must be numeric"
[[ "$REMAINING_FOLD_ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || ptm_die "REMAINING_FOLD_ARRAY_JOB_ID must be numeric"
[[ "$POPULATION_MANIFEST" == "$EARLY_SIDECAR_ROOT/SIDECAR_POPULATION.json" ]] || \
  ptm_die "sidecar population manifest path differs from the frozen root"

ptm_stage_gate "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_early_opposite_pair_kinematics_runner.py
ptm_activate_runtime aggregate_five "${SLURM_CPUS_PER_TASK:-32}"
ptm_targeted_preflight
ptm_require_file_sha256 "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "kinematic input manifest"
ptm_require_file_sha256 "$SYNTHETIC_PASS" "$SYNTHETIC_PASS_SHA256" "production synthetic PASS"
ptm_require_file_sha256 "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" "sealed 32-sidecar population"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly -a OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)
RUN_DIRS=()
for TASK_ID in 0 1 2 3 4; do
  FAMILY=${OUTER_FAMILIES[$TASK_ID]}
  if [[ "$TASK_ID" == 0 ]]; then
    SOURCE_JOB_ID=$FIRST_FOLD_JOB_ID
  else
    SOURCE_JOB_ID=$REMAINING_FOLD_ARRAY_JOB_ID
  fi
  RUN_DIR="$EARLY_EXPERIMENT_ROOT/runs/slurm_${SOURCE_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${FAMILY}"
  [[ -d "$RUN_DIR" ]] || ptm_die "required fold directory is missing: $RUN_DIR"
  RUN_DIRS+=("$RUN_DIR")
done
readonly OUTPUT_DIR="$EARLY_EXPERIMENT_ROOT/aggregation/slurm_${JOB_ID}_${SHORT_COMMIT}_first_${FIRST_FOLD_JOB_ID}_remaining_${REMAINING_FOLD_ARRAY_JOB_ID}"
[[ ! -e "$OUTPUT_DIR" ]] || ptm_die "immutable aggregate output already exists: $OUTPUT_DIR"

echo "experiment=Verify_EarlyOppositePairKinematics_1.1"
echo "stage=complete_five_unique_outer_family_aggregation"
echo "formal_confirmation=false"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "first_fold_job_id=$FIRST_FOLD_JOB_ID"
echo "remaining_fold_array_job_id=$REMAINING_FOLD_ARRAY_JOB_ID"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

/usr/bin/time -v python "$EARLY_AGGREGATOR" \
  --config "$EARLY_CONFIG" \
  --expected-config-sha256 "$EARLY_CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_GIT_COMMIT" \
  --run-dir "${RUN_DIRS[0]}" \
  --run-dir "${RUN_DIRS[1]}" \
  --run-dir "${RUN_DIRS[2]}" \
  --run-dir "${RUN_DIRS[3]}" \
  --run-dir "${RUN_DIRS[4]}" \
  --output-dir "$OUTPUT_DIR" \
  --mode complete-five-fold \
  --device cpu \
  --kinematic-input-manifest "$INPUT_MANIFEST" \
  --kinematic-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$EARLY_SIDECAR_ROOT" \
  --sidecar-population-manifest "$POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$POPULATION_MANIFEST_SHA256"

ptm_stage_unchanged "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_early_opposite_pair_kinematics_runner.py
echo "aggregate_status=complete_five_fold_artifacts_freshly_authenticated"
