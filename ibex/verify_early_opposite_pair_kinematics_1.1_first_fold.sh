#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMEarlyFirst
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-early-kinematics
#SBATCH -o /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail
source ibex/verify_early_opposite_pair_kinematics_1.1_common.sh
readonly WRAPPER=ibex/verify_early_opposite_pair_kinematics_1.1_first_fold.sh
readonly INPUT_MANIFEST=${KINEMATIC_INPUT_MANIFEST:?KINEMATIC_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${KINEMATIC_INPUT_MANIFEST_SHA256:?KINEMATIC_INPUT_MANIFEST_SHA256 is required}
readonly SYNTHETIC_PASS=${SYNTHETIC_PASS:?SYNTHETIC_PASS is required}
readonly SYNTHETIC_PASS_SHA256=${SYNTHETIC_PASS_SHA256:?SYNTHETIC_PASS_SHA256 is required}
readonly POPULATION_MANIFEST=${SIDECAR_POPULATION_MANIFEST:?SIDECAR_POPULATION_MANIFEST is required}
readonly POPULATION_MANIFEST_SHA256=${SIDECAR_POPULATION_MANIFEST_SHA256:?SIDECAR_POPULATION_MANIFEST_SHA256 is required}
[[ "$POPULATION_MANIFEST" == "$EARLY_SIDECAR_ROOT/SIDECAR_POPULATION.json" ]] || ptm_die "population path differs from frozen root"

ptm_stage_gate "$WRAPPER" "$EARLY_PREPARER" "$EARLY_RUNNER" "$EARLY_AGGREGATOR" tests/test_early_opposite_pair_kinematics_runner.py
ptm_activate_runtime first_fold "${SLURM_CPUS_PER_TASK:-32}"
ptm_targeted_preflight
ptm_require_file_sha256 "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "kinematic input manifest"
ptm_require_file_sha256 "$SYNTHETIC_PASS" "$SYNTHETIC_PASS_SHA256" "production synthetic PASS"
ptm_require_file_sha256 "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" "sealed 32-sidecar population"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly RUN_DIR="$EARLY_EXPERIMENT_ROOT/runs/slurm_${JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
[[ ! -e "$RUN_DIR" ]] || ptm_die "immutable first-fold output already exists: $RUN_DIR"

echo "experiment=Verify_EarlyOppositePairKinematics_1.1"
echo "stage=first_outer_fold_before_any_remaining_fold_submission"
echo "outer_family=half_cylinder"
echo "run_dir=$RUN_DIR"
hostname
lscpu

/usr/bin/time -v python "$EARLY_RUNNER" \
  --config "$EARLY_CONFIG" --expected-config-sha256 "$EARLY_CONFIG_SHA256" \
  --outer-family half_cylinder --output-dir "$RUN_DIR" --device cpu \
  --kinematic-input-manifest "$INPUT_MANIFEST" --kinematic-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$SYNTHETIC_PASS" --synthetic-pass-sha256 "$SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$EARLY_SIDECAR_ROOT" \
  --sidecar-population-manifest "$POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$POPULATION_MANIFEST_SHA256"

ptm_stage_unchanged "$WRAPPER" "$EARLY_PREPARER" "$EARLY_RUNNER" "$EARLY_AGGREGATOR" tests/test_early_opposite_pair_kinematics_runner.py
echo "first_fold_job_id=$JOB_ID"
echo "first_fold_run_dir=$RUN_DIR"
