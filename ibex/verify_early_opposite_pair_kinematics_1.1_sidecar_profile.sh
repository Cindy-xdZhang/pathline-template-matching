#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMEarlyProfile
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-early-kinematics
#SBATCH -o /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail

source ibex/verify_early_opposite_pair_kinematics_1.1_common.sh
readonly WRAPPER=ibex/verify_early_opposite_pair_kinematics_1.1_sidecar_profile.sh
readonly INPUT_MANIFEST=${KINEMATIC_INPUT_MANIFEST:?KINEMATIC_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${KINEMATIC_INPUT_MANIFEST_SHA256:?KINEMATIC_INPUT_MANIFEST_SHA256 is required}
readonly SYNTHETIC_PASS=${SYNTHETIC_PASS:?SYNTHETIC_PASS is required}
readonly SYNTHETIC_PASS_SHA256=${SYNTHETIC_PASS_SHA256:?SYNTHETIC_PASS_SHA256 is required}

ptm_stage_gate "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_early_opposite_pair_kinematics_runner.py
ptm_activate_runtime sidecar_profile "${SLURM_CPUS_PER_TASK:-8}"
ptm_targeted_preflight
ptm_require_file_sha256 "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "kinematic input manifest"
ptm_require_file_sha256 "$SYNTHETIC_PASS" "$SYNTHETIC_PASS_SHA256" "production synthetic PASS"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly PROFILE_ROOT="$EARLY_EXPERIMENT_ROOT/resource_profiles/slurm_${JOB_ID}_${SHORT_COMMIT}/single_row"
readonly PROFILE_COMPLETION="$PROFILE_ROOT/cylinder3d/source_00_index_000000/SIDECAR_COMPLETE.json"
[[ ! -e "$PROFILE_ROOT" ]] || ptm_die "immutable profile root already exists: $PROFILE_ROOT"

echo "experiment=Verify_EarlyOppositePairKinematics_1.1"
echo "stage=single_exact_row_resource_profile_before_population_array"
echo "profile_row_index=0"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "profile_root=$PROFILE_ROOT"
hostname
lscpu

/usr/bin/time -v python "$EARLY_PREPARER" \
  --project-root "$EARLY_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  build-sidecar \
  --sidecar-root "$PROFILE_ROOT" \
  --row-index 0 \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$SYNTHETIC_PASS_SHA256"

readonly PROFILE_COMPLETION_SHA256=$(sha256sum "$PROFILE_COMPLETION" | awk '{print $1}')
/usr/bin/time -v python "$EARLY_PREPARER" \
  --project-root "$EARLY_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  authenticate-profile \
  --sidecar-root "$PROFILE_ROOT" \
  --row-index 0 \
  --completion-sha256 "$PROFILE_COMPLETION_SHA256" \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$SYNTHETIC_PASS_SHA256"

ptm_stage_unchanged "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_early_opposite_pair_kinematics_runner.py
echo "profile_sidecar_root=$PROFILE_ROOT"
echo "profile_completion=$PROFILE_COMPLETION"
echo "profile_completion_sha256=$PROFILE_COMPLETION_SHA256"
echo "resource_decision=production_array_may_run_only_with_explicit_profile_evidence_and_array_concurrency_at_most_2"
