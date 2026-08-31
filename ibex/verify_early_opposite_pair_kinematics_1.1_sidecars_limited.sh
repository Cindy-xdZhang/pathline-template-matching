#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMEarlySidecars
#SBATCH --array=0-31%2
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-early-kinematics
#SBATCH -o /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail

source ibex/verify_early_opposite_pair_kinematics_1.1_common.sh
readonly WRAPPER=ibex/verify_early_opposite_pair_kinematics_1.1_sidecars_limited.sh
readonly INPUT_MANIFEST=${KINEMATIC_INPUT_MANIFEST:?KINEMATIC_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${KINEMATIC_INPUT_MANIFEST_SHA256:?KINEMATIC_INPUT_MANIFEST_SHA256 is required}
readonly SYNTHETIC_PASS=${SYNTHETIC_PASS:?SYNTHETIC_PASS is required}
readonly SYNTHETIC_PASS_SHA256=${SYNTHETIC_PASS_SHA256:?SYNTHETIC_PASS_SHA256 is required}
readonly PROFILE_ROOT=${PROFILE_SIDECAR_ROOT:?PROFILE_SIDECAR_ROOT is required}
readonly PROFILE_COMPLETION=${PROFILE_COMPLETION:?PROFILE_COMPLETION is required}
readonly PROFILE_COMPLETION_SHA256=${PROFILE_COMPLETION_SHA256:?PROFILE_COMPLETION_SHA256 is required}
readonly ROW_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
[[ "$ROW_INDEX" =~ ^([0-9]|[12][0-9]|3[01])$ ]] || ptm_die "sidecar row index must be 0..31"
[[ "$PROFILE_COMPLETION" == "$PROFILE_ROOT/cylinder3d/source_00_index_000000/SIDECAR_COMPLETE.json" ]] || \
  ptm_die "profile completion path is not the exact frozen row-0 artifact"

ptm_stage_gate "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_early_opposite_pair_kinematics_runner.py
ptm_activate_runtime "sidecar_${ROW_INDEX}" "${SLURM_CPUS_PER_TASK:-8}"
ptm_targeted_preflight
ptm_require_file_sha256 "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "kinematic input manifest"
ptm_require_file_sha256 "$SYNTHETIC_PASS" "$SYNTHETIC_PASS_SHA256" "production synthetic PASS"
ptm_require_file_sha256 "$PROFILE_COMPLETION" "$PROFILE_COMPLETION_SHA256" "single-row profile completion"

# Reauthenticate the actual profile sidecar, not a caller-supplied PASS flag.
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

echo "experiment=Verify_EarlyOppositePairKinematics_1.1"
echo "stage=exact_32_sidecar_population_capped_at_two_concurrent_rows"
echo "row_index=$ROW_INDEX"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "sidecar_root=$EARLY_SIDECAR_ROOT"
hostname
lscpu

/usr/bin/time -v python "$EARLY_PREPARER" \
  --project-root "$EARLY_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  build-sidecar \
  --sidecar-root "$EARLY_SIDECAR_ROOT" \
  --row-index "$ROW_INDEX" \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$SYNTHETIC_PASS_SHA256"

ptm_stage_unchanged "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_early_opposite_pair_kinematics_runner.py
echo "row_status=sidecar_and_completion_freshly_authenticated"
