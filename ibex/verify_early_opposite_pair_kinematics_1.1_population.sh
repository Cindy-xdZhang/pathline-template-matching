#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMEarlySeal32
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-early-kinematics
#SBATCH -o /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_early_opposite_pair_kinematics_1.1_common.sh
readonly WRAPPER=ibex/verify_early_opposite_pair_kinematics_1.1_population.sh
readonly INPUT_MANIFEST=${KINEMATIC_INPUT_MANIFEST:?KINEMATIC_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${KINEMATIC_INPUT_MANIFEST_SHA256:?KINEMATIC_INPUT_MANIFEST_SHA256 is required}
readonly SYNTHETIC_PASS=${SYNTHETIC_PASS:?SYNTHETIC_PASS is required}
readonly SYNTHETIC_PASS_SHA256=${SYNTHETIC_PASS_SHA256:?SYNTHETIC_PASS_SHA256 is required}
readonly POPULATION_MANIFEST="$EARLY_SIDECAR_ROOT/SIDECAR_POPULATION.json"

ptm_stage_gate "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_early_opposite_pair_kinematics_runner.py
ptm_activate_runtime population_seal "${SLURM_CPUS_PER_TASK:-16}"
ptm_targeted_preflight
ptm_require_file_sha256 "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "kinematic input manifest"
ptm_require_file_sha256 "$SYNTHETIC_PASS" "$SYNTHETIC_PASS_SHA256" "production synthetic PASS"
[[ ! -e "$POPULATION_MANIFEST" ]] || ptm_die "immutable population manifest already exists: $POPULATION_MANIFEST"

echo "experiment=Verify_EarlyOppositePairKinematics_1.1"
echo "stage=authenticate_every_and_only_32_sidecars_then_publish_population_last"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "sidecar_root=$EARLY_SIDECAR_ROOT"
hostname
lscpu

/usr/bin/time -v python "$EARLY_PREPARER" \
  --project-root "$EARLY_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  seal-population \
  --sidecar-root "$EARLY_SIDECAR_ROOT" \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$SYNTHETIC_PASS_SHA256"

readonly POPULATION_MANIFEST_SHA256=$(sha256sum "$POPULATION_MANIFEST" | awk '{print $1}')
ptm_require_file_sha256 "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" "sealed 32-sidecar population"
ptm_stage_unchanged "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_early_opposite_pair_kinematics_runner.py
echo "sidecar_population_manifest=$POPULATION_MANIFEST"
echo "sidecar_population_manifest_sha256=$POPULATION_MANIFEST_SHA256"
