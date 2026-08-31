#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMEarlyPrep
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-early-kinematics
#SBATCH -o /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

source ibex/verify_early_opposite_pair_kinematics_1.1_common.sh
readonly WRAPPER=ibex/verify_early_opposite_pair_kinematics_1.1_synthetic_input.sh
readonly PARENT_INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_50998592_260a07ad380d/train_cache_input_manifest.json
readonly PARENT_INPUT_SHA256=e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393
readonly TRAIN_PORTABLE_MARKER=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/portable_population/train_only/slurm_50998449_260a07ad380d/TRAIN_PORTABLES_PASS.json
readonly TRAIN_PORTABLE_SHA256=489d303b4430be7eded4fe39ab87107c778e1f7db2579cb9e3bb1fdfce209341

ptm_stage_gate "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_all.py \
  tests/test_early_kinematic_preparation.py \
  tests/test_early_opposite_pair_kinematics.py \
  tests/test_seed_time_kinematic_sidecar.py \
  tests/test_early_opposite_pair_kinematics_runner.py
ptm_activate_runtime synthetic_input "${SLURM_CPUS_PER_TASK:-8}"
ptm_targeted_preflight
ptm_full_preflight
ptm_require_file_sha256 "$PARENT_INPUT_MANIFEST" "$PARENT_INPUT_SHA256" "parent 32-row input manifest"
ptm_require_file_sha256 "$TRAIN_PORTABLE_MARKER" "$TRAIN_PORTABLE_SHA256" "train portable population marker"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly PREPARATION_DIR="$EARLY_EXPERIMENT_ROOT/preparation/slurm_${JOB_ID}_${SHORT_COMMIT}"
readonly SYNTHETIC_PASS="$PREPARATION_DIR/SYNTHETIC_PASS.json"
readonly INPUT_MANIFEST="$PREPARATION_DIR/kinematic_input_manifest.json"
[[ ! -e "$PREPARATION_DIR" ]] || ptm_die "immutable preparation directory already exists: $PREPARATION_DIR"

echo "experiment=Verify_EarlyOppositePairKinematics_1.1"
echo "stage=synthetic_production_oracles_then_exact_32_input_freeze"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "preparation_dir=$PREPARATION_DIR"
hostname
lscpu

/usr/bin/time -v python "$EARLY_PREPARER" \
  --project-root "$EARLY_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  synthetic \
  --run-dir "$PREPARATION_DIR"

readonly SYNTHETIC_SHA256=$(sha256sum "$SYNTHETIC_PASS" | awk '{print $1}')
ptm_require_file_sha256 "$SYNTHETIC_PASS" "$SYNTHETIC_SHA256" "production synthetic PASS"

/usr/bin/time -v python "$EARLY_PREPARER" \
  --project-root "$EARLY_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  freeze-input \
  --output-path "$INPUT_MANIFEST" \
  --parent-input-manifest "$PARENT_INPUT_MANIFEST" \
  --train-portable-marker "$TRAIN_PORTABLE_MARKER" \
  --synthetic-pass "$SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$SYNTHETIC_SHA256"

readonly INPUT_MANIFEST_SHA256=$(sha256sum "$INPUT_MANIFEST" | awk '{print $1}')
ptm_require_file_sha256 "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "frozen kinematic input manifest"
python - "$PREPARATION_DIR" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
expected = {"synthetic_oracle_evidence.json", "SYNTHETIC_PASS.json", "kinematic_input_manifest.json"}
assert root.is_dir()
assert {path.name for path in root.iterdir()} == expected
PY

ptm_stage_unchanged "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_all.py \
  tests/test_early_kinematic_preparation.py \
  tests/test_early_opposite_pair_kinematics.py \
  tests/test_seed_time_kinematic_sidecar.py \
  tests/test_early_opposite_pair_kinematics_runner.py
echo "synthetic_pass=$SYNTHETIC_PASS"
echo "synthetic_pass_sha256=$SYNTHETIC_SHA256"
echo "kinematic_input_manifest=$INPUT_MANIFEST"
echo "kinematic_input_manifest_sha256=$INPUT_MANIFEST_SHA256"
