#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMHeadroom
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-headroom
#SBATCH -o /home/zhanx0o/pathline-template-matching-headroom/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-headroom/slurm_logs/%x.%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=cpu_amd_epyc_7702

set -euo pipefail

readonly PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-headroom
readonly EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Other_FirstPrinciplesHeadroom_1.1
readonly CONFIG=config/Other_FirstPrinciplesHeadroom_1.1.yaml
readonly CONFIG_SHA256=a76ae95710f72a6432e4d392606fe4ca5ad4c0fb89b8d50e6d3868f546117477
readonly RUNNER=scripts/run_other_first_principles_headroom_1_1.py
readonly TEST=tests/test_first_principles_headroom.py
readonly WRAPPER=ibex/other_first_principles_headroom_1.1.sh
readonly EARLY_CONFIG_RELATIVE=config/Verify_EarlyOppositePairKinematics_1.1.yaml
readonly EARLY_CONFIG=/home/zhanx0o/pathline-template-matching-early-kinematics/config/Verify_EarlyOppositePairKinematics_1.1.yaml
readonly EARLY_CONFIG_SHA256=e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767
readonly EARLY_RUNNER=scripts/run_verify_early_opposite_pair_kinematics_1_1.py
readonly EARLY_RUNNER_SHA256=e999960ac06d3fedd355e1d6135d9e69316bfe1e798318a22dadf5a8e2063796
readonly EARLY_AGGREGATOR=scripts/aggregate_verify_early_opposite_pair_kinematics_1_1.py
readonly EARLY_AGGREGATOR_SHA256=631909159387cba854f471b3179ff0f0cd97404905e29b74589b2b8cf71f089e
readonly EARLY_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1
readonly INPUT_MANIFEST="$EARLY_ROOT/preparation/slurm_51068863_fd0412dc134d/kinematic_input_manifest.json"
readonly INPUT_MANIFEST_SHA256=1b9df53a9010c6c3c46345639cfbf1d5ab2fe3a43187c79c7dfa0f7d840b102f
readonly SYNTHETIC_PASS="$EARLY_ROOT/preparation/slurm_51068863_fd0412dc134d/SYNTHETIC_PASS.json"
readonly SYNTHETIC_PASS_SHA256=78d0990352777e488f26bb84f3b0fc16e18845fc7cedb8a7d7fc598f32c0afe3
readonly SIDECAR_ROOT="$EARLY_ROOT/kinematic_cache/train"
readonly SIDECAR_POPULATION="$SIDECAR_ROOT/SIDECAR_POPULATION.json"
readonly SIDECAR_POPULATION_SHA256=9f96835b9185218f40df4cc3c52bf3d80a93056681d922a30abfc5c0246f88a7
readonly -a FOLD_DIRECTORIES=(
  "$EARLY_ROOT/runs/slurm_51070299_0_2c3774dca0d8_outer_half_cylinder"
  "$EARLY_ROOT/runs/slurm_51070386_1_2c3774dca0d8_outer_delta_wing"
  "$EARLY_ROOT/runs/slurm_51070386_2_2c3774dca0d8_outer_f22_raptor"
  "$EARLY_ROOT/runs/slurm_51070386_3_2c3774dca0d8_outer_channel"
  "$EARLY_ROOT/runs/slurm_51070386_4_2c3774dca0d8_outer_boeing_747"
)

die() {
  echo "$*" >&2
  exit 2
}

require_file_sha256() {
  local path=$1
  local expected=$2
  local label=$3
  [[ -f "$path" ]] || die "$label is missing: $path"
  local observed
  observed=$(sha256sum "$path" | awk '{print $1}')
  [[ "$observed" == "$expected" ]] || die "$label SHA-256 mismatch: $observed"
}

readonly EXPECTED_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "EXPECTED_GIT_COMMIT must be lowercase 40-hex"
readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}

cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || die "worktree is not clean"
readonly COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
[[ "$COMMIT_ID" == "$EXPECTED_COMMIT" ]] || die "checkout commit differs from EXPECTED_GIT_COMMIT"
for source in "$CONFIG" "$RUNNER" "$TEST" "$WRAPPER" "$EARLY_CONFIG_RELATIVE" "$EARLY_RUNNER" "$EARLY_AGGREGATOR"; do
  git ls-files --error-unmatch "$source" >/dev/null || die "required source is not tracked: $source"
done
require_file_sha256 "$CONFIG" "$CONFIG_SHA256" "headroom config"
require_file_sha256 "$EARLY_CONFIG_RELATIVE" "$EARLY_CONFIG_SHA256" "reporting-checkout Early config"
require_file_sha256 "$EARLY_CONFIG" "$EARLY_CONFIG_SHA256" "Early config"
require_file_sha256 "$EARLY_RUNNER" "$EARLY_RUNNER_SHA256" "Early runner"
require_file_sha256 "$EARLY_AGGREGATOR" "$EARLY_AGGREGATOR_SHA256" "Early aggregator"
require_file_sha256 "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "kinematic input manifest"
require_file_sha256 "$SYNTHETIC_PASS" "$SYNTHETIC_PASS_SHA256" "production synthetic PASS"
require_file_sha256 "$SIDECAR_POPULATION" "$SIDECAR_POPULATION_SHA256" "sidecar population manifest"
for fold in "${FOLD_DIRECTORIES[@]}"; do
  [[ -d "$fold" ]] || die "required exact Early fold directory is missing: $fold"
done

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONOPTIMIZE
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
readonly JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_headroom_${JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

readonly SHORT_COMMIT=${COMMIT_ID:0:12}
readonly OUTPUT_DIR="$EXPERIMENT_ROOT/runs/slurm_${JOB_ID}_${SHORT_COMMIT}"
[[ ! -e "$OUTPUT_DIR" ]] || die "immutable headroom output already exists: $OUTPUT_DIR"

echo "experiment=Other_FirstPrinciplesHeadroom_1.1"
echo "evidence_scope=exposed_development_posthoc_diagnostic_only"
echo "formal_confirmation=false"
echo "oracle_is_deployable=false"
echo "git_commit=$COMMIT_ID"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

python -c 'assert __debug__, "Python assertions must remain enabled"'
python -m py_compile "$RUNNER" "$TEST"
python "$TEST"
/usr/bin/time -v python tests/test_all.py
[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || \
  die "preflight changed the commit or clean worktree"

/usr/bin/time -v python "$RUNNER" \
  --config "$CONFIG" \
  --early-config "$EARLY_CONFIG" \
  --expected-reporting-commit "$COMMIT_ID" \
  --run-dir "${FOLD_DIRECTORIES[0]}" \
  --run-dir "${FOLD_DIRECTORIES[1]}" \
  --run-dir "${FOLD_DIRECTORIES[2]}" \
  --run-dir "${FOLD_DIRECTORIES[3]}" \
  --run-dir "${FOLD_DIRECTORIES[4]}" \
  --kinematic-input-manifest "$INPUT_MANIFEST" \
  --kinematic-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$SIDECAR_ROOT" \
  --sidecar-population-manifest "$SIDECAR_POPULATION" \
  --sidecar-population-manifest-sha256 "$SIDECAR_POPULATION_SHA256" \
  --output-dir "$OUTPUT_DIR"

python - "$OUTPUT_DIR" "$COMMIT_ID" "$CONFIG_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import aggregate_verify_early_opposite_pair_kinematics_1_1 as early_aggregate
from scripts import run_other_first_principles_headroom_1_1 as runner

root = Path(sys.argv[1]).resolve()
completion, _ = early_aggregate._load_self_hashed_json(root / "RUN_COMPLETE.json")
assert completion["schema"] == runner.COMPLETE_SCHEMA
assert completion["reporting_git_commit"] == sys.argv[2]
assert completion["config_sha256"] == sys.argv[3]
result, _ = early_aggregate._load_self_hashed_json(root / "result_manifest.json")
assert result["schema"] == runner.RESULT_SCHEMA
assert result["reporting_git_commit"] == sys.argv[2]
summary, _ = early_aggregate._load_self_hashed_json(root / "aggregate_summary.json")
assert summary["schema"] == runner.SUMMARY_SCHEMA
assert summary["parent_f1_reproduced"] is True
assert summary["oracle_is_deployable"] is False
print("headroom_output_authentication=passed")
PY

[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || \
  die "commit or clean worktree changed during the diagnostic"
echo "headroom_status=complete_posthoc_diagnostic_authenticated"
