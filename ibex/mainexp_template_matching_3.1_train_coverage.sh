#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM31coverage
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
DATA_ROOT=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development
CONFIG=config/mainExp_TemplateMatching_3.1.yaml
VERIFY_CONFIG=config/Verify_LongArcHorizon_1.1.yaml
CACHE_ROOT="$DATA_ROOT/primitive_cache"
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing Phase B" >&2
  exit 2
fi
: "${SYNTHETIC_PASS_PATH:?export the immutable Phase A SYNTHETIC_PASS.json path}"
: "${TRAIN_PORTABLES_PASS_PATH:?export the immutable TRAIN_PORTABLES_PASS.json path}"
if [[ ! -f "$SYNTHETIC_PASS_PATH" || ! -f "$TRAIN_PORTABLES_PASS_PATH" ]]; then
  echo "synthetic or train-portable marker is missing" >&2
  exit 3
fi
TRAIN_PORTABLE_PREFIX="$DATA_ROOT/verification/portable_population/train_only/"
RESOLVED_TRAIN_PORTABLES_PASS=$(readlink -f "$TRAIN_PORTABLES_PASS_PATH")
case "$RESOLVED_TRAIN_PORTABLES_PASS" in
  "$TRAIN_PORTABLE_PREFIX"*/TRAIN_PORTABLES_PASS.json) ;;
  *)
    echo "TRAIN_PORTABLES_PASS is outside the frozen Weka train-only root: $RESOLVED_TRAIN_PORTABLES_PASS" >&2
    exit 4
    ;;
esac

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm31_train_coverage_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$DATA_ROOT/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"
TRAIN_COVERAGE_PASS="$RUN_DIR/TRAIN_COVERAGE_PASS.json"

echo "experiment=Verify_LongArcHorizon_1.1"
echo "phase=train_coverage"
echo "input_scope=exactly_32_train_caches_no_test_access"
echo "git_commit=$COMMIT_ID"
echo "synthetic_pass=$SYNTHETIC_PASS_PATH"
echo "train_portables_pass=$RESOLVED_TRAIN_PORTABLES_PASS"
echo "run_dir=$RUN_DIR"
hostname
sha256sum "$CONFIG" "$VERIFY_CONFIG" "$SYNTHETIC_PASS_PATH" "$RESOLVED_TRAIN_PORTABLES_PASS"

python scripts/run_mainexp_template_matching_3_1.py \
  --mode audit-train-coverage \
  --config "$CONFIG" \
  --verify-config "$VERIFY_CONFIG" \
  --cache-root "$CACHE_ROOT" \
  --run-dir "$RUN_DIR" \
  --synthetic-pass "$SYNTHETIC_PASS_PATH" \
  --portable-population-pass "$RESOLVED_TRAIN_PORTABLES_PASS"

python - "$CONFIG" "$VERIFY_CONFIG" "$SYNTHETIC_PASS_PATH" "$RESOLVED_TRAIN_PORTABLES_PASS" "$TRAIN_COVERAGE_PASS" "$COMMIT_ID" <<'PY'
from pathlib import Path
import sys

from pathline_template_matching.phase21_pipeline import (
    load_phase31_plan,
    validate_phase31_train_coverage_pass,
)
from pathline_template_matching.portable_flow import sha256_file

config, verify, synthetic, train_portables, coverage, commit = sys.argv[1:7]
plan = load_phase31_plan(config)
evidence = validate_phase31_train_coverage_pass(
    plan,
    coverage,
    synthetic_pass_path=synthetic,
    verify_config_path=verify,
    current_git_commit=commit,
)
assert evidence["train_portable_population_pass"]["file_sha256"] == sha256_file(
    train_portables
)
print(f"train_coverage_status=passed")
print(f"train_coverage_marker={Path(coverage).resolve()}")
print(f"train_coverage_marker_sha256={evidence['file_sha256']}")
print(f"synthetic_pass_file_sha256={evidence['synthetic_pass_file_sha256']}")
print(
    "train_portables_pass_file_sha256="
    f"{evidence['train_portable_population_pass']['file_sha256']}"
)
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during Phase B" >&2
  exit 5
fi
