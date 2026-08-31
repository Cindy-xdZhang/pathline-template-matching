#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMDimAuth1
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-dimensionless-deformation
#SBATCH -o /home/zhanx0o/pathline-template-matching-dimensionless-deformation/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-dimensionless-deformation/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=cpu_amd_epyc_7702

set -euo pipefail

readonly PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-dimensionless-deformation
readonly EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_DimensionlessDeformationFMT_1.1
readonly CONFIG=config/Verify_DimensionlessDeformationFMT_1.1.yaml
readonly CONFIG_SHA256=c689b1d265bbc39327b2ed4147e8ffb22450dcd26f87b7c19ceae346c9ecfe18
readonly PARENT_CONFIG_SHA256=b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79
readonly CORE_SHA256=5fc4acb47c52c6505737e661cac7f8f503c429c5d88910992655e83cdc53a649
readonly AGGREGATOR=scripts/aggregate_verify_dimensionless_deformation_fmt_1_1.py
readonly FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required}
readonly EXPECTED_FOLD_COMMIT=${EXPECTED_FOLD_COMMIT:?EXPECTED_FOLD_COMMIT is required}

[[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "FIRST_FOLD_JOB_ID must be numeric" >&2
  exit 2
}
[[ "$EXPECTED_FOLD_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "EXPECTED_FOLD_COMMIT must be a lowercase 40-character Git commit" >&2
  exit 3
}

cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain)" ]] || {
  echo "worktree contains tracked or untracked changes; refusing authentication" >&2
  exit 4
}
readonly COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
[[ "$COMMIT_ID" == "$EXPECTED_FOLD_COMMIT" ]] || {
  echo "checkout does not match EXPECTED_FOLD_COMMIT: $COMMIT_ID" >&2
  exit 5
}

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
readonly JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_dimensionless_auth_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

readonly SHORT_COMMIT=${EXPECTED_FOLD_COMMIT:0:12}
readonly FOLD_RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${FIRST_FOLD_JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
readonly OUTPUT_DIR="$EXPERIMENT_ROOT/early_stop/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}_firstfold_${FIRST_FOLD_JOB_ID}"
[[ -d "$FOLD_RUN_DIR" ]] || {
  echo "first fold is missing: $FOLD_RUN_DIR" >&2
  exit 6
}
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "immutable first-fold authentication output exists: $OUTPUT_DIR" >&2
  exit 7
}

python - "$CONFIG" "$CONFIG_SHA256" "$PARENT_CONFIG_SHA256" "$CORE_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import run_verify_dimensionless_deformation_fmt_1_1 as runner

plan = runner.load_plan(Path(sys.argv[1]))
assert plan.sha256 == sys.argv[2] == runner.EXPECTED_CONFIG_SHA256
assert plan.parent_experiment_config_sha256 == sys.argv[3] == runner.EXPECTED_PARENT_CONFIG_SHA256
assert plan.core_sha256 == sys.argv[4] == runner.EXPECTED_CORE_SHA256
assert plan.required_fold_files == runner.REQUIRED_FOLD_FILES
PY

python -c 'assert __debug__, "Python assertions must remain enabled"'
/usr/bin/time -v python tests/test_all.py
[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain)" ]] || {
  echo "preflight changed the commit or worktree" >&2
  exit 8
}

/usr/bin/time -v python "$AGGREGATOR" \
  --config "$CONFIG" \
  --run-dir "$FOLD_RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mode single-fold \
  --device cpu \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_FOLD_COMMIT"

readonly COMPLETION="$OUTPUT_DIR/AGGREGATE_COMPLETE.json"
readonly COMPLETION_SHA256=$(sha256sum "$COMPLETION" | awk '{print $1}')
python - "$OUTPUT_DIR" "$COMPLETION_SHA256" "$EXPECTED_FOLD_COMMIT" "$CONFIG_SHA256" "$FOLD_RUN_DIR" <<'PY'
from pathlib import Path
import json
import sys

from scripts import aggregate_verify_dimensionless_deformation_fmt_1_1 as aggregate

release = aggregate.authenticate_single_fold_release(
    Path(sys.argv[1]),
    expected_completion_sha256=sys.argv[2],
    expected_fold_commit=sys.argv[3],
    expected_config_sha256=sys.argv[4],
    expected_fold_directory=Path(sys.argv[5]),
)
print(json.dumps(release, indent=2, sort_keys=True))
PY

[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain)" ]] || {
  echo "commit or clean worktree changed during authentication" >&2
  exit 9
}
echo "first_fold_auth_dir=$OUTPUT_DIR"
echo "first_fold_auth_complete_sha256=$COMPLETION_SHA256"
