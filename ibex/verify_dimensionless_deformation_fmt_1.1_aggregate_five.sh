#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMDimAgg5
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
readonly REMAINING_FOLD_ARRAY_JOB_ID=${REMAINING_FOLD_ARRAY_JOB_ID:?REMAINING_FOLD_ARRAY_JOB_ID is required}
readonly EXPECTED_FOLD_COMMIT=${EXPECTED_FOLD_COMMIT:?EXPECTED_FOLD_COMMIT is required}
readonly -a OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)

[[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || { echo "FIRST_FOLD_JOB_ID must be numeric" >&2; exit 2; }
[[ "$REMAINING_FOLD_ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || { echo "REMAINING_FOLD_ARRAY_JOB_ID must be numeric" >&2; exit 3; }
[[ "$EXPECTED_FOLD_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "EXPECTED_FOLD_COMMIT must be a lowercase 40-character Git commit" >&2; exit 4; }

cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "worktree is not clean" >&2; exit 5; }
readonly COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
[[ "$COMMIT_ID" == "$EXPECTED_FOLD_COMMIT" ]] || { echo "checkout differs from fold commit: $COMMIT_ID" >&2; exit 6; }

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
readonly JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_dimensionless_aggregate_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

python - "$CONFIG" "$CONFIG_SHA256" "$PARENT_CONFIG_SHA256" "$CORE_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import run_verify_dimensionless_deformation_fmt_1_1 as runner

plan = runner.load_plan(Path(sys.argv[1]))
assert plan.sha256 == sys.argv[2] == runner.EXPECTED_CONFIG_SHA256
assert plan.parent_experiment_config_sha256 == sys.argv[3] == runner.EXPECTED_PARENT_CONFIG_SHA256
assert plan.core_sha256 == sys.argv[4] == runner.EXPECTED_CORE_SHA256
assert plan.required_fold_files == runner.REQUIRED_FOLD_FILES
assert len(runner.REQUIRED_FOLD_FILES) == 15
PY

readonly SHORT_COMMIT=${EXPECTED_FOLD_COMMIT:0:12}
RUN_DIRS=()
for TASK_ID in 0 1 2 3 4; do
  FAMILY=${OUTER_FAMILIES[$TASK_ID]}
  if [[ "$TASK_ID" == 0 ]]; then
    SOURCE_JOB_ID=$FIRST_FOLD_JOB_ID
  else
    SOURCE_JOB_ID=$REMAINING_FOLD_ARRAY_JOB_ID
  fi
  RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${SOURCE_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${FAMILY}"
  [[ -d "$RUN_DIR" ]] || { echo "required fold directory is missing: $RUN_DIR" >&2; exit 7; }
  RUN_DIRS+=("$RUN_DIR")
done
readonly OUTPUT_DIR="$EXPERIMENT_ROOT/aggregation/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}_first_${FIRST_FOLD_JOB_ID}_remaining_${REMAINING_FOLD_ARRAY_JOB_ID}"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "immutable aggregate output already exists: $OUTPUT_DIR" >&2; exit 8; }

echo "experiment=Verify_DimensionlessDeformationFMT_1.1"
echo "phase=complete_five_unique_outer_family_aggregation"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "first_fold_job_id=$FIRST_FOLD_JOB_ID"
echo "remaining_fold_array_job_id=$REMAINING_FOLD_ARRAY_JOB_ID"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

python -c 'assert __debug__, "Python assertions must remain enabled"'
/usr/bin/time -v python tests/test_all.py
/usr/bin/time -v python scripts/validate_matcher_backend.py --device cpu
[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain)" ]] || {
  echo "preflight changed the commit or worktree" >&2
  exit 9
}

/usr/bin/time -v python "$AGGREGATOR" \
  --config "$CONFIG" \
  --run-dir "${RUN_DIRS[0]}" \
  --run-dir "${RUN_DIRS[1]}" \
  --run-dir "${RUN_DIRS[2]}" \
  --run-dir "${RUN_DIRS[3]}" \
  --run-dir "${RUN_DIRS[4]}" \
  --output-dir "$OUTPUT_DIR" \
  --mode complete-five-fold \
  --device cpu \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_FOLD_COMMIT"

# Read producer-owned schema constants at verification time; never duplicate
# their literal values in this wrapper.
python - "$OUTPUT_DIR" "$EXPECTED_FOLD_COMMIT" "$CONFIG_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import aggregate_verify_dimensionless_deformation_fmt_1_1 as aggregate

root = Path(sys.argv[1]).resolve()
completion, _ = aggregate._load_self_hashed_json(root / "AGGREGATE_COMPLETE.json")
assert completion["schema"] == aggregate.AGGREGATE_COMPLETE_SCHEMA
assert completion["mode"] == "complete_five_fold_aggregate"
assert completion["fold_numerical_git_commit"] == sys.argv[2]
assert completion["aggregator_git_commit"] == sys.argv[2]
assert completion["config_sha256"] == sys.argv[3]
report, _ = aggregate._load_self_hashed_json(root / completion["report_file"])
assert report["schema"] == aggregate.AGGREGATE_SUMMARY_SCHEMA
assert report["outer_families"] == list(aggregate.runner.FAMILY_ORDER)
assert report["outer_family_count"] == 5
print("five_fold_aggregate_authentication=passed")
PY

[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain)" ]] || {
  echo "commit or clean worktree changed during aggregation" >&2
  exit 10
}
echo "aggregate_status=complete_five_fold_artifacts_freshly_authenticated"
