#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMPerScaleAgg5
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-tail-cpu
#SBATCH -o /home/zhanx0o/pathline-template-matching-tail-cpu/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-tail-cpu/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-tail-cpu
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_PerScaleNegativeMetric_1.1
CONFIG=config/Verify_PerScaleNegativeMetric_1.1.yaml
CONFIG_SHA256=b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79
RUNNER=scripts/run_verify_per_scale_negative_metric_1_1.py
RUNNER_SHA256=6bf3dba258833918e165a56c9f5141140e3ee422a0c651634769779a6a23b672
AGGREGATOR=scripts/aggregate_verify_per_scale_negative_metric_1_1.py
AGGREGATOR_SHA256=38677afdb95d2fa9178f76a3aa66035778ab6e31bc4ca96f0fe40ee3064c06e8
INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_50998592_260a07ad380d/train_cache_input_manifest.json
INPUT_MANIFEST_SHA256=e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393
FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)

FOLD_ARRAY_JOB_ID=${FOLD_ARRAY_JOB_ID:?FOLD_ARRAY_JOB_ID is required}
EXPECTED_FOLD_COMMIT=${EXPECTED_FOLD_COMMIT:?EXPECTED_FOLD_COMMIT is required}
if [[ ! "$FOLD_ARRAY_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "FOLD_ARRAY_JOB_ID must be a numeric Slurm array job ID: $FOLD_ARRAY_JOB_ID" >&2
  exit 2
fi
if [[ ! "$EXPECTED_FOLD_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_FOLD_COMMIT must be a lowercase 40-character Git commit" >&2
  exit 3
fi

cd "$PROJECT_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing aggregation" >&2
  exit 4
fi
if ! git cat-file -e "${EXPECTED_FOLD_COMMIT}^{commit}"; then
  echo "expected fold commit is unavailable: $EXPECTED_FOLD_COMMIT" >&2
  exit 5
fi
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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_per_scale_aggregate_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
if [[ "$COMMIT_ID" != "$EXPECTED_FOLD_COMMIT" ]]; then
  echo "aggregation checkout does not match EXPECTED_FOLD_COMMIT: $COMMIT_ID" >&2
  exit 6
fi
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
FOLD_SHORT_COMMIT=${EXPECTED_FOLD_COMMIT:0:12}
OUTPUT_DIR="$EXPERIMENT_ROOT/aggregation/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}_foldarray_${FOLD_ARRAY_JOB_ID}_foldcommit_${FOLD_SHORT_COMMIT}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "immutable aggregation output already exists: $OUTPUT_DIR" >&2
  exit 7
fi

RUN_DIRS=()
for TASK_ID in 0 1 2 3 4; do
  FAMILY=${FAMILIES[$TASK_ID]}
  RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${FOLD_ARRAY_JOB_ID}_${TASK_ID}_${FOLD_SHORT_COMMIT}_outer_${FAMILY}"
  [[ -d "$RUN_DIR" ]] || { echo "required fold is missing: $RUN_DIR" >&2; exit 8; }
  RUN_DIRS+=("$RUN_DIR")
done

ACTUAL_CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
ACTUAL_RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
ACTUAL_AGGREGATOR_SHA=$(sha256sum "$AGGREGATOR" | awk '{print $1}')
FOLD_RUNNER_SHA=$(git show "${EXPECTED_FOLD_COMMIT}:${RUNNER}" | sha256sum | awk '{print $1}')
FOLD_AGGREGATOR_SHA=$(git show "${EXPECTED_FOLD_COMMIT}:${AGGREGATOR}" | sha256sum | awk '{print $1}')
ACTUAL_INPUT_SHA=$(sha256sum "$INPUT_MANIFEST" | awk '{print $1}')
[[ "$ACTUAL_CONFIG_SHA" == "$CONFIG_SHA256" ]] || { echo "config SHA-256 mismatch" >&2; exit 9; }
[[ "$ACTUAL_RUNNER_SHA" == "$RUNNER_SHA256" && "$FOLD_RUNNER_SHA" == "$RUNNER_SHA256" ]] || { echo "runner SHA-256 mismatch" >&2; exit 10; }
[[ "$ACTUAL_AGGREGATOR_SHA" == "$AGGREGATOR_SHA256" && "$FOLD_AGGREGATOR_SHA" == "$AGGREGATOR_SHA256" ]] || { echo "aggregator SHA-256 mismatch" >&2; exit 11; }
[[ "$ACTUAL_INPUT_SHA" == "$INPUT_MANIFEST_SHA256" ]] || { echo "input manifest SHA-256 mismatch" >&2; exit 12; }

echo "experiment=Verify_PerScaleNegativeMetric_1.1"
echo "phase=complete_five_fold_aggregation"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "expected_fold_commit=$EXPECTED_FOLD_COMMIT"
echo "fold_array_job_id=$FOLD_ARRAY_JOB_ID"
echo "output_dir=$OUTPUT_DIR"
echo "job_tmp_root=$JOB_TMP_ROOT"
hostname
lscpu

python -c 'assert __debug__, "Python assertions must remain enabled"'
/usr/bin/time -v python tests/test_all.py
/usr/bin/time -v python scripts/validate_matcher_backend.py --device cpu
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 13
fi

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

[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA256" ]] || { echo "runner changed during aggregation" >&2; exit 14; }
[[ "$(sha256sum "$AGGREGATOR" | awk '{print $1}')" == "$AGGREGATOR_SHA256" ]] || { echo "aggregator changed during aggregation" >&2; exit 15; }
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during aggregation" >&2
  exit 16
fi
