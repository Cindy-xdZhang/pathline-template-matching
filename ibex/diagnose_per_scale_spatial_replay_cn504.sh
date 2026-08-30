#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMSpatialReplayDiag
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-spatial-diag
#SBATCH -o /home/zhanx0o/pathline-template-matching-spatial-diag/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-spatial-diag/slurm_logs/%x.%j.err
#SBATCH --nodelist=cn504-17
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-spatial-diag
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_PerScaleNegativeMetric_1.1
CONFIG=config/Verify_PerScaleNegativeMetric_1.1.yaml
CONFIG_SHA256=b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79
FOLD_COMMIT=809ffa3b9490ca4f5b0817d77759b5d88cce628c
FOLD_ARRAY_JOB_ID=51063738
FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)

EXPECTED_DIAGNOSTIC_COMMIT=${EXPECTED_DIAGNOSTIC_COMMIT:?EXPECTED_DIAGNOSTIC_COMMIT is required}
if [[ ! "$EXPECTED_DIAGNOSTIC_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_DIAGNOSTIC_COMMIT must be a lowercase 40-character Git commit" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "diagnostic worktree is dirty" >&2
  exit 3
fi
ACTUAL_COMMIT=$(git rev-parse --verify HEAD^{commit})
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_DIAGNOSTIC_COMMIT" ]]; then
  echo "diagnostic checkout mismatch: $ACTUAL_COMMIT" >&2
  exit 4
fi
if [[ "$(sha256sum "$CONFIG" | awk '{print $1}')" != "$CONFIG_SHA256" ]]; then
  echo "frozen config SHA-256 mismatch" >&2
  exit 5
fi

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "access_scope=outer_prediction_manifest_and_npz_only_no_labels_no_metrics"
echo "diagnostic_commit=$ACTUAL_COMMIT"
echo "fold_commit=$FOLD_COMMIT"
echo "fold_array_job_id=$FOLD_ARRAY_JOB_ID"
hostname
lscpu

for TASK_ID in 0 1 2 3 4; do
  FAMILY=${FAMILIES[$TASK_ID]}
  RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${FOLD_ARRAY_JOB_ID}_${TASK_ID}_${FOLD_COMMIT:0:12}_outer_${FAMILY}"
  echo "diagnostic_fold_start=$TASK_ID family=$FAMILY"
  python scripts/diagnose_per_scale_spatial_replay.py \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR" \
    --full-query-replay
  echo "diagnostic_fold_end=$TASK_ID family=$FAMILY"
done

if [[ "$(git rev-parse --verify HEAD^{commit})" != "$ACTUAL_COMMIT" || -n "$(git status --porcelain)" ]]; then
  echo "diagnostic changed its checkout" >&2
  exit 6
fi
