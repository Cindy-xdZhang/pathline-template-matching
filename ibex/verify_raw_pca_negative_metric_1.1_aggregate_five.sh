#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMRawPCAAgg5
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-raw-pca-cpu
#SBATCH -o /home/zhanx0o/pathline-template-matching-raw-pca-cpu/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-raw-pca-cpu/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-raw-pca-cpu
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_RawPCANegativeMetric_1.1
CONFIG=config/Verify_RawPCANegativeMetric_1.1.yaml
CONFIG_SHA256=6f4718ce6d6385bd0bd5b41a7a04e74cb8f2064fee64097f162999e9eefe6440
RUNNER=scripts/run_verify_raw_pca_negative_metric_1_1.py
RUNNER_SHA256=12785cae503d4a64fff838ad6a377c91ac7191adcfe856147a7009fb5e307dee
AGGREGATOR=scripts/aggregate_verify_raw_pca_negative_metric_1_1.py
AGGREGATOR_SHA256=cf6fc43100db62d45f5f83f4d9ecf449c7ed96cad736462f250898659250b2aa
INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_50998592_260a07ad380d/train_cache_input_manifest.json
INPUT_MANIFEST_SHA256=e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393
FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)

FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required}
REMAINING_FOLD_ARRAY_JOB_ID=${REMAINING_FOLD_ARRAY_JOB_ID:?REMAINING_FOLD_ARRAY_JOB_ID is required}
EXPECTED_FOLD_COMMIT=${EXPECTED_FOLD_COMMIT:?EXPECTED_FOLD_COMMIT is required}
if [[ ! "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "FIRST_FOLD_JOB_ID must be a numeric Slurm job ID: $FIRST_FOLD_JOB_ID" >&2
  exit 2
fi
if [[ ! "$REMAINING_FOLD_ARRAY_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "REMAINING_FOLD_ARRAY_JOB_ID must be a numeric Slurm array job ID: $REMAINING_FOLD_ARRAY_JOB_ID" >&2
  exit 3
fi
if [[ ! "$EXPECTED_FOLD_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_FOLD_COMMIT must be a lowercase 40-character Git commit" >&2
  exit 4
fi

cd "$PROJECT_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing aggregation" >&2
  exit 5
fi
if ! git cat-file -e "${EXPECTED_FOLD_COMMIT}^{commit}"; then
  echo "expected fold commit is unavailable: $EXPECTED_FOLD_COMMIT" >&2
  exit 6
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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_raw_pca_aggregate_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
if [[ "$COMMIT_ID" != "$EXPECTED_FOLD_COMMIT" ]]; then
  echo "aggregation checkout does not match EXPECTED_FOLD_COMMIT: $COMMIT_ID" >&2
  exit 7
fi
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
FOLD_SHORT_COMMIT=${EXPECTED_FOLD_COMMIT:0:12}
OUTPUT_DIR="$EXPERIMENT_ROOT/aggregation/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}_first_${FIRST_FOLD_JOB_ID}_remaining_${REMAINING_FOLD_ARRAY_JOB_ID}_foldcommit_${FOLD_SHORT_COMMIT}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "immutable aggregation output already exists: $OUTPUT_DIR" >&2
  exit 8
fi

RUN_DIRS=()
for TASK_ID in 0 1 2 3 4; do
  FAMILY=${FAMILIES[$TASK_ID]}
  if [[ "$TASK_ID" == 0 ]]; then
    FOLD_JOB_ID=$FIRST_FOLD_JOB_ID
  else
    FOLD_JOB_ID=$REMAINING_FOLD_ARRAY_JOB_ID
  fi
  RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${FOLD_JOB_ID}_${TASK_ID}_${FOLD_SHORT_COMMIT}_outer_${FAMILY}"
  [[ -d "$RUN_DIR" ]] || { echo "required fold is missing: $RUN_DIR" >&2; exit 9; }
  RUN_DIRS+=("$RUN_DIR")
done

ACTUAL_CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
ACTUAL_RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
ACTUAL_AGGREGATOR_SHA=$(sha256sum "$AGGREGATOR" | awk '{print $1}')
FOLD_RUNNER_SHA=$(git show "${EXPECTED_FOLD_COMMIT}:${RUNNER}" | sha256sum | awk '{print $1}')
FOLD_AGGREGATOR_SHA=$(git show "${EXPECTED_FOLD_COMMIT}:${AGGREGATOR}" | sha256sum | awk '{print $1}')
ACTUAL_INPUT_SHA=$(sha256sum "$INPUT_MANIFEST" | awk '{print $1}')
[[ "$ACTUAL_CONFIG_SHA" == "$CONFIG_SHA256" ]] || { echo "config SHA-256 mismatch: $ACTUAL_CONFIG_SHA" >&2; exit 10; }
[[ "$ACTUAL_RUNNER_SHA" == "$RUNNER_SHA256" && "$FOLD_RUNNER_SHA" == "$RUNNER_SHA256" ]] || { echo "runner SHA-256 mismatch: $ACTUAL_RUNNER_SHA/$FOLD_RUNNER_SHA" >&2; exit 11; }
[[ "$ACTUAL_AGGREGATOR_SHA" == "$AGGREGATOR_SHA256" && "$FOLD_AGGREGATOR_SHA" == "$AGGREGATOR_SHA256" ]] || { echo "aggregator SHA-256 mismatch: $ACTUAL_AGGREGATOR_SHA/$FOLD_AGGREGATOR_SHA" >&2; exit 12; }
[[ "$ACTUAL_INPUT_SHA" == "$INPUT_MANIFEST_SHA256" ]] || { echo "input manifest SHA-256 mismatch: $ACTUAL_INPUT_SHA" >&2; exit 13; }

echo "experiment=Verify_RawPCANegativeMetric_1.1"
echo "phase=complete_five_fold_aggregation"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "expected_fold_commit=$EXPECTED_FOLD_COMMIT"
echo "first_fold_job_id=$FIRST_FOLD_JOB_ID"
echo "remaining_fold_array_job_id=$REMAINING_FOLD_ARRAY_JOB_ID"
echo "output_dir=$OUTPUT_DIR"
echo "job_tmp_root=$JOB_TMP_ROOT"
hostname
lscpu

python -c 'assert __debug__, "Python assertions must remain enabled"'
/usr/bin/time -v python tests/test_all.py
/usr/bin/time -v python scripts/validate_matcher_backend.py --device cpu
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 14
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

[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA256" ]] || { echo "runner changed during aggregation" >&2; exit 15; }
[[ "$(sha256sum "$AGGREGATOR" | awk '{print $1}')" == "$AGGREGATOR_SHA256" ]] || { echo "aggregator changed during aggregation" >&2; exit 16; }
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during aggregation" >&2
  exit 17
fi
