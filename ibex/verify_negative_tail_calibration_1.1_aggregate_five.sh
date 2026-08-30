#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMNegTailAgg5
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-tail-cpu
#SBATCH -o /home/zhanx0o/pathline-template-matching-tail-cpu/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-tail-cpu/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-tail-cpu
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_NegativeTailCalibration_1.1
CONFIG=config/Verify_NegativeTailCalibration_1.1.yaml
CONFIG_SHA256=4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b
RUNNER=scripts/run_verify_negative_tail_calibration_1_1.py
RUNNER_SHA256=ab62453215a7ecf508aad50e94e244093d898c2baa148908c215e71ce994b6d5
AGGREGATOR=scripts/aggregate_verify_negative_tail_calibration_1_1.py
AGGREGATOR_SHA256=212e402cf287f780a0e8def4949a38dfde1d96d59b27ad61d50c35dff7730e58
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
  echo "expected fold commit is unavailable in this clone: $EXPECTED_FOLD_COMMIT" >&2
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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_negative_tail_aggregate_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
FOLD_SHORT_COMMIT=${EXPECTED_FOLD_COMMIT:0:12}
OUTPUT_DIR="$EXPERIMENT_ROOT/aggregation/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}_foldarray_${FOLD_ARRAY_JOB_ID}_foldcommit_${FOLD_SHORT_COMMIT}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "immutable aggregation output already exists: $OUTPUT_DIR" >&2
  exit 6
fi

RUN_DIRS=()
for TASK_ID in 0 1 2 3 4; do
  FAMILY=${FAMILIES[$TASK_ID]}
  RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${FOLD_ARRAY_JOB_ID}_${TASK_ID}_cpu_${FOLD_SHORT_COMMIT}_outer_${FAMILY}"
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "required immutable fold run directory is missing: $RUN_DIR" >&2
    exit 7
  fi
  RUN_DIRS+=("$RUN_DIR")
done

echo "experiment=Verify_NegativeTailCalibration_1.1"
echo "phase=complete_five_fold_aggregation"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "expected_fold_commit=$EXPECTED_FOLD_COMMIT"
echo "fold_array_job_id=$FOLD_ARRAY_JOB_ID"
echo "output_dir=$OUTPUT_DIR"
echo "job_tmp_root=$JOB_TMP_ROOT"
hostname
lscpu

ACTUAL_CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
ACTUAL_RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
ACTUAL_AGGREGATOR_SHA=$(sha256sum "$AGGREGATOR" | awk '{print $1}')
FOLD_COMMIT_RUNNER_SHA=$(git show "${EXPECTED_FOLD_COMMIT}:${RUNNER}" | sha256sum | awk '{print $1}')
FOLD_COMMIT_AGGREGATOR_SHA=$(git show "${EXPECTED_FOLD_COMMIT}:${AGGREGATOR}" | sha256sum | awk '{print $1}')
ACTUAL_INPUT_SHA=$(sha256sum "$INPUT_MANIFEST" | awk '{print $1}')
if [[ "$ACTUAL_CONFIG_SHA" != "$CONFIG_SHA256" ]]; then
  echo "config SHA-256 mismatch: $ACTUAL_CONFIG_SHA" >&2
  exit 8
fi
if [[ "$ACTUAL_RUNNER_SHA" != "$RUNNER_SHA256" ]]; then
  echo "current runner SHA-256 mismatch: $ACTUAL_RUNNER_SHA" >&2
  exit 9
fi
if [[ "$FOLD_COMMIT_RUNNER_SHA" != "$RUNNER_SHA256" ]]; then
  echo "fold-commit runner SHA-256 mismatch: $FOLD_COMMIT_RUNNER_SHA" >&2
  exit 10
fi
if [[ "$ACTUAL_AGGREGATOR_SHA" != "$AGGREGATOR_SHA256" ]]; then
  echo "current aggregator SHA-256 mismatch: $ACTUAL_AGGREGATOR_SHA" >&2
  exit 11
fi
if [[ "$FOLD_COMMIT_AGGREGATOR_SHA" != "$AGGREGATOR_SHA256" ]]; then
  echo "fold-commit aggregator SHA-256 mismatch: $FOLD_COMMIT_AGGREGATOR_SHA" >&2
  exit 12
fi
if [[ "$ACTUAL_INPUT_SHA" != "$INPUT_MANIFEST_SHA256" ]]; then
  echo "input manifest SHA-256 mismatch: $ACTUAL_INPUT_SHA" >&2
  exit 13
fi
echo "config_sha256=$ACTUAL_CONFIG_SHA"
echo "runner_sha256=$ACTUAL_RUNNER_SHA"
echo "aggregator_sha256=$ACTUAL_AGGREGATOR_SHA"
echo "fold_commit_aggregator_sha256=$FOLD_COMMIT_AGGREGATOR_SHA"
echo "input_manifest_sha256=$ACTUAL_INPUT_SHA"

echo "profile_phase=preflight_203_tests"
python -c 'assert __debug__, "Python assertions must remain enabled"'
/usr/bin/time -v python tests/test_all.py
echo "profile_phase=cpu_backend_gate"
/usr/bin/time -v python scripts/validate_matcher_backend.py --device cpu
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 14
fi

echo "profile_phase=authenticated_complete_five_fold_aggregation"
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

FINAL_RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
FINAL_AGGREGATOR_SHA=$(sha256sum "$AGGREGATOR" | awk '{print $1}')
if [[ "$FINAL_RUNNER_SHA" != "$RUNNER_SHA256" ]]; then
  echo "runner changed during aggregation: $FINAL_RUNNER_SHA" >&2
  exit 15
fi
if [[ "$FINAL_AGGREGATOR_SHA" != "$AGGREGATOR_SHA256" ]]; then
  echo "aggregator changed during aggregation: $FINAL_AGGREGATOR_SHA" >&2
  exit 16
fi
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during aggregation" >&2
  exit 17
fi
