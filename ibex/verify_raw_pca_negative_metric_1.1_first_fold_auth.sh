#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMRawPCAAuth1
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

FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required}
EXPECTED_FOLD_COMMIT=${EXPECTED_FOLD_COMMIT:?EXPECTED_FOLD_COMMIT is required}
[[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "FIRST_FOLD_JOB_ID must be numeric: $FIRST_FOLD_JOB_ID" >&2
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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_raw_pca_first_auth_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
[[ "$COMMIT_ID" == "$EXPECTED_FOLD_COMMIT" ]] || {
  echo "checkout does not match EXPECTED_FOLD_COMMIT: $COMMIT_ID" >&2
  exit 5
}
SHORT_COMMIT=${EXPECTED_FOLD_COMMIT:0:12}
FOLD_RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${FIRST_FOLD_JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
OUTPUT_DIR="$EXPERIMENT_ROOT/early_stop/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}_firstfold_${FIRST_FOLD_JOB_ID}"
[[ -d "$FOLD_RUN_DIR" ]] || {
  echo "first fold is missing: $FOLD_RUN_DIR" >&2
  exit 6
}
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "immutable first-fold authentication output exists: $OUTPUT_DIR" >&2
  exit 7
}

ACTUAL_CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
ACTUAL_RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
ACTUAL_AGGREGATOR_SHA=$(sha256sum "$AGGREGATOR" | awk '{print $1}')
COMMITTED_RUNNER_SHA=$(git show "${EXPECTED_FOLD_COMMIT}:${RUNNER}" | sha256sum | awk '{print $1}')
COMMITTED_AGGREGATOR_SHA=$(git show "${EXPECTED_FOLD_COMMIT}:${AGGREGATOR}" | sha256sum | awk '{print $1}')
ACTUAL_INPUT_SHA=$(sha256sum "$INPUT_MANIFEST" | awk '{print $1}')
[[ "$ACTUAL_CONFIG_SHA" == "$CONFIG_SHA256" ]] || { echo "config SHA-256 mismatch: $ACTUAL_CONFIG_SHA" >&2; exit 8; }
[[ "$ACTUAL_RUNNER_SHA" == "$RUNNER_SHA256" && "$COMMITTED_RUNNER_SHA" == "$RUNNER_SHA256" ]] || { echo "runner SHA-256 mismatch: $ACTUAL_RUNNER_SHA/$COMMITTED_RUNNER_SHA" >&2; exit 9; }
[[ "$ACTUAL_AGGREGATOR_SHA" == "$AGGREGATOR_SHA256" && "$COMMITTED_AGGREGATOR_SHA" == "$AGGREGATOR_SHA256" ]] || { echo "aggregator SHA-256 mismatch: $ACTUAL_AGGREGATOR_SHA/$COMMITTED_AGGREGATOR_SHA" >&2; exit 10; }
[[ "$ACTUAL_INPUT_SHA" == "$INPUT_MANIFEST_SHA256" ]] || { echo "input manifest SHA-256 mismatch: $ACTUAL_INPUT_SHA" >&2; exit 11; }

echo "experiment=Verify_RawPCANegativeMetric_1.1"
echo "stage=first_fold_complete_authentication_and_early_stop_certificate"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "first_fold_job_id=$FIRST_FOLD_JOB_ID"
echo "first_fold_run_dir=$FOLD_RUN_DIR"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

python -c 'assert __debug__, "Python assertions must remain enabled"'
/usr/bin/time -v python tests/test_all.py
[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain)" ]] || {
  echo "preflight changed the commit or worktree" >&2
  exit 12
}

/usr/bin/time -v python "$AGGREGATOR" \
  --config "$CONFIG" \
  --run-dir "$FOLD_RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mode single-fold \
  --device cpu \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_FOLD_COMMIT"

COMPLETION="$OUTPUT_DIR/AGGREGATE_COMPLETE.json"
COMPLETION_SHA256=$(sha256sum "$COMPLETION" | awk '{print $1}')
CERTIFICATE="$OUTPUT_DIR/early_stop_certificate.json"
CERTIFICATE_SHA256=$(sha256sum "$CERTIFICATE" | awk '{print $1}')
STOP_VERSION=$(python -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1], encoding="utf-8"))["stop_version"])).lower())' "$CERTIFICATE")
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA256" ]] || { echo "runner changed during authentication" >&2; exit 13; }
[[ "$(sha256sum "$AGGREGATOR" | awk '{print $1}')" == "$AGGREGATOR_SHA256" ]] || { echo "aggregator changed during authentication" >&2; exit 14; }
[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain)" ]] || {
  echo "commit or clean worktree changed during authentication" >&2
  exit 15
}
echo "first_fold_auth_dir=$OUTPUT_DIR"
echo "first_fold_auth_complete_sha256=$COMPLETION_SHA256"
echo "early_stop_certificate_sha256=$CERTIFICATE_SHA256"
echo "stop_version=$STOP_VERSION"
