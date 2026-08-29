#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM31evaluate
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --constraint="a100|v100"
#SBATCH --mem=128G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
DATA_ROOT=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development
CONFIG=config/mainExp_TemplateMatching_3.1.yaml
VERIFY_CONFIG=config/Verify_LongArcHorizon_1.1.yaml
CACHE_ROOT="$DATA_ROOT/primitive_cache"
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing evaluation" >&2
  exit 2
fi
: "${SYNTHETIC_PASS_PATH:?export the immutable Phase A SYNTHETIC_PASS.json path}"
: "${TRAIN_COVERAGE_PASS_PATH:?export the immutable Phase B TRAIN_COVERAGE_PASS.json path}"
if [[ ! -f "$SYNTHETIC_PASS_PATH" || ! -f "$TRAIN_COVERAGE_PASS_PATH" ]]; then
  echo "one or both Verify markers are missing" >&2
  exit 3
fi

module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm31_evaluate_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
export MPLCONFIGDIR="$JOB_TMP_ROOT/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$DATA_ROOT/runs/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"

echo "experiment=mainExp_TemplateMatching_3.1"
echo "phase=frozen_eight_train_two_test_development_evaluation"
echo "evidence_scope=exposed_development_only"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "synthetic_pass=$SYNTHETIC_PASS_PATH"
echo "train_coverage_pass=$TRAIN_COVERAGE_PASS_PATH"
echo "run_dir=$RUN_DIR"
hostname
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
sha256sum "$CONFIG" "$VERIFY_CONFIG" "$SYNTHETIC_PASS_PATH" "$TRAIN_COVERAGE_PASS_PATH"

python - "$CONFIG" "$VERIFY_CONFIG" "$SYNTHETIC_PASS_PATH" "$TRAIN_COVERAGE_PASS_PATH" "$COMMIT_ID" <<'PY'
import sys

from pathline_template_matching.phase21_pipeline import (
    load_phase31_plan,
    validate_phase31_train_coverage_pass,
)

config, verify, synthetic, coverage, commit = sys.argv[1:6]
plan = load_phase31_plan(config)
evidence = validate_phase31_train_coverage_pass(
    plan,
    coverage,
    synthetic_pass_path=synthetic,
    verify_config_path=verify,
    current_git_commit=commit,
)
print("verify_status=passed")
print(f"train_coverage_marker_sha256={evidence['file_sha256']}")
PY

echo "preflight_test_command=python tests/test_all.py"
python tests/test_all.py
python scripts/validate_matcher_backend.py --device cuda
if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 4
fi

python scripts/run_mainexp_template_matching_3_1.py \
  --mode evaluate \
  --config "$CONFIG" \
  --verify-config "$VERIFY_CONFIG" \
  --cache-root "$CACHE_ROOT" \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --synthetic-pass "$SYNTHETIC_PASS_PATH" \
  --train-coverage-pass "$TRAIN_COVERAGE_PASS_PATH" \
  --query-chunk-size 512 \
  --library-chunk-size 8192

python - "$RUN_DIR" "$SYNTHETIC_PASS_PATH" "$TRAIN_COVERAGE_PASS_PATH" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import sha256_file

root = Path(sys.argv[1]).resolve()
synthetic_path = Path(sys.argv[2]).resolve()
coverage_path = Path(sys.argv[3]).resolve()
completion = json.loads((root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
assert completion["schema"] == "pathline_template_matching.phase31_completion.v1"
assert completion["experiment"] == "mainExp_TemplateMatching_3.1"
assert completion["status"] == "development_completed_confirmation_not_run"
manifest_path = root / "result_manifest.json"
assert sha256_file(manifest_path) == completion["result_manifest_file_sha256"]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["schema"] == "pathline_template_matching.phase31_result_manifest.v1"
assert manifest["status"] == "development_completed_confirmation_not_run"
assert manifest["assigned_primitive_count_per_source_time"] == 128000
assert manifest["maximum_source_frame_intervals"] == 48.0
gates = manifest["verification_gates"]
assert gates["synthetic_pass"]["file_sha256"] == sha256_file(synthetic_path)
assert gates["train_coverage_pass"]["file_sha256"] == sha256_file(coverage_path)
assert gates["train_coverage_pass"]["synthetic_pass_file_sha256"] == sha256_file(
    synthetic_path
)
portable_passes = manifest["portable_population_passes"]
assert len(portable_passes) == 2
by_scope = {row["access_scope"]: row for row in portable_passes}
assert set(by_scope) == {"train-only", "all"}
for scope, row in by_scope.items():
    marker_path = Path(row["path"]).resolve()
    assert marker_path.is_file()
    assert marker_path.stat().st_size == row["file_size"]
    assert sha256_file(marker_path) == row["file_sha256"]
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["access_scope"] == scope and marker["status"] == "passed"
train_marker_sha = by_scope["train-only"]["file_sha256"]
all_marker = json.loads(Path(by_scope["all"]["path"]).read_text(encoding="utf-8"))
coverage_marker = json.loads(coverage_path.read_text(encoding="utf-8"))
assert coverage_marker["train_portable_population_pass_file_sha256"] == train_marker_sha
assert all_marker["synthetic_pass_file_sha256"] == sha256_file(synthetic_path)
assert all_marker["train_coverage_pass_file_sha256"] == sha256_file(coverage_path)
print(f"evaluation_status={completion['status']}")
print(f"result_manifest_file_sha256={completion['result_manifest_file_sha256']}")
print(f"portable_population_singletons=train-only,all")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during evaluation" >&2
  exit 5
fi
