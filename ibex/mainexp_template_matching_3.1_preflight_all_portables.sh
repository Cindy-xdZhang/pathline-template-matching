#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM31portAll
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
DATA_ROOT=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development
CONFIG=config/mainExp_TemplateMatching_3.1.yaml
VERIFY_CONFIG=config/Verify_LongArcHorizon_1.1.yaml
PORTABLE_ROOT="$DATA_ROOT/portable_windows"
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing all-portable preflight" >&2
  exit 2
fi
: "${SYNTHETIC_PASS_PATH:?export the immutable Phase A SYNTHETIC_PASS.json path}"
: "${TRAIN_COVERAGE_PASS_PATH:?export the immutable Phase B TRAIN_COVERAGE_PASS.json path}"
if [[ ! -f "$SYNTHETIC_PASS_PATH" || ! -f "$TRAIN_COVERAGE_PASS_PATH" ]]; then
  echo "one or both Verify markers are missing" >&2
  exit 3
fi

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm31_portable_all_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$DATA_ROOT/verification/portable_population/all/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"
MARKER="$RUN_DIR/ALL_PORTABLES_PASS.json"

echo "experiment=mainExp_TemplateMatching_3.1"
echo "phase=preflight_all_portables"
echo "access_scope=all"
echo "expected_dataset_count=10"
echo "expected_window_count=40"
echo "git_commit=$COMMIT_ID"
echo "synthetic_pass=$SYNTHETIC_PASS_PATH"
echo "train_coverage_pass=$TRAIN_COVERAGE_PASS_PATH"
echo "run_dir=$RUN_DIR"
hostname
sha256sum "$CONFIG" "$VERIFY_CONFIG" "$SYNTHETIC_PASS_PATH" "$TRAIN_COVERAGE_PASS_PATH"

python scripts/run_mainexp_template_matching_3_1.py \
  --mode preflight-portables \
  --config "$CONFIG" \
  --verify-config "$VERIFY_CONFIG" \
  --portable-root "$PORTABLE_ROOT" \
  --access-scope all \
  --run-dir "$RUN_DIR" \
  --synthetic-pass "$SYNTHETIC_PASS_PATH" \
  --train-coverage-pass "$TRAIN_COVERAGE_PASS_PATH"

python - "$CONFIG" "$PORTABLE_ROOT" "$SYNTHETIC_PASS_PATH" "$TRAIN_COVERAGE_PASS_PATH" "$MARKER" "$COMMIT_ID" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.phase21_pipeline import load_phase31_plan
from pathline_template_matching.portable_flow import canonical_json_sha256, sha256_file

config, portable_root, synthetic, coverage, marker_path, commit = sys.argv[1:7]
plan = load_phase31_plan(config)
path = Path(marker_path).resolve()
assert {item.name for item in path.parent.iterdir()} == {"ALL_PORTABLES_PASS.json"}
marker = json.loads(path.read_text(encoding="utf-8"))
assert marker["schema"] == "pathline_template_matching.phase31_portable_population_pass.v1"
assert marker["experiment"] == "mainExp_TemplateMatching_3.1"
assert marker["status"] == "passed" and marker["access_scope"] == "all"
assert marker["git_commit"] == commit and marker["worktree_clean"] is True
assert marker["config_sha256"] == plan.config_sha256
assert marker["dataset_registry_sha256"] == plan.dataset_registry_sha256
assert marker["portable_root"] == str(Path(portable_root).resolve())
assert marker["dataset_count"] == 10 and marker["window_count"] == 40
assert marker["synthetic_pass_file_sha256"] == sha256_file(synthetic)
assert marker["train_coverage_pass_file_sha256"] == sha256_file(coverage)
rows = marker["rows"]
assert marker["rows_content_sha256"] == canonical_json_sha256(rows)
expected = {(dataset, ordinal) for dataset in plan.datasets for ordinal in range(4)}
found = {(row["dataset"], int(row["source_ordinal"])) for row in rows}
assert len(rows) == 40 and found == expected
assert {row["split"] for row in rows} == {"train", "test"}
print(f"all_portables_status=passed")
print(f"all_portables_marker={path}")
print(f"all_portables_marker_sha256={sha256_file(path)}")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during all-portable preflight" >&2
  exit 4
fi
