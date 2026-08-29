#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM31cacheTest
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
DATA_ROOT=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development
CONFIG=config/mainExp_TemplateMatching_3.1.yaml
VERIFY_CONFIG=config/Verify_LongArcHorizon_1.1.yaml
PORTABLE_ROOT="$DATA_ROOT/portable_windows"
CACHE_ROOT="$DATA_ROOT/primitive_cache"
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing test cache build" >&2
  exit 2
fi
: "${SYNTHETIC_PASS_PATH:?export the immutable Phase A SYNTHETIC_PASS.json path}"
: "${TRAIN_COVERAGE_PASS_PATH:?export the immutable Phase B TRAIN_COVERAGE_PASS.json path}"
: "${ALL_PORTABLES_PASS_PATH:?export the immutable ALL_PORTABLES_PASS.json path}"
if [[ ! -f "$SYNTHETIC_PASS_PATH" || ! -f "$TRAIN_COVERAGE_PASS_PATH" || ! -f "$ALL_PORTABLES_PASS_PATH" ]]; then
  echo "a Verify or all-portable marker is missing" >&2
  exit 3
fi

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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm31_cache_test_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

DATASETS=(tangaroa smokeBuoyancy)
TASK_ID="${SLURM_ARRAY_TASK_ID:?missing Slurm array task ID}"
if (( TASK_ID < 0 || TASK_ID >= 8 || ${#DATASETS[@]} != 2 )); then
  echo "test cache mapping requires task 0..7 and exactly two datasets" >&2
  exit 4
fi
DATASET_INDEX=$((TASK_ID / 4))
ORDINAL=$((TASK_ID % 4))
DATASET="${DATASETS[$DATASET_INDEX]}"
COMMIT_ID=$(git rev-parse HEAD)

echo "experiment=mainExp_TemplateMatching_3.1"
echo "phase=build_test_cache_after_complete_verify"
echo "array_task_id=$TASK_ID"
echo "dataset_index=$DATASET_INDEX"
echo "dataset=$DATASET"
echo "source_ordinal=$ORDINAL"
echo "git_commit=$COMMIT_ID"
echo "synthetic_pass=$SYNTHETIC_PASS_PATH"
echo "train_coverage_pass=$TRAIN_COVERAGE_PASS_PATH"
echo "all_portables_pass=$ALL_PORTABLES_PASS_PATH"
hostname
sha256sum "$CONFIG" "$VERIFY_CONFIG" "$SYNTHETIC_PASS_PATH" "$TRAIN_COVERAGE_PASS_PATH" "$ALL_PORTABLES_PASS_PATH"

python scripts/run_mainexp_template_matching_3_1.py \
  --mode build-slice \
  --config "$CONFIG" \
  --verify-config "$VERIFY_CONFIG" \
  --access-scope all \
  --portable-root "$PORTABLE_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --dataset "$DATASET" \
  --ordinal "$ORDINAL" \
  --synthetic-pass "$SYNTHETIC_PASS_PATH" \
  --train-coverage-pass "$TRAIN_COVERAGE_PASS_PATH" \
  --portable-population-pass "$ALL_PORTABLES_PASS_PATH" \
  --integration-chunk-size 1024 \
  --encoding-chunk-size 4096

python - "$PORTABLE_ROOT/$DATASET/manifest.json" "$CACHE_ROOT/test/$DATASET" "$DATASET" "$ORDINAL" "$COMMIT_ID" "$ALL_PORTABLES_PASS_PATH" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.phase21_pipeline import load_cache_summary_sidecar
from pathline_template_matching.portable_flow import sha256_file

manifest_path = Path(sys.argv[1]).resolve()
cache_dir = Path(sys.argv[2]).resolve()
dataset, ordinal_text, commit, population_marker = sys.argv[3:7]
ordinal = int(ordinal_text)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
rows = sorted(manifest["windows"], key=lambda row: int(row["source_ordinal"]))
assert [int(row["source_ordinal"]) for row in rows] == [0, 1, 2, 3]
source_index = int(rows[ordinal]["source_start_index"])
sidecar = cache_dir / f"source_{source_index:06d}.summary.json"
summary = load_cache_summary_sidecar(sidecar)
row = summary.cache_row
assert row["experiment"] == "mainExp_TemplateMatching_3.1"
assert row["dataset"] == dataset and row["split"] == "test"
assert int(row["source_ordinal"]) == ordinal
assert int(row["source_index"]) == source_index
assert int(row["assigned_count"]) == 128000
assert int(row["valid_count"]) + int(row["invalid_count"]) == 128000
assert float(row["maximum_source_frame_intervals"]) == 48.0
assert row["cache_builder_git_commit"] == commit
assert row["portable_population_scope"] == "all"
assert row["portable_population_pass_file_sha256"] == sha256_file(population_marker)
cache = Path(row["path"])
assert cache.is_file() and sha256_file(cache) == row["file_sha256"]
print(f"validated_test_cache_sidecar={sidecar}")
print(f"cache_file_sha256={row['file_sha256']}")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during test cache build" >&2
  exit 5
fi
