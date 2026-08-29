#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM31stageTrain
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# This array stages only the three train raw fields already present on Ibex.
# The five Windows-only train datasets must be staged from the same clean Git
# commit with the same Phase A marker and uploaded per dataset. This job does
# not mean that the complete eight-dataset train portable population exists.

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
DATA_ROOT=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development
CONFIG=config/mainExp_TemplateMatching_3.1.yaml
VERIFY_CONFIG=config/Verify_LongArcHorizon_1.1.yaml
PORTABLE_ROOT="$DATA_ROOT/portable_windows"
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing staging" >&2
  exit 2
fi
: "${SYNTHETIC_PASS_PATH:?export the immutable Phase A SYNTHETIC_PASS.json path}"
if [[ ! -f "$SYNTHETIC_PASS_PATH" ]]; then
  echo "synthetic marker is not a file: $SYNTHETIC_PASS_PATH" >&2
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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm31_stage_train_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

DATASETS=(cylinder3d halfcylinderRe640 deltaWing_resampled)
TASK_ID="${SLURM_ARRAY_TASK_ID:?missing Slurm array task ID}"
if (( TASK_ID < 0 || TASK_ID >= ${#DATASETS[@]} )); then
  echo "train staging task ID is outside 0..2: $TASK_ID" >&2
  exit 4
fi
DATASET="${DATASETS[$TASK_ID]}"
COMMIT_ID=$(git rev-parse HEAD)

echo "experiment=mainExp_TemplateMatching_3.1"
echo "phase=stage_ibex_raw_subset_3_of_8_train_only"
echo "dataset=$DATASET"
echo "git_commit=$COMMIT_ID"
echo "synthetic_pass=$SYNTHETIC_PASS_PATH"
echo "portable_root=$PORTABLE_ROOT"
hostname
sha256sum "$CONFIG" "$VERIFY_CONFIG" "$SYNTHETIC_PASS_PATH"

python scripts/stage_mainexp_template_matching_3_1_windows.py \
  --config "$CONFIG" \
  --registry config/datasets.yaml \
  --environment ibex \
  --access-scope train-only \
  --dataset "$DATASET" \
  --output-root "$PORTABLE_ROOT" \
  --synthetic-pass "$SYNTHETIC_PASS_PATH" \
  --verify-config "$VERIFY_CONFIG" \
  --resume

python - "$PORTABLE_ROOT/$DATASET/manifest.json" "$DATASET" "$COMMIT_ID" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import canonical_json_sha256, sha256_file

path = Path(sys.argv[1]).resolve()
dataset, commit = sys.argv[2:4]
manifest = json.loads(path.read_text(encoding="utf-8"))
claimed = manifest["manifest_content_sha256"]
content = dict(manifest)
content.pop("manifest_content_sha256")
assert claimed == canonical_json_sha256(content)
assert manifest["experiment"] == "mainExp_TemplateMatching_3.1"
assert manifest["dataset"] == dataset and manifest["split"] == "train"
assert manifest["builder_git_commit"] == commit
assert manifest["window_count"] == 4
windows = manifest["windows"]
assert [row["source_ordinal"] for row in windows] == [0, 1, 2, 3]
assert len({row["source_start_index"] for row in windows}) == 4
assert all(row["frame_count"] == 49 for row in windows)
for row in windows:
    window = path.parent / row["relative_path"]
    assert window.is_file() and window.stat().st_size == row["file_size"]
    assert sha256_file(window) == row["file_sha256"]
print(f"validated_train_portable_manifest={path}")
print(f"manifest_content_sha256={claimed}")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during train staging" >&2
  exit 5
fi
