#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM31stageTest
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
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
  echo "worktree contains tracked or untracked changes; refusing test staging" >&2
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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm31_stage_test_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse HEAD)

echo "experiment=mainExp_TemplateMatching_3.1"
echo "phase=stage_test_only_after_complete_verify"
echo "datasets=tangaroa,smokeBuoyancy"
echo "git_commit=$COMMIT_ID"
echo "synthetic_pass=$SYNTHETIC_PASS_PATH"
echo "train_coverage_pass=$TRAIN_COVERAGE_PASS_PATH"
hostname
sha256sum "$CONFIG" "$VERIFY_CONFIG" "$SYNTHETIC_PASS_PATH" "$TRAIN_COVERAGE_PASS_PATH"

python scripts/stage_mainexp_template_matching_3_1_windows.py \
  --config "$CONFIG" \
  --registry config/datasets.yaml \
  --environment ibex \
  --access-scope test-only \
  --dataset all \
  --output-root "$PORTABLE_ROOT" \
  --synthetic-pass "$SYNTHETIC_PASS_PATH" \
  --train-coverage-pass "$TRAIN_COVERAGE_PASS_PATH" \
  --verify-config "$VERIFY_CONFIG" \
  --resume

python - "$PORTABLE_ROOT" "$COMMIT_ID" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import canonical_json_sha256, sha256_file

root = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
for dataset in ("tangaroa", "smokeBuoyancy"):
    path = root / dataset / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    content = dict(manifest)
    claimed = content.pop("manifest_content_sha256")
    assert claimed == canonical_json_sha256(content)
    assert manifest["experiment"] == "mainExp_TemplateMatching_3.1"
    assert manifest["dataset"] == dataset and manifest["split"] == "test"
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
    print(f"validated_test_portable_manifest={path}")
    print(f"manifest_content_sha256={claimed}")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during test staging" >&2
  exit 4
fi
