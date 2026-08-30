#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMNegSpatialViz
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-negviz
#SBATCH -o /home/zhanx0o/pathline-template-matching-negviz/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-negviz/slurm_logs/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-negviz
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Other_NegativeDistanceSpatialVisualization_1.1
CONFIG=config/Other_NegativeDistanceSpatialVisualization_1.1.yaml
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing experiment" >&2
  exit 2
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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_negative_spatial_viz_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$JOB_TMP_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"

echo "experiment=Other_NegativeDistanceSpatialVisualization_1.1"
echo "evidence_scope=family-held-out_exposed-development"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "run_dir=$RUN_DIR"
hostname
sha256sum "$CONFIG"

python tests/test_all.py
if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 3
fi

python scripts/run_other_negative_distance_spatial_visualization_1_1.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR"

python - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import sha256_file

root = Path(sys.argv[1]).resolve()
completion = json.loads((root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
assert completion["schema"] == "pathline_template_matching.negative_distance_spatial_visualization_run_complete.v1"
assert completion["experiment"] == "Other_NegativeDistanceSpatialVisualization_1.1"
assert completion["status"] == "complete"
assert completion["figure_count"] == 8
assert sha256_file(root / "result_manifest.json") == completion["result_manifest_file_sha256"]
visualization = json.loads((root / "visualization_manifest.json").read_text(encoding="utf-8"))
assert visualization["entry_count"] == 8
assert len({(row["dataset"], row["scale_block_id"]) for row in visualization["entries"]}) == 8
for row in visualization["entries"]:
    assert len(row["required_exports"]) == 7
    for item in row["required_exports"]:
        path = root / item["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == item["size_bytes"]
        assert sha256_file(path) == item["sha256"]
print(f"evaluation_status={completion['status']}")
print(f"query_count={completion['query_count']}")
print(f"result_manifest_file_sha256={completion['result_manifest_file_sha256']}")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during experiment" >&2
  exit 4
fi
