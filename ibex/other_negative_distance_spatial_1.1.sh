#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMnegspatial
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Other_NegativeDistanceSpatial_1.1
CONFIG=config/Other_NegativeDistanceSpatial_1.1.yaml
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

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"

echo "experiment=Other_NegativeDistanceSpatial_1.1"
echo "evidence_scope=exposed-development-mechanism-diagnostic"
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

python scripts/run_other_negative_distance_spatial_1_1.py \
  --config "$CONFIG" \
  --output-dir "$RUN_DIR"

python - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
completion = json.loads((root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
assert completion["experiment"] == "Other_NegativeDistanceSpatial_1.1"
assert completion["status"] == "complete"
assert (root / "predictions.csv").is_file()
assert (root / "prediction_manifest.json").is_file()
assert (root / "per_group_metrics.csv").is_file()
assert (root / "aggregate_metrics.csv").is_file()
assert (root / "oracle_upper_bound.csv").is_file()
assert (root / "result_manifest.json").is_file()
print(f"evaluation_status={completion['status']}")
print(f"run_dir={root}")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during experiment" >&2
  exit 4
fi
