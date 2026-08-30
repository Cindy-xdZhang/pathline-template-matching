#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMScaleAgg
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${PTM_FOLD_RUN_DIRS:?export five comma-separated immutable fold run directories}"
PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_ScaleConditionedRetrieval_1.1
CONFIG=config/Verify_ScaleConditionedRetrieval_1.1.yaml
CONFIG_SHA256=f5dbdae08e2e13140245a6a9fd12dba67b4eaf6a7ae1aaea8d600f89a409a6a2
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing aggregate" >&2
  exit 2
fi
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

IFS=',' read -r -a FOLD_DIRS <<< "$PTM_FOLD_RUN_DIRS"
if [[ "${#FOLD_DIRS[@]}" -ne 5 ]]; then
  echo "PTM_FOLD_RUN_DIRS must contain exactly five comma-separated paths" >&2
  exit 3
fi
COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
OUTPUT_DIR="$EXPERIMENT_ROOT/aggregate/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"
ARGS=()
for directory in "${FOLD_DIRS[@]}"; do
  if [[ ! -f "$directory/RUN_COMPLETE.json" ]]; then
    echo "missing fold completion: $directory" >&2
    exit 4
  fi
  ARGS+=(--run-dir "$directory")
done

python scripts/aggregate_verify_scale_conditioned_retrieval_1_1.py \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA256" \
  "${ARGS[@]}" \
  --output-dir "$OUTPUT_DIR"

python - "$OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import sha256_file

root = Path(sys.argv[1]).resolve()
completion = json.loads((root / "AGGREGATE_COMPLETE.json").read_text(encoding="utf-8"))
manifest_path = root / completion["aggregate_manifest_file"]
assert sha256_file(manifest_path) == completion["aggregate_manifest_file_sha256"]
summary = json.loads((root / "aggregate_summary.json").read_text(encoding="utf-8"))
assert summary["outer_family_count"] == 5
print(f"family_macro_f1={summary['family_macro']['f1']}")
print(f"family_macro_average_precision={summary['family_macro']['average_precision']}")
print(f"all_success_conditions_pass={summary['all_success_conditions_pass']}")
print(f"aggregate_manifest_file_sha256={completion['aggregate_manifest_file_sha256']}")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during aggregate" >&2
  exit 5
fi
