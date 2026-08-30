#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM31heldviz
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --constraint="a100|v100"
#SBATCH --mem=128G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
PARENT_DATA_ROOT=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Other_MainExp31FamilyHeldOutVisualization_1.1
CONFIG=config/Other_MainExp31FamilyHeldOutVisualization_1.1.yaml
CACHE_ROOT="$PARENT_DATA_ROOT/primitive_cache"
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing experiment" >&2
  exit 2
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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm31_heldviz_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
export MPLCONFIGDIR="$JOB_TMP_ROOT/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"

echo "experiment=Other_MainExp31FamilyHeldOutVisualization_1.1"
echo "evidence_scope=physical-family-held-out_exposed-development"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "cache_root=$CACHE_ROOT"
echo "run_dir=$RUN_DIR"
hostname
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
sha256sum "$CONFIG" config/mainExp_TemplateMatching_3.1.yaml

echo "preflight_test_command=python tests/test_all.py"
python tests/test_all.py
python scripts/validate_matcher_backend.py --device cuda
if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 3
fi

python scripts/run_other_mainexp31_family_heldout_visualization_1_1.py \
  --config "$CONFIG" \
  --cache-root "$CACHE_ROOT" \
  --run-dir "$RUN_DIR" \
  --device cuda

python - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import sha256_file

root = Path(sys.argv[1]).resolve()
completion = json.loads((root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
assert completion["schema"] == "pathline_template_matching.family_heldout_run_complete.v1"
assert completion["experiment"] == "Other_MainExp31FamilyHeldOutVisualization_1.1"
assert completion["status"] == "complete"
assert completion["figure_count"] == 8
manifest_path = root / "result_manifest.json"
assert sha256_file(manifest_path) == completion["result_manifest_file_sha256"]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["schema"] == "pathline_template_matching.family_heldout_result.v1"
assert manifest["status"] == "family_held_out_exposed_development_completed"
visualization = json.loads(
    (root / "visualization_manifest.json").read_text(encoding="utf-8")
)
assert visualization["entry_count"] == 8
assert len({(row["dataset"], row["scale_block_id"]) for row in visualization["entries"]}) == 8
for row in visualization["entries"]:
    assert len(row["required_exports"]) == 5
    for item in row["required_exports"] + row["additional_audit_files"]:
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
