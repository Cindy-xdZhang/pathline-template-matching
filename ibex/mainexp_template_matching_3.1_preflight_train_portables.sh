#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM31portTrain
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
  echo "worktree contains tracked or untracked changes; refusing train portable preflight" >&2
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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm31_portable_train_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$DATA_ROOT/verification/portable_population/train_only/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"
MARKER="$RUN_DIR/TRAIN_PORTABLES_PASS.json"

echo "experiment=mainExp_TemplateMatching_3.1"
echo "phase=preflight_train_portables"
echo "access_scope=train-only"
echo "expected_dataset_count=8"
echo "expected_window_count=32"
echo "git_commit=$COMMIT_ID"
echo "synthetic_pass=$SYNTHETIC_PASS_PATH"
echo "run_dir=$RUN_DIR"
hostname
sha256sum "$CONFIG" "$VERIFY_CONFIG" "$SYNTHETIC_PASS_PATH"

python scripts/run_mainexp_template_matching_3_1.py \
  --mode preflight-portables \
  --config "$CONFIG" \
  --verify-config "$VERIFY_CONFIG" \
  --portable-root "$PORTABLE_ROOT" \
  --access-scope train-only \
  --run-dir "$RUN_DIR" \
  --synthetic-pass "$SYNTHETIC_PASS_PATH"

python - "$CONFIG" "$PORTABLE_ROOT" "$SYNTHETIC_PASS_PATH" "$MARKER" "$COMMIT_ID" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.phase21_pipeline import load_phase31_plan
from pathline_template_matching.portable_flow import canonical_json_sha256, sha256_file

config, portable_root, synthetic, marker_path, commit = sys.argv[1:6]
plan = load_phase31_plan(config)
path = Path(marker_path).resolve()
assert {item.name for item in path.parent.iterdir()} == {"TRAIN_PORTABLES_PASS.json"}
marker = json.loads(path.read_text(encoding="utf-8"))
assert marker["schema"] == "pathline_template_matching.phase31_portable_population_pass.v1"
assert marker["experiment"] == "mainExp_TemplateMatching_3.1"
assert marker["status"] == "passed" and marker["access_scope"] == "train-only"
assert marker["git_commit"] == commit and marker["worktree_clean"] is True
assert marker["config_sha256"] == plan.config_sha256
assert marker["dataset_registry_sha256"] == plan.dataset_registry_sha256
assert marker["portable_root"] == str(Path(portable_root).resolve())
assert marker["dataset_count"] == 8 and marker["window_count"] == 32
assert marker["synthetic_pass_file_sha256"] == sha256_file(synthetic)
assert marker["train_coverage_pass_file_sha256"] is None
rows = marker["rows"]
assert marker["rows_content_sha256"] == canonical_json_sha256(rows)
expected = {(dataset, ordinal) for dataset in plan.train_datasets for ordinal in range(4)}
found = {(row["dataset"], int(row["source_ordinal"])) for row in rows}
assert len(rows) == 32 and found == expected
assert all(row["split"] == "train" for row in rows)
print(f"train_portables_status=passed")
print(f"train_portables_marker={path}")
print(f"train_portables_marker_sha256={sha256_file(path)}")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during train portable preflight" >&2
  exit 4
fi
