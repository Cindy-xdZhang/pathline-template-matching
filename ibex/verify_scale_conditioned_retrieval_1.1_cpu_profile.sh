#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMScaleCPU
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-scale-cpu
#SBATCH -o /home/zhanx0o/pathline-template-matching-scale-cpu/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-scale-cpu/slurm_logs/%x.%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-scale-cpu
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_ScaleConditionedRetrieval_1.1
CONFIG=config/Verify_ScaleConditionedRetrieval_1.1.yaml
CONFIG_SHA256=f5dbdae08e2e13140245a6a9fd12dba67b4eaf6a7ae1aaea8d600f89a409a6a2
INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_50998592_260a07ad380d/train_cache_input_manifest.json
INPUT_MANIFEST_SHA256=e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393
OUTER_FAMILY=half_cylinder

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
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_scale_cpu_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${SLURM_JOB_ID}_cpu_${SHORT_COMMIT}_outer_${OUTER_FAMILY}"

echo "experiment=Verify_ScaleConditionedRetrieval_1.1"
echo "phase=cpu_outer_fold_profile"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "outer_family=$OUTER_FAMILY"
echo "run_dir=$RUN_DIR"
hostname
lscpu

ACTUAL_CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
ACTUAL_INPUT_SHA=$(sha256sum "$INPUT_MANIFEST" | awk '{print $1}')
if [[ "$ACTUAL_CONFIG_SHA" != "$CONFIG_SHA256" ]]; then
  echo "config SHA-256 mismatch: $ACTUAL_CONFIG_SHA" >&2
  exit 3
fi
if [[ "$ACTUAL_INPUT_SHA" != "$INPUT_MANIFEST_SHA256" ]]; then
  echo "input manifest SHA-256 mismatch: $ACTUAL_INPUT_SHA" >&2
  exit 4
fi
echo "config_sha256=$ACTUAL_CONFIG_SHA"
echo "input_manifest_sha256=$ACTUAL_INPUT_SHA"

echo "preflight_test_command=python tests/test_all.py"
python tests/test_all.py
python scripts/validate_matcher_backend.py --device cpu
if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 5
fi

python scripts/run_verify_scale_conditioned_retrieval_1_1.py \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --outer-family "$OUTER_FAMILY" \
  --output-dir "$RUN_DIR" \
  --device cpu

python - "$RUN_DIR" "$OUTER_FAMILY" "$CONFIG_SHA256" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import sha256_file

root = Path(sys.argv[1]).resolve()
outer_family = sys.argv[2]
config_sha = sys.argv[3]
completion_path = root / "RUN_COMPLETE.json"
completion = json.loads(completion_path.read_text(encoding="utf-8"))
assert completion["schema"] == "pathline_template_matching.run_complete.v1"
assert completion["experiment"] == "Verify_ScaleConditionedRetrieval_1.1"
assert completion["outer_family"] == outer_family
assert completion["config_sha256"] == config_sha
result_path = root / completion["result_manifest_file"]
assert sha256_file(result_path) == completion["result_manifest_file_sha256"]
result = json.loads(result_path.read_text(encoding="utf-8"))
assert result["schema"] == "pathline_template_matching.scale_conditioned_outer_result.v1"
assert result["status"] == "completed"
assert result["outer_family"] == outer_family
assert result["config_sha256"] == config_sha
contract = result["prediction_before_reference_contract"]
assert contract["prediction_manifest_outer_reference_opened"] is False
assert contract["reference_first_open_phase"] == "after_prediction_file_and_manifest_authenticated"
for name, identity in result["artifacts"].items():
    path = root / name
    assert path.is_file()
    assert path.stat().st_size == identity["size_bytes"]
    assert sha256_file(path) == identity["sha256"]
print(f"outer_family={outer_family}")
print(f"selected_candidate={result['selected_candidate']['candidate_id']}")
print(f"outer_f1={result['outer_summary']['f1']}")
print(f"outer_average_precision={result['outer_summary']['average_precision']}")
print(f"result_manifest_file_sha256={completion['result_manifest_file_sha256']}")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during experiment" >&2
  exit 6
fi
