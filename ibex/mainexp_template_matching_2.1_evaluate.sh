#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM21eval
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --constraint="a100|v100"
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
DATA_ROOT=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_2.1_development
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "tracked worktree is dirty; refusing non-reproducible evaluation" >&2
  exit 2
fi

module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_GPU:-16}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$DATA_ROOT/runs/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"

echo "experiment=mainExp_TemplateMatching_2.1"
echo "phase=frozen_eight_train_two_test_evaluation"
echo "evidence_scope=exposed_development_only"
echo "formal_confirmation=false"
echo "git_commit=${COMMIT_ID}"
echo "run_dir=${RUN_DIR}"
hostname
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

python tests/test_all.py
python scripts/validate_matcher_backend.py --device cuda

python scripts/run_mainexp_template_matching_2_1.py \
  --mode evaluate \
  --config config/mainExp_TemplateMatching_2.1.yaml \
  --cache-root "$DATA_ROOT/primitive_cache" \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --query-chunk-size 1024 \
  --library-chunk-size 8192

python -c "import json,pathlib; p=pathlib.Path('${RUN_DIR}')/'RUN_COMPLETE.json'; d=json.loads(p.read_text()); print('status='+d['status']); print('result_manifest_file_sha256='+d['result_manifest_file_sha256'])"
