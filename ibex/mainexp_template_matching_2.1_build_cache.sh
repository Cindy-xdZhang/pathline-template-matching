#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM21cache
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%A_%a.err
#SBATCH --array=0-39
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
DATA_ROOT=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_2.1_development
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "tracked worktree is dirty; refusing non-reproducible cache construction" >&2
  exit 2
fi

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

DATASETS=(
  cylinder3d
  halfcylinderRe640
  halfcylinderRe6400
  deltaWing_resampled
  deltaWing_LBM
  f22raptor
  channel
  boeing747
  tangaroa
  smokeBuoyancy
)
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
ORDINAL=$((SLURM_ARRAY_TASK_ID % 4))
DATASET="${DATASETS[$DATASET_INDEX]}"
COMMIT_ID=$(git rev-parse HEAD)

echo "experiment=mainExp_TemplateMatching_2.1"
echo "phase=primitive_cache_build"
echo "evidence_scope=exposed_development_only"
echo "dataset=${DATASET}"
echo "source_ordinal=${ORDINAL}"
echo "git_commit=${COMMIT_ID}"
hostname

python scripts/run_mainexp_template_matching_2_1.py \
  --mode build-slice \
  --config config/mainExp_TemplateMatching_2.1.yaml \
  --portable-root "$DATA_ROOT/portable_windows" \
  --cache-root "$DATA_ROOT/primitive_cache" \
  --dataset "$DATASET" \
  --ordinal "$ORDINAL" \
  --integration-chunk-size 2048 \
  --encoding-chunk-size 4096
