#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM21stage
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing non-reproducible staging" >&2
  exit 2
fi

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

DATASETS=(
  cylinder3d
  halfcylinderRe640
  tangaroa
  deltaWing_resampled
  smokeBuoyancy
)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
COMMIT_ID=$(git rev-parse HEAD)

echo "experiment=mainExp_TemplateMatching_2.1"
echo "phase=portable_window_staging"
echo "evidence_scope=exposed_development_only"
echo "dataset=${DATASET}"
echo "git_commit=${COMMIT_ID}"
hostname

python scripts/stage_mainexp_template_matching_2_1_windows.py \
  --environment ibex \
  --dataset "$DATASET" \
  --resume \
  --output-root /ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_2.1_development/portable_windows

echo "dataset=${DATASET} staging_complete"
