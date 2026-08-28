#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMarcverify
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "tracked worktree is dirty; refusing non-reproducible verification" >&2
  exit 2
fi

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

COMMIT_ID=$(git rev-parse HEAD)
OUTPUT=outputs/Verify_ArcLengthResampling_1.1/slurm_${SLURM_JOB_ID}_${COMMIT_ID:0:12}/verification.json

echo "experiment=Verify_ArcLengthResampling_1.1"
echo "git_commit=${COMMIT_ID}"
echo "output=${OUTPUT}"
hostname

python tests/test_all.py
python scripts/verify_arc_length_resampling_1_1.py --output "$OUTPUT"
sha256sum "$OUTPUT"
