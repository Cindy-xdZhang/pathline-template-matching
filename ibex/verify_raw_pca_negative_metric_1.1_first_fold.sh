#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMRawPCAFirst
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-raw-pca-cpu
#SBATCH -o /home/zhanx0o/pathline-template-matching-raw-pca-cpu/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-raw-pca-cpu/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

# Profile half_cylinder and complete only the shared label-free postvalidation.
# This stage cannot release the remaining-family array; the separate
# first_fold_auth wrapper must first perform complete fold authentication and
# publish a frozen no-stop certificate.
export SLURM_ARRAY_TASK_ID=0
export SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
exec bash ibex/verify_raw_pca_negative_metric_1.1_all_folds.sh
