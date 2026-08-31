#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMDimFirst
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-dimensionless-deformation
#SBATCH -o /home/zhanx0o/pathline-template-matching-dimensionless-deformation/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-dimensionless-deformation/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=cpu_amd_epyc_7702

set -euo pipefail

# Task zero is always run alone.  Only the separate authentication job may
# release tasks 1--4 after reconstructing the mathematical stop certificate.
export SLURM_ARRAY_TASK_ID=0
export SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
exec bash ibex/verify_dimensionless_deformation_fmt_1.1_all_folds.sh
