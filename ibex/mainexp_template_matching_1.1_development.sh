#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMdev11
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --constraint="a100|v100"
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

if [[ -n "$(git status --porcelain)" ]]; then
  echo "tracked worktree is dirty; refusing a non-reproducible experiment" >&2
  exit 2
fi

module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="outputs/mainExp_TemplateMatching_1.1_development/runs/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"

echo "experiment=mainExp_TemplateMatching_1.1"
echo "phase=development_cache_backed"
echo "evidence_scope=exposed_development_only"
echo "sealed_confirmation_access=forbidden"
echo "git_commit=${COMMIT_ID}"
echo "run_dir=${RUN_DIR}"
hostname
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

python -c "import importlib.metadata as m, matplotlib, netCDF4, numpy, skimage, tifffile, torch, yaml; print('runtime_versions='+','.join(f'{name}={m.version(name)}' for name in ('matplotlib','netCDF4','numpy','PyYAML','scikit-image','tifffile','torch')))"
python tests/test_all.py
python scripts/validate_matcher_backend.py --device cuda

python scripts/run_mainexp_development.py \
  --config config/mainExp_TemplateMatching_1.1_development.yaml \
  --run-dir "$RUN_DIR" \
  --mode run-and-finalize \
  --environment ibex \
  --device cuda \
  --figure-dpi 360

python -c "import json,pathlib; p=pathlib.Path('${RUN_DIR}')/'run_state.json'; d=json.loads(p.read_text()); print('result_manifest='+d['result_manifest']); print('result_manifest_file_sha256='+d['result_manifest_file_sha256']); print('status='+d['status'])"
