#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTM31longverify
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching
#SBATCH -o /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching/slurm_logs/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching
WEKA_ROOT=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development
cd "$PROJECT_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing non-reproducible verification" >&2
  exit 2
fi

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_verify_long_arc_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$WEKA_ROOT/verification/Verify_LongArcHorizon_1.1/synthetic/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"
OUTPUT="$RUN_DIR/SYNTHETIC_PASS.json"

echo "experiment=Verify_LongArcHorizon_1.1"
echo "phase=synthetic_numeric_gate_only"
echo "evidence_scope=synthetic_only_no_real_flow_access"
echo "train_only_coverage_gate_run=false"
echo "git_commit=${COMMIT_ID}"
echo "run_dir=${RUN_DIR}"
hostname

echo "preflight_test_command=python tests/test_all.py"
python tests/test_all.py
echo "preflight_test_status=passed"

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" ]]; then
  echo "Git commit changed while preflight tests were running" >&2
  exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "preflight tests changed the worktree; refusing verification" >&2
  exit 4
fi

python scripts/verify_long_arc_horizon_1_1.py \
  --phase synthetic \
  --main-config config/mainExp_TemplateMatching_3.1.yaml \
  --verify-config config/Verify_LongArcHorizon_1.1.yaml \
  --run-dir "$RUN_DIR"

python -c "import hashlib,json,pathlib; p=pathlib.Path('${OUTPUT}'); d=json.loads(p.read_text()); assert d['status']=='synthetic_gate_passed_train_only_coverage_not_run'; assert d['train_only_coverage_gate_run'] is False; assert d['final_verify_pass'] is False; assert len(d['outputs'])==6; print('status='+d['status']); print('marker_sha256='+hashlib.sha256(p.read_bytes()).hexdigest())"
sha256sum "$OUTPUT"
