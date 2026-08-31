#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMDimRemain
#SBATCH --array=1-4%2
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-dimensionless-deformation
#SBATCH -o /home/zhanx0o/pathline-template-matching-dimensionless-deformation/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-dimensionless-deformation/slurm_logs/%x.%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=cpu_amd_epyc_7702

set -euo pipefail

readonly PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-dimensionless-deformation
readonly EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_DimensionlessDeformationFMT_1.1
readonly CONFIG=config/Verify_DimensionlessDeformationFMT_1.1.yaml
readonly CONFIG_SHA256=c689b1d265bbc39327b2ed4147e8ffb22450dcd26f87b7c19ceae346c9ecfe18
readonly PARENT_CONFIG=config/Verify_PerScaleNegativeMetric_1.1.yaml
readonly PARENT_CONFIG_SHA256=b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79
readonly CORE=src/pathline_template_matching/dimensionless_deformation_fmt.py
readonly CORE_SHA256=5fc4acb47c52c6505737e661cac7f8f503c429c5d88910992655e83cdc53a649
readonly RUNNER=scripts/run_verify_dimensionless_deformation_fmt_1_1.py
readonly AGGREGATOR=scripts/aggregate_verify_dimensionless_deformation_fmt_1_1.py
readonly EXPECTED_FOLD_COMMIT=${EXPECTED_FOLD_COMMIT:?EXPECTED_FOLD_COMMIT is required}
readonly TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
readonly -a OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)

[[ "$EXPECTED_FOLD_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "EXPECTED_FOLD_COMMIT must be a lowercase 40-character Git commit" >&2
  exit 2
}
[[ "$TASK_ID" =~ ^[0-4]$ ]] || {
  echo "array task must be one of 0,1,2,3,4: $TASK_ID" >&2
  exit 3
}
readonly OUTER_FAMILY=${OUTER_FAMILIES[$TASK_ID]}

cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain)" ]] || {
  echo "worktree contains tracked or untracked changes; refusing experiment" >&2
  exit 4
}
readonly COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
[[ "$COMMIT_ID" == "$EXPECTED_FOLD_COMMIT" ]] || {
  echo "checkout does not match EXPECTED_FOLD_COMMIT: $COMMIT_ID" >&2
  exit 5
}

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONOPTIMIZE
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
readonly JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_dimensionless_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

for binding in \
  "$CONFIG:$CONFIG_SHA256" \
  "$PARENT_CONFIG:$PARENT_CONFIG_SHA256" \
  "$CORE:$CORE_SHA256"; do
  path=${binding%%:*}
  expected=${binding##*:}
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "frozen SHA-256 mismatch for $path: $actual" >&2
    exit 6
  }
done
for source_path in "$RUNNER" "$AGGREGATOR"; do
  worktree_sha=$(sha256sum "$source_path" | awk '{print $1}')
  committed_sha=$(git show "${EXPECTED_FOLD_COMMIT}:${source_path}" | sha256sum | awk '{print $1}')
  [[ "$worktree_sha" == "$committed_sha" ]] || {
    echo "worktree/commit source mismatch for $source_path" >&2
    exit 7
  }
done

# Read every schema and method identity from the committed runner itself.  The
# wrapper never repeats schema literals that can drift from the real producer.
python - "$CONFIG" "$CONFIG_SHA256" "$PARENT_CONFIG_SHA256" "$CORE_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import run_verify_dimensionless_deformation_fmt_1_1 as runner

config, config_sha, parent_sha, core_sha = sys.argv[1:5]
plan = runner.load_plan(Path(config))
assert runner.EXPECTED_CONFIG_SHA256 == config_sha == plan.sha256
assert runner.EXPECTED_PARENT_CONFIG_SHA256 == parent_sha
assert runner.EXPECTED_CORE_SHA256 == core_sha == plan.core_sha256
assert plan.parent_experiment_config_sha256 == parent_sha
assert plan.required_fold_files == runner.REQUIRED_FOLD_FILES
assert len(runner.REQUIRED_FOLD_FILES) == len(set(runner.REQUIRED_FOLD_FILES)) == 15
for name in (
    "SCALER_ARTIFACT_SCHEMA", "SCALER_MANIFEST_SCHEMA",
    "CALIBRATION_ARTIFACT_SCHEMA", "CALIBRATION_MANIFEST_SCHEMA",
    "SELECTED_SCHEMA", "PREDICTION_SCHEMA", "PREDICTION_MANIFEST_SCHEMA",
    "OUTER_SUMMARY_SCHEMA", "REFERENCE_AUDIT_SCHEMA", "RESULT_SCHEMA",
    "COMPLETE_SCHEMA",
):
    value = getattr(runner, name)
    assert isinstance(value, str) and value
PY

if [[ "$TASK_ID" != 0 ]]; then
  readonly FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required for remaining folds}
  readonly FIRST_FOLD_AUTH_DIR=${FIRST_FOLD_AUTH_DIR:?FIRST_FOLD_AUTH_DIR is required for remaining folds}
  readonly FIRST_FOLD_AUTH_COMPLETE_SHA256=${FIRST_FOLD_AUTH_COMPLETE_SHA256:?FIRST_FOLD_AUTH_COMPLETE_SHA256 is required for remaining folds}
  [[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "FIRST_FOLD_JOB_ID must be numeric" >&2
    exit 8
  }
  readonly FIRST_FOLD_DIR="$EXPERIMENT_ROOT/runs/slurm_${FIRST_FOLD_JOB_ID}_0_${EXPECTED_FOLD_COMMIT:0:12}_outer_half_cylinder"
  python - "$FIRST_FOLD_AUTH_DIR" "$FIRST_FOLD_AUTH_COMPLETE_SHA256" "$EXPECTED_FOLD_COMMIT" "$CONFIG_SHA256" "$FIRST_FOLD_DIR" <<'PY'
from pathlib import Path
import sys

from scripts import aggregate_verify_dimensionless_deformation_fmt_1_1 as aggregate

release = aggregate.authenticate_single_fold_release(
    Path(sys.argv[1]),
    expected_completion_sha256=sys.argv[2],
    expected_fold_commit=sys.argv[3],
    expected_config_sha256=sys.argv[4],
    expected_fold_directory=Path(sys.argv[5]),
)
assert release["stop_version"] is False, (
    "authenticated mathematical certificate forbids remaining folds"
)
print("first_fold_authenticated_release=continue_remaining_folds")
PY
fi

readonly SHORT_COMMIT=${EXPECTED_FOLD_COMMIT:0:12}
readonly ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
readonly RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${ARRAY_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${OUTER_FAMILY}"
[[ ! -e "$RUN_DIR" ]] || {
  echo "immutable fold output already exists: $RUN_DIR" >&2
  exit 9
}

echo "experiment=Verify_DimensionlessDeformationFMT_1.1"
echo "phase=cpu_outer_fold_after_frozen_release_gate"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "outer_family=$OUTER_FAMILY"
echo "run_dir=$RUN_DIR"
hostname
lscpu

python -c 'assert __debug__, "Python assertions must remain enabled"'
/usr/bin/time -v python tests/test_all.py
/usr/bin/time -v python scripts/validate_matcher_backend.py --device cpu
[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain)" ]] || {
  echo "preflight changed the commit or worktree" >&2
  exit 10
}

/usr/bin/time -v python "$RUNNER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --outer-family "$OUTER_FAMILY" \
  --output-dir "$RUN_DIR" \
  --device cpu

# Check only the label-free producer contract here; performance acceptance is
# reserved for the separate independent aggregator and its fresh replay.
python - "$RUN_DIR" "$CONFIG" <<'PY'
from pathlib import Path
import sys

from scripts import run_verify_dimensionless_deformation_fmt_1_1 as runner

root = Path(sys.argv[1]).resolve()
plan = runner.load_plan(Path(sys.argv[2]))
assert {path.name for path in root.iterdir()} == set(runner.REQUIRED_FOLD_FILES)
assert plan.required_fold_files == runner.REQUIRED_FOLD_FILES
PY

[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain)" ]] || {
  echo "commit or clean worktree changed during experiment" >&2
  exit 11
}
echo "fold_status=completed_pending_independent_aggregate_authentication"
