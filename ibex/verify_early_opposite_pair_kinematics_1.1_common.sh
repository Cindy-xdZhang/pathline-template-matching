#!/bin/bash
# Shared fail-closed gates for Verify_EarlyOppositePairKinematics_1.1 stages.

set -euo pipefail

readonly EARLY_PROJECT_ROOT="${EARLY_PROJECT_ROOT:-/home/zhanx0o/pathline-template-matching-early-kinematics}"
readonly EARLY_EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1
readonly EARLY_CONFIG=config/Verify_EarlyOppositePairKinematics_1.1.yaml
readonly EARLY_CONFIG_SHA256=e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767
readonly EARLY_PREPARER=scripts/prepare_verify_early_opposite_pair_kinematics_1_1.py
readonly EARLY_PREPARER_SHA256=79303ed04c0885d56a57acaad1549a98eed5f88f6155e3ba432a8844ca5420d3
readonly EARLY_RUNNER=scripts/run_verify_early_opposite_pair_kinematics_1_1.py
readonly EARLY_RUNNER_SHA256=e999960ac06d3fedd355e1d6135d9e69316bfe1e798318a22dadf5a8e2063796
readonly EARLY_AGGREGATOR=scripts/aggregate_verify_early_opposite_pair_kinematics_1_1.py
readonly EARLY_AGGREGATOR_SHA256=2b5233f8e708d3242ae9191d2eb745aba3294f38b0656d3d81dde4a0074e02bf
readonly EARLY_SIDECAR_ROOT="$EARLY_EXPERIMENT_ROOT/kinematic_cache/train"
readonly EARLY_COMMON=ibex/verify_early_opposite_pair_kinematics_1.1_common.sh

readonly -a EARLY_IDENTITY_SOURCES=(
  config/Verify_EarlyOppositePairKinematics_1.1.yaml
  config/mainExp_TemplateMatching_3.1.yaml
  src/pathline_template_matching/early_opposite_pair_kinematics.py
  src/pathline_template_matching/seed_time_kinematic_sidecar.py
  src/pathline_template_matching/early_kinematic_preparation.py
  src/pathline_template_matching/portable_flow.py
  src/pathline_template_matching/nested_scale_validation.py
  src/pathline_template_matching/arc_length_primitives.py
  src/pathline_template_matching/vector_field.py
  src/pathline_template_matching/netcdf_io.py
)

ptm_die() {
  echo "$*" >&2
  exit 2
}

ptm_require_sha256() {
  local value=$1
  local name=$2
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || ptm_die "$name must be lowercase SHA-256"
}

ptm_reject_confirmation_value() {
  local value=$1
  local name=$2
  local normalized
  normalized=$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')
  if [[ "$normalized" == *tangaroa* || "$normalized" == *smokebuoyancy* ]]; then
    ptm_die "$name contains a forbidden confirmation dataset token"
  fi
}

ptm_require_file_sha256() {
  local path=$1
  local expected=$2
  local name=$3
  ptm_reject_confirmation_value "$path" "$name path"
  ptm_require_sha256 "$expected" "$name SHA-256"
  [[ -f "$path" ]] || ptm_die "$name does not exist: $path"
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || ptm_die "$name SHA-256 mismatch: $actual"
}

ptm_stage_gate() {
  local wrapper_path=$1
  shift
  local expected_commit=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
  [[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || ptm_die "EXPECTED_GIT_COMMIT must be lowercase 40-hex"
  ptm_reject_confirmation_value "$EARLY_PROJECT_ROOT" "project root"
  ptm_reject_confirmation_value "$EARLY_EXPERIMENT_ROOT" "experiment root"
  cd "$EARLY_PROJECT_ROOT"
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || ptm_die "worktree is not clean"
  local observed_commit
  observed_commit=$(git rev-parse --verify HEAD^{commit})
  [[ "$observed_commit" == "$expected_commit" ]] || ptm_die "checkout commit differs from EXPECTED_GIT_COMMIT"
  git cat-file -e "${expected_commit}^{commit}"
  ptm_require_file_sha256 "$EARLY_CONFIG" "$EARLY_CONFIG_SHA256" "Verify config"
  ptm_require_file_sha256 "$EARLY_PREPARER" "$EARLY_PREPARER_SHA256" "Early preparation CLI"
  ptm_require_file_sha256 "$EARLY_RUNNER" "$EARLY_RUNNER_SHA256" "Early nested runner"
  ptm_require_file_sha256 "$EARLY_AGGREGATOR" "$EARLY_AGGREGATOR_SHA256" "Early five-fold aggregator"

  local -a sources=("$EARLY_COMMON" "$wrapper_path" "${EARLY_IDENTITY_SOURCES[@]}" "$@")
  local source actual committed
  declare -A seen=()
  for source in "${sources[@]}"; do
    [[ -n "$source" ]] || ptm_die "empty source path in stage gate"
    [[ -z "${seen[$source]:-}" ]] || continue
    seen[$source]=1
    git ls-files --error-unmatch "$source" >/dev/null
    actual=$(sha256sum "$source" | awk '{print $1}')
    committed=$(git show "${expected_commit}:${source}" | sha256sum | awk '{print $1}')
    [[ "$actual" == "$committed" ]] || ptm_die "source differs from expected commit: $source"
    echo "source_sha256[$source]=$actual"
  done
}

ptm_activate_runtime() {
  local stage_name=$1
  local threads=${2:-4}
  source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
  conda activate deepvortex
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$EARLY_PROJECT_ROOT/src:$EARLY_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  unset PYTHONOPTIMIZE
  export OPENBLAS_NUM_THREADS="$threads"
  export OMP_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads"
  export NUMEXPR_NUM_THREADS="$threads"
  local job_identity=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
  local task_identity=${SLURM_ARRAY_TASK_ID:-single}
  export EARLY_JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_early_${stage_name}_${job_identity}_${task_identity}"
  export NUMBA_CACHE_DIR="$EARLY_JOB_TMP_ROOT/numba_cache"
  mkdir -p "$NUMBA_CACHE_DIR"
  python -c 'assert __debug__, "Python assertions must remain enabled"'
}

ptm_targeted_preflight() {
  python -m py_compile \
    "$EARLY_PREPARER" \
    "$EARLY_RUNNER" \
    "$EARLY_AGGREGATOR"
  python -c 'import tests.test_early_opposite_pair_kinematics_runner as t; names=sorted(n for n in dir(t) if n.startswith("test_")); [getattr(t,n)() for n in names]; print(f"early_targeted_tests={len(names)}_passed")'
}

ptm_full_preflight() {
  python tests/test_all.py
}

ptm_stage_unchanged() {
  local wrapper_path=$1
  shift
  ptm_stage_gate "$wrapper_path" "$@"
}
