#!/bin/bash
# Shared fail-closed gates for Verify_SourceCenteredPairedScaleTemplate_1.1.

set -euo pipefail

readonly SOURCE_CENTERED_PROJECT_ROOT="${SOURCE_CENTERED_PROJECT_ROOT:-/home/zhanx0o/pathline-template-matching-source-centered}"
readonly SOURCE_CENTERED_EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_SourceCenteredPairedScaleTemplate_1.1
readonly SOURCE_CENTERED_CONFIG=config/Verify_SourceCenteredPairedScaleTemplate_1.1.yaml
readonly SOURCE_CENTERED_CONFIG_SHA256=15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc
readonly SOURCE_CENTERED_PREPARER=scripts/prepare_verify_source_centered_paired_scale_template_1_1.py
readonly SOURCE_CENTERED_RUNNER=scripts/run_verify_source_centered_paired_scale_template_1_1.py
readonly SOURCE_CENTERED_AGGREGATOR=scripts/aggregate_verify_source_centered_paired_scale_template_1_1.py
readonly SOURCE_CENTERED_COMMON=ibex/verify_source_centered_paired_scale_template_1.1_common.sh
readonly SOURCE_CENTERED_SIDECAR_ROOT="$SOURCE_CENTERED_EXPERIMENT_ROOT/source_centered_cache/train"
readonly SOURCE_CENTERED_EARLY_INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/preparation/slurm_51068863_fd0412dc134d/kinematic_input_manifest.json
readonly SOURCE_CENTERED_EARLY_INPUT_SHA256=1b9df53a9010c6c3c46345639cfbf1d5ab2fe3a43187c79c7dfa0f7d840b102f
readonly SOURCE_CENTERED_REMOTE_URL=git@github.com:Cindy-xdZhang/pathline-template-matching.git

readonly -a SOURCE_CENTERED_IDENTITY_SOURCES=(
  config/Verify_SourceCenteredPairedScaleTemplate_1.1.yaml
  config/Verify_EarlyOppositePairKinematics_1.1.yaml
  config/mainExp_TemplateMatching_3.1.yaml
  src/pathline_template_matching/source_centered_seed_time_kinematics.py
  src/pathline_template_matching/source_centered_sidecar.py
  src/pathline_template_matching/paired_scale_center_fusion.py
  src/pathline_template_matching/per_scale_negative_metric.py
  src/pathline_template_matching/early_opposite_pair_kinematics.py
  src/pathline_template_matching/nested_scale_validation.py
  src/pathline_template_matching/portable_flow.py
  scripts/run_verify_early_opposite_pair_kinematics_1_1.py
  tests/test_source_centered_seed_time_kinematics.py
  tests/test_source_centered_sidecar.py
  tests/test_paired_scale_center_fusion.py
  tests/test_source_centered_aggregate.py
  tests/test_source_centered_runner_contract.py
  tests/test_source_centered_ibex.py
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

ptm_require_job_id() {
  local value=$1
  local name=$2
  [[ "$value" =~ ^[0-9]+$ ]] || ptm_die "$name must be numeric"
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
  [[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || \
    ptm_die "EXPECTED_GIT_COMMIT must be lowercase 40-hex"
  ptm_reject_confirmation_value "$SOURCE_CENTERED_PROJECT_ROOT" "project root"
  ptm_reject_confirmation_value "$SOURCE_CENTERED_EXPERIMENT_ROOT" "experiment root"
  cd "$SOURCE_CENTERED_PROJECT_ROOT"
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || \
    ptm_die "worktree is not clean"
  local observed_commit
  observed_commit=$(git rev-parse --verify HEAD^{commit})
  [[ "$observed_commit" == "$expected_commit" ]] || \
    ptm_die "checkout commit differs from EXPECTED_GIT_COMMIT"
  git cat-file -e "${expected_commit}^{commit}"
  local origin_url
  origin_url=$(git remote get-url origin)
  [[ "$origin_url" == "$SOURCE_CENTERED_REMOTE_URL" ]] || \
    ptm_die "origin URL differs from the frozen SSH repository"
  git show-ref --verify --quiet refs/remotes/origin/main || \
    ptm_die "origin/main is unavailable; clone or fetch the pushed revision first"
  git merge-base --is-ancestor "$expected_commit" refs/remotes/origin/main || \
    ptm_die "EXPECTED_GIT_COMMIT is not contained in the pushed origin/main history"
  ptm_require_file_sha256 \
    "$SOURCE_CENTERED_CONFIG" "$SOURCE_CENTERED_CONFIG_SHA256" "Verify config"

  local -a sources=(
    "$SOURCE_CENTERED_COMMON"
    "$wrapper_path"
    "$SOURCE_CENTERED_PREPARER"
    "$SOURCE_CENTERED_RUNNER"
    "$SOURCE_CENTERED_AGGREGATOR"
    "${SOURCE_CENTERED_IDENTITY_SOURCES[@]}"
    "$@"
  )
  local source actual committed
  declare -A seen=()
  for source in "${sources[@]}"; do
    [[ -n "$source" ]] || ptm_die "empty source path in stage gate"
    [[ -z "${seen[$source]:-}" ]] || continue
    seen[$source]=1
    git ls-files --error-unmatch "$source" >/dev/null
    actual=$(sha256sum "$source" | awk '{print $1}')
    committed=$(git show "${expected_commit}:${source}" | sha256sum | awk '{print $1}')
    [[ "$actual" == "$committed" ]] || \
      ptm_die "source differs from expected commit: $source"
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
  export PYTHONPATH="$SOURCE_CENTERED_PROJECT_ROOT/src:$SOURCE_CENTERED_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  unset PYTHONOPTIMIZE
  export OPENBLAS_NUM_THREADS="$threads"
  export OMP_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads"
  export NUMEXPR_NUM_THREADS="$threads"
  local job_identity=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
  local task_identity=${SLURM_ARRAY_TASK_ID:-single}
  export SOURCE_CENTERED_JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_source_centered_${stage_name}_${job_identity}_${task_identity}"
  export NUMBA_CACHE_DIR="$SOURCE_CENTERED_JOB_TMP_ROOT/numba_cache"
  mkdir -p "$NUMBA_CACHE_DIR"
  python -c 'assert __debug__, "Python assertions must remain enabled"'
}

ptm_targeted_preflight() {
  python -m py_compile \
    "$SOURCE_CENTERED_PREPARER" \
    "$SOURCE_CENTERED_RUNNER" \
    "$SOURCE_CENTERED_AGGREGATOR"
  python - <<'PY'
import importlib
import inspect

modules = (
    "tests.test_source_centered_seed_time_kinematics",
    "tests.test_source_centered_sidecar",
    "tests.test_paired_scale_center_fusion",
    "tests.test_source_centered_aggregate",
    "tests.test_source_centered_runner_contract",
    "tests.test_source_centered_ibex",
)
count = 0
for module_name in modules:
    module = importlib.import_module(module_name)
    for name in sorted(value for value in dir(module) if value.startswith("test_")):
        function = getattr(module, name)
        if inspect.signature(function).parameters:
            raise AssertionError(f"targeted test requires a fixture: {module_name}.{name}")
        function()
        count += 1
print(f"source_centered_targeted_tests={count}_passed")
PY
}

ptm_full_preflight() {
  python tests/test_all.py
}

ptm_stage_unchanged() {
  local wrapper_path=$1
  shift
  ptm_stage_gate "$wrapper_path" "$@"
}
