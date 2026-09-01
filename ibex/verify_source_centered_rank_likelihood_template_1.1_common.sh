#!/bin/bash
# Shared fail-closed gates for Verify_SourceCenteredRankLikelihoodTemplate_1.1.

set -euo pipefail

readonly RANK_PROJECT_ROOT="${RANK_PROJECT_ROOT:-/home/zhanx0o/pathline-template-matching-rank-likelihood}"
readonly RANK_EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_SourceCenteredRankLikelihoodTemplate_1.1
readonly RANK_CONFIG=config/Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml
readonly RANK_CONFIG_SHA256=41d6e7be70b898715c6df6f92cfb17176d2f1bb6153fa37b09dd4da9a6059ffa
readonly RANK_PREPARER=scripts/prepare_verify_source_centered_rank_likelihood_template_1_1.py
readonly RANK_RUNNER=scripts/run_verify_source_centered_rank_likelihood_template_1_1.py
readonly RANK_AGGREGATOR=scripts/aggregate_verify_source_centered_rank_likelihood_template_1_1.py
readonly RANK_COMMON=ibex/verify_source_centered_rank_likelihood_template_1.1_common.sh
readonly RANK_REMOTE_URL=git@github.com:Cindy-xdZhang/pathline-template-matching.git

readonly -a RANK_IDENTITY_SOURCES=(
  config/Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml
  config/Verify_SourceCenteredPairedScaleTemplate_1.1.yaml
  src/pathline_template_matching/source_centered_rank_likelihood.py
  src/pathline_template_matching/source_centered_sidecar.py
  src/pathline_template_matching/source_centered_seed_time_kinematics.py
  src/pathline_template_matching/paired_scale_center_fusion.py
  src/pathline_template_matching/portable_flow.py
  scripts/run_verify_source_centered_paired_scale_template_1_1.py
  tests/test_source_centered_rank_likelihood.py
  tests/test_source_centered_rank_likelihood_runner.py
  tests/test_source_centered_rank_likelihood_aggregate.py
  tests/test_source_centered_rank_likelihood_ibex.py
)

rank_die() {
  echo "$*" >&2
  exit 2
}

rank_require_sha256() {
  local value=$1
  local name=$2
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || rank_die "$name must be lowercase SHA-256"
}

rank_require_job_id() {
  local value=$1
  local name=$2
  [[ "$value" =~ ^[0-9]+$ ]] || rank_die "$name must be numeric"
}

rank_reject_confirmation_value() {
  local value=$1
  local name=$2
  local normalized
  normalized=$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')
  if [[ "$normalized" == *tangaroa* || "$normalized" == *smokebuoyancy* ]]; then
    rank_die "$name contains a forbidden confirmation dataset token"
  fi
}

rank_require_file_sha256() {
  local path=$1
  local expected=$2
  local name=$3
  rank_reject_confirmation_value "$path" "$name path"
  rank_require_sha256 "$expected" "$name SHA-256"
  [[ -f "$path" ]] || rank_die "$name does not exist: $path"
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || rank_die "$name SHA-256 mismatch: $actual"
}

rank_stage_gate() {
  local wrapper_path=$1
  shift
  local expected_commit=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
  [[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || \
    rank_die "EXPECTED_GIT_COMMIT must be lowercase 40-hex"
  rank_reject_confirmation_value "$RANK_PROJECT_ROOT" "project root"
  rank_reject_confirmation_value "$RANK_EXPERIMENT_ROOT" "experiment root"
  cd "$RANK_PROJECT_ROOT"
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || \
    rank_die "worktree is not clean"
  local observed_commit
  observed_commit=$(git rev-parse --verify HEAD^{commit})
  [[ "$observed_commit" == "$expected_commit" ]] || \
    rank_die "checkout commit differs from EXPECTED_GIT_COMMIT"
  git cat-file -e "${expected_commit}^{commit}"
  local origin_url
  origin_url=$(git remote get-url origin)
  [[ "$origin_url" == "$RANK_REMOTE_URL" ]] || \
    rank_die "origin URL differs from the frozen SSH repository"
  git show-ref --verify --quiet refs/remotes/origin/main || \
    rank_die "origin/main is unavailable; clone or fetch the pushed revision first"
  git merge-base --is-ancestor "$expected_commit" refs/remotes/origin/main || \
    rank_die "EXPECTED_GIT_COMMIT is not contained in pushed origin/main"
  rank_require_file_sha256 "$RANK_CONFIG" "$RANK_CONFIG_SHA256" "Verify config"

  local -a sources=(
    "$RANK_COMMON"
    "$wrapper_path"
    "$RANK_PREPARER"
    "$RANK_RUNNER"
    "$RANK_AGGREGATOR"
    "${RANK_IDENTITY_SOURCES[@]}"
    "$@"
  )
  local source actual committed
  declare -A seen=()
  for source in "${sources[@]}"; do
    [[ -n "$source" ]] || rank_die "empty source path in stage gate"
    [[ -z "${seen[$source]:-}" ]] || continue
    seen[$source]=1
    git ls-files --error-unmatch "$source" >/dev/null
    actual=$(sha256sum "$source" | awk '{print $1}')
    committed=$(git show "${expected_commit}:${source}" | sha256sum | awk '{print $1}')
    [[ "$actual" == "$committed" ]] || \
      rank_die "source differs from expected commit: $source"
    echo "source_sha256[$source]=$actual"
  done
}

rank_activate_runtime() {
  local stage_name=$1
  local threads=${2:-4}
  source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
  conda activate deepvortex
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$RANK_PROJECT_ROOT/src:$RANK_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  unset PYTHONOPTIMIZE
  export OPENBLAS_NUM_THREADS="$threads"
  export OMP_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads"
  export NUMEXPR_NUM_THREADS="$threads"
  local job_identity=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
  local task_identity=${SLURM_ARRAY_TASK_ID:-single}
  export RANK_JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_rank_${stage_name}_${job_identity}_${task_identity}"
  export NUMBA_CACHE_DIR="$RANK_JOB_TMP_ROOT/numba_cache"
  mkdir -p "$NUMBA_CACHE_DIR"
  python -c 'assert __debug__, "Python assertions must remain enabled"'
}

rank_targeted_preflight() {
  python -m py_compile "$RANK_PREPARER" "$RANK_RUNNER" "$RANK_AGGREGATOR"
  python -m pytest -q \
    tests/test_source_centered_rank_likelihood.py \
    tests/test_source_centered_rank_likelihood_runner.py \
    tests/test_source_centered_rank_likelihood_aggregate.py \
    tests/test_source_centered_rank_likelihood_ibex.py
}

rank_full_preflight() {
  python tests/test_all.py
}

rank_authenticate_parent_binding() {
  local binding=$1
  local binding_sha=$2
  local completion=$3
  local completion_sha=$4
  rank_require_file_sha256 "$binding" "$binding_sha" "parent sidecar binding"
  rank_require_file_sha256 "$completion" "$completion_sha" "binding completion"
  python "$RANK_PREPARER" \
    --config "$RANK_CONFIG" \
    --expected-git-commit "$EXPECTED_GIT_COMMIT" \
    authenticate \
    --parent-binding "$binding" \
    --parent-binding-sha256 "$binding_sha" \
    --binding-completion "$completion" \
    --binding-completion-sha256 "$completion_sha"
}

rank_stage_unchanged() {
  local wrapper_path=$1
  shift
  rank_stage_gate "$wrapper_path" "$@"
}
