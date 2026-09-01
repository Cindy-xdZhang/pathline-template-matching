#!/bin/bash
# Shared fail-closed gates for Verify_ClassConditionalTemplateScore_1.1.

set -euo pipefail

readonly CLASS_PROJECT_ROOT="${CLASS_PROJECT_ROOT:-/home/zhanx0o/pathline-template-matching-class-conditional-score}"
readonly CLASS_EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1
readonly CLASS_CONFIG=config/Verify_ClassConditionalTemplateScore_1.1.yaml
readonly CLASS_CONFIG_SHA256=814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea
readonly CLASS_CORE=src/pathline_template_matching/class_conditional_template_score.py
readonly CLASS_RUNNER=scripts/run_verify_class_conditional_template_score_1_1.py
readonly CLASS_AGGREGATOR=scripts/aggregate_verify_class_conditional_template_score_1_1.py
readonly CLASS_SMOKE=scripts/run_verify_class_conditional_template_score_resource_smoke_1_1.py
readonly CLASS_COMMON=ibex/verify_class_conditional_template_score_1.1_common.sh
readonly CLASS_RUNTIME_SLURM_ACCOUNT=pi-hadwigm
readonly CLASS_RUNTIME_SLURM_PARTITION=batch

readonly CLASS_INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/preparation/slurm_51068863_fd0412dc134d/kinematic_input_manifest.json
readonly CLASS_INPUT_MANIFEST_SHA256=1b9df53a9010c6c3c46345639cfbf1d5ab2fe3a43187c79c7dfa0f7d840b102f
readonly CLASS_SYNTHETIC_PASS=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/preparation/slurm_51068863_fd0412dc134d/SYNTHETIC_PASS.json
readonly CLASS_SYNTHETIC_PASS_SHA256=78d0990352777e488f26bb84f3b0fc16e18845fc7cedb8a7d7fc598f32c0afe3
readonly CLASS_SIDECAR_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/kinematic_cache/train
readonly CLASS_POPULATION_MANIFEST="$CLASS_SIDECAR_ROOT/SIDECAR_POPULATION.json"
readonly CLASS_POPULATION_MANIFEST_SHA256=9f96835b9185218f40df4cc3c52bf3d80a93056681d922a30abfc5c0246f88a7

readonly -a CLASS_IDENTITY_SOURCES=(
  config/Verify_ClassConditionalTemplateScore_1.1.yaml
  config/Verify_EarlyOppositePairKinematics_1.1.yaml
  src/pathline_template_matching/class_conditional_template_score.py
  scripts/run_verify_class_conditional_template_score_1_1.py
  scripts/aggregate_verify_class_conditional_template_score_1_1.py
  scripts/run_verify_class_conditional_template_score_resource_smoke_1_1.py
  scripts/run_verify_early_opposite_pair_kinematics_1_1.py
  scripts/aggregate_verify_early_opposite_pair_kinematics_1_1.py
  tests/test_class_conditional_template_score.py
  tests/test_class_conditional_template_score_runner.py
  tests/test_class_conditional_template_score_aggregate.py
  tests/test_class_conditional_template_score_resource_smoke.py
  tests/test_class_conditional_template_score_ibex.py
  tests/test_all.py
)

ptm_class_die() {
  echo "$*" >&2
  exit 2
}

ptm_class_require_sha256() {
  local value=$1
  local name=$2
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || ptm_class_die "$name must be lowercase SHA-256"
}

ptm_class_reject_confirmation_value() {
  local value=$1
  local name=$2
  local normalized
  normalized=$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')
  if [[ "$normalized" == *tangaroa* || "$normalized" == *smokebuoyancy* ]]; then
    ptm_class_die "$name contains a forbidden confirmation dataset token"
  fi
}

ptm_class_require_file_sha256() {
  local path=$1
  local expected=$2
  local name=$3
  ptm_class_reject_confirmation_value "$path" "$name path"
  ptm_class_require_sha256 "$expected" "$name SHA-256"
  [[ -f "$path" ]] || ptm_class_die "$name is missing: $path"
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || ptm_class_die "$name SHA-256 mismatch: $actual"
}

ptm_class_stage_gate() {
  local wrapper_path=$1
  shift
  local expected_commit=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
  [[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || ptm_class_die "EXPECTED_GIT_COMMIT must be lowercase 40-hex"
  ptm_class_reject_confirmation_value "$CLASS_PROJECT_ROOT" "project root"
  ptm_class_reject_confirmation_value "$CLASS_EXPERIMENT_ROOT" "experiment root"
  cd "$CLASS_PROJECT_ROOT"
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || ptm_class_die "worktree is not clean"
  local observed_commit
  observed_commit=$(git rev-parse --verify HEAD^{commit})
  [[ "$observed_commit" == "$expected_commit" ]] || ptm_class_die "checkout differs from EXPECTED_GIT_COMMIT"
  git cat-file -e "${expected_commit}^{commit}"
  ptm_class_require_file_sha256 "$CLASS_CONFIG" "$CLASS_CONFIG_SHA256" "ClassConditional config"

  local -a sources=("$CLASS_COMMON" "$wrapper_path" "${CLASS_IDENTITY_SOURCES[@]}" "$@")
  local source actual committed
  declare -A seen=()
  for source in "${sources[@]}"; do
    [[ -n "$source" ]] || ptm_class_die "empty source path in stage gate"
    [[ -z "${seen[$source]:-}" ]] || continue
    seen[$source]=1
    git ls-files --error-unmatch "$source" >/dev/null
    actual=$(sha256sum "$source" | awk '{print $1}')
    committed=$(git show "${expected_commit}:${source}" | sha256sum | awk '{print $1}')
    [[ "$actual" == "$committed" ]] || ptm_class_die "source differs from expected commit: $source"
    echo "source_sha256[$source]=$actual"
  done
}

ptm_class_activate_runtime() {
  local stage_name=$1
  local threads=${2:-32}
  source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
  conda activate deepvortex
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$CLASS_PROJECT_ROOT/src:$CLASS_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  unset PYTHONOPTIMIZE
  export OPENBLAS_NUM_THREADS="$threads"
  export OMP_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads"
  export NUMEXPR_NUM_THREADS="$threads"
  local job_identity=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
  local task_identity=${SLURM_ARRAY_TASK_ID:-single}
  export CLASS_JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_class_${stage_name}_${job_identity}_${task_identity}"
  mkdir -p "$CLASS_JOB_TMP_ROOT"
  export TMPDIR="$CLASS_JOB_TMP_ROOT"
  export TMP="$CLASS_JOB_TMP_ROOT"
  export TEMP="$CLASS_JOB_TMP_ROOT"
  export NUMBA_CACHE_DIR="$CLASS_JOB_TMP_ROOT/numba_cache"
  mkdir -p "$NUMBA_CACHE_DIR"
  python -c 'import os, pathlib, tempfile; assert __debug__, "Python assertions must remain enabled"; assert pathlib.Path(tempfile.gettempdir()).resolve() == pathlib.Path(os.environ["CLASS_JOB_TMP_ROOT"]).resolve(), "Python tempfile escaped the job-local root"'
}

ptm_class_require_slurm_resources() {
  local job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
  command -v scontrol >/dev/null || ptm_class_die "scontrol is required to authenticate the Slurm allocation"
  local scontrol_record
  scontrol_record=$(scontrol show job -o "$job_id")
  [[ -n "$scontrol_record" ]] || ptm_class_die "scontrol returned an empty Slurm allocation"
  PTM_SLURM_SCONTROL_JOB_RECORD="$scontrol_record" python - "$job_id" <<'PY'
import os
import sys

from scripts.run_verify_class_conditional_template_score_resource_smoke_1_1 import (
    _validated_scontrol_allocation,
)

allocation = _validated_scontrol_allocation(
    os.environ.get("PTM_SLURM_SCONTROL_JOB_RECORD"),
    expected_job_id=sys.argv[1],
    expected_time_limit="12:00:00",
)
print(
    "slurm_allocation_authenticated="
    f"nodes={allocation['num_nodes']},"
    f"time={allocation['time_limit']},"
    f"features={allocation['features']},"
    f"gpu={allocation['gpu_allocation']},"
    f"record_sha256={allocation['record_sha256']}"
)
PY
  [[ "${SLURM_JOB_PARTITION:-}" == "$CLASS_RUNTIME_SLURM_PARTITION" ]] || \
    ptm_class_die "Slurm partition must be $CLASS_RUNTIME_SLURM_PARTITION"
  [[ "${SLURM_JOB_ACCOUNT:-}" == "$CLASS_RUNTIME_SLURM_ACCOUNT" ]] || \
    ptm_class_die "Slurm account must be $CLASS_RUNTIME_SLURM_ACCOUNT"
  [[ "${SLURM_CPUS_PER_TASK:-}" == 32 ]] || ptm_class_die "Slurm CPUs per task must be 32"
  [[ "${SLURM_MEM_PER_NODE:-}" == 131072 ]] || ptm_class_die "Slurm memory must be exactly 128 GiB"
}

ptm_class_targeted_preflight() {
  python -m py_compile \
    "$CLASS_CORE" \
    "$CLASS_RUNNER" \
    "$CLASS_AGGREGATOR" \
    "$CLASS_SMOKE" \
    tests/test_class_conditional_template_score.py \
    tests/test_class_conditional_template_score_runner.py \
    tests/test_class_conditional_template_score_aggregate.py \
    tests/test_class_conditional_template_score_resource_smoke.py \
    tests/test_class_conditional_template_score_ibex.py
  python - <<'PY'
import importlib
import inspect

modules = tuple(
    importlib.import_module(name)
    for name in (
        "tests.test_class_conditional_template_score",
        "tests.test_class_conditional_template_score_runner",
        "tests.test_class_conditional_template_score_aggregate",
        "tests.test_class_conditional_template_score_resource_smoke",
        "tests.test_class_conditional_template_score_ibex",
    )
)
tests = [
    getattr(module, name)
    for module in modules
    for name in sorted(value for value in dir(module) if value.startswith("test_"))
    if getattr(getattr(module, name), "__module__", None) == module.__name__
]
assert all(not inspect.signature(test).parameters for test in tests)
for test in tests:
    test()
print(f"class_conditional_targeted_tests={len(tests)}_passed")
PY
}

ptm_class_require_evidence() {
  ptm_class_require_file_sha256 "$CLASS_INPUT_MANIFEST" "$CLASS_INPUT_MANIFEST_SHA256" "kinematic input manifest"
  ptm_class_require_file_sha256 "$CLASS_SYNTHETIC_PASS" "$CLASS_SYNTHETIC_PASS_SHA256" "parent synthetic PASS"
  ptm_class_require_file_sha256 "$CLASS_POPULATION_MANIFEST" "$CLASS_POPULATION_MANIFEST_SHA256" "sealed sidecar population"
}

ptm_class_require_resource_smoke() {
  local pass_path=$1
  local pass_sha256=$2
  ptm_class_require_file_sha256 "$pass_path" "$pass_sha256" "ClassConditional resource-smoke PASS"
  python - "$pass_path" "$pass_sha256" "$EXPECTED_GIT_COMMIT" "$CLASS_CONFIG_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts.run_verify_class_conditional_template_score_resource_smoke_1_1 import (
    authenticate_resource_smoke_release,
)

result = authenticate_resource_smoke_release(
    Path(sys.argv[1]),
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
)
print(f"resource_smoke_release_authenticated={result['marker']['sha256']}")
PY
}

ptm_class_stage_unchanged() {
  local wrapper_path=$1
  shift
  ptm_class_stage_gate "$wrapper_path" "$@"
}
