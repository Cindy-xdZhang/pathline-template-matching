#!/bin/bash
# Shared fail-closed gates for the post-stop Boeing-only diagnostic.

set -euo pipefail

readonly BOEING_DIAG_PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-class-conditional-boeing
readonly BOEING_DIAG_VERIFY_PRODUCER_ROOT=/home/zhanx0o/pathline-template-matching-class-conditional-score
readonly BOEING_DIAG_VERIFY_AUDIT_CONFIG_PATH=/home/zhanx0o/pathline-template-matching-class-conditional-score/config/Verify_ClassConditionalTemplateScore_1.1.yaml
readonly BOEING_DIAG_EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1
readonly BOEING_DIAG_CONFIG=config/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.yaml
readonly BOEING_DIAG_CONFIG_SHA256=6112e7588efecf29cf2690b270385053d8ccd94f8e11037a6e247815afcc5856
readonly BOEING_DIAG_CORE=src/pathline_template_matching/class_conditional_template_score.py
readonly BOEING_DIAG_RUNNER=scripts/run_other_class_conditional_template_score_boeing_diagnostic_1_1.py
readonly BOEING_DIAG_AGGREGATOR=scripts/aggregate_other_class_conditional_template_score_boeing_diagnostic_1_1.py
readonly BOEING_DIAG_COMMON=ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_common.sh
readonly BOEING_DIAG_FOLD_WRAPPER=ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_fold.sh
readonly BOEING_DIAG_AUTH_WRAPPER=ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_auth.sh
readonly BOEING_DIAG_RUNTIME_SLURM_ACCOUNT=pi-hadwigm
readonly BOEING_DIAG_RUNTIME_SLURM_PARTITION=batch

readonly BOEING_DIAG_VERIFY_CONFIG=config/Verify_ClassConditionalTemplateScore_1.1.yaml
readonly BOEING_DIAG_VERIFY_RUNNER=scripts/run_verify_class_conditional_template_score_1_1.py
readonly BOEING_DIAG_VERIFY_AGGREGATOR=scripts/aggregate_verify_class_conditional_template_score_1_1.py
readonly BOEING_DIAG_VERIFY_RESOURCE_AUTHENTICATOR=scripts/run_verify_class_conditional_template_score_resource_smoke_1_1.py
readonly BOEING_DIAG_VERIFY_EXECUTION_COMMIT=58b0bc0b0c7385f1b356eb343a150fcd50dad94f
readonly BOEING_DIAG_VERIFY_CONFIG_SHA256=814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea
readonly BOEING_DIAG_VERIFY_CORE_SHA256=9c009376f7cea1481f6f47a49362d54d0e78530717f480fda3e8a109f841ef99
readonly BOEING_DIAG_VERIFY_RUNNER_SHA256=e5063887475029320e66da1f1eb221d7988598e8918d37fbe47ee213e5ff1b48
readonly BOEING_DIAG_VERIFY_AGGREGATOR_SHA256=77a561930ca85e3c1e6193a12e27b0b61bf7cc99be96889070962b8bfaf04e9c
readonly BOEING_DIAG_VERIFY_RESOURCE_AUTHENTICATOR_SHA256=97f02e58cf571e81466fc1d14bbc605d89c48e7d79fe0d6af86f0fdb0e780371
readonly BOEING_DIAG_VERIFY_RESOURCE_TEST_SHA256=31b0de79c34ccf13b2ff8559801d144d816212009cc07785e37cc7f5403908c7
readonly BOEING_DIAG_VERIFY_RESOURCE_WRAPPER_SHA256=64c77ae20ed97edba0a57152d2fe0c51ca925b19f2c156a5c504bbd0e830c667
readonly BOEING_DIAG_VERIFY_FIRST_FOLD_JOB_ID=51146327
readonly BOEING_DIAG_VERIFY_FIRST_FOLD_DIR=/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1/runs/slurm_51146327_0_58b0bc0b0c73_outer_half_cylinder
readonly BOEING_DIAG_VERIFY_AUTH_DIR=/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1/aggregate/slurm_51146768_58b0bc0b0c73
readonly BOEING_DIAG_VERIFY_AUTH_COMPLETE_SHA256=f8515858efe531c24471a11f64f014692a5d4774146c8908f07ee4ca49476844
readonly BOEING_DIAG_VERIFY_RESOURCE_PASS=/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1/resource_smoke/slurm_51146125_58b0bc0b0c73/RESOURCE_SMOKE_PASS.json
readonly BOEING_DIAG_VERIFY_RESOURCE_PASS_SHA256=3f9197a19407906b0b13a2b9eaa09dbc647b166a9fe9d2ef4dc90cda532557ea

readonly BOEING_DIAG_INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/preparation/slurm_51068863_fd0412dc134d/kinematic_input_manifest.json
readonly BOEING_DIAG_INPUT_MANIFEST_SHA256=1b9df53a9010c6c3c46345639cfbf1d5ab2fe3a43187c79c7dfa0f7d840b102f
readonly BOEING_DIAG_SYNTHETIC_PASS=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/preparation/slurm_51068863_fd0412dc134d/SYNTHETIC_PASS.json
readonly BOEING_DIAG_SYNTHETIC_PASS_SHA256=78d0990352777e488f26bb84f3b0fc16e18845fc7cedb8a7d7fc598f32c0afe3
readonly BOEING_DIAG_SIDECAR_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/kinematic_cache/train
readonly BOEING_DIAG_POPULATION_MANIFEST="$BOEING_DIAG_SIDECAR_ROOT/SIDECAR_POPULATION.json"
readonly BOEING_DIAG_POPULATION_MANIFEST_SHA256=9f96835b9185218f40df4cc3c52bf3d80a93056681d922a30abfc5c0246f88a7

readonly -a BOEING_DIAG_IDENTITY_SOURCES=(
  config/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.yaml
  config/Verify_ClassConditionalTemplateScore_1.1.yaml
  src/pathline_template_matching/class_conditional_template_score.py
  scripts/run_other_class_conditional_template_score_boeing_diagnostic_1_1.py
  scripts/aggregate_other_class_conditional_template_score_boeing_diagnostic_1_1.py
  scripts/run_verify_class_conditional_template_score_1_1.py
  scripts/aggregate_verify_class_conditional_template_score_1_1.py
  scripts/run_verify_class_conditional_template_score_resource_smoke_1_1.py
  ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_common.sh
  ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_fold.sh
  ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_auth.sh
  tests/test_other_class_conditional_template_score_boeing_diagnostic_runner.py
  tests/test_other_class_conditional_template_score_boeing_diagnostic_aggregate.py
  tests/test_other_class_conditional_template_score_boeing_diagnostic_ibex.py
  tests/test_class_conditional_template_score.py
  tests/test_class_conditional_template_score_runner.py
  tests/test_class_conditional_template_score_aggregate.py
  tests/test_class_conditional_template_score_resource_smoke.py
  tests/test_all.py
)

ptm_boeing_diag_die() {
  echo "$*" >&2
  exit 2
}

ptm_boeing_diag_require_sha256() {
  local value=$1
  local name=$2
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || ptm_boeing_diag_die "$name must be lowercase SHA-256"
}

ptm_boeing_diag_reject_confirmation_value() {
  local value=$1
  local name=$2
  local normalized
  normalized=$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')
  if [[ "$normalized" == *tangaroa* || "$normalized" == *smokebuoyancy* ]]; then
    ptm_boeing_diag_die "$name contains a forbidden confirmation dataset token"
  fi
}

ptm_boeing_diag_require_file_sha256() {
  local path=$1
  local expected=$2
  local name=$3
  ptm_boeing_diag_reject_confirmation_value "$path" "$name path"
  ptm_boeing_diag_require_sha256 "$expected" "$name SHA-256"
  [[ -f "$path" ]] || ptm_boeing_diag_die "$name is missing: $path"
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || ptm_boeing_diag_die "$name SHA-256 mismatch: $actual"
}

ptm_boeing_diag_require_parent_source_identity() {
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_VERIFY_CONFIG" "$BOEING_DIAG_VERIFY_CONFIG_SHA256" \
    "inherited Verify config"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_CORE" "$BOEING_DIAG_VERIFY_CORE_SHA256" \
    "inherited class-conditional core"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_VERIFY_RUNNER" "$BOEING_DIAG_VERIFY_RUNNER_SHA256" \
    "inherited Verify runner"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_VERIFY_AGGREGATOR" "$BOEING_DIAG_VERIFY_AGGREGATOR_SHA256" \
    "authenticated Verify aggregator"
}

ptm_boeing_diag_require_verify_producer_source_identity() {
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_VERIFY_PRODUCER_ROOT/$BOEING_DIAG_VERIFY_CONFIG" \
    "$BOEING_DIAG_VERIFY_CONFIG_SHA256" "producer Verify config"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_VERIFY_PRODUCER_ROOT/$BOEING_DIAG_CORE" \
    "$BOEING_DIAG_VERIFY_CORE_SHA256" "producer class-conditional core"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_VERIFY_PRODUCER_ROOT/$BOEING_DIAG_VERIFY_RUNNER" \
    "$BOEING_DIAG_VERIFY_RUNNER_SHA256" "producer Verify runner"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_VERIFY_PRODUCER_ROOT/$BOEING_DIAG_VERIFY_AGGREGATOR" \
    "$BOEING_DIAG_VERIFY_AGGREGATOR_SHA256" "producer Verify aggregator"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_VERIFY_PRODUCER_ROOT/$BOEING_DIAG_VERIFY_RESOURCE_AUTHENTICATOR" \
    "$BOEING_DIAG_VERIFY_RESOURCE_AUTHENTICATOR_SHA256" \
    "producer resource authenticator"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_VERIFY_PRODUCER_ROOT/tests/test_class_conditional_template_score_resource_smoke.py" \
    "$BOEING_DIAG_VERIFY_RESOURCE_TEST_SHA256" "producer resource test"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_VERIFY_PRODUCER_ROOT/ibex/verify_class_conditional_template_score_1.1_resource_smoke.sh" \
    "$BOEING_DIAG_VERIFY_RESOURCE_WRAPPER_SHA256" "producer resource wrapper"
}

ptm_boeing_diag_require_verify_producer_checkout() {
  [[ -d "$BOEING_DIAG_VERIFY_PRODUCER_ROOT/.git" ]] || \
    ptm_boeing_diag_die "immutable Verify producer checkout is missing"
  local producer_root producer_head
  producer_root=$(git -C "$BOEING_DIAG_VERIFY_PRODUCER_ROOT" rev-parse --show-toplevel)
  [[ "$producer_root" == "$BOEING_DIAG_VERIFY_PRODUCER_ROOT" ]] || \
    ptm_boeing_diag_die "Verify producer checkout resolved to the wrong absolute root"
  producer_head=$(git -C "$BOEING_DIAG_VERIFY_PRODUCER_ROOT" rev-parse --verify HEAD^{commit})
  [[ "$producer_head" == "$BOEING_DIAG_VERIFY_EXECUTION_COMMIT" ]] || \
    ptm_boeing_diag_die "Verify producer checkout is not at the frozen execution commit"
  if git -C "$BOEING_DIAG_VERIFY_PRODUCER_ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then
    ptm_boeing_diag_die "Verify producer checkout must remain detached"
  fi
  [[ -z "$(git -C "$BOEING_DIAG_VERIFY_PRODUCER_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || \
    ptm_boeing_diag_die "Verify producer checkout is not clean"
  git -C "$BOEING_DIAG_VERIFY_PRODUCER_ROOT" cat-file -e \
    "${BOEING_DIAG_VERIFY_EXECUTION_COMMIT}^{commit}"
  ptm_boeing_diag_require_verify_producer_source_identity
}

ptm_boeing_diag_stage_gate() {
  local expected_commit=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
  [[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || \
    ptm_boeing_diag_die "EXPECTED_GIT_COMMIT must be lowercase 40-hex"
  ptm_boeing_diag_reject_confirmation_value "$BOEING_DIAG_PROJECT_ROOT" "project root"
  ptm_boeing_diag_reject_confirmation_value \
    "$BOEING_DIAG_VERIFY_PRODUCER_ROOT" "Verify producer root"
  ptm_boeing_diag_reject_confirmation_value "$BOEING_DIAG_EXPERIMENT_ROOT" "experiment root"
  [[ "$BOEING_DIAG_PROJECT_ROOT" != "$BOEING_DIAG_VERIFY_PRODUCER_ROOT" ]] || \
    ptm_boeing_diag_die "Other and Verify producer checkouts must be separate"
  cd "$BOEING_DIAG_PROJECT_ROOT"
  [[ "$(pwd -P)" == "$BOEING_DIAG_PROJECT_ROOT" ]] || \
    ptm_boeing_diag_die "Other checkout resolved to the wrong absolute root"
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || \
    ptm_boeing_diag_die "worktree is not clean"
  local observed_commit
  observed_commit=$(git rev-parse --verify HEAD^{commit})
  [[ "$observed_commit" == "$expected_commit" ]] || \
    ptm_boeing_diag_die "checkout differs from EXPECTED_GIT_COMMIT"
  git cat-file -e "${expected_commit}^{commit}"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_CONFIG" "$BOEING_DIAG_CONFIG_SHA256" "Boeing diagnostic config"
  ptm_boeing_diag_require_parent_source_identity

  local source actual committed
  for source in "${BOEING_DIAG_IDENTITY_SOURCES[@]}"; do
    git ls-files --error-unmatch "$source" >/dev/null
    actual=$(sha256sum "$source" | awk '{print $1}')
    committed=$(git show "${expected_commit}:${source}" | sha256sum | awk '{print $1}')
    [[ "$actual" == "$committed" ]] || \
      ptm_boeing_diag_die "source differs from expected commit: $source"
    echo "source_sha256[$source]=$actual"
  done
}

ptm_boeing_diag_activate_runtime() {
  local stage_name=$1
  local threads=${2:-32}
  source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
  conda activate deepvortex
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$BOEING_DIAG_PROJECT_ROOT/src:$BOEING_DIAG_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  unset PYTHONOPTIMIZE
  export OPENBLAS_NUM_THREADS="$threads"
  export OMP_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads"
  export NUMEXPR_NUM_THREADS="$threads"
  local job_identity=${SLURM_JOB_ID:-manual}
  export BOEING_DIAG_JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_boeing_diag_${stage_name}_${job_identity}"
  mkdir -p "$BOEING_DIAG_JOB_TMP_ROOT"
  export TMPDIR="$BOEING_DIAG_JOB_TMP_ROOT"
  export TMP="$BOEING_DIAG_JOB_TMP_ROOT"
  export TEMP="$BOEING_DIAG_JOB_TMP_ROOT"
  export NUMBA_CACHE_DIR="$BOEING_DIAG_JOB_TMP_ROOT/numba_cache"
  mkdir -p "$NUMBA_CACHE_DIR"
  python -c 'import os, pathlib, tempfile; assert __debug__, "Python assertions must remain enabled"; assert pathlib.Path(tempfile.gettempdir()).resolve() == pathlib.Path(os.environ["BOEING_DIAG_JOB_TMP_ROOT"]).resolve(), "Python tempfile escaped the job-local root"'
}

ptm_boeing_diag_require_slurm_resources() {
  local job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
  command -v scontrol >/dev/null || \
    ptm_boeing_diag_die "scontrol is required to authenticate the Slurm allocation"
  local scontrol_record
  scontrol_record=$(scontrol show job -o "$job_id")
  [[ -n "$scontrol_record" ]] || \
    ptm_boeing_diag_die "scontrol returned an empty Slurm allocation"
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
    f"partition={allocation['partition']},"
    f"account={allocation['account']},"
    f"time={allocation['time_limit']},"
    f"features={allocation['features']},"
    f"gpu={allocation['gpu_allocation']},"
    f"record_sha256={allocation['record_sha256']}"
)
PY
  [[ "${SLURM_JOB_PARTITION:-}" == "$BOEING_DIAG_RUNTIME_SLURM_PARTITION" ]] || \
    ptm_boeing_diag_die "Slurm partition must be $BOEING_DIAG_RUNTIME_SLURM_PARTITION"
  [[ "${SLURM_JOB_ACCOUNT:-}" == "$BOEING_DIAG_RUNTIME_SLURM_ACCOUNT" ]] || \
    ptm_boeing_diag_die "Slurm account must be $BOEING_DIAG_RUNTIME_SLURM_ACCOUNT"
  [[ "${SLURM_CPUS_PER_TASK:-}" == 32 ]] || \
    ptm_boeing_diag_die "Slurm CPUs per task must be 32"
  [[ "${SLURM_MEM_PER_NODE:-}" == 131072 ]] || \
    ptm_boeing_diag_die "Slurm memory must be exactly 128 GiB"
}

ptm_boeing_diag_targeted_preflight() {
  python -m py_compile \
    "$BOEING_DIAG_CORE" \
    "$BOEING_DIAG_RUNNER" \
    "$BOEING_DIAG_AGGREGATOR" \
    "$BOEING_DIAG_VERIFY_RUNNER" \
    "$BOEING_DIAG_VERIFY_AGGREGATOR" \
    "$BOEING_DIAG_VERIFY_RESOURCE_AUTHENTICATOR" \
    tests/test_other_class_conditional_template_score_boeing_diagnostic_runner.py \
    tests/test_other_class_conditional_template_score_boeing_diagnostic_aggregate.py \
    tests/test_other_class_conditional_template_score_boeing_diagnostic_ibex.py
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
        "tests.test_other_class_conditional_template_score_boeing_diagnostic_runner",
        "tests.test_other_class_conditional_template_score_boeing_diagnostic_aggregate",
        "tests.test_other_class_conditional_template_score_boeing_diagnostic_ibex",
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
print(f"boeing_diagnostic_targeted_tests={len(tests)}_passed")
PY
  python tests/test_all.py
}

ptm_boeing_diag_require_input_evidence() {
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_INPUT_MANIFEST" "$BOEING_DIAG_INPUT_MANIFEST_SHA256" \
    "kinematic input manifest"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_SYNTHETIC_PASS" "$BOEING_DIAG_SYNTHETIC_PASS_SHA256" \
    "parent synthetic PASS"
  ptm_boeing_diag_require_file_sha256 \
    "$BOEING_DIAG_POPULATION_MANIFEST" "$BOEING_DIAG_POPULATION_MANIFEST_SHA256" \
    "sealed sidecar population"
}

ptm_boeing_diag_require_parent_releases() {
  local first_fold_job_id=$1
  local auth_dir=$2
  local auth_complete_sha256=$3
  local resource_pass=$4
  local resource_pass_sha256=$5
  local original_directory original_pythonpath project_directory producer_directory

  [[ "$first_fold_job_id" == "$BOEING_DIAG_VERIFY_FIRST_FOLD_JOB_ID" ]] || \
    ptm_boeing_diag_die "VERIFY_FIRST_FOLD_JOB_ID drifted from the frozen stopped fold"
  [[ "$auth_dir" == "$BOEING_DIAG_VERIFY_AUTH_DIR" ]] || \
    ptm_boeing_diag_die "VERIFY_FIRST_FOLD_AUTH_DIR drifted from the frozen stopped release"
  [[ "$auth_complete_sha256" == "$BOEING_DIAG_VERIFY_AUTH_COMPLETE_SHA256" ]] || \
    ptm_boeing_diag_die "VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256 drifted"
  [[ "$resource_pass" == "$BOEING_DIAG_VERIFY_RESOURCE_PASS" ]] || \
    ptm_boeing_diag_die "VERIFY_RESOURCE_SMOKE_PASS drifted from the frozen resource release"
  [[ "$resource_pass_sha256" == "$BOEING_DIAG_VERIFY_RESOURCE_PASS_SHA256" ]] || \
    ptm_boeing_diag_die "VERIFY_RESOURCE_SMOKE_PASS_SHA256 drifted"

  ptm_boeing_diag_require_file_sha256 \
    "$resource_pass" "$resource_pass_sha256" "Verify resource-smoke PASS"
  ptm_boeing_diag_require_file_sha256 \
    "$auth_dir/AGGREGATE_COMPLETE.json" "$auth_complete_sha256" \
    "stopped Verify single-fold completion"

  original_directory=$(pwd -P)
  original_pythonpath=${PYTHONPATH-}
  project_directory=$(realpath -- "$BOEING_DIAG_PROJECT_ROOT")
  [[ "$original_directory" == "$project_directory" ]] || \
    ptm_boeing_diag_die "parent release authentication must start in the current Other checkout"
  producer_directory=$(realpath -- "$BOEING_DIAG_VERIFY_PRODUCER_ROOT")
  [[ "$producer_directory" == "$BOEING_DIAG_VERIFY_PRODUCER_ROOT" ]] || \
    ptm_boeing_diag_die "Verify producer root is not the frozen absolute path"
  [[ "$producer_directory" != "$project_directory" ]] || \
    ptm_boeing_diag_die "Other and Verify producer roots unexpectedly coincide"
  ptm_boeing_diag_require_verify_producer_checkout

  (
    cd "$BOEING_DIAG_VERIFY_PRODUCER_ROOT"
    export PYTHONPATH="$BOEING_DIAG_VERIFY_PRODUCER_ROOT/src:$BOEING_DIAG_VERIFY_PRODUCER_ROOT"
    python - \
      "$resource_pass" "$resource_pass_sha256" \
      "$auth_dir" "$auth_complete_sha256" \
      "$BOEING_DIAG_VERIFY_EXECUTION_COMMIT" "$BOEING_DIAG_VERIFY_CONFIG_SHA256" \
      "$BOEING_DIAG_VERIFY_FIRST_FOLD_DIR" \
      "$BOEING_DIAG_VERIFY_PRODUCER_ROOT" "$BOEING_DIAG_VERIFY_AUDIT_CONFIG_PATH" <<'PY'
from pathlib import Path
import inspect
import sys

from scripts import aggregate_verify_class_conditional_template_score_1_1 as verify_aggregate
from scripts import run_verify_class_conditional_template_score_resource_smoke_1_1 as verify_resource

producer_root = Path(sys.argv[8]).resolve()
expected_config_path = Path(sys.argv[9]).resolve()
assert Path.cwd().resolve() == producer_root
assert Path(verify_aggregate.__file__).resolve().is_relative_to(producer_root)
assert Path(verify_resource.__file__).resolve().is_relative_to(producer_root)
assert Path(verify_aggregate.runner.__file__).resolve().is_relative_to(producer_root)
assert verify_aggregate.runner.CORE_PATH.resolve() == (
    producer_root / "src/pathline_template_matching/class_conditional_template_score.py"
)
inherited_git_module = inspect.getmodule(
    verify_aggregate.runner._INHERITED_GIT_IDENTITY
)
assert inherited_git_module is not None
assert Path(inherited_git_module.__file__).resolve().is_relative_to(producer_root)
assert verify_resource.CONFIG_PATH.resolve() == expected_config_path
assert verify_aggregate.runner.CONFIG_PATH.resolve() == expected_config_path

resource = verify_resource.authenticate_resource_smoke_release(
    Path(sys.argv[1]),
    sys.argv[2],
    sys.argv[5],
    sys.argv[6],
)
release = verify_aggregate.authenticate_single_fold_release(
    Path(sys.argv[3]),
    expected_completion_sha256=sys.argv[4],
    expected_fold_commit=sys.argv[5],
    expected_config_sha256=sys.argv[6],
    expected_fold_directory=Path(sys.argv[7]),
)
assert resource["status"] == "authenticated"
assert resource["git_commit"] == sys.argv[5]
assert resource["config_sha256"] == sys.argv[6]
assert release["outer_family"] == "half_cylinder"
assert release["fold_numerical_git_commit"] == sys.argv[5]
assert release["stop_version"] is True
assert release["mathematically_impossible_to_pass"] is True
print(f"verify_resource_release_authenticated={resource['marker']['sha256']}")
print(f"verify_stopped_single_fold_authenticated={release['completion_sha256']}")
PY
  )

  [[ "$(pwd -P)" == "$original_directory" ]] || \
    ptm_boeing_diag_die "current directory was not restored after Verify authentication"
  [[ "${PYTHONPATH-}" == "$original_pythonpath" ]] || \
    ptm_boeing_diag_die "PYTHONPATH was not restored after Verify authentication"
  ptm_boeing_diag_require_verify_producer_checkout
  ptm_boeing_diag_stage_gate
}

ptm_boeing_diag_stage_unchanged() {
  ptm_boeing_diag_stage_gate
}
