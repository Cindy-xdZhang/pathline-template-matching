#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMClassSmoke
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-class-conditional-score
#SBATCH -o /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=rome
#SBATCH --partition=cpu
#SBATCH --account=deepvortex

set -euo pipefail
export PTM_RESOURCE_SMOKE_WRAPPER_START_EPOCH
PTM_RESOURCE_SMOKE_WRAPPER_START_EPOCH=$(date +%s)
readonly RESOURCE_SMOKE_JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
command -v scontrol >/dev/null || {
  echo "scontrol is required to authenticate the Slurm allocation" >&2
  exit 2
}
export PTM_SLURM_SCONTROL_JOB_RECORD
PTM_SLURM_SCONTROL_JOB_RECORD=$(scontrol show job -o "$RESOURCE_SMOKE_JOB_ID")
[[ -n "$PTM_SLURM_SCONTROL_JOB_RECORD" ]] || {
  echo "scontrol returned an empty Slurm allocation" >&2
  exit 2
}

readonly PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-class-conditional-score
readonly EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1
readonly CONFIG=config/Verify_ClassConditionalTemplateScore_1.1.yaml
readonly CONFIG_SHA256=814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea
readonly CORE=src/pathline_template_matching/class_conditional_template_score.py
readonly RUNNER=scripts/run_verify_class_conditional_template_score_1_1.py
readonly SMOKE=scripts/run_verify_class_conditional_template_score_resource_smoke_1_1.py
readonly SMOKE_TEST=tests/test_class_conditional_template_score_resource_smoke.py
readonly WRAPPER=ibex/verify_class_conditional_template_score_1.1_resource_smoke.sh

readonly KINEMATIC_INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/preparation/slurm_51068863_fd0412dc134d/kinematic_input_manifest.json
readonly KINEMATIC_INPUT_MANIFEST_SHA256=1b9df53a9010c6c3c46345639cfbf1d5ab2fe3a43187c79c7dfa0f7d840b102f
readonly SYNTHETIC_PASS=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/preparation/slurm_51068863_fd0412dc134d/SYNTHETIC_PASS.json
readonly SYNTHETIC_PASS_SHA256=78d0990352777e488f26bb84f3b0fc16e18845fc7cedb8a7d7fc598f32c0afe3
readonly SIDECAR_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/kinematic_cache/train
readonly SIDECAR_POPULATION_MANIFEST="$SIDECAR_ROOT/SIDECAR_POPULATION.json"
readonly SIDECAR_POPULATION_MANIFEST_SHA256=9f96835b9185218f40df4cc3c52bf3d80a93056681d922a30abfc5c0246f88a7

die() {
  echo "$*" >&2
  exit 2
}

require_sha256() {
  local value=$1
  local name=$2
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die "$name must be lowercase SHA-256"
}

reject_forbidden_dataset_token() {
  local value=$1
  local name=$2
  local normalized
  normalized=$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')
  if [[ "$normalized" == *tangaroa* || "$normalized" == *smokebuoyancy* ]]; then
    die "$name contains a forbidden dataset token"
  fi
}

require_file_sha256() {
  local path=$1
  local expected=$2
  local name=$3
  reject_forbidden_dataset_token "$path" "$name path"
  require_sha256 "$expected" "$name SHA-256"
  [[ -f "$path" ]] || die "$name is missing: $path"
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || die "$name SHA-256 mismatch: $actual"
}

stage_gate() {
  local expected_commit=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
  [[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || die "EXPECTED_GIT_COMMIT must be lowercase 40-hex"
  cd "$PROJECT_ROOT"
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || die "worktree is not clean"
  local observed_commit
  observed_commit=$(git rev-parse --verify HEAD^{commit})
  [[ "$observed_commit" == "$expected_commit" ]] || die "checkout differs from EXPECTED_GIT_COMMIT"
  git cat-file -e "${expected_commit}^{commit}"
  require_file_sha256 "$CONFIG" "$CONFIG_SHA256" "ClassConditional config"

  local -a sources=(
    "$CONFIG"
    "$CORE"
    "$RUNNER"
    "$SMOKE"
    "$SMOKE_TEST"
    "$WRAPPER"
    tests/test_class_conditional_template_score.py
    tests/test_class_conditional_template_score_runner.py
  )
  local source actual committed
  for source in "${sources[@]}"; do
    git ls-files --error-unmatch "$source" >/dev/null
    actual=$(sha256sum "$source" | awk '{print $1}')
    committed=$(git show "${expected_commit}:${source}" | sha256sum | awk '{print $1}')
    [[ "$actual" == "$committed" ]] || die "source differs from expected commit: $source"
    echo "source_sha256[$source]=$actual"
  done
}

stage_gate
require_file_sha256 "$KINEMATIC_INPUT_MANIFEST" "$KINEMATIC_INPUT_MANIFEST_SHA256" "kinematic input manifest"
require_file_sha256 "$SYNTHETIC_PASS" "$SYNTHETIC_PASS_SHA256" "parent synthetic PASS"
require_file_sha256 "$SIDECAR_POPULATION_MANIFEST" "$SIDECAR_POPULATION_MANIFEST_SHA256" "sealed sidecar population"

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONOPTIMIZE
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export CLASS_JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_class_resource_smoke_${SLURM_JOB_ID:?SLURM_JOB_ID is required}_single"
mkdir -p "$CLASS_JOB_TMP_ROOT"
export TMPDIR="$CLASS_JOB_TMP_ROOT"
export TMP="$CLASS_JOB_TMP_ROOT"
export TEMP="$CLASS_JOB_TMP_ROOT"
export NUMBA_CACHE_DIR="$CLASS_JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"
python -c 'import os, pathlib, tempfile; assert __debug__, "Python assertions must remain enabled"; assert pathlib.Path(tempfile.gettempdir()).resolve() == pathlib.Path(os.environ["CLASS_JOB_TMP_ROOT"]).resolve(), "Python tempfile escaped the job-local root"'

python -m py_compile \
  "$CORE" \
  "$RUNNER" \
  "$SMOKE" \
  tests/test_class_conditional_template_score.py \
  tests/test_class_conditional_template_score_runner.py \
  "$SMOKE_TEST"
python tests/test_class_conditional_template_score_runner.py
python - <<'PY'
import importlib
import inspect

modules = (
    "tests.test_class_conditional_template_score",
    "tests.test_class_conditional_template_score_resource_smoke",
)
count = 0
for module_name in modules:
    module = importlib.import_module(module_name)
    for name in sorted(value for value in dir(module) if value.startswith("test_")):
        function = getattr(module, name)
        if inspect.signature(function).parameters:
            continue
        function()
        count += 1
print(f"class_conditional_resource_smoke_targeted_tests={count}_passed")
PY
python tests/test_all.py

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly EXPECTED_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
readonly SHORT_COMMIT=${EXPECTED_COMMIT:0:12}
readonly RUN_DIR="$EXPERIMENT_ROOT/resource_smoke/slurm_${JOB_ID}_${SHORT_COMMIT}"
[[ ! -e "$RUN_DIR" ]] || die "immutable resource-smoke output already exists: $RUN_DIR"

echo "experiment=Verify_ClassConditionalTemplateScore_1.1"
echo "stage=mandatory_resource_smoke_before_first_real_outer_fold"
echo "fit_families=f22_raptor,channel,boeing_747"
echo "reserved_outer_family=half_cylinder"
echo "reserved_inner_family=delta_wing"
echo "representation=fmt161_plus_seed4"
echo "k=31"
echo "run_dir=$RUN_DIR"
hostname
lscpu

/usr/bin/time -v python "$SMOKE" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --expected-git-commit "$EXPECTED_COMMIT" \
  --output-dir "$RUN_DIR" \
  --kinematic-input-manifest "$KINEMATIC_INPUT_MANIFEST" \
  --kinematic-input-manifest-sha256 "$KINEMATIC_INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$SIDECAR_ROOT" \
  --sidecar-population-manifest "$SIDECAR_POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$SIDECAR_POPULATION_MANIFEST_SHA256"

[[ -f "$RUN_DIR/resource_smoke_audit.json" ]] || die "detailed resource-smoke audit is missing"
[[ -f "$RUN_DIR/RESOURCE_SMOKE_PASS.json" ]] || die "resource-smoke PASS marker is missing"
[[ "$(find "$RUN_DIR" -maxdepth 1 -type f | wc -l)" -eq 2 ]] || die "resource-smoke output file set drifted"
stage_gate
echo "resource_smoke_job_id=$JOB_ID"
echo "resource_smoke_run_dir=$RUN_DIR"
