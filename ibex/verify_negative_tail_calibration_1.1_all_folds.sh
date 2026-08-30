#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMNegTailAll
#SBATCH --array=0-4
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-tail-cpu
#SBATCH -o /home/zhanx0o/pathline-template-matching-tail-cpu/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-tail-cpu/slurm_logs/%x.%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-tail-cpu
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_NegativeTailCalibration_1.1
CONFIG=config/Verify_NegativeTailCalibration_1.1.yaml
CONFIG_SHA256=4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b
RUNNER=scripts/run_verify_negative_tail_calibration_1_1.py
RUNNER_SHA256=ab62453215a7ecf508aad50e94e244093d898c2baa148908c215e71ce994b6d5
INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_50998592_260a07ad380d/train_cache_input_manifest.json
INPUT_MANIFEST_SHA256=e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393
OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)

EXPECTED_FOLD_COMMIT=${EXPECTED_FOLD_COMMIT:?EXPECTED_FOLD_COMMIT is required}
if [[ ! "$EXPECTED_FOLD_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_FOLD_COMMIT must be a lowercase 40-character Git commit" >&2
  exit 2
fi
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if [[ ! "$TASK_ID" =~ ^[0-4]$ ]]; then
  echo "array task must be one of 0,1,2,3,4: $TASK_ID" >&2
  exit 3
fi
OUTER_FAMILY=${OUTER_FAMILIES[$TASK_ID]}

cd "$PROJECT_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing experiment" >&2
  exit 4
fi

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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_negative_tail_all_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
if [[ "$COMMIT_ID" != "$EXPECTED_FOLD_COMMIT" ]]; then
  echo "checked-out commit does not match EXPECTED_FOLD_COMMIT: $COMMIT_ID" >&2
  exit 5
fi
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${SLURM_ARRAY_JOB_ID}_${TASK_ID}_cpu_${SHORT_COMMIT}_outer_${OUTER_FAMILY}"

echo "experiment=Verify_NegativeTailCalibration_1.1"
echo "phase=cpu_complete_five_outer_folds"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$TASK_ID"
echo "outer_family=$OUTER_FAMILY"
echo "run_dir=$RUN_DIR"
echo "job_tmp_root=$JOB_TMP_ROOT"
hostname
lscpu

ACTUAL_CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
ACTUAL_RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
ACTUAL_INPUT_SHA=$(sha256sum "$INPUT_MANIFEST" | awk '{print $1}')
if [[ "$ACTUAL_CONFIG_SHA" != "$CONFIG_SHA256" ]]; then
  echo "config SHA-256 mismatch: $ACTUAL_CONFIG_SHA" >&2
  exit 6
fi
if [[ "$ACTUAL_RUNNER_SHA" != "$RUNNER_SHA256" ]]; then
  echo "runner SHA-256 mismatch: $ACTUAL_RUNNER_SHA" >&2
  exit 7
fi
if [[ "$ACTUAL_INPUT_SHA" != "$INPUT_MANIFEST_SHA256" ]]; then
  echo "input manifest SHA-256 mismatch: $ACTUAL_INPUT_SHA" >&2
  exit 8
fi
echo "config_sha256=$ACTUAL_CONFIG_SHA"
echo "runner_sha256=$ACTUAL_RUNNER_SHA"
echo "input_manifest_sha256=$ACTUAL_INPUT_SHA"

echo "profile_phase=preflight_203_tests"
python -c 'assert __debug__, "Python assertions must remain enabled"'
/usr/bin/time -v python tests/test_all.py
echo "profile_phase=cpu_backend_gate"
/usr/bin/time -v python scripts/validate_matcher_backend.py --device cpu
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 9
fi

echo "profile_phase=numerical_runner"
/usr/bin/time -v python "$RUNNER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --outer-family "$OUTER_FAMILY" \
  --output-dir "$RUN_DIR" \
  --device cpu

# Reauthenticate the complete artifact envelope and replay only label-free outer
# projections. This postvalidation block does not call the aggregator, reopen
# labels, print metric values, or use metrics for candidate selection.
echo "profile_phase=label_free_artifact_postvalidation"
/usr/bin/time -v python - "$RUN_DIR" "$CONFIG" "$OUTER_FAMILY" "$CONFIG_SHA256" \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "$COMMIT_ID" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import sha256_file
from scripts.run_verify_negative_tail_calibration_1_1 import (
    CALIBRATION_MANIFEST_SCHEMA,
    COMPLETE_SCHEMA,
    EXPERIMENT,
    PREDICTION_MANIFEST_SCHEMA,
    PREDICTION_SCHEMA,
    RESULT_SCHEMA,
    SELECTED_SCHEMA,
    TailCandidateSpec,
    _authenticate_self_hash,
    authenticate_and_rebuild_final_calibration,
    authenticate_outer_prediction,
    authenticate_selected_candidate,
    load_cache_projection,
    load_cache_rows,
    load_plan,
)

root = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
outer_family = sys.argv[3]
config_sha = sys.argv[4]
input_manifest_path = Path(sys.argv[5]).resolve()
input_manifest_sha = sys.argv[6]
git_commit = sys.argv[7]

plan = load_plan(config_path)
assert plan.sha256 == config_sha
assert plan.manifest_path == input_manifest_path
assert plan.manifest_sha256 == input_manifest_sha
assert input_manifest_path.stat().st_size == plan.manifest_size
assert sha256_file(input_manifest_path) == input_manifest_sha

completion_path = root / "RUN_COMPLETE.json"
completion = json.loads(completion_path.read_text(encoding="utf-8"))
_authenticate_self_hash(completion)
assert completion["schema"] == COMPLETE_SCHEMA
assert completion["experiment"] == EXPERIMENT
assert completion["outer_family"] == outer_family
assert completion["config_sha256"] == config_sha
assert completion["git_commit"] == git_commit

result_path = root / completion["result_manifest_file"]
assert result_path.name == "result_manifest.json"
assert sha256_file(result_path) == completion["result_manifest_file_sha256"]
result = json.loads(result_path.read_text(encoding="utf-8"))
_authenticate_self_hash(result)
assert result["content_sha256"] == completion["result_manifest_content_sha256"]
assert result["schema"] == RESULT_SCHEMA
assert result["experiment"] == EXPERIMENT
assert result["status"] == "completed"
assert result["outer_family"] == outer_family
assert result["config_sha256"] == config_sha
assert result["git_commit"] == git_commit
assert result["environment"]["requested_device"] == "cpu"
assert result["environment"]["slurm_job_id"] is not None
assert result["input_manifest"]["path"] == str(input_manifest_path)
assert result["input_manifest"]["size_bytes"] == plan.manifest_size
assert result["input_manifest"]["sha256"] == input_manifest_sha
assert result["input_manifest"]["schema"] == plan.manifest_schema
assert result["input_manifest"]["rows_content_sha256"] == plan.manifest_rows_sha256

expected_artifacts = set(plan.required_fold_files) - {
    "result_manifest.json",
    "RUN_COMPLETE.json",
}
assert set(result["artifacts"]) == expected_artifacts
assert {path.name for path in root.iterdir()} == set(plan.required_fold_files)
assert len(plan.required_fold_files) == 13
for name, identity in result["artifacts"].items():
    path = root / name
    assert path.is_file()
    assert path.stat().st_size == identity["size_bytes"]
    assert sha256_file(path) == identity["sha256"]

selected_payload = result["selected_candidate"]
selected = TailCandidateSpec(
    representation=str(selected_payload["representation"]),
    k=int(selected_payload["k"]),
    sigma=float(selected_payload["sigma"]),
    decision_rule=str(selected_payload["decision_rule"]),
    decision_value=float(selected_payload["decision_value"]),
)
assert selected.candidate_id == selected_payload["candidate_id"]
fit_families = tuple(family for family in plan.family_order if family != outer_family)

calibration_manifest_path = root / "final_tail_calibration_manifest.json"
calibration_manifest = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
_authenticate_self_hash(calibration_manifest)
assert calibration_manifest["schema"] == CALIBRATION_MANIFEST_SCHEMA
calibration = authenticate_and_rebuild_final_calibration(
    root / "final_tail_calibration.npz",
    calibration_manifest_path,
    plan=plan,
    selected=selected,
    outer_family=outer_family,
    fit_families=fit_families,
    git_commit=git_commit,
    expected_manifest_file_sha256=result["final_calibration_manifest_file_sha256"],
)
assert calibration.calibration_file_sha256 == result["final_calibration_file_sha256"]

selected_path = root / "selected_candidate.json"
selected_manifest = json.loads(selected_path.read_text(encoding="utf-8"))
_authenticate_self_hash(selected_manifest)
assert selected_manifest["schema"] == SELECTED_SCHEMA
selected_artifact = authenticate_selected_candidate(
    selected_path,
    plan=plan,
    selected=selected,
    calibration=calibration,
    inner_group_metrics_path=root / "inner_group_metrics.csv",
    inner_group_metrics_sha256=result["inner_group_metrics_file_sha256"],
    inner_candidate_summary_path=root / "inner_candidate_summary.csv",
    inner_candidate_summary_sha256=result["inner_candidate_summary_file_sha256"],
    inner_fit_audits_path=root / "inner_fit_audits.json",
    inner_fit_audits_sha256=result["inner_fit_audits_file_sha256"],
    outer_family=outer_family,
    git_commit=git_commit,
    expected_file_sha256=result["selected_candidate_file_sha256"],
)
assert selected_artifact.manifest["content_sha256"] == result["selected_candidate_content_sha256"]

cache_rows, _ = load_cache_rows(plan)
outer_rows = [row for row in cache_rows if row.family == outer_family]
outer_projections = [
    load_cache_projection(plan, row, include_labels=False)
    for row in outer_rows
]
prediction_manifest_path = root / "outer_prediction_manifest.json"
prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
_authenticate_self_hash(prediction_manifest)
assert prediction_manifest["schema"] == PREDICTION_MANIFEST_SCHEMA
assert prediction_manifest["prediction_schema"] == PREDICTION_SCHEMA
prediction = authenticate_outer_prediction(
    root / "outer_predictions.npz",
    prediction_manifest_path,
    plan=plan,
    selected=selected,
    selected_artifact=selected_artifact,
    calibration=calibration,
    outer_projections=outer_projections,
    expected_outer_rows=outer_rows,
    outer_family=outer_family,
    git_commit=git_commit,
    device="cpu",
    expected_manifest_file_sha256=result["prediction_manifest_file_sha256"],
)
assert prediction.prediction_file_sha256 == result["prediction_file_sha256"]

inner_fit_audits = json.loads((root / "inner_fit_audits.json").read_text(encoding="utf-8"))
_authenticate_self_hash(inner_fit_audits)
assert inner_fit_audits["schema"] == "pathline_template_matching.negative_tail_inner_fit_audits.v1"
assert inner_fit_audits["experiment"] == EXPERIMENT
assert inner_fit_audits["outer_family"] == outer_family

reference_audit = json.loads((root / "outer_reference_access_audit.json").read_text(encoding="utf-8"))
_authenticate_self_hash(reference_audit)
assert reference_audit["schema"] == "pathline_template_matching.negative_tail_outer_reference_access.v1"
assert reference_audit["experiment"] == EXPERIMENT
assert reference_audit["outer_family"] == outer_family
assert reference_audit["first_open_phase"] == "after_outer_prediction_file_and_manifest_authentication"
assert reference_audit["prediction_manifest_file_sha256"] == prediction.manifest_file_sha256
assert reference_audit["prediction_file_sha256"] == prediction.prediction_file_sha256

assert result["selected_candidate_file_sha256"] == result["artifacts"]["selected_candidate.json"]["sha256"]
assert result["final_calibration_manifest_file_sha256"] == result["artifacts"]["final_tail_calibration_manifest.json"]["sha256"]
assert result["final_calibration_file_sha256"] == result["artifacts"]["final_tail_calibration.npz"]["sha256"]
assert result["prediction_manifest_file_sha256"] == result["artifacts"]["outer_prediction_manifest.json"]["sha256"]
assert result["prediction_file_sha256"] == result["artifacts"]["outer_predictions.npz"]["sha256"]
assert result["inner_group_metrics_file_sha256"] == result["artifacts"]["inner_group_metrics.csv"]["sha256"]
assert result["inner_candidate_summary_file_sha256"] == result["artifacts"]["inner_candidate_summary.csv"]["sha256"]
assert result["inner_fit_audits_file_sha256"] == result["artifacts"]["inner_fit_audits.json"]["sha256"]
assert result["outer_group_metrics_file_sha256"] == result["artifacts"]["outer_group_metrics.csv"]["sha256"]
assert result["outer_summary_file_sha256"] == result["artifacts"]["outer_summary.json"]["sha256"]
assert result["outer_reference_access_audit_file_sha256"] == result["artifacts"]["outer_reference_access_audit.json"]["sha256"]
print("postvalidation=passed")
print(f"experiment={EXPERIMENT}")
print(f"outer_family={outer_family}")
print(f"git_commit={git_commit}")
print(f"result_manifest_file_sha256={completion['result_manifest_file_sha256']}")
PY

FINAL_RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
if [[ "$FINAL_RUNNER_SHA" != "$RUNNER_SHA256" ]]; then
  echo "runner changed during experiment: $FINAL_RUNNER_SHA" >&2
  exit 10
fi
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during experiment" >&2
  exit 11
fi
