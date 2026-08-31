#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMRawPCAAll
#SBATCH --array=1-4%1
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-raw-pca-cpu
#SBATCH -o /home/zhanx0o/pathline-template-matching-raw-pca-cpu/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-raw-pca-cpu/slurm_logs/%x.%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-raw-pca-cpu
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_RawPCANegativeMetric_1.1
CONFIG=config/Verify_RawPCANegativeMetric_1.1.yaml
CONFIG_SHA256=6f4718ce6d6385bd0bd5b41a7a04e74cb8f2064fee64097f162999e9eefe6440
RUNNER=scripts/run_verify_raw_pca_negative_metric_1_1.py
RUNNER_SHA256=12785cae503d4a64fff838ad6a377c91ac7191adcfe856147a7009fb5e307dee
AGGREGATOR=scripts/aggregate_verify_raw_pca_negative_metric_1_1.py
AGGREGATOR_SHA256=cf6fc43100db62d45f5f83f4d9ecf449c7ed96cad736462f250898659250b2aa
INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_50998592_260a07ad380d/train_cache_input_manifest.json
INPUT_MANIFEST_SHA256=e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393
OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)

EXPECTED_FOLD_COMMIT=${EXPECTED_FOLD_COMMIT:?EXPECTED_FOLD_COMMIT is required}
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if [[ ! "$EXPECTED_FOLD_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_FOLD_COMMIT must be a lowercase 40-character Git commit" >&2
  exit 2
fi
if [[ ! "$TASK_ID" =~ ^[0-4]$ ]]; then
  echo "array task must be one of 0,1,2,3,4: $TASK_ID" >&2
  exit 3
fi
OUTER_FAMILY=${OUTER_FAMILIES[$TASK_ID]}
if [[ "$TASK_ID" != 0 ]]; then
  FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required for remaining folds}
  FIRST_AUTH_DIR=${FIRST_FOLD_AUTH_DIR:?FIRST_FOLD_AUTH_DIR is required for remaining folds}
  FIRST_AUTH_COMPLETE_SHA256=${FIRST_FOLD_AUTH_COMPLETE_SHA256:?FIRST_FOLD_AUTH_COMPLETE_SHA256 is required for remaining folds}
  [[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "FIRST_FOLD_JOB_ID must be numeric: $FIRST_FOLD_JOB_ID" >&2
    exit 4
  }
fi

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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_raw_pca_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
if [[ "$COMMIT_ID" != "$EXPECTED_FOLD_COMMIT" ]]; then
  echo "checked-out commit does not match EXPECTED_FOLD_COMMIT: $COMMIT_ID" >&2
  exit 5
fi
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${SLURM_ARRAY_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${OUTER_FAMILY}"
if [[ -e "$RUN_DIR" ]]; then
  echo "immutable fold output already exists: $RUN_DIR" >&2
  exit 6
fi

ACTUAL_CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
ACTUAL_RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
ACTUAL_AGGREGATOR_SHA=$(sha256sum "$AGGREGATOR" | awk '{print $1}')
COMMITTED_RUNNER_SHA=$(git show "${EXPECTED_FOLD_COMMIT}:${RUNNER}" | sha256sum | awk '{print $1}')
COMMITTED_AGGREGATOR_SHA=$(git show "${EXPECTED_FOLD_COMMIT}:${AGGREGATOR}" | sha256sum | awk '{print $1}')
ACTUAL_INPUT_SHA=$(sha256sum "$INPUT_MANIFEST" | awk '{print $1}')
[[ "$ACTUAL_CONFIG_SHA" == "$CONFIG_SHA256" ]] || { echo "config SHA-256 mismatch: $ACTUAL_CONFIG_SHA" >&2; exit 7; }
[[ "$ACTUAL_RUNNER_SHA" == "$RUNNER_SHA256" && "$COMMITTED_RUNNER_SHA" == "$RUNNER_SHA256" ]] || { echo "runner SHA-256 mismatch: $ACTUAL_RUNNER_SHA/$COMMITTED_RUNNER_SHA" >&2; exit 8; }
[[ "$ACTUAL_AGGREGATOR_SHA" == "$AGGREGATOR_SHA256" && "$COMMITTED_AGGREGATOR_SHA" == "$AGGREGATOR_SHA256" ]] || { echo "aggregator SHA-256 mismatch: $ACTUAL_AGGREGATOR_SHA/$COMMITTED_AGGREGATOR_SHA" >&2; exit 9; }
[[ "$ACTUAL_INPUT_SHA" == "$INPUT_MANIFEST_SHA256" ]] || { echo "input manifest SHA-256 mismatch: $ACTUAL_INPUT_SHA" >&2; exit 9; }

if [[ "$TASK_ID" != 0 ]]; then
  [[ -f "$FIRST_AUTH_DIR/AGGREGATE_COMPLETE.json" ]] || {
    echo "first-fold authentication completion is missing: $FIRST_AUTH_DIR" >&2
    exit 10
  }
  ACTUAL_FIRST_AUTH_COMPLETE_SHA256=$(sha256sum "$FIRST_AUTH_DIR/AGGREGATE_COMPLETE.json" | awk '{print $1}')
  [[ "$ACTUAL_FIRST_AUTH_COMPLETE_SHA256" == "$FIRST_AUTH_COMPLETE_SHA256" ]] || {
    echo "first-fold authentication completion SHA-256 mismatch: $ACTUAL_FIRST_AUTH_COMPLETE_SHA256" >&2
    exit 11
  }
  python - "$FIRST_AUTH_DIR" "$FIRST_AUTH_COMPLETE_SHA256" "$EXPECTED_FOLD_COMMIT" "$CONFIG_SHA256" "$FIRST_FOLD_JOB_ID" "$EXPERIMENT_ROOT" "$CONFIG" <<'PY'
from pathlib import Path
import sys

from scripts import aggregate_verify_raw_pca_negative_metric_1_1 as aggregate

root = Path(sys.argv[1]).resolve()
completion_sha, commit, config_sha, first_job_id, experiment_root, config_path = sys.argv[2:8]
completion, authenticated_completion_sha = aggregate._load_self_hashed_json(
    root / "AGGREGATE_COMPLETE.json",
    expected_file_sha256=completion_sha,
)
assert authenticated_completion_sha == completion_sha
assert completion["schema"] == aggregate.AGGREGATE_COMPLETE_SCHEMA
assert completion["mode"] == "single_fold_authentication"
assert completion["fold_numerical_git_commit"] == commit
assert completion["aggregator_git_commit"] == commit
assert completion["config_sha256"] == config_sha
assert completion["report_file"] == "single_fold_authentication_report.json"
assert completion["aggregate_manifest_file"] == "aggregate_manifest.json"

certificate_record = completion["early_stop_certificate"]
assert isinstance(certificate_record, dict)
assert certificate_record["path"] == "early_stop_certificate.json"
certificate_snapshot = aggregate._read_file_snapshot(root / certificate_record["path"])
assert certificate_snapshot.identity.size == certificate_record["size_bytes"]
certificate, certificate_sha = aggregate._load_self_hashed_json(
    root / certificate_record["path"],
    expected_file_sha256=certificate_record["sha256"],
)
assert certificate_sha == certificate_record["sha256"]
assert certificate["content_sha256"] == certificate_record["content_sha256"]
assert certificate["schema"] == aggregate.EARLY_STOP_CERTIFICATE_SCHEMA
assert certificate["fold_numerical_git_commit"] == commit
assert certificate["config_sha256"] == config_sha
assert certificate["observed_outer_families"] == ["half_cylinder"]
assert certificate["five_fold_success_evaluated"] is False
assert certificate["five_fold_success"] is None

report, report_sha = aggregate._load_self_hashed_json(
    root / completion["report_file"],
    expected_file_sha256=completion["report_file_sha256"],
)
assert report_sha == completion["report_file_sha256"]
assert report["schema"] == aggregate.SINGLE_FOLD_REPORT_SCHEMA
assert report["mode"] == "single_fold_authentication"
assert report["aggregator_git_commit"] == commit
assert report["fold_numerical_git_commit"] == commit
assert report["config_sha256"] == config_sha
assert report["outer_family"] == "half_cylinder"
assert report["early_stop_certificate"] == certificate_record
assert report["stop_version"] == certificate["stop_version"]
assert report["five_fold_success_evaluated"] is False
assert report["five_fold_success"] is None
plan = aggregate.runner.load_plan(Path(config_path))
assert plan.sha256 == config_sha
expected_certificate = aggregate.runner._manifest_with_self_hash(
    aggregate._early_stop_certificate(
        plan,
        [report["fold"]],
        numerical_git_commit=commit,
    )
)
assert certificate == expected_certificate

manifest, manifest_sha = aggregate._load_self_hashed_json(
    root / completion["aggregate_manifest_file"],
    expected_file_sha256=completion["aggregate_manifest_file_sha256"],
)
assert manifest_sha == completion["aggregate_manifest_file_sha256"]
assert manifest["schema"] == aggregate.AGGREGATE_MANIFEST_SCHEMA
assert manifest["mode"] == "single_fold_authentication"
assert manifest["aggregator_git_commit"] == commit
assert manifest["fold_numerical_git_commit"] == commit
assert manifest["config_sha256"] == config_sha
assert manifest["report_file"] == completion["report_file"]
assert manifest["report_file_sha256"] == report_sha
assert manifest["early_stop_certificate"] == certificate_record
assert len(manifest["source_folds"]) == 1
expected_fold = (
    Path(experiment_root)
    / "runs"
    / f"slurm_{first_job_id}_0_{commit[:12]}_outer_half_cylinder"
).resolve()
assert Path(manifest["source_folds"][0]["run_directory"]).resolve() == expected_fold
table_snapshot = aggregate._read_file_snapshot(root / "outer_family_summary.csv")
assert table_snapshot.sha256 == manifest["outer_family_summary_file_sha256"]
assert table_snapshot.sha256 == report["outer_family_summary_file_sha256"]
assert {path.name for path in root.iterdir()} == {
    "outer_family_summary.csv",
    "early_stop_certificate.json",
    "single_fold_authentication_report.json",
    "aggregate_manifest.json",
    "AGGREGATE_COMPLETE.json",
}
assert certificate["stop_version"] is False, (
    "frozen early-stop certificate forbids remaining folds"
)
print("first_fold_authenticated_release=continue_remaining_folds")
PY
fi

echo "experiment=Verify_RawPCANegativeMetric_1.1"
echo "phase=cpu_outer_fold_after_frozen_release_gate"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "outer_family=$OUTER_FAMILY"
echo "run_dir=$RUN_DIR"
echo "job_tmp_root=$JOB_TMP_ROOT"
hostname
lscpu

python -c 'assert __debug__, "Python assertions must remain enabled"'
/usr/bin/time -v python tests/test_all.py
/usr/bin/time -v python scripts/validate_matcher_backend.py --device cpu
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 10
fi

/usr/bin/time -v python "$RUNNER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --outer-family "$OUTER_FAMILY" \
  --output-dir "$RUN_DIR" \
  --device cpu

# Authenticate the label-free subset from stable private snapshots, then
# fresh-load outer Raw672 and replay PCA/scaler/calibrator predictions.  This
# block never opens result_manifest, outer metrics, metadata_json or labels.
/usr/bin/time -v python - "$RUN_DIR" "$CONFIG" "$OUTER_FAMILY" "$CONFIG_SHA256" "$COMMIT_ID" <<'PY'
from pathlib import Path
import sys
import tempfile

from scripts.aggregate_verify_raw_pca_negative_metric_1_1 import (
    LABEL_FREE_PRE_RESULT_FILES,
    _candidate_from_payload,
    _json_from_snapshot,
    _read_file_snapshot,
    _require_same_snapshot,
    _stage_snapshot,
)
from scripts.run_verify_raw_pca_negative_metric_1_1 import (
    COMPLETE_SCHEMA,
    authenticate_and_rebuild_final_calibration,
    authenticate_and_rebuild_final_pca,
    authenticate_and_rebuild_final_scaler,
    authenticate_outer_prediction,
    authenticate_selected_candidate,
    load_cache_rows,
    load_plan,
)

root = Path(sys.argv[1]).resolve()
plan = load_plan(Path(sys.argv[2]).resolve())
outer_family, config_sha, git_commit = sys.argv[3:6]
assert plan.sha256 == config_sha
assert len(plan.required_fold_files) == 17
assert len(set(plan.required_fold_files) - {"result_manifest.json", "RUN_COMPLETE.json"}) == 15
assert {path.name for path in root.iterdir()} == set(plan.required_fold_files)

completion_snapshot = _read_file_snapshot(root / "RUN_COMPLETE.json")
completion = _json_from_snapshot(
    completion_snapshot, path=root / "RUN_COMPLETE.json", self_hashed=True
)
assert completion["schema"] == COMPLETE_SCHEMA
assert completion["outer_family"] == outer_family
assert completion["git_commit"] == git_commit
snapshots = {
    name: _read_file_snapshot(root / name) for name in LABEL_FREE_PRE_RESULT_FILES
}
with tempfile.TemporaryDirectory(prefix="ptm_raw_pca_wrapper_auth_") as temporary:
    staged = Path(temporary)
    for name in LABEL_FREE_PRE_RESULT_FILES:
        _stage_snapshot(staged / name, snapshots[name])
    selected_payload = _json_from_snapshot(
        snapshots["selected_candidate.json"],
        path=root / "selected_candidate.json",
        self_hashed=True,
    )
    selected = _candidate_from_payload(plan, selected_payload["candidate"])
    assert selected_payload["config_sha256"] == config_sha
    assert selected_payload["git_commit"] == git_commit
    assert selected_payload["outer_family"] == outer_family
    fit_families = tuple(value for value in plan.family_order if value != outer_family)
    pca = authenticate_and_rebuild_final_pca(
        staged / "final_pca.npz", staged / "final_pca_manifest.json",
        plan=plan, outer_family=outer_family, fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=snapshots["final_pca_manifest.json"].sha256,
    )
    scaler = authenticate_and_rebuild_final_scaler(
        staged / "final_per_scale_scaler.npz",
        staged / "final_per_scale_scaler_manifest.json",
        plan=plan, selected=selected, pca=pca, outer_family=outer_family,
        fit_families=fit_families, git_commit=git_commit,
        expected_manifest_file_sha256=snapshots["final_per_scale_scaler_manifest.json"].sha256,
    )
    calibration = authenticate_and_rebuild_final_calibration(
        staged / "final_tail_calibration.npz",
        staged / "final_tail_calibration_manifest.json",
        plan=plan, selected=selected, pca=pca, scaler=scaler,
        outer_family=outer_family, fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=snapshots["final_tail_calibration_manifest.json"].sha256,
    )
    selected_artifact = authenticate_selected_candidate(
        staged / "selected_candidate.json", plan=plan, selected=selected,
        pca=pca, scaler=scaler, calibration=calibration,
        inner_group_metrics_path=staged / "inner_group_metrics.csv",
        inner_group_metrics_sha256=snapshots["inner_group_metrics.csv"].sha256,
        inner_candidate_summary_path=staged / "inner_candidate_summary.csv",
        inner_candidate_summary_sha256=snapshots["inner_candidate_summary.csv"].sha256,
        inner_fit_audits_path=staged / "inner_fit_audits.json",
        inner_fit_audits_sha256=snapshots["inner_fit_audits.json"].sha256,
        outer_family=outer_family, git_commit=git_commit,
        expected_file_sha256=snapshots["selected_candidate.json"].sha256,
    )
    rows, _ = load_cache_rows(plan)
    outer_rows = [row for row in rows if row.family == outer_family]
    prediction = authenticate_outer_prediction(
        staged / "outer_predictions.npz",
        staged / "outer_prediction_manifest.json",
        plan=plan, selected=selected, selected_artifact=selected_artifact,
        pca=pca, scaler=scaler, calibration=calibration, outer_rows=outer_rows,
        outer_family=outer_family, git_commit=git_commit, device="cpu",
        expected_manifest_file_sha256=snapshots["outer_prediction_manifest.json"].sha256,
    )
    assert prediction.prediction_file_sha256 == snapshots["outer_predictions.npz"].sha256
for name, snapshot in snapshots.items():
    _require_same_snapshot(root / name, snapshot)
_require_same_snapshot(root / "RUN_COMPLETE.json", completion_snapshot)
print("label_free_raw_pca_postvalidation=passed")
PY

[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA256" ]] || { echo "runner changed during experiment" >&2; exit 11; }
[[ "$(sha256sum "$AGGREGATOR" | awk '{print $1}')" == "$AGGREGATOR_SHA256" ]] || { echo "aggregator changed during experiment" >&2; exit 12; }
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during experiment" >&2
  exit 12
fi
