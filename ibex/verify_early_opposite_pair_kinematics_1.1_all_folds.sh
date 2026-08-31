#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMEarlyRemain
#SBATCH --array=1-4%2
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-early-kinematics
#SBATCH -o /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_early_opposite_pair_kinematics_1.1_common.sh
readonly WRAPPER=ibex/verify_early_opposite_pair_kinematics_1.1_all_folds.sh
readonly INPUT_MANIFEST=${KINEMATIC_INPUT_MANIFEST:?KINEMATIC_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${KINEMATIC_INPUT_MANIFEST_SHA256:?KINEMATIC_INPUT_MANIFEST_SHA256 is required}
readonly SYNTHETIC_PASS=${SYNTHETIC_PASS:?SYNTHETIC_PASS is required}
readonly SYNTHETIC_PASS_SHA256=${SYNTHETIC_PASS_SHA256:?SYNTHETIC_PASS_SHA256 is required}
readonly POPULATION_MANIFEST=${SIDECAR_POPULATION_MANIFEST:?SIDECAR_POPULATION_MANIFEST is required}
readonly POPULATION_MANIFEST_SHA256=${SIDECAR_POPULATION_MANIFEST_SHA256:?SIDECAR_POPULATION_MANIFEST_SHA256 is required}
readonly FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required}
readonly FIRST_AUTH_DIR=${FIRST_FOLD_AUTH_DIR:?FIRST_FOLD_AUTH_DIR is required}
readonly FIRST_AUTH_COMPLETE_SHA256=${FIRST_FOLD_AUTH_COMPLETE_SHA256:?FIRST_FOLD_AUTH_COMPLETE_SHA256 is required}
readonly TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
readonly -a OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)
[[ "$TASK_ID" =~ ^[1-4]$ ]] || ptm_die "remaining-fold array task must be 1..4"
[[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || ptm_die "FIRST_FOLD_JOB_ID must be numeric"
readonly OUTER_FAMILY=${OUTER_FAMILIES[$TASK_ID]}
[[ "$POPULATION_MANIFEST" == "$EARLY_SIDECAR_ROOT/SIDECAR_POPULATION.json" ]] || \
  ptm_die "sidecar population manifest path differs from the frozen root"

ptm_stage_gate "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_early_opposite_pair_kinematics_runner.py
ptm_activate_runtime "fold_${TASK_ID}" "${SLURM_CPUS_PER_TASK:-32}"
ptm_targeted_preflight
ptm_require_file_sha256 "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "kinematic input manifest"
ptm_require_file_sha256 "$SYNTHETIC_PASS" "$SYNTHETIC_PASS_SHA256" "production synthetic PASS"
ptm_require_file_sha256 "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" "sealed 32-sidecar population"
ptm_require_file_sha256 "$FIRST_AUTH_DIR/AGGREGATE_COMPLETE.json" "$FIRST_AUTH_COMPLETE_SHA256" "first-fold authentication completion"

# Authenticate the complete first-fold release chain. No caller-provided stop
# boolean is trusted: reconstruct the certificate from a report row that is
# bound to the exact immutable first-fold job directory and its artifacts.
python - "$FIRST_AUTH_DIR" "$FIRST_AUTH_COMPLETE_SHA256" "$EXPECTED_GIT_COMMIT" "$EARLY_CONFIG_SHA256" "$FIRST_FOLD_JOB_ID" "$EARLY_EXPERIMENT_ROOT" "$EARLY_CONFIG" "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "$SYNTHETIC_PASS" "$SYNTHETIC_PASS_SHA256" "$EARLY_SIDECAR_ROOT" "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" <<'PY'
import csv
import io
from pathlib import Path
import sys

from scripts import aggregate_verify_early_opposite_pair_kinematics_1_1 as aggregate

root = Path(sys.argv[1]).resolve()
(
    completion_sha,
    commit,
    config_sha,
    first_job_id,
    experiment_root_text,
    config_text,
    input_text,
    input_sha,
    synthetic_text,
    synthetic_sha,
    sidecar_root_text,
    population_text,
    population_sha,
) = sys.argv[2:15]
experiment_root = Path(experiment_root_text).resolve()
config_path = Path(config_text).resolve()
input_path = Path(input_text).resolve()
synthetic_path = Path(synthetic_text).resolve()
sidecar_root = Path(sidecar_root_text).resolve()
population_path = Path(population_text).resolve()

expected_names = {
    "outer_family_summary.csv",
    "early_stop_certificate.json",
    "single_fold_authentication_report.json",
    "aggregate_manifest.json",
    "AGGREGATE_COMPLETE.json",
}
assert {path.name for path in root.iterdir()} == expected_names
snapshots = {
    name: aggregate._read_file_snapshot(root / name) for name in expected_names
}
assert all(snapshot.identity.size > 0 for snapshot in snapshots.values())

def self_hashed(name):
    return aggregate._json_from_snapshot(
        snapshots[name], path=root / name, self_hashed=True
    )

completion = self_hashed("AGGREGATE_COMPLETE.json")
assert snapshots["AGGREGATE_COMPLETE.json"].sha256 == completion_sha
assert set(completion) == {
    "schema", "experiment", "status", "mode", "config_sha256",
    "early_evidence", "aggregator_git_commit", "aggregator_worktree_clean",
    "fold_numerical_git_commit", "aggregate_manifest_file",
    "aggregate_manifest_file_sha256", "report_file", "report_file_sha256",
    "early_stop_certificate", "completed_utc", "content_sha256",
}
assert completion["schema"] == aggregate.AGGREGATE_COMPLETE_SCHEMA
assert completion["experiment"] == aggregate.runner.EXPERIMENT
assert completion["status"] == "completed"
assert completion["mode"] == "single_fold_authentication"
assert completion["fold_numerical_git_commit"] == commit
assert completion["aggregator_git_commit"] == commit
assert completion["aggregator_worktree_clean"] is True
assert completion["config_sha256"] == config_sha
assert completion["report_file"] == "single_fold_authentication_report.json"
assert completion["aggregate_manifest_file"] == "aggregate_manifest.json"

report = self_hashed(completion["report_file"])
manifest = self_hashed(completion["aggregate_manifest_file"])
record = completion["early_stop_certificate"]
assert isinstance(record, dict) and set(record) == {
    "path", "size_bytes", "sha256", "content_sha256"
}
assert record["path"] == "early_stop_certificate.json"
certificate = self_hashed(record["path"])
assert snapshots[completion["report_file"]].sha256 == completion["report_file_sha256"]
assert snapshots[completion["aggregate_manifest_file"]].sha256 == completion["aggregate_manifest_file_sha256"]
assert snapshots[record["path"]].identity.size == record["size_bytes"]
assert snapshots[record["path"]].sha256 == record["sha256"]
assert certificate["content_sha256"] == record["content_sha256"]
assert certificate["schema"] == aggregate.EARLY_STOP_CERTIFICATE_SCHEMA
assert certificate["fold_numerical_git_commit"] == commit
assert certificate["config_sha256"] == config_sha
assert certificate["observed_outer_families"] == ["half_cylinder"]
assert certificate["five_fold_success_evaluated"] is False
assert certificate["five_fold_success"] is None

assert set(report) == {
    "schema", "experiment", "status", "mode", "config_sha256",
    "early_evidence", "aggregator_git_commit", "aggregator_worktree_clean",
    "fold_numerical_git_commit", "input_manifest_sha256",
    "input_manifest_rows_sha256", "outer_family", "fold_summary_source",
    "fold", "early_stop_certificate", "stop_version",
    "five_fold_success_evaluated", "five_fold_success",
    "outer_family_summary_file_sha256", "formal_confirmation",
    "evidence_scope", "content_sha256",
}
assert report["schema"] == aggregate.SINGLE_FOLD_REPORT_SCHEMA
assert report["experiment"] == aggregate.runner.EXPERIMENT
assert report["status"] == "completed"
assert report["mode"] == "single_fold_authentication"
assert report["aggregator_git_commit"] == commit
assert report["aggregator_worktree_clean"] is True
assert report["fold_numerical_git_commit"] == commit
assert report["config_sha256"] == config_sha
assert report["outer_family"] == "half_cylinder"
assert report["early_stop_certificate"] == record
assert report["stop_version"] == certificate["stop_version"]
assert report["five_fold_success_evaluated"] is False
assert report["five_fold_success"] is None
assert report["formal_confirmation"] is False

assert set(manifest) == {
    "schema", "experiment", "status", "mode", "config_sha256",
    "early_evidence", "aggregator_git_commit", "aggregator_worktree_clean",
    "fold_numerical_git_commit", "outer_family_summary_file",
    "outer_family_summary_file_sha256", "report_file", "report_file_sha256",
    "early_stop_certificate", "source_folds", "content_sha256",
}
assert manifest["schema"] == aggregate.AGGREGATE_MANIFEST_SCHEMA
assert manifest["experiment"] == aggregate.runner.EXPERIMENT
assert manifest["status"] == "completed"
assert manifest["mode"] == "single_fold_authentication"
assert manifest["aggregator_git_commit"] == commit
assert manifest["aggregator_worktree_clean"] is True
assert manifest["fold_numerical_git_commit"] == commit
assert manifest["config_sha256"] == config_sha
assert manifest["report_file"] == completion["report_file"]
assert manifest["report_file_sha256"] == snapshots[completion["report_file"]].sha256
assert manifest["early_stop_certificate"] == record
assert manifest["outer_family_summary_file"] == "outer_family_summary.csv"

table_snapshot = snapshots["outer_family_summary.csv"]
assert table_snapshot.sha256 == manifest["outer_family_summary_file_sha256"]
assert table_snapshot.sha256 == report["outer_family_summary_file_sha256"]

# Bind aggregate evidence to the exact input, synthetic gate, and sealed
# 32-sidecar population supplied to this Slurm stage.
input_snapshot = aggregate._read_file_snapshot(input_path)
synthetic_snapshot = aggregate._read_file_snapshot(synthetic_path)
population_snapshot = aggregate._read_file_snapshot(population_path)
assert input_snapshot.sha256 == input_sha
assert synthetic_snapshot.sha256 == synthetic_sha
assert population_snapshot.sha256 == population_sha
input_manifest = aggregate._json_from_snapshot(
    input_snapshot, path=input_path, self_hashed=True
)
synthetic_marker = aggregate._json_from_snapshot(
    synthetic_snapshot, path=synthetic_path, self_hashed=True
)
population_manifest = aggregate._json_from_snapshot(
    population_snapshot, path=population_path, self_hashed=True
)
early = report["early_evidence"]
assert completion["early_evidence"] == early
assert manifest["early_evidence"] == early
assert set(early) == {
    "kinematic_input_manifest", "synthetic_pass",
    "sidecar_population_manifest", "composite_descriptor_ids",
    "clean_git_commit", "config_sha256",
}
assert Path(early["kinematic_input_manifest"]["path"]).resolve() == input_path
assert early["kinematic_input_manifest"]["file_sha256"] == input_sha
assert early["kinematic_input_manifest"]["content_sha256"] == input_manifest["content_sha256"]
assert Path(early["synthetic_pass"]["path"]).resolve() == synthetic_path
assert early["synthetic_pass"]["file_sha256"] == synthetic_sha
assert Path(early["sidecar_population_manifest"]["path"]).resolve() == population_path
assert population_path == sidecar_root / "SIDECAR_POPULATION.json"
assert early["sidecar_population_manifest"]["file_sha256"] == population_sha
assert early["sidecar_population_manifest"]["content_sha256"] == population_manifest["content_sha256"]
assert early["sidecar_population_manifest"]["sidecar_count"] == 32
assert early["clean_git_commit"] == commit
assert early["config_sha256"] == config_sha
aggregate._require_preparation_release_binding(
    early=early,
    input_manifest=input_manifest,
    synthetic_marker=synthetic_marker,
    population_manifest=population_manifest,
    current_fold_commit=commit,
)

plan = aggregate.runner.load_plan(config_path)
assert plan.sha256 == config_sha
assert report["input_manifest_sha256"] == plan.manifest_sha256
assert report["input_manifest_rows_sha256"] == plan.manifest_rows_sha256

# Bind the report row to the exact first-fold job directory and every artifact
# recorded by its authenticated result manifest.
source_folds = manifest["source_folds"]
assert isinstance(source_folds, list) and len(source_folds) == 1
source_fold = source_folds[0]
assert set(source_fold) == {
    "outer_family", "run_directory", "completion_file_sha256",
    "result_manifest_file_sha256", "artifact_count", "artifacts",
}
expected_fold = (
    experiment_root
    / "runs"
    / f"slurm_{first_job_id}_0_{commit[:12]}_outer_half_cylinder"
).resolve()
assert source_fold["outer_family"] == "half_cylinder"
assert Path(source_fold["run_directory"]).resolve() == expected_fold
assert {path.name for path in expected_fold.iterdir()} == set(plan.required_fold_files)

fold_completion_snapshot = aggregate._read_file_snapshot(
    expected_fold / "RUN_COMPLETE.json"
)
fold_completion = aggregate._json_from_snapshot(
    fold_completion_snapshot,
    path=expected_fold / "RUN_COMPLETE.json",
    self_hashed=True,
)
assert set(fold_completion) == aggregate.COMPLETION_FIELDS
assert fold_completion_snapshot.sha256 == source_fold["completion_file_sha256"]
assert fold_completion["schema"] == aggregate.runner.COMPLETE_SCHEMA
assert fold_completion["outer_family"] == "half_cylinder"
assert fold_completion["git_commit"] == commit
assert fold_completion["config_sha256"] == config_sha
assert fold_completion["result_manifest_file"] == "result_manifest.json"

result_snapshot = aggregate._read_file_snapshot(expected_fold / "result_manifest.json")
result = aggregate._json_from_snapshot(
    result_snapshot,
    path=expected_fold / "result_manifest.json",
    self_hashed=True,
)
assert set(result) == aggregate.RESULT_FIELDS
assert result_snapshot.sha256 == source_fold["result_manifest_file_sha256"]
assert result_snapshot.sha256 == fold_completion["result_manifest_file_sha256"]
assert result["content_sha256"] == fold_completion["result_manifest_content_sha256"]
assert result["git_commit"] == commit
assert result["config_sha256"] == config_sha
assert result["outer_family"] == "half_cylinder"
aggregate._require_result_input_manifest_binding(result, plan)
result_early = result["early_evidence"]
assert result_early["kinematic_input_manifest"] == early["kinematic_input_manifest"]
assert result_early["sidecar_population_manifest"] == early["sidecar_population_manifest"]
assert result_early["config_sha256"] == config_sha
assert result_early["clean_git_commit"] == commit
assert result_early["representation"] == result["selected_candidate"]["representation"]
assert result_early["composite_descriptor_id"] == early["composite_descriptor_ids"][result_early["representation"]]
assert result_early["fit_families"] == [
    family for family in plan.family_order if family != "half_cylinder"
]

artifacts = source_fold["artifacts"]
assert source_fold["artifact_count"] == len(aggregate.EXPECTED_RESULT_ARTIFACTS)
assert set(artifacts) == set(aggregate.EXPECTED_RESULT_ARTIFACTS)
assert result["artifacts"] == artifacts
for name, artifact in artifacts.items():
    assert set(artifact) == {"size_bytes", "sha256"}
    snapshot = aggregate._read_file_snapshot(expected_fold / name)
    assert snapshot.identity.size == artifact["size_bytes"]
    assert snapshot.sha256 == artifact["sha256"]

fold = report["fold"]
expected_fold_fields = {
    "outer_family", "run_directory", "numerical_git_commit", "config_sha256",
    "input_manifest_sha256", "input_manifest_rows_sha256", "requested_device",
    "selected_candidate_id", "completion_file_sha256",
    "completion_content_sha256", "result_manifest_file_sha256",
    "result_manifest_content_sha256", "outer_group_metrics_file_sha256",
    *aggregate.FAMILY_METRIC_FIELDS, *aggregate.FAMILY_COUNT_FIELDS,
}
assert set(fold) == expected_fold_fields
assert fold["outer_family"] == "half_cylinder"
assert Path(fold["run_directory"]).resolve() == expected_fold
assert fold["numerical_git_commit"] == commit
assert fold["config_sha256"] == config_sha
assert fold["input_manifest_sha256"] == plan.manifest_sha256
assert fold["input_manifest_rows_sha256"] == plan.manifest_rows_sha256
assert fold["requested_device"] == result["environment"]["requested_device"]
assert fold["selected_candidate_id"] == result["selected_candidate"]["candidate_id"]
assert fold["completion_file_sha256"] == fold_completion_snapshot.sha256
assert fold["completion_content_sha256"] == fold_completion["content_sha256"]
assert fold["result_manifest_file_sha256"] == result_snapshot.sha256
assert fold["result_manifest_content_sha256"] == result["content_sha256"]
assert fold["outer_group_metrics_file_sha256"] == artifacts["outer_group_metrics.csv"]["sha256"]

outer_summary_snapshot = aggregate._read_file_snapshot(
    expected_fold / "outer_summary.json"
)
outer_summary = aggregate._json_from_snapshot(
    outer_summary_snapshot,
    path=expected_fold / "outer_summary.json",
    self_hashed=True,
)
assert outer_summary_snapshot.sha256 == artifacts["outer_summary.json"]["sha256"]
assert outer_summary["schema"] == aggregate.runner.OUTER_SUMMARY_SCHEMA
assert outer_summary["outer_family"] == "half_cylinder"
for field in (*aggregate.FAMILY_METRIC_FIELDS, *aggregate.FAMILY_COUNT_FIELDS):
    assert fold[field] == outer_summary[field]

with io.StringIO(table_snapshot.content.decode("utf-8"), newline="") as stream:
    table = csv.DictReader(stream)
    assert tuple(table.fieldnames or ()) == aggregate.FAMILY_SUMMARY_FIELDS
    rows = list(table)
assert len(rows) == 1
assert all(rows[0][field] == aggregate._csv_text(fold[field]) for field in fold)

expected_certificate = aggregate.runner._manifest_with_self_hash(
    aggregate._early_stop_certificate(
        plan,
        [fold],
        numerical_git_commit=commit,
    )
)
assert certificate == expected_certificate
assert certificate["stop_version"] is False, (
    "frozen early-stop certificate forbids remaining folds"
)
print("first_fold_authenticated_release=continue_remaining_folds")
PY

readonly ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly RUN_DIR="$EARLY_EXPERIMENT_ROOT/runs/slurm_${ARRAY_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${OUTER_FAMILY}"
[[ ! -e "$RUN_DIR" ]] || ptm_die "immutable fold output already exists: $RUN_DIR"

echo "experiment=Verify_EarlyOppositePairKinematics_1.1"
echo "stage=remaining_nested_outer_family_fold_after_authenticated_no_stop"
echo "formal_confirmation=false"
echo "outer_family=$OUTER_FAMILY"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "run_dir=$RUN_DIR"
hostname
lscpu

# The runner authenticates only the population byte envelope at startup.  It
# does not deserialize the held-out sidecar until scaler, calibrator, and
# selected-candidate artifacts have been closed and freshly authenticated.
/usr/bin/time -v python "$EARLY_RUNNER" \
  --config "$EARLY_CONFIG" \
  --expected-config-sha256 "$EARLY_CONFIG_SHA256" \
  --outer-family "$OUTER_FAMILY" \
  --output-dir "$RUN_DIR" \
  --device cpu \
  --kinematic-input-manifest "$INPUT_MANIFEST" \
  --kinematic-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$EARLY_SIDECAR_ROOT" \
  --sidecar-population-manifest "$POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$POPULATION_MANIFEST_SHA256"

ptm_stage_unchanged "$WRAPPER" \
  "$EARLY_PREPARER" \
  "$EARLY_RUNNER" \
  "$EARLY_AGGREGATOR" \
  tests/test_early_opposite_pair_kinematics_runner.py
echo "remaining_fold_status=completed_with_internal_fresh_prediction_before_outer_label_gate"
