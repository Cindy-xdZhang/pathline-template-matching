#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMSCAuth1
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-source-centered
#SBATCH --output=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_source_centered_paired_scale_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_paired_scale_template_1.1_first_fold_auth.sh
readonly FIRST_FOLD_JOB_ID=${SOURCE_CENTERED_FIRST_FOLD_JOB_ID:?SOURCE_CENTERED_FIRST_FOLD_JOB_ID is required}
readonly INPUT_MANIFEST=${SOURCE_CENTERED_INPUT_MANIFEST:?SOURCE_CENTERED_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${SOURCE_CENTERED_INPUT_MANIFEST_SHA256:?SOURCE_CENTERED_INPUT_MANIFEST_SHA256 is required}
readonly POPULATION_MANIFEST=${SOURCE_CENTERED_POPULATION_MANIFEST:?SOURCE_CENTERED_POPULATION_MANIFEST is required}
readonly POPULATION_MANIFEST_SHA256=${SOURCE_CENTERED_POPULATION_MANIFEST_SHA256:?SOURCE_CENTERED_POPULATION_MANIFEST_SHA256 is required}
ptm_require_job_id "$FIRST_FOLD_JOB_ID" "SOURCE_CENTERED_FIRST_FOLD_JOB_ID"

ptm_stage_gate "$WRAPPER"
ptm_activate_runtime first_fold_auth "${SLURM_CPUS_PER_TASK:-32}"
ptm_targeted_preflight
ptm_require_file_sha256 \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "source-centered input manifest"
ptm_require_file_sha256 \
  "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" "sealed 32-sidecar population"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly FOLD_RUN_DIR="$SOURCE_CENTERED_EXPERIMENT_ROOT/runs/slurm_${FIRST_FOLD_JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
readonly OUTPUT_DIR="$SOURCE_CENTERED_EXPERIMENT_ROOT/authentication/slurm_${JOB_ID}_${SHORT_COMMIT}_firstfold_${FIRST_FOLD_JOB_ID}"
[[ -d "$FOLD_RUN_DIR" ]] || ptm_die "first fold is missing: $FOLD_RUN_DIR"
[[ ! -e "$OUTPUT_DIR" ]] || \
  ptm_die "immutable first-fold authentication output exists: $OUTPUT_DIR"

echo "experiment=Verify_SourceCenteredPairedScaleTemplate_1.1"
echo "stage=fresh_single_fold_replay_and_authenticated_stop_decision"
echo "first_fold_job_id=$FIRST_FOLD_JOB_ID"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

/usr/bin/time -v python "$SOURCE_CENTERED_AGGREGATOR" \
  --config "$SOURCE_CENTERED_CONFIG" \
  --expected-config-sha256 "$SOURCE_CENTERED_CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_GIT_COMMIT" \
  --run-dir "$FOLD_RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device cpu \
  --sidecar-input-manifest "$INPUT_MANIFEST" \
  --sidecar-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --sidecar-root "$SOURCE_CENTERED_SIDECAR_ROOT" \
  --sidecar-population-manifest "$POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$POPULATION_MANIFEST_SHA256"

readonly AUTH_COMPLETE="$OUTPUT_DIR/AGGREGATE_COMPLETE.json"
readonly AUTH_COMPLETE_SHA256=$(sha256sum "$AUTH_COMPLETE" | awk '{print $1}')
readonly AUTH_REPORT="$OUTPUT_DIR/single_fold_authentication_report.json"
readonly AUTH_REPORT_SHA256=$(sha256sum "$AUTH_REPORT" | awk '{print $1}')
ptm_require_file_sha256 \
  "$AUTH_COMPLETE" "$AUTH_COMPLETE_SHA256" "first-fold authentication completion"
ptm_require_file_sha256 \
  "$AUTH_REPORT" "$AUTH_REPORT_SHA256" "first-fold authentication report"

readonly STOP_VERSION=$(python - "$OUTPUT_DIR" "$AUTH_COMPLETE_SHA256" "$EXPECTED_GIT_COMMIT" "$SOURCE_CENTERED_CONFIG_SHA256" "$FOLD_RUN_DIR" "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "$SOURCE_CENTERED_SIDECAR_ROOT" "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import aggregate_verify_source_centered_paired_scale_template_1_1 as aggregate

root = Path(sys.argv[1]).resolve()
(
    completion_sha,
    commit,
    config_sha,
    expected_fold,
    input_text,
    input_sha,
    sidecar_root_text,
    population_text,
    population_sha,
) = sys.argv[2:11]
expected_names = {
    "outer_family_summary.csv",
    "single_fold_authentication_report.json",
    "aggregate_manifest.json",
    "AGGREGATE_COMPLETE.json",
}
assert {path.name for path in root.iterdir()} == expected_names
completion = aggregate.runner._read_self_hashed_json(
    root / "AGGREGATE_COMPLETE.json", expected_file_sha256=completion_sha
)
assert completion["schema"] == aggregate.AGGREGATE_COMPLETE_SCHEMA
assert completion["experiment"] == aggregate.runner.EXPERIMENT
assert completion["status"] == "completed"
assert completion["mode"] == "single_fold_authentication"
assert completion["aggregator_git_commit"] == commit
assert completion["fold_git_commit"] == commit
assert completion["config_sha256"] == config_sha
assert completion["report_file"] == "single_fold_authentication_report.json"
report = aggregate.runner._read_self_hashed_json(
    root / completion["report_file"],
    expected_file_sha256=completion["report_file_sha256"],
)
manifest = aggregate.runner._read_self_hashed_json(
    root / completion["aggregate_manifest_file"],
    expected_file_sha256=completion["aggregate_manifest_file_sha256"],
)
assert report["schema"] == aggregate.SINGLE_FOLD_SCHEMA
assert report["status"] == "completed"
assert report["mode"] == "single_fold_authentication"
assert report["outer_families"] == ["half_cylinder"]
assert len(report["folds"]) == 1
assert Path(report["folds"][0]["run_directory"]).resolve() == Path(expected_fold).resolve()
assert report["all_template_success_conditions_pass"] is None
assert isinstance(report["stop_version"], bool)
assert manifest["schema"] == aggregate.AGGREGATE_MANIFEST_SCHEMA
assert manifest["mode"] == "single_fold_authentication"
assert manifest["report_file_sha256"] == completion["report_file_sha256"]
assert len(manifest["source_folds"]) == 1
assert Path(manifest["source_folds"][0]["run_directory"]).resolve() == Path(expected_fold).resolve()
assert aggregate.sha256_file(root / manifest["outer_family_summary_file"]) == (
    manifest["outer_family_summary_file_sha256"]
)
assert report["outer_family_summary_file_sha256"] == (
    manifest["outer_family_summary_file_sha256"]
)
evidence = completion["source_centered_evidence"]
assert report["source_centered_evidence"] == evidence
assert manifest["source_centered_evidence"] == evidence
assert evidence["git_commit"] == commit
assert evidence["config_sha256"] == config_sha
assert Path(evidence["source_centered_input_manifest"]["path"]).resolve() == Path(input_text).resolve()
assert evidence["source_centered_input_manifest"]["file_sha256"] == input_sha
sidecars = evidence["source_centered_sidecars"]
assert Path(sidecars["root"]).resolve() == Path(sidecar_root_text).resolve()
assert Path(sidecars["population_manifest_path"]).resolve() == Path(population_text).resolve()
assert sidecars["population_manifest_file_sha256"] == population_sha
assert sidecars["sidecar_count"] == 32
print(str(report["stop_version"]).lower())
PY
)

ptm_stage_unchanged "$WRAPPER"
echo "first_fold_auth_dir=$OUTPUT_DIR"
echo "first_fold_auth_complete_sha256=$AUTH_COMPLETE_SHA256"
echo "first_fold_auth_report_sha256=$AUTH_REPORT_SHA256"
echo "stop_version=$STOP_VERSION"
