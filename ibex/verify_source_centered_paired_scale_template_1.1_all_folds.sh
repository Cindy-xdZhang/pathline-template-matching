#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMSCRemain
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --array=1-4%2
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-source-centered
#SBATCH --output=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%A_%a.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_source_centered_paired_scale_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_paired_scale_template_1.1_all_folds.sh
readonly INPUT_MANIFEST=${SOURCE_CENTERED_INPUT_MANIFEST:?SOURCE_CENTERED_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${SOURCE_CENTERED_INPUT_MANIFEST_SHA256:?SOURCE_CENTERED_INPUT_MANIFEST_SHA256 is required}
readonly POPULATION_MANIFEST=${SOURCE_CENTERED_POPULATION_MANIFEST:?SOURCE_CENTERED_POPULATION_MANIFEST is required}
readonly POPULATION_MANIFEST_SHA256=${SOURCE_CENTERED_POPULATION_MANIFEST_SHA256:?SOURCE_CENTERED_POPULATION_MANIFEST_SHA256 is required}
readonly FIRST_FOLD_JOB_ID=${SOURCE_CENTERED_FIRST_FOLD_JOB_ID:?SOURCE_CENTERED_FIRST_FOLD_JOB_ID is required}
readonly FIRST_AUTH_DIR=${SOURCE_CENTERED_FIRST_AUTH_DIR:?SOURCE_CENTERED_FIRST_AUTH_DIR is required}
readonly FIRST_AUTH_COMPLETE_SHA256=${SOURCE_CENTERED_FIRST_AUTH_COMPLETE_SHA256:?SOURCE_CENTERED_FIRST_AUTH_COMPLETE_SHA256 is required}
readonly TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
readonly -a OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)
[[ "$TASK_ID" =~ ^[1-4]$ ]] || \
  ptm_die "remaining-fold array task must be 1..4"
ptm_require_job_id "$FIRST_FOLD_JOB_ID" "SOURCE_CENTERED_FIRST_FOLD_JOB_ID"
readonly OUTER_FAMILY=${OUTER_FAMILIES[$TASK_ID]}

ptm_stage_gate "$WRAPPER"
ptm_activate_runtime "fold_${TASK_ID}" "${SLURM_CPUS_PER_TASK:-32}"
ptm_targeted_preflight
ptm_require_file_sha256 \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "source-centered input manifest"
ptm_require_file_sha256 \
  "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" "sealed 32-sidecar population"
ptm_require_file_sha256 \
  "$FIRST_AUTH_DIR/AGGREGATE_COMPLETE.json" \
  "$FIRST_AUTH_COMPLETE_SHA256" \
  "first-fold authentication completion"

# Reauthenticate the immutable release chain. No caller-provided boolean is
# trusted as permission to run the four remaining outer families.
python - "$FIRST_AUTH_DIR" "$FIRST_AUTH_COMPLETE_SHA256" "$EXPECTED_GIT_COMMIT" "$SOURCE_CENTERED_CONFIG_SHA256" "$FIRST_FOLD_JOB_ID" "$SOURCE_CENTERED_EXPERIMENT_ROOT" "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "$SOURCE_CENTERED_SIDECAR_ROOT" "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import aggregate_verify_source_centered_paired_scale_template_1_1 as aggregate

root = Path(sys.argv[1]).resolve()
(
    completion_sha,
    commit,
    config_sha,
    first_job_id,
    experiment_root_text,
    input_text,
    input_sha,
    sidecar_root_text,
    population_text,
    population_sha,
) = sys.argv[2:12]
assert {path.name for path in root.iterdir()} == {
    "outer_family_summary.csv",
    "single_fold_authentication_report.json",
    "aggregate_manifest.json",
    "AGGREGATE_COMPLETE.json",
}
completion = aggregate.runner._read_self_hashed_json(
    root / "AGGREGATE_COMPLETE.json", expected_file_sha256=completion_sha
)
assert completion["schema"] == aggregate.AGGREGATE_COMPLETE_SCHEMA
assert completion["status"] == "completed"
assert completion["mode"] == "single_fold_authentication"
assert completion["aggregator_git_commit"] == commit
assert completion["fold_git_commit"] == commit
assert completion["config_sha256"] == config_sha
report = aggregate.runner._read_self_hashed_json(
    root / completion["report_file"],
    expected_file_sha256=completion["report_file_sha256"],
)
manifest = aggregate.runner._read_self_hashed_json(
    root / completion["aggregate_manifest_file"],
    expected_file_sha256=completion["aggregate_manifest_file_sha256"],
)
assert report["schema"] == aggregate.SINGLE_FOLD_SCHEMA
assert report["outer_families"] == ["half_cylinder"]
assert report["stop_version"] is False
assert report["all_template_success_conditions_pass"] is None
assert manifest["schema"] == aggregate.AGGREGATE_MANIFEST_SCHEMA
assert manifest["mode"] == "single_fold_authentication"
assert manifest["report_file_sha256"] == completion["report_file_sha256"]
assert len(manifest["source_folds"]) == 1
expected_fold = (
    Path(experiment_root_text)
    / "runs"
    / f"slurm_{first_job_id}_0_{commit[:12]}_outer_half_cylinder"
).resolve()
assert Path(manifest["source_folds"][0]["run_directory"]).resolve() == expected_fold
assert Path(report["folds"][0]["run_directory"]).resolve() == expected_fold
source_fold = manifest["source_folds"][0]
assert aggregate.sha256_file(expected_fold / "RUN_COMPLETE.json") == (
    source_fold["completion_file_sha256"]
)
assert aggregate.sha256_file(expected_fold / "result_manifest.json") == (
    source_fold["result_manifest_file_sha256"]
)
assert aggregate.sha256_file(root / manifest["outer_family_summary_file"]) == (
    manifest["outer_family_summary_file_sha256"]
)
evidence = completion["source_centered_evidence"]
assert report["source_centered_evidence"] == evidence
assert manifest["source_centered_evidence"] == evidence
assert Path(evidence["source_centered_input_manifest"]["path"]).resolve() == Path(input_text).resolve()
assert evidence["source_centered_input_manifest"]["file_sha256"] == input_sha
sidecars = evidence["source_centered_sidecars"]
assert Path(sidecars["root"]).resolve() == Path(sidecar_root_text).resolve()
assert Path(sidecars["population_manifest_path"]).resolve() == Path(population_text).resolve()
assert sidecars["population_manifest_file_sha256"] == population_sha
assert sidecars["sidecar_count"] == 32
print("first_fold_authenticated_release=continue_remaining_folds")
PY

readonly ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly RUN_DIR="$SOURCE_CENTERED_EXPERIMENT_ROOT/runs/slurm_${ARRAY_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${OUTER_FAMILY}"
[[ ! -e "$RUN_DIR" ]] || \
  ptm_die "immutable fold output already exists: $RUN_DIR"

echo "experiment=Verify_SourceCenteredPairedScaleTemplate_1.1"
echo "stage=remaining_outer_fold_after_authenticated_first_fold_release"
echo "outer_family=$OUTER_FAMILY"
echo "formal_confirmation=false"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "run_dir=$RUN_DIR"
hostname
lscpu

/usr/bin/time -v python "$SOURCE_CENTERED_RUNNER" \
  --config "$SOURCE_CENTERED_CONFIG" \
  --expected-config-sha256 "$SOURCE_CENTERED_CONFIG_SHA256" \
  --outer-family "$OUTER_FAMILY" \
  --output-dir "$RUN_DIR" \
  --device cpu \
  --sidecar-input-manifest "$INPUT_MANIFEST" \
  --sidecar-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --sidecar-root "$SOURCE_CENTERED_SIDECAR_ROOT" \
  --sidecar-population-manifest "$POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$POPULATION_MANIFEST_SHA256"

readonly RUN_COMPLETE="$RUN_DIR/RUN_COMPLETE.json"
readonly RUN_COMPLETE_SHA256=$(sha256sum "$RUN_COMPLETE" | awk '{print $1}')
ptm_require_file_sha256 "$RUN_COMPLETE" "$RUN_COMPLETE_SHA256" "outer-fold completion"
ptm_stage_unchanged "$WRAPPER"
echo "remaining_fold_completion_sha256=$RUN_COMPLETE_SHA256"
echo "remaining_fold_status=completed_with_internal_fresh_prediction_before_reference_access"
