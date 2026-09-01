#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMSCAgg5
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-source-centered
#SBATCH --output=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_source_centered_paired_scale_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_paired_scale_template_1.1_aggregate_five.sh
readonly INPUT_MANIFEST=${SOURCE_CENTERED_INPUT_MANIFEST:?SOURCE_CENTERED_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${SOURCE_CENTERED_INPUT_MANIFEST_SHA256:?SOURCE_CENTERED_INPUT_MANIFEST_SHA256 is required}
readonly POPULATION_MANIFEST=${SOURCE_CENTERED_POPULATION_MANIFEST:?SOURCE_CENTERED_POPULATION_MANIFEST is required}
readonly POPULATION_MANIFEST_SHA256=${SOURCE_CENTERED_POPULATION_MANIFEST_SHA256:?SOURCE_CENTERED_POPULATION_MANIFEST_SHA256 is required}
readonly FIRST_FOLD_JOB_ID=${SOURCE_CENTERED_FIRST_FOLD_JOB_ID:?SOURCE_CENTERED_FIRST_FOLD_JOB_ID is required}
readonly REMAINING_FOLD_ARRAY_JOB_ID=${SOURCE_CENTERED_REMAINING_FOLD_ARRAY_JOB_ID:?SOURCE_CENTERED_REMAINING_FOLD_ARRAY_JOB_ID is required}
ptm_require_job_id "$FIRST_FOLD_JOB_ID" "SOURCE_CENTERED_FIRST_FOLD_JOB_ID"
ptm_require_job_id \
  "$REMAINING_FOLD_ARRAY_JOB_ID" "SOURCE_CENTERED_REMAINING_FOLD_ARRAY_JOB_ID"

ptm_stage_gate "$WRAPPER" tests/test_all.py
ptm_activate_runtime aggregate_five "${SLURM_CPUS_PER_TASK:-32}"
ptm_targeted_preflight
ptm_full_preflight
ptm_require_file_sha256 \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "source-centered input manifest"
ptm_require_file_sha256 \
  "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" "sealed 32-sidecar population"

readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly -a OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)
RUN_DIRS=()
for TASK_ID in 0 1 2 3 4; do
  FAMILY=${OUTER_FAMILIES[$TASK_ID]}
  if [[ "$TASK_ID" == 0 ]]; then
    SOURCE_JOB_ID=$FIRST_FOLD_JOB_ID
  else
    SOURCE_JOB_ID=$REMAINING_FOLD_ARRAY_JOB_ID
  fi
  RUN_DIR="$SOURCE_CENTERED_EXPERIMENT_ROOT/runs/slurm_${SOURCE_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${FAMILY}"
  [[ -d "$RUN_DIR" ]] || ptm_die "required fold directory is missing: $RUN_DIR"
  RUN_DIRS+=("$RUN_DIR")
done

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly OUTPUT_DIR="$SOURCE_CENTERED_EXPERIMENT_ROOT/aggregate/slurm_${JOB_ID}_${SHORT_COMMIT}_first_${FIRST_FOLD_JOB_ID}_remaining_${REMAINING_FOLD_ARRAY_JOB_ID}"
[[ ! -e "$OUTPUT_DIR" ]] || \
  ptm_die "immutable aggregate output already exists: $OUTPUT_DIR"

echo "experiment=Verify_SourceCenteredPairedScaleTemplate_1.1"
echo "stage=complete_five_unique_outer_family_fresh_aggregation"
echo "formal_confirmation=false"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "first_fold_job_id=$FIRST_FOLD_JOB_ID"
echo "remaining_fold_array_job_id=$REMAINING_FOLD_ARRAY_JOB_ID"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

/usr/bin/time -v python "$SOURCE_CENTERED_AGGREGATOR" \
  --config "$SOURCE_CENTERED_CONFIG" \
  --expected-config-sha256 "$SOURCE_CENTERED_CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_GIT_COMMIT" \
  --run-dir "${RUN_DIRS[0]}" \
  --run-dir "${RUN_DIRS[1]}" \
  --run-dir "${RUN_DIRS[2]}" \
  --run-dir "${RUN_DIRS[3]}" \
  --run-dir "${RUN_DIRS[4]}" \
  --output-dir "$OUTPUT_DIR" \
  --device cpu \
  --sidecar-input-manifest "$INPUT_MANIFEST" \
  --sidecar-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --sidecar-root "$SOURCE_CENTERED_SIDECAR_ROOT" \
  --sidecar-population-manifest "$POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$POPULATION_MANIFEST_SHA256"

readonly AGGREGATE_COMPLETE="$OUTPUT_DIR/AGGREGATE_COMPLETE.json"
readonly AGGREGATE_COMPLETE_SHA256=$(sha256sum "$AGGREGATE_COMPLETE" | awk '{print $1}')
readonly AGGREGATE_REPORT="$OUTPUT_DIR/aggregate_summary.json"
readonly AGGREGATE_REPORT_SHA256=$(sha256sum "$AGGREGATE_REPORT" | awk '{print $1}')
ptm_require_file_sha256 \
  "$AGGREGATE_COMPLETE" "$AGGREGATE_COMPLETE_SHA256" "five-fold aggregate completion"
ptm_require_file_sha256 \
  "$AGGREGATE_REPORT" "$AGGREGATE_REPORT_SHA256" "five-fold aggregate report"

python - "$OUTPUT_DIR" "$AGGREGATE_COMPLETE_SHA256" "$EXPECTED_GIT_COMMIT" "$SOURCE_CENTERED_CONFIG_SHA256" "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "$SOURCE_CENTERED_SIDECAR_ROOT" "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import aggregate_verify_source_centered_paired_scale_template_1_1 as aggregate

root = Path(sys.argv[1]).resolve()
(
    completion_sha,
    commit,
    config_sha,
    input_text,
    input_sha,
    sidecar_root_text,
    population_text,
    population_sha,
) = sys.argv[2:10]
assert {path.name for path in root.iterdir()} == {
    "outer_family_summary.csv",
    "aggregate_summary.json",
    "aggregate_manifest.json",
    "AGGREGATE_COMPLETE.json",
}
completion = aggregate.runner._read_self_hashed_json(
    root / "AGGREGATE_COMPLETE.json", expected_file_sha256=completion_sha
)
assert completion["schema"] == aggregate.AGGREGATE_COMPLETE_SCHEMA
assert completion["status"] == "completed"
assert completion["mode"] == "complete_five_fold_aggregate"
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
assert report["schema"] == aggregate.AGGREGATE_SUMMARY_SCHEMA
assert report["outer_families"] == list(aggregate.runner.FAMILY_ORDER)
assert len(report["folds"]) == 5
assert isinstance(report["all_template_success_conditions_pass"], bool)
assert isinstance(report["stop_version"], bool)
assert report["paired_bootstrap"]["replicates"] == 5000
assert report["paired_bootstrap"]["seed"] == 17068
assert manifest["schema"] == aggregate.AGGREGATE_MANIFEST_SCHEMA
assert manifest["mode"] == "complete_five_fold_aggregate"
assert len(manifest["source_folds"]) == 5
assert aggregate.sha256_file(root / manifest["outer_family_summary_file"]) == (
    manifest["outer_family_summary_file_sha256"]
)
assert report["outer_family_summary_file_sha256"] == (
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
print(f"five_family_macro_f1={report['success_rule']['family_macro']['f1']:.12g}")
print(f"all_template_success_conditions_pass={str(report['all_template_success_conditions_pass']).lower()}")
print(f"stop_version={str(report['stop_version']).lower()}")
PY

ptm_stage_unchanged "$WRAPPER" tests/test_all.py
echo "aggregate_dir=$OUTPUT_DIR"
echo "aggregate_complete_sha256=$AGGREGATE_COMPLETE_SHA256"
echo "aggregate_report_sha256=$AGGREGATE_REPORT_SHA256"
echo "aggregate_status=complete_five_fold_artifacts_freshly_authenticated"
