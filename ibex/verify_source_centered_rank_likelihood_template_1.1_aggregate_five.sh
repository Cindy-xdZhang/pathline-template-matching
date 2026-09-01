#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMRLAgg5
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-rank-likelihood
#SBATCH --output=/home/zhanx0o/pathline-template-matching-rank-likelihood/slurm_logs/%x.%j.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-rank-likelihood/slurm_logs/%x.%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_source_centered_rank_likelihood_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_rank_likelihood_template_1.1_aggregate_five.sh
readonly PARENT_BINDING=${RANK_PARENT_BINDING:?RANK_PARENT_BINDING is required}
readonly PARENT_BINDING_SHA256=${RANK_PARENT_BINDING_SHA256:?RANK_PARENT_BINDING_SHA256 is required}
readonly BINDING_COMPLETION=${RANK_BINDING_COMPLETION:?RANK_BINDING_COMPLETION is required}
readonly BINDING_COMPLETION_SHA256=${RANK_BINDING_COMPLETION_SHA256:?RANK_BINDING_COMPLETION_SHA256 is required}
readonly FIRST_FOLD_JOB_ID=${RANK_FIRST_FOLD_JOB_ID:?RANK_FIRST_FOLD_JOB_ID is required}
readonly REMAINING_FOLD_ARRAY_JOB_ID=${RANK_REMAINING_FOLD_ARRAY_JOB_ID:?RANK_REMAINING_FOLD_ARRAY_JOB_ID is required}
rank_require_job_id "$FIRST_FOLD_JOB_ID" "RANK_FIRST_FOLD_JOB_ID"
rank_require_job_id "$REMAINING_FOLD_ARRAY_JOB_ID" "RANK_REMAINING_FOLD_ARRAY_JOB_ID"

rank_stage_gate "$WRAPPER" tests/test_all.py
rank_activate_runtime aggregate_five "${SLURM_CPUS_PER_TASK:-32}"
rank_targeted_preflight
rank_full_preflight
rank_authenticate_parent_binding \
  "$PARENT_BINDING" "$PARENT_BINDING_SHA256" \
  "$BINDING_COMPLETION" "$BINDING_COMPLETION_SHA256"

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
  RUN_DIR="$RANK_EXPERIMENT_ROOT/runs/slurm_${SOURCE_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${FAMILY}"
  [[ -d "$RUN_DIR" ]] || rank_die "required fold directory is missing: $RUN_DIR"
  RUN_DIRS+=("$RUN_DIR")
done

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly OUTPUT_DIR="$RANK_EXPERIMENT_ROOT/aggregate/slurm_${JOB_ID}_${SHORT_COMMIT}_first_${FIRST_FOLD_JOB_ID}_remaining_${REMAINING_FOLD_ARRAY_JOB_ID}"
[[ ! -e "$OUTPUT_DIR" ]] || rank_die "immutable aggregate output already exists: $OUTPUT_DIR"

echo "experiment=Verify_SourceCenteredRankLikelihoodTemplate_1.1"
echo "stage=complete_five_unique_outer_family_fresh_aggregation"
echo "formal_confirmation=false"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "first_fold_job_id=$FIRST_FOLD_JOB_ID"
echo "remaining_fold_array_job_id=$REMAINING_FOLD_ARRAY_JOB_ID"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

/usr/bin/time -v python "$RANK_AGGREGATOR" \
  --config "$RANK_CONFIG" \
  --expected-config-sha256 "$RANK_CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_GIT_COMMIT" \
  --run-dir "${RUN_DIRS[0]}" \
  --run-dir "${RUN_DIRS[1]}" \
  --run-dir "${RUN_DIRS[2]}" \
  --run-dir "${RUN_DIRS[3]}" \
  --run-dir "${RUN_DIRS[4]}" \
  --output-dir "$OUTPUT_DIR" \
  --parent-binding "$PARENT_BINDING" \
  --parent-binding-sha256 "$PARENT_BINDING_SHA256" \
  --binding-completion "$BINDING_COMPLETION" \
  --binding-completion-sha256 "$BINDING_COMPLETION_SHA256"

readonly AGGREGATE_COMPLETE="$OUTPUT_DIR/AGGREGATE_COMPLETE.json"
readonly AGGREGATE_COMPLETE_SHA256=$(sha256sum "$AGGREGATE_COMPLETE" | awk '{print $1}')
readonly AGGREGATE_REPORT="$OUTPUT_DIR/aggregate_summary.json"
readonly AGGREGATE_REPORT_SHA256=$(sha256sum "$AGGREGATE_REPORT" | awk '{print $1}')
rank_require_file_sha256 "$AGGREGATE_COMPLETE" "$AGGREGATE_COMPLETE_SHA256" "five-fold aggregate completion"
rank_require_file_sha256 "$AGGREGATE_REPORT" "$AGGREGATE_REPORT_SHA256" "five-fold aggregate report"

python - "$OUTPUT_DIR" "$AGGREGATE_COMPLETE_SHA256" "$EXPECTED_GIT_COMMIT" "$RANK_CONFIG_SHA256" "$PARENT_BINDING_SHA256" "$BINDING_COMPLETION_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import aggregate_verify_source_centered_rank_likelihood_template_1_1 as aggregate

root = Path(sys.argv[1]).resolve()
completion_sha, commit, config_sha, binding_sha, binding_completion_sha = sys.argv[2:7]
assert {path.name for path in root.iterdir()} == {
    "outer_family_summary.csv",
    "aggregate_summary.json",
    "aggregate_manifest.json",
    "AGGREGATE_COMPLETE.json",
}
completion = aggregate.runner.source_runner._read_self_hashed_json(
    root / "AGGREGATE_COMPLETE.json", expected_file_sha256=completion_sha
)
assert completion["schema"] == aggregate.AGGREGATE_COMPLETE_SCHEMA
assert completion["status"] == "completed"
assert completion["mode"] == "complete_five_fold_aggregate"
assert completion["aggregator_git_commit"] == commit
assert completion["fold_git_commit"] == commit
assert completion["config_sha256"] == config_sha
report = aggregate.runner.source_runner._read_self_hashed_json(
    root / completion["report_file"],
    expected_file_sha256=completion["report_file_sha256"],
)
manifest = aggregate.runner.source_runner._read_self_hashed_json(
    root / completion["aggregate_manifest_file"],
    expected_file_sha256=completion["aggregate_manifest_file_sha256"],
)
assert report["schema"] == aggregate.AGGREGATE_SUMMARY_SCHEMA
assert report["outer_families"] == list(aggregate.runner.FAMILY_ORDER)
assert len(report["folds"]) == 5
assert isinstance(report["all_primary_success_conditions_pass"], bool)
assert isinstance(report["stop_version"], bool)
assert report["controls_can_satisfy_primary_success"] is False
assert report["paired_bootstrap"]["replicates"] == 5000
assert report["paired_bootstrap"]["seed"] == 17068
assert manifest["schema"] == aggregate.AGGREGATE_MANIFEST_SCHEMA
assert manifest["mode"] == "complete_five_fold_aggregate"
assert len(manifest["source_folds"]) == 5
for source in manifest["source_folds"]:
    fold = Path(source["run_directory"]).resolve()
    assert aggregate.sha256_file(fold / "RUN_COMPLETE.json") == source["completion_file_sha256"]
    assert aggregate.sha256_file(fold / "result_manifest.json") == source["result_manifest_file_sha256"]
assert aggregate.sha256_file(root / manifest["outer_family_summary_file"]) == manifest["outer_family_summary_file_sha256"]
assert report["outer_family_summary_file_sha256"] == manifest["outer_family_summary_file_sha256"]
evidence = completion["source_centered_evidence"]
assert report["source_centered_evidence"] == evidence
assert manifest["source_centered_evidence"] == evidence
assert evidence["parent_binding_file_sha256"] == binding_sha
assert evidence["binding_completion_file_sha256"] == binding_completion_sha
print(f"five_family_macro_f1={report['success_rule']['family_macro']['f1']:.12g}")
print(f"all_primary_success_conditions_pass={str(report['all_primary_success_conditions_pass']).lower()}")
print(f"stop_version={str(report['stop_version']).lower()}")
PY

rank_stage_unchanged "$WRAPPER" tests/test_all.py
echo "aggregate_dir=$OUTPUT_DIR"
echo "aggregate_complete_sha256=$AGGREGATE_COMPLETE_SHA256"
echo "aggregate_report_sha256=$AGGREGATE_REPORT_SHA256"
echo "aggregate_status=complete_five_fold_artifacts_freshly_authenticated"
