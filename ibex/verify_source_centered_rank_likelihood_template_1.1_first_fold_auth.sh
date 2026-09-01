#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMRLAuth1
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-rank-likelihood
#SBATCH --output=/home/zhanx0o/pathline-template-matching-rank-likelihood/slurm_logs/%x.%j.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-rank-likelihood/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_source_centered_rank_likelihood_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_rank_likelihood_template_1.1_first_fold_auth.sh
readonly FIRST_FOLD_JOB_ID=${RANK_FIRST_FOLD_JOB_ID:?RANK_FIRST_FOLD_JOB_ID is required}
readonly PARENT_BINDING=${RANK_PARENT_BINDING:?RANK_PARENT_BINDING is required}
readonly PARENT_BINDING_SHA256=${RANK_PARENT_BINDING_SHA256:?RANK_PARENT_BINDING_SHA256 is required}
readonly BINDING_COMPLETION=${RANK_BINDING_COMPLETION:?RANK_BINDING_COMPLETION is required}
readonly BINDING_COMPLETION_SHA256=${RANK_BINDING_COMPLETION_SHA256:?RANK_BINDING_COMPLETION_SHA256 is required}
rank_require_job_id "$FIRST_FOLD_JOB_ID" "RANK_FIRST_FOLD_JOB_ID"

rank_stage_gate "$WRAPPER"
rank_activate_runtime first_fold_auth "${SLURM_CPUS_PER_TASK:-32}"
rank_targeted_preflight
rank_authenticate_parent_binding \
  "$PARENT_BINDING" "$PARENT_BINDING_SHA256" \
  "$BINDING_COMPLETION" "$BINDING_COMPLETION_SHA256"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly FOLD_RUN_DIR="$RANK_EXPERIMENT_ROOT/runs/slurm_${FIRST_FOLD_JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
readonly OUTPUT_DIR="$RANK_EXPERIMENT_ROOT/authentication/slurm_${JOB_ID}_${SHORT_COMMIT}_firstfold_${FIRST_FOLD_JOB_ID}"
[[ -d "$FOLD_RUN_DIR" ]] || rank_die "first fold is missing: $FOLD_RUN_DIR"
[[ ! -e "$OUTPUT_DIR" ]] || \
  rank_die "immutable first-fold authentication output exists: $OUTPUT_DIR"

echo "experiment=Verify_SourceCenteredRankLikelihoodTemplate_1.1"
echo "stage=fresh_single_fold_replay_and_authenticated_stop_decision"
echo "first_fold_job_id=$FIRST_FOLD_JOB_ID"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

/usr/bin/time -v python "$RANK_AGGREGATOR" \
  --config "$RANK_CONFIG" \
  --expected-config-sha256 "$RANK_CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_GIT_COMMIT" \
  --run-dir "$FOLD_RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --parent-binding "$PARENT_BINDING" \
  --parent-binding-sha256 "$PARENT_BINDING_SHA256" \
  --binding-completion "$BINDING_COMPLETION" \
  --binding-completion-sha256 "$BINDING_COMPLETION_SHA256"

readonly AUTH_COMPLETE="$OUTPUT_DIR/AGGREGATE_COMPLETE.json"
readonly AUTH_COMPLETE_SHA256=$(sha256sum "$AUTH_COMPLETE" | awk '{print $1}')
readonly AUTH_REPORT="$OUTPUT_DIR/single_fold_authentication_report.json"
readonly AUTH_REPORT_SHA256=$(sha256sum "$AUTH_REPORT" | awk '{print $1}')
rank_require_file_sha256 \
  "$AUTH_COMPLETE" "$AUTH_COMPLETE_SHA256" "first-fold authentication completion"
rank_require_file_sha256 \
  "$AUTH_REPORT" "$AUTH_REPORT_SHA256" "first-fold authentication report"

readonly STOP_VERSION=$(python - "$OUTPUT_DIR" "$AUTH_COMPLETE_SHA256" "$EXPECTED_GIT_COMMIT" "$RANK_CONFIG_SHA256" "$FOLD_RUN_DIR" "$PARENT_BINDING_SHA256" "$BINDING_COMPLETION_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import aggregate_verify_source_centered_rank_likelihood_template_1_1 as aggregate

root = Path(sys.argv[1]).resolve()
completion_sha, commit, config_sha, expected_fold, binding_sha, binding_completion_sha = sys.argv[2:8]
assert {path.name for path in root.iterdir()} == {
    "outer_family_summary.csv",
    "single_fold_authentication_report.json",
    "aggregate_manifest.json",
    "AGGREGATE_COMPLETE.json",
}
completion = aggregate.runner.source_runner._read_self_hashed_json(
    root / "AGGREGATE_COMPLETE.json", expected_file_sha256=completion_sha
)
assert completion["schema"] == aggregate.AGGREGATE_COMPLETE_SCHEMA
assert completion["status"] == "completed"
assert completion["mode"] == "single_fold_authentication"
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
assert report["schema"] == aggregate.SINGLE_FOLD_SCHEMA
assert report["status"] == "completed"
assert report["mode"] == "single_fold_authentication"
assert report["outer_families"] == ["half_cylinder"]
assert len(report["folds"]) == 1
assert Path(report["folds"][0]["run_directory"]).resolve() == Path(expected_fold).resolve()
assert report["all_primary_success_conditions_pass"] is None
assert isinstance(report["stop_version"], bool)
assert manifest["schema"] == aggregate.AGGREGATE_MANIFEST_SCHEMA
assert manifest["mode"] == "single_fold_authentication"
assert len(manifest["source_folds"]) == 1
source = manifest["source_folds"][0]
assert Path(source["run_directory"]).resolve() == Path(expected_fold).resolve()
assert aggregate.sha256_file(Path(expected_fold) / "RUN_COMPLETE.json") == source["completion_file_sha256"]
assert aggregate.sha256_file(Path(expected_fold) / "result_manifest.json") == source["result_manifest_file_sha256"]
assert aggregate.sha256_file(root / manifest["outer_family_summary_file"]) == manifest["outer_family_summary_file_sha256"]
assert report["outer_family_summary_file_sha256"] == manifest["outer_family_summary_file_sha256"]
evidence = completion["source_centered_evidence"]
assert report["source_centered_evidence"] == evidence
assert manifest["source_centered_evidence"] == evidence
assert evidence["parent_binding_file_sha256"] == binding_sha
assert evidence["binding_completion_file_sha256"] == binding_completion_sha
print(str(report["stop_version"]).lower())
PY
)

rank_stage_unchanged "$WRAPPER"
echo "first_fold_auth_dir=$OUTPUT_DIR"
echo "first_fold_auth_complete_sha256=$AUTH_COMPLETE_SHA256"
echo "first_fold_auth_report_sha256=$AUTH_REPORT_SHA256"
echo "stop_version=$STOP_VERSION"
