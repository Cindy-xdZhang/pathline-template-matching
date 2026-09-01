#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMRLRemain
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --array=1-4%2
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-rank-likelihood
#SBATCH --output=/home/zhanx0o/pathline-template-matching-rank-likelihood/slurm_logs/%x.%A_%a.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-rank-likelihood/slurm_logs/%x.%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_source_centered_rank_likelihood_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_rank_likelihood_template_1.1_all_folds.sh
readonly PARENT_BINDING=${RANK_PARENT_BINDING:?RANK_PARENT_BINDING is required}
readonly PARENT_BINDING_SHA256=${RANK_PARENT_BINDING_SHA256:?RANK_PARENT_BINDING_SHA256 is required}
readonly BINDING_COMPLETION=${RANK_BINDING_COMPLETION:?RANK_BINDING_COMPLETION is required}
readonly BINDING_COMPLETION_SHA256=${RANK_BINDING_COMPLETION_SHA256:?RANK_BINDING_COMPLETION_SHA256 is required}
readonly FIRST_FOLD_JOB_ID=${RANK_FIRST_FOLD_JOB_ID:?RANK_FIRST_FOLD_JOB_ID is required}
readonly FIRST_AUTH_DIR=${RANK_FIRST_AUTH_DIR:?RANK_FIRST_AUTH_DIR is required}
readonly FIRST_AUTH_COMPLETE_SHA256=${RANK_FIRST_AUTH_COMPLETE_SHA256:?RANK_FIRST_AUTH_COMPLETE_SHA256 is required}
readonly TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
readonly -a OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)
[[ "$TASK_ID" =~ ^[1-4]$ ]] || rank_die "remaining-fold array task must be 1..4"
rank_require_job_id "$FIRST_FOLD_JOB_ID" "RANK_FIRST_FOLD_JOB_ID"
readonly OUTER_FAMILY=${OUTER_FAMILIES[$TASK_ID]}

rank_stage_gate "$WRAPPER"
rank_activate_runtime "fold_${TASK_ID}" "${SLURM_CPUS_PER_TASK:-32}"
rank_targeted_preflight
rank_authenticate_parent_binding \
  "$PARENT_BINDING" "$PARENT_BINDING_SHA256" \
  "$BINDING_COMPLETION" "$BINDING_COMPLETION_SHA256"
rank_require_file_sha256 \
  "$FIRST_AUTH_DIR/AGGREGATE_COMPLETE.json" \
  "$FIRST_AUTH_COMPLETE_SHA256" \
  "first-fold authentication completion"

# Do not trust a caller-provided continue flag.  Freshly authenticate the
# first-fold release and require its mathematically derived stop decision.
python - "$FIRST_AUTH_DIR" "$FIRST_AUTH_COMPLETE_SHA256" "$EXPECTED_GIT_COMMIT" "$RANK_CONFIG_SHA256" "$FIRST_FOLD_JOB_ID" "$RANK_EXPERIMENT_ROOT" "$PARENT_BINDING_SHA256" "$BINDING_COMPLETION_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import aggregate_verify_source_centered_rank_likelihood_template_1_1 as aggregate

root = Path(sys.argv[1]).resolve()
completion_sha, commit, config_sha, first_job_id, experiment_root_text, binding_sha, binding_completion_sha = sys.argv[2:9]
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
assert report["outer_families"] == ["half_cylinder"]
assert report["all_primary_success_conditions_pass"] is None
assert report["stop_version"] is False
assert manifest["schema"] == aggregate.AGGREGATE_MANIFEST_SCHEMA
assert manifest["mode"] == "single_fold_authentication"
assert len(manifest["source_folds"]) == 1
expected_fold = (
    Path(experiment_root_text)
    / "runs"
    / f"slurm_{first_job_id}_0_{commit[:12]}_outer_half_cylinder"
).resolve()
source = manifest["source_folds"][0]
assert Path(source["run_directory"]).resolve() == expected_fold
assert Path(report["folds"][0]["run_directory"]).resolve() == expected_fold
assert aggregate.sha256_file(expected_fold / "RUN_COMPLETE.json") == source["completion_file_sha256"]
assert aggregate.sha256_file(expected_fold / "result_manifest.json") == source["result_manifest_file_sha256"]
assert aggregate.sha256_file(root / manifest["outer_family_summary_file"]) == manifest["outer_family_summary_file_sha256"]
evidence = completion["source_centered_evidence"]
assert report["source_centered_evidence"] == evidence
assert manifest["source_centered_evidence"] == evidence
assert evidence["parent_binding_file_sha256"] == binding_sha
assert evidence["binding_completion_file_sha256"] == binding_completion_sha
print("first_fold_authenticated_release=continue_remaining_folds")
PY

readonly ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly RUN_DIR="$RANK_EXPERIMENT_ROOT/runs/slurm_${ARRAY_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${OUTER_FAMILY}"
[[ ! -e "$RUN_DIR" ]] || rank_die "immutable fold output already exists: $RUN_DIR"

echo "experiment=Verify_SourceCenteredRankLikelihoodTemplate_1.1"
echo "stage=remaining_outer_fold_after_authenticated_first_fold_release"
echo "outer_family=$OUTER_FAMILY"
echo "formal_confirmation=false"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "run_dir=$RUN_DIR"
hostname
lscpu

/usr/bin/time -v python "$RANK_RUNNER" \
  --config "$RANK_CONFIG" \
  --expected-config-sha256 "$RANK_CONFIG_SHA256" \
  --outer-family "$OUTER_FAMILY" \
  --output-dir "$RUN_DIR" \
  --device cpu \
  --parent-binding "$PARENT_BINDING" \
  --parent-binding-sha256 "$PARENT_BINDING_SHA256" \
  --binding-completion "$BINDING_COMPLETION" \
  --binding-completion-sha256 "$BINDING_COMPLETION_SHA256"

readonly RUN_COMPLETE="$RUN_DIR/RUN_COMPLETE.json"
readonly RUN_COMPLETE_SHA256=$(sha256sum "$RUN_COMPLETE" | awk '{print $1}')
rank_require_file_sha256 "$RUN_COMPLETE" "$RUN_COMPLETE_SHA256" "outer-fold completion"
python - "$RUN_DIR" "$RUN_COMPLETE_SHA256" "$OUTER_FAMILY" <<'PY'
from pathlib import Path
import sys

from scripts import run_verify_source_centered_rank_likelihood_template_1_1 as runner

root = Path(sys.argv[1]).resolve()
assert {path.name for path in root.iterdir()} == set(runner.REQUIRED_FOLD_FILES)
completion = runner.source_runner._read_self_hashed_json(
    root / "RUN_COMPLETE.json", expected_file_sha256=sys.argv[2]
)
assert completion["schema"] == runner.COMPLETE_SCHEMA
assert completion["outer_family"] == sys.argv[3]
print("remaining_fold_exact_18_file_completion=passed")
PY

rank_stage_unchanged "$WRAPPER"
echo "remaining_fold_completion_sha256=$RUN_COMPLETE_SHA256"
echo "remaining_fold_status=completed_with_prediction_sealed_before_outer_labels"
