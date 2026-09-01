#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMRLFirst
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
readonly WRAPPER=ibex/verify_source_centered_rank_likelihood_template_1.1_first_fold.sh
readonly PARENT_BINDING=${RANK_PARENT_BINDING:?RANK_PARENT_BINDING is required}
readonly PARENT_BINDING_SHA256=${RANK_PARENT_BINDING_SHA256:?RANK_PARENT_BINDING_SHA256 is required}
readonly BINDING_COMPLETION=${RANK_BINDING_COMPLETION:?RANK_BINDING_COMPLETION is required}
readonly BINDING_COMPLETION_SHA256=${RANK_BINDING_COMPLETION_SHA256:?RANK_BINDING_COMPLETION_SHA256 is required}

rank_stage_gate "$WRAPPER" tests/test_all.py
rank_activate_runtime first_fold "${SLURM_CPUS_PER_TASK:-32}"
rank_targeted_preflight
rank_full_preflight
rank_authenticate_parent_binding \
  "$PARENT_BINDING" "$PARENT_BINDING_SHA256" \
  "$BINDING_COMPLETION" "$BINDING_COMPLETION_SHA256"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly RUN_DIR="$RANK_EXPERIMENT_ROOT/runs/slurm_${JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
[[ ! -e "$RUN_DIR" ]] || \
  rank_die "immutable first-fold output already exists: $RUN_DIR"

echo "experiment=Verify_SourceCenteredRankLikelihoodTemplate_1.1"
echo "stage=first_outer_fold_before_any_remaining_fold_submission"
echo "outer_family=half_cylinder"
echo "formal_confirmation=false"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "run_dir=$RUN_DIR"
hostname
lscpu

/usr/bin/time -v python "$RANK_RUNNER" \
  --config "$RANK_CONFIG" \
  --expected-config-sha256 "$RANK_CONFIG_SHA256" \
  --outer-family half_cylinder \
  --output-dir "$RUN_DIR" \
  --device cpu \
  --parent-binding "$PARENT_BINDING" \
  --parent-binding-sha256 "$PARENT_BINDING_SHA256" \
  --binding-completion "$BINDING_COMPLETION" \
  --binding-completion-sha256 "$BINDING_COMPLETION_SHA256"

readonly RUN_COMPLETE="$RUN_DIR/RUN_COMPLETE.json"
readonly RUN_COMPLETE_SHA256=$(sha256sum "$RUN_COMPLETE" | awk '{print $1}')
rank_require_file_sha256 "$RUN_COMPLETE" "$RUN_COMPLETE_SHA256" "first-fold completion"
python - "$RUN_DIR" "$RUN_COMPLETE_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import run_verify_source_centered_rank_likelihood_template_1_1 as runner

root = Path(sys.argv[1]).resolve()
assert {path.name for path in root.iterdir()} == set(runner.REQUIRED_FOLD_FILES)
completion = runner.source_runner._read_self_hashed_json(
    root / "RUN_COMPLETE.json", expected_file_sha256=sys.argv[2]
)
assert completion["schema"] == runner.COMPLETE_SCHEMA
assert completion["outer_family"] == "half_cylinder"
print("first_fold_exact_18_file_completion=passed")
PY

rank_stage_unchanged "$WRAPPER" tests/test_all.py
echo "first_fold_job_id=$JOB_ID"
echo "first_fold_run_dir=$RUN_DIR"
echo "first_fold_completion_sha256=$RUN_COMPLETE_SHA256"
echo "first_fold_status=completed_with_prediction_sealed_before_outer_labels"
