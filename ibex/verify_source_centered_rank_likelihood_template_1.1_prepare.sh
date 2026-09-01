#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMRLPrepare
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-rank-likelihood
#SBATCH --output=/home/zhanx0o/pathline-template-matching-rank-likelihood/slurm_logs/%x.%j.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-rank-likelihood/slurm_logs/%x.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_source_centered_rank_likelihood_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_rank_likelihood_template_1.1_prepare.sh

rank_stage_gate "$WRAPPER" tests/test_all.py
rank_activate_runtime prepare "${SLURM_CPUS_PER_TASK:-16}"
rank_targeted_preflight
rank_full_preflight

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly PREPARATION_DIR="$RANK_EXPERIMENT_ROOT/preparation/slurm_${JOB_ID}_${SHORT_COMMIT}"
readonly PARENT_BINDING="$PREPARATION_DIR/parent_sidecar_binding.json"
readonly BINDING_COMPLETION="$PREPARATION_DIR/BINDING_COMPLETE.json"
[[ ! -e "$PREPARATION_DIR" ]] || \
  rank_die "immutable preparation directory already exists: $PREPARATION_DIR"

echo "experiment=Verify_SourceCenteredRankLikelihoodTemplate_1.1"
echo "stage=opaque_parent_sidecar_binding_without_sidecar_member_access"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "config_sha256=$RANK_CONFIG_SHA256"
echo "preparation_dir=$PREPARATION_DIR"
hostname
lscpu

/usr/bin/time -v python "$RANK_PREPARER" \
  --config "$RANK_CONFIG" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  build \
  --output-dir "$PREPARATION_DIR"

readonly PARENT_BINDING_SHA256=$(sha256sum "$PARENT_BINDING" | awk '{print $1}')
readonly BINDING_COMPLETION_SHA256=$(sha256sum "$BINDING_COMPLETION" | awk '{print $1}')
rank_authenticate_parent_binding \
  "$PARENT_BINDING" "$PARENT_BINDING_SHA256" \
  "$BINDING_COMPLETION" "$BINDING_COMPLETION_SHA256"

python - "$PREPARATION_DIR" "$PARENT_BINDING_SHA256" "$BINDING_COMPLETION_SHA256" <<'PY'
from pathlib import Path
import sys

from scripts import run_verify_source_centered_rank_likelihood_template_1_1 as runner

root = Path(sys.argv[1]).resolve()
binding_sha, completion_sha = sys.argv[2:4]
assert {path.name for path in root.iterdir()} == {
    "parent_sidecar_binding.json",
    "BINDING_COMPLETE.json",
}
binding = runner.source_runner._read_self_hashed_json(
    root / "parent_sidecar_binding.json", expected_file_sha256=binding_sha
)
completion = runner.source_runner._read_self_hashed_json(
    root / "BINDING_COMPLETE.json", expected_file_sha256=completion_sha
)
assert binding["schema"] == runner.PARENT_BINDING_SCHEMA
assert binding["status"] == "passed"
assert binding["sidecar_npz_members_opened"] == []
assert binding["labels_or_references_opened"] == []
assert completion["schema"] == runner.BINDING_COMPLETE_SCHEMA
assert completion["status"] == "passed"
assert completion["parent_binding_file_sha256"] == binding_sha
assert completion["parent_binding_content_sha256"] == binding["content_sha256"]
print("opaque_parent_binding_fresh_replay=passed")
PY

rank_stage_unchanged "$WRAPPER" tests/test_all.py
echo "parent_binding=$PARENT_BINDING"
echo "parent_binding_sha256=$PARENT_BINDING_SHA256"
echo "binding_completion=$BINDING_COMPLETION"
echo "binding_completion_sha256=$BINDING_COMPLETION_SHA256"
echo "preparation_status=historical_sidecars_bound_without_rebuild_or_member_open"
