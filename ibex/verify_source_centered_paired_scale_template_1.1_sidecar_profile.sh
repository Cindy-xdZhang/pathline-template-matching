#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMSCProfile
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-source-centered
#SBATCH --output=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail

source ibex/verify_source_centered_paired_scale_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_paired_scale_template_1.1_sidecar_profile.sh
readonly INPUT_MANIFEST=${SOURCE_CENTERED_INPUT_MANIFEST:?SOURCE_CENTERED_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${SOURCE_CENTERED_INPUT_MANIFEST_SHA256:?SOURCE_CENTERED_INPUT_MANIFEST_SHA256 is required}

ptm_stage_gate "$WRAPPER"
ptm_activate_runtime sidecar_profile "${SLURM_CPUS_PER_TASK:-8}"
ptm_targeted_preflight
ptm_require_file_sha256 \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "source-centered input manifest"
/usr/bin/time -v python "$SOURCE_CENTERED_PREPARER" \
  --project-root "$SOURCE_CENTERED_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  authenticate-input \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --authenticate-all-rows

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly PROFILE_ROOT="$SOURCE_CENTERED_EXPERIMENT_ROOT/resource_profiles/slurm_${JOB_ID}_${SHORT_COMMIT}/single_row"
[[ ! -e "$PROFILE_ROOT" ]] || \
  ptm_die "immutable profile root already exists: $PROFILE_ROOT"

echo "experiment=Verify_SourceCenteredPairedScaleTemplate_1.1"
echo "stage=single_exact_assigned_row_resource_profile_before_array"
echo "profile_row_index=0"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "profile_root=$PROFILE_ROOT"
hostname
lscpu

/usr/bin/time -v python "$SOURCE_CENTERED_PREPARER" \
  --project-root "$SOURCE_CENTERED_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  build-sidecar \
  --sidecar-root "$PROFILE_ROOT" \
  --row-index 0 \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256"

readonly PROFILE_COMPLETION=$(python - "$INPUT_MANIFEST" "$PROFILE_ROOT" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
row = manifest["rows"][0]
relative = (
    Path(row["dataset"])
    / f"source_{int(row['source_ordinal']):02d}_index_{int(row['source_index']):06d}"
    / "SIDECAR_COMPLETE.json"
)
print((Path(sys.argv[2]) / relative).resolve())
PY
)
readonly PROFILE_COMPLETION_SHA256=$(sha256sum "$PROFILE_COMPLETION" | awk '{print $1}')
ptm_require_file_sha256 \
  "$PROFILE_COMPLETION" "$PROFILE_COMPLETION_SHA256" "profile row completion"
/usr/bin/time -v python "$SOURCE_CENTERED_PREPARER" \
  --project-root "$SOURCE_CENTERED_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  authenticate-row \
  --sidecar-root "$PROFILE_ROOT" \
  --row-index 0 \
  --completion-sha256 "$PROFILE_COMPLETION_SHA256" \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256"

python - "$PROFILE_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = sorted(path.name for path in root.rglob("*") if path.is_file())
assert files == ["SIDECAR_COMPLETE.json", "source_centered_seed_time_kinematics.npz"]
PY

ptm_stage_unchanged "$WRAPPER"
echo "profile_sidecar_root=$PROFILE_ROOT"
echo "profile_completion=$PROFILE_COMPLETION"
echo "profile_completion_sha256=$PROFILE_COMPLETION_SHA256"
echo "resource_decision=production_array_requires_this_exact_profile_and_concurrency_at_most_2"
