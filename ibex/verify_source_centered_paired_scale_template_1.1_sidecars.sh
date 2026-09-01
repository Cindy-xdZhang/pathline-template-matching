#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMSCSidecars
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --array=0-31%2
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-source-centered
#SBATCH --output=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%A_%a.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail

source ibex/verify_source_centered_paired_scale_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_paired_scale_template_1.1_sidecars.sh
readonly INPUT_MANIFEST=${SOURCE_CENTERED_INPUT_MANIFEST:?SOURCE_CENTERED_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${SOURCE_CENTERED_INPUT_MANIFEST_SHA256:?SOURCE_CENTERED_INPUT_MANIFEST_SHA256 is required}
readonly PROFILE_ROOT=${SOURCE_CENTERED_PROFILE_ROOT:?SOURCE_CENTERED_PROFILE_ROOT is required}
readonly PROFILE_COMPLETION=${SOURCE_CENTERED_PROFILE_COMPLETION:?SOURCE_CENTERED_PROFILE_COMPLETION is required}
readonly PROFILE_COMPLETION_SHA256=${SOURCE_CENTERED_PROFILE_COMPLETION_SHA256:?SOURCE_CENTERED_PROFILE_COMPLETION_SHA256 is required}
readonly ROW_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
[[ "$ROW_INDEX" =~ ^([0-9]|[12][0-9]|3[01])$ ]] || \
  ptm_die "sidecar row index must be 0..31"

ptm_stage_gate "$WRAPPER"
ptm_activate_runtime "sidecar_${ROW_INDEX}" "${SLURM_CPUS_PER_TASK:-8}"
ptm_targeted_preflight
ptm_require_file_sha256 \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "source-centered input manifest"
ptm_require_file_sha256 \
  "$PROFILE_COMPLETION" "$PROFILE_COMPLETION_SHA256" "profile row completion"

readonly EXPECTED_PROFILE_COMPLETION=$(python - "$INPUT_MANIFEST" "$PROFILE_ROOT" <<'PY'
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
[[ "$(realpath "$PROFILE_COMPLETION")" == "$EXPECTED_PROFILE_COMPLETION" ]] || \
  ptm_die "profile completion is not the authenticated row-0 artifact"

# Reauthenticate the actual profiled NPZ and completion; a caller-provided
# boolean or resource note is never accepted as the release gate.
/usr/bin/time -v python "$SOURCE_CENTERED_PREPARER" \
  --project-root "$SOURCE_CENTERED_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  authenticate-row \
  --sidecar-root "$PROFILE_ROOT" \
  --row-index 0 \
  --completion-sha256 "$PROFILE_COMPLETION_SHA256" \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256"

echo "experiment=Verify_SourceCenteredPairedScaleTemplate_1.1"
echo "stage=exact_32_source_centered_sidecars_capped_at_two_concurrent_rows"
echo "row_index=$ROW_INDEX"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "sidecar_root=$SOURCE_CENTERED_SIDECAR_ROOT"
hostname
lscpu

/usr/bin/time -v python "$SOURCE_CENTERED_PREPARER" \
  --project-root "$SOURCE_CENTERED_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  build-sidecar \
  --sidecar-root "$SOURCE_CENTERED_SIDECAR_ROOT" \
  --row-index "$ROW_INDEX" \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256"

readonly ROW_COMPLETION=$(python - "$INPUT_MANIFEST" "$SOURCE_CENTERED_SIDECAR_ROOT" "$ROW_INDEX" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
row = manifest["rows"][int(sys.argv[3])]
relative = (
    Path(row["dataset"])
    / f"source_{int(row['source_ordinal']):02d}_index_{int(row['source_index']):06d}"
    / "SIDECAR_COMPLETE.json"
)
print((Path(sys.argv[2]) / relative).resolve())
PY
)
readonly ROW_COMPLETION_SHA256=$(sha256sum "$ROW_COMPLETION" | awk '{print $1}')
ptm_require_file_sha256 \
  "$ROW_COMPLETION" "$ROW_COMPLETION_SHA256" "sidecar row completion"
/usr/bin/time -v python "$SOURCE_CENTERED_PREPARER" \
  --project-root "$SOURCE_CENTERED_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  authenticate-row \
  --sidecar-root "$SOURCE_CENTERED_SIDECAR_ROOT" \
  --row-index "$ROW_INDEX" \
  --completion-sha256 "$ROW_COMPLETION_SHA256" \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256"

ptm_stage_unchanged "$WRAPPER"
echo "row_completion=$ROW_COMPLETION"
echo "row_completion_sha256=$ROW_COMPLETION_SHA256"
echo "row_status=sidecar_and_completion_freshly_authenticated"
