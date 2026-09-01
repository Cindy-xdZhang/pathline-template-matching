#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMSCPrepare
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-source-centered
#SBATCH --output=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_source_centered_paired_scale_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_paired_scale_template_1.1_prepare.sh

ptm_stage_gate \
  "$WRAPPER" \
  tests/test_all.py
ptm_activate_runtime prepare "${SLURM_CPUS_PER_TASK:-16}"
ptm_targeted_preflight
ptm_full_preflight
ptm_require_file_sha256 \
  "$SOURCE_CENTERED_EARLY_INPUT_MANIFEST" \
  "$SOURCE_CENTERED_EARLY_INPUT_SHA256" \
  "authenticated Early input manifest"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly PREPARATION_DIR="$SOURCE_CENTERED_EXPERIMENT_ROOT/preparation/slurm_${JOB_ID}_${SHORT_COMMIT}"
readonly INPUT_MANIFEST="$PREPARATION_DIR/source_centered_input_manifest.json"
[[ ! -e "$PREPARATION_DIR" ]] || \
  ptm_die "immutable preparation directory already exists: $PREPARATION_DIR"

echo "experiment=Verify_SourceCenteredPairedScaleTemplate_1.1"
echo "stage=freeze_and_fresh_authenticate_exact_32_label_free_inputs"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "config_sha256=$SOURCE_CENTERED_CONFIG_SHA256"
echo "preparation_dir=$PREPARATION_DIR"
hostname
lscpu

/usr/bin/time -v python "$SOURCE_CENTERED_PREPARER" \
  --project-root "$SOURCE_CENTERED_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  freeze-input \
  --output-path "$INPUT_MANIFEST" \
  --early-input-manifest "$SOURCE_CENTERED_EARLY_INPUT_MANIFEST"

readonly INPUT_MANIFEST_SHA256=$(sha256sum "$INPUT_MANIFEST" | awk '{print $1}')
ptm_require_file_sha256 \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "source-centered input manifest"
/usr/bin/time -v python "$SOURCE_CENTERED_PREPARER" \
  --project-root "$SOURCE_CENTERED_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  authenticate-input \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --authenticate-all-rows

python - "$PREPARATION_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
assert root.is_dir()
assert {path.name for path in root.iterdir()} == {
    "source_centered_input_manifest.json"
}
PY

ptm_stage_unchanged "$WRAPPER" tests/test_all.py
echo "source_centered_input_manifest=$INPUT_MANIFEST"
echo "source_centered_input_manifest_sha256=$INPUT_MANIFEST_SHA256"
echo "preparation_status=exact_32_inputs_freshly_authenticated"
