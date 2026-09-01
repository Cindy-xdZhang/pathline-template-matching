#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMClassBoeingAuth
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-class-conditional-score
#SBATCH -o /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --constraint=rome
#SBATCH --account=pi-hadwigm

set -euo pipefail
source ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_common.sh
readonly BOEING_DIAGNOSTIC_FOLD_JOB_ID=${BOEING_DIAGNOSTIC_FOLD_JOB_ID:?BOEING_DIAGNOSTIC_FOLD_JOB_ID is required}
readonly VERIFY_FIRST_FOLD_JOB_ID=${VERIFY_FIRST_FOLD_JOB_ID:?VERIFY_FIRST_FOLD_JOB_ID is required}
readonly VERIFY_FIRST_FOLD_AUTH_DIR=${VERIFY_FIRST_FOLD_AUTH_DIR:?VERIFY_FIRST_FOLD_AUTH_DIR is required}
readonly VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256=${VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256:?VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256 is required}
readonly VERIFY_RESOURCE_SMOKE_PASS=${VERIFY_RESOURCE_SMOKE_PASS:?VERIFY_RESOURCE_SMOKE_PASS is required}
readonly VERIFY_RESOURCE_SMOKE_PASS_SHA256=${VERIFY_RESOURCE_SMOKE_PASS_SHA256:?VERIFY_RESOURCE_SMOKE_PASS_SHA256 is required}
[[ "$BOEING_DIAGNOSTIC_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || \
  ptm_boeing_diag_die "BOEING_DIAGNOSTIC_FOLD_JOB_ID must be numeric"

ptm_boeing_diag_stage_gate
ptm_boeing_diag_activate_runtime auth "${SLURM_CPUS_PER_TASK:-32}"
ptm_boeing_diag_require_slurm_resources
ptm_boeing_diag_targeted_preflight
ptm_boeing_diag_require_input_evidence
ptm_boeing_diag_require_parent_releases \
  "$VERIFY_FIRST_FOLD_JOB_ID" \
  "$VERIFY_FIRST_FOLD_AUTH_DIR" \
  "$VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256" \
  "$VERIFY_RESOURCE_SMOKE_PASS" \
  "$VERIFY_RESOURCE_SMOKE_PASS_SHA256"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly FOLD_DIR="$BOEING_DIAG_EXPERIMENT_ROOT/runs/slurm_${BOEING_DIAGNOSTIC_FOLD_JOB_ID}_0_${SHORT_COMMIT}_outer_boeing_747"
readonly OUTPUT_DIR="$BOEING_DIAG_EXPERIMENT_ROOT/authentication/slurm_${JOB_ID}_${SHORT_COMMIT}"
[[ -d "$FOLD_DIR" ]] || ptm_boeing_diag_die "Boeing diagnostic fold is missing: $FOLD_DIR"
[[ ! -e "$OUTPUT_DIR" ]] || \
  ptm_boeing_diag_die "immutable Boeing diagnostic authentication output exists: $OUTPUT_DIR"

/usr/bin/time -v python "$BOEING_DIAG_AGGREGATOR" \
  --config "$BOEING_DIAG_CONFIG" \
  --expected-config-sha256 "$BOEING_DIAG_CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_GIT_COMMIT" \
  --run-dir "$FOLD_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device cpu \
  --kinematic-input-manifest "$BOEING_DIAG_INPUT_MANIFEST" \
  --kinematic-input-manifest-sha256 "$BOEING_DIAG_INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$BOEING_DIAG_SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$BOEING_DIAG_SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$BOEING_DIAG_SIDECAR_ROOT" \
  --sidecar-population-manifest "$BOEING_DIAG_POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$BOEING_DIAG_POPULATION_MANIFEST_SHA256"

readonly COMPLETION="$OUTPUT_DIR/DIAGNOSTIC_COMPLETE.json"
COMPLETION_SHA256=$(sha256sum "$COMPLETION" | awk '{print $1}')
readonly COMPLETION_SHA256
ptm_boeing_diag_stage_unchanged

# This public release authenticator is intentionally the final command.  It
# freshly reconstructs the Boeing result and is the only release boundary.
python - \
  "$OUTPUT_DIR" "$COMPLETION_SHA256" \
  "$EXPECTED_GIT_COMMIT" "$BOEING_DIAG_CONFIG_SHA256" "$FOLD_DIR" <<'PY'
from pathlib import Path
import sys

from scripts.aggregate_other_class_conditional_template_score_boeing_diagnostic_1_1 import (
    authenticate_diagnostic_release,
)

release = authenticate_diagnostic_release(
    Path(sys.argv[1]),
    expected_completion_sha256=sys.argv[2],
    expected_fold_commit=sys.argv[3],
    expected_config_sha256=sys.argv[4],
    expected_fold_directory=Path(sys.argv[5]),
)
print(f"boeing_diagnostic_auth_dir={sys.argv[1]}")
print(
    "boeing_diagnostic_complete_sha256="
    f"{release['release_files']['DIAGNOSTIC_COMPLETE.json']['sha256']}"
)
print(f"boeing_diagnostic_release_authenticated={release['outer_family']}")
PY
