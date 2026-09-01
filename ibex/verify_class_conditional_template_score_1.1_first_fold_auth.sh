#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMClassAuth1
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
source ibex/verify_class_conditional_template_score_1.1_common.sh
readonly WRAPPER=ibex/verify_class_conditional_template_score_1.1_first_fold_auth.sh
readonly FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required}
readonly SMOKE_PASS=${RESOURCE_SMOKE_PASS:?RESOURCE_SMOKE_PASS is required}
readonly SMOKE_PASS_SHA256=${RESOURCE_SMOKE_PASS_SHA256:?RESOURCE_SMOKE_PASS_SHA256 is required}
[[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || ptm_class_die "FIRST_FOLD_JOB_ID must be numeric"

ptm_class_stage_gate "$WRAPPER"
ptm_class_activate_runtime first_fold_auth "${SLURM_CPUS_PER_TASK:-32}"
ptm_class_require_slurm_resources
ptm_class_targeted_preflight
ptm_class_require_evidence
ptm_class_require_resource_smoke "$SMOKE_PASS" "$SMOKE_PASS_SHA256"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly FOLD_DIR="$CLASS_EXPERIMENT_ROOT/runs/slurm_${FIRST_FOLD_JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
readonly OUTPUT_DIR="$CLASS_EXPERIMENT_ROOT/aggregate/slurm_${JOB_ID}_${SHORT_COMMIT}"
[[ -d "$FOLD_DIR" ]] || ptm_class_die "first fold is missing: $FOLD_DIR"
[[ ! -e "$OUTPUT_DIR" ]] || ptm_class_die "immutable first-fold authentication output exists: $OUTPUT_DIR"

/usr/bin/time -v python "$CLASS_AGGREGATOR" \
  --config "$CLASS_CONFIG" --expected-config-sha256 "$CLASS_CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_GIT_COMMIT" \
  --run-dir "$FOLD_DIR" --output-dir "$OUTPUT_DIR" \
  --mode single-fold --device cpu \
  --kinematic-input-manifest "$CLASS_INPUT_MANIFEST" \
  --kinematic-input-manifest-sha256 "$CLASS_INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$CLASS_SYNTHETIC_PASS" \
  --synthetic-pass-sha256 "$CLASS_SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$CLASS_SIDECAR_ROOT" \
  --sidecar-population-manifest "$CLASS_POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$CLASS_POPULATION_MANIFEST_SHA256"

readonly COMPLETION="$OUTPUT_DIR/AGGREGATE_COMPLETE.json"
COMPLETION_SHA256=$(sha256sum "$COMPLETION" | awk '{print $1}')
readonly COMPLETION_SHA256
STOP_VERSION=$(python - "$OUTPUT_DIR" "$COMPLETION_SHA256" "$EXPECTED_GIT_COMMIT" "$CLASS_CONFIG_SHA256" "$FOLD_DIR" <<'PY'
from pathlib import Path
import sys
from scripts.aggregate_verify_class_conditional_template_score_1_1 import (
    authenticate_single_fold_release,
)

release = authenticate_single_fold_release(
    Path(sys.argv[1]),
    expected_completion_sha256=sys.argv[2],
    expected_fold_commit=sys.argv[3],
    expected_config_sha256=sys.argv[4],
    expected_fold_directory=Path(sys.argv[5]),
)
print(str(bool(release["stop_version"])).lower())
PY
)
readonly STOP_VERSION
ptm_class_stage_unchanged "$WRAPPER"
echo "first_fold_auth_dir=$OUTPUT_DIR"
echo "first_fold_auth_complete_sha256=$COMPLETION_SHA256"
echo "stop_version=$STOP_VERSION"
