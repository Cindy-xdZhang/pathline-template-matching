#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMSCFirst
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-source-centered
#SBATCH --output=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_source_centered_paired_scale_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_paired_scale_template_1.1_first_fold.sh
readonly INPUT_MANIFEST=${SOURCE_CENTERED_INPUT_MANIFEST:?SOURCE_CENTERED_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${SOURCE_CENTERED_INPUT_MANIFEST_SHA256:?SOURCE_CENTERED_INPUT_MANIFEST_SHA256 is required}
readonly POPULATION_MANIFEST=${SOURCE_CENTERED_POPULATION_MANIFEST:?SOURCE_CENTERED_POPULATION_MANIFEST is required}
readonly POPULATION_MANIFEST_SHA256=${SOURCE_CENTERED_POPULATION_MANIFEST_SHA256:?SOURCE_CENTERED_POPULATION_MANIFEST_SHA256 is required}
[[ "$(realpath "$POPULATION_MANIFEST")" == "$(realpath "$SOURCE_CENTERED_SIDECAR_ROOT")/SIDECAR_POPULATION.json" ]] || \
  ptm_die "population manifest path differs from the frozen sidecar root"

ptm_stage_gate "$WRAPPER" tests/test_all.py
ptm_activate_runtime first_fold "${SLURM_CPUS_PER_TASK:-32}"
ptm_targeted_preflight
ptm_full_preflight
ptm_require_file_sha256 \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "source-centered input manifest"
ptm_require_file_sha256 \
  "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" "sealed 32-sidecar population"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly RUN_DIR="$SOURCE_CENTERED_EXPERIMENT_ROOT/runs/slurm_${JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
[[ ! -e "$RUN_DIR" ]] || \
  ptm_die "immutable first-fold output already exists: $RUN_DIR"

echo "experiment=Verify_SourceCenteredPairedScaleTemplate_1.1"
echo "stage=first_outer_fold_before_any_remaining_fold_submission"
echo "outer_family=half_cylinder"
echo "formal_confirmation=false"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "run_dir=$RUN_DIR"
hostname
lscpu

/usr/bin/time -v python "$SOURCE_CENTERED_RUNNER" \
  --config "$SOURCE_CENTERED_CONFIG" \
  --expected-config-sha256 "$SOURCE_CENTERED_CONFIG_SHA256" \
  --outer-family half_cylinder \
  --output-dir "$RUN_DIR" \
  --device cpu \
  --sidecar-input-manifest "$INPUT_MANIFEST" \
  --sidecar-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --sidecar-root "$SOURCE_CENTERED_SIDECAR_ROOT" \
  --sidecar-population-manifest "$POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$POPULATION_MANIFEST_SHA256"

readonly RUN_COMPLETE="$RUN_DIR/RUN_COMPLETE.json"
readonly RUN_COMPLETE_SHA256=$(sha256sum "$RUN_COMPLETE" | awk '{print $1}')
ptm_require_file_sha256 "$RUN_COMPLETE" "$RUN_COMPLETE_SHA256" "first-fold completion"
ptm_stage_unchanged "$WRAPPER" tests/test_all.py
echo "first_fold_job_id=$JOB_ID"
echo "first_fold_run_dir=$RUN_DIR"
echo "first_fold_completion_sha256=$RUN_COMPLETE_SHA256"
echo "first_fold_status=completed_with_internal_fresh_prediction_before_reference_access"
