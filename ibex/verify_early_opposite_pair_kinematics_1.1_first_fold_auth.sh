#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMEarlyAuth1
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-early-kinematics
#SBATCH -o /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-early-kinematics/slurm_logs/%x.%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail
source ibex/verify_early_opposite_pair_kinematics_1.1_common.sh
readonly WRAPPER=ibex/verify_early_opposite_pair_kinematics_1.1_first_fold_auth.sh
readonly FIRST_FOLD_JOB_ID=${FIRST_FOLD_JOB_ID:?FIRST_FOLD_JOB_ID is required}
[[ "$FIRST_FOLD_JOB_ID" =~ ^[0-9]+$ ]] || ptm_die "FIRST_FOLD_JOB_ID must be numeric"
readonly INPUT_MANIFEST=${KINEMATIC_INPUT_MANIFEST:?KINEMATIC_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${KINEMATIC_INPUT_MANIFEST_SHA256:?KINEMATIC_INPUT_MANIFEST_SHA256 is required}
readonly SYNTHETIC_PASS=${SYNTHETIC_PASS:?SYNTHETIC_PASS is required}
readonly SYNTHETIC_PASS_SHA256=${SYNTHETIC_PASS_SHA256:?SYNTHETIC_PASS_SHA256 is required}
readonly POPULATION_MANIFEST=${SIDECAR_POPULATION_MANIFEST:?SIDECAR_POPULATION_MANIFEST is required}
readonly POPULATION_MANIFEST_SHA256=${SIDECAR_POPULATION_MANIFEST_SHA256:?SIDECAR_POPULATION_MANIFEST_SHA256 is required}

ptm_stage_gate "$WRAPPER" "$EARLY_PREPARER" "$EARLY_RUNNER" "$EARLY_AGGREGATOR" tests/test_early_opposite_pair_kinematics_runner.py
ptm_activate_runtime first_fold_auth "${SLURM_CPUS_PER_TASK:-32}"
ptm_targeted_preflight
ptm_require_file_sha256 "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "kinematic input manifest"
ptm_require_file_sha256 "$SYNTHETIC_PASS" "$SYNTHETIC_PASS_SHA256" "production synthetic PASS"
ptm_require_file_sha256 "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" "sealed 32-sidecar population"

readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
readonly SHORT_COMMIT=${EXPECTED_GIT_COMMIT:0:12}
readonly FOLD_RUN_DIR="$EARLY_EXPERIMENT_ROOT/runs/slurm_${FIRST_FOLD_JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder"
readonly OUTPUT_DIR="$EARLY_EXPERIMENT_ROOT/early_stop/slurm_${JOB_ID}_${SHORT_COMMIT}_firstfold_${FIRST_FOLD_JOB_ID}"
[[ -d "$FOLD_RUN_DIR" ]] || ptm_die "first fold is missing: $FOLD_RUN_DIR"
[[ ! -e "$OUTPUT_DIR" ]] || ptm_die "immutable first-fold authentication output exists: $OUTPUT_DIR"

/usr/bin/time -v python "$EARLY_AGGREGATOR" \
  --config "$EARLY_CONFIG" --expected-config-sha256 "$EARLY_CONFIG_SHA256" \
  --expected-fold-commit "$EXPECTED_GIT_COMMIT" \
  --run-dir "$FOLD_RUN_DIR" --output-dir "$OUTPUT_DIR" \
  --mode single-fold --device cpu \
  --kinematic-input-manifest "$INPUT_MANIFEST" --kinematic-input-manifest-sha256 "$INPUT_MANIFEST_SHA256" \
  --synthetic-pass "$SYNTHETIC_PASS" --synthetic-pass-sha256 "$SYNTHETIC_PASS_SHA256" \
  --sidecar-root "$EARLY_SIDECAR_ROOT" \
  --sidecar-population-manifest "$POPULATION_MANIFEST" \
  --sidecar-population-manifest-sha256 "$POPULATION_MANIFEST_SHA256"

readonly COMPLETION="$OUTPUT_DIR/AGGREGATE_COMPLETE.json"
readonly COMPLETION_SHA256=$(sha256sum "$COMPLETION" | awk '{print $1}')
readonly CERTIFICATE="$OUTPUT_DIR/early_stop_certificate.json"
readonly CERTIFICATE_SHA256=$(sha256sum "$CERTIFICATE" | awk '{print $1}')
readonly STOP_VERSION=$(python -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1], encoding="utf-8"))["stop_version"])).lower())' "$CERTIFICATE")
ptm_stage_unchanged "$WRAPPER" "$EARLY_PREPARER" "$EARLY_RUNNER" "$EARLY_AGGREGATOR" tests/test_early_opposite_pair_kinematics_runner.py
echo "first_fold_auth_dir=$OUTPUT_DIR"
echo "first_fold_auth_complete_sha256=$COMPLETION_SHA256"
echo "early_stop_certificate_sha256=$CERTIFICATE_SHA256"
echo "stop_version=$STOP_VERSION"
