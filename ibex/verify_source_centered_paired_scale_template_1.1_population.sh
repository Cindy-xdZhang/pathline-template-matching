#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=PTMSCSeal32
#SBATCH --account=pi-hadwigm
#SBATCH --partition=batch
#SBATCH --constraint=rome
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-source-centered
#SBATCH --output=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.out
#SBATCH --error=/home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

source ibex/verify_source_centered_paired_scale_template_1.1_common.sh
readonly WRAPPER=ibex/verify_source_centered_paired_scale_template_1.1_population.sh
readonly INPUT_MANIFEST=${SOURCE_CENTERED_INPUT_MANIFEST:?SOURCE_CENTERED_INPUT_MANIFEST is required}
readonly INPUT_MANIFEST_SHA256=${SOURCE_CENTERED_INPUT_MANIFEST_SHA256:?SOURCE_CENTERED_INPUT_MANIFEST_SHA256 is required}
readonly SIDECAR_ARRAY_JOB_ID=${SOURCE_CENTERED_SIDECAR_ARRAY_JOB_ID:?SOURCE_CENTERED_SIDECAR_ARRAY_JOB_ID is required}
ptm_require_job_id "$SIDECAR_ARRAY_JOB_ID" "SOURCE_CENTERED_SIDECAR_ARRAY_JOB_ID"
readonly POPULATION_MANIFEST="$SOURCE_CENTERED_SIDECAR_ROOT/SIDECAR_POPULATION.json"

ptm_stage_gate "$WRAPPER"
ptm_activate_runtime population "${SLURM_CPUS_PER_TASK:-16}"
ptm_targeted_preflight
ptm_require_file_sha256 \
  "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "source-centered input manifest"
[[ ! -e "$POPULATION_MANIFEST" ]] || \
  ptm_die "immutable population manifest already exists: $POPULATION_MANIFEST"

echo "experiment=Verify_SourceCenteredPairedScaleTemplate_1.1"
echo "stage=fresh_authenticate_every_and_only_32_sidecars_then_publish_population_last"
echo "git_commit=$EXPECTED_GIT_COMMIT"
echo "sidecar_array_job_id=$SIDECAR_ARRAY_JOB_ID"
echo "sidecar_root=$SOURCE_CENTERED_SIDECAR_ROOT"
hostname
lscpu

/usr/bin/time -v python "$SOURCE_CENTERED_PREPARER" \
  --project-root "$SOURCE_CENTERED_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  seal-population \
  --sidecar-root "$SOURCE_CENTERED_SIDECAR_ROOT" \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256"

readonly POPULATION_MANIFEST_SHA256=$(sha256sum "$POPULATION_MANIFEST" | awk '{print $1}')
ptm_require_file_sha256 \
  "$POPULATION_MANIFEST" "$POPULATION_MANIFEST_SHA256" "sealed 32-sidecar population"
/usr/bin/time -v python "$SOURCE_CENTERED_PREPARER" \
  --project-root "$SOURCE_CENTERED_PROJECT_ROOT" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  authenticate-population \
  --sidecar-root "$SOURCE_CENTERED_SIDECAR_ROOT" \
  --population-manifest "$POPULATION_MANIFEST" \
  --population-manifest-sha256 "$POPULATION_MANIFEST_SHA256" \
  --input-manifest "$INPUT_MANIFEST" \
  --input-manifest-sha256 "$INPUT_MANIFEST_SHA256"

ptm_stage_unchanged "$WRAPPER"
echo "source_centered_population_manifest=$POPULATION_MANIFEST"
echo "source_centered_population_manifest_sha256=$POPULATION_MANIFEST_SHA256"
echo "population_status=exact_32_sidecars_freshly_authenticated"
