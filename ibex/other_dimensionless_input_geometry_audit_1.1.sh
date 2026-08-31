#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMDimGeomAudit
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-dimensionless-deformation
#SBATCH -o /home/zhanx0o/pathline-template-matching-dimensionless-deformation/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-dimensionless-deformation/slurm_logs/%x.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=cpu_amd_epyc_7702

set -euo pipefail

readonly PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-dimensionless-deformation
readonly EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Other_DimensionlessInputGeometryAudit_1.1
readonly CONFIG=config/Other_DimensionlessInputGeometryAudit_1.1.yaml
readonly CONFIG_SHA256=c874a8d9f6abbab452c6543139073eea2ac88e3db99ea13f78e0c3d43e03f566
readonly RUNNER=scripts/run_other_dimensionless_input_geometry_audit_1_1.py
readonly TEST=tests/test_dimensionless_input_geometry_audit.py
readonly WRAPPER=ibex/other_dimensionless_input_geometry_audit_1.1.sh
readonly DOCUMENT=docs/Other_DimensionlessInputGeometryAudit_1.1.md
readonly INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_50998592_260a07ad380d/train_cache_input_manifest.json
readonly INPUT_MANIFEST_SHA256=e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393
readonly INPUT_MANIFEST_SIZE=24009

die() {
  echo "$*" >&2
  exit 2
}

require_file_sha256() {
  local path=$1
  local expected=$2
  local label=$3
  [[ -f "$path" ]] || die "$label is missing: $path"
  local observed
  observed=$(sha256sum "$path" | awk '{print $1}')
  [[ "$observed" == "$expected" ]] || die "$label SHA-256 mismatch: $observed"
}

readonly EXPECTED_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "EXPECTED_GIT_COMMIT must be lowercase 40-hex"
readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}

cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || die "worktree is not clean"
readonly COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
[[ "$COMMIT_ID" == "$EXPECTED_COMMIT" ]] || die "checkout commit differs from EXPECTED_GIT_COMMIT"
for source in "$CONFIG" "$RUNNER" "$TEST" "$WRAPPER" "$DOCUMENT"; do
  git ls-files --error-unmatch "$source" >/dev/null || die "required source is not tracked: $source"
done
require_file_sha256 "$CONFIG" "$CONFIG_SHA256" "geometry-audit config"
require_file_sha256 "$INPUT_MANIFEST" "$INPUT_MANIFEST_SHA256" "frozen 32-shard input manifest"
[[ "$(stat -c %s "$INPUT_MANIFEST")" == "$INPUT_MANIFEST_SIZE" ]] || die "input manifest size mismatch"

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONOPTIMIZE
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
readonly JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_dimensionless_geometry_audit_${JOB_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

readonly SHORT_COMMIT=${COMMIT_ID:0:12}
readonly OUTPUT_DIR="$EXPERIMENT_ROOT/runs/slurm_${JOB_ID}_${SHORT_COMMIT}"
[[ ! -e "$OUTPUT_DIR" ]] || die "immutable geometry-audit output already exists: $OUTPUT_DIR"

echo "experiment=Other_DimensionlessInputGeometryAudit_1.1"
echo "evidence_scope=exposed_development_label_free_geometry_only"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "input_manifest=$INPUT_MANIFEST"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

python -c 'assert __debug__, "Python assertions must remain enabled"'
python -m py_compile "$RUNNER" "$TEST"
python tests/test_dimensionless_input_geometry_audit.py
[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || \
  die "preflight changed the commit or clean worktree"

/usr/bin/time -v python "$RUNNER" \
  --config "$CONFIG" \
  --expected-git-commit "$COMMIT_ID" \
  --output-dir "$OUTPUT_DIR"

python - "$OUTPUT_DIR" "$COMMIT_ID" "$CONFIG_SHA256" <<'PY'
import csv
import hashlib
import json
from pathlib import Path
import sys

from scripts import run_other_dimensionless_input_geometry_audit_1_1 as runner


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1]).resolve()
expected_commit = sys.argv[2]
expected_config_sha256 = sys.argv[3]
assert {path.name for path in root.iterdir()} == {
    "per_shard_geometry.csv",
    "per_scale_geometry.csv",
    "summary.json",
    "RUN_COMPLETE.json",
}
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
completion = json.loads((root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
runner.authenticate_self_hash(summary)
runner.authenticate_self_hash(completion)
assert summary["schema"] == runner.SUMMARY_SCHEMA
assert completion["schema"] == runner.COMPLETE_SCHEMA
assert summary["git_commit"] == completion["git_commit"] == expected_commit
assert summary["config"]["sha256"] == completion["config_sha256"] == expected_config_sha256
assert summary["authenticated_cache_file_count"] == 32
assert summary["opened_cache_members_in_exact_order"] == list(runner.ALLOWED_MEMBERS)
assert summary["forbidden_cache_member_open_count"] == 0
assert summary["test_dataset_access"] is False
assert summary["raw_flow_access"] is False
assert summary["cache_sidecar_access"] is False
assert summary["label_access"] is False
assert summary["metric_access"] is False
assert completion["summary_file_sha256"] == file_sha256(root / "summary.json")
assert completion["summary_content_sha256"] == summary["content_sha256"]
for name, expected_rows in (
    ("per_shard_geometry.csv", 32),
    ("per_scale_geometry.csv", summary["counts"]["observed_shard_scale_count"]),
):
    with (root / name).open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == expected_rows
    assert summary["artifacts"][name]["row_count"] == expected_rows
    assert summary["artifacts"][name]["sha256"] == file_sha256(root / name)
print("dimensionless_input_geometry_audit_authentication=passed")
PY

[[ "$(git rev-parse --verify HEAD^{commit})" == "$COMMIT_ID" && -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || \
  die "commit or clean worktree changed during geometry audit"
echo "dimensionless_input_geometry_audit_status=complete_authenticated"
