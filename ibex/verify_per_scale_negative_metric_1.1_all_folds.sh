#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMPerScaleAll
#SBATCH --array=0-4
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-tail-cpu
#SBATCH -o /home/zhanx0o/pathline-template-matching-tail-cpu/slurm_logs/%x.%A_%a.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-tail-cpu/slurm_logs/%x.%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-tail-cpu
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Verify_PerScaleNegativeMetric_1.1
CONFIG=config/Verify_PerScaleNegativeMetric_1.1.yaml
CONFIG_SHA256=b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79
RUNNER=scripts/run_verify_per_scale_negative_metric_1_1.py
RUNNER_SHA256=189b58205be983aa15858f84d54b97aad1e8e78f11c41c1111bdf938ead42497
INPUT_MANIFEST=/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_50998592_260a07ad380d/train_cache_input_manifest.json
INPUT_MANIFEST_SHA256=e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393
OUTER_FAMILIES=(half_cylinder delta_wing f22_raptor channel boeing_747)

EXPECTED_FOLD_COMMIT=${EXPECTED_FOLD_COMMIT:?EXPECTED_FOLD_COMMIT is required}
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if [[ ! "$EXPECTED_FOLD_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_FOLD_COMMIT must be a lowercase 40-character Git commit" >&2
  exit 2
fi
if [[ ! "$TASK_ID" =~ ^[0-4]$ ]]; then
  echo "array task must be one of 0,1,2,3,4: $TASK_ID" >&2
  exit 3
fi
OUTER_FAMILY=${OUTER_FAMILIES[$TASK_ID]}

cd "$PROJECT_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing experiment" >&2
  exit 4
fi
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
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_per_scale_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
export NUMBA_CACHE_DIR="$JOB_TMP_ROOT/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

COMMIT_ID=$(git rev-parse --verify HEAD^{commit})
if [[ "$COMMIT_ID" != "$EXPECTED_FOLD_COMMIT" ]]; then
  echo "checked-out commit does not match EXPECTED_FOLD_COMMIT: $COMMIT_ID" >&2
  exit 5
fi
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${SLURM_ARRAY_JOB_ID}_${TASK_ID}_${SHORT_COMMIT}_outer_${OUTER_FAMILY}"

ACTUAL_CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
ACTUAL_RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
ACTUAL_INPUT_SHA=$(sha256sum "$INPUT_MANIFEST" | awk '{print $1}')
[[ "$ACTUAL_CONFIG_SHA" == "$CONFIG_SHA256" ]] || { echo "config SHA-256 mismatch: $ACTUAL_CONFIG_SHA" >&2; exit 6; }
[[ "$ACTUAL_RUNNER_SHA" == "$RUNNER_SHA256" ]] || { echo "runner SHA-256 mismatch: $ACTUAL_RUNNER_SHA" >&2; exit 7; }
[[ "$ACTUAL_INPUT_SHA" == "$INPUT_MANIFEST_SHA256" ]] || { echo "input manifest SHA-256 mismatch: $ACTUAL_INPUT_SHA" >&2; exit 8; }

echo "experiment=Verify_PerScaleNegativeMetric_1.1"
echo "phase=cpu_complete_five_outer_folds"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "outer_family=$OUTER_FAMILY"
echo "run_dir=$RUN_DIR"
hostname
lscpu

python -c 'assert __debug__, "Python assertions must remain enabled"'
/usr/bin/time -v python tests/test_all.py
/usr/bin/time -v python scripts/validate_matcher_backend.py --device cpu
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 9
fi

/usr/bin/time -v python "$RUNNER" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --outer-family "$OUTER_FAMILY" \
  --output-dir "$RUN_DIR" \
  --device cpu

# Rebuild both artifacts and replay only label-free outer projections. This
# intentionally does not open labels, metrics, or invoke aggregation.
/usr/bin/time -v python - "$RUN_DIR" "$CONFIG" "$OUTER_FAMILY" "$CONFIG_SHA256" "$COMMIT_ID" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import sha256_file
from scripts.run_verify_per_scale_negative_metric_1_1 import (
    COMPLETE_SCHEMA,
    RESULT_SCHEMA,
    TailCandidateSpec,
    _authenticate_self_hash,
    authenticate_and_rebuild_final_calibration,
    authenticate_and_rebuild_final_scaler,
    authenticate_outer_prediction,
    authenticate_selected_candidate,
    load_cache_projection,
    load_cache_rows,
    load_plan,
)

root = Path(sys.argv[1]).resolve()
plan = load_plan(Path(sys.argv[2]).resolve())
outer_family, config_sha, git_commit = sys.argv[3:6]
assert plan.sha256 == config_sha
completion = json.loads((root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
_authenticate_self_hash(completion)
assert completion["schema"] == COMPLETE_SCHEMA
assert completion["outer_family"] == outer_family
assert completion["git_commit"] == git_commit
result_path = root / completion["result_manifest_file"]
assert sha256_file(result_path) == completion["result_manifest_file_sha256"]
result = json.loads(result_path.read_text(encoding="utf-8"))
_authenticate_self_hash(result)
assert result["schema"] == RESULT_SCHEMA
assert result["content_sha256"] == completion["result_manifest_content_sha256"]
assert result["config_sha256"] == config_sha
assert result["git_commit"] == git_commit
assert set(result["artifacts"]) == set(plan.required_fold_files) - {
    "result_manifest.json", "RUN_COMPLETE.json"
}
assert len(plan.required_fold_files) == 15
assert {path.name for path in root.iterdir()} == set(plan.required_fold_files)
for name, identity in result["artifacts"].items():
    path = root / name
    assert path.stat().st_size == identity["size_bytes"]
    assert sha256_file(path) == identity["sha256"]

payload = result["selected_candidate"]
selected = TailCandidateSpec(
    str(payload["representation"]), int(payload["k"]), float(payload["sigma"]),
    str(payload["decision_rule"]), float(payload["decision_value"]),
)
assert selected.candidate_id == payload["candidate_id"]
fit_families = tuple(value for value in plan.family_order if value != outer_family)
scaler = authenticate_and_rebuild_final_scaler(
    root / "final_per_scale_scaler.npz",
    root / "final_per_scale_scaler_manifest.json",
    plan=plan, selected=selected, outer_family=outer_family,
    fit_families=fit_families, git_commit=git_commit,
    expected_manifest_file_sha256=result["final_scaler_manifest_file_sha256"],
)
assert scaler.scaler_file_sha256 == result["final_scaler_file_sha256"]
calibration = authenticate_and_rebuild_final_calibration(
    root / "final_tail_calibration.npz",
    root / "final_tail_calibration_manifest.json",
    plan=plan, selected=selected, scaler=scaler, outer_family=outer_family,
    fit_families=fit_families, git_commit=git_commit,
    expected_manifest_file_sha256=result["final_calibration_manifest_file_sha256"],
)
selected_artifact = authenticate_selected_candidate(
    root / "selected_candidate.json", plan=plan, selected=selected,
    scaler=scaler, calibration=calibration,
    inner_group_metrics_path=root / "inner_group_metrics.csv",
    inner_group_metrics_sha256=result["inner_group_metrics_file_sha256"],
    inner_candidate_summary_path=root / "inner_candidate_summary.csv",
    inner_candidate_summary_sha256=result["inner_candidate_summary_file_sha256"],
    inner_fit_audits_path=root / "inner_fit_audits.json",
    inner_fit_audits_sha256=result["inner_fit_audits_file_sha256"],
    outer_family=outer_family, git_commit=git_commit,
    expected_file_sha256=result["selected_candidate_file_sha256"],
)
rows, _ = load_cache_rows(plan)
outer_rows = [row for row in rows if row.family == outer_family]
outer_projections = [
    load_cache_projection(plan, row, include_labels=False) for row in outer_rows
]
prediction = authenticate_outer_prediction(
    root / "outer_predictions.npz", root / "outer_prediction_manifest.json",
    plan=plan, selected=selected, selected_artifact=selected_artifact,
    scaler=scaler, calibration=calibration,
    outer_projections=outer_projections, expected_outer_rows=outer_rows,
    outer_family=outer_family, git_commit=git_commit, device="cpu",
    expected_manifest_file_sha256=result["prediction_manifest_file_sha256"],
)
assert prediction.prediction_file_sha256 == result["prediction_file_sha256"]
print("label_free_postvalidation=passed")
PY

[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA256" ]] || { echo "runner changed during experiment" >&2; exit 10; }
if [[ "$(git rev-parse --verify HEAD^{commit})" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during experiment" >&2
  exit 11
fi
