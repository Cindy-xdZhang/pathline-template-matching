#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMNegTailViz
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-tail-viz
#SBATCH -o /home/zhanx0o/pathline-template-matching-tail-viz/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-tail-viz/slurm_logs/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-tail-viz
EXPERIMENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Other_NegativeTailVisualization_1.1
CONFIG=config/Other_NegativeTailVisualization_1.1.yaml
EXPECTED_CONFIG_SHA256=5a82a9d1af406043066316262e5dcefb1a0d559f6d66e82da16440a2066df131
: "${EXPECTED_NUMERICAL_COMMIT:?submit with --export=ALL,EXPECTED_NUMERICAL_COMMIT=<40-char committed revision>}"
cd "$PROJECT_ROOT"

if [[ ! "$EXPECTED_NUMERICAL_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_NUMERICAL_COMMIT must be a 40-character lowercase Git commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree contains tracked or untracked changes; refusing experiment" >&2
  exit 3
fi

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_negative_tail_viz_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$JOB_TMP_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"

COMMIT_ID=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
RUN_DIR="$EXPERIMENT_ROOT/runs/slurm_${SLURM_JOB_ID}_${SHORT_COMMIT}"
CONFIG_SHA256=$(sha256sum "$CONFIG" | awk '{print $1}')
if [[ "$COMMIT_ID" != "$EXPECTED_NUMERICAL_COMMIT" ]]; then
  echo "checked-out commit $COMMIT_ID differs from expected $EXPECTED_NUMERICAL_COMMIT" >&2
  exit 4
fi
if [[ "$CONFIG_SHA256" != "$EXPECTED_CONFIG_SHA256" ]]; then
  echo "config SHA-256 $CONFIG_SHA256 differs from frozen $EXPECTED_CONFIG_SHA256" >&2
  exit 5
fi

echo "experiment=Other_NegativeTailVisualization_1.1"
echo "evidence_scope=family-held-out_exposed-development"
echo "formal_confirmation=false"
echo "git_commit=$COMMIT_ID"
echo "config_sha256=$CONFIG_SHA256"
echo "run_dir=$RUN_DIR"
hostname

python tests/test_all.py
python tests/test_negative_tail_visualization.py
python -m py_compile \
  src/pathline_template_matching/negative_tail_visualization.py \
  scripts/run_other_negative_tail_visualization_1_1.py \
  tests/test_negative_tail_visualization.py
if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "preflight changed the commit or worktree" >&2
  exit 6
fi

python scripts/run_other_negative_tail_visualization_1_1.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR"

python - "$RUN_DIR" "$COMMIT_ID" "$EXPECTED_CONFIG_SHA256" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import canonical_json_sha256, sha256_file

root = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
config_sha = sys.argv[3]

def self_hashed(name, field):
    path = root / name
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    stored = payload.pop(field)
    assert stored == canonical_json_sha256(payload), name
    return value

input_manifest = self_hashed("input_manifest.json", "manifest_content_sha256")
visualization = self_hashed("visualization_manifest.json", "manifest_content_sha256")
result = self_hashed("result_manifest.json", "manifest_content_sha256")
completion = self_hashed("RUN_COMPLETE.json", "marker_content_sha256")
assert input_manifest["npz_array_access_before_manifest_write"] is False
assert completion["schema"] == "pathline_template_matching.negative_tail_visualization_run_complete.v1"
assert completion["experiment"] == "Other_NegativeTailVisualization_1.1"
assert completion["status"] == "complete"
assert completion["git_commit"] == commit
assert completion["config_sha256"] == config_sha
assert completion["figure_count"] == 8
assert sha256_file(root / "result_manifest.json") == completion["result_manifest_file_sha256"]
assert result["artifact_count"] == 70
assert visualization["entry_count"] == 8
assert len({(row["dataset"], row["scale_block_id"]) for row in visualization["entries"]}) == 8

expected_candidates = {
    "half_cylinder": "representation=chirality_all35|k=15|sigma=1.0|fixed_top_fraction=0.05",
    "boeing_747": "representation=real_neighbor36|k=1|sigma=1.0|fixed_top_fraction=0.05",
}
for row in visualization["entries"]:
    assert row["candidate_id"] == expected_candidates[row["outer_family"]]
    assert len(row["required_exports"]) == 7
    assert len(row["additional_audit_files"]) == 1
    kinds = {item["export_kind"] for item in row["required_exports"]}
    assert kinds == {
        "scene_npz",
        "scene_manifest_json",
        "svg_with_editable_text_and_rasterized_3d_marks",
        "pdf_with_editable_text_and_rasterized_3d_marks",
        "png_360dpi",
        "panel_alignment_json",
        "render_metadata_json",
    }
    for item in row["required_exports"]:
        path = root / item["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == item["size_bytes"]
        assert sha256_file(path) == item["sha256"]
    audit = row["additional_audit_files"][0]
    assert audit["export_kind"] == "pdf_text_minimum_5pt_audit_json"
    audit_path = root / audit["relative_path"]
    assert audit_path.is_file()
    assert audit_path.stat().st_size == audit["size_bytes"]
    assert sha256_file(audit_path) == audit["sha256"]
    audit_value = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_value["status"] == "PASS"
    assert audit_value["auditable"] is True
    assert audit_value["minimum_found_pt"] >= 5.0
    assert audit_value["below_minimum_count"] == 0
    assert row["post_download_qa_required"]["original_png_visual_review"] is True
for item in result["artifacts"]:
    path = root / item["relative_path"]
    assert path.is_file()
    assert path.stat().st_size == item["size_bytes"]
    assert sha256_file(path) == item["sha256"]
print(f"evaluation_status={completion['status']}")
print(f"query_count={completion['query_count']}")
print(f"result_manifest_file_sha256={completion['result_manifest_file_sha256']}")
print("post_download_qa_required=PyMuPDF rendered-collision audit plus original-PNG visual review")
PY

if [[ "$(git rev-parse HEAD)" != "$COMMIT_ID" || -n "$(git status --porcelain)" ]]; then
  echo "commit or clean worktree changed during experiment" >&2
  exit 7
fi
