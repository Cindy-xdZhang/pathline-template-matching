#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMClassScoreViz
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-class-conditional-score
#SBATCH -o /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --constraint=rome
#SBATCH --account=pi-hadwigm

# ``cpu`` is the portable default.  Ibex submission may override it with
# ``sbatch --partition=batch ...``; the authenticated runtime accepts only
# those two CPU partitions.

set -euo pipefail

die() {
  echo "$*" >&2
  exit 2
}

require_lower_hex() {
  local value=$1
  local length=$2
  local name=$3
  [[ "$value" =~ ^[0-9a-f]+$ && ${#value} -eq $length ]] || \
    die "$name must be lowercase ${length}-hex"
}

readonly PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-class-conditional-score
readonly REPORTER=scripts/render_class_conditional_template_score_visualizations.py
readonly REPORT_TEST=tests/test_class_conditional_template_score_visualization.py
readonly WRAPPER_TEST=tests/test_class_conditional_template_score_visualization_ibex.py
readonly WRAPPER=ibex/other_class_conditional_template_score_visualization_1.1.sh

readonly EXPECTED_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
readonly REPORT_CONFIG_VALUE=${REPORT_CONFIG:?REPORT_CONFIG is required}
readonly EXPECTED_REPORT_CONFIG_SHA256=${REPORT_CONFIG_SHA256:?REPORT_CONFIG_SHA256 is required}
readonly OUTPUT_ROOT_VALUE=${OUTPUT_ROOT:?OUTPUT_ROOT is required}

require_lower_hex "$EXPECTED_COMMIT" 40 EXPECTED_GIT_COMMIT
require_lower_hex "$EXPECTED_REPORT_CONFIG_SHA256" 64 REPORT_CONFIG_SHA256
[[ "$REPORT_CONFIG_VALUE" == /* ]] || die "REPORT_CONFIG must be an absolute path"
[[ "$OUTPUT_ROOT_VALUE" == /* ]] || die "OUTPUT_ROOT must be an absolute path"

cd "$PROJECT_ROOT"
[[ -f "$REPORT_CONFIG_VALUE" ]] || die "REPORT_CONFIG is missing: $REPORT_CONFIG_VALUE"
readonly REPORT_CONFIG_PATH=$(realpath -- "$REPORT_CONFIG_VALUE")
readonly OUTPUT_DIR=$(realpath -m -- "$OUTPUT_ROOT_VALUE")
[[ "$OUTPUT_DIR" != "$PROJECT_ROOT" && "$OUTPUT_DIR" != "$PROJECT_ROOT/"* ]] || \
  die "OUTPUT_ROOT must be outside the committed checkout"
[[ ! -e "$OUTPUT_DIR" ]] || die "immutable report output already exists: $OUTPUT_DIR"

stage_gate() {
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || \
    die "worktree is not clean"
  local observed_commit
  observed_commit=$(git rev-parse --verify HEAD^{commit})
  [[ "$observed_commit" == "$EXPECTED_COMMIT" ]] || \
    die "checkout differs from EXPECTED_GIT_COMMIT"
  git cat-file -e "${EXPECTED_COMMIT}^{commit}"

  local -a sources=(
    "$REPORTER"
    "$REPORT_TEST"
    "$WRAPPER_TEST"
    "$WRAPPER"
    tests/test_all.py
  )
  local source actual committed
  for source in "${sources[@]}"; do
    git ls-files --error-unmatch "$source" >/dev/null
    actual=$(sha256sum "$source" | awk '{print $1}')
    committed=$(git show "${EXPECTED_COMMIT}:${source}" | sha256sum | awk '{print $1}')
    [[ "$actual" == "$committed" ]] || \
      die "source differs from expected commit: $source"
    echo "source_sha256[$source]=$actual"
  done

  local config_sha256
  config_sha256=$(sha256sum "$REPORT_CONFIG_PATH" | awk '{print $1}')
  [[ "$config_sha256" == "$EXPECTED_REPORT_CONFIG_SHA256" ]] || \
    die "REPORT_CONFIG SHA-256 mismatch: $config_sha256"
}

require_slurm_resources() {
  local job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
  [[ "${SLURM_CPUS_PER_TASK:-}" == 32 ]] || \
    die "Slurm CPUs per task must be exactly 32"
  [[ "${SLURM_MEM_PER_NODE:-}" == 131072 ]] || \
    die "Slurm memory must be exactly 128 GiB"
  [[ "${SLURM_JOB_ACCOUNT:-}" == pi-hadwigm ]] || \
    die "Slurm account must be pi-hadwigm"
  case "${SLURM_JOB_PARTITION:-}" in
    cpu|batch) ;;
    *) die "Slurm partition must be cpu or batch" ;;
  esac
  command -v scontrol >/dev/null || \
    die "scontrol is required to authenticate the Slurm allocation"
  local scontrol_record
  scontrol_record=$(scontrol show job -o "$job_id")
  [[ -n "$scontrol_record" ]] || die "scontrol returned an empty allocation"
  PTM_CLASS_SCORE_VIZ_SCONTROL_RECORD="$scontrol_record" python - "$job_id" <<'PY'
import os
import re
import sys

record = os.environ["PTM_CLASS_SCORE_VIZ_SCONTROL_RECORD"].strip()
assert "\n" not in record and "\r" not in record and record
fields = {}
for name in (
    "JobId",
    "Partition",
    "Account",
    "NumNodes",
    "NumCPUs",
    "TimeLimit",
    "Features",
    "AllocTRES",
    "ReqTRES",
    "TresPerNode",
    "Gres",
):
    matches = re.findall(rf"(?:^|\s){re.escape(name)}=([^\s]*)", record)
    assert len(matches) <= 1, f"duplicated scontrol field: {name}"
    if matches:
        fields[name] = matches[0]
required = {
    "JobId",
    "Partition",
    "Account",
    "NumNodes",
    "NumCPUs",
    "TimeLimit",
    "Features",
    "AllocTRES",
    "ReqTRES",
}
assert required <= set(fields), "scontrol allocation fields are incomplete"
assert fields["JobId"] == sys.argv[1]
assert fields["Partition"] in {"cpu", "batch"}
assert fields["Account"] == "pi-hadwigm"
assert fields["NumNodes"] == "1"
assert fields["NumCPUs"] == "32"
assert fields["TimeLimit"] == "12:00:00"
assert fields["Features"].casefold() == "rome"
surfaces = " ".join(
    fields.get(name, "") for name in ("AllocTRES", "ReqTRES", "TresPerNode", "Gres")
).casefold()
assert "gpu" not in surfaces, "GPU allocation/request is forbidden"
requested = fields["ReqTRES"].casefold()
assert "cpu=32" in requested
assert "mem=128g" in requested
assert "node=1" in requested
print(
    "slurm_allocation_authenticated="
    f"partition={fields['Partition']},account={fields['Account']},"
    f"nodes={fields['NumNodes']},cpus={fields['NumCPUs']},"
    f"memory=128G,features={fields['Features']},gpu=none"
)
PY
}

stage_gate
require_slurm_resources

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONOPTIMIZE
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
readonly JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
export CLASS_SCORE_VIZ_JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}/ptm_class_score_viz_${JOB_ID}_single"
mkdir -p "$CLASS_SCORE_VIZ_JOB_TMP_ROOT"
export TMPDIR="$CLASS_SCORE_VIZ_JOB_TMP_ROOT"
export TMP="$CLASS_SCORE_VIZ_JOB_TMP_ROOT"
export TEMP="$CLASS_SCORE_VIZ_JOB_TMP_ROOT"
export NUMBA_CACHE_DIR="$CLASS_SCORE_VIZ_JOB_TMP_ROOT/numba_cache"
export MPLCONFIGDIR="$CLASS_SCORE_VIZ_JOB_TMP_ROOT/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
python -c 'import os, pathlib, tempfile; assert __debug__, "Python assertions must remain enabled"; assert pathlib.Path(tempfile.gettempdir()).resolve() == pathlib.Path(os.environ["CLASS_SCORE_VIZ_JOB_TMP_ROOT"]).resolve(), "Python tempfile escaped the job-local root"'

python -m py_compile "$REPORTER" "$REPORT_TEST" "$WRAPPER_TEST" tests/test_all.py
python "$REPORT_TEST"
python "$WRAPPER_TEST"
python tests/test_all.py
stage_gate
[[ ! -e "$OUTPUT_DIR" ]] || die "immutable report output appeared during preflight: $OUTPUT_DIR"

echo "experiment=Other_ClassConditionalTemplateScoreVisualization_1.1"
echo "formal_confirmation=false"
echo "reporting_git_commit=$EXPECTED_COMMIT"
echo "report_config=$REPORT_CONFIG_PATH"
echo "report_config_sha256=$EXPECTED_REPORT_CONFIG_SHA256"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

/usr/bin/time -v python "$REPORTER" \
  --config "$REPORT_CONFIG_PATH" \
  --config-sha256 "$EXPECTED_REPORT_CONFIG_SHA256" \
  --output-root "$OUTPUT_DIR" \
  --expected-reporting-commit "$EXPECTED_COMMIT" \
  --device cpu

python - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$EXPECTED_REPORT_CONFIG_SHA256" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import canonical_json_sha256, sha256_file

root = Path(sys.argv[1]).resolve()
expected_commit = sys.argv[2]
expected_config_sha256 = sys.argv[3]

def self_hashed(name: str, field: str) -> dict:
    path = root / name
    assert path.is_file(), f"missing report artifact: {name}"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), name
    unsigned = dict(value)
    claimed = unsigned.pop(field)
    assert claimed == canonical_json_sha256(unsigned), name
    return value

visualization = self_hashed("visualization_manifest.json", "manifest_content_sha256")
result = self_hashed("result_manifest.json", "manifest_content_sha256")
completion = self_hashed("RUN_COMPLETE.json", "marker_content_sha256")
assert completion["schema"] == "pathline_template_matching.class_conditional_template_score_visualization_run_complete.v1"
assert result["schema"] == "pathline_template_matching.class_conditional_template_score_visualization_result.v1"
for payload in (visualization, result, completion):
    assert payload["experiment"] == "Other_ClassConditionalTemplateScoreVisualization_1.1"
    assert payload["reporting_git_commit"] == expected_commit
    assert payload["report_config_sha256"] == expected_config_sha256
assert completion["status"] == "complete_pending_local_pdf_collision_and_visual_QA"
assert result["status"] == "completed_pending_local_pdf_collision_and_visual_QA"
assert completion["figure_count"] == result["figure_count"] == visualization["figure_count"] == 8
assert sha256_file(root / "result_manifest.json") == completion["result_manifest_file_sha256"]
assert result["manifest_content_sha256"] == completion["result_manifest_content_sha256"]
assert sha256_file(root / "visualization_manifest.json") == result["visualization_manifest_file_sha256"]

entries = visualization["entries"]
assert isinstance(entries, list) and len(entries) == 8
assert len({(row["dataset"], row["scale_block"]) for row in entries}) == 8
expected_kinds = {
    "scene_npz",
    "scene_manifest",
    "png",
    "pdf",
    "svg",
    "alignment",
    "render_metadata",
}
for row in entries:
    exports = row["exports"]
    assert len(exports) == 7
    assert {item["export_kind"] for item in exports} == expected_kinds
    for item in exports:
        path = root / item["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == item["size_bytes"]
        assert sha256_file(path) == item["sha256"]
for item in result["artifacts"]:
    path = root / item["relative_path"]
    assert path.is_file()
    assert path.stat().st_size == item["size_bytes"]
    assert sha256_file(path) == item["sha256"]
print(f"evaluation_status={completion['status']}")
print(f"figure_count={completion['figure_count']}")
print(f"result_manifest_file_sha256={completion['result_manifest_file_sha256']}")
PY

stage_gate
echo "output_dir=$OUTPUT_DIR"
