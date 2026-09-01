#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMSCPairedViz
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-source-centered
#SBATCH -o /home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-source-centered/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --constraint=rome
#SBATCH --account=pi-hadwigm

# ``cpu`` is the portable default. Submission may override it with
# ``sbatch --partition=batch``; the runtime accepts only these CPU partitions.

set -euo pipefail

readonly PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-source-centered
readonly WRAPPER=ibex/other_source_centered_paired_scale_template_visualization_1.1.sh
readonly COMMON=ibex/verify_source_centered_paired_scale_template_1.1_common.sh
readonly REPORTER=scripts/render_source_centered_paired_scale_template_visualizations.py
readonly AUDITOR=scripts/audit_source_centered_paired_scale_template_visualizations.py
readonly REPORT_TEST=tests/test_source_centered_visualization.py
readonly WRAPPER_TEST=tests/test_source_centered_visualization_ibex.py
readonly REPORT_CONFIG=config/Other_SourceCenteredPairedScaleTemplateVisualization_1.1.yaml
readonly REPORT_CONFIG_SHA256=c9c9a14b02fc3f47a4ee934ccd1091a7c7accefdbd28f569100605bf8230ca4e
readonly METHOD_CONFIG_SHA256=15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc
readonly TRUSTED_METHOD_COMMIT=a85c007ef961ce53bb40946ca3f38f033bf7a646
readonly PARENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Other_MainExp31FamilyHeldOutVisualization_1.1/runs/slurm_51029080_86be29698eb6
readonly PARENT_RESULT_SHA256=57f03ba16ad8cfa0e1e0a9efd93f2dde7ae5866f173fad20055efb6939d4188e

cd "$PROJECT_ROOT"
source "$COMMON"

readonly EXPECTED_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
readonly RELEASE_ROOT_VALUE=${SOURCE_CENTERED_VIZ_RELEASE_ROOT:?SOURCE_CENTERED_VIZ_RELEASE_ROOT is required}
readonly RELEASE_COMPLETE_SHA256=${SOURCE_CENTERED_VIZ_RELEASE_COMPLETE_SHA256:?SOURCE_CENTERED_VIZ_RELEASE_COMPLETE_SHA256 is required}
readonly HALF_FOLD_ROOT_VALUE=${SOURCE_CENTERED_VIZ_HALF_FOLD_ROOT:?SOURCE_CENTERED_VIZ_HALF_FOLD_ROOT is required}
readonly HALF_RUN_COMPLETE_SHA256=${SOURCE_CENTERED_VIZ_HALF_RUN_COMPLETE_SHA256:?SOURCE_CENTERED_VIZ_HALF_RUN_COMPLETE_SHA256 is required}
readonly BOEING_FOLD_ROOT_VALUE=${SOURCE_CENTERED_VIZ_BOEING_FOLD_ROOT:?SOURCE_CENTERED_VIZ_BOEING_FOLD_ROOT is required}
readonly BOEING_RUN_COMPLETE_SHA256=${SOURCE_CENTERED_VIZ_BOEING_RUN_COMPLETE_SHA256:?SOURCE_CENTERED_VIZ_BOEING_RUN_COMPLETE_SHA256 is required}
readonly OUTPUT_ROOT_VALUE=${SOURCE_CENTERED_VIZ_OUTPUT_ROOT:?SOURCE_CENTERED_VIZ_OUTPUT_ROOT is required}

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  ptm_die "EXPECTED_GIT_COMMIT must be lowercase 40-hex"
for runtime_path in \
  "$RELEASE_ROOT_VALUE" \
  "$HALF_FOLD_ROOT_VALUE" \
  "$BOEING_FOLD_ROOT_VALUE" \
  "$OUTPUT_ROOT_VALUE"; do
  [[ "$runtime_path" == /* ]] || ptm_die "all visualization runtime paths must be absolute"
done
ptm_require_sha256 "$RELEASE_COMPLETE_SHA256" SOURCE_CENTERED_VIZ_RELEASE_COMPLETE_SHA256
ptm_require_sha256 "$HALF_RUN_COMPLETE_SHA256" SOURCE_CENTERED_VIZ_HALF_RUN_COMPLETE_SHA256
ptm_require_sha256 "$BOEING_RUN_COMPLETE_SHA256" SOURCE_CENTERED_VIZ_BOEING_RUN_COMPLETE_SHA256

ptm_reject_confirmation_value "$RELEASE_ROOT_VALUE" SOURCE_CENTERED_VIZ_RELEASE_ROOT
ptm_reject_confirmation_value "$HALF_FOLD_ROOT_VALUE" SOURCE_CENTERED_VIZ_HALF_FOLD_ROOT
ptm_reject_confirmation_value "$BOEING_FOLD_ROOT_VALUE" SOURCE_CENTERED_VIZ_BOEING_FOLD_ROOT
ptm_reject_confirmation_value "$OUTPUT_ROOT_VALUE" SOURCE_CENTERED_VIZ_OUTPUT_ROOT
ptm_reject_confirmation_value "$PARENT_ROOT" PARENT_ROOT

readonly RELEASE_ROOT=$(realpath -- "$RELEASE_ROOT_VALUE")
readonly HALF_FOLD_ROOT=$(realpath -- "$HALF_FOLD_ROOT_VALUE")
readonly BOEING_FOLD_ROOT=$(realpath -- "$BOEING_FOLD_ROOT_VALUE")
readonly OUTPUT_DIR=$(realpath -m -- "$OUTPUT_ROOT_VALUE")
[[ -d "$RELEASE_ROOT" ]] || ptm_die "release root is missing"
[[ -d "$HALF_FOLD_ROOT" ]] || ptm_die "half-cylinder fold root is missing"
[[ -d "$BOEING_FOLD_ROOT" ]] || ptm_die "Boeing fold root is missing"
[[ "$OUTPUT_DIR" != "$PROJECT_ROOT" && "$OUTPUT_DIR" != "$PROJECT_ROOT/"* ]] || \
  ptm_die "SOURCE_CENTERED_VIZ_OUTPUT_ROOT must be outside the committed checkout"
[[ ! -e "$OUTPUT_DIR" ]] || ptm_die "immutable report output already exists: $OUTPUT_DIR"

ptm_require_file_sha256 \
  "$RELEASE_ROOT/AGGREGATE_COMPLETE.json" \
  "$RELEASE_COMPLETE_SHA256" \
  "complete-five release completion"
ptm_require_file_sha256 \
  "$HALF_FOLD_ROOT/RUN_COMPLETE.json" \
  "$HALF_RUN_COMPLETE_SHA256" \
  "half-cylinder fold completion"
ptm_require_file_sha256 \
  "$BOEING_FOLD_ROOT/RUN_COMPLETE.json" \
  "$BOEING_RUN_COMPLETE_SHA256" \
  "Boeing fold completion"
ptm_require_file_sha256 \
  "$PARENT_ROOT/result_manifest.json" \
  "$PARENT_RESULT_SHA256" \
  "frozen parent result manifest"
ptm_require_file_sha256 "$REPORT_CONFIG" "$REPORT_CONFIG_SHA256" "frozen report config"

readonly -a REPORTING_SOURCES=(
  "$REPORT_CONFIG"
  src/pathline_template_matching/source_centered_visualization.py
  src/pathline_template_matching/visualization.py
  src/pathline_template_matching/phase21_pipeline.py
  src/pathline_template_matching/portable_flow.py
  scripts/render_early_opposite_pair_kinematics_visualizations.py
  "$REPORTER"
  "$AUDITOR"
  "$REPORT_TEST"
  "$WRAPPER_TEST"
  tests/test_all.py
)

require_slurm_resources() {
  local job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
  [[ "${SLURM_JOB_NAME:-}" == PTMSCPairedViz ]] || ptm_die "Slurm job name changed"
  [[ "${SLURM_CPUS_PER_TASK:-}" == 32 ]] || ptm_die "Slurm CPUs per task must be exactly 32"
  [[ "${SLURM_MEM_PER_NODE:-}" == 131072 ]] || ptm_die "Slurm memory must be exactly 128 GiB"
  [[ "${SLURM_JOB_ACCOUNT:-}" == pi-hadwigm ]] || ptm_die "Slurm account must be pi-hadwigm"
  case "${SLURM_JOB_PARTITION:-}" in
    cpu|batch) ;;
    *) ptm_die "Slurm partition must be cpu or batch" ;;
  esac
  command -v scontrol >/dev/null || ptm_die "scontrol is required"
  local record
  record=$(scontrol show job -o "$job_id")
  [[ -n "$record" ]] || ptm_die "scontrol returned an empty allocation"
  PTM_SOURCE_CENTERED_VIZ_SCONTROL_RECORD="$record" python - "$job_id" <<'PY'
import os
import re
import sys

record = os.environ["PTM_SOURCE_CENTERED_VIZ_SCONTROL_RECORD"].strip()
assert record and "\n" not in record and "\r" not in record
fields = {}
for name in (
    "JobId", "Partition", "Account", "NumNodes", "NumCPUs", "TimeLimit",
    "Features", "AllocTRES", "ReqTRES", "TresPerNode", "Gres",
):
    matches = re.findall(rf"(?:^|\s){re.escape(name)}=([^\s]*)", record)
    assert len(matches) <= 1, f"duplicated scontrol field: {name}"
    if matches:
        fields[name] = matches[0]
required = {
    "JobId", "Partition", "Account", "NumNodes", "NumCPUs", "TimeLimit",
    "Features", "AllocTRES", "ReqTRES",
}
assert required <= set(fields)
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
assert "gpu" not in surfaces
requested = fields["ReqTRES"].casefold()
assert "cpu=32" in requested and "mem=128g" in requested and "node=1" in requested
print(
    "slurm_allocation_authenticated="
    f"job={fields['JobId']},partition={fields['Partition']},account={fields['Account']},"
    f"nodes={fields['NumNodes']},cpus={fields['NumCPUs']},memory=128G,"
    f"features={fields['Features']},gpu=none"
)
PY
}

ptm_stage_gate "$WRAPPER" "${REPORTING_SOURCES[@]}"
require_slurm_resources
ptm_activate_runtime source_centered_visualization 32
export TMPDIR="$SOURCE_CENTERED_JOB_TMP_ROOT"
export TMP="$SOURCE_CENTERED_JOB_TMP_ROOT"
export TEMP="$SOURCE_CENTERED_JOB_TMP_ROOT"
export MPLCONFIGDIR="$SOURCE_CENTERED_JOB_TMP_ROOT/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
python -c 'import os, pathlib, tempfile; assert __debug__; assert pathlib.Path(tempfile.gettempdir()).resolve() == pathlib.Path(os.environ["SOURCE_CENTERED_JOB_TMP_ROOT"]).resolve()'

# Authenticate the fixed complete-five release identity before tests or any NPZ member access.
python - "$RELEASE_ROOT/AGGREGATE_COMPLETE.json" <<'PY'
import json
import os
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import canonical_json_sha256

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
unsigned = dict(value)
claimed = unsigned.pop("content_sha256")
assert claimed == canonical_json_sha256(unsigned)
assert value["experiment"] == "Verify_SourceCenteredPairedScaleTemplate_1.1"
assert value["status"] == "completed"
assert value["mode"] == "complete_five_fold_aggregate"
assert value["config_sha256"] == "15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc"
assert value["aggregator_git_commit"] == "a85c007ef961ce53bb40946ca3f38f033bf7a646"
assert value["fold_git_commit"] == "a85c007ef961ce53bb40946ca3f38f033bf7a646"
assert value["source_centered_evidence"]["git_commit"] == value["fold_git_commit"]
PY

python -m py_compile "$REPORTER" "$AUDITOR" "$REPORT_TEST" "$WRAPPER_TEST" tests/test_all.py
python "$REPORT_TEST"
python "$WRAPPER_TEST"
python tests/test_all.py
ptm_stage_unchanged "$WRAPPER" "${REPORTING_SOURCES[@]}"
[[ ! -e "$OUTPUT_DIR" ]] || ptm_die "immutable report output appeared during preflight: $OUTPUT_DIR"

echo "experiment=Other_SourceCenteredPairedScaleTemplateVisualization_1.1"
echo "formal_confirmation=false"
echo "reporting_git_commit=$EXPECTED_COMMIT"
echo "method_git_commit=$TRUSTED_METHOD_COMMIT"
echo "report_config=$PROJECT_ROOT/$REPORT_CONFIG"
echo "report_config_sha256=$REPORT_CONFIG_SHA256"
echo "parent_root=$PARENT_ROOT"
echo "parent_result_sha256=$PARENT_RESULT_SHA256"
echo "release_root=$RELEASE_ROOT"
echo "release_complete_sha256=$RELEASE_COMPLETE_SHA256"
echo "half_fold_root=$HALF_FOLD_ROOT"
echo "half_RUN_COMPLETE_sha256=$HALF_RUN_COMPLETE_SHA256"
echo "boeing_fold_root=$BOEING_FOLD_ROOT"
echo "boeing_RUN_COMPLETE_sha256=$BOEING_RUN_COMPLETE_SHA256"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

/usr/bin/time -v python "$REPORTER" \
  --parent-root "$PARENT_ROOT" \
  --release-root "$RELEASE_ROOT" \
  --half-fold-root "$HALF_FOLD_ROOT" \
  --boeing-fold-root "$BOEING_FOLD_ROOT" \
  --output-root "$OUTPUT_DIR" \
  --expected-reporting-commit "$EXPECTED_COMMIT"

python - "$OUTPUT_DIR" "$EXPECTED_COMMIT" <<'PY'
import json
import os
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import canonical_json_sha256, sha256_file

root = Path(sys.argv[1]).resolve()
expected_commit = sys.argv[2]
experiment = "Other_SourceCenteredPairedScaleTemplateVisualization_1.1"
method_experiment = "Verify_SourceCenteredPairedScaleTemplate_1.1"
method_commit = "a85c007ef961ce53bb40946ca3f38f033bf7a646"
method_config = "15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc"
report_config = "c9c9a14b02fc3f47a4ee934ccd1091a7c7accefdbd28f569100605bf8230ca4e"

def self_hashed(name: str) -> dict:
    path = root / name
    assert path.is_file(), name
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256")
    assert claimed == canonical_json_sha256(unsigned), name
    return value

input_manifest = self_hashed("input_manifest.json")
self_hashed("figure_contract.json")
visualization = self_hashed("visualization_manifest.json")
result = self_hashed("result_manifest.json")
completion = self_hashed("RUN_COMPLETE.json")
assert result["schema"] == "pathline_template_matching.source_centered_paired_scale_visualization_result.v1"
assert completion["schema"] == "pathline_template_matching.source_centered_paired_scale_visualization_run_complete.v1"
assert visualization["schema"] == "pathline_template_matching.source_centered_paired_scale_visualization.v1"
for payload in (input_manifest, visualization, result, completion):
    assert payload["experiment"] == experiment
    assert payload["reporting_git_commit"] == expected_commit
    assert payload["frozen_report_config_sha256"] == report_config
    assert payload["method_experiment"] == method_experiment
    assert payload["method_config_sha256"] == method_config
    assert payload["method_fold_git_commit"] == method_commit
    releases = payload["method_release_authentication"]
    assert len(releases) == 1
    assert releases[0]["mode"] == "complete_five_fold_aggregate"
    assert releases[0]["aggregator_git_commit"] == method_commit
    assert releases[0]["fold_git_commit"] == method_commit
    assert len(releases[0]["source_centered_evidence_sha256"]) == 64
    runtime = payload["slurm_runtime"]
    assert runtime["job_id"] == str(os.environ["SLURM_JOB_ID"])
    assert runtime["job_name"] == "PTMSCPairedViz"
    assert runtime["partition"] in {"cpu", "batch"}
    assert runtime["account"] == "pi-hadwigm"
    assert runtime["cpus_per_task"] == 32
    assert runtime["memory_mib_per_node"] == 131072
    assert runtime["gpu"] == "none"
assert result["status"] == "completed_pending_local_rendered_qa"
assert completion["status"] == "complete_pending_local_rendered_qa"
assert result["figure_count"] == completion["figure_count"] == visualization["figure_count"] == 4
assert sha256_file(root / "result_manifest.json") == completion["result_manifest_file_sha256"]
assert result["content_sha256"] == completion["result_manifest_content_sha256"]
assert sha256_file(root / "visualization_manifest.json") == result["visualization_manifest_file_sha256"]

datasets = ("cylinder3d", "halfcylinderRe640", "halfcylinderRe6400", "boeing747")
entries = visualization["entries"]
assert tuple(row["dataset"] for row in entries) == datasets
expected_artifacts = {
    "frozen_config.yaml", "input_manifest.json", "figure_contract.json",
    "per_figure_metrics.csv", "visualization_manifest.json",
}
export_fields = (
    ("scene_npz", "scene_npz_sha256"),
    ("scene_manifest", "scene_manifest_sha256"),
    ("png", "png_sha256"),
    ("pdf", "pdf_sha256"),
    ("svg", "svg_sha256"),
    ("alignment", "alignment_sha256"),
    ("render_metadata", "render_metadata_sha256"),
)
for row in entries:
    assert row["source_ordinal"] == 2
    assert row["population"] == "combined_valid_unique_centers"
    assert set(row["pending_local_qa"]) == {
        "svg_text_audit", "pdf_text_audit", "collision_audit", "collision_overlay_pdf",
    }
    for relative in row["pending_local_qa"].values():
        assert not (root / relative).exists()
    for path_field, hash_field in export_fields:
        relative = row[path_field]
        assert relative not in expected_artifacts
        expected_artifacts.add(relative)
        path = root / relative
        assert path.is_file() and sha256_file(path) == row[hash_field]
assert len(expected_artifacts) == 33
assert result["artifact_count"] == 33
assert isinstance(result["artifacts"], list) and len(result["artifacts"]) == 33
assert {row["relative_path"] for row in result["artifacts"]} == expected_artifacts
for row in result["artifacts"]:
    path = root / row["relative_path"]
    assert path.is_file()
    assert path.stat().st_size == row["size_bytes"]
    assert sha256_file(path) == row["sha256"]
final_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
assert len(final_paths) == 35
assert final_paths == expected_artifacts | {"result_manifest.json", "RUN_COMPLETE.json"}
assert not (root / "delivery_qa_summary.json").exists()
print(f"evaluation_status={completion['status']}")
print(f"figure_count={completion['figure_count']}")
print(f"result_manifest_file_sha256={completion['result_manifest_file_sha256']}")
PY

ptm_stage_unchanged "$WRAPPER" "${REPORTING_SOURCES[@]}"
echo "output_dir=$OUTPUT_DIR"
