#!/bin/bash
#SBATCH -N 1
#SBATCH -J PTMSCRankViz
#SBATCH --chdir=/home/zhanx0o/pathline-template-matching-rank-likelihood
#SBATCH -o /home/zhanx0o/pathline-template-matching-rank-likelihood/slurm_logs/%x.%j.out
#SBATCH -e /home/zhanx0o/pathline-template-matching-rank-likelihood/slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --constraint=rome
#SBATCH --account=pi-hadwigm

set -euo pipefail

readonly PROJECT_ROOT=/home/zhanx0o/pathline-template-matching-rank-likelihood
readonly WRAPPER=ibex/other_source_centered_rank_likelihood_template_visualization_1.1.sh
readonly COMMON=ibex/verify_source_centered_rank_likelihood_template_1.1_common.sh
readonly REPORTER=scripts/render_source_centered_rank_likelihood_template_visualizations.py
readonly AUDITOR=scripts/audit_source_centered_rank_likelihood_template_visualizations.py
readonly REPORT_TEST=tests/test_source_centered_rank_likelihood_visualization.py
readonly WRAPPER_TEST=tests/test_source_centered_rank_likelihood_visualization_ibex.py
readonly REPORT_CONFIG=config/Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1.yaml
readonly REPORT_CONFIG_SHA256=a464761eb8df3ebf43d55b6f05eee2e90302be770b43f3e5e75a5944f13ff9a3
readonly METHOD_CONFIG_SHA256=41d6e7be70b898715c6df6f92cfb17176d2f1bb6153fa37b09dd4da9a6059ffa
readonly PARENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/Other_MainExp31FamilyHeldOutVisualization_1.1/runs/slurm_51029080_86be29698eb6
readonly PARENT_RESULT_SHA256=57f03ba16ad8cfa0e1e0a9efd93f2dde7ae5866f173fad20055efb6939d4188e

cd "$PROJECT_ROOT"
source "$COMMON"

readonly EXPECTED_COMMIT=${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}
readonly METHOD_COMMIT=${RANK_LIKELIHOOD_VIZ_METHOD_COMMIT:?RANK_LIKELIHOOD_VIZ_METHOD_COMMIT is required}
readonly RELEASE_ROOT_VALUE=${RANK_LIKELIHOOD_VIZ_RELEASE_ROOT:?RANK_LIKELIHOOD_VIZ_RELEASE_ROOT is required}
readonly RELEASE_COMPLETE_SHA256=${RANK_LIKELIHOOD_VIZ_RELEASE_COMPLETE_SHA256:?RANK_LIKELIHOOD_VIZ_RELEASE_COMPLETE_SHA256 is required}
readonly HALF_FOLD_ROOT_VALUE=${RANK_LIKELIHOOD_VIZ_HALF_FOLD_ROOT:?RANK_LIKELIHOOD_VIZ_HALF_FOLD_ROOT is required}
readonly HALF_RUN_COMPLETE_SHA256=${RANK_LIKELIHOOD_VIZ_HALF_RUN_COMPLETE_SHA256:?RANK_LIKELIHOOD_VIZ_HALF_RUN_COMPLETE_SHA256 is required}
readonly BOEING_FOLD_ROOT_VALUE=${RANK_LIKELIHOOD_VIZ_BOEING_FOLD_ROOT:?RANK_LIKELIHOOD_VIZ_BOEING_FOLD_ROOT is required}
readonly BOEING_RUN_COMPLETE_SHA256=${RANK_LIKELIHOOD_VIZ_BOEING_RUN_COMPLETE_SHA256:?RANK_LIKELIHOOD_VIZ_BOEING_RUN_COMPLETE_SHA256 is required}
readonly OUTPUT_ROOT_VALUE=${RANK_LIKELIHOOD_VIZ_OUTPUT_ROOT:?RANK_LIKELIHOOD_VIZ_OUTPUT_ROOT is required}

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || rank_die "EXPECTED_GIT_COMMIT must be lowercase 40-hex"
[[ "$METHOD_COMMIT" =~ ^[0-9a-f]{40}$ ]] || rank_die "RANK_LIKELIHOOD_VIZ_METHOD_COMMIT must be lowercase 40-hex"
for runtime_path in \
  "$RELEASE_ROOT_VALUE" \
  "$HALF_FOLD_ROOT_VALUE" \
  "$BOEING_FOLD_ROOT_VALUE" \
  "$OUTPUT_ROOT_VALUE"; do
  [[ "$runtime_path" == /* ]] || rank_die "all visualization runtime paths must be absolute"
done
rank_require_sha256 "$RELEASE_COMPLETE_SHA256" RANK_LIKELIHOOD_VIZ_RELEASE_COMPLETE_SHA256
rank_require_sha256 "$HALF_RUN_COMPLETE_SHA256" RANK_LIKELIHOOD_VIZ_HALF_RUN_COMPLETE_SHA256
rank_require_sha256 "$BOEING_RUN_COMPLETE_SHA256" RANK_LIKELIHOOD_VIZ_BOEING_RUN_COMPLETE_SHA256
for name in RELEASE_ROOT_VALUE HALF_FOLD_ROOT_VALUE BOEING_FOLD_ROOT_VALUE OUTPUT_ROOT_VALUE; do
  rank_reject_confirmation_value "${!name}" "$name"
done
rank_reject_confirmation_value "$PARENT_ROOT" PARENT_ROOT

readonly RELEASE_ROOT=$(realpath -- "$RELEASE_ROOT_VALUE")
readonly HALF_FOLD_ROOT=$(realpath -- "$HALF_FOLD_ROOT_VALUE")
readonly BOEING_FOLD_ROOT=$(realpath -- "$BOEING_FOLD_ROOT_VALUE")
readonly OUTPUT_DIR=$(realpath -m -- "$OUTPUT_ROOT_VALUE")
[[ -d "$RELEASE_ROOT" ]] || rank_die "release root is missing"
[[ -d "$HALF_FOLD_ROOT" ]] || rank_die "half-cylinder fold root is missing"
[[ -d "$BOEING_FOLD_ROOT" ]] || rank_die "Boeing fold root is missing"
[[ "$OUTPUT_DIR" != "$PROJECT_ROOT" && "$OUTPUT_DIR" != "$PROJECT_ROOT/"* ]] || \
  rank_die "RANK_LIKELIHOOD_VIZ_OUTPUT_ROOT must be outside the committed checkout"
[[ ! -e "$OUTPUT_DIR" ]] || rank_die "immutable report output already exists: $OUTPUT_DIR"
python - "$OUTPUT_DIR" "$RELEASE_ROOT" "$HALF_FOLD_ROOT" "$BOEING_FOLD_ROOT" "$PARENT_ROOT" <<'PY'
from pathlib import Path
import sys

output = Path(sys.argv[1]).resolve()
for value in sys.argv[2:]:
    source = Path(value).resolve()
    assert output != source
    assert source not in output.parents
    assert output not in source.parents
PY

rank_require_file_sha256 "$RELEASE_ROOT/AGGREGATE_COMPLETE.json" "$RELEASE_COMPLETE_SHA256" "complete-five release completion"
rank_require_file_sha256 "$HALF_FOLD_ROOT/RUN_COMPLETE.json" "$HALF_RUN_COMPLETE_SHA256" "half-cylinder fold completion"
rank_require_file_sha256 "$BOEING_FOLD_ROOT/RUN_COMPLETE.json" "$BOEING_RUN_COMPLETE_SHA256" "Boeing fold completion"
rank_require_file_sha256 "$PARENT_ROOT/result_manifest.json" "$PARENT_RESULT_SHA256" "frozen parent result manifest"
rank_require_file_sha256 "$REPORT_CONFIG" "$REPORT_CONFIG_SHA256" "frozen report config"

readonly -a REPORTING_SOURCES=(
  "$REPORT_CONFIG"
  src/pathline_template_matching/source_centered_rank_likelihood_visualization.py
  src/pathline_template_matching/source_centered_visualization.py
  src/pathline_template_matching/visualization.py
  src/pathline_template_matching/phase21_pipeline.py
  src/pathline_template_matching/phase21_visualization.py
  src/pathline_template_matching/negative_tail_visualization.py
  src/pathline_template_matching/metrics.py
  src/pathline_template_matching/portable_flow.py
  scripts/render_early_opposite_pair_kinematics_visualizations.py
  "$REPORTER"
  "$AUDITOR"
  "$REPORT_TEST"
  "$WRAPPER_TEST"
  tests/test_all.py
)
readonly -a METHOD_INTERPRETATION_SOURCES=(
  config/Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml
  config/Verify_SourceCenteredPairedScaleTemplate_1.1.yaml
  scripts/run_verify_source_centered_rank_likelihood_template_1_1.py
  scripts/aggregate_verify_source_centered_rank_likelihood_template_1_1.py
  scripts/run_verify_source_centered_paired_scale_template_1_1.py
  scripts/run_verify_early_opposite_pair_kinematics_1_1.py
  scripts/run_verify_scale_conditioned_retrieval_1_1.py
  src/pathline_template_matching/paired_scale_center_fusion.py
  src/pathline_template_matching/per_scale_negative_metric.py
  src/pathline_template_matching/nested_scale_validation.py
  src/pathline_template_matching/portable_flow.py
  src/pathline_template_matching/source_centered_sidecar.py
  src/pathline_template_matching/source_centered_rank_likelihood.py
)

require_slurm_resources() {
  local job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
  [[ "${SLURM_JOB_NAME:-}" == PTMSCRankViz ]] || rank_die "Slurm job name changed"
  [[ "${SLURM_CPUS_PER_TASK:-}" == 32 ]] || rank_die "Slurm CPUs must be exactly 32"
  [[ "${SLURM_MEM_PER_NODE:-}" == 131072 ]] || rank_die "Slurm memory must be 128 GiB"
  [[ "${SLURM_JOB_ACCOUNT:-}" == pi-hadwigm ]] || rank_die "Slurm account changed"
  case "${SLURM_JOB_PARTITION:-}" in cpu|batch) ;; *) rank_die "Slurm partition must be cpu or batch" ;; esac
  command -v scontrol >/dev/null || rank_die "scontrol is required"
  local record
  record=$(scontrol show job -o "$job_id")
  [[ -n "$record" ]] || rank_die "scontrol returned an empty allocation"
  RANK_VIZ_SCONTROL_RECORD="$record" python - "$job_id" <<'PY'
import os
import re
import sys

record = os.environ["RANK_VIZ_SCONTROL_RECORD"].strip()
assert record and "\n" not in record and "\r" not in record
fields = {}
for name in (
    "JobId", "Partition", "Account", "NumNodes", "NumCPUs", "TimeLimit",
    "Features", "AllocTRES", "ReqTRES", "TresPerNode", "Gres",
):
    matches = re.findall(rf"(?:^|\s){re.escape(name)}=([^\s]*)", record)
    assert len(matches) <= 1
    if matches:
        fields[name] = matches[0]
required = {"JobId", "Partition", "Account", "NumNodes", "NumCPUs", "TimeLimit", "Features", "AllocTRES", "ReqTRES"}
assert required <= set(fields)
assert fields["JobId"] == sys.argv[1]
assert fields["Partition"] in {"cpu", "batch"}
assert fields["Account"] == "pi-hadwigm"
assert fields["NumNodes"] == "1"
assert fields["NumCPUs"] == "32"
assert fields["TimeLimit"] == "12:00:00"
assert fields["Features"].casefold() == "rome"
surfaces = " ".join(fields.get(name, "") for name in ("AllocTRES", "ReqTRES", "TresPerNode", "Gres")).casefold()
assert "gpu" not in surfaces
requested = fields["ReqTRES"].casefold()
assert "cpu=32" in requested and "mem=128g" in requested and "node=1" in requested
PY
}

rank_stage_gate "$WRAPPER" "${REPORTING_SOURCES[@]}" "${METHOD_INTERPRETATION_SOURCES[@]}"
require_slurm_resources
git cat-file -e "${METHOD_COMMIT}^{commit}" || rank_die "method commit is unavailable"
if ! git diff --quiet "$METHOD_COMMIT" "$EXPECTED_COMMIT" -- \
  "${METHOD_INTERPRETATION_SOURCES[@]}"; then
  rank_die "method interpretation sources differ between method and reporting commits"
fi
rank_activate_runtime source_centered_rank_likelihood_visualization 32
export TMPDIR="$RANK_JOB_TMP_ROOT"
export TMP="$RANK_JOB_TMP_ROOT"
export TEMP="$RANK_JOB_TMP_ROOT"
export MPLCONFIGDIR="$RANK_JOB_TMP_ROOT/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
python -c 'import os, pathlib, tempfile; assert __debug__; assert pathlib.Path(tempfile.gettempdir()).resolve() == pathlib.Path(os.environ["RANK_JOB_TMP_ROOT"]).resolve()'

# Opaque aggregate identity gate: no fold NPZ member is opened here.
python - "$RELEASE_ROOT/AGGREGATE_COMPLETE.json" "$METHOD_COMMIT" <<'PY'
import json
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import canonical_json_sha256

path = Path(sys.argv[1])
commit = sys.argv[2]
value = json.loads(path.read_text(encoding="utf-8"))
unsigned = dict(value)
claimed = unsigned.pop("content_sha256")
assert claimed == canonical_json_sha256(unsigned)
assert value["experiment"] == "Verify_SourceCenteredRankLikelihoodTemplate_1.1"
assert value["status"] == "completed"
assert value["mode"] == "complete_five_fold_aggregate"
assert value["config_sha256"] == "41d6e7be70b898715c6df6f92cfb17176d2f1bb6153fa37b09dd4da9a6059ffa"
assert value["aggregator_git_commit"] == commit
assert value["fold_git_commit"] == commit
PY

python -m py_compile "$REPORTER" "$AUDITOR" "$REPORT_TEST" "$WRAPPER_TEST" tests/test_all.py
python "$REPORT_TEST"
python "$WRAPPER_TEST"
python tests/test_all.py
rank_stage_unchanged "$WRAPPER" "${REPORTING_SOURCES[@]}" "${METHOD_INTERPRETATION_SOURCES[@]}"
[[ ! -e "$OUTPUT_DIR" ]] || rank_die "immutable report output appeared during preflight"

echo "experiment=Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1"
echo "formal_confirmation=false"
echo "plotted_arm=dual_histogram_llr"
echo "reporting_git_commit=$EXPECTED_COMMIT"
echo "method_git_commit=$METHOD_COMMIT"
echo "report_config_sha256=$REPORT_CONFIG_SHA256"
echo "parent_root=$PARENT_ROOT"
echo "release_root=$RELEASE_ROOT"
echo "half_fold_root=$HALF_FOLD_ROOT"
echo "boeing_fold_root=$BOEING_FOLD_ROOT"
echo "output_dir=$OUTPUT_DIR"
hostname
lscpu

/usr/bin/time -v python "$REPORTER" \
  --parent-root "$PARENT_ROOT" \
  --release-root "$RELEASE_ROOT" \
  --half-fold-root "$HALF_FOLD_ROOT" \
  --boeing-fold-root "$BOEING_FOLD_ROOT" \
  --output-root "$OUTPUT_DIR" \
  --expected-reporting-commit "$EXPECTED_COMMIT" \
  --expected-method-commit "$METHOD_COMMIT"

python - "$OUTPUT_DIR" "$EXPECTED_COMMIT" "$METHOD_COMMIT" <<'PY'
import json
import os
from pathlib import Path
import sys

from pathline_template_matching.portable_flow import canonical_json_sha256, sha256_file

root = Path(sys.argv[1]).resolve()
reporting_commit = sys.argv[2]
method_commit = sys.argv[3]

def self_hashed(name):
    path = root / name
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
assert result["schema"] == "pathline_template_matching.source_centered_rank_likelihood_visualization_result.v1"
assert completion["schema"] == "pathline_template_matching.source_centered_rank_likelihood_visualization_run_complete.v1"
assert visualization["schema"] == "pathline_template_matching.source_centered_rank_likelihood_visualization.v1"
for payload in (input_manifest, visualization, result, completion):
    assert payload["experiment"] == "Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1"
    assert payload["reporting_git_commit"] == reporting_commit
    assert payload["frozen_report_config_sha256"] == "a464761eb8df3ebf43d55b6f05eee2e90302be770b43f3e5e75a5944f13ff9a3"
    assert payload["method_experiment"] == "Verify_SourceCenteredRankLikelihoodTemplate_1.1"
    assert payload["method_config_sha256"] == "41d6e7be70b898715c6df6f92cfb17176d2f1bb6153fa37b09dd4da9a6059ffa"
    assert payload["method_fold_git_commit"] == method_commit
    assert payload["method_release_authentication"][0]["mode"] == "complete_five_fold_aggregate"
assert input_manifest["all_18_files_authenticated_per_required_fold"] is True
assert input_manifest["fold_sidecar_or_label_member_access"] is False
assert input_manifest["method_interpretation_git_commit"] == method_commit
assert len(input_manifest["method_interpretation_git_blob_sha1"]) == 13
assert result["status"] == "completed_pending_local_rendered_qa"
assert completion["status"] == "complete_pending_local_rendered_qa"
assert result["plotted_arm"] == completion["plotted_arm"] == visualization["plotted_arm"] == "dual_histogram_llr"
assert result["table_only_control_arms"] == visualization["table_only_control_arms"] == ["negative_ecdf", "direct_rank_mean_top5"]
assert result["table_only_control_metrics_reported"] is visualization["table_only_control_metrics_reported"] is True
assert result["figure_count"] == completion["figure_count"] == visualization["figure_count"] == 4
assert sha256_file(root / "result_manifest.json") == completion["result_manifest_file_sha256"]
assert sha256_file(root / "visualization_manifest.json") == result["visualization_manifest_file_sha256"]
entries = visualization["entries"]
datasets = ("cylinder3d", "halfcylinderRe640", "halfcylinderRe6400", "boeing747")
assert tuple(row["dataset"] for row in entries) == datasets
expected_artifacts = {"frozen_config.yaml", "input_manifest.json", "figure_contract.json", "per_figure_metrics.csv", "visualization_manifest.json"}
for row in entries:
    assert row["source_ordinal"] == 2
    assert row["population"] == "combined_valid_unique_centers"
    assert row["plotted_arm"] == "dual_histogram_llr"
    assert row["table_only_control_arms"] == ["negative_ecdf", "direct_rank_mean_top5"]
    for path_field, hash_field in (
        ("scene_npz", "scene_npz_sha256"), ("scene_manifest", "scene_manifest_sha256"),
        ("png", "png_sha256"), ("pdf", "pdf_sha256"), ("svg", "svg_sha256"),
        ("alignment", "alignment_sha256"), ("render_metadata", "render_metadata_sha256"),
    ):
        relative = row[path_field]
        expected_artifacts.add(relative)
        assert (root / relative).is_file()
        assert sha256_file(root / relative) == row[hash_field]
assert len(expected_artifacts) == 33
assert result["artifact_count"] == 33 and len(result["artifacts"]) == 33
assert {row["relative_path"] for row in result["artifacts"]} == expected_artifacts
final_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
assert len(final_paths) == 35
assert final_paths == expected_artifacts | {"result_manifest.json", "RUN_COMPLETE.json"}
assert not (root / "delivery_qa_summary.json").exists()
print(f"evaluation_status={completion['status']}")
PY

rank_stage_unchanged "$WRAPPER" "${REPORTING_SOURCES[@]}" "${METHOD_INTERPRETATION_SOURCES[@]}"
echo "output_dir=$OUTPUT_DIR"
