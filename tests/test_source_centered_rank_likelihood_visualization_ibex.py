from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "ibex" / "other_source_centered_rank_likelihood_template_visualization_1.1.sh"
COMMON = ROOT / "ibex" / "verify_source_centered_rank_likelihood_template_1.1_common.sh"
PROJECT_ROOT = "/home/zhanx0o/pathline-template-matching-rank-likelihood"

for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import (  # noqa: E402
    render_source_centered_rank_likelihood_template_visualizations as reporter_module,
)


def _text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_rank_likelihood_visualization_wrapper_freezes_cpu_rome_resources() -> None:
    text = _text()
    for directive in (
        "#SBATCH -N 1\n",
        "#SBATCH -J PTMSCRankViz\n",
        "#SBATCH --time=12:00:00\n",
        "#SBATCH --cpus-per-task=32\n",
        "#SBATCH --mem=128G\n",
        "#SBATCH --partition=cpu\n",
        "#SBATCH --constraint=rome\n",
        "#SBATCH --account=pi-hadwigm\n",
    ):
        assert directive in text
    assert "#SBATCH --gpus" not in text and "#SBATCH --gres=gpu" not in text
    assert '[[ "${SLURM_CPUS_PER_TASK:-}" == 32 ]]' in text
    assert '[[ "${SLURM_MEM_PER_NODE:-}" == 131072 ]]' in text
    assert 'scontrol show job -o "$job_id"' in text
    assert 'fields["Features"].casefold() == "rome"' in text
    assert '"gpu" not in surfaces' in text


def test_rank_likelihood_visualization_wrapper_pins_config_parent_and_checkout() -> None:
    text = _text()
    assert f"#SBATCH --chdir={PROJECT_ROOT}" in text
    assert f"readonly PROJECT_ROOT={PROJECT_ROOT}" in text
    assert "Other_MainExp31FamilyHeldOutVisualization_1.1/runs/slurm_51029080_86be29698eb6" in text
    assert "PARENT_RESULT_SHA256=57f03ba16ad8cfa0e1e0a9efd93f2dde7ae5866f173fad20055efb6939d4188e" in text
    assert "REPORT_CONFIG=config/Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1.yaml" in text
    expected = "a464761eb8df3ebf43d55b6f05eee2e90302be770b43f3e5e75a5944f13ff9a3"
    assert f"REPORT_CONFIG_SHA256={expected}" in text
    assert hashlib.sha256((ROOT / "config" / "Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1.yaml").read_bytes()).hexdigest() == expected


def test_rank_likelihood_visualization_wrapper_requires_complete_runtime_identities() -> None:
    text = _text()
    for value in (
        "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}",
        "${RANK_LIKELIHOOD_VIZ_METHOD_COMMIT:?RANK_LIKELIHOOD_VIZ_METHOD_COMMIT is required}",
        "${RANK_LIKELIHOOD_VIZ_RELEASE_ROOT:?RANK_LIKELIHOOD_VIZ_RELEASE_ROOT is required}",
        "${RANK_LIKELIHOOD_VIZ_RELEASE_COMPLETE_SHA256:?RANK_LIKELIHOOD_VIZ_RELEASE_COMPLETE_SHA256 is required}",
        "${RANK_LIKELIHOOD_VIZ_HALF_FOLD_ROOT:?RANK_LIKELIHOOD_VIZ_HALF_FOLD_ROOT is required}",
        "${RANK_LIKELIHOOD_VIZ_HALF_RUN_COMPLETE_SHA256:?RANK_LIKELIHOOD_VIZ_HALF_RUN_COMPLETE_SHA256 is required}",
        "${RANK_LIKELIHOOD_VIZ_BOEING_FOLD_ROOT:?RANK_LIKELIHOOD_VIZ_BOEING_FOLD_ROOT is required}",
        "${RANK_LIKELIHOOD_VIZ_BOEING_RUN_COMPLETE_SHA256:?RANK_LIKELIHOOD_VIZ_BOEING_RUN_COMPLETE_SHA256 is required}",
        "${RANK_LIKELIHOOD_VIZ_OUTPUT_ROOT:?RANK_LIKELIHOOD_VIZ_OUTPUT_ROOT is required}",
    ):
        assert value in text
    assert text.count("rank_require_file_sha256") >= 5
    assert '"$RELEASE_ROOT/AGGREGATE_COMPLETE.json"' in text
    assert '"$HALF_FOLD_ROOT/RUN_COMPLETE.json"' in text
    assert '"$BOEING_FOLD_ROOT/RUN_COMPLETE.json"' in text


def test_rank_likelihood_visualization_wrapper_authenticates_complete_five_before_tests() -> None:
    text = _text()
    release_gate = text.index('value["mode"] == "complete_five_fold_aggregate"')
    compile_gate = text.index('python -m py_compile "$REPORTER"')
    reporter = text.index('/usr/bin/time -v python "$REPORTER"')
    assert release_gate < compile_gate < reporter
    assert 'value["experiment"] == "Verify_SourceCenteredRankLikelihoodTemplate_1.1"' in text
    assert 'value["config_sha256"] == "41d6e7be70b898715c6df6f92cfb17176d2f1bb6153fa37b09dd4da9a6059ffa"' in text
    assert 'value["aggregator_git_commit"] == commit' in text
    assert 'value["fold_git_commit"] == commit' in text
    assert 'python "$REPORT_TEST"' in text
    assert 'python "$WRAPPER_TEST"' in text
    assert "python tests/test_all.py" in text
    assert 'git cat-file -e "${METHOD_COMMIT}^{commit}"' in text
    assert 'git diff --quiet "$METHOD_COMMIT" "$EXPECTED_COMMIT"' in text
    assert "method interpretation sources differ" in text
    assert all(
        path in text
        for path in reporter_module.METHOD_INTERPRETATION_RELATIVE_PATHS
    )


def test_rank_likelihood_visualization_wrapper_stages_all_reporting_sources_and_refuses_overwrite() -> None:
    text = _text()
    common = COMMON.read_text(encoding="utf-8")
    assert "git status --porcelain=v1 --untracked-files=all" in common
    assert "git merge-base --is-ancestor" in common
    for source in (
        '"$REPORT_CONFIG"',
        "src/pathline_template_matching/source_centered_rank_likelihood_visualization.py",
        "src/pathline_template_matching/source_centered_visualization.py",
        "src/pathline_template_matching/visualization.py",
        "src/pathline_template_matching/phase21_visualization.py",
        "src/pathline_template_matching/negative_tail_visualization.py",
        "src/pathline_template_matching/metrics.py",
        "scripts/render_early_opposite_pair_kinematics_visualizations.py",
        '"$REPORTER"',
        '"$AUDITOR"',
        '"$REPORT_TEST"',
        '"$WRAPPER_TEST"',
        "tests/test_all.py",
    ):
        assert source in text
    first = text.index('[[ ! -e "$OUTPUT_DIR" ]]')
    compile_gate = text.index('python -m py_compile "$REPORTER"')
    second = text.index('[[ ! -e "$OUTPUT_DIR" ]]', first + 1)
    reporter = text.index('/usr/bin/time -v python "$REPORTER"')
    assert first < compile_gate < second < reporter
    assert 'assert source not in output.parents' in text
    assert 'assert output not in source.parents' in text


def test_rank_likelihood_visualization_wrapper_invokes_primary_only_report_contract() -> None:
    text = _text()
    for argument in (
        '--parent-root "$PARENT_ROOT"',
        '--release-root "$RELEASE_ROOT"',
        '--half-fold-root "$HALF_FOLD_ROOT"',
        '--boeing-fold-root "$BOEING_FOLD_ROOT"',
        '--output-root "$OUTPUT_DIR"',
        '--expected-reporting-commit "$EXPECTED_COMMIT"',
        '--expected-method-commit "$METHOD_COMMIT"',
    ):
        assert argument in text
    assert 'echo "plotted_arm=dual_histogram_llr"' in text
    assert 'visualization["plotted_arm"] == "dual_histogram_llr"' in text
    assert 'input_manifest["all_18_files_authenticated_per_required_fold"] is True' in text
    assert 'input_manifest["fold_sidecar_or_label_member_access"] is False' in text
    assert 'input_manifest["method_interpretation_git_commit"] == method_commit' in text


def test_rank_likelihood_visualization_wrapper_postauths_exact_35_machine_files() -> None:
    text = _text()
    for schema in (
        "pathline_template_matching.source_centered_rank_likelihood_visualization_result.v1",
        "pathline_template_matching.source_centered_rank_likelihood_visualization_run_complete.v1",
        "pathline_template_matching.source_centered_rank_likelihood_visualization.v1",
    ):
        assert schema in text
    assert 'result["status"] == "completed_pending_local_rendered_qa"' in text
    assert 'completion["status"] == "complete_pending_local_rendered_qa"' in text
    assert 'len(expected_artifacts) == 33' in text
    assert 'result["artifact_count"] == 33' in text
    assert 'len(final_paths) == 35' in text
    assert 'not (root / "delivery_qa_summary.json").exists()' in text


def _run_standalone() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
        and callable(value)
        and not inspect.signature(value).parameters
    ]
    assert len(tests) == 7
    for function in tests:
        function()


if __name__ == "__main__":
    _run_standalone()
