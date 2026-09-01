from __future__ import annotations

import hashlib
import inspect
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = "/home/zhanx0o/pathline-template-matching-source-centered"
WRAPPER = (
    ROOT
    / "ibex"
    / "other_source_centered_paired_scale_template_visualization_1.1.sh"
)
COMMON = ROOT / "ibex" / "verify_source_centered_paired_scale_template_1.1_common.sh"


def _text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_source_centered_visualization_wrapper_freezes_cpu_rome_resources() -> None:
    text = _text()
    for directive in (
        "#SBATCH -N 1\n",
        "#SBATCH -J PTMSCPairedViz\n",
        "#SBATCH --time=12:00:00\n",
        "#SBATCH --cpus-per-task=32\n",
        "#SBATCH --mem=128G\n",
        "#SBATCH --partition=cpu\n",
        "#SBATCH --constraint=rome\n",
        "#SBATCH --account=pi-hadwigm\n",
    ):
        assert directive in text
    assert "#SBATCH --gpus" not in text
    assert "#SBATCH --gres=gpu" not in text
    assert 'case "${SLURM_JOB_PARTITION:-}" in\n    cpu|batch)' in text
    assert '[[ "${SLURM_CPUS_PER_TASK:-}" == 32 ]]' in text
    assert '[[ "${SLURM_MEM_PER_NODE:-}" == 131072 ]]' in text
    assert '[[ "${SLURM_JOB_ACCOUNT:-}" == pi-hadwigm ]]' in text
    assert 'scontrol show job -o "$job_id"' in text
    assert 'fields["Features"].casefold() == "rome"' in text
    assert 'fields["NumNodes"] == "1"' in text
    assert 'fields["NumCPUs"] == "32"' in text
    assert '"gpu" not in surfaces' in text
    assert '"cpu=32" in requested' in text
    assert '"mem=128g" in requested' in text
    assert '"node=1" in requested' in text


def test_source_centered_visualization_wrapper_uses_fixed_checkout_parent_and_config() -> None:
    text = _text()
    assert f"#SBATCH --chdir={PROJECT_ROOT}" in text
    assert f"readonly PROJECT_ROOT={PROJECT_ROOT}" in text
    assert (
        "readonly PARENT_ROOT=/ibex/user/zhanx0o/pathline-template-matching/"
        "Other_MainExp31FamilyHeldOutVisualization_1.1/runs/"
        "slurm_51029080_86be29698eb6"
    ) in text
    assert (
        "readonly PARENT_RESULT_SHA256="
        "57f03ba16ad8cfa0e1e0a9efd93f2dde7ae5866f173fad20055efb6939d4188e"
    ) in text
    assert (
        "readonly REPORT_CONFIG="
        "config/Other_SourceCenteredPairedScaleTemplateVisualization_1.1.yaml"
    ) in text
    assert (
        "readonly REPORT_CONFIG_SHA256="
        "c9c9a14b02fc3f47a4ee934ccd1091a7c7accefdbd28f569100605bf8230ca4e"
    ) in text
    config = ROOT / "config" / "Other_SourceCenteredPairedScaleTemplateVisualization_1.1.yaml"
    assert hashlib.sha256(config.read_bytes()).hexdigest() == (
        "c9c9a14b02fc3f47a4ee934ccd1091a7c7accefdbd28f569100605bf8230ca4e"
    )


def test_source_centered_visualization_wrapper_requires_all_runtime_identities() -> None:
    text = _text()
    required = (
        "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}",
        "${SOURCE_CENTERED_VIZ_RELEASE_ROOT:?SOURCE_CENTERED_VIZ_RELEASE_ROOT is required}",
        "${SOURCE_CENTERED_VIZ_RELEASE_COMPLETE_SHA256:?SOURCE_CENTERED_VIZ_RELEASE_COMPLETE_SHA256 is required}",
        "${SOURCE_CENTERED_VIZ_HALF_FOLD_ROOT:?SOURCE_CENTERED_VIZ_HALF_FOLD_ROOT is required}",
        "${SOURCE_CENTERED_VIZ_HALF_RUN_COMPLETE_SHA256:?SOURCE_CENTERED_VIZ_HALF_RUN_COMPLETE_SHA256 is required}",
        "${SOURCE_CENTERED_VIZ_BOEING_FOLD_ROOT:?SOURCE_CENTERED_VIZ_BOEING_FOLD_ROOT is required}",
        "${SOURCE_CENTERED_VIZ_BOEING_RUN_COMPLETE_SHA256:?SOURCE_CENTERED_VIZ_BOEING_RUN_COMPLETE_SHA256 is required}",
        "${SOURCE_CENTERED_VIZ_OUTPUT_ROOT:?SOURCE_CENTERED_VIZ_OUTPUT_ROOT is required}",
    )
    for value in required:
        assert value in text
    for name in (
        "RELEASE_COMPLETE_SHA256",
        "HALF_RUN_COMPLETE_SHA256",
        "BOEING_RUN_COMPLETE_SHA256",
    ):
        assert f'ptm_require_sha256 "${name}"' in text
    for name in (
        "RELEASE_ROOT_VALUE",
        "HALF_FOLD_ROOT_VALUE",
        "BOEING_FOLD_ROOT_VALUE",
        "OUTPUT_ROOT_VALUE",
    ):
        assert f'ptm_reject_confirmation_value "${name}"' in text
    assert '[[ "$runtime_path" == /* ]]' in text
    assert text.count("ptm_require_file_sha256") >= 5
    assert '"$RELEASE_ROOT/AGGREGATE_COMPLETE.json"' in text
    assert '"$HALF_FOLD_ROOT/RUN_COMPLETE.json"' in text
    assert '"$BOEING_FOLD_ROOT/RUN_COMPLETE.json"' in text


def test_source_centered_visualization_wrapper_authenticates_exact_clean_sources() -> None:
    text = _text()
    common = COMMON.read_text(encoding="utf-8")
    assert "git status --porcelain=v1 --untracked-files=all" in common
    assert "git rev-parse --verify HEAD^{commit}" in common
    assert 'git show "${expected_commit}:${source}"' in common
    assert "git merge-base --is-ancestor" in common
    stage = text.index('ptm_stage_gate "$WRAPPER"')
    preflight = text.index('python -m py_compile "$REPORTER"')
    reporter = text.index('/usr/bin/time -v python "$REPORTER"')
    final_stage = text.rindex('ptm_stage_unchanged "$WRAPPER"')
    assert stage < preflight < reporter < final_stage
    for source in (
        '"$REPORT_CONFIG"',
        "src/pathline_template_matching/source_centered_visualization.py",
        "src/pathline_template_matching/visualization.py",
        "src/pathline_template_matching/phase21_pipeline.py",
        "src/pathline_template_matching/portable_flow.py",
        "scripts/render_early_opposite_pair_kinematics_visualizations.py",
        '"$REPORTER"',
        '"$AUDITOR"',
        '"$REPORT_TEST"',
        '"$WRAPPER_TEST"',
        "tests/test_all.py",
    ):
        assert source in text
    assert 'python "$REPORT_TEST"' in text
    assert 'python "$WRAPPER_TEST"' in text
    assert "python tests/test_all.py" in text


def test_source_centered_visualization_wrapper_rejects_overwrite_twice() -> None:
    text = _text()
    first = text.index('[[ ! -e "$OUTPUT_DIR" ]]')
    preflight = text.index('python -m py_compile "$REPORTER"')
    second = text.index('[[ ! -e "$OUTPUT_DIR" ]]', first + 1)
    reporter = text.index('/usr/bin/time -v python "$REPORTER"')
    assert first < preflight < second < reporter
    assert "SOURCE_CENTERED_VIZ_OUTPUT_ROOT must be outside the committed checkout" in text


def test_source_centered_visualization_wrapper_uses_job_local_cpu_runtime() -> None:
    text = _text()
    assert "ptm_activate_runtime source_centered_visualization 32" in text
    assert 'export TMPDIR="$SOURCE_CENTERED_JOB_TMP_ROOT"' in text
    assert 'export TMP="$SOURCE_CENTERED_JOB_TMP_ROOT"' in text
    assert 'export TEMP="$SOURCE_CENTERED_JOB_TMP_ROOT"' in text
    assert 'export MPLCONFIGDIR="$SOURCE_CENTERED_JOB_TMP_ROOT/matplotlib"' in text
    assert "tempfile.gettempdir()" in text
    common = COMMON.read_text(encoding="utf-8")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f'export {name}="$threads"' in common
    assert "unset PYTHONOPTIMIZE" in common


def test_source_centered_visualization_wrapper_invokes_exact_production_contract() -> None:
    text = _text()
    release_auth = text.index('value["mode"] == "complete_five_fold_aggregate"')
    preflight = text.index('python -m py_compile "$REPORTER"')
    reporter = text.index('/usr/bin/time -v python "$REPORTER"')
    assert release_auth < preflight < reporter
    for argument in (
        '--parent-root "$PARENT_ROOT"',
        '--release-root "$RELEASE_ROOT"',
        '--half-fold-root "$HALF_FOLD_ROOT"',
        '--boeing-fold-root "$BOEING_FOLD_ROOT"',
        '--output-root "$OUTPUT_DIR"',
        '--expected-reporting-commit "$EXPECTED_COMMIT"',
    ):
        assert argument in text
    assert (
        'value["aggregator_git_commit"] == '
        '"a85c007ef961ce53bb40946ca3f38f033bf7a646"'
    ) in text
    assert (
        'value["fold_git_commit"] == '
        '"a85c007ef961ce53bb40946ca3f38f033bf7a646"'
    ) in text
    assert "--device" not in text


def test_source_centered_visualization_wrapper_postauths_every_machine_artifact() -> None:
    text = _text()
    for schema in (
        "pathline_template_matching.source_centered_paired_scale_visualization_result.v1",
        "pathline_template_matching.source_centered_paired_scale_visualization_run_complete.v1",
        "pathline_template_matching.source_centered_paired_scale_visualization.v1",
    ):
        assert schema in text
    assert 'result["status"] == "completed_pending_local_rendered_qa"' in text
    assert 'completion["status"] == "complete_pending_local_rendered_qa"' in text
    assert '== visualization["figure_count"] == 4' in text
    assert 'row["source_ordinal"] == 2' in text
    assert 'row["population"] == "combined_valid_unique_centers"' in text
    assert 'len(expected_artifacts) == 33' in text
    assert 'result["artifact_count"] == 33' in text
    assert 'len(final_paths) == 35' in text
    assert 'sha256_file(path) == row["sha256"]' in text
    assert 'not (root / "delivery_qa_summary.json").exists()' in text
    for kind in (
        "scene_npz",
        "scene_manifest",
        "png",
        "pdf",
        "svg",
        "alignment",
        "render_metadata",
    ):
        assert f'("{kind}",' in text
    for pending in (
        "svg_text_audit",
        "pdf_text_audit",
        "collision_audit",
        "collision_overlay_pdf",
    ):
        assert f'"{pending}"' in text


def _run_standalone() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
        and callable(value)
        and not inspect.signature(value).parameters
    ]
    assert len(tests) == 8
    for function in tests:
        function()


if __name__ == "__main__":
    _run_standalone()
