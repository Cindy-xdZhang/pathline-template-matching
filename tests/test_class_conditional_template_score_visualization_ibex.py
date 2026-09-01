from __future__ import annotations

import inspect
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    ROOT
    / "ibex"
    / "other_class_conditional_template_score_visualization_1.1.sh"
)


def _text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_class_score_visualization_wrapper_freezes_cpu_rome_resources() -> None:
    text = _text()
    for directive in (
        "#SBATCH -N 1\n",
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
    assert 'fields["NumCPUs"] == "32"' in text
    assert '"gpu" not in surfaces' in text


def test_class_score_visualization_wrapper_requires_four_external_identities() -> None:
    text = _text()
    required = (
        "${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}",
        "${REPORT_CONFIG:?REPORT_CONFIG is required}",
        "${REPORT_CONFIG_SHA256:?REPORT_CONFIG_SHA256 is required}",
        "${OUTPUT_ROOT:?OUTPUT_ROOT is required}",
    )
    for value in required:
        assert value in text
    assert 'require_lower_hex "$EXPECTED_COMMIT" 40 EXPECTED_GIT_COMMIT' in text
    assert (
        'require_lower_hex "$EXPECTED_REPORT_CONFIG_SHA256" 64 '
        "REPORT_CONFIG_SHA256"
    ) in text
    assert '[[ "$REPORT_CONFIG_VALUE" == /* ]]' in text
    assert '[[ "$OUTPUT_ROOT_VALUE" == /* ]]' in text
    assert "config/Other_ClassConditionalTemplateScoreVisualization_1.1.yaml" not in text
    assert not (
        ROOT
        / "config"
        / "Other_ClassConditionalTemplateScoreVisualization_1.1.yaml"
    ).exists()


def test_class_score_visualization_wrapper_authenticates_clean_exact_sources() -> None:
    text = _text()
    clean = text.index("git status --porcelain=v1 --untracked-files=all")
    commit = text.index("git rev-parse --verify HEAD^{commit}")
    committed_source = text.index('git show "${EXPECTED_COMMIT}:${source}"')
    config_hash = text.index('sha256sum "$REPORT_CONFIG_PATH"')
    preflight = text.index('python -m py_compile "$REPORTER"')
    reporter = text.index('/usr/bin/time -v python "$REPORTER"')
    assert clean < commit < committed_source < config_hash < preflight < reporter
    for source in (
        '"$REPORTER"',
        '"$REPORT_TEST"',
        '"$WRAPPER_TEST"',
        '"$WRAPPER"',
        "tests/test_all.py",
    ):
        assert source in text
    assert text.count("stage_gate") >= 4
    assert "python tests/test_all.py" in text
    assert 'python "$REPORT_TEST"' in text
    assert 'python "$WRAPPER_TEST"' in text


def test_class_score_visualization_wrapper_rejects_overwrite_before_any_run() -> None:
    text = _text()
    first_no_replace = text.index('[[ ! -e "$OUTPUT_DIR" ]]')
    preflight = text.index('python -m py_compile "$REPORTER"')
    second_no_replace = text.index(
        '[[ ! -e "$OUTPUT_DIR" ]]', first_no_replace + 1
    )
    reporter = text.index('/usr/bin/time -v python "$REPORTER"')
    assert first_no_replace < preflight < second_no_replace < reporter
    assert "OUTPUT_ROOT must be outside the committed checkout" in text
    assert "--output-root \"$OUTPUT_DIR\"" in text


def test_class_score_visualization_wrapper_invokes_frozen_reporter_contract() -> None:
    text = _text()
    reporter = text.index('/usr/bin/time -v python "$REPORTER"')
    completion = text.index('self_hashed("RUN_COMPLETE.json"', reporter)
    final_gate = text.rindex("stage_gate")
    final_echo = text.rindex('echo "output_dir=$OUTPUT_DIR"')
    assert reporter < completion < final_gate < final_echo
    for argument in (
        '--config "$REPORT_CONFIG_PATH"',
        '--config-sha256 "$EXPECTED_REPORT_CONFIG_SHA256"',
        '--output-root "$OUTPUT_DIR"',
        '--expected-reporting-commit "$EXPECTED_COMMIT"',
        "--device cpu",
    ):
        assert argument in text
    assert (
        "pathline_template_matching.class_conditional_template_score_"
        "visualization_run_complete.v2"
    ) in text
    assert (
        "pathline_template_matching.class_conditional_template_score_"
        "visualization_result.v2"
    ) in text
    assert 'completion["figure_count"] == result["figure_count"] == visualization["figure_count"] == 8' in text
    assert 'len({(row["dataset"], row["scale_block"]) for row in entries}) == 8' in text
    assert 'payload["source_release_count"] == 2' in text
    assert 'payload["complete_five_fold"] is False' in text
    assert 'row["evidence_source"]' in text
    for export_kind in (
        "scene_npz",
        "scene_manifest",
        "png",
        "pdf",
        "svg",
        "alignment",
        "render_metadata",
    ):
        assert f'"{export_kind}"' in text
    assert 'sha256_file(root / "result_manifest.json")' in text
    assert 'sha256_file(root / "visualization_manifest.json")' in text
    assert 'sha256_file(path) == item["sha256"]' in text


def test_class_score_visualization_wrapper_uses_job_local_cpu_runtime() -> None:
    text = _text()
    assert "conda activate deepvortex" in text
    assert 'export OPENBLAS_NUM_THREADS=32' in text
    assert 'export OMP_NUM_THREADS=32' in text
    assert 'export MKL_NUM_THREADS=32' in text
    assert 'export NUMEXPR_NUM_THREADS=32' in text
    assert 'export TMPDIR="$CLASS_SCORE_VIZ_JOB_TMP_ROOT"' in text
    assert 'export NUMBA_CACHE_DIR="$CLASS_SCORE_VIZ_JOB_TMP_ROOT/numba_cache"' in text
    assert 'export MPLCONFIGDIR="$CLASS_SCORE_VIZ_JOB_TMP_ROOT/matplotlib"' in text
    assert "unset PYTHONOPTIMIZE" in text


def _run_standalone() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
        and callable(value)
        and not inspect.signature(value).parameters
    ]
    assert len(tests) == 6
    for function in tests:
        function()


if __name__ == "__main__":
    _run_standalone()
