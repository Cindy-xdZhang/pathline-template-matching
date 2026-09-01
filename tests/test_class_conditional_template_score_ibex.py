from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IBEX = ROOT / "ibex"


def _text(name: str) -> str:
    return (IBEX / name).read_text(encoding="utf-8")


def test_class_conditional_ibex_runtime_resources_apply_documented_scheduler_overrides() -> None:
    config_text = (
        ROOT / "config" / "Verify_ClassConditionalTemplateScore_1.1.yaml"
    ).read_text(encoding="utf-8")
    assert config_text.count("account: deepvortex") == 2
    assert "account: pi-hadwigm" not in config_text
    wrappers = (
        "verify_class_conditional_template_score_1.1_resource_smoke.sh",
        "verify_class_conditional_template_score_1.1_first_fold.sh",
        "verify_class_conditional_template_score_1.1_first_fold_auth.sh",
        "verify_class_conditional_template_score_1.1_remaining_folds.sh",
        "verify_class_conditional_template_score_1.1_aggregate_five.sh",
    )
    for name in wrappers:
        text = _text(name)
        assert "#SBATCH -N 1\n" in text
        assert "#SBATCH --partition=cpu" in text
        assert "#SBATCH --constraint=rome" in text
        assert "#SBATCH --account=pi-hadwigm" in text
        assert "#SBATCH --cpus-per-task=32" in text
        assert "#SBATCH --mem=128G" in text
        assert "cpu_amd_epyc_7702" not in text
    assert "#SBATCH --time=04:00:00" in _text(wrappers[0])
    smoke = _text(wrappers[0])
    assert 'scontrol show job -o "$RESOURCE_SMOKE_JOB_ID"' in smoke
    assert "PTM_SLURM_SCONTROL_JOB_RECORD" in smoke
    for name in wrappers[1:]:
        assert "#SBATCH --time=12:00:00" in _text(name)
    common = _text("verify_class_conditional_template_score_1.1_common.sh")
    assert "readonly CLASS_RUNTIME_SLURM_ACCOUNT=pi-hadwigm" in common
    assert "readonly CLASS_RUNTIME_SLURM_PARTITION=batch" in common
    assert '[[ "${SLURM_JOB_ACCOUNT:-}" == "$CLASS_RUNTIME_SLURM_ACCOUNT" ]]' in common
    assert '[[ "${SLURM_JOB_PARTITION:-}" == "$CLASS_RUNTIME_SLURM_PARTITION" ]]' in common
    assert "conda activate deepvortex" in common
    assert "conda activate deepvortex" in smoke
    assert 'scontrol show job -o "$job_id"' in common
    assert 'expected_time_limit="12:00:00"' in common
    assert "allocation['num_nodes']" in common
    assert "allocation['gpu_allocation']" in common


def test_common_slurm_constraint_gate_matches_the_resource_smoke_token_rule() -> None:
    common = _text("verify_class_conditional_template_score_1.1_common.sh")
    assert "SLURM_JOB_CONSTRAINTS" not in common
    assert "_validated_scontrol_allocation" in common
    assert "allocation['features']" in common
    assert "features=" in common


def test_first_fold_is_gated_by_the_authenticated_resource_smoke() -> None:
    text = _text("verify_class_conditional_template_score_1.1_first_fold.sh")
    smoke = text.index("ptm_class_require_resource_smoke")
    runner = text.index('/usr/bin/time -v python "$CLASS_RUNNER"')
    assert smoke < runner
    assert "RESOURCE_SMOKE_PASS" in text
    assert "RESOURCE_SMOKE_PASS_SHA256" in text
    common = _text("verify_class_conditional_template_score_1.1_common.sh")
    assert "authenticate_resource_smoke_release" in common
    assert "ptm_class_require_file_sha256" in common


def test_common_smoke_gate_consumes_the_public_authenticator_return_schema() -> None:
    common = _text("verify_class_conditional_template_score_1.1_common.sh")
    assert "result['marker']['sha256']" in common
    assert "result['pass_marker_sha256']" not in common


def test_remaining_folds_require_a_reconstructed_no_stop_certificate() -> None:
    text = _text("verify_class_conditional_template_score_1.1_remaining_folds.sh")
    smoke = text.index('ptm_class_require_resource_smoke "$SMOKE_PASS"')
    authenticate = text.index("authenticate_single_fold_release")
    no_stop = text.index('[[ "$STOP_VERSION" == false ]]')
    runner = text.index('/usr/bin/time -v python "$CLASS_RUNNER"')
    assert smoke < authenticate < no_stop < runner
    assert "#SBATCH --array=1-4%2" in text
    assert "RESOURCE_SMOKE_PASS" in text
    assert "RESOURCE_SMOKE_PASS_SHA256" in text
    assert "FIRST_FOLD_AUTH_COMPLETE_SHA256" in text


def test_five_fold_aggregate_reauthenticates_both_release_gates_before_running() -> None:
    text = _text("verify_class_conditional_template_score_1.1_aggregate_five.sh")
    resource = text.index('ptm_class_require_resource_smoke "$SMOKE_PASS"')
    single_fold = text.index("authenticate_single_fold_release")
    no_stop = text.index('[[ "$STOP_VERSION" == false ]]')
    aggregate = text.index('/usr/bin/time -v python "$CLASS_AGGREGATOR"')
    assert resource < single_fold < no_stop < aggregate
    assert "RESOURCE_SMOKE_PASS" in text
    assert "RESOURCE_SMOKE_PASS_SHA256" in text
    assert "FIRST_FOLD_AUTH_DIR" in text
    assert "FIRST_FOLD_AUTH_COMPLETE_SHA256" in text
    assert 'expected_fold_commit=sys.argv[3]' in text
    assert 'expected_fold_directory=Path(sys.argv[5])' in text
    assert "FIRST_FOLD_JOB_ID}_0_${SHORT_COMMIT}_outer_half_cylinder" in text
    assert 'RUN_DIR="$FIRST_FOLD_DIR"' in text


def test_all_numerical_wrappers_enforce_clean_exact_commit_and_no_overwrite() -> None:
    common = _text("verify_class_conditional_template_score_1.1_common.sh")
    assert 'git status --porcelain=v1 --untracked-files=all' in common
    assert '[[ "$observed_commit" == "$expected_commit" ]]' in common
    assert 'git show "${expected_commit}:${source}"' in common
    assert "tests/test_all.py" in common
    for name in (
        "verify_class_conditional_template_score_1.1_first_fold.sh",
        "verify_class_conditional_template_score_1.1_first_fold_auth.sh",
        "verify_class_conditional_template_score_1.1_remaining_folds.sh",
        "verify_class_conditional_template_score_1.1_aggregate_five.sh",
    ):
        text = _text(name)
        assert "ptm_class_stage_gate" in text
        assert "ptm_class_stage_unchanged" in text
        assert "ptm_class_require_slurm_resources" in text
        assert "[[ ! -e" in text


def test_fresh_clone_contains_the_slurm_log_directory() -> None:
    marker = ROOT / "slurm_logs" / ".gitkeep"
    assert marker.is_file()
    assert "Slurm output directory" in marker.read_text(encoding="utf-8")
    ignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "slurm_logs/*" in ignore_lines
    assert "!slurm_logs/.gitkeep" in ignore_lines


def test_job_local_tmp_is_exported_before_python_tempfile_use() -> None:
    common = _text("verify_class_conditional_template_score_1.1_common.sh")
    smoke = _text("verify_class_conditional_template_score_1.1_resource_smoke.sh")
    for text in (common, smoke):
        root = text.index('export CLASS_JOB_TMP_ROOT=')
        tmpdir = text.index('export TMPDIR="$CLASS_JOB_TMP_ROOT"')
        numba = text.index('export NUMBA_CACHE_DIR="$CLASS_JOB_TMP_ROOT/numba_cache"')
        guard = text.index("tempfile.gettempdir()")
        assert root < tmpdir < numba < guard
        assert 'export TMP="$CLASS_JOB_TMP_ROOT"' in text
        assert 'export TEMP="$CLASS_JOB_TMP_ROOT"' in text


def test_aggregate_output_directories_match_the_frozen_pattern() -> None:
    expected = (
        'readonly OUTPUT_DIR="$CLASS_EXPERIMENT_ROOT/aggregate/'
        'slurm_${JOB_ID}_${SHORT_COMMIT}"'
    )
    for name in (
        "verify_class_conditional_template_score_1.1_first_fold_auth.sh",
        "verify_class_conditional_template_score_1.1_aggregate_five.sh",
    ):
        text = _text(name)
        assert expected in text
