from __future__ import annotations

import hashlib
import inspect
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IBEX = ROOT / "ibex"
COMMON_NAME = "other_class_conditional_template_score_boeing_diagnostic_1.1_common.sh"
FOLD_NAME = "other_class_conditional_template_score_boeing_diagnostic_1.1_fold.sh"
AUTH_NAME = "other_class_conditional_template_score_boeing_diagnostic_1.1_auth.sh"
CONFIG = ROOT / "config" / "Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.yaml"
CONFIG_SHA256 = "6112e7588efecf29cf2690b270385053d8ccd94f8e11037a6e247815afcc5856"
VERIFY_COMMIT = "58b0bc0b0c7385f1b356eb343a150fcd50dad94f"
OTHER_ROOT = "/home/zhanx0o/pathline-template-matching-class-conditional-boeing"
VERIFY_PRODUCER_ROOT = "/home/zhanx0o/pathline-template-matching-class-conditional-score"


def _text(name: str) -> str:
    return (IBEX / name).read_text(encoding="utf-8")


def test_boeing_diagnostic_config_and_all_wrapper_sources_are_frozen() -> None:
    common = _text(COMMON_NAME)
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    assert f"readonly BOEING_DIAG_CONFIG_SHA256={CONFIG_SHA256}" in common
    assert 'git status --porcelain=v1 --untracked-files=all' in common
    assert '[[ "$observed_commit" == "$expected_commit" ]]' in common
    assert 'git show "${expected_commit}:${source}"' in common
    required_sources = (
        "config/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.yaml",
        "config/Verify_ClassConditionalTemplateScore_1.1.yaml",
        "src/pathline_template_matching/class_conditional_template_score.py",
        "scripts/run_other_class_conditional_template_score_boeing_diagnostic_1_1.py",
        "scripts/aggregate_other_class_conditional_template_score_boeing_diagnostic_1_1.py",
        "scripts/run_verify_class_conditional_template_score_1_1.py",
        "scripts/aggregate_verify_class_conditional_template_score_1_1.py",
        "scripts/run_verify_class_conditional_template_score_resource_smoke_1_1.py",
        f"ibex/{COMMON_NAME}",
        f"ibex/{FOLD_NAME}",
        f"ibex/{AUTH_NAME}",
        "tests/test_other_class_conditional_template_score_boeing_diagnostic_runner.py",
        "tests/test_other_class_conditional_template_score_boeing_diagnostic_aggregate.py",
        "tests/test_other_class_conditional_template_score_boeing_diagnostic_ibex.py",
        "tests/test_all.py",
    )
    for source in required_sources:
        assert source in common


def test_boeing_diagnostic_pins_the_exact_inherited_scientific_sources() -> None:
    common = _text(COMMON_NAME)
    expected = {
        "BOEING_DIAG_VERIFY_EXECUTION_COMMIT": VERIFY_COMMIT,
        "BOEING_DIAG_VERIFY_CONFIG_SHA256": "814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea",
        "BOEING_DIAG_VERIFY_CORE_SHA256": "9c009376f7cea1481f6f47a49362d54d0e78530717f480fda3e8a109f841ef99",
        "BOEING_DIAG_VERIFY_RUNNER_SHA256": "e5063887475029320e66da1f1eb221d7988598e8918d37fbe47ee213e5ff1b48",
        "BOEING_DIAG_VERIFY_AGGREGATOR_SHA256": "77a561930ca85e3c1e6193a12e27b0b61bf7cc99be96889070962b8bfaf04e9c",
        "BOEING_DIAG_VERIFY_RESOURCE_AUTHENTICATOR_SHA256": "97f02e58cf571e81466fc1d14bbc605d89c48e7d79fe0d6af86f0fdb0e780371",
        "BOEING_DIAG_VERIFY_RESOURCE_TEST_SHA256": "31b0de79c34ccf13b2ff8559801d144d816212009cc07785e37cc7f5403908c7",
        "BOEING_DIAG_VERIFY_RESOURCE_WRAPPER_SHA256": "64c77ae20ed97edba0a57152d2fe0c51ca925b19f2c156a5c504bbd0e830c667",
    }
    for name, value in expected.items():
        assert f"readonly {name}={value}" in common
    assert "ptm_boeing_diag_require_parent_source_identity" in common
    assert "ptm_boeing_diag_require_verify_producer_source_identity" in common
    assert common.count("ptm_boeing_diag_require_file_sha256") >= 16


def test_boeing_diagnostic_reauthenticates_the_exact_stopped_parent_and_resource_release() -> None:
    common = _text(COMMON_NAME)
    assert "readonly BOEING_DIAG_VERIFY_FIRST_FOLD_JOB_ID=51146327" in common
    assert "slurm_51146327_0_58b0bc0b0c73_outer_half_cylinder" in common
    assert "slurm_51146768_58b0bc0b0c73" in common
    assert "f8515858efe531c24471a11f64f014692a5d4774146c8908f07ee4ca49476844" in common
    assert "slurm_51146125_58b0bc0b0c73/RESOURCE_SMOKE_PASS.json" in common
    assert "3f9197a19407906b0b13a2b9eaa09dbc647b166a9fe9d2ef4dc90cda532557ea" in common
    assert "authenticate_resource_smoke_release" in common
    assert "authenticate_single_fold_release" in common
    assert 'resource["git_commit"] == sys.argv[5]' in common
    assert 'resource["config_sha256"] == sys.argv[6]' in common
    assert 'resource["marker"]["git_commit"]' not in common
    assert 'release["outer_family"] == "half_cylinder"' in common
    assert 'release["stop_version"] is True' in common
    assert 'release["mathematically_impossible_to_pass"] is True' in common
    assert 'sys.argv[5],\n    sys.argv[6],' in common


def test_parent_release_auth_uses_separate_immutable_absolute_checkouts() -> None:
    common = _text(COMMON_NAME)
    assert f"readonly BOEING_DIAG_PROJECT_ROOT={OTHER_ROOT}" in common
    assert (
        f"readonly BOEING_DIAG_VERIFY_PRODUCER_ROOT={VERIFY_PRODUCER_ROOT}"
        in common
    )
    assert (
        "readonly BOEING_DIAG_VERIFY_AUDIT_CONFIG_PATH="
        f"{VERIFY_PRODUCER_ROOT}/config/Verify_ClassConditionalTemplateScore_1.1.yaml"
        in common
    )
    section = common[
        common.index("ptm_boeing_diag_require_parent_releases() {") :
        common.index("ptm_boeing_diag_stage_unchanged() {")
    ]
    producer_gate = common[
        common.index("ptm_boeing_diag_require_verify_producer_checkout() {") :
        common.index("ptm_boeing_diag_stage_gate() {")
    ]
    assert 'rev-parse --verify HEAD^{commit}' in producer_gate
    assert 'symbolic-ref -q HEAD' in producer_gate
    assert 'status --porcelain=v1 --untracked-files=all' in producer_gate
    assert "ptm_boeing_diag_require_verify_producer_source_identity" in producer_gate
    assert section.count("ptm_boeing_diag_require_verify_producer_checkout") == 2
    assert 'original_directory=$(pwd -P)' in section
    assert 'original_pythonpath=${PYTHONPATH-}' in section
    assert 'cd "$BOEING_DIAG_VERIFY_PRODUCER_ROOT"' in section
    assert (
        'export PYTHONPATH="$BOEING_DIAG_VERIFY_PRODUCER_ROOT/src:'
        '$BOEING_DIAG_VERIFY_PRODUCER_ROOT"'
    ) in section
    assert "Path(verify_aggregate.__file__).resolve().is_relative_to(producer_root)" in section
    assert "Path(verify_resource.__file__).resolve().is_relative_to(producer_root)" in section
    assert "Path(verify_aggregate.runner.__file__).resolve().is_relative_to(producer_root)" in section
    assert "verify_aggregate.runner.CORE_PATH.resolve()" in section
    assert "inspect.getmodule(" in section
    assert "verify_aggregate.runner._INHERITED_GIT_IDENTITY" in section
    assert "Path(inherited_git_module.__file__).resolve().is_relative_to(producer_root)" in section
    assert "verify_resource.CONFIG_PATH.resolve() == expected_config_path" in section
    assert "verify_aggregate.runner.CONFIG_PATH.resolve() == expected_config_path" in section
    assert '[[ "$(pwd -P)" == "$original_directory" ]]' in section
    assert '[[ "${PYTHONPATH-}" == "$original_pythonpath" ]]' in section
    assert section.rfind("ptm_boeing_diag_stage_gate") > section.rfind(
        "PYTHONPATH was not restored"
    )
    for forbidden in ("git clone", "git worktree", "https://", "git@", "ssh "):
        assert forbidden not in section


def test_all_boeing_wrappers_use_the_other_checkout_and_leave_producer_root_reserved() -> None:
    common = _text(COMMON_NAME)
    assert f"readonly BOEING_DIAG_PROJECT_ROOT={OTHER_ROOT}" in common
    assert f"readonly BOEING_DIAG_VERIFY_PRODUCER_ROOT={VERIFY_PRODUCER_ROOT}" in common
    for name in (FOLD_NAME, AUTH_NAME):
        text = _text(name)
        assert f"#SBATCH --chdir={OTHER_ROOT}" in text
        assert f"#SBATCH -o {OTHER_ROOT}/slurm_logs/%x.%j.out" in text
        assert f"#SBATCH -e {OTHER_ROOT}/slurm_logs/%x.%j.err" in text
        assert VERIFY_PRODUCER_ROOT not in text


def test_boeing_diagnostic_wrappers_request_and_authenticate_the_frozen_cpu_allocation() -> None:
    for name in (FOLD_NAME, AUTH_NAME):
        text = _text(name)
        assert "#SBATCH -N 1\n" in text
        assert "#SBATCH --time=12:00:00" in text
        assert "#SBATCH --cpus-per-task=32" in text
        assert "#SBATCH --mem=128G" in text
        assert "#SBATCH --partition=cpu" in text
        assert "#SBATCH --constraint=rome" in text
        assert "#SBATCH --account=pi-hadwigm" in text
        assert "#SBATCH --gres" not in text
        assert "ptm_boeing_diag_require_slurm_resources" in text
    common = _text(COMMON_NAME)
    assert "readonly BOEING_DIAG_RUNTIME_SLURM_PARTITION=batch" in common
    assert "readonly BOEING_DIAG_RUNTIME_SLURM_ACCOUNT=pi-hadwigm" in common
    assert 'scontrol show job -o "$job_id"' in common
    assert 'expected_time_limit="12:00:00"' in common
    assert "allocation['partition']" in common
    assert "allocation['account']" in common
    assert "allocation['features']" in common
    assert "allocation['gpu_allocation']" in common
    assert '[[ "${SLURM_CPUS_PER_TASK:-}" == 32 ]]' in common
    assert '[[ "${SLURM_MEM_PER_NODE:-}" == 131072 ]]' in common


def test_fold_is_boeing_only_and_all_gates_precede_the_runner() -> None:
    text = _text(FOLD_NAME)
    assert "--outer-family boeing_747" in text
    assert "--array" not in text
    assert "OUTER_FAMILIES" not in text
    assert "remaining" not in text.casefold()
    parent = text.index("ptm_boeing_diag_require_parent_releases")
    evidence = text.index("ptm_boeing_diag_require_input_evidence")
    runner = text.index('/usr/bin/time -v python "$BOEING_DIAG_RUNNER"')
    assert evidence < parent < runner
    assert (
        'RUN_DIR="$BOEING_DIAG_EXPERIMENT_ROOT/runs/'
        'slurm_${JOB_ID}_0_${SHORT_COMMIT}_outer_boeing_747"'
    ) in text
    assert '[[ ! -e "$RUN_DIR" ]]' in text
    assert "VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256" in text
    assert "VERIFY_RESOURCE_SMOKE_PASS_SHA256" in text
    assert "ptm_boeing_diag_stage_unchanged" in text


def test_auth_uses_the_dedicated_aggregator_and_public_authenticator_last() -> None:
    text = _text(AUTH_NAME)
    parent = text.index("ptm_boeing_diag_require_parent_releases")
    aggregate = text.index('/usr/bin/time -v python "$BOEING_DIAG_AGGREGATOR"')
    public = text.index("authenticate_diagnostic_release")
    assert parent < aggregate < public
    assert "--mode" not in text
    assert "--expected-fold-commit \"$EXPECTED_GIT_COMMIT\"" in text
    assert "DIAGNOSTIC_COMPLETE.json" in text
    assert (
        'OUTPUT_DIR="$BOEING_DIAG_EXPERIMENT_ROOT/authentication/'
        'slurm_${JOB_ID}_${SHORT_COMMIT}"'
    ) in text
    assert '[[ ! -e "$OUTPUT_DIR" ]]' in text
    assert 'expected_fold_directory=Path(sys.argv[5])' in text
    assert "release['release_files']['DIAGNOSTIC_COMPLETE.json']['sha256']" in text
    assert "release['completion_sha256']" not in text
    assert text.rstrip().endswith("PY")
    final_python = text.rsplit("python - \\\n", 1)[1]
    assert "authenticate_diagnostic_release" in final_python
    assert "ptm_boeing_diag_stage_unchanged" not in final_python


def test_job_local_tmp_and_full_test_gate_are_part_of_each_job() -> None:
    common = _text(COMMON_NAME)
    root = common.index('export BOEING_DIAG_JOB_TMP_ROOT=')
    tmpdir = common.index('export TMPDIR="$BOEING_DIAG_JOB_TMP_ROOT"')
    numba = common.index('export NUMBA_CACHE_DIR="$BOEING_DIAG_JOB_TMP_ROOT/numba_cache"')
    guard = common.index("tempfile.gettempdir()")
    assert root < tmpdir < numba < guard
    assert 'export TMP="$BOEING_DIAG_JOB_TMP_ROOT"' in common
    assert 'export TEMP="$BOEING_DIAG_JOB_TMP_ROOT"' in common
    assert "tests.test_other_class_conditional_template_score_boeing_diagnostic_runner" in common
    assert "tests.test_other_class_conditional_template_score_boeing_diagnostic_aggregate" in common
    assert "tests.test_other_class_conditional_template_score_boeing_diagnostic_ibex" in common
    assert "python tests/test_all.py" in common


def test_common_rejects_forbidden_confirmation_dataset_paths() -> None:
    common = _text(COMMON_NAME)
    assert "ptm_boeing_diag_reject_confirmation_value" in common
    normalized = common.casefold()
    assert re.search(r"\*tangaroa\*", normalized)
    assert re.search(r"\*smokebuoyancy\*", normalized)
    assert 'ptm_boeing_diag_reject_confirmation_value "$path"' in common


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
        and callable(value)
        and not inspect.signature(value).parameters
    ]
    for test in tests:
        test()
    print(f"other_class_conditional_template_score_boeing_diagnostic_ibex_tests={len(tests)}_passed")
