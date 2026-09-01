from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import run_other_class_conditional_template_score_boeing_diagnostic_1_1 as runner  # noqa: E402
from scripts import run_verify_class_conditional_template_score_1_1 as verify  # noqa: E402


CONFIG = ROOT / "config" / "Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.yaml"


def _expect_error(error_types, function, *args, contains: str | None = None, **kwargs):
    try:
        function(*args, **kwargs)
    except error_types as error:
        if contains is not None:
            assert contains in str(error), str(error)
        return error
    raise AssertionError("expected an exception")


def test_diagnostic_plan_is_boeing_only_and_candidate_sequence_is_identical() -> None:
    plan = runner.load_plan(CONFIG)
    parent = verify.load_plan(verify.CONFIG_PATH)
    assert plan.sha256 == runner.EXPECTED_CONFIG_SHA256
    assert plan.output_root == runner.DEFAULT_OUTPUT_ROOT
    assert plan.required_fold_files == verify.REQUIRED_FOLD_FILES
    assert len(plan.required_fold_files) == 15
    observed = runner.candidate_specs(plan)
    expected = verify.candidate_specs(parent)
    assert len(observed) == len(expected) == 3060
    assert observed == expected
    scope = plan.raw["diagnostic_contract"]["diagnostic_scope"]
    assert scope["only_outer_family"] == "boeing_747"
    assert scope["outer_family_count"] == 1
    assert scope["verify_half_cylinder_selected_candidate_or_result_reuse"] == "forbidden"
    inherited = plan.raw["diagnostic_contract"]["scientific_method_parent"]
    assert tuple(inherited["exact_inherited_sections"]) == (
        "direct_parent",
        "evidence_scope",
        "input_identity",
        "families",
        "nested_split",
        "representations",
        "labels",
        "shared_negative_scaler",
        "family_class_exact_scale_conformity",
        "joint_family_support_and_score",
        "group_transform",
        "decision_candidates",
        "inner_selection",
        "final_refit_and_outer_label_gate",
        "metrics",
    )
    assert inherited["inherited_numerical_override"] == "forbidden"


def test_diagnostic_config_and_all_inherited_sources_fail_closed_on_drift() -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == runner.EXPECTED_CONFIG_SHA256
    assert hashlib.sha256(runner.VERIFY_CONFIG_PATH.read_bytes()).hexdigest() == runner.EXPECTED_VERIFY_CONFIG_SHA256
    assert hashlib.sha256(runner.VERIFY_RUNNER_PATH.read_bytes()).hexdigest() == runner.EXPECTED_VERIFY_RUNNER_SHA256
    assert hashlib.sha256(runner.VERIFY_AGGREGATOR_PATH.read_bytes()).hexdigest() == runner.EXPECTED_VERIFY_AGGREGATOR_SHA256
    assert hashlib.sha256(runner.CORE_PATH.read_bytes()).hexdigest() == runner.EXPECTED_VERIFY_CORE_SHA256
    with tempfile.TemporaryDirectory() as directory:
        changed = Path(directory) / CONFIG.name
        changed.write_bytes(CONFIG.read_bytes() + b"\n")
        _expect_error(
            ValueError,
            runner.load_plan,
            changed,
            contains="frozen diagnostic config SHA-256 drifted",
        )


def test_method_binding_records_verify_stop_resource_and_no_override_scope() -> None:
    plan = runner.load_plan(CONFIG)
    binding = runner._method_binding(plan, "1" * 40)
    parent = binding["diagnostic_parent"]
    assert binding["experiment"] == runner.EXPERIMENT
    assert binding["config"]["sha256"] == runner.EXPECTED_CONFIG_SHA256
    assert binding["fit"]["k"] == [1, 5, 15, 31]
    assert binding["score"]["outer_support"] == "3_of_4_joint_families"
    assert binding["threshold"]["comparison"] == "strict_greater_than"
    assert binding["fold_transaction"] == "unchanged_parent_15_files"
    assert parent["config_sha256"] == runner.EXPECTED_VERIFY_CONFIG_SHA256
    assert parent["runner_sha256"] == runner.EXPECTED_VERIFY_RUNNER_SHA256
    assert parent["aggregator_sha256"] == runner.EXPECTED_VERIFY_AGGREGATOR_SHA256
    assert parent["stop_release_completion_sha256"] == runner.EXPECTED_VERIFY_STOP_COMPLETION_SHA256
    assert parent["resource_pass_sha256"] == runner.EXPECTED_VERIFY_RESOURCE_PASS_SHA256
    assert parent["role"].endswith("not_a_numerical_input")
    assert binding["diagnostic_scope"] == {
        "only_outer_family": "boeing_747",
        "only_outer_dataset": "boeing747",
        "evidence_scope": runner.EVIDENCE_SCOPE,
        "success_stop_or_macro_semantics": "forbidden",
    }
    assert binding["diagnostic_adapter"]["sha256"] == hashlib.sha256(
        runner.ADAPTER_PATH.read_bytes()
    ).hexdigest()


def test_runtime_binding_is_exclusive_and_restores_verify_after_error() -> None:
    names = (
        "EXPERIMENT",
        "CONFIG_PATH",
        "EXPECTED_CONFIG_SHA256",
        "DEFAULT_OUTPUT_ROOT",
        "load_plan",
        "_method_binding",
    )
    before = {name: getattr(verify, name) for name in names}
    try:
        with runner.diagnostic_verify_runtime():
            assert verify.EXPERIMENT == runner.EXPERIMENT
            assert verify.CONFIG_PATH == runner.CONFIG_PATH
            assert verify.load_plan is not runner.load_plan
            assert verify.load_plan().sha256 == runner.EXPECTED_CONFIG_SHA256
            assert verify._method_binding is runner._method_binding
            _expect_error(
                ValueError,
                lambda: runner.diagnostic_verify_runtime().__enter__(),
                contains="already active",
            )
            raise RuntimeError("synthetic exit")
    except RuntimeError as error:
        assert str(error) == "synthetic exit"
    for name, value in before.items():
        assert getattr(verify, name) is value
    with runner.diagnostic_verify_runtime():
        assert verify.EXPERIMENT == runner.EXPERIMENT
    for name, value in before.items():
        assert getattr(verify, name) is value


def test_restore_failure_permanently_poisons_runtime_before_any_rebind() -> None:
    code = r"""
from scripts import run_other_class_conditional_template_score_boeing_diagnostic_1_1 as runner

plan = runner.load_plan(runner.CONFIG_PATH)
original_setter = runner._set_verify_global
calls = []

def fail_first_restore(name, value):
    calls.append(name)
    if len(calls) == 7:
        raise RuntimeError("synthetic restore failure")
    original_setter(name, value)

runner._set_verify_global = fail_first_restore
try:
    with runner.diagnostic_verify_runtime(plan):
        pass
except RuntimeError as error:
    assert "failed to restore Verify globals" in str(error), error
else:
    raise AssertionError("restore failure was not raised")

assert runner._ADAPTER_POISONED == "_method_binding", runner._ADAPTER_POISONED

def forbidden_rebind(name, value):
    raise AssertionError(f"poisoned runtime attempted to rebind {name}")

runner._set_verify_global = forbidden_rebind
try:
    runner.diagnostic_verify_runtime(plan).__enter__()
except ValueError as error:
    assert "permanently poisoned" in str(error), error
else:
    raise AssertionError("poisoned runtime allowed a second entry")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_runner_rejects_non_boeing_before_git_or_numerical_access() -> None:
    with (
        patch.object(runner, "_git_identity") as git_identity,
        patch.object(runner, "_VERIFY_RUN") as numerical,
    ):
        _expect_error(
            ValueError,
            runner.run,
            CONFIG,
            "channel",
            Path("never_written"),
            device="cpu",
            kinematic_input_manifest_path="kinematic.json",
            kinematic_input_manifest_file_sha256="1" * 64,
            synthetic_pass_path="pass.json",
            synthetic_pass_file_sha256="2" * 64,
            sidecar_root="sidecars",
            sidecar_population_manifest_path="population.json",
            sidecar_population_manifest_file_sha256="3" * 64,
            contains="rejects every non-Boeing",
        )
    git_identity.assert_not_called()
    numerical.assert_not_called()


def test_runner_delegates_boeing_to_the_pinned_verify_path_under_other_identity() -> None:
    commit = "2" * 40
    calls = []

    def numerical(config_path, outer_family, output_dir, **kwargs):
        calls.append((config_path, outer_family, output_dir, dict(kwargs)))
        assert verify.EXPERIMENT == runner.EXPERIMENT
        assert verify.load_plan is not runner.load_plan
        plan = verify.load_plan(config_path)
        return {
            "schema": runner.RESULT_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "outer_family": runner.ONLY_OUTER_FAMILY,
            "config_sha256": plan.sha256,
            "git_commit": commit,
            runner.METHOD_BINDING_KEY: runner._method_binding(plan, commit),
            "content_sha256": "3" * 64,
        }

    with (
        patch.object(runner, "_git_identity", return_value=(commit, False)),
        patch.object(runner, "_VERIFY_RUN", side_effect=numerical),
    ):
        result = runner.run(
            CONFIG,
            "boeing_747",
            Path("synthetic_output"),
            device="cpu",
            kinematic_input_manifest_path="kinematic.json",
            kinematic_input_manifest_file_sha256="4" * 64,
            synthetic_pass_path="pass.json",
            synthetic_pass_file_sha256="5" * 64,
            sidecar_root="sidecars",
            sidecar_population_manifest_path="population.json",
            sidecar_population_manifest_file_sha256="6" * 64,
        )
    assert result["experiment"] == runner.EXPERIMENT
    assert result["outer_family"] == "boeing_747"
    assert len(calls) == 1
    assert calls[0][1] == "boeing_747"
    assert calls[0][3]["expected_config_sha256"] == runner.EXPECTED_CONFIG_SHA256
    assert verify.EXPERIMENT != runner.EXPERIMENT
    assert verify.load_plan is not runner.load_plan


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
    print(
        "other_class_conditional_template_score_boeing_diagnostic_"
        f"runner_tests={len(tests)}_passed"
    )
