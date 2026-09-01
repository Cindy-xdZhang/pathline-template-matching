#!/usr/bin/env python3
"""Boeing-only post-stop diagnostic adapter for the frozen ClassConditional method.

The numerical implementation is not copied here.  A short-lived, locked
runtime binding gives the authenticated Verify runner the Other experiment and
config identity, while every representation, candidate, fit, score, spatial
transform, prediction, and label gate continues to execute in the pinned
Verify implementation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.portable_flow import sha256_file  # noqa: E402
from scripts import run_verify_class_conditional_template_score_1_1 as verify  # noqa: E402


EXPERIMENT = "Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1"
CONFIG_SCHEMA = (
    "pathline_template_matching."
    "other_class_conditional_template_score_boeing_diagnostic_config.v1"
)
CONFIG_PATH = (
    ROOT / "config" / "Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.yaml"
)
EXPECTED_CONFIG_SHA256 = (
    "6112e7588efecf29cf2690b270385053d8ccd94f8e11037a6e247815afcc5856"
)
VERIFY_CONFIG_PATH = ROOT / "config" / "Verify_ClassConditionalTemplateScore_1.1.yaml"
EXPECTED_VERIFY_CONFIG_SHA256 = verify.EXPECTED_CONFIG_SHA256
EXPECTED_VERIFY_RUNNER_SHA256 = (
    "e5063887475029320e66da1f1eb221d7988598e8918d37fbe47ee213e5ff1b48"
)
EXPECTED_VERIFY_AGGREGATOR_SHA256 = (
    "77a561930ca85e3c1e6193a12e27b0b61bf7cc99be96889070962b8bfaf04e9c"
)
EXPECTED_VERIFY_CORE_SHA256 = verify.EXPECTED_CORE_SHA256
EXPECTED_VERIFY_EXECUTION_COMMIT = (
    "58b0bc0b0c7385f1b356eb343a150fcd50dad94f"
)
EXPECTED_VERIFY_STOP_COMPLETION_SHA256 = (
    "f8515858efe531c24471a11f64f014692a5d4774146c8908f07ee4ca49476844"
)
EXPECTED_VERIFY_RESOURCE_PASS_SHA256 = (
    "3f9197a19407906b0b13a2b9eaa09dbc647b166a9fe9d2ef4dc90cda532557ea"
)
VERIFY_EXPERIMENT = "Verify_ClassConditionalTemplateScore_1.1"
VERIFY_RUNNER_PATH = ROOT / "scripts" / "run_verify_class_conditional_template_score_1_1.py"
VERIFY_AGGREGATOR_PATH = (
    ROOT / "scripts" / "aggregate_verify_class_conditional_template_score_1_1.py"
)
ADAPTER_PATH = Path(__file__).resolve()
ONLY_OUTER_FAMILY = "boeing_747"
ONLY_OUTER_DATASET = "boeing747"
DEFAULT_OUTPUT_ROOT = Path(
    "/ibex/user/zhanx0o/pathline-template-matching/"
    "Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1"
)
EVIDENCE_SCOPE = "exposed_post_stop_visualization_diagnostic"

# Structural and numerical identities deliberately remain the generic
# ClassConditional schemas.  The experiment, config, method binding, and Git
# identities distinguish the Other artifacts from the stopped Verify version.
PARENT_EXPERIMENT = verify.PARENT_EXPERIMENT
EXPECTED_PARENT_CONFIG_SHA256 = verify.EXPECTED_PARENT_CONFIG_SHA256
EXPECTED_PARENT_RUNNER_SHA256 = verify.EXPECTED_PARENT_RUNNER_SHA256
EXPECTED_PARENT_AGGREGATOR_SHA256 = verify.EXPECTED_PARENT_AGGREGATOR_SHA256
EXPECTED_PARENT_NUMERICAL_GIT_COMMIT = verify.EXPECTED_PARENT_NUMERICAL_GIT_COMMIT
EXPECTED_CORE_SHA256 = verify.EXPECTED_CORE_SHA256
CORE_PATH = verify.CORE_PATH
SELECTED_SCHEMA = verify.SELECTED_SCHEMA
PREDICTION_SCHEMA = verify.PREDICTION_SCHEMA
PREDICTION_MANIFEST_SCHEMA = verify.PREDICTION_MANIFEST_SCHEMA
OUTER_SUMMARY_SCHEMA = verify.OUTER_SUMMARY_SCHEMA
REFERENCE_AUDIT_SCHEMA = verify.REFERENCE_AUDIT_SCHEMA
RESULT_SCHEMA = verify.RESULT_SCHEMA
COMPLETE_SCHEMA = verify.COMPLETE_SCHEMA
METHOD_BINDING_KEY = verify.METHOD_BINDING_KEY
REQUIRED_FOLD_FILES = verify.REQUIRED_FOLD_FILES
PREDICTION_ARRAY_DTYPES = verify.PREDICTION_ARRAY_DTYPES
FROZEN_CANDIDATE_COUNT = verify.FROZEN_CANDIDATE_COUNT
FAMILY_ORDER = verify.FAMILY_ORDER
TailCandidateSpec = verify.TailCandidateSpec
OuterMetricRows = verify.OuterMetricRows
Plan = verify.Plan
inherited = verify.inherited

_VERIFY_LOAD_PLAN = verify.load_plan
_VERIFY_RUN = verify.run
_VERIFY_EVALUATE_OUTER_PREDICTION = verify.evaluate_outer_prediction
_VERIFY_METHOD_BINDING = verify._method_binding
_VERIFY_BIND_EARLY_EVIDENCE = verify.bind_early_evidence
_VERIFY_GIT_IDENTITY = verify._git_identity
_ADAPTER_LOCK = threading.Lock()
_ADAPTER_POISONED: str | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is missing")
    assert isinstance(value, Mapping)
    return value


def _lower_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _strict_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        return set(left) == set(right) and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        assert isinstance(right, list)
        return len(left) == len(right) and all(
            _strict_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _diagnostic_parent_binding(plan: Plan) -> dict[str, Any]:
    contract = _mapping(plan.raw.get("diagnostic_contract"), "diagnostic contract")
    parent = _mapping(contract.get("scientific_method_parent"), "scientific method parent")
    stopped = _mapping(contract.get("authenticated_stop_parent"), "authenticated stop parent")
    resource = _mapping(contract.get("resource_evidence"), "resource evidence")
    return {
        "schema": (
            "pathline_template_matching."
            "class_conditional_template_score_boeing_diagnostic_parent.v1"
        ),
        "experiment": str(parent["experiment"]),
        "config_sha256": str(_mapping(parent["config"], "parent config")["sha256"]),
        "runner_sha256": str(_mapping(parent["runner"], "parent runner")["sha256"]),
        "aggregator_sha256": str(
            _mapping(parent["authenticated_aggregator"], "parent aggregator")["sha256"]
        ),
        "core_sha256": str(_mapping(parent["core"], "parent core")["sha256"]),
        "stopped_execution_git_commit": str(parent["stopped_execution_git_commit"]),
        "stop_release_completion_sha256": str(stopped["completion_file_sha256"]),
        "resource_pass_sha256": str(resource["pass_file_sha256"]),
        "role": "provenance_and_execution_authorization_only_not_a_numerical_input",
    }


def _method_binding(plan: Plan, git_commit: str) -> dict[str, Any]:
    """Return the unchanged scientific binding plus explicit diagnostic provenance."""

    _require(_lower_hex(git_commit, 40), "diagnostic Git commit must be lowercase SHA-1")
    binding = dict(_VERIFY_METHOD_BINDING(plan, git_commit))
    binding["experiment"] = EXPERIMENT
    binding["diagnostic_parent"] = _diagnostic_parent_binding(plan)
    binding["diagnostic_adapter"] = {
        "path": "scripts/run_other_class_conditional_template_score_boeing_diagnostic_1_1.py",
        "sha256": sha256_file(ADAPTER_PATH),
    }
    binding["diagnostic_scope"] = {
        "only_outer_family": ONLY_OUTER_FAMILY,
        "only_outer_dataset": ONLY_OUTER_DATASET,
        "evidence_scope": EVIDENCE_SCOPE,
        "success_stop_or_macro_semantics": "forbidden",
    }
    return inherited._json_safe(binding)


def _validate_parent_contract(raw: Mapping[str, Any]) -> None:
    parent = _mapping(raw.get("scientific_method_parent"), "scientific method parent")
    expected_parent_fields = {
        "experiment",
        "inheritance",
        "stopped_execution_git_commit",
        "config",
        "core",
        "runner",
        "authenticated_aggregator",
        "exact_inherited_sections",
        "inherited_candidate_count",
        "inherited_candidate_identity_sequence_must_match_exactly",
        "inherited_numerical_override",
    }
    _require(set(parent) == expected_parent_fields, "scientific parent fields drifted")
    config = _mapping(parent.get("config"), "scientific parent config")
    core = _mapping(parent.get("core"), "scientific parent core")
    runner = _mapping(parent.get("runner"), "scientific parent runner")
    aggregator = _mapping(
        parent.get("authenticated_aggregator"), "scientific parent aggregator"
    )
    _require(
        parent.get("experiment") == VERIFY_EXPERIMENT
        and parent.get("inheritance")
        == "exact_complete_numerical_contract_without_override"
        and parent.get("stopped_execution_git_commit")
        == EXPECTED_VERIFY_EXECUTION_COMMIT
        and config.get("path")
        == "config/Verify_ClassConditionalTemplateScore_1.1.yaml"
        and config.get("sha256") == EXPECTED_VERIFY_CONFIG_SHA256
        and core.get("path")
        == "src/pathline_template_matching/class_conditional_template_score.py"
        and core.get("sha256") == EXPECTED_VERIFY_CORE_SHA256
        and runner.get("path")
        == "scripts/run_verify_class_conditional_template_score_1_1.py"
        and runner.get("sha256") == EXPECTED_VERIFY_RUNNER_SHA256
        and aggregator.get("path")
        == "scripts/aggregate_verify_class_conditional_template_score_1_1.py"
        and aggregator.get("sha256") == EXPECTED_VERIFY_AGGREGATOR_SHA256
        and parent.get("inherited_candidate_count") == FROZEN_CANDIDATE_COUNT
        and parent.get("inherited_candidate_identity_sequence_must_match_exactly")
        is True
        and parent.get("inherited_numerical_override") == "forbidden",
        "scientific parent identity or no-override contract drifted",
    )
    inherited_sections = tuple(parent.get("exact_inherited_sections", ()))
    _require(
        inherited_sections
        == (
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
        ),
        "inherited method section order or identity drifted",
    )


def _validate_stop_and_scope(raw: Mapping[str, Any]) -> None:
    stopped = _mapping(raw.get("authenticated_stop_parent"), "authenticated stop parent")
    _require(
        stopped.get("role")
        == "execution_authorization_and_provenance_only_not_a_numerical_input"
        and stopped.get("expected_outer_family") == "half_cylinder"
        and stopped.get("expected_stop_version") is True
        and stopped.get("fold_job_id") == 51146327
        and stopped.get("authentication_job_id") == 51146768
        and stopped.get("completion_file") == "AGGREGATE_COMPLETE.json"
        and stopped.get("completion_file_sha256")
        == EXPECTED_VERIFY_STOP_COMPLETION_SHA256,
        "authenticated Verify stop binding drifted",
    )
    resource = _mapping(raw.get("resource_evidence"), "resource evidence")
    _require(
        resource.get("experiment") == VERIFY_EXPERIMENT
        and resource.get("execution_git_commit") == EXPECTED_VERIFY_EXECUTION_COMMIT
        and resource.get("config_sha256") == EXPECTED_VERIFY_CONFIG_SHA256
        and resource.get("job_id") == 51146125
        and resource.get("pass_file_sha256") == EXPECTED_VERIFY_RESOURCE_PASS_SHA256
        and resource.get("performance_evidence") is False,
        "resource evidence binding drifted",
    )
    scope = _mapping(raw.get("diagnostic_scope"), "diagnostic scope")
    _require(
        scope.get("only_outer_family") == ONLY_OUTER_FAMILY
        and scope.get("only_outer_dataset") == ONLY_OUTER_DATASET
        and scope.get("fit_families")
        == ["half_cylinder", "delta_wing", "f22_raptor", "channel"]
        and scope.get("inner_query_families")
        == ["half_cylinder", "delta_wing", "f22_raptor", "channel"]
        and scope.get("outer_family_count") == 1
        and scope.get("verify_half_cylinder_selected_candidate_or_result_reuse")
        == "forbidden",
        "Boeing-only diagnostic scope drifted",
    )
    evidence = _mapping(raw.get("evidence_scope"), "evidence scope")
    _require(
        evidence.get("level") == EVIDENCE_SCOPE
        and evidence.get("formal_confirmation") is False
        and evidence.get("forbidden_datasets") == ["tangaroa", "smokeBuoyancy"],
        "diagnostic evidence scope drifted",
    )


def load_plan(config_path: str | Path = CONFIG_PATH) -> Plan:
    """Authenticate the thin diagnostic config and derive the exact Verify plan."""

    path = Path(config_path).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    _require(digest == EXPECTED_CONFIG_SHA256, "frozen diagnostic config SHA-256 drifted")
    raw = yaml.safe_load(payload.decode("utf-8"))
    _require(isinstance(raw, Mapping), "diagnostic config root is invalid")
    assert isinstance(raw, Mapping)
    expected_top = {
        "schema",
        "experiment",
        "phase",
        "status",
        "freeze",
        "purpose",
        "scientific_method_parent",
        "authenticated_stop_parent",
        "resource_evidence",
        "diagnostic_scope",
        "evidence_scope",
        "input_identity",
        "authentication",
        "execution",
        "output",
    }
    _require(set(raw) == expected_top, "diagnostic config fields drifted")
    _require(
        raw.get("schema") == CONFIG_SCHEMA
        and raw.get("experiment") == EXPERIMENT
        and raw.get("phase")
        == "post_stop_exposed_boeing_outer_visualization_diagnostic"
        and raw.get("status") == "frozen_pre_run_not_implemented",
        "diagnostic experiment identity drifted",
    )
    freeze = _mapping(raw.get("freeze"), "freeze history")
    _require(
        freeze.get("frozen_before_first_read_of_any_diagnostic_real_array_or_result")
        is True
        and freeze.get("new_version_runner_exists") is False
        and freeze.get(
            "new_version_real_feature_label_valid_rate_prediction_or_metric_read"
        )
        is False,
        "diagnostic pre-read freeze history drifted",
    )
    _validate_parent_contract(raw)
    _validate_stop_and_scope(raw)
    _require(
        sha256_file(VERIFY_CONFIG_PATH) == EXPECTED_VERIFY_CONFIG_SHA256
        and sha256_file(VERIFY_RUNNER_PATH) == EXPECTED_VERIFY_RUNNER_SHA256
        and sha256_file(VERIFY_AGGREGATOR_PATH) == EXPECTED_VERIFY_AGGREGATOR_SHA256
        and sha256_file(CORE_PATH) == EXPECTED_VERIFY_CORE_SHA256,
        "pinned Verify scientific or authentication source changed",
    )
    parent_plan = _VERIFY_LOAD_PLAN(VERIFY_CONFIG_PATH)
    parent_candidates = verify.candidate_specs(parent_plan)
    _require(
        len(parent_candidates) == FROZEN_CANDIDATE_COUNT,
        "pinned Verify candidate count drifted",
    )
    output = _mapping(raw.get("output"), "diagnostic output contract")
    _require(
        Path(str(output.get("root"))) == DEFAULT_OUTPUT_ROOT
        and output.get("overwrite") == "forbidden"
        and output.get("atomic_publish") == "hard_link_without_replace",
        "diagnostic output contract drifted",
    )
    merged_raw = dict(inherited._json_safe(parent_plan.raw))
    merged_raw["experiment"] = EXPERIMENT
    merged_raw["diagnostic_contract"] = inherited._json_safe(dict(raw))
    merged_raw["diagnostic_scope"] = inherited._json_safe(raw["diagnostic_scope"])
    merged_raw["evidence_scope"] = inherited._json_safe(raw["evidence_scope"])
    merged_raw["output"] = inherited._json_safe(raw["output"])
    plan = replace(
        parent_plan,
        path=path,
        sha256=digest,
        raw=merged_raw,
        output_root=DEFAULT_OUTPUT_ROOT,
    )
    observed_candidates = candidate_specs(plan)
    _require(
        len(observed_candidates) == len(parent_candidates)
        and all(
            observed == expected
            for observed, expected in zip(
                observed_candidates, parent_candidates, strict=True
            )
        ),
        "diagnostic and Verify candidate sequences differ",
    )
    return plan


def candidate_specs(plan: Plan) -> tuple[TailCandidateSpec, ...]:
    candidates = verify.candidate_specs(plan)
    _require(len(candidates) == FROZEN_CANDIDATE_COUNT, "candidate count drifted")
    return candidates


def bind_early_evidence(plan: Plan, **kwargs: Any) -> Plan:
    return _VERIFY_BIND_EARLY_EVIDENCE(plan, **kwargs)


def _set_verify_global(name: str, value: Any) -> None:
    """Single injectable mutation seam used by restoration-failure tests."""

    setattr(verify, name, value)


@contextmanager
def diagnostic_verify_runtime(plan: Plan | None = None) -> Iterator[None]:
    """Bind and restore the stopped Verify adapter without allowing overlap."""

    global _ADAPTER_POISONED

    _require(
        _ADAPTER_POISONED is None,
        "Boeing diagnostic Verify runtime is permanently poisoned after a "
        f"restoration failure: {_ADAPTER_POISONED}",
    )
    acquired = _ADAPTER_LOCK.acquire(blocking=False)
    _require(acquired, "Boeing diagnostic Verify runtime is already active")
    old: dict[str, Any] = {}
    attempted: list[str] = []
    failures: list[tuple[str, BaseException]] = []
    try:
        active_plan = load_plan(CONFIG_PATH) if plan is None else plan
        _require(
            active_plan.path == CONFIG_PATH.resolve()
            and active_plan.sha256 == EXPECTED_CONFIG_SHA256,
            "Boeing diagnostic runtime received another plan",
        )

        def bound_load_plan(config_path: str | Path = CONFIG_PATH) -> Plan:
            active_path = Path(config_path).resolve()
            _require(
                active_path == active_plan.path,
                "Verify adapter requested another diagnostic config",
            )
            _require(
                sha256_file(active_path) == active_plan.sha256,
                "diagnostic config changed inside Verify adapter",
            )
            return active_plan

        replacements = {
            "EXPERIMENT": EXPERIMENT,
            "CONFIG_PATH": CONFIG_PATH,
            "EXPECTED_CONFIG_SHA256": EXPECTED_CONFIG_SHA256,
            "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT,
            "load_plan": bound_load_plan,
            "_method_binding": _method_binding,
        }
        old = {name: getattr(verify, name) for name in replacements}
        for name, value in replacements.items():
            attempted.append(name)
            _set_verify_global(name, value)
        yield
    finally:
        for name in reversed(attempted):
            try:
                _set_verify_global(name, old[name])
            except BaseException as error:  # pragma: no cover - catastrophic
                failures.append((name, error))
        if failures:
            names = ", ".join(name for name, _ in failures)
            _ADAPTER_POISONED = names
        _ADAPTER_LOCK.release()
        if failures:
            raise RuntimeError(
                f"failed to restore Verify globals after Boeing diagnostic: {names}"
            ) from failures[0][1]


def evaluate_outer_prediction(
    plan: Plan,
    selected: TailCandidateSpec,
    output_directory: Path,
    **kwargs: Any,
) -> tuple[OuterMetricRows, list[dict[str, Any]]]:
    _require(
        kwargs.get("outer_family") == ONLY_OUTER_FAMILY,
        "diagnostic fresh replay is Boeing-only",
    )
    with diagnostic_verify_runtime(plan):
        return _VERIFY_EVALUATE_OUTER_PREDICTION(
            plan, selected, Path(output_directory), **kwargs
        )


def _git_identity() -> tuple[str, bool]:
    return _VERIFY_GIT_IDENTITY()


def run(
    config_path: str | Path,
    outer_family: str,
    output_dir: str | Path,
    *,
    device: str,
    kinematic_input_manifest_path: str | Path,
    kinematic_input_manifest_file_sha256: str,
    synthetic_pass_path: str | Path,
    synthetic_pass_file_sha256: str,
    sidecar_root: str | Path,
    sidecar_population_manifest_path: str | Path,
    sidecar_population_manifest_file_sha256: str,
    expected_config_sha256: str | None = EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    """Run exactly one Boeing outer fold under a distinct Other identity."""

    _require(
        outer_family == ONLY_OUTER_FAMILY,
        "Other Boeing diagnostic rejects every non-Boeing outer family",
    )
    plan = load_plan(config_path)
    if expected_config_sha256 is not None:
        _require(
            expected_config_sha256 == plan.sha256,
            "explicit diagnostic config SHA-256 drifted",
        )
    commit, dirty = _git_identity()
    _require(not dirty, "Boeing diagnostic requires a clean committed worktree")
    with diagnostic_verify_runtime(plan):
        result = _VERIFY_RUN(
            config_path,
            outer_family,
            output_dir,
            device=device,
            kinematic_input_manifest_path=kinematic_input_manifest_path,
            kinematic_input_manifest_file_sha256=(
                kinematic_input_manifest_file_sha256
            ),
            synthetic_pass_path=synthetic_pass_path,
            synthetic_pass_file_sha256=synthetic_pass_file_sha256,
            sidecar_root=sidecar_root,
            sidecar_population_manifest_path=sidecar_population_manifest_path,
            sidecar_population_manifest_file_sha256=(
                sidecar_population_manifest_file_sha256
            ),
            expected_config_sha256=plan.sha256,
        )
    _require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("experiment") == EXPERIMENT
        and result.get("outer_family") == ONLY_OUTER_FAMILY
        and result.get("config_sha256") == plan.sha256
        and result.get("git_commit") == commit
        and _strict_json_equal(
            result.get(METHOD_BINDING_KEY), _method_binding(plan, commit)
        ),
        "Boeing diagnostic result identity drifted",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--expected-config-sha256", default=EXPECTED_CONFIG_SHA256)
    parser.add_argument("--outer-family", required=True, choices=(ONLY_OUTER_FAMILY,))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kinematic-input-manifest", required=True)
    parser.add_argument("--kinematic-input-manifest-sha256", required=True)
    parser.add_argument("--synthetic-pass", required=True)
    parser.add_argument("--synthetic-pass-sha256", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--sidecar-population-manifest", required=True)
    parser.add_argument("--sidecar-population-manifest-sha256", required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.outer_family,
        arguments.output_dir,
        device=arguments.device,
        kinematic_input_manifest_path=arguments.kinematic_input_manifest,
        kinematic_input_manifest_file_sha256=(
            arguments.kinematic_input_manifest_sha256
        ),
        synthetic_pass_path=arguments.synthetic_pass,
        synthetic_pass_file_sha256=arguments.synthetic_pass_sha256,
        sidecar_root=arguments.sidecar_root,
        sidecar_population_manifest_path=arguments.sidecar_population_manifest,
        sidecar_population_manifest_file_sha256=(
            arguments.sidecar_population_manifest_sha256
        ),
        expected_config_sha256=arguments.expected_config_sha256,
    )
    print(f"experiment={result['experiment']}")
    print(f"outer_family={result['outer_family']}")
    print(f"result_content_sha256={result['content_sha256']}")


if __name__ == "__main__":
    main()
