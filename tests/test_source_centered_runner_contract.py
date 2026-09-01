from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import (
    aggregate_verify_source_centered_paired_scale_template_1_1 as aggregate,
)
from scripts import run_verify_source_centered_paired_scale_template_1_1 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "Verify_SourceCenteredPairedScaleTemplate_1.1.yaml"


def _expect_value_error(callable_object: object, *args: object, match: str) -> None:
    try:
        callable_object(*args)  # type: ignore[operator]
    except (KeyError, ValueError) as error:
        assert match in str(error), (match, str(error))
    else:
        raise AssertionError("expected a closed failure")


def test_source_centered_plan_freezes_complete_1800_candidate_grid() -> None:
    plan = runner.load_plan(CONFIG)
    candidates = runner.candidate_specs(plan)
    assert plan.weights == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert plan.top_fractions == (0.025, 0.04, 0.05, 0.06, 0.075, 0.1)
    assert len(candidates) == runner.FROZEN_CANDIDATE_COUNT == 1800
    assert len({candidate.candidate_id for candidate in candidates}) == 1800
    assert {candidate.top_fraction for candidate in candidates} == set(
        plan.top_fractions
    )
    assert {candidate.weight for candidate in candidates} == set(plan.weights)


def test_selected_candidate_payload_round_trips_fraction_and_rejects_tamper() -> None:
    plan = runner.load_plan(CONFIG)
    candidate = runner.candidate_specs(plan)[-1]
    payload = runner._candidate_payload(candidate)
    assert payload["decision_rule"] == "fixed_top_fraction"
    assert payload["decision_value"] == candidate.top_fraction == 0.1
    assert aggregate._candidate_from_payload(plan, payload) == candidate

    tampered = dict(payload)
    tampered["decision_value"] = 0.03
    tampered["candidate_id"] = str(payload["candidate_id"]).replace(
        "fixed_top_fraction=0.100", "fixed_top_fraction=0.030"
    )
    _expect_value_error(
        aggregate._candidate_from_payload,
        plan,
        tampered,
        match="outside the frozen 1,800-member set",
    )


def test_single_fold_stop_certificate_uses_optimistic_complete_five_bounds() -> None:
    primary = {
        "f1": 0.90,
        "average_precision": 0.90,
        "balanced_accuracy": 0.90,
        "precision": 0.90,
        "recall": 0.90,
        "unique_center_combined_coverage": 0.49,
    }
    fold = SimpleNamespace(
        outer_family="half_cylinder",
        fresh_summary={"primary": primary},
    )
    certificate = aggregate._single_fold_stop_certificate(fold)  # type: ignore[arg-type]
    coverage = certificate["optimistic_macro_bounds"][
        "unique_center_combined_coverage"
    ]
    assert coverage["complete_five_macro_upper_bound"] == (0.49 + 4.0) / 5.0
    assert not coverage["possible"]
    assert certificate["stop_version"]
    assert (
        "five_family_macro_unique_center_combined_coverage"
        in certificate["impossible_complete_five_thresholds"]
    )

    primary["unique_center_combined_coverage"] = 0.50
    possible = aggregate._single_fold_stop_certificate(fold)  # type: ignore[arg-type]
    assert not possible["stop_version"]
