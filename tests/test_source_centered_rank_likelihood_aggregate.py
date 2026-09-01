from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import (
    aggregate_verify_source_centered_rank_likelihood_template_1_1 as aggregate,
)
from scripts import (
    run_verify_source_centered_rank_likelihood_template_1_1 as runner,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml"


def _expect_value_error(function: object, *args: object, match: str) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except (KeyError, TypeError, ValueError) as error:
        assert match in str(error), (match, str(error))
    else:
        raise AssertionError("expected closed failure")


def test_primary_and_control_payloads_round_trip_and_reject_cross_arm_tamper() -> None:
    plan = runner.load_plan(CONFIG)
    primary = runner.candidate_specs(plan)[-1]
    control = runner.control_specs(plan)[-1]
    assert aggregate._candidate_from_payload(
        plan, runner._candidate_payload(primary)
    ) == primary
    assert aggregate._control_from_payload(
        plan, runner._control_payload(control)
    ) == control
    tampered = runner._candidate_payload(primary)
    tampered["arm"] = "negative_ecdf"
    _expect_value_error(
        aggregate._candidate_from_payload,
        plan,
        tampered,
        match="not canonical",
    )


def test_single_fold_stop_certificate_uses_primary_only_and_optimistic_bounds() -> None:
    primary = {
        "f1": 0.90,
        "average_precision": 0.90,
        "balanced_accuracy": 0.90,
        "precision": 0.90,
        "recall": 0.90,
        "unique_center_combined_coverage": 0.49,
    }
    fold = SimpleNamespace(
        outer_family="half_cylinder", fresh_summary={"primary": primary}
    )
    certificate = aggregate._single_fold_stop_certificate(fold)  # type: ignore[arg-type]
    coverage = certificate["optimistic_macro_bounds"][
        "unique_center_combined_coverage"
    ]
    assert coverage["complete_five_macro_upper_bound"] == (0.49 + 4.0) / 5.0
    assert certificate["stop_version"]
    primary["unique_center_combined_coverage"] = 0.50
    assert not aggregate._single_fold_stop_certificate(fold)["stop_version"]  # type: ignore[arg-type]


def test_partial_stop_certificate_stops_after_two_completed_sub_065_folds() -> None:
    folds = tuple(
        SimpleNamespace(
            outer_family=family,
            fresh_summary={
                "primary": {
                    "f1": f1,
                    "average_precision": 0.90,
                    "balanced_accuracy": 0.90,
                    "precision": 0.90,
                    "recall": 0.90,
                    "unique_center_combined_coverage": 0.95,
                }
            },
        )
        for family, f1 in (("half_cylinder", 0.64), ("delta_wing", 0.63))
    )
    certificate = aggregate._partial_stop_certificate(folds)  # type: ignore[arg-type]
    assert certificate["completed_family_count"] == 2
    assert certificate["remaining_family_count"] == 3
    assert not certificate["threshold_possible"][
        "families_at_or_above_f1_0_65"
    ]
    assert certificate["stop_version"]


def test_partial_stop_certificate_stops_on_observed_sub_050_family() -> None:
    folds = tuple(
        SimpleNamespace(
            outer_family=family,
            fresh_summary={
                "primary": {
                    "f1": f1,
                    "average_precision": 0.90,
                    "balanced_accuracy": 0.90,
                    "precision": 0.90,
                    "recall": 0.90,
                    "unique_center_combined_coverage": 0.95,
                }
            },
        )
        for family, f1 in (
            ("half_cylinder", 0.49),
            ("delta_wing", 0.90),
            ("f22_raptor", 0.90),
        )
    )
    certificate = aggregate._partial_stop_certificate(folds)  # type: ignore[arg-type]
    assert not certificate["threshold_possible"]["minimum_single_family_f1"]
    assert certificate["stop_version"]


def test_outer_metric_contract_contains_exact_identity_and_all_four_arms() -> None:
    assert runner.OUTER_METRIC_FIELDS[:6] == (
        "outer_family",
        "dataset",
        "source_ordinal",
        "source_index",
        "arm",
        "population",
    )
    source = Path(
        runner.__file__  # type: ignore[arg-type]
    ).read_text(encoding="utf-8")
    for required in (
        '"valid_assigned_row_index"',
        '"valid_center_seed_index"',
        '"valid_scale_block_index"',
        '"valid_scale_id"',
        '"dual_histogram_llr"',
        '"parent_source_centered_paired_scale"',
        '"negative_ecdf"',
        '"direct_rank_mean_top5"',
    ):
        assert required in source


def test_aggregate_contract_keeps_5000_paired_bootstrap_and_controls_ineligible() -> None:
    source = Path(
        aggregate.__file__  # type: ignore[arg-type]
    ).read_text(encoding="utf-8")
    assert "np.empty(5000" in source
    assert "np.random.default_rng(17068)" in source
    assert '"controls_can_satisfy_primary_success": False' in source
    assert 'row["arm"] == "dual_histogram_llr"' in source
    assert 'row["arm"] == "parent_source_centered_paired_scale"' in source
