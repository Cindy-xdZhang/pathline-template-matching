"""Freeze-only contract tests for Verify_DimensionlessDeformationFMT_1.1.

These tests intentionally validate only the preregistered YAML.  They do not
implement the transform and must not open any cache, feature, or outer result.
"""

from __future__ import annotations

from decimal import Decimal
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "Verify_DimensionlessDeformationFMT_1.1.yaml"
EXPECTED_CONFIG_SHA256 = (
    "c689b1d265bbc39327b2ed4147e8ffb22450dcd26f87b7c19ceae346c9ecfe18"
)
EXPECTED_FAMILY_ORDER = [
    "half_cylinder",
    "delta_wing",
    "f22_raptor",
    "channel",
    "boeing_747",
]


def _load_frozen_config() -> dict:
    payload = CONFIG_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_CONFIG_SHA256
    parsed = yaml.safe_load(payload.decode("utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def test_dimensionless_freeze_identity_input_and_forbidden_scope_are_exact():
    spec = _load_frozen_config()

    assert spec["experiment"] == "Verify_DimensionlessDeformationFMT_1.1"
    assert spec["phase"] == "exposed_train_only_nested_family_validation"
    assert spec["status"] == "frozen_pre_run_not_implemented"
    assert spec["frozen_date"].isoformat() == "2026-08-31"
    assert spec["freeze_provenance"] == {
        "parent_experiment": "Verify_PerScaleNegativeMetric_1.1",
        "parent_config_sha256": (
            "b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79"
        ),
        "timing": (
            "before_reading_any_outer_result_from_"
            "Verify_EarlyOppositePairKinematics_1.1_or_"
            "Verify_RawPCANegativeMetric_1.1"
        ),
        "historical_status_is_immutable": True,
    }

    scope = spec["evidence_scope"]
    assert scope["allowed_inputs"] == "mainExp_TemplateMatching_3.1_train_caches_only"
    assert scope["formal_confirmation"] is False
    assert scope["forbidden_datasets"] == ["tangaroa", "smokeBuoyancy"]
    assert scope["input_manifest"] == {
        "path": (
            "/ibex/user/zhanx0o/pathline-template-matching/"
            "mainExp_TemplateMatching_3.1_development/verification/"
            "Verify_LongArcHorizon_1.1/train_coverage/"
            "slurm_50998592_260a07ad380d/train_cache_input_manifest.json"
        ),
        "size_bytes": 24009,
        "sha256": (
            "e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393"
        ),
        "rows_content_sha256": (
            "ceb6d0e3fb7a2c90fcaae98583f8d1def7ee75fa7968f38d2821ee3040ae156f"
        ),
        "row_count": 32,
        "test_dataset_access": False,
    }
    assert "Tangaroa_or_Smoke_access" in spec["forbidden_changes_within_version"]


def test_dimensionless_raw_shape_order_and_transform_formula_are_exact():
    spec = _load_frozen_config()

    assert spec["primitive_contract"] == {
        "input_member": "raw_features",
        "input_dtype": "float32",
        "input_shape": ["N", 672],
        "reshape_order": "C",
        "reshaped_shape": ["N", 7, 32, 3],
        "primitive_order": [
            "center",
            "x_plus",
            "x_minus",
            "y_plus",
            "y_minus",
            "z_plus",
            "z_minus",
        ],
        "centered_origin_contract": "center_path_sample_0_is_exact_zero",
    }

    transform = spec["dimensionless_transform"]
    assert transform["arithmetic_dtype"] == "float64"
    assert transform["serialization_dtype"] == "float32"
    assert transform["center_arc_length"] == {
        "formula": "sum_t_norm2(center[t+1]-center[t])",
        "sample_interval": "t_0_to_t_31",
        "require_finite_and_strictly_positive": True,
        "forbidden_source": "configured_arc_level_or_scale_table",
    }
    assert transform["realized_initial_dx"] == {
        "relative_vectors": "neighbor_j[0]-center[0]",
        "formula": "mean_of_six_relative_vector_l2_norms",
        "require_finite_and_strictly_positive": True,
        "require_six_norms_equal_within": {"rtol": 0.00005, "atol": 0.0000001},
        "require_opposite_pair_midpoint_within": {
            "rtol": 0.00005,
            "atol": 0.0000001,
        },
        "forbidden_source": "scale_id_or_dataset_grid_metadata",
    }
    assert transform["output_coordinates"] == {
        "center": "center_path / center_arc_length",
        "neighbor": (
            "center_path / center_arc_length + "
            "(neighbor_path-center_path) / realized_initial_dx"
        ),
    }
    assert transform["invariants"] == {
        "output_shape": ["N", 7, 32, 3],
        "row_local_only": True,
        "query_batch_statistics": "forbidden",
        "flow_statistics": "forbidden",
        "labels_or_ivd": "forbidden",
        "train_fit": "none",
        "hidden_clipping_or_log": "forbidden",
    }


def test_dimensionless_representations_and_3060_candidate_grid_are_exact():
    spec = _load_frozen_config()

    assert spec["descriptor"] == {
        "implementation": "unchanged_independent_FMT",
        "parent_descriptor_id": "fmt_independent_3d_161d_sha256_25fce29499c9089e",
        "full_width": 161,
        "representations": [
            "fmt161_dimensionless_deformation",
            "real_neighbor36_dimensionless_deformation",
            "chirality_all35_dimensionless_deformation",
        ],
        "parent_coordinate_index_sets_are_unchanged": True,
        "descriptor_weights": 1.0,
        "trainable_parameters": "none",
    }
    inherited = spec["inherited_negative_metric_and_calibration"]
    assert inherited == {
        "source": "Verify_PerScaleNegativeMetric_1.1",
        "library_population": "all_natural_negative_rows_from_fit_families",
        "exact_same_scale_retrieval": True,
        "per_scale_variance_ddof": 0,
        "variance_shrinkage_lambda": 64.0,
        "tail_reference": "fit_negative_leave_one_out",
        "query_rank": "forbidden",
        "query_distribution_calibration": "forbidden",
    }

    grid = spec["candidate_grid"]
    assert grid["representation_count"] == 3
    assert grid["k"] == [1, 5, 15, 31]
    assert grid["spatial_sigma_grid_cells"] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert grid["fixed_top_fraction"] == [0.05]
    assert grid["calibrated_tail_threshold"] == {
        "start": 0.50,
        "stop_inclusive": 0.99,
        "step": 0.01,
    }
    threshold = grid["calibrated_tail_threshold"]
    threshold_count = int(
        (Decimal(str(threshold["stop_inclusive"])) - Decimal(str(threshold["start"])))
        / Decimal(str(threshold["step"]))
    ) + 1
    computed_count = (
        grid["representation_count"]
        * len(grid["k"])
        * len(grid["spatial_sigma_grid_cells"])
        * (len(grid["fixed_top_fraction"]) + threshold_count)
    )
    assert threshold_count == 50
    assert computed_count == grid["candidate_count"] == 3060
    assert grid["selection_unit"] == "equal_weighted_complete_inner_family"
    assert grid["outer_metrics_in_selection"] == "forbidden"


def test_dimensionless_split_gates_success_and_stop_rules_are_exact():
    spec = _load_frozen_config()

    assert spec["families"] == {
        "half_cylinder": ["cylinder3d", "halfcylinderRe640", "halfcylinderRe6400"],
        "delta_wing": ["deltaWing_resampled", "deltaWing_LBM"],
        "f22_raptor": ["f22raptor"],
        "channel": ["channel"],
        "boeing_747": ["boeing747"],
    }
    assert spec["nested_split"] == {
        "outer_order": EXPECTED_FAMILY_ORDER,
        "inner_order": EXPECTED_FAMILY_ORDER,
        "split_unit": "complete_physical_family",
        "random_seed_split": "forbidden",
        "outer_features_available_to_selection": False,
        "outer_labels_available_to_selection": False,
    }
    assert spec["access_gates"] == {
        "fold": [
            "authenticate_clean_exact_commit_and_config_before_cache_access",
            "fit_transform_and_select_on_nonouter_families_only",
            "write_and_authenticate_label_free_outer_prediction_before_outer_labels",
            "fresh_reload_transform_and_prediction_replay_before_outer_labels",
            "read_outer_labels_and_metrics_only_after_replay_passes",
        ],
        "aggregate": [
            "stage_only_label_free_artifacts_before_fresh_replay",
            "never_open_result_or_outer_metric_artifacts_before_fresh_replay",
            "open_outer_labels_only_after_prediction_replay",
            "recompute_group_family_and_five_family_metrics",
        ],
        "artifact_overwrite": "forbidden",
        "publish_semantics": (
            "same_directory_fsync_then_hard_link_no_replace_then_parent_fsync"
        ),
    }
    assert spec["success_rule"] == {
        "five_family_macro_f1_min": 0.70,
        "families_with_f1_at_least_0_65_min": 4,
        "single_family_f1_min": 0.50,
        "five_family_macro_average_precision_min": 0.60,
        "five_family_macro_balanced_accuracy_min": 0.70,
        "five_family_macro_precision_min": 0.60,
        "five_family_macro_recall_min": 0.60,
        "all_conditions_required": True,
    }
    assert spec["early_stop_rule"] == {
        "permitted_only_for_mathematical_impossibility": True,
        "conditions": [
            "any_completed_family_f1_below_0_50",
            "two_completed_families_f1_below_0_65",
            "optimistic_remaining_family_bound_cannot_meet_any_macro_threshold",
        ],
    }
    assert spec["forbidden_changes_within_version"] == [
        "alternate_normalization_formula",
        "learned_or_label_fitted_transform",
        "descriptor_weight_scan",
        "PCA_or_whitening",
        "scale_grid_or_assignment_change",
        "k_sigma_threshold_or_stop_rule_change",
        "query_unlabeled_adaptation",
        "Tangaroa_or_Smoke_access",
    ]
