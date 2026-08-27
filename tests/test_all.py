"""Standard-library test runner so validation does not require pytest."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_fmt_descriptor import (
    test_default_descriptor_is_161d_and_rigid_motion_invariant,
    test_descriptor_does_not_depend_on_query_batch_composition_or_chunks,
    test_descriptor_metadata_and_width_are_frozen,
    test_frozen_descriptor_rejects_wrong_tensor_contract,
    test_default_numeric_recipe_matches_legacy_task5_cache_encoder,
)
from test_data_access import (
    test_cache_validator_checks_every_slice_and_full_contract,
    test_cache_validator_detects_damage_after_first_slice,
    test_cache_validator_rejects_consistent_but_noncanonical_config_hash,
)
from test_development_core import (
    test_balanced_library_is_stratum_balanced_reproducible_and_leak_free,
    test_binary_metrics_are_tie_aware_and_match_hand_computation,
    test_deterministic_full_svd_pca_is_library_only_and_sign_frozen,
    test_exhaustive_matcher_matches_direct_all_pair_distances_and_chunks,
    test_fold_completion_hashes_reject_corrupted_artifact,
    test_high_dimensional_matcher_has_exact_self_distance_and_duplicate_tie,
    test_bootstrap_point_estimate_equals_reported_family_timeslice_macro_delta,
    test_completed_report_is_immutable,
    test_empty_class_library_stratum_policy_is_versioned_and_audited,
    test_timeslice_coverage_rejects_missing_cartesian_key,
    test_frozen_development_config_rejects_sealed_confirmation_access,
)
from test_integration import (
    test_multiscale_zero_flow_returns_frozen_7x32_contract_and_physical_scale,
    test_rk4_constant_velocity_matches_analytic_cross_primitive,
    test_rk4_linear_velocity_matches_exponential_solution_not_euler,
    test_n48_n64_rounded_sampling_and_chunk_order_are_frozen,
)
from test_ivd import (
    test_percentile_mask_rejects_nonfinite_volume,
    test_percentile_mask_uses_greater_than_or_equal,
    test_solid_body_rotation_has_zero_ivd,
)
from test_library import (
    test_cross_class_distance_tie_is_deterministically_non_vortex,
    test_exact_one_nearest_neighbor_and_signed_class_margin,
    test_library_round_trip_without_pickle,
    test_query_descriptor_mismatch_and_nonbinary_labels_are_rejected,
    test_query_does_not_refit_library_normalization,
)
from test_netcdf_io import (
    test_dimension_aware_window_load_and_stride,
    test_extra_velocity_dimension_requires_explicit_policy,
    test_invalid_or_missing_coordinate_fails_closed,
    test_masked_velocity_fails_closed,
)
from test_scales import (
    test_balanced_assignment_is_reproducible_and_shuffled,
    test_frozen_scale_sets_are_valid_and_disjoint,
    test_scale_table_rejects_insufficient_integration_steps,
    test_scale_table_rejects_duplicate_numeric_tuple,
)
from test_visualization import (
    test_confusion_masks_are_mutually_exclusive_and_exhaustive,
    test_scene_validation_rejects_invalid_shapes_and_mismatched_seed_copies,
    test_triptych_uses_positive_reference_seeds_when_ivd_points_are_absent,
    test_triptych_renders_audited_ivd_isosurface_without_fallback,
    test_triptych_writes_png_counts_and_identical_camera_metadata,
)


TESTS = (
    test_default_descriptor_is_161d_and_rigid_motion_invariant,
    test_descriptor_does_not_depend_on_query_batch_composition_or_chunks,
    test_descriptor_metadata_and_width_are_frozen,
    test_frozen_descriptor_rejects_wrong_tensor_contract,
    test_default_numeric_recipe_matches_legacy_task5_cache_encoder,
    test_binary_metrics_are_tie_aware_and_match_hand_computation,
    test_exhaustive_matcher_matches_direct_all_pair_distances_and_chunks,
    test_fold_completion_hashes_reject_corrupted_artifact,
    test_high_dimensional_matcher_has_exact_self_distance_and_duplicate_tie,
    test_bootstrap_point_estimate_equals_reported_family_timeslice_macro_delta,
    test_completed_report_is_immutable,
    test_empty_class_library_stratum_policy_is_versioned_and_audited,
    test_timeslice_coverage_rejects_missing_cartesian_key,
    test_deterministic_full_svd_pca_is_library_only_and_sign_frozen,
    test_balanced_library_is_stratum_balanced_reproducible_and_leak_free,
    test_frozen_development_config_rejects_sealed_confirmation_access,
    test_cache_validator_checks_every_slice_and_full_contract,
    test_cache_validator_detects_damage_after_first_slice,
    test_cache_validator_rejects_consistent_but_noncanonical_config_hash,
    test_rk4_constant_velocity_matches_analytic_cross_primitive,
    test_multiscale_zero_flow_returns_frozen_7x32_contract_and_physical_scale,
    test_rk4_linear_velocity_matches_exponential_solution_not_euler,
    test_n48_n64_rounded_sampling_and_chunk_order_are_frozen,
    test_percentile_mask_uses_greater_than_or_equal,
    test_percentile_mask_rejects_nonfinite_volume,
    test_solid_body_rotation_has_zero_ivd,
    test_exact_one_nearest_neighbor_and_signed_class_margin,
    test_library_round_trip_without_pickle,
    test_query_descriptor_mismatch_and_nonbinary_labels_are_rejected,
    test_query_does_not_refit_library_normalization,
    test_cross_class_distance_tie_is_deterministically_non_vortex,
    test_dimension_aware_window_load_and_stride,
    test_masked_velocity_fails_closed,
    test_invalid_or_missing_coordinate_fails_closed,
    test_extra_velocity_dimension_requires_explicit_policy,
    test_balanced_assignment_is_reproducible_and_shuffled,
    test_frozen_scale_sets_are_valid_and_disjoint,
    test_scale_table_rejects_insufficient_integration_steps,
    test_scale_table_rejects_duplicate_numeric_tuple,
    test_triptych_writes_png_counts_and_identical_camera_metadata,
    test_triptych_uses_positive_reference_seeds_when_ivd_points_are_absent,
    test_triptych_renders_audited_ivd_isosurface_without_fallback,
    test_scene_validation_rejects_invalid_shapes_and_mismatched_seed_copies,
    test_confusion_masks_are_mutually_exclusive_and_exhaustive,
)


def load_tests(loader, standard_tests, pattern):
    del loader, standard_tests, pattern
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in TESTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
