from __future__ import annotations

from dataclasses import replace

import numpy as np

from pathline_template_matching.paired_scale_center_fusion import (
    DEFAULT_CENTER_COUNT,
    direct_source_centered_diagnostics,
    empirical_midrank,
    fixed_top_fraction_over_centers,
    fuse_paired_scale_centers,
    separate_block_center_predictions,
)


def _expect_value_error(
    callable_object: object,
    *args: object,
    match: str,
    **kwargs: object,
) -> None:
    try:
        callable_object(*args, **kwargs)  # type: ignore[operator]
    except ValueError as error:
        assert match in str(error), (match, str(error))
    else:
        raise AssertionError("expected ValueError")


def test_fixed_top_fraction_uses_all_centers_and_deterministic_ties() -> None:
    scores = np.asarray([0.2, 0.8, 0.8, 0.0, 0.7], dtype=np.float64)
    eligible = np.asarray([True, True, True, True, False], dtype=np.bool_)
    observed = fixed_top_fraction_over_centers(
        scores,
        eligible,
        fraction=0.4,
        require_strictly_positive_score=True,
    )
    np.testing.assert_array_equal(observed, [False, True, True, False, False])


def test_paired_center_fusion_covers_missing_blocks_and_projects_rows() -> None:
    centers = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int64)
    blocks = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    scores = np.asarray([0.9, 0.6, 0.1, 0.5, 0.7, 0.8], dtype=np.float64)
    supported = np.asarray([True, True, False, True, True, True], dtype=np.bool_)
    fusion = fuse_paired_scale_centers(
        centers,
        blocks,
        scores,
        supported,
        weight=0.75,
        center_count=4,
        top_fraction=0.5,
    )
    np.testing.assert_allclose(fusion.paired_score, [0.8, 0.6, 0.7, 0.8])
    np.testing.assert_array_equal(fusion.legacy_valid, [True, True, True, False])
    np.testing.assert_array_equal(fusion.expanded_valid, [True, False, True, True])
    np.testing.assert_array_equal(fusion.combined_eligible, [True, True, True, True])
    # Centers zero and three tie at 0.8; both are selected for top 50%.
    np.testing.assert_array_equal(fusion.prediction, [True, False, False, True])
    np.testing.assert_array_equal(
        fusion.valid_row_prediction, fusion.prediction[centers]
    )
    assert fusion.combined_coverage == 1.0
    np.testing.assert_array_equal(fusion.both_valid, [True, False, True, False])
    np.testing.assert_array_equal(fusion.legacy_only, [False, True, False, False])
    np.testing.assert_array_equal(fusion.expanded_only, [False, False, False, True])
    np.testing.assert_array_equal(fusion.neither_valid, [False, False, False, False])

    legacy, expanded = separate_block_center_predictions(fusion)
    np.testing.assert_array_equal(legacy, [True, True, False, False])
    np.testing.assert_array_equal(expanded, [False, False, True, True])


def test_weight_one_keeps_expanded_only_center_score() -> None:
    fusion = fuse_paired_scale_centers(
        np.asarray([0, 0, 1], dtype=np.int64),
        np.asarray([0, 1, 1], dtype=np.int8),
        np.asarray([0.4, 0.9, 0.7], dtype=np.float64),
        np.asarray([True, True, True], dtype=np.bool_),
        weight=1.0,
        center_count=2,
        top_fraction=0.5,
    )
    np.testing.assert_allclose(fusion.paired_score, [0.4, 0.7])
    np.testing.assert_array_equal(fusion.prediction, [False, True])


def test_weight_zero_keeps_legacy_only_center_score() -> None:
    fusion = fuse_paired_scale_centers(
        np.asarray([0, 1, 1], dtype=np.int64),
        np.asarray([0, 0, 1], dtype=np.int8),
        np.asarray([0.7, 0.4, 0.9], dtype=np.float64),
        np.asarray([True, True, True], dtype=np.bool_),
        weight=0.0,
        center_count=2,
        top_fraction=0.5,
    )
    np.testing.assert_allclose(fusion.paired_score, [0.7, 0.9])
    np.testing.assert_array_equal(fusion.prediction, [False, True])


def test_duplicate_block_center_and_invalid_projection_fail_closed() -> None:
    _expect_value_error(
        lambda: fuse_paired_scale_centers(
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int8),
            np.asarray([0.2, 0.3], dtype=np.float64),
            np.asarray([True, True], dtype=np.bool_),
            weight=0.5,
            center_count=2,
        ),
        match="at most one",
    )

    fusion = fuse_paired_scale_centers(
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int8),
        np.asarray([0.5], dtype=np.float64),
        np.asarray([True], dtype=np.bool_),
        weight=0.5,
        center_count=2,
    )
    bad = fusion.valid_row_prediction.copy()
    bad[0] = ~bad[0]
    _expect_value_error(
        replace,
        fusion,
        match="exact center projection",
        valid_row_prediction=bad,
    )


def test_derived_fusion_arrays_reject_post_construction_tampering() -> None:
    fusion = fuse_paired_scale_centers(
        np.asarray([0, 1, 0, 1], dtype=np.int64),
        np.asarray([0, 0, 1, 1], dtype=np.int8),
        np.asarray([0.8, 0.6, 0.4, 0.9], dtype=np.float64),
        np.asarray([True, True, True, True], dtype=np.bool_),
        weight=0.25,
        center_count=2,
        top_fraction=0.5,
    )

    bad_score = fusion.paired_score.copy()
    bad_score[0] += 0.01
    _expect_value_error(
        replace,
        fusion,
        match="paired_score does not reproduce",
        paired_score=bad_score,
    )

    bad_eligible = fusion.combined_eligible.copy()
    bad_eligible[0] = False
    _expect_value_error(
        replace,
        fusion,
        match="combined eligibility is not the union",
        combined_eligible=bad_eligible,
    )

    bad_valid = fusion.legacy_valid.copy()
    bad_valid[0] = False
    _expect_value_error(
        replace,
        fusion,
        match="legacy_valid does not reproduce valid-row identities",
        legacy_valid=bad_valid,
    )

    duplicate_centers = fusion.valid_row_center_seed_index.copy()
    duplicate_centers[1] = duplicate_centers[0]
    _expect_value_error(
        replace,
        fusion,
        match="duplicate block/center pair",
        valid_row_center_seed_index=duplicate_centers,
    )

    bad_prediction = fusion.prediction.copy()
    bad_prediction[:] = ~bad_prediction
    _expect_value_error(
        replace,
        fusion,
        match="prediction does not reproduce",
        prediction=bad_prediction,
        valid_row_prediction=bad_prediction[fusion.valid_row_center_seed_index],
    )


def test_empirical_midrank_exact_ties() -> None:
    observed = empirical_midrank(np.asarray([3.0, 1.0, 1.0, 2.0], dtype=np.float32))
    # Sorted one-based ranks are 1.5, 1.5, 3, 4; subtract 0.5 and divide by 4.
    np.testing.assert_allclose(observed, [0.875, 0.25, 0.25, 0.625])
    assert observed.dtype == np.dtype(np.float64)
    assert not observed.flags.writeable


def _complete_assigned_population() -> tuple[np.ndarray, ...]:
    centers_one = np.arange(DEFAULT_CENTER_COUNT, dtype=np.int64)
    centers = np.concatenate((centers_one, centers_one))
    blocks = np.concatenate(
        (
            np.zeros(DEFAULT_CENTER_COUNT, dtype=np.int8),
            np.ones(DEFAULT_CENTER_COUNT, dtype=np.int8),
        )
    )
    local_scale = (centers_one // 64).astype(np.int32)
    scales = np.concatenate((local_scale, local_scale + 1000))
    level = ((scales % 1000) // 100).astype(np.float64)
    # Expanded dx is reversed so min-dx selection uses both blocks.
    dx = np.where(blocks == 0, level + 1.0, 10.0 - level).astype(np.float64)
    feature = np.zeros((2 * DEFAULT_CENTER_COUNT, 4), dtype=np.float32)
    feature[:DEFAULT_CENTER_COUNT, 0] = (
        centers_one.astype(np.float32) / DEFAULT_CENTER_COUNT
    )
    feature[DEFAULT_CENTER_COUNT:, 0] = (
        (DEFAULT_CENTER_COUNT - centers_one).astype(np.float32)
        / DEFAULT_CENTER_COUNT
    )
    return centers, blocks, scales, dx, feature


def test_direct_diagnostics_use_all_assigned_rows_and_fixed_top_five_percent() -> None:
    centers, blocks, scales, dx, feature = _complete_assigned_population()
    observed = direct_source_centered_diagnostics(
        centers,
        blocks,
        scales,
        dx,
        feature,
    )
    assert int(observed.min_dx_prediction.sum()) == 3200
    assert int(observed.dx_rank_mean_prediction.sum()) == 3200
    assert np.isfinite(observed.min_dx_centered_curl_score).all()
    assert np.all(
        (observed.dx_rank_mean_score > 0.0)
        & (observed.dx_rank_mean_score < 1.0)
    )
    assert not observed.dx_rank_mean_score.flags.writeable

    bad_scale = scales.copy()
    bad_scale[0] = 100
    _expect_value_error(
        direct_source_centered_diagnostics,
        centers,
        blocks,
        bad_scale,
        dx,
        feature,
        match="dx level 0",
    )

    bad_prediction = observed.min_dx_prediction.copy()
    bad_prediction[0] = ~bad_prediction[0]
    _expect_value_error(
        replace,
        observed,
        match="min-dx prediction does not reproduce",
        min_dx_prediction=bad_prediction,
    )
