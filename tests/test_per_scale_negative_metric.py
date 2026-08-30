from __future__ import annotations

import numpy as np

from pathline_template_matching.per_scale_negative_metric import (
    SCALER_ARRAY_NAMES,
    SCALER_LOCAL_BLOCK_SHRINK,
    SCALER_LOCAL_GLOBAL_SHRINK,
    SCALER_LOCAL_ONLY,
    SCALER_NO_LOCAL_ROWS,
    PerScaleNegativeScaler,
    PerScaleNegativeTailModel,
)


def _expect_value_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _independent_population_variance(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    mean = numeric.mean(axis=0, dtype=np.float64)
    return np.square(numeric - mean).sum(axis=0, dtype=np.float64) / len(numeric)


def test_per_scale_scaler_matches_frozen_ddof0_prior_and_variance_shrinkage():
    features = np.asarray(
        [
            [0.1, 100.0],
            [1.7, 102.0],
            [4.2, 104.0],
            [1000.2, -20.0],
            [1003.9, -16.0],
            [-400.5, 7.0],
        ],
        dtype=np.float32,
    )
    scales = np.asarray([0, 0, 0, 1, 1, 1000], dtype=np.int64)
    scaler = PerScaleNegativeScaler(features, scales)
    arrays = scaler.export_arrays()

    assert tuple(arrays) == SCALER_ARRAY_NAMES
    assert arrays["local_row_count_int64"].tolist()[:2] == [3, 2]
    assert arrays["local_row_count_int64"][1000] == 1
    local0 = _independent_population_variance(features[:3])
    local1 = _independent_population_variance(features[3:5])
    local1000 = _independent_population_variance(features[5:6])
    np.testing.assert_array_equal(arrays["local_variance_float64"][0], local0)
    np.testing.assert_array_equal(arrays["local_variance_float64"][1], local1)
    np.testing.assert_array_equal(
        arrays["local_variance_float64"][1000], local1000
    )

    # Other-scale means are removed before pooling, so the enormous mean gap
    # between scale 0 and scale 1 never enters either prior.
    np.testing.assert_array_equal(arrays["prior_variance_float64"][0], local1)
    np.testing.assert_array_equal(arrays["prior_variance_float64"][1], local0)
    pooled_block0 = (3.0 * local0 + 2.0 * local1) / 5.0
    np.testing.assert_array_equal(
        arrays["prior_variance_float64"][1000], pooled_block0
    )
    expected0 = (3.0 / 67.0) * local0 + (64.0 / 67.0) * local1
    np.testing.assert_array_equal(arrays["shrunk_variance_float64"][0], expected0)
    np.testing.assert_array_equal(
        arrays["effective_std_float64"][0], np.sqrt(expected0)
    )
    assert arrays["scaler_mode_int8"][0] == SCALER_LOCAL_BLOCK_SHRINK
    assert arrays["scaler_mode_int8"][1] == SCALER_LOCAL_BLOCK_SHRINK
    assert arrays["scaler_mode_int8"][1000] == SCALER_LOCAL_GLOBAL_SHRINK
    assert arrays["scaler_mode_int8"][999] == SCALER_NO_LOCAL_ROWS


def test_per_scale_scaler_local_only_floor_and_absent_placeholders_are_exact():
    scaler = PerScaleNegativeScaler(
        np.asarray([[3.0, -2.0]], dtype=np.float32), np.asarray([17])
    )
    arrays = scaler.export_arrays()
    assert arrays["scaler_mode_int8"][17] == SCALER_LOCAL_ONLY
    np.testing.assert_array_equal(arrays["prior_variance_float64"][17], [0.0, 0.0])
    np.testing.assert_array_equal(arrays["shrunk_variance_float64"][17], [0.0, 0.0])
    np.testing.assert_array_equal(arrays["effective_std_float64"][17], [1.0, 1.0])
    assert arrays["local_support_bool"][18] == np.bool_(False)
    np.testing.assert_array_equal(arrays["local_mean_float64"][18], [0.0, 0.0])
    np.testing.assert_array_equal(arrays["effective_std_float64"][18], [1.0, 1.0])


def test_per_scale_scaler_nonintegral_variance_round_trip_is_bitwise_exact():
    features = np.asarray(
        [[0.1, 1.0], [0.2, 4.0], [0.4, 9.0], [8.3, -1.0], [9.7, 2.0]],
        dtype=np.float32,
    )
    scales = np.asarray([2, 2, 2, 7, 7], dtype=np.int64)
    original = PerScaleNegativeScaler(features, scales)
    exported = original.export_arrays()
    restored = PerScaleNegativeScaler.from_arrays(exported)
    for name in SCALER_ARRAY_NAMES:
        np.testing.assert_array_equal(exported[name], restored.export_arrays()[name])
    query = np.asarray([[0.33, 3.7], [9.2, 0.1]], dtype=np.float32)
    np.testing.assert_array_equal(
        original.transform(query, [2, 7]), restored.transform(query, [2, 7])
    )


def test_per_scale_scaler_rejects_prior_shrink_and_effective_std_tamper():
    scaler = PerScaleNegativeScaler(
        np.asarray([[0.0], [1.0], [3.0], [10.0], [14.0]], dtype=np.float32),
        np.asarray([0, 0, 0, 1, 1]),
    )
    arrays = scaler.export_arrays()
    for name in (
        "prior_variance_float64",
        "shrunk_variance_float64",
        "effective_std_float64",
    ):
        tampered = {key: value.copy() for key, value in arrays.items()}
        tampered[name][0, 0] += 0.125
        _expect_value_error(PerScaleNegativeScaler.from_arrays, tampered)
    local_only = PerScaleNegativeScaler(
        np.asarray([[1.0], [2.0]], dtype=np.float32), np.asarray([9, 9])
    ).export_arrays()
    local_only["prior_variance_float64"][9, 0] = 1.0
    _expect_value_error(PerScaleNegativeScaler.from_arrays, local_only)


def test_no_local_transform_never_reads_poisoned_placeholders_and_fails_closed():
    valid = PerScaleNegativeScaler(
        np.asarray([[0.0], [1.0]], dtype=np.float32), np.asarray([0, 0])
    )
    arrays = valid.export_arrays()
    # Build an intentionally nonserializable in-memory audit object whose
    # absent placeholders would yield NaN if transform indexed them.
    means = arrays["local_mean_float64"].copy()
    std = arrays["effective_std_float64"].copy()
    means[77, 0] = np.nan
    std[77, 0] = np.nan
    poisoned = PerScaleNegativeScaler.__new__(PerScaleNegativeScaler)
    poisoned._install_state(
        shrinkage_lambda=64.0,
        scale_ids=arrays["scale_id_int32"],
        local_counts=arrays["local_row_count_int64"],
        block_other_counts=arrays["block_other_row_count_int64"],
        global_other_counts=arrays["global_other_row_count_int64"],
        support=arrays["local_support_bool"],
        modes=arrays["scaler_mode_int8"],
        means=means,
        local_variances=arrays["local_variance_float64"],
        prior_variances=arrays["prior_variance_float64"],
        shrunk_variances=arrays["shrunk_variance_float64"],
        effective_std=std,
    )
    np.testing.assert_array_equal(
        poisoned.transform(np.asarray([[1.0e30]], dtype=np.float32), [77]),
        np.asarray([[0.0]], dtype=np.float32),
    )

    model = PerScaleNegativeTailModel(
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([0, 0]),
        ks=(1,),
    )
    result = model.query(np.asarray([[1.0e30]], dtype=np.float32), [77], ks=(1,))
    assert not result.retrieval_supported[1][0]
    assert not result.calibration_supported[1][0]
    assert np.isnan(result.raw_distances[1][0])
    assert result.tail_probabilities[1][0] == 1.0
    assert result.anomaly_scores[1][0] == 0.0


def test_composite_model_uses_exact_scale_std_without_second_global_weighting():
    negative = np.asarray(
        [[0.0, 0.0], [2.0, 10.0], [10.0, 1000.0], [14.0, 1002.0]],
        dtype=np.float32,
    )
    scales = np.asarray([0, 0, 1, 1], dtype=np.int64)
    model = PerScaleNegativeTailModel(negative, scales, ks=(1,))
    tail_arrays = model.tail_calibrator.export_arrays()
    np.testing.assert_array_equal(tail_arrays["mean"], np.zeros(2))
    np.testing.assert_array_equal(tail_arrays["raw_std"], np.ones(2))
    np.testing.assert_array_equal(tail_arrays["effective_std"], np.ones(2))
    assert not tail_arrays["zero_variance_feature_mask"].any()

    query = np.asarray([[1.0, 7.0]], dtype=np.float32)
    transformed_negative = model.scaler.transform(negative[:2], [0, 0])
    transformed_query = model.scaler.transform(query, [0])
    expected = np.min(
        np.linalg.norm(
            transformed_query[:, None, :] - transformed_negative[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    result = model.query(query, [0], ks=(1,))
    np.testing.assert_allclose(result.raw_distances[1], expected, rtol=1e-6, atol=1e-6)


def test_composite_artifact_reconstruction_rejects_tail_lambda_or_count_tamper():
    model = PerScaleNegativeTailModel(
        np.asarray([[0.0], [1.0], [3.0], [10.0]], dtype=np.float32),
        np.asarray([0, 0, 0, 1]),
        ks=(1,),
    )
    scaler_arrays = model.scaler.export_arrays()
    tail_arrays = model.tail_calibrator.export_arrays()
    restored = PerScaleNegativeTailModel.from_artifacts(scaler_arrays, tail_arrays)
    assert restored.ks == (1,)

    bad_lambda = {name: value.copy() for name, value in tail_arrays.items()}
    bad_lambda["shrinkage_lambda"] = np.asarray(63.0, dtype=np.float64)
    _expect_value_error(
        PerScaleNegativeTailModel.from_artifacts, scaler_arrays, bad_lambda
    )
    bad_count = {name: value.copy() for name, value in tail_arrays.items()}
    bad_count["negative_scale_offsets"][2:] += 1
    _expect_value_error(
        PerScaleNegativeTailModel.from_artifacts, scaler_arrays, bad_count
    )


def test_same_scale_mean_cancels_and_query_membership_is_invariant():
    model = PerScaleNegativeTailModel(
        np.asarray([[100.0], [102.0], [105.0], [-50.0], [-47.0], [-41.0]], dtype=np.float32),
        np.asarray([4, 4, 4, 1004, 1004, 1004]),
        ks=(1,),
        query_chunk_size=1,
        library_chunk_size=1,
    )
    target = np.asarray([[101.0]], dtype=np.float32)
    alone = model.query(target, [4], ks=(1,), query_chunk_size=1, library_chunk_size=1)
    together = model.query(
        np.asarray([[101.0], [1.0e6]], dtype=np.float32),
        [4, 4],
        ks=(1,),
        query_chunk_size=8,
        library_chunk_size=8,
    )
    np.testing.assert_array_equal(alone.raw_distances[1], together.raw_distances[1][:1])
    np.testing.assert_array_equal(alone.tail_probabilities[1], together.tail_probabilities[1][:1])
