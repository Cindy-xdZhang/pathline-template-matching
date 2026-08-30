from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import torch
import unittest

from pathline_template_matching.negative_tail_calibration import (
    CALIBRATION_BLOCK_FALLBACK,
    CALIBRATION_GLOBAL_FALLBACK,
    CALIBRATION_LOCAL_BLOCK_SHRINK,
    CALIBRATION_LOCAL_GLOBAL_SHRINK,
    CALIBRATION_LOCAL_ONLY,
    CALIBRATION_NONE,
    SHRINKAGE_LAMBDA,
    ScaleConditionedNegativeTailCalibrator,
    empirical_upper_tail_probability,
)


@contextmanager
def _raises(*error_types: type[BaseException]):
    try:
        yield
    except error_types:
        return
    raise AssertionError(
        f"expected one of {[error_type.__name__ for error_type in error_types]}"
    )


class _PytestCompatibility:
    raises = staticmethod(_raises)

    @staticmethod
    def skip(message: str = "") -> None:
        raise unittest.SkipTest(message)


pytest = _PytestCompatibility()


def _scale_part(arrays: dict[str, np.ndarray], k: int, scale_id: int) -> np.ndarray:
    offsets = arrays[f"loo_scale_offsets_k_{k}"]
    values = arrays[f"loo_distances_k_{k}"]
    return values[int(offsets[scale_id]) : int(offsets[scale_id + 1])]


def _result_arrays(result):
    return (
        result.raw_distances,
        result.tail_probabilities,
        result.anomaly_scores,
        result.retrieval_supported,
        result.calibration_supported,
        result.calibration_modes,
    )


def _independent_standardized(
    negative: np.ndarray, query: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Test-only scaler implementation independent of production helpers."""

    canonical_negative = np.asarray(negative, dtype=np.float32)
    negative64 = canonical_negative.astype(np.float64)
    mean = negative64.mean(axis=0, dtype=np.float64)
    raw_std = negative64.std(axis=0, dtype=np.float64, ddof=0)
    effective_std = np.where(raw_std < 1.0e-12, 1.0, raw_std)
    scaled_negative = np.asarray(
        (negative64 - mean) / effective_std, dtype=np.float32
    )
    if query is None:
        return scaled_negative, None
    canonical_query = np.asarray(query, dtype=np.float32).astype(np.float64)
    scaled_query = np.asarray(
        (canonical_query - mean) / effective_std, dtype=np.float32
    )
    return scaled_negative, scaled_query


def _independent_loo_reference(
    negative: np.ndarray,
    scale_ids: np.ndarray,
    *,
    scale_id: int,
    k: int,
) -> np.ndarray:
    """Brute-force exact kth LOO reference with an explicit infinite diagonal."""

    scaled, _ = _independent_standardized(negative)
    selected = scaled[np.asarray(scale_ids, dtype=np.int64) == scale_id]
    pairwise = np.linalg.norm(
        selected[:, None, :] - selected[None, :, :], axis=2
    )
    np.fill_diagonal(pairwise, np.inf)
    return np.sort(np.sort(pairwise, axis=1)[:, k - 1])


def _independent_query_k_distance(
    negative: np.ndarray,
    scale_ids: np.ndarray,
    query: np.ndarray,
    *,
    scale_id: int,
    k: int,
) -> np.ndarray:
    """Brute-force exact query kth distance under the fit-negative scaler."""

    scaled_negative, scaled_query = _independent_standardized(negative, query)
    assert scaled_query is not None
    selected = scaled_negative[np.asarray(scale_ids, dtype=np.int64) == scale_id]
    distances = np.linalg.norm(
        scaled_query[:, None, :] - selected[None, :, :], axis=2
    )
    return np.sort(distances, axis=1)[:, k - 1]


def _independent_plus_one_tail(
    reference: np.ndarray, distances: np.ndarray
) -> np.ndarray:
    """Direct count implementation; intentionally does not call production code."""

    reference_values = np.asarray(reference, dtype=np.float64)
    query_values = np.asarray(distances, dtype=np.float64)
    return np.asarray(
        [
            (1.0 + float(np.count_nonzero(reference_values >= distance)))
            / float(len(reference_values) + 1)
            for distance in query_values
        ],
        dtype=np.float64,
    )


def test_empirical_upper_tail_probability_is_plus_one_inclusive_and_conservative():
    reference = np.asarray([1.0, 2.0, 2.0, 4.0], dtype=np.float32)
    distances = np.asarray([0.0, 2.0, 3.0, 5.0], dtype=np.float32)
    probability = empirical_upper_tail_probability(reference, distances)

    np.testing.assert_array_equal(probability, [1.0, 0.8, 0.4, 0.2])
    with pytest.raises(ValueError):
        empirical_upper_tail_probability(reference[::-1], distances)
    with pytest.raises(ValueError):
        empirical_upper_tail_probability([], distances)


def test_exact_loo_explicitly_excludes_only_the_self_row_with_duplicate_features():
    negative = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [5.0, 1.0]],
        dtype=np.float32,
    )
    model = ScaleConditionedNegativeTailCalibrator(
        negative,
        np.full(len(negative), 7, dtype=np.int64),
        ks=(1, 2, 3),
        query_chunk_size=2,
        library_chunk_size=2,
    )
    arrays = model.export_arrays()

    mean = negative.astype(np.float64).mean(axis=0)
    std = negative.astype(np.float64).std(axis=0, ddof=0)
    standardized = ((negative.astype(np.float64) - mean) / std).astype(np.float32)
    pairwise = np.linalg.norm(
        standardized[:, None, :] - standardized[None, :, :], axis=2
    )
    np.fill_diagonal(pairwise, np.inf)
    expected = np.sort(pairwise, axis=1)
    for k in (1, 2, 3):
        np.testing.assert_allclose(
            _scale_part(arrays, k, 7),
            np.sort(expected[:, k - 1]),
            atol=1e-6,
            rtol=1e-6,
        )

    # The two distinct duplicate rows remain valid zero-distance neighbours.
    assert np.count_nonzero(_scale_part(arrays, 1, 7) == 0.0) == 2


def test_all_supported_k_values_share_one_maximum_k_loo_pass_per_scale():
    negative = np.arange(16, dtype=np.float32).reshape(8, 2)
    original_cdist = torch.cdist
    call_count = 0

    def counted_cdist(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_cdist(*args, **kwargs)

    torch.cdist = counted_cdist
    try:
        model = ScaleConditionedNegativeTailCalibrator(
            negative,
            np.full(8, 9, dtype=np.int64),
            ks=(1, 3, 6),
            query_chunk_size=2,
            library_chunk_size=3,
        )
    finally:
        torch.cdist = original_cdist

    # ceil(8/2) query chunks times ceil(8/3) library chunks.  Repeating this
    # for three k values would make 36 calls instead of 12.
    assert call_count == 12
    assert model.ks == (1, 3, 6)
    assert model.fit_audit["loo_reference_counts_by_k"] == {
        "1": 8,
        "3": 8,
        "6": 8,
    }


def test_frozen_default_k_candidates_include_exact_1_5_15_31_references():
    negative = np.column_stack(
        (
            np.linspace(-2.0, 3.0, 32, dtype=np.float32),
            np.linspace(1.0, 5.0, 32, dtype=np.float32) ** 2,
        )
    )
    model = ScaleConditionedNegativeTailCalibrator(
        negative,
        np.full(32, 41, dtype=np.int64),
        query_chunk_size=7,
        library_chunk_size=9,
    )
    result = model.query(
        np.asarray([[0.5, 4.0]], dtype=np.float32), np.asarray([41])
    )

    assert model.ks == (1, 5, 15, 31)
    assert tuple(result.raw_distances) == (1, 5, 15, 31)
    assert model.fit_audit["loo_reference_counts_by_k"] == {
        "1": 32,
        "5": 32,
        "15": 32,
        "31": 32,
    }
    for k in model.ks:
        assert result.retrieval_supported[k][0]
        assert result.calibration_supported[k][0]
        assert result.calibration_modes[k][0] == CALIBRATION_LOCAL_ONLY


def test_local_block_shrink_uses_lambda_64_and_excludes_the_local_scale_from_prior():
    negative = np.asarray(
        [[0.0], [1.0], [4.0], [10.0], [14.0], [20.0]], dtype=np.float32
    )
    scales = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = ScaleConditionedNegativeTailCalibrator(negative, scales, ks=(2,))
    query = np.asarray([[8.0]], dtype=np.float32)
    result = model.query(query, np.asarray([0]), ks=(2,))

    local = _independent_loo_reference(negative, scales, scale_id=0, k=2)
    other_scale = _independent_loo_reference(negative, scales, scale_id=1, k=2)
    distance = _independent_query_k_distance(
        negative, scales, query, scale_id=0, k=2
    )
    local_probability = _independent_plus_one_tail(local, distance)
    prior_probability = _independent_plus_one_tail(other_scale, distance)
    weight = len(local) / (len(local) + SHRINKAGE_LAMBDA)
    expected = weight * local_probability + (1.0 - weight) * prior_probability
    incorrectly_included_current_scale = _independent_plus_one_tail(
        np.concatenate((local, other_scale)), distance
    )
    wrong_expected = (
        weight * local_probability
        + (1.0 - weight) * incorrectly_included_current_scale
    )

    np.testing.assert_allclose(result.raw_distances[2], distance, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(result.tail_probabilities[2], expected)
    np.testing.assert_allclose(result.anomaly_scores[2], 1.0 - expected)
    assert not np.allclose(expected, wrong_expected)
    np.testing.assert_array_equal(
        result.calibration_modes[2], [CALIBRATION_LOCAL_BLOCK_SHRINK]
    )
    assert result.retrieval_supported[2][0]
    assert result.calibration_supported[2][0]


def test_block_fallback_exact_tail_probability_matches_independent_reference():
    negative = np.asarray(
        [[0.0], [1.0], [4.0], [10.0], [12.0]], dtype=np.float32
    )
    scales = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    query = np.asarray([[20.0]], dtype=np.float32)
    model = ScaleConditionedNegativeTailCalibrator(negative, scales, ks=(2,))
    result = model.query(query, np.asarray([1]), ks=(2,))

    reference = _independent_loo_reference(negative, scales, scale_id=0, k=2)
    distance = _independent_query_k_distance(
        negative, scales, query, scale_id=1, k=2
    )
    expected_tail = _independent_plus_one_tail(reference, distance)

    np.testing.assert_allclose(result.raw_distances[2], distance, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(result.tail_probabilities[2], expected_tail)
    np.testing.assert_array_equal(result.anomaly_scores[2], 1.0 - expected_tail)
    np.testing.assert_array_equal(
        result.calibration_modes[2], [CALIBRATION_BLOCK_FALLBACK]
    )


def test_global_fallback_exact_tail_probability_matches_independent_reference():
    negative = np.asarray(
        [[0.0], [1.0], [4.0], [10.0], [12.0]], dtype=np.float32
    )
    scales = np.asarray([0, 0, 0, 1000, 1000], dtype=np.int64)
    query = np.asarray([[20.0]], dtype=np.float32)
    model = ScaleConditionedNegativeTailCalibrator(negative, scales, ks=(2,))
    result = model.query(query, np.asarray([1000]), ks=(2,))

    opposite_block_reference = _independent_loo_reference(
        negative, scales, scale_id=0, k=2
    )
    distance = _independent_query_k_distance(
        negative, scales, query, scale_id=1000, k=2
    )
    expected_tail = _independent_plus_one_tail(opposite_block_reference, distance)

    np.testing.assert_allclose(result.raw_distances[2], distance, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(result.tail_probabilities[2], expected_tail)
    np.testing.assert_array_equal(result.anomaly_scores[2], 1.0 - expected_tail)
    np.testing.assert_array_equal(
        result.calibration_modes[2], [CALIBRATION_GLOBAL_FALLBACK]
    )


def test_local_global_shrink_exact_tail_probability_excludes_current_scale():
    negative = np.asarray(
        [[0.0], [1.0], [4.0], [10.0], [12.0], [18.0]], dtype=np.float32
    )
    scales = np.asarray([0, 0, 0, 1000, 1000, 1000], dtype=np.int64)
    query = np.asarray([[8.0]], dtype=np.float32)
    model = ScaleConditionedNegativeTailCalibrator(negative, scales, ks=(2,))
    result = model.query(query, np.asarray([0]), ks=(2,))

    local = _independent_loo_reference(negative, scales, scale_id=0, k=2)
    opposite_block = _independent_loo_reference(
        negative, scales, scale_id=1000, k=2
    )
    distance = _independent_query_k_distance(
        negative, scales, query, scale_id=0, k=2
    )
    local_tail = _independent_plus_one_tail(local, distance)
    prior_tail = _independent_plus_one_tail(opposite_block, distance)
    weight = len(local) / (len(local) + SHRINKAGE_LAMBDA)
    expected_tail = weight * local_tail + (1.0 - weight) * prior_tail
    incorrectly_included_current_scale = _independent_plus_one_tail(
        np.concatenate((local, opposite_block)), distance
    )
    wrong_tail = (
        weight * local_tail
        + (1.0 - weight) * incorrectly_included_current_scale
    )

    np.testing.assert_allclose(result.raw_distances[2], distance, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(result.tail_probabilities[2], expected_tail)
    np.testing.assert_allclose(result.anomaly_scores[2], 1.0 - expected_tail)
    assert not np.allclose(expected_tail, wrong_tail)
    np.testing.assert_array_equal(
        result.calibration_modes[2], [CALIBRATION_LOCAL_GLOBAL_SHRINK]
    )


def test_local_only_exact_tail_probability_matches_independent_reference():
    negative = np.asarray([[0.0], [1.0], [4.0]], dtype=np.float32)
    scales = np.asarray([0, 0, 0], dtype=np.int64)
    query = np.asarray([[8.0]], dtype=np.float32)
    model = ScaleConditionedNegativeTailCalibrator(negative, scales, ks=(2,))
    result = model.query(query, np.asarray([0]), ks=(2,))

    local_reference = _independent_loo_reference(
        negative, scales, scale_id=0, k=2
    )
    distance = _independent_query_k_distance(
        negative, scales, query, scale_id=0, k=2
    )
    expected_tail = _independent_plus_one_tail(local_reference, distance)

    np.testing.assert_allclose(result.raw_distances[2], distance, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(result.tail_probabilities[2], expected_tail)
    np.testing.assert_array_equal(result.anomaly_scores[2], 1.0 - expected_tail)
    np.testing.assert_array_equal(
        result.calibration_modes[2], [CALIBRATION_LOCAL_ONLY]
    )


def test_retrieval_and_calibration_boundaries_use_frozen_fallback_chain():
    negative = np.asarray(
        [
            [0.0],
            [1.0],
            [3.0],  # scale 0: local k=2 reference
            [10.0],
            [11.0],  # scale 1: n=k, block fallback
            [20.0],
            [22.0],
            [25.0],  # scale 2: second same-block local reference
            [100.0],
            [101.0],  # scale 1000: n=k, global fallback
            [200.0],  # scale 1001: n<k, retrieval unsupported
        ],
        dtype=np.float32,
    )
    scales = np.asarray(
        [0, 0, 0, 1, 1, 2, 2, 2, 1000, 1000, 1001], dtype=np.int64
    )
    model = ScaleConditionedNegativeTailCalibrator(negative, scales, ks=(2,))
    result = model.query(
        np.asarray([[2.0], [10.5], [100.5], [200.0]], dtype=np.float32),
        np.asarray([0, 1, 1000, 1001], dtype=np.int64),
        ks=(2,),
    )

    np.testing.assert_array_equal(
        result.retrieval_supported[2], [True, True, True, False]
    )
    np.testing.assert_array_equal(
        result.calibration_supported[2], [True, True, True, False]
    )
    np.testing.assert_array_equal(
        result.calibration_modes[2],
        [
            CALIBRATION_LOCAL_BLOCK_SHRINK,
            CALIBRATION_BLOCK_FALLBACK,
            CALIBRATION_GLOBAL_FALLBACK,
            CALIBRATION_NONE,
        ],
    )
    assert np.isnan(result.raw_distances[2][-1])
    assert result.tail_probabilities[2][-1] == 1.0
    assert result.anomaly_scores[2][-1] == 0.0


def test_local_global_shrink_and_local_only_modes_are_reachable_without_labels():
    two_blocks = ScaleConditionedNegativeTailCalibrator(
        np.asarray([[0.0], [1.0], [3.0], [10.0], [12.0], [15.0]], dtype=np.float32),
        np.asarray([0, 0, 0, 1000, 1000, 1000], dtype=np.int64),
        ks=(2,),
    )
    result = two_blocks.query(
        np.asarray([[2.0], [13.0]], dtype=np.float32),
        np.asarray([0, 1000], dtype=np.int64),
        ks=(2,),
    )
    np.testing.assert_array_equal(
        result.calibration_modes[2],
        [CALIBRATION_LOCAL_GLOBAL_SHRINK, CALIBRATION_LOCAL_GLOBAL_SHRINK],
    )

    one_scale = ScaleConditionedNegativeTailCalibrator(
        np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32),
        np.asarray([0, 0, 0], dtype=np.int64),
        ks=(2,),
    )
    local_only = one_scale.query(
        np.asarray([[2.0]], dtype=np.float32), np.asarray([0]), ks=(2,)
    )
    np.testing.assert_array_equal(
        local_only.calibration_modes[2], [CALIBRATION_LOCAL_ONLY]
    )


def test_no_reference_keeps_n_equals_k_retrieved_but_calibration_unsupported():
    model = ScaleConditionedNegativeTailCalibrator(
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([17, 17], dtype=np.int64),
        ks=(2,),
    )
    result = model.query(
        np.asarray([[0.5]], dtype=np.float32), np.asarray([17]), ks=(2,)
    )

    assert result.retrieval_supported[2][0]
    assert np.isfinite(result.raw_distances[2][0])
    assert not result.calibration_supported[2][0]
    assert result.calibration_modes[2][0] == CALIBRATION_NONE
    assert result.tail_probabilities[2][0] == 1.0
    assert result.anomaly_scores[2][0] == 0.0


def test_fit_and_query_are_invariant_to_distance_chunk_sizes_and_query_membership():
    generator = np.random.default_rng(26091)
    negative = generator.normal(size=(24, 4)).astype(np.float32)
    scales = np.repeat(np.asarray([3, 4, 1003]), 8)
    small = ScaleConditionedNegativeTailCalibrator(
        negative,
        scales,
        ks=(1, 3, 5),
        query_chunk_size=1,
        library_chunk_size=1,
    )
    large = ScaleConditionedNegativeTailCalibrator(
        negative,
        scales,
        ks=(1, 3, 5),
        query_chunk_size=100,
        library_chunk_size=100,
    )
    for name, values in small.export_arrays().items():
        np.testing.assert_array_equal(values, large.export_arrays()[name])

    query = generator.normal(size=(9, 4)).astype(np.float32)
    query_scales = np.asarray([3, 4, 1003, 3, 1003, 4, 3, 4, 1003])
    together = small.query(
        query,
        query_scales,
        ks=(1, 5),
        query_chunk_size=1,
        library_chunk_size=1,
    )
    one_chunk = large.query(
        query,
        query_scales,
        ks=(1, 5),
        query_chunk_size=100,
        library_chunk_size=100,
    )
    for together_mapping, large_mapping in zip(
        _result_arrays(together), _result_arrays(one_chunk)
    ):
        for k in together_mapping:
            np.testing.assert_array_equal(together_mapping[k], large_mapping[k])

    for row in range(len(query)):
        alone = small.query(
            query[row : row + 1], query_scales[row : row + 1], ks=(1, 5)
        )
        for together_mapping, alone_mapping in zip(
            _result_arrays(together), _result_arrays(alone)
        ):
            for k in together_mapping:
                np.testing.assert_array_equal(
                    together_mapping[k][row : row + 1], alone_mapping[k]
                )


def test_export_is_pure_arrays_and_allow_pickle_false_round_trip_is_exact(tmp_path):
    generator = np.random.default_rng(7331)
    negative = generator.normal(size=(18, 3)).astype(np.float32)
    scales = np.repeat(np.asarray([2, 6, 1004]), 6)
    model = ScaleConditionedNegativeTailCalibrator(
        negative, scales, ks=(1, 3, 5), query_chunk_size=2, library_chunk_size=3
    )
    exported = model.export_arrays()
    assert exported
    assert all(isinstance(value, np.ndarray) for value in exported.values())
    assert all(value.dtype.kind not in {"O", "U", "S"} for value in exported.values())

    destination = tmp_path / "tail_calibration.npz"
    np.savez_compressed(destination, **exported)
    with np.load(destination, allow_pickle=False) as archive:
        restored = ScaleConditionedNegativeTailCalibrator.from_arrays(
            {name: archive[name] for name in archive.files}
        )

    query = generator.normal(size=(7, 3)).astype(np.float32)
    query_scales = np.asarray([2, 6, 1004, 2, 1004, 6, 2])
    original_result = model.query(query, query_scales, ks=(5, 1))
    restored_result = restored.query(query, query_scales, ks=(5, 1))
    for original_mapping, restored_mapping in zip(
        _result_arrays(original_result), _result_arrays(restored_result)
    ):
        for k in original_mapping:
            np.testing.assert_array_equal(original_mapping[k], restored_mapping[k])
    for name, values in exported.items():
        np.testing.assert_array_equal(values, restored.export_arrays()[name])


def test_reconstruction_rejects_tampered_offsets_and_lambda():
    model = ScaleConditionedNegativeTailCalibrator(
        np.arange(8, dtype=np.float32).reshape(4, 2),
        np.full(4, 3, dtype=np.int64),
        ks=(1, 3),
    )
    arrays = model.export_arrays()

    bad_offsets = {name: value.copy() for name, value in arrays.items()}
    bad_offsets["loo_scale_offsets_k_1"][4] += 1
    with pytest.raises(ValueError):
        ScaleConditionedNegativeTailCalibrator.from_arrays(bad_offsets)

    bad_lambda = {name: value.copy() for name, value in arrays.items()}
    bad_lambda["shrinkage_lambda"] = np.asarray(63.0, dtype=np.float64)
    with pytest.raises(ValueError):
        ScaleConditionedNegativeTailCalibrator.from_arrays(bad_lambda)

    impossible_empty = {name: value.copy() for name, value in arrays.items()}
    impossible_empty["negative_features"] = np.empty((0, 2), dtype=np.float32)
    impossible_empty["negative_scale_offsets"] = np.zeros(2001, dtype=np.int64)
    for k in model.ks:
        impossible_empty[f"loo_distances_k_{k}"] = np.empty(0, dtype=np.float32)
        impossible_empty[f"loo_scale_offsets_k_{k}"] = np.zeros(
            2001, dtype=np.int64
        )
    with pytest.raises(ValueError):
        ScaleConditionedNegativeTailCalibrator.from_arrays(impossible_empty)


def test_inputs_and_outputs_are_copied_and_result_arrays_are_read_only():
    negative = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32)
    scales = np.asarray([0, 0, 0], dtype=np.int64)
    model = ScaleConditionedNegativeTailCalibrator(negative, scales, ks=(1,))
    before = model.fit_audit
    negative[:] = -999.0
    scales[:] = 1999
    result = model.query(np.asarray([[2.0]], dtype=np.float32), [0], ks=(1,))

    assert model.fit_audit == before
    assert result.calibration_supported[1][0]
    for mapping in _result_arrays(result):
        with pytest.raises(ValueError):
            mapping[1][0] = 0


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        ScaleConditionedNegativeTailCalibrator(
            np.ones((2, 2)), np.asarray([0, 2000]), ks=(1,)
        )
    with pytest.raises(ValueError):
        ScaleConditionedNegativeTailCalibrator(
            np.ones((2, 2)), np.asarray([0, 0]), ks=(1,), shrinkage_lambda=32
        )

    model = ScaleConditionedNegativeTailCalibrator(
        np.ones((2, 2)), np.asarray([0, 0]), ks=(1,)
    )
    with pytest.raises(ValueError):
        model.query(np.ones((1, 3)), np.asarray([0]))
    with pytest.raises(ValueError):
        model.query(np.ones((1, 2)), np.asarray([2000]))
    with pytest.raises(ValueError):
        model.query(np.ones((1, 2)), np.asarray([0]), ks=(5,))


def test_cuda_fit_and_query_match_cpu_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = np.random.default_rng(81731)
    negative = generator.normal(size=(36, 5)).astype(np.float32)
    scales = np.repeat(np.asarray([3, 8, 1003]), 12)
    cpu_model = ScaleConditionedNegativeTailCalibrator(
        negative, scales, ks=(1, 5, 10), device="cpu"
    )
    cuda_model = ScaleConditionedNegativeTailCalibrator(
        negative,
        scales,
        ks=(1, 5, 10),
        device="cuda",
        query_chunk_size=3,
        library_chunk_size=7,
    )
    cuda_one_chunk_model = ScaleConditionedNegativeTailCalibrator(
        negative,
        scales,
        ks=(1, 5, 10),
        device="cuda",
        query_chunk_size=100,
        library_chunk_size=100,
    )
    for name, cpu_values in cpu_model.export_arrays().items():
        cuda_values = cuda_model.export_arrays()[name]
        one_chunk_values = cuda_one_chunk_model.export_arrays()[name]
        if cpu_values.dtype.kind == "f":
            np.testing.assert_allclose(cuda_values, cpu_values, atol=2e-6, rtol=2e-6)
            np.testing.assert_array_equal(one_chunk_values, cuda_values)
        else:
            np.testing.assert_array_equal(cuda_values, cpu_values)
            np.testing.assert_array_equal(one_chunk_values, cuda_values)
    query = generator.normal(size=(13, 5)).astype(np.float32)
    query_scales = np.resize(np.asarray([8, 3, 1003]), len(query))
    cpu = cpu_model.query(query, query_scales, ks=(1, 5, 10), device="cpu")
    cuda = cuda_model.query(
        query,
        query_scales,
        ks=(1, 5, 10),
        device="cuda",
        query_chunk_size=2,
        library_chunk_size=5,
    )
    cuda_one_chunk = cuda_one_chunk_model.query(
        query,
        query_scales,
        ks=(1, 5, 10),
        device="cuda",
        query_chunk_size=100,
        library_chunk_size=100,
    )

    for cpu_mapping, cuda_mapping, one_chunk_mapping in zip(
        _result_arrays(cpu), _result_arrays(cuda), _result_arrays(cuda_one_chunk)
    ):
        for k in cpu_mapping:
            if cpu_mapping[k].dtype.kind == "f":
                np.testing.assert_allclose(
                    cuda_mapping[k], cpu_mapping[k], atol=2e-6, rtol=2e-6
                )
                np.testing.assert_array_equal(one_chunk_mapping[k], cuda_mapping[k])
            else:
                np.testing.assert_array_equal(cuda_mapping[k], cpu_mapping[k])
                np.testing.assert_array_equal(one_chunk_mapping[k], cuda_mapping[k])


def test_cuda_duplicate_self_exclusion_and_equal_tail_ties_match_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    negative = np.asarray([[0.0], [0.0], [2.0], [4.0]], dtype=np.float32)
    scales = np.full(4, 7, dtype=np.int64)
    cpu_model = ScaleConditionedNegativeTailCalibrator(
        negative,
        scales,
        ks=(1, 2, 3),
        device="cpu",
        query_chunk_size=2,
        library_chunk_size=2,
    )
    cuda_model = ScaleConditionedNegativeTailCalibrator(
        negative,
        scales,
        ks=(1, 2, 3),
        device="cuda",
        query_chunk_size=1,
        library_chunk_size=3,
    )

    cpu_arrays = cpu_model.export_arrays()
    cuda_arrays = cuda_model.export_arrays()
    for k in (1, 2, 3):
        expected_reference = _independent_loo_reference(
            negative, scales, scale_id=7, k=k
        )
        cpu_reference = _scale_part(cpu_arrays, k, 7)
        cuda_reference = _scale_part(cuda_arrays, k, 7)
        np.testing.assert_allclose(
            cpu_reference, expected_reference, atol=2e-6, rtol=2e-6
        )
        np.testing.assert_allclose(
            cuda_reference, expected_reference, atol=2e-6, rtol=2e-6
        )

    query = np.asarray([[2.0], [4.0]], dtype=np.float32)
    query_scales = np.asarray([7, 7], dtype=np.int64)
    cpu = cpu_model.query(
        query,
        query_scales,
        ks=(1, 2, 3),
        device="cpu",
        query_chunk_size=2,
        library_chunk_size=2,
    )
    cuda = cuda_model.query(
        query,
        query_scales,
        ks=(1, 2, 3),
        device="cuda",
        query_chunk_size=1,
        library_chunk_size=3,
    )
    expected_k2_distance = _independent_query_k_distance(
        negative, scales, query, scale_id=7, k=2
    )
    expected_k2_reference = _independent_loo_reference(
        negative, scales, scale_id=7, k=2
    )
    expected_k2_tail = _independent_plus_one_tail(
        expected_k2_reference, expected_k2_distance
    )
    # Each query k=2 distance equals at least three reference entries, so this
    # explicitly exercises the inclusive >= equal-tail convention.
    for distance in expected_k2_distance:
        assert np.count_nonzero(
            np.isclose(expected_k2_reference, distance, atol=2e-6, rtol=2e-6)
        ) >= 3
    np.testing.assert_array_equal(expected_k2_tail, np.ones(2, dtype=np.float64))
    np.testing.assert_allclose(
        cpu.raw_distances[2], expected_k2_distance, atol=2e-6, rtol=2e-6
    )
    np.testing.assert_array_equal(cpu.tail_probabilities[2], expected_k2_tail)
    np.testing.assert_array_equal(cpu.anomaly_scores[2], 1.0 - expected_k2_tail)

    for cpu_mapping, cuda_mapping in zip(_result_arrays(cpu), _result_arrays(cuda)):
        for k in cpu_mapping:
            if cpu_mapping[k].dtype.kind == "f":
                np.testing.assert_allclose(
                    cuda_mapping[k], cpu_mapping[k], atol=2e-6, rtol=2e-6
                )
            else:
                np.testing.assert_array_equal(cuda_mapping[k], cpu_mapping[k])
