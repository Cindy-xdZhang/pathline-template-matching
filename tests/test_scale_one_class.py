from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from pathline_template_matching.scale_one_class import ScaleConditionedNegativeKNN


@contextmanager
def _assert_raises(*error_types: type[BaseException]):
    try:
        yield
    except error_types:
        return
    raise AssertionError(f"expected one of {[item.__name__ for item in error_types]}")


def test_query_never_crosses_scale():
    model = ScaleConditionedNegativeKNN(
        np.asarray([[0.0], [1.0], [100.0], [101.0]], dtype=np.float32),
        np.asarray([10, 10, 20, 20], dtype=np.int64),
    )
    result = model.query(
        np.asarray([[100.0], [0.0]], dtype=np.float32),
        np.asarray([10, 20], dtype=np.int64),
        ks=(1, 2),
    )

    # Each query has an exact or near-exact feature match in the other scale,
    # but exact-scale isolation forces both nearest distances to remain large.
    assert result[1][0] > 1.0
    assert result[1][1] > 1.0
    assert np.all(result[2] > result[1])


def test_missing_or_insufficient_scale_fails_before_any_distance_calculation():
    model = ScaleConditionedNegativeKNN(
        np.asarray([[0.0], [100.0], [101.0]], dtype=np.float32),
        np.asarray([10, 20, 20], dtype=np.int64),
    )
    original_cdist = torch.cdist
    distance_called = False

    def forbidden_cdist(*args, **kwargs):
        nonlocal distance_called
        distance_called = True
        raise AssertionError("distance calculation must not begin")

    torch.cdist = forbidden_cdist
    try:
        try:
            model.query(
                np.asarray([[0.0], [50.0]], dtype=np.float32),
                np.asarray([10, 30], dtype=np.int64),
                ks=(1, 2),
            )
        except ValueError as error:
            message = str(error)
        else:
            raise AssertionError("insufficient scale support must raise ValueError")
    finally:
        torch.cdist = original_cdist

    assert not distance_called
    assert "scale=10, available=1, required_max_k=2" in message
    assert "scale=30, available=0, required_max_k=2" in message


def test_k_semantics_and_one_pass_results_match_exact_order_statistics():
    negative = np.asarray([[0.0], [2.0], [5.0], [20.0]], dtype=np.float64)
    model = ScaleConditionedNegativeKNN(negative, np.asarray([4, 4, 4, 9]))
    query = np.asarray([[1.0], [4.0]], dtype=np.float64)
    result = model.query(query, np.asarray([4, 4]), ks=(3, 1, 2))

    mean = negative.mean(axis=0)
    std = negative.std(axis=0)
    scaled_negative = (negative[:3] - mean) / std
    scaled_query = (query - mean) / std
    expected = np.sort(
        np.abs(scaled_query[:, None, 0] - scaled_negative[None, :, 0]), axis=1
    )
    assert tuple(result) == (3, 1, 2)
    for k in (1, 2, 3):
        np.testing.assert_allclose(result[k], expected[:, k - 1], atol=1e-6, rtol=1e-6)


def test_multiple_k_values_share_each_chunk_distance_calculation():
    negative = np.arange(14, dtype=np.float32).reshape(7, 2)
    model = ScaleConditionedNegativeKNN(negative, np.full(7, 4, dtype=np.int64))
    query = np.arange(10, dtype=np.float32).reshape(5, 2) + 0.25
    original_cdist = torch.cdist
    call_count = 0

    def counted_cdist(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_cdist(*args, **kwargs)

    torch.cdist = counted_cdist
    try:
        result = model.query(
            query,
            np.full(5, 4, dtype=np.int64),
            ks=(1, 3, 6),
            query_chunk_size=2,
            library_chunk_size=3,
        )
    finally:
        torch.cdist = original_cdist

    # ceil(5 / 2) query chunks * ceil(7 / 3) library chunks.  Three separate
    # k searches would make 27 calls rather than the observed nine.
    assert call_count == 9
    assert tuple(result) == (1, 3, 6)


def test_query_is_invariant_to_both_chunk_sizes():
    generator = np.random.default_rng(6203)
    negative = generator.normal(size=(31, 5)).astype(np.float32)
    negative_scales = np.repeat(np.asarray([2, 7, 11]), [9, 11, 11])
    query = generator.normal(size=(17, 5)).astype(np.float32)
    query_scales = np.resize(np.asarray([11, 2, 7]), len(query))
    model = ScaleConditionedNegativeKNN(negative, negative_scales)

    small = model.query(
        query,
        query_scales,
        ks=(1, 5),
        query_chunk_size=1,
        library_chunk_size=1,
    )
    large = model.query(
        query,
        query_scales,
        ks=(1, 5),
        query_chunk_size=100,
        library_chunk_size=100,
    )
    for k in small:
        np.testing.assert_array_equal(small[k], large[k])


def test_query_never_changes_training_scaler_or_fit_audit():
    negative = np.asarray(
        [[0.0, 0.0, 4.0], [2.0, 5.0e-13, 4.0], [10.0, 1.0e-12, 4.0]]
    )
    model = ScaleConditionedNegativeKNN(
        negative,
        np.asarray([1, 1, 2]),
        negative_family_ids=np.asarray(["a", "a", "b"]),
    )
    before = model.fit_audit
    first_alone = model.query(
        np.asarray([[1.0, 2.5e-13, 4.0]]), np.asarray([1]), ks=(1,)
    )[1]
    first_in_shifted_batch = model.query(
        np.asarray(
            [[1.0, 2.5e-13, 4.0], [1.0e6, -1.0e-6, -1.0e6]]
        ),
        np.asarray([1, 1]),
        ks=(1,),
    )[1][:1]
    after = model.fit_audit

    np.testing.assert_array_equal(first_alone, first_in_shifted_batch)
    assert before == after
    assert before["count"] == 3
    assert before["dim"] == 3
    assert before["canonical_feature_dtype"] == "float32"
    assert before["scaler_statistics_dtype"] == "float64"
    assert before["distance_dtype"] == "float32"
    assert before["std_ddof"] == 0
    assert before["scales"] == (1, 2)
    assert before["scale_counts"] == {1: 2, 2: 1}
    np.testing.assert_allclose(before["mean"], [4.0, 5.0e-13, 4.0])
    # Both a near-zero-variance feature and a constant feature use effective
    # standard deviation one, matching the parent matcher threshold.
    assert 0.0 < before["raw_std"][1] < 1.0e-12
    assert before["raw_std"][2] == 0.0
    assert before["effective_std"][1:] == [1.0, 1.0]
    assert before["std_floor_exclusive"] == 1.0e-12
    assert before["zero_variance_feature_mask"] == [False, True, True]
    assert before["family_counts"] == {"a": 2, "b": 1}


def test_fit_uses_every_natural_negative_once_and_copies_caller_inputs():
    negative = np.asarray(
        [[0.0, 1.0], [2.0, 3.0], [100.0, 101.0], [200.0, 201.0]],
        dtype=np.float32,
    )
    scales = np.asarray([1, 1, 2, 2], dtype=np.int64)
    families = np.asarray(["a", "a", "b", "c"])
    expected_mean = negative.astype(np.float64).mean(axis=0)
    expected_raw_std = negative.astype(np.float64).std(axis=0, ddof=0)
    model = ScaleConditionedNegativeKNN(negative, scales, families)

    negative[:] = -999.0
    scales[:] = 99
    families[:] = "z"
    audit = model.fit_audit

    assert audit["count"] == 4
    assert audit["scale_counts"] == {1: 2, 2: 2}
    assert audit["family_counts"] == {"a": 2, "b": 1, "c": 1}
    np.testing.assert_array_equal(audit["mean"], expected_mean)
    np.testing.assert_array_equal(audit["raw_std"], expected_raw_std)
    np.testing.assert_array_equal(audit["effective_std"], expected_raw_std)


def test_query_batch_membership_order_and_fit_audit_are_invariant():
    generator = np.random.default_rng(943)
    negative = generator.normal(size=(24, 4)).astype(np.float32)
    negative_scales = np.repeat(np.asarray([3, 8]), 12)
    query = generator.normal(size=(9, 4)).astype(np.float32)
    query_scales = np.asarray([8, 3, 8, 8, 3, 3, 8, 3, 8])
    model = ScaleConditionedNegativeKNN(negative, negative_scales)
    before = model.fit_audit

    together = model.query(query, query_scales, ks=(1, 5))
    permutation = np.asarray([8, 0, 4, 2, 6, 1, 7, 5, 3])
    permuted = model.query(query[permutation], query_scales[permutation], ks=(1, 5))
    inverse = np.argsort(permutation)
    for k in together:
        np.testing.assert_array_equal(together[k], permuted[k][inverse])
        for row in range(len(query)):
            alone = model.query(query[row : row + 1], query_scales[row : row + 1], ks=(k,))
            np.testing.assert_array_equal(together[k][row : row + 1], alone[k])
    assert model.fit_audit == before


def test_deterministic_algorithm_setting_is_restored_after_cpu_query():
    model = ScaleConditionedNegativeKNN(
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.ones(6, dtype=np.int64),
    )
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        first = model.query(np.asarray([[0.5, 1.5]], dtype=np.float32), [1], ks=(1, 3))
        second = model.query(np.asarray([[0.5, 1.5]], dtype=np.float32), [1], ks=(1, 3))
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        for k in first:
            np.testing.assert_array_equal(first[k], second[k])
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )


def test_cuda_matches_cpu_when_available():
    if not torch.cuda.is_available():
        return
    generator = np.random.default_rng(817)
    negative = generator.normal(size=(35, 6)).astype(np.float32)
    negative_scales = np.resize(np.asarray([3, 8]), len(negative))
    query = generator.normal(size=(13, 6)).astype(np.float32)
    query_scales = np.resize(np.asarray([8, 3]), len(query))
    model = ScaleConditionedNegativeKNN(negative, negative_scales)
    cpu = model.query(query, query_scales, ks=(1, 5, 15), device="cpu")
    cuda = model.query(
        query,
        query_scales,
        ks=(1, 5, 15),
        device="cuda",
        query_chunk_size=3,
        library_chunk_size=7,
    )
    cuda_repeat = model.query(
        query,
        query_scales,
        ks=(1, 5, 15),
        device="cuda",
        query_chunk_size=3,
        library_chunk_size=7,
    )
    for k in cpu:
        np.testing.assert_array_equal(cuda_repeat[k], cuda[k])
        np.testing.assert_allclose(cuda[k], cpu[k], atol=2e-6, rtol=2e-6)


def test_invalid_fit_inputs_are_rejected():
    cases = [
        (np.empty((0, 2)), np.asarray([], dtype=np.int64), None),
        (np.ones(3), np.asarray([1, 1, 1]), None),
        (np.asarray([[0.0, np.nan]]), np.asarray([1]), None),
        (np.ones((2, 2)), np.asarray([1]), None),
        (np.ones((2, 2)), np.asarray([1.0, 2.0]), None),
        (np.ones((2, 2)), np.asarray([True, False]), None),
        (np.ones((2, 2)), np.asarray([1, 2]), np.asarray(["a"])),
        (np.ones((2, 2)), np.asarray([1, 2]), np.asarray([1.0, 2.0])),
    ]
    for features, scales, families in cases:
        with _assert_raises(ValueError):
            ScaleConditionedNegativeKNN(features, scales, families)


def test_invalid_query_inputs_are_rejected():
    model = ScaleConditionedNegativeKNN(np.ones((3, 2)), np.asarray([1, 1, 1]))
    cases = [
        (np.ones(2), np.asarray([1]), {}),
        (np.ones((1, 3)), np.asarray([1]), {}),
        (np.asarray([[0.0, np.inf]]), np.asarray([1]), {}),
        (np.ones((2, 2)), np.asarray([1]), {}),
        (np.ones((1, 2)), np.asarray([1.0]), {}),
        (np.ones((1, 2)), np.asarray([1]), {"ks": ()}),
        (np.ones((1, 2)), np.asarray([1]), {"ks": (1, 1)}),
        (np.ones((1, 2)), np.asarray([1]), {"ks": (0,)}),
        (np.ones((1, 2)), np.asarray([1]), {"ks": (1.0,)}),
        (np.ones((1, 2)), np.asarray([1]), {"query_chunk_size": 0}),
        (np.ones((1, 2)), np.asarray([1]), {"library_chunk_size": False}),
        (np.ones((1, 2)), np.asarray([1]), {"device": "meta", "ks": (1,)}),
    ]
    for features, scales, kwargs in cases:
        with _assert_raises(ValueError, RuntimeError):
            model.query(features, scales, **kwargs)


def test_empty_query_returns_requested_empty_float32_arrays():
    model = ScaleConditionedNegativeKNN(np.ones((3, 2)), np.asarray([1, 1, 1]))
    result = model.query(
        np.empty((0, 2)), np.asarray([], dtype=np.int64), ks=(5, 1)
    )
    assert tuple(result) == (5, 1)
    for values in result.values():
        assert values.shape == (0,)
        assert values.dtype == np.float32
