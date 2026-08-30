from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from pathline_template_matching.one_class_spatial import (
    high_score_two_means_predictions,
    masked_gaussian_grid_scores,
    negative_knn_scores,
    rank_scores,
)


@contextmanager
def _assert_raises(*error_types: type[BaseException]):
    try:
        yield
    except error_types:
        return
    raise AssertionError(f"expected one of {[item.__name__ for item in error_types]}")


def test_negative_knn_scores_are_exact_and_chunk_invariant_on_cpu():
    train = np.asarray([[0.0, 0.0], [3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    query = np.asarray([[0.0, 1.0], [6.0, 8.0]], dtype=np.float32)
    all_distances = np.linalg.norm(query[:, None, :] - train[None, :, :], axis=2)
    expected_second = np.sort(all_distances, axis=1)[:, 1]

    first = negative_knn_scores(
        train,
        query,
        k=2,
        device="cpu",
        query_chunk_size=1,
        library_chunk_size=1,
    )
    second = negative_knn_scores(
        train,
        query,
        k=2,
        device=torch.device("cpu"),
        query_chunk_size=8,
        library_chunk_size=8,
    )
    np.testing.assert_allclose(first, expected_second, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32


def test_negative_knn_scores_include_zero_self_distance_and_empty_queries():
    train = np.asarray([[1.0, -2.0], [4.0, 5.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        negative_knn_scores(train, train[:1], k=1), np.asarray([0.0], np.float32)
    )
    empty = negative_knn_scores(train, np.empty((0, 2), dtype=np.float32), k=1)
    assert empty.shape == (0,)
    assert empty.dtype == np.float32


def test_negative_knn_scores_cuda_matches_cpu():
    if not torch.cuda.is_available():
        return
    generator = np.random.default_rng(25068)
    train = generator.normal(size=(19, 7)).astype(np.float32)
    query = generator.normal(size=(11, 7)).astype(np.float32)
    cpu = negative_knn_scores(train, query, k=4, device="cpu", library_chunk_size=5)
    cuda = negative_knn_scores(
        train,
        query,
        k=4,
        device="cuda",
        query_chunk_size=3,
        library_chunk_size=5,
    )
    np.testing.assert_allclose(cuda, cpu, rtol=2e-6, atol=2e-6)


def test_negative_knn_scores_reject_invalid_inputs():
    cases = [
        (np.empty((0, 2)), np.empty((1, 2)), {}),
        (np.ones(3), np.ones((1, 3)), {}),
        (np.ones((2, 3)), np.ones((1, 2)), {}),
        (np.asarray([[0.0, np.nan]]), np.ones((1, 2)), {}),
        (np.ones((2, 3)), np.ones((1, 3)), {"k": 0}),
        (np.ones((2, 3)), np.ones((1, 3)), {"k": 3}),
        (np.ones((2, 3)), np.ones((1, 3)), {"k": 1.0}),
        (np.ones((2, 3)), np.ones((1, 3)), {"query_chunk_size": 0}),
        (np.ones((2, 3)), np.ones((1, 3)), {"library_chunk_size": False}),
        (np.ones((2, 3)), np.ones((1, 3)), {"device": "meta"}),
    ]
    for train, query, kwargs in cases:
        with _assert_raises(ValueError, RuntimeError):
            negative_knn_scores(train, query, **kwargs)


def test_rank_scores_use_center_index_to_break_score_ties():
    scores = np.asarray([3.0, 1.0, 1.0, 2.0])
    centers = np.asarray([8, 4, 2, 9], dtype=np.int64)
    np.testing.assert_array_equal(
        rank_scores(scores, centers), np.asarray([1.0, 0.5, 0.25, 0.75])
    )
    np.testing.assert_array_equal(rank_scores([7.0], np.asarray([3])), [1.0])
    assert rank_scores([], np.asarray([], dtype=np.int64)).shape == (0,)


def test_rank_scores_reject_invalid_inputs():
    cases = [
        (np.ones((2, 1)), np.asarray([0, 1])),
        ([0.0, np.inf], np.asarray([0, 1])),
        ([0.0], np.asarray([0, 1])),
        ([0.0], np.asarray([0.0])),
        ([0.0, 1.0], np.asarray([3, 3])),
    ]
    for scores, centers in cases:
        with _assert_raises(ValueError):
            rank_scores(scores, centers)


def test_masked_gaussian_normalizes_only_over_valid_centers():
    weight_at_distance_two = np.exp(-2.0)
    expected = np.asarray(
        [
            10.0 * weight_at_distance_two / (1.0 + weight_at_distance_two),
            10.0 / (1.0 + weight_at_distance_two),
        ]
    )
    result = masked_gaussian_grid_scores(
        [0.0, 10.0],
        np.asarray([0, 2]),
        grid_shape=(1, 1, 3),
        sigma=1.0,
        truncate=2.0,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_masked_gaussian_flat_indices_follow_zyx_c_order():
    # In shape (Z=2,Y=3,X=4), flat indices 0 and 12 differ by one z step.
    adjacent_weight = np.exp(-0.5)
    expected = np.asarray(
        [2.0 * adjacent_weight / (1.0 + adjacent_weight), 2.0 / (1.0 + adjacent_weight)]
    )
    result = masked_gaussian_grid_scores(
        [0.0, 2.0],
        np.asarray([0, 12]),
        grid_shape=(2, 3, 4),
        sigma=1.0,
        truncate=1.0,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_masked_gaussian_sigma_zero_preserves_input_rows_and_order():
    scores = np.asarray([5.0, -1.0, 2.0])
    indices = np.asarray([7, 0, 3], dtype=np.int64)
    np.testing.assert_array_equal(
        masked_gaussian_grid_scores(
            scores, indices, grid_shape=(2, 2, 2), sigma=0.0
        ),
        scores,
    )


def test_masked_gaussian_rejects_invalid_inputs():
    cases = [
        ([1.0, 2.0], np.asarray([0, 0]), (1, 1, 1), 1.0, 3.0),
        ([1.0], np.asarray([-1]), (1, 1, 1), 1.0, 3.0),
        ([1.0], np.asarray([1]), (1, 1, 1), 1.0, 3.0),
        ([1.0], np.asarray([0.0]), (1, 1, 1), 1.0, 3.0),
        ([1.0], np.asarray([0]), (1, 1), 1.0, 3.0),
        ([1.0], np.asarray([0]), (1, 0, 1), 1.0, 3.0),
        ([1.0], np.asarray([0]), (1, 1, 1), -1.0, 3.0),
        ([1.0], np.asarray([0]), (1, 1, 1), 1.0, 0.0),
        ([np.nan], np.asarray([0]), (1, 1, 1), 1.0, 3.0),
    ]
    for scores, indices, shape, sigma, truncate in cases:
        with _assert_raises(ValueError):
            masked_gaussian_grid_scores(
                scores,
                indices,
                grid_shape=shape,
                sigma=sigma,
                truncate=truncate,
            )


def test_high_score_two_means_finds_global_high_mean_cluster():
    scores = np.asarray([10.0, 0.2, 9.0, 0.0])
    np.testing.assert_array_equal(
        high_score_two_means_predictions(scores),
        np.asarray([True, False, True, False]),
    )
    # Equal high scores are never split across clusters.
    np.testing.assert_array_equal(
        high_score_two_means_predictions([0.0, 1.0, 1.0]),
        np.asarray([False, True, True]),
    )


def test_high_score_two_means_is_translation_stable_at_large_offset():
    centered = np.asarray([0.0, 1.0, 2.0, 100.0])
    shifted = centered + 1.0e12
    expected = np.asarray([False, False, False, True])
    np.testing.assert_array_equal(
        high_score_two_means_predictions(centered), expected
    )
    np.testing.assert_array_equal(
        high_score_two_means_predictions(shifted), expected
    )


def test_high_score_two_means_constant_or_insufficient_input_fails_closed():
    for scores in ([], [3.0], [3.0, 3.0, 3.0]):
        prediction = high_score_two_means_predictions(scores)
        assert prediction.dtype == np.bool_
        assert not prediction.any()


def test_high_score_two_means_rejects_invalid_scores():
    for scores in (np.ones((2, 1)), [0.0, np.nan], [1.0, np.inf]):
        with _assert_raises(ValueError):
            high_score_two_means_predictions(scores)
