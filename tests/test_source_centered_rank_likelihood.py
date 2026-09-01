from __future__ import annotations

import numpy as np
import unittest

from pathline_template_matching.source_centered_rank_likelihood import (
    FamilyBalancedRankLikelihoodModel,
    FamilySourceRankBatch,
    conservative_strict_ecdf,
    assigned_block_dx_midranks,
    empirical_midrank,
    pair_assigned_center_ranks,
    strict_absolute_threshold,
)


def _fit_batches() -> dict[str, FamilySourceRankBatch]:
    return {
        "family_a": FamilySourceRankBatch(
            ranks=np.asarray(
                [0.10, 0.15, 0.75, 0.80, 0.20, 0.25, 0.85, 0.90],
                dtype=np.float64,
            ),
            labels=np.asarray(
                [False, False, True, True, False, False, True, True],
                dtype=np.bool_,
            ),
            source_ids=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        ),
        "family_b": FamilySourceRankBatch(
            ranks=np.asarray(
                [0.05, 0.10, 0.20, 0.70, 0.80, 0.15, 0.25, 0.85, 0.95],
                dtype=np.float64,
            ),
            labels=np.asarray(
                [False, False, False, True, True, False, False, True, True],
                dtype=np.bool_,
            ),
            source_ids=np.asarray(
                [10, 10, 10, 10, 10, 11, 11, 11, 11], dtype=np.int64
            ),
        ),
    }


def _model() -> FamilyBalancedRankLikelihoodModel:
    return FamilyBalancedRankLikelihoodModel(
        _fit_batches(), bin_count=64, beta=0.5
    )


def test_empirical_midrank_uses_average_tie_rank() -> None:
    observed = empirical_midrank(
        np.asarray([3.0, 1.0, 1.0, 2.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(observed, [0.875, 0.25, 0.25, 0.625])
    assert observed.dtype == np.dtype(np.float64)
    assert not observed.flags.writeable


def test_assigned_midrank_is_separate_for_every_block_dx_group_and_order_safe() -> None:
    center_count = 100
    centers = np.tile(np.arange(center_count, dtype=np.int64), 2)
    blocks = np.repeat(np.asarray([0, 1], dtype=np.int8), center_count)
    dx_level = centers % 10
    scales = (
        blocks.astype(np.int32) * 1000 + dx_level.astype(np.int32) * 100
    )
    within_group_value = (centers // 10).astype(np.float32)
    curl = within_group_value + blocks.astype(np.float32) * 1000.0
    permutation = np.random.default_rng(17068).permutation(len(centers))

    observed = assigned_block_dx_midranks(
        centers[permutation],
        blocks[permutation],
        scales[permutation],
        curl[permutation],
        center_count=center_count,
    )
    expected = ((centers[permutation] // 10) + 0.5) / 10.0
    np.testing.assert_array_equal(observed, expected)
    assert not observed.flags.writeable

    broken_scales = scales[permutation].copy()
    broken_scales[0] = int(blocks[permutation][0]) * 1000
    with unittest.TestCase().assertRaisesRegex(
        ValueError, "must contain exactly 10 assigned rows"
    ):
        assigned_block_dx_midranks(
            centers[permutation],
            blocks[permutation],
            broken_scales,
            curl[permutation],
            center_count=center_count,
        )


def test_pairing_uses_both_assigned_ranks_but_library_mask_is_combined_valid() -> None:
    centers = np.tile(np.arange(4, dtype=np.int64), 2)
    blocks = np.repeat(np.asarray([0, 1], dtype=np.int8), 4)
    ranks = np.asarray(
        [0.10, 0.20, 0.30, 0.40, 0.90, 0.80, 0.70, 0.60],
        dtype=np.float64,
    )
    valid_centers = np.asarray([0, 1, 2, 2], dtype=np.int64)
    valid_blocks = np.asarray([0, 1, 0, 1], dtype=np.int8)
    paired = pair_assigned_center_ranks(
        centers,
        blocks,
        ranks,
        valid_centers,
        valid_blocks,
        weight=0.25,
        center_count=4,
    )

    expected = 0.25 * ranks[:4] + 0.75 * ranks[4:]
    np.testing.assert_array_equal(paired.paired_rank, expected)
    np.testing.assert_array_equal(paired.combined_valid, [True, True, True, False])
    np.testing.assert_array_equal(
        paired.valid_row_paired_rank, expected[valid_centers]
    )
    # Center 0 has only a valid legacy pathline, but its assigned expanded rank
    # is still part of the fixed paired representation.
    np.testing.assert_allclose(paired.paired_rank[0], 0.70, rtol=0.0, atol=1e-15)
    for value in paired.__dataclass_fields__:
        array = getattr(paired, value)
        if isinstance(array, np.ndarray):
            assert not array.flags.writeable


def test_family_histograms_are_normalized_before_equal_family_average() -> None:
    model = _model()
    counts = model.family_class_histogram_counts
    totals = model.family_class_totals
    expected_family = (counts + 0.5) / (totals[:, :, None] + 0.5 * 64.0)
    np.testing.assert_array_equal(model.family_class_density, expected_family)
    np.testing.assert_array_equal(
        model.full_class_density, expected_family.mean(axis=0)
    )

    pooled = (counts.sum(axis=0) + 0.5) / (
        totals.sum(axis=0)[:, None] + 0.5 * 64.0
    )
    assert not np.array_equal(model.full_class_density, pooled)


def test_family_order_permutation_does_not_change_model_or_scores() -> None:
    batches = _fit_batches()
    forward = FamilyBalancedRankLikelihoodModel(
        batches, bin_count=64, beta=2.0
    )
    reverse = FamilyBalancedRankLikelihoodModel(
        {name: batches[name] for name in reversed(tuple(batches))},
        bin_count=64,
        beta=2.0,
    )
    assert forward.family_order == reverse.family_order
    for name, values in forward.export_arrays().items():
        np.testing.assert_array_equal(values, reverse.export_arrays()[name])
    query = np.asarray([0.05, 0.21, 0.77, 0.99], dtype=np.float64)
    first = forward.query(query)
    second = reverse.query(query)
    for name in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_leave_one_source_out_reference_removes_both_classes_from_whole_source() -> None:
    batches = _fit_batches()
    model = _model()
    family_index = model.family_order.index("family_a")
    batch = batches["family_a"]
    bins = np.minimum((batch.ranks * 64).astype(np.int64), 63)
    family_density_sum = model.family_class_density.sum(axis=0)
    expected_parts: list[np.ndarray] = []
    wrong_parts: list[np.ndarray] = []
    for source_id in np.unique(batch.source_ids):
        source = batch.source_ids == source_id
        source_counts = np.zeros((2, 64), dtype=np.int64)
        source_counts[0] = np.bincount(
            bins[source & ~batch.labels], minlength=64
        )
        source_counts[1] = np.bincount(
            bins[source & batch.labels], minlength=64
        )
        remaining = model.family_class_histogram_counts[family_index] - source_counts
        loo_family = (remaining + 0.5) / (
            remaining.sum(axis=1)[:, None] + 0.5 * 64.0
        )
        loo_density = (
            family_density_sum
            - model.family_class_density[family_index]
            + loo_family
        ) / model.family_count
        negative_bins = bins[source & ~batch.labels]
        expected_parts.append(
            np.log(loo_density[1, negative_bins])
            - np.log(loo_density[0, negative_bins])
        )

        # Incorrect control: remove only this source's negative templates and
        # leave its positive templates in the likelihood fit.
        wrong_remaining = model.family_class_histogram_counts[family_index].copy()
        wrong_remaining[0] -= source_counts[0]
        wrong_family = (wrong_remaining + 0.5) / (
            wrong_remaining.sum(axis=1)[:, None] + 0.5 * 64.0
        )
        wrong_density = (
            family_density_sum
            - model.family_class_density[family_index]
            + wrong_family
        ) / model.family_count
        wrong_parts.append(
            np.log(wrong_density[1, negative_bins])
            - np.log(wrong_density[0, negative_bins])
        )

    start = int(model.dual_negative_reference_offsets[family_index])
    stop = int(model.dual_negative_reference_offsets[family_index + 1])
    observed = model.dual_negative_reference_values[start:stop]
    expected = np.sort(np.concatenate(expected_parts), kind="mergesort")
    wrong = np.sort(np.concatenate(wrong_parts), kind="mergesort")
    np.testing.assert_array_equal(observed, expected)
    assert not np.array_equal(observed, wrong)


def test_leave_one_source_out_fails_when_remaining_family_loses_a_class() -> None:
    unsupported = FamilySourceRankBatch(
        ranks=np.asarray([0.10, 0.80, 0.85, 0.20, 0.25], dtype=np.float64),
        labels=np.asarray([False, True, True, False, False], dtype=np.bool_),
        source_ids=np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
    )
    supported = _fit_batches()["family_b"]
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        (
            "leave-one-source-out histogram must retain both classes: "
            "family='unsupported', source_id=0, missing=positive"
        ),
    ):
        FamilyBalancedRankLikelihoodModel(
            {"unsupported": unsupported, "supported": supported},
            bin_count=64,
            beta=0.5,
        )


def test_conservative_strict_ecdf_has_frozen_tie_and_boundary_rule() -> None:
    observed = conservative_strict_ecdf(
        np.asarray([1.0, 1.0, 3.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(observed, [0.0, 0.0, 0.5, 0.5, 0.75])
    assert not observed.flags.writeable


def test_query_is_order_chunk_and_membership_invariant() -> None:
    model = _model()
    query = np.asarray([0.03, 0.10, 0.22, 0.70, 0.83, 0.99], dtype=np.float64)
    together = model.query(query)
    order = np.asarray([4, 0, 5, 2, 1, 3], dtype=np.int64)
    reordered = model.query(query[order])
    for name in together.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(reordered, name), getattr(together, name)[order]
        )
    for row in range(len(query)):
        alone = model.query(query[row : row + 1])
        for name in together.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(alone, name), getattr(together, name)[row : row + 1]
            )


def test_fit_class_labels_change_dual_score_and_absolute_prediction() -> None:
    batches = _fit_batches()
    original = FamilyBalancedRankLikelihoodModel(
        batches, bin_count=64, beta=0.5
    )
    relabeled = {
        name: FamilySourceRankBatch(
            ranks=batch.ranks,
            labels=~batch.labels,
            source_ids=batch.source_ids,
        )
        for name, batch in batches.items()
    }
    changed = FamilyBalancedRankLikelihoodModel(
        relabeled, bin_count=64, beta=0.5
    )
    query = np.asarray([0.90], dtype=np.float64)
    original_score = original.query(query).dual_template_score
    changed_score = changed.query(query).dual_template_score
    assert not np.array_equal(original_score, changed_score)
    eligible = np.ones(1, dtype=np.bool_)
    original_prediction = strict_absolute_threshold(
        original_score, eligible, threshold=0.75
    )
    changed_prediction = strict_absolute_threshold(
        changed_score, eligible, threshold=0.75
    )
    np.testing.assert_array_equal(original_prediction, [True])
    np.testing.assert_array_equal(changed_prediction, [False])


def test_negative_ecdf_has_no_histogram_axis_and_is_family_equal() -> None:
    model = _model()
    query = np.asarray([0.10, 0.90], dtype=np.float64)
    result = model.query(query)
    expected = np.zeros(len(query), dtype=np.float64)
    for family_index in range(model.family_count):
        start = int(model.negative_rank_reference_offsets[family_index])
        stop = int(model.negative_rank_reference_offsets[family_index + 1])
        expected += conservative_strict_ecdf(
            model.negative_rank_reference_values[start:stop], query
        )
    expected /= model.family_count
    np.testing.assert_array_equal(result.negative_ecdf_score, expected)

    other_histogram = FamilyBalancedRankLikelihoodModel(
        _fit_batches(), bin_count=256, beta=2.0
    )
    np.testing.assert_array_equal(
        result.negative_ecdf_score,
        other_histogram.query(query).negative_ecdf_score,
    )


def test_strict_absolute_threshold_keeps_equal_score_negative() -> None:
    result = strict_absolute_threshold(
        np.asarray([0.89, 0.90, 0.91, 0.99], dtype=np.float64),
        np.asarray([True, True, True, False], dtype=np.bool_),
        threshold=0.90,
    )
    np.testing.assert_array_equal(result, [False, False, True, False])
    assert not result.flags.writeable


def test_model_export_round_trip_is_allow_pickle_false_and_exact(tmp_path) -> None:
    model = _model()
    arrays = model.export_arrays()
    assert all(value.dtype.kind != "O" for value in arrays.values())
    destination = tmp_path / "rank_likelihood.npz"
    np.savez_compressed(destination, **arrays)
    with np.load(destination, allow_pickle=False) as archive:
        restored = FamilyBalancedRankLikelihoodModel.from_arrays(
            {name: archive[name] for name in archive.files}
        )
    query = np.asarray([0.01, 0.12, 0.51, 0.88, 1.0], dtype=np.float64)
    original_result = model.query(query)
    restored_result = restored.query(query)
    for name in original_result.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(original_result, name), getattr(restored_result, name)
        )
    for name, value in arrays.items():
        np.testing.assert_array_equal(value, restored.export_arrays()[name])
        assert not value.flags.writeable


def test_inputs_are_copied_and_outputs_are_read_only() -> None:
    batches = _fit_batches()
    family_a_ranks = batches["family_a"].ranks
    with unittest.TestCase().assertRaises(ValueError):
        family_a_ranks[0] = 0.99
    model = _model()
    query = np.asarray([0.2, 0.8], dtype=np.float64)
    result = model.query(query)
    query[:] = 0.0
    np.testing.assert_array_equal(result.ranks, [0.2, 0.8])
    for name in result.__dataclass_fields__:
        with unittest.TestCase().assertRaises(ValueError):
            getattr(result, name)[0] = 0
    for name, value in model.export_arrays().items():
        with unittest.TestCase().assertRaisesRegex(ValueError, "read-only"):
            value.flat[0] = value.flat[0]


def test_invalid_population_and_model_parameters_fail_closed() -> None:
    with unittest.TestCase().assertRaisesRegex(
        ValueError, "at least two complete sources"
    ):
        FamilySourceRankBatch(
            ranks=np.asarray([0.1, 0.9], dtype=np.float64),
            labels=np.asarray([False, True], dtype=np.bool_),
            source_ids=np.asarray([0, 0], dtype=np.int64),
        )
    with unittest.TestCase().assertRaisesRegex(ValueError, "both template classes"):
        FamilySourceRankBatch(
            ranks=np.asarray([0.1, 0.2], dtype=np.float64),
            labels=np.asarray([False, False], dtype=np.bool_),
            source_ids=np.asarray([0, 1], dtype=np.int64),
        )
    with unittest.TestCase().assertRaisesRegex(ValueError, "bin_count"):
        FamilyBalancedRankLikelihoodModel(
            _fit_batches(), bin_count=32, beta=0.5
        )
    with unittest.TestCase().assertRaisesRegex(ValueError, "beta"):
        FamilyBalancedRankLikelihoodModel(
            _fit_batches(), bin_count=64, beta=1.0
        )
    model = _model()
    with unittest.TestCase().assertRaisesRegex(ValueError, "dtype float64"):
        model.query(np.asarray([0.5], dtype=np.float32))
    with unittest.TestCase().assertRaisesRegex(ValueError, r"lie in \[0,1\]"):
        model.query(np.asarray([1.1], dtype=np.float64))


def test_serialized_tampering_fails_closed() -> None:
    arrays = _model().export_arrays()
    bad_density = {name: value.copy() for name, value in arrays.items()}
    bad_density["full_class_density"][0, 0] += 0.01
    with unittest.TestCase().assertRaisesRegex(ValueError, "family-equal mean"):
        FamilyBalancedRankLikelihoodModel.from_arrays(bad_density)

    bad_offsets = {name: value.copy() for name, value in arrays.items()}
    bad_offsets["dual_negative_reference_offsets"][1] += 1
    with unittest.TestCase().assertRaises(ValueError):
        FamilyBalancedRankLikelihoodModel.from_arrays(bad_offsets)
