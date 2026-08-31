from __future__ import annotations

import tempfile

import numpy as np

from pathline_template_matching.class_conditional_template_score import (
    ClassConditionalTemplateScoreModel,
    FamilyFitBatch,
    combine_joint_family_conformity,
    strict_threshold_predictions,
)
from pathline_template_matching.negative_tail_calibration import (
    CALIBRATION_BLOCK_FALLBACK,
    CALIBRATION_LOCAL_BLOCK_SHRINK,
    CALIBRATION_LOCAL_ONLY,
    empirical_upper_tail_probability,
)
from pathline_template_matching.per_scale_negative_metric import SCALER_ARRAY_NAMES


def _expect_value_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _batch(features, scales, labels) -> FamilyFitBatch:
    return FamilyFitBatch(
        np.asarray(features, dtype=np.float32).reshape(-1, 1),
        np.asarray(scales, dtype=np.int64),
        np.asarray(labels, dtype=np.bool_),
    )


def _small_family_batch(
    family_offset: float,
    *,
    include_positive: bool = True,
    positive_shift: float = 0.0,
) -> FamilyFitBatch:
    features: list[list[float]] = []
    scales: list[int] = []
    labels: list[bool] = []
    for scale_id in (3, 4):
        for row in range(3):
            features.append([family_offset + scale_id * 0.2 + row * 0.7])
            scales.append(scale_id)
            labels.append(False)
        if include_positive:
            for row in range(3):
                features.append(
                    [
                        10.0
                        + positive_shift
                        + family_offset
                        + scale_id * 0.2
                        + row * 0.7
                    ]
                )
                scales.append(scale_id)
                labels.append(True)
    return _batch(features, scales, labels)


def _small_model(
    *,
    positive_shift_b: float = 0.0,
    include_positive=(True, True, True),
    mapping_order=("a", "b", "c"),
) -> ClassConditionalTemplateScoreModel:
    source = {
        "a": _small_family_batch(0.0, include_positive=include_positive[0]),
        "b": _small_family_batch(
            0.3,
            include_positive=include_positive[1],
            positive_shift=positive_shift_b,
        ),
        "c": _small_family_batch(0.6, include_positive=include_positive[2]),
    }
    batches = {family: source[family] for family in mapping_order}
    return ClassConditionalTemplateScoreModel(
        batches,
        family_order=("a", "b", "c"),
        ks=(1,),
        query_chunk_size=2,
        library_chunk_size=2,
    )


def _sparse_all_k_model() -> tuple[
    ClassConditionalTemplateScoreModel, dict[int, tuple[int, int, int]]
]:
    cases: dict[int, tuple[int, int, int]] = {}
    family_batches: dict[str, FamilyFitBatch] = {}
    for family_index, family in enumerate(("a", "b", "c")):
        features: list[list[float]] = []
        scales: list[int] = []
        labels: list[bool] = []
        for k_index, k in enumerate((1, 5, 15, 31)):
            under_scale = 20 + 3 * k_index
            exact_scale = under_scale + 1
            local_scale = under_scale + 2
            cases[k] = (under_scale, exact_scale, local_scale)
            for scale_id, count in (
                (under_scale, k - 1),
                (exact_scale, k),
                (local_scale, k + 1),
            ):
                for positive in (False, True):
                    class_offset = 8.0 if positive else 0.0
                    for row in range(count):
                        features.append(
                            [
                                class_offset
                                + 0.4 * family_index
                                + 0.01 * scale_id
                                + 0.03 * row
                            ]
                        )
                        scales.append(scale_id)
                        labels.append(positive)
        family_batches[family] = _batch(features, scales, labels)
    return (
        ClassConditionalTemplateScoreModel(
            family_batches,
            family_order=("a", "b", "c"),
            ks=(1, 5, 15, 31),
            query_chunk_size=16,
            library_chunk_size=16,
        ),
        cases,
    )


def test_empirical_q_uses_plus_one_upper_tail_and_conservative_ties():
    reference = np.asarray([0.0, 1.0, 1.0, 3.0], dtype=np.float32)
    observed = empirical_upper_tail_probability(
        reference, np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(observed, [4.0 / 5.0, 2.0 / 5.0, 1.0 / 5.0])


def test_joint_combiner_uses_one_family_set_equal_weights_and_strict_majority():
    positive = np.asarray(
        [[0.9, 0.7, 0.8], [0.5, 0.3, 0.2], [0.1, 1.0, 0.6]],
        dtype=np.float64,
    )
    negative = np.asarray(
        [[0.2, 0.5, 0.4], [0.4, 0.1, 0.9], [0.8, 0.0, 0.3]],
        dtype=np.float64,
    )
    retrieval = np.ones((3, 3), dtype=bool)
    calibration = np.asarray(
        [[True, True, True], [True, True, False], [True, False, False]],
        dtype=bool,
    )
    result = combine_joint_family_conformity(
        positive,
        negative,
        retrieval,
        calibration,
        retrieval,
        calibration,
        required_family_count=2,
    )
    assert result.supporting_family_counts.tolist() == [3, 2, 1]
    assert result.supported.tolist() == [True, True, False]
    np.testing.assert_array_equal(
        result.positive_conformity[:2],
        [np.mean([0.9, 0.5, 0.1]), np.mean([0.7, 0.3])],
    )
    np.testing.assert_array_equal(
        result.negative_conformity[:2],
        [np.mean([0.2, 0.4, 0.8]), np.mean([0.5, 0.1])],
    )
    assert result.scores[2] == 0.0
    assert result.positive_conformity[2] == 0.0
    assert result.negative_conformity[2] == 0.0

    swapped = combine_joint_family_conformity(
        negative,
        positive,
        retrieval,
        calibration,
        retrieval,
        calibration,
        required_family_count=2,
    )
    np.testing.assert_array_equal(
        swapped.scores[result.supported], 1.0 - result.scores[result.supported]
    )

    four_family = np.ones((4, 2), dtype=np.float64) * 0.5
    four_support = np.asarray(
        [[True, True], [True, True], [True, False], [False, False]], dtype=bool
    )
    final = combine_joint_family_conformity(
        four_family,
        four_family,
        four_support,
        four_support,
        four_support,
        four_support,
        required_family_count=3,
    )
    assert final.supported.tolist() == [True, False]


def test_joint_combiner_rejects_calibration_without_retrieval():
    values = np.asarray([[0.5]], dtype=np.float64)
    no = np.asarray([[False]], dtype=bool)
    yes = np.asarray([[True]], dtype=bool)
    _expect_value_error(
        combine_joint_family_conformity,
        values,
        values,
        no,
        yes,
        yes,
        yes,
        required_family_count=1,
    )


def test_sparse_positive_scale_contract_for_every_frozen_k():
    model, cases = _sparse_all_k_model()
    positive = model.calibrator_for("a", positive=True)
    assert positive is not None
    for k, (under_scale, exact_scale, local_scale) in cases.items():
        query_scales = np.asarray(
            [under_scale, exact_scale, local_scale], dtype=np.int64
        )
        raw_query = np.asarray([[8.2], [8.2], [8.2]], dtype=np.float32)
        transformed = model.scaler.transform(raw_query, query_scales)
        result = positive.query(
            transformed,
            query_scales,
            ks=(k,),
            query_chunk_size=4,
            library_chunk_size=8,
        )
        assert result.retrieval_supported[k].tolist() == [False, True, True]
        assert result.calibration_supported[k].tolist() == [False, True, True]
        assert np.isnan(result.raw_distances[k][0])
        assert result.tail_probabilities[k][0] == 1.0
        assert result.anomaly_scores[k][0] == 0.0
        assert result.calibration_modes[k][1] == CALIBRATION_BLOCK_FALLBACK
        expected_local_mode = (
            CALIBRATION_LOCAL_ONLY
            if k == 31
            else CALIBRATION_LOCAL_BLOCK_SHRINK
        )
        assert result.calibration_modes[k][2] == expected_local_mode


def test_n_equals_k_without_any_same_class_prior_is_retrieved_but_not_joint_supported():
    family_batches = {
        family: _batch(
            [[family_index], [10.0 + family_index]],
            [9, 9],
            [False, True],
        )
        for family_index, family in enumerate(("a", "b", "c"))
    }
    model = ClassConditionalTemplateScoreModel(
        family_batches, family_order=("a", "b", "c"), ks=(1,)
    )
    result = model.query([[5.0]], [9], ks=(1,))
    assert result.retrieval_supported[1][0]
    assert not result.joint_supported[1][0]
    assert result.joint_family_count[1][0] == 0
    assert np.isfinite(result.mean_negative_distances[1][0])
    assert result.mean_negative_distances[1][0] >= 0.0
    assert result.scores[1][0] == 0.0
    transformed = model.scaler.transform([[5.0]], [9])
    per_family_negative_distance = []
    for family in model.family_order:
        negative = model.calibrator_for(family, positive=False)
        assert negative is not None
        negative_result = negative.query(transformed, [9], ks=(1,))
        assert negative_result.retrieval_supported[1][0]
        assert not negative_result.calibration_supported[1][0]
        per_family_negative_distance.append(negative_result.raw_distances[1][0])
    np.testing.assert_array_equal(
        result.mean_negative_distances[1],
        np.asarray([np.mean(per_family_negative_distance)], dtype=np.float32),
    )


def test_leave_one_out_excludes_only_self_and_keeps_duplicate_templates():
    family_batches = {
        family: _batch(
            [[0.0], [0.0], [2.0], [8.0], [8.0], [10.0]],
            [2, 2, 2, 2, 2, 2],
            [False, False, False, True, True, True],
        )
        for family in ("a", "b", "c")
    }
    model = ClassConditionalTemplateScoreModel(
        family_batches, family_order=("a", "b", "c"), ks=(1,)
    )
    calibrator = model.calibrator_for("a", positive=False)
    assert calibrator is not None
    loo = calibrator.export_arrays()["loo_distances_k_1"]
    assert len(loo) == 3
    assert np.count_nonzero(loo == 0.0) == 2
    assert loo[-1] > 0.0


def test_absent_positive_class_is_explicit_and_never_borrowed_from_other_family():
    model = _small_model(include_positive=(True, True, False))
    assert model.required_family_count == 2
    assert model.calibrator_for("c", positive=True) is None
    assert model.tail_calibrator.calibrator_for("c", positive=True) is None
    result = model.query([[10.8]], [3], ks=(1,))
    assert result.retrieval_supported[1][0]
    assert result.joint_supported[1][0]
    assert result.joint_family_count[1][0] == 2
    assert np.isfinite(result.mean_negative_distances[1][0])
    assert result.per_family_positive_retrieval_supported[1].shape == (1, 3)
    assert result.per_family_positive_calibration_supported[1][0].tolist() == [
        True,
        True,
        False,
    ]
    assert result.per_family_negative_retrieval_supported[1][0].tolist() == [
        True,
        True,
        True,
    ]
    assert result.per_family_negative_calibration_supported[1][0].tolist() == [
        True,
        True,
        True,
    ]

    insufficient = _small_model(include_positive=(True, False, False))
    failed = insufficient.query([[10.8]], [3], ks=(1,))
    assert not failed.retrieval_supported[1][0]
    assert not failed.joint_supported[1][0]
    assert failed.joint_family_count[1][0] == 1
    assert np.isnan(failed.mean_negative_distances[1][0])
    assert failed.scores[1][0] == 0.0


def test_exact_scale_without_any_fit_negative_disables_both_class_supports():
    batches = {
        family: _batch(
            [[0.0], [1.0], [10.0], [11.0]],
            [3, 3, 4, 4],
            [False, False, True, True],
        )
        for family in ("a", "b", "c")
    }
    model = ClassConditionalTemplateScoreModel(
        batches, family_order=("a", "b", "c"), ks=(1,)
    )
    result = model.query([[10.5]], [4], ks=(1,))
    assert not result.retrieval_supported[1][0]
    assert not result.joint_supported[1][0]
    assert result.joint_family_count[1][0] == 0
    assert not result.per_family_positive_retrieval_supported[1][0].any()
    assert not result.per_family_positive_calibration_supported[1][0].any()
    assert not result.per_family_negative_retrieval_supported[1][0].any()
    assert not result.per_family_negative_calibration_supported[1][0].any()


def test_query_contract_dtypes_distance_mean_and_batch_chunk_invariance():
    model = _small_model()
    target = np.asarray([[10.8]], dtype=np.float32)
    alone = model.query(
        target, [3], ks=(1,), query_chunk_size=1, library_chunk_size=1
    )
    together = model.query(
        np.asarray([[10.8], [100000.0]], dtype=np.float32),
        [3, 3],
        ks=(1,),
        query_chunk_size=8,
        library_chunk_size=8,
    )
    assert alone.scores[1].dtype == np.dtype(np.float64)
    assert alone.mean_negative_distances[1].dtype == np.dtype(np.float32)
    assert alone.retrieval_supported[1].dtype == np.dtype(np.bool_)
    assert alone.joint_supported[1].dtype == np.dtype(np.bool_)
    assert alone.joint_family_count[1].dtype == np.dtype(np.int16)
    assert not alone.scores[1].flags.writeable
    for field in (
        "scores",
        "mean_negative_distances",
        "retrieval_supported",
        "joint_supported",
        "joint_family_count",
        "positive_conformity",
        "negative_conformity",
        "per_family_positive_retrieval_supported",
        "per_family_positive_calibration_supported",
        "per_family_negative_retrieval_supported",
        "per_family_negative_calibration_supported",
    ):
        np.testing.assert_array_equal(
            getattr(alone, field)[1], getattr(together, field)[1][:1]
        )

    transformed = model.scaler.transform(target, [3])
    expected_negative_distances = []
    for family in model.family_order:
        positive = model.calibrator_for(family, positive=True)
        negative = model.calibrator_for(family, positive=False)
        assert positive is not None and negative is not None
        positive_result = positive.query(transformed, [3], ks=(1,))
        negative_result = negative.query(transformed, [3], ks=(1,))
        assert positive_result.calibration_supported[1][0]
        assert negative_result.calibration_supported[1][0]
        expected_negative_distances.append(negative_result.raw_distances[1][0])
    np.testing.assert_allclose(
        alone.mean_negative_distances[1],
        np.asarray([np.mean(expected_negative_distances)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    absent = model.query([[1.0e20]], [77], ks=(1,))
    assert not absent.retrieval_supported[1][0]
    assert not absent.joint_supported[1][0]
    assert np.isnan(absent.mean_negative_distances[1][0])
    assert absent.scores[1][0] == 0.0


def test_other_class_and_family_positive_rows_do_not_enter_target_tail_prior():
    original = _small_model()
    perturbed = _small_model(positive_shift_b=1000.0)
    for name in SCALER_ARRAY_NAMES:
        np.testing.assert_array_equal(
            original.scaler.export_arrays()[name],
            perturbed.scaler.export_arrays()[name],
        )
    for family, positive in (("a", True), ("a", False), ("c", True)):
        first = original.calibrator_for(family, positive=positive)
        second = perturbed.calibrator_for(family, positive=positive)
        assert first is not None and second is not None
        first_arrays = first.export_arrays()
        second_arrays = second.export_arrays()
        assert set(first_arrays) == set(second_arrays)
        for name in first_arrays:
            np.testing.assert_array_equal(first_arrays[name], second_arrays[name])


def test_family_order_not_mapping_insertion_controls_fit_and_serialization():
    forward = _small_model(mapping_order=("a", "b", "c"))
    reverse = _small_model(mapping_order=("c", "b", "a"))
    first = forward.export_arrays()
    second = reverse.export_arrays()
    assert tuple(first) == tuple(second)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_separate_artifacts_round_trip_and_fail_closed_tamper_checks():
    model = _small_model()
    scaler_arrays = model.scaler.export_arrays()
    calibrator_arrays = model.tail_calibrator.export_arrays()
    assert model.tail_calibrator.fit_audit["probability_claim"] is False
    assert all(np.asarray(value).dtype.kind != "O" for value in scaler_arrays.values())
    assert all(
        np.asarray(value).dtype.kind != "O" for value in calibrator_arrays.values()
    )

    with tempfile.TemporaryDirectory() as directory:
        artifact = f"{directory}/calibrators.npz"
        np.savez(artifact, **calibrator_arrays)
        with np.load(artifact, allow_pickle=False) as archive:
            loaded_calibrators = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
        loaded = ClassConditionalTemplateScoreModel.from_artifacts(
            scaler_arrays, loaded_calibrators
        )
        assert loaded.family_order == model.family_order

    restored = ClassConditionalTemplateScoreModel.from_artifacts(
        scaler_arrays, calibrator_arrays
    )
    combined_restored = ClassConditionalTemplateScoreModel.from_arrays(
        model.export_arrays()
    )
    query = np.asarray([[10.8], [1.0]], dtype=np.float32)
    scales = np.asarray([3, 4], dtype=np.int64)
    expected = model.query(query, scales, ks=(1,))
    for candidate in (restored, combined_restored):
        observed = candidate.query(query, scales, ks=(1,))
        for field in (
            "scores",
            "mean_negative_distances",
            "retrieval_supported",
            "joint_supported",
            "joint_family_count",
            "positive_conformity",
            "negative_conformity",
            "per_family_positive_retrieval_supported",
            "per_family_positive_calibration_supported",
            "per_family_negative_retrieval_supported",
            "per_family_negative_calibration_supported",
        ):
            np.testing.assert_array_equal(
                getattr(expected, field)[1], getattr(observed, field)[1]
            )

    extra = {name: value.copy() for name, value in calibrator_arrays.items()}
    extra["unexpected"] = np.asarray(1, dtype=np.int8)
    _expect_value_error(
        ClassConditionalTemplateScoreModel.from_artifacts, scaler_arrays, extra
    )

    family_tamper = {
        name: value.copy() for name, value in calibrator_arrays.items()
    }
    family_tamper["family_order_copy_unicode"][0] = "c"
    _expect_value_error(
        ClassConditionalTemplateScoreModel.from_artifacts,
        scaler_arrays,
        family_tamper,
    )

    count_tamper = {name: value.copy() for name, value in calibrator_arrays.items()}
    count_tamper["class_scale_counts_int64"][0, 1, 3] += 1
    _expect_value_error(
        ClassConditionalTemplateScoreModel.from_artifacts,
        scaler_arrays,
        count_tamper,
    )

    identity_tamper = {
        name: value.copy() for name, value in calibrator_arrays.items()
    }
    mean_key = next(
        name
        for name in identity_tamper
        if name.startswith("calibrator_f0_c0__") and name.endswith("__mean")
    )
    identity_tamper[mean_key][0] = 1.0
    _expect_value_error(
        ClassConditionalTemplateScoreModel.from_artifacts,
        scaler_arrays,
        identity_tamper,
    )


def test_strict_threshold_keeps_neutral_and_exact_threshold_ties_negative():
    scores = np.asarray([0.49, 0.5, 0.5000001, 0.9], dtype=np.float64)
    supported = np.asarray([True, True, True, False], dtype=bool)
    predictions = strict_threshold_predictions(
        scores, supported, threshold=0.5
    )
    assert predictions.tolist() == [False, False, True, False]


def test_input_contract_rejects_nonboolean_labels_missing_family_and_no_negatives():
    _expect_value_error(
        FamilyFitBatch,
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    )
    valid = {
        "a": _small_family_batch(0.0),
        "b": _small_family_batch(0.2),
    }
    _expect_value_error(
        ClassConditionalTemplateScoreModel,
        valid,
        family_order=("a", "c"),
        ks=(1,),
    )
    all_positive = {
        family: _batch([[1.0], [2.0]], [0, 0], [True, True])
        for family in ("a", "b", "c")
    }
    _expect_value_error(
        ClassConditionalTemplateScoreModel,
        all_positive,
        family_order=("a", "b", "c"),
        ks=(1,),
    )
