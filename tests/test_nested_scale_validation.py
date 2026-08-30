from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from contextlib import contextmanager
import inspect
import math
import re

import numpy as np

from pathline_template_matching.metrics import binary_metrics
from pathline_template_matching.nested_scale_validation import (
    DECISION_FIXED_TOP_FRACTION,
    DECISION_RANK_THRESHOLD,
    FROZEN_THRESHOLD_GRID,
    FROZEN_K_VALUES,
    CandidateSpec,
    InnerCandidateMacro,
    InnerGroupKey,
    aggregate_inner_group_metrics,
    evaluate_inner_group,
    fixed_top_fraction_predictions,
    representation_indices,
    select_inner_candidate,
    select_representation,
    spatial_rank_scores,
    spatial_support_scores,
    supported_rank_scores,
    threshold_predictions,
)


class _Approx:
    def __init__(self, expected: float, rel: float = 1.0e-12, abs_: float = 1.0e-12):
        self.expected = float(expected)
        self.rel = float(rel)
        self.abs = float(abs_)

    def __eq__(self, observed: object) -> bool:
        try:
            return math.isclose(
                float(observed), self.expected, rel_tol=self.rel, abs_tol=self.abs
            )
        except (TypeError, ValueError):
            return False


@contextmanager
def _raises(error_type: type[BaseException], match: str | None = None):
    try:
        yield
    except error_type as error:
        if match is not None and re.search(match, str(error)) is None:
            raise AssertionError(
                f"exception text {str(error)!r} did not match {match!r}"
            ) from error
        return
    raise AssertionError(f"expected {error_type.__name__}")


class _PytestCompatibility:
    raises = staticmethod(_raises)

    @staticmethod
    def approx(expected: float, rel: float = 1.0e-12, abs: float = 1.0e-12):
        return _Approx(expected, rel=rel, abs_=abs)


pytest = _PytestCompatibility()


def _threshold_candidate(
    *,
    representation: str = "fmt161",
    k: int = 5,
    sigma: float = 0.0,
    threshold: float = 0.5,
) -> CandidateSpec:
    return CandidateSpec(
        representation=representation,
        k=k,
        sigma=sigma,
        decision_rule=DECISION_RANK_THRESHOLD,
        decision_value=threshold,
    )


def _macro(candidate: CandidateSpec, **overrides: float) -> InnerCandidateMacro:
    values = {
        "average_precision": 0.7,
        "auroc": 0.8,
        "precision": 0.6,
        "recall": 0.5,
        "f1": 0.55,
        "balanced_accuracy": 0.65,
    }
    values.update(overrides)
    return InnerCandidateMacro(
        candidate=candidate,
        physical_families=("family_a", "family_b"),
        family_count=2,
        group_count=2,
        total_sample_count=10,
        total_positive_count=2,
        total_negative_count=8,
        **values,
    )


def test_frozen_representation_indices_and_matrix_contract():
    matrix = np.arange(2 * 161, dtype=np.float64).reshape(2, 161)
    full = select_representation(matrix, "fmt161")
    real_neighbor = select_representation(matrix, "real_neighbor36")
    chirality = select_representation(matrix, "chirality_all35")

    expected_real = tuple(
        line * 23 + local
        for line in range(1, 7)
        for local in (0, 3, 6, 9, 12, 15)
    )
    expected_chirality = tuple(
        line * 23 + local for line in range(7) for local in range(18, 23)
    )
    assert representation_indices("fmt161") == tuple(range(161))
    assert representation_indices("real_neighbor36") == expected_real
    assert representation_indices("chirality_all35") == expected_chirality
    assert full.shape == (2, 161)
    assert real_neighbor.shape == (2, 36)
    assert chirality.shape == (2, 35)
    np.testing.assert_array_equal(real_neighbor, matrix[:, expected_real])
    np.testing.assert_array_equal(chirality, matrix[:, expected_chirality])
    assert full.dtype == np.float32

    with pytest.raises(ValueError, match="N x 161"):
        select_representation(np.zeros((2, 160)), "fmt161")
    damaged = matrix.copy()
    damaged[0, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        select_representation(damaged, "fmt161")


def test_supported_only_rank_is_deterministic_and_keeps_unsupported_rows():
    distances = np.array([2.0, 2.0, np.nan, 1.0])
    centers = np.array([10, 5, 9, 7], dtype=np.int64)
    supported = np.array([True, True, False, True])

    ranked = supported_rank_scores(distances, centers, supported)

    np.testing.assert_allclose(ranked, [1.0, 2.0 / 3.0, 0.0, 1.0 / 3.0])
    assert ranked.shape == distances.shape
    assert ranked[2] == 0.0
    with pytest.raises(ValueError, match="supported raw distance"):
        supported_rank_scores([1.0, np.nan], [0, 1], [True, True])


def test_mask_normalized_gaussian_imputes_from_supported_ranks_only():
    centers = np.array([0, 1, 2], dtype=np.int64)
    supported = np.array([True, False, True])
    ranks = np.array([0.25, 0.0, 1.0])

    smoothed = spatial_rank_scores(ranks, centers, supported, sigma=1.0)

    assert smoothed.shape == ranks.shape
    # At center 1 the two supported centers have equal Gaussian weights.  The
    # unsupported center contributes neither a zero-valued sample nor mask mass.
    assert smoothed[1] == pytest.approx((0.25 + 1.0) / 2.0)
    assert smoothed[1] > 0.0
    np.testing.assert_array_equal(
        spatial_rank_scores(ranks, centers, supported, sigma=0.0), ranks
    )


def test_mask_normalized_gaussian_zero_denominator_fails_closed_to_zero():
    centers = np.array([0, 40**3 - 1], dtype=np.int64)
    supported = np.array([True, False])
    ranks = np.array([1.0, 0.0])
    result = spatial_rank_scores(ranks, centers, supported, sigma=0.5)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == 0.0

    all_missing = spatial_rank_scores(
        np.zeros(2), centers, np.zeros(2, dtype=bool), sigma=2.0
    )
    np.testing.assert_array_equal(all_missing, np.zeros(2))

    with pytest.raises(ValueError, match="duplicates"):
        spatial_rank_scores([1.0, 0.0], [1, 1], [True, False], sigma=1.0)
    with pytest.raises(ValueError, match="outside"):
        spatial_rank_scores([1.0], [40**3], [True], sigma=1.0)


def test_spatial_support_audit_partitions_supported_imputed_and_unimputable():
    centers = np.array([0, 1, 40**3 - 1], dtype=np.int64)
    support = np.array([True, False, False])
    result = spatial_support_scores(
        [2.0, np.nan, np.nan],
        centers,
        support,
        sigma=0.5,
    )
    np.testing.assert_array_equal(result.supported_mask, [True, False, False])
    np.testing.assert_array_equal(result.imputed_mask, [False, True, False])
    np.testing.assert_array_equal(result.unimputable_mask, [False, False, True])
    assert result.denominator[0] > 0.0
    assert result.denominator[1] > 0.0
    assert result.denominator[2] == 0.0
    assert result.scores[0] == pytest.approx(1.0)
    assert result.scores[1] == pytest.approx(1.0)
    assert result.scores[2] == 0.0
    assert not result.scores.flags.writeable

    sigma_zero = spatial_support_scores(
        [2.0, np.nan, np.nan], centers, support, sigma=0.0
    )
    np.testing.assert_array_equal(sigma_zero.imputed_mask, np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(
        sigma_zero.unimputable_mask, [False, True, True]
    )


def test_fixed_top_fraction_uses_exact_ceil_and_center_tie_break():
    scores = np.array([0.5, 0.5, 0.2, 0.0])
    centers = np.array([10, 5, 7, 8])
    eligible = np.ones(4, dtype=bool)
    one = fixed_top_fraction_predictions(scores, centers, eligible, fraction=0.25)
    two = fixed_top_fraction_predictions(scores, centers, eligible, fraction=0.26)
    none = fixed_top_fraction_predictions(scores, centers, eligible, fraction=0.0)

    np.testing.assert_array_equal(one, [False, True, False, False])
    np.testing.assert_array_equal(two, [True, True, False, False])
    assert not none.any()

    # A high-score ineligible row and a zero-score eligible row cannot be made
    # positive; the all-row target is capped by usable score sources.
    capped = fixed_top_fraction_predictions(
        [9.0, 0.8, 0.0], [0, 1, 2], [False, True, True], fraction=1.0
    )
    np.testing.assert_array_equal(capped, [False, True, False])


def test_threshold_grid_candidate_id_and_immutability_are_frozen():
    assert FROZEN_K_VALUES == (1, 5, 15, 31)
    assert _threshold_candidate(k=1).k == 1
    assert len(FROZEN_THRESHOLD_GRID) == 50
    assert FROZEN_THRESHOLD_GRID[0] == 0.5
    assert FROZEN_THRESHOLD_GRID[-1] == pytest.approx(0.99)
    np.testing.assert_array_equal(
        threshold_predictions(
            [0.49, 0.5, 0.8], [True, True, False], threshold=0.5
        ),
        [False, True, False],
    )
    assert 0.57 in FROZEN_THRESHOLD_GRID
    assert threshold_predictions([0.57], [True], threshold=0.57)[0]
    with pytest.raises(ValueError, match="exact member"):
        threshold_predictions([0.5], [True], threshold=0.5000001)

    candidate = CandidateSpec(
        representation="real_neighbor36",
        k=15,
        sigma=1.5,
        decision_rule=DECISION_FIXED_TOP_FRACTION,
        decision_value=0.05,
    )
    assert candidate.candidate_id == (
        "real_neighbor36__k=15__sigma=1.50__"
        "rule=fixed_top_fraction__value=0.05"
    )
    assert candidate.candidate_id == replace(candidate).candidate_id
    with pytest.raises(FrozenInstanceError):
        candidate.k = 31  # type: ignore[misc]


def test_group_macro_is_equal_weighted_not_pooled():
    candidate = _threshold_candidate()
    first = evaluate_inner_group(
        candidate,
        "family_a",
        InnerGroupKey("small", 0, 0),
        inner_labels=np.array([1, 0]),
        scores=np.array([1.0, 0.0]),
        center_indices=np.array([0, 1]),
        eligible_mask=np.ones(2, dtype=bool),
    )
    second_labels = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    second_scores = np.zeros(8)
    second = evaluate_inner_group(
        candidate,
        "family_b",
        InnerGroupKey("large", 0, 0),
        inner_labels=second_labels,
        scores=second_scores,
        center_indices=np.arange(8),
        eligible_mask=np.ones(8, dtype=bool),
    )
    third = evaluate_inner_group(
        candidate,
        "family_a",
        InnerGroupKey("small_again", 0, 0),
        inner_labels=np.array([1, 0]),
        scores=np.array([1.0, 0.0]),
        center_indices=np.array([2, 3]),
        eligible_mask=np.ones(2, dtype=bool),
    )

    macro = aggregate_inner_group_metrics([first, second, third])
    pooled = binary_metrics(
        np.r_[np.array([1, 0]), second_labels, np.array([1, 0])],
        np.r_[
            np.array([True, False]),
            np.zeros(8, dtype=bool),
            np.array([True, False]),
        ],
        np.r_[np.array([1.0, 0.0]), second_scores, np.array([1.0, 0.0])],
    )
    assert first.metrics.f1 == 1.0
    assert second.metrics.f1 == 0.0
    assert macro.group_count == 3
    assert macro.family_count == 2
    assert macro.physical_families == ("family_a", "family_b")
    assert macro.f1 == pytest.approx(0.5)
    assert pooled.f1 == pytest.approx(0.8)
    assert macro.f1 != pooled.f1
    assert macro.f1 != pytest.approx(2.0 / 3.0)  # not flat over three groups


def test_selector_tie_break_order_and_direction_are_exact():
    base = _threshold_candidate(representation="fmt161", k=5, sigma=0.0)
    alternatives = [
        _threshold_candidate(representation="fmt161", k=15, sigma=0.0),
        _threshold_candidate(representation="fmt161", k=31, sigma=0.0),
        _threshold_candidate(representation="real_neighbor36", k=5, sigma=0.0),
        _threshold_candidate(representation="chirality_all35", k=5, sigma=0.0),
    ]

    assert select_inner_candidate(
        [_macro(base, f1=0.7), _macro(alternatives[0], f1=0.71)]
    ).candidate == alternatives[0]
    assert select_inner_candidate(
        [
            _macro(base, f1=0.7, average_precision=0.8),
            _macro(alternatives[0], f1=0.7, average_precision=0.81),
        ]
    ).candidate == alternatives[0]
    assert select_inner_candidate(
        [
            _macro(base, f1=0.7, average_precision=0.8, balanced_accuracy=0.8),
            _macro(
                alternatives[1],
                f1=0.7,
                average_precision=0.8,
                balanced_accuracy=0.81,
            ),
        ]
    ).candidate == alternatives[1]
    assert select_inner_candidate(
        [
            _macro(base, f1=0.7, average_precision=0.8, precision=0.8),
            _macro(
                alternatives[2], f1=0.7, average_precision=0.8, precision=0.81
            ),
        ]
    ).candidate == alternatives[2]
    assert select_inner_candidate(
        [
            _macro(base, f1=0.7, average_precision=0.8, recall=0.8),
            _macro(
                alternatives[3], f1=0.7, average_precision=0.8, recall=0.81
            ),
        ]
    ).candidate == alternatives[3]

    exact_tie = [_macro(base), _macro(alternatives[3])]
    selected = select_inner_candidate(exact_tie)
    assert selected.candidate_id == min(row.candidate_id for row in exact_tie)

    incomplete = replace(
        _macro(alternatives[0]),
        physical_families=("family_a",),
        family_count=1,
    )
    with pytest.raises(ValueError, match="same complete physical-family set"):
        select_inner_candidate([_macro(base), incomplete])


def test_selection_api_cannot_accept_outer_fields_or_outer_labels():
    candidate = _threshold_candidate()
    macro = _macro(candidate)
    signature = inspect.signature(select_inner_candidate)
    assert tuple(signature.parameters) == ("inner_candidate_macros",)
    assert "label" not in str(signature).lower()
    assert "outer" not in str(signature).lower()
    with pytest.raises(TypeError):
        select_inner_candidate([macro], outer_labels=np.array([1, 0]))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        InnerGroupKey("flow", 0, 0, outer_family="held_out")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="only InnerCandidateMacro"):
        select_inner_candidate([{"outer_labels": [1, 0]}])  # type: ignore[list-item]
