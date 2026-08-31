"""Family-balanced positive/negative exact-scale template conformity.

This module implements the scoring change proposed for
``Verify_ClassConditionalTemplateScore_1.1``.  It deliberately reuses the
frozen fit-negative per-scale metric and tail calibration.  A single scaler is
fitted from all negative rows in the fit families.  After that transform, one
independent exact-scale tail calibrator is fitted for every
``physical family x class`` population.

For a query, a family contributes only when both its positive and negative
calibrators have exact retrieval and tail-calibration support.  Supported
families receive equal weight, irrespective of their template counts.  If a
strict majority of fit families contributes, the classifier score is

``0.5 * (1 + mean(positive conformity) - mean(negative conformity))``.

The empirical tail values are conformity scores.  Cross-scale shrinkage or
fallback priors mean they must not be described as exact conformal p-values or
posterior probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch

from .negative_tail_calibration import (
    DEFAULT_KS,
    SCALE_COUNT,
    SHRINKAGE_LAMBDA,
    ScaleConditionedNegativeTailCalibrator,
    _validated_lambda,
    _validated_scale_ids,
)
from .per_scale_negative_metric import (
    SCALER_ARRAY_NAMES,
    PerScaleNegativeScaler,
    _identity_tail_calibrator,
)
from .scale_one_class import _feature_matrix, _ks, _positive_integer


NEGATIVE_CLASS_INDEX = 0
POSITIVE_CLASS_INDEX = 1
CLASS_NAMES = ("negative", "positive")

_SERIALIZATION_VERSION = 1
_CALIBRATOR_BASE_ARRAY_NAMES = (
    "serialization_version_int16",
    "family_order_unicode",
    "family_order_copy_unicode",
    "required_family_count_int64",
    "ks_int64",
    "shrinkage_lambda_float64",
    "class_present_bool",
    "class_scale_counts_int64",
)
_SCALER_PREFIX = "scaler__"


def _freeze_array(
    values: object, *, dtype: np.dtype | type | None = None
) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _validated_family_order(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("family_order must be a non-empty sequence of names")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("family_order must be a non-empty sequence of names") from error
    if not items:
        raise ValueError("family_order must be a non-empty sequence of names")
    if any(
        not isinstance(item, str) or not item or "\x00" in item for item in items
    ):
        raise ValueError("each family name must be a non-empty string without NUL")
    if len(set(items)) != len(items):
        raise ValueError("family_order must not contain duplicate names")
    return items


def _strict_majority(family_count: int) -> int:
    return family_count // 2 + 1


def _tail_array_names(ks: Sequence[int]) -> tuple[str, ...]:
    fitted = _ks(ks)
    base = (
        "serialization_version",
        "ks",
        "shrinkage_lambda",
        "mean",
        "raw_std",
        "effective_std",
        "zero_variance_feature_mask",
        "negative_features",
        "negative_scale_offsets",
    )
    dynamic = tuple(
        name
        for k in fitted
        for name in (f"loo_distances_k_{k}", f"loo_scale_offsets_k_{k}")
    )
    return base + dynamic


def _calibrator_prefix(family_index: int, class_index: int) -> str:
    return f"calibrator_f{family_index}_c{class_index}__"


@dataclass(frozen=True, slots=True)
class FamilyFitBatch:
    """One physical family's fit-only features, exact scales, and labels."""

    features: np.ndarray
    scale_ids: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        features = _feature_matrix(
            self.features, name="features", allow_empty=False
        )
        scales = _validated_scale_ids(self.scale_ids, name="scale_ids")
        labels = np.asarray(self.labels)
        if labels.ndim != 1 or labels.dtype != np.dtype(np.bool_):
            raise ValueError("labels must be a one-dimensional boolean array")
        if len(features) != len(scales) or len(features) != len(labels):
            raise ValueError(
                "features, scale_ids, and labels must contain the same rows"
            )
        object.__setattr__(self, "features", _freeze_array(features, dtype=np.float32))
        object.__setattr__(self, "scale_ids", _freeze_array(scales, dtype=np.int64))
        object.__setattr__(self, "labels", _freeze_array(labels, dtype=np.bool_))

    @property
    def count(self) -> int:
        return len(self.features)

    @property
    def dimension(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True, slots=True)
class CombinedClassConditionalScore:
    """One neighbour count's immutable family-balanced score arrays."""

    positive_conformity: np.ndarray
    negative_conformity: np.ndarray
    scores: np.ndarray
    supported: np.ndarray
    supporting_family_counts: np.ndarray
    required_family_count: int
    fit_family_count: int

    def __post_init__(self) -> None:
        positive = np.asarray(self.positive_conformity)
        negative = np.asarray(self.negative_conformity)
        scores = np.asarray(self.scores)
        supported = np.asarray(self.supported)
        counts = np.asarray(self.supporting_family_counts)
        required = _positive_integer(
            self.required_family_count, name="required_family_count"
        )
        family_count = _positive_integer(
            self.fit_family_count, name="fit_family_count"
        )
        if required != _strict_majority(family_count):
            raise ValueError("required_family_count must be the strict majority")
        if (
            positive.ndim != 1
            or negative.shape != positive.shape
            or scores.shape != positive.shape
            or supported.shape != positive.shape
            or counts.shape != positive.shape
            or positive.dtype != np.dtype(np.float64)
            or negative.dtype != np.dtype(np.float64)
            or scores.dtype != np.dtype(np.float64)
            or supported.dtype != np.dtype(np.bool_)
            or counts.dtype != np.dtype(np.int64)
        ):
            raise ValueError("combined score arrays have invalid shapes or dtypes")
        if (
            not np.isfinite(positive).all()
            or not np.isfinite(negative).all()
            or not np.isfinite(scores).all()
            or np.any((positive < 0.0) | (positive > 1.0))
            or np.any((negative < 0.0) | (negative > 1.0))
            or np.any((scores < 0.0) | (scores > 1.0))
            or np.any((counts < 0) | (counts > family_count))
            or not np.array_equal(supported, counts >= required)
        ):
            raise ValueError("combined score arrays contain invalid values")
        expected_scores = np.zeros(len(scores), dtype=np.float64)
        expected_scores[supported] = 0.5 * (
            1.0 + positive[supported] - negative[supported]
        )
        if (
            not np.array_equal(scores, expected_scores)
            or np.any(positive[~supported] != 0.0)
            or np.any(negative[~supported] != 0.0)
        ):
            raise ValueError("combined score formula or unsupported sentinel drifted")
        object.__setattr__(
            self, "positive_conformity", _freeze_array(positive, dtype=np.float64)
        )
        object.__setattr__(
            self, "negative_conformity", _freeze_array(negative, dtype=np.float64)
        )
        object.__setattr__(self, "scores", _freeze_array(scores, dtype=np.float64))
        object.__setattr__(self, "supported", _freeze_array(supported, dtype=np.bool_))
        object.__setattr__(
            self,
            "supporting_family_counts",
            _freeze_array(counts, dtype=np.int64),
        )
        object.__setattr__(self, "required_family_count", required)
        object.__setattr__(self, "fit_family_count", family_count)


def _finalize_family_sums(
    positive_sums: np.ndarray,
    negative_sums: np.ndarray,
    family_counts: np.ndarray,
    *,
    required_family_count: int,
    fit_family_count: int,
) -> CombinedClassConditionalScore:
    positive_sum = np.asarray(positive_sums, dtype=np.float64)
    negative_sum = np.asarray(negative_sums, dtype=np.float64)
    counts = np.asarray(family_counts, dtype=np.int64)
    if (
        positive_sum.ndim != 1
        or negative_sum.shape != positive_sum.shape
        or counts.shape != positive_sum.shape
        or not np.isfinite(positive_sum).all()
        or not np.isfinite(negative_sum).all()
        or np.any(positive_sum < 0.0)
        or np.any(negative_sum < 0.0)
    ):
        raise ValueError("family conformity accumulators are invalid")
    supported = counts >= required_family_count
    positive = np.zeros(len(counts), dtype=np.float64)
    negative = np.zeros(len(counts), dtype=np.float64)
    positive[supported] = positive_sum[supported] / counts[supported]
    negative[supported] = negative_sum[supported] / counts[supported]
    scores = np.zeros(len(counts), dtype=np.float64)
    scores[supported] = 0.5 * (
        1.0 + positive[supported] - negative[supported]
    )
    return CombinedClassConditionalScore(
        positive_conformity=positive,
        negative_conformity=negative,
        scores=scores,
        supported=supported,
        supporting_family_counts=counts,
        required_family_count=required_family_count,
        fit_family_count=fit_family_count,
    )


def combine_joint_family_conformity(
    positive_conformity: object,
    negative_conformity: object,
    positive_retrieval_supported: object,
    positive_calibration_supported: object,
    negative_retrieval_supported: object,
    negative_calibration_supported: object,
    *,
    required_family_count: int,
) -> CombinedClassConditionalScore:
    """Combine family/class tails using one common, jointly supported set.

    Arrays have shape ``(fit_family_count, query_count)``.  A prior may make
    calibration possible only after an exact distance exists; consequently
    calibration support is required to be a subset of retrieval support.
    """

    positive = np.asarray(positive_conformity)
    negative = np.asarray(negative_conformity)
    masks = tuple(
        np.asarray(values)
        for values in (
            positive_retrieval_supported,
            positive_calibration_supported,
            negative_retrieval_supported,
            negative_calibration_supported,
        )
    )
    if (
        positive.ndim != 2
        or positive.shape[0] < 1
        or negative.shape != positive.shape
        or not np.issubdtype(positive.dtype, np.number)
        or not np.issubdtype(negative.dtype, np.number)
        or np.issubdtype(positive.dtype, np.bool_)
        or np.issubdtype(negative.dtype, np.bool_)
        or any(mask.shape != positive.shape for mask in masks)
        or any(mask.dtype != np.dtype(np.bool_) for mask in masks)
    ):
        raise ValueError("family conformity/support arrays have invalid contracts")
    positive64 = np.asarray(positive, dtype=np.float64)
    negative64 = np.asarray(negative, dtype=np.float64)
    if (
        not np.isfinite(positive64).all()
        or not np.isfinite(negative64).all()
        or np.any((positive64 < 0.0) | (positive64 > 1.0))
        or np.any((negative64 < 0.0) | (negative64 > 1.0))
    ):
        raise ValueError("family conformity values must be finite within [0, 1]")
    positive_retrieval, positive_calibration, negative_retrieval, negative_calibration = masks
    if np.any(positive_calibration & ~positive_retrieval) or np.any(
        negative_calibration & ~negative_retrieval
    ):
        raise ValueError("calibration support cannot exceed retrieval support")
    family_count = int(positive.shape[0])
    required = _positive_integer(
        required_family_count, name="required_family_count"
    )
    if required != _strict_majority(family_count):
        raise ValueError("required_family_count must be the strict majority")
    joint = (
        positive_retrieval
        & positive_calibration
        & negative_retrieval
        & negative_calibration
    )
    counts = joint.sum(axis=0, dtype=np.int64)
    positive_sums = np.where(joint, positive64, 0.0).sum(
        axis=0, dtype=np.float64
    )
    negative_sums = np.where(joint, negative64, 0.0).sum(
        axis=0, dtype=np.float64
    )
    return _finalize_family_sums(
        positive_sums,
        negative_sums,
        counts,
        required_family_count=required,
        fit_family_count=family_count,
    )


@dataclass(frozen=True, slots=True)
class ClassConditionalQueryResult:
    """Immutable query outputs keyed by fitted neighbour count."""

    scores: Mapping[int, np.ndarray]
    mean_negative_distances: Mapping[int, np.ndarray]
    retrieval_supported: Mapping[int, np.ndarray]
    joint_supported: Mapping[int, np.ndarray]
    joint_family_count: Mapping[int, np.ndarray]
    positive_conformity: Mapping[int, np.ndarray]
    negative_conformity: Mapping[int, np.ndarray]
    per_family_positive_retrieval_supported: Mapping[int, np.ndarray]
    per_family_positive_calibration_supported: Mapping[int, np.ndarray]
    per_family_negative_retrieval_supported: Mapping[int, np.ndarray]
    per_family_negative_calibration_supported: Mapping[int, np.ndarray]
    required_family_count: int
    fit_family_count: int

    def __post_init__(self) -> None:
        mappings = (
            self.scores,
            self.mean_negative_distances,
            self.retrieval_supported,
            self.joint_supported,
            self.joint_family_count,
            self.positive_conformity,
            self.negative_conformity,
            self.per_family_positive_retrieval_supported,
            self.per_family_positive_calibration_supported,
            self.per_family_negative_retrieval_supported,
            self.per_family_negative_calibration_supported,
        )
        keys = tuple(mappings[0])
        if not keys or any(tuple(mapping) != keys for mapping in mappings[1:]):
            raise ValueError("all class-conditional result mappings must share keys")
        frozen = [dict() for _ in mappings]
        expected_length: int | None = None
        for k in keys:
            distances = np.asarray(mappings[1][k])
            retrieval = np.asarray(mappings[2][k])
            joint = np.asarray(mappings[3][k])
            counts16 = np.asarray(mappings[4][k])
            family_masks = tuple(np.asarray(mapping[k]) for mapping in mappings[7:])
            if (
                distances.ndim != 1
                or retrieval.shape != distances.shape
                or joint.shape != distances.shape
                or counts16.shape != distances.shape
                or distances.dtype != np.dtype(np.float32)
                or retrieval.dtype != np.dtype(np.bool_)
                or joint.dtype != np.dtype(np.bool_)
                or counts16.dtype != np.dtype(np.int16)
                or np.any(counts16 < 0)
                or np.any(counts16 > self.fit_family_count)
                or np.any(joint & ~retrieval)
                or not np.isfinite(distances[retrieval]).all()
                or np.any(distances[retrieval] < 0.0)
                or not np.isnan(distances[~retrieval]).all()
                or any(
                    mask.shape != (len(distances), self.fit_family_count)
                    or mask.dtype != np.dtype(np.bool_)
                    for mask in family_masks
                )
            ):
                raise ValueError("query distance/support audit arrays are invalid")
            (
                positive_retrieval,
                positive_calibration,
                negative_retrieval,
                negative_calibration,
            ) = family_masks
            if np.any(positive_calibration & ~positive_retrieval) or np.any(
                negative_calibration & ~negative_retrieval
            ):
                raise ValueError("per-family calibration support exceeds retrieval")
            family_retrieval = positive_retrieval & negative_retrieval
            family_joint = (
                family_retrieval
                & positive_calibration
                & negative_calibration
            )
            if (
                not np.array_equal(
                    retrieval,
                    family_retrieval.sum(axis=1, dtype=np.int64)
                    >= self.required_family_count,
                )
                or not np.array_equal(
                    counts16,
                    family_joint.sum(axis=1, dtype=np.int64).astype(np.int16),
                )
            ):
                raise ValueError("per-family support does not reproduce joint audits")
            combined = CombinedClassConditionalScore(
                positive_conformity=np.asarray(mappings[5][k]),
                negative_conformity=np.asarray(mappings[6][k]),
                scores=np.asarray(mappings[0][k]),
                supported=joint,
                supporting_family_counts=np.asarray(counts16, dtype=np.int64),
                required_family_count=self.required_family_count,
                fit_family_count=self.fit_family_count,
            )
            if expected_length is None:
                expected_length = len(combined.scores)
            elif len(combined.scores) != expected_length:
                raise ValueError("all query-result arrays must have identical length")
            values = (
                combined.scores,
                _freeze_array(distances, dtype=np.float32),
                _freeze_array(retrieval, dtype=np.bool_),
                combined.supported,
                _freeze_array(counts16, dtype=np.int16),
                combined.positive_conformity,
                combined.negative_conformity,
                *(
                    _freeze_array(mask, dtype=np.bool_)
                    for mask in family_masks
                ),
            )
            for output, value in zip(frozen, values):
                output[int(k)] = value
        names = (
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
        )
        for name, values in zip(names, frozen):
            object.__setattr__(self, name, MappingProxyType(values))

    @property
    def supported(self) -> Mapping[int, np.ndarray]:
        """Compatibility alias for the stricter joint calibration support."""

        return self.joint_supported

    @property
    def supporting_family_counts(self) -> Mapping[int, np.ndarray]:
        """Compatibility alias for ``joint_family_count``."""

        return self.joint_family_count


def strict_threshold_predictions(
    scores: object,
    supported: object,
    *,
    threshold: float,
) -> np.ndarray:
    """Return ``score > threshold`` predictions; exact ties are negative."""

    values = np.asarray(scores)
    mask = np.asarray(supported)
    if (
        values.ndim != 1
        or mask.shape != values.shape
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.bool_)
        or mask.dtype != np.dtype(np.bool_)
    ):
        raise ValueError("scores and supported must be matching one-dimensional arrays")
    values64 = np.asarray(values, dtype=np.float64)
    try:
        selected = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be finite within [0, 1]") from error
    if (
        isinstance(threshold, (bool, np.bool_))
        or not np.isfinite(values64).all()
        or np.any((values64 < 0.0) | (values64 > 1.0))
        or not np.isfinite(selected)
        or selected < 0.0
        or selected > 1.0
    ):
        raise ValueError("scores and threshold must be finite within [0, 1]")
    return np.ascontiguousarray(mask & (values64 > selected), dtype=np.bool_)


def _require_identity_calibrator_state(
    calibrator: ScaleConditionedNegativeTailCalibrator,
    *,
    expected_scale_counts: np.ndarray,
    dimension: int,
    ks: tuple[int, ...],
    shrinkage_lambda: float,
) -> None:
    arrays = calibrator.export_arrays()
    if (
        calibrator.ks != ks
        or not np.array_equal(arrays["ks"], np.asarray(ks, dtype=np.int64))
        or not np.array_equal(arrays["mean"], np.zeros(dimension, dtype=np.float64))
        or not np.array_equal(
            arrays["raw_std"], np.ones(dimension, dtype=np.float64)
        )
        or not np.array_equal(
            arrays["effective_std"], np.ones(dimension, dtype=np.float64)
        )
        or np.any(arrays["zero_variance_feature_mask"])
        or arrays["shrinkage_lambda"].shape != ()
        or arrays["shrinkage_lambda"].dtype != np.dtype(np.float64)
        or float(arrays["shrinkage_lambda"]) != shrinkage_lambda
        or not np.array_equal(
            np.diff(arrays["negative_scale_offsets"]), expected_scale_counts
        )
        or arrays["negative_features"].shape
        != (int(expected_scale_counts.sum(dtype=np.int64)), dimension)
    ):
        raise ValueError(
            "class calibrator is not the required identity-tail state for its family/class"
        )


class FamilyClassTailCalibratorBundle:
    """Read-only family/class calibrator collection used by runner artifacts."""

    def __init__(self, model: "ClassConditionalTemplateScoreModel") -> None:
        self._model = model

    @property
    def fit_audit(self) -> dict[str, object]:
        return self._model._tail_fit_audit()

    def export_arrays(self) -> dict[str, np.ndarray]:
        return self._model._export_calibrator_arrays()

    def calibrator_for(
        self, family: str, *, positive: bool
    ) -> ScaleConditionedNegativeTailCalibrator | None:
        return self._model.calibrator_for(family, positive=positive)


class ClassConditionalTemplateScoreModel:
    """Fit-only class-conditional template scorer with equal family weights."""

    def __init__(
        self,
        family_batches: Mapping[str, FamilyFitBatch],
        *,
        family_order: Sequence[str],
        ks: Sequence[int] = DEFAULT_KS,
        shrinkage_lambda: float = SHRINKAGE_LAMBDA,
        device: str | torch.device = "cpu",
        query_chunk_size: int = 1024,
        library_chunk_size: int = 8192,
    ) -> None:
        if not isinstance(family_batches, Mapping):
            raise ValueError("family_batches must map family names to FamilyFitBatch")
        families = _validated_family_order(family_order)
        if set(family_batches) != set(families):
            raise ValueError("family_batches keys must exactly match family_order")
        batches: list[FamilyFitBatch] = []
        for family in families:
            batch = family_batches[family]
            if not isinstance(batch, FamilyFitBatch):
                raise ValueError("each family batch must be a FamilyFitBatch")
            batches.append(batch)
        dimensions = {batch.dimension for batch in batches}
        if len(dimensions) != 1:
            raise ValueError("all family feature matrices must have the same width")
        dimension = dimensions.pop()
        fitted_ks = _ks(ks)
        selected_lambda = _validated_lambda(shrinkage_lambda)

        negative_features = [
            batch.features[~batch.labels] for batch in batches if (~batch.labels).any()
        ]
        negative_scales = [
            batch.scale_ids[~batch.labels] for batch in batches if (~batch.labels).any()
        ]
        if not negative_features:
            raise ValueError("fit families contain no negative templates")
        pooled_negative_features = np.ascontiguousarray(
            np.concatenate(negative_features, axis=0), dtype=np.float32
        )
        pooled_negative_scales = np.ascontiguousarray(
            np.concatenate(negative_scales), dtype=np.int64
        )
        scaler = PerScaleNegativeScaler(
            pooled_negative_features,
            pooled_negative_scales,
            shrinkage_lambda=selected_lambda,
        )
        del pooled_negative_features, pooled_negative_scales

        class_counts = np.zeros(
            (len(families), len(CLASS_NAMES), SCALE_COUNT), dtype=np.int64
        )
        calibrators: list[
            tuple[
                ScaleConditionedNegativeTailCalibrator | None,
                ScaleConditionedNegativeTailCalibrator | None,
            ]
        ] = []
        for family_index, batch in enumerate(batches):
            transformed = scaler.transform(batch.features, batch.scale_ids)
            current: list[ScaleConditionedNegativeTailCalibrator | None] = []
            for class_index in (NEGATIVE_CLASS_INDEX, POSITIVE_CLASS_INDEX):
                selected_rows = batch.labels == bool(class_index)
                selected_scales = np.asarray(
                    batch.scale_ids[selected_rows], dtype=np.int64
                )
                class_counts[family_index, class_index] = np.bincount(
                    selected_scales, minlength=SCALE_COUNT
                ).astype(np.int64, copy=False)
                if len(selected_scales) == 0:
                    current.append(None)
                    continue
                current.append(
                    _identity_tail_calibrator(
                        np.ascontiguousarray(
                            transformed[selected_rows], dtype=np.float32
                        ),
                        np.ascontiguousarray(selected_scales, dtype=np.int64),
                        ks=fitted_ks,
                        shrinkage_lambda=selected_lambda,
                        device=device,
                        query_chunk_size=query_chunk_size,
                        library_chunk_size=library_chunk_size,
                    )
                )
            calibrators.append((current[0], current[1]))
            del transformed

        self._install_state(
            family_order=families,
            ks=fitted_ks,
            scaler=scaler,
            calibrators=tuple(calibrators),
            class_scale_counts=class_counts,
            dimension=dimension,
            shrinkage_lambda=selected_lambda,
        )

    def _install_state(
        self,
        *,
        family_order: tuple[str, ...],
        ks: tuple[int, ...],
        scaler: PerScaleNegativeScaler,
        calibrators: tuple[
            tuple[
                ScaleConditionedNegativeTailCalibrator | None,
                ScaleConditionedNegativeTailCalibrator | None,
            ],
            ...,
        ],
        class_scale_counts: np.ndarray,
        dimension: int,
        shrinkage_lambda: float,
    ) -> None:
        families = _validated_family_order(family_order)
        fitted_ks = _ks(ks)
        selected_lambda = _validated_lambda(shrinkage_lambda)
        counts = np.asarray(class_scale_counts)
        if not isinstance(scaler, PerScaleNegativeScaler):
            raise ValueError("scaler must be a PerScaleNegativeScaler")
        if len(families) > np.iinfo(np.int16).max:
            raise ValueError("fit family count exceeds the int16 query-audit contract")
        if scaler.dimension != dimension or scaler.shrinkage_lambda != selected_lambda:
            raise ValueError("scaler dimension or shrinkage lambda drifted")
        if (
            counts.shape != (len(families), len(CLASS_NAMES), SCALE_COUNT)
            or counts.dtype != np.dtype(np.int64)
            or np.any(counts < 0)
            or len(calibrators) != len(families)
            or any(len(pair) != len(CLASS_NAMES) for pair in calibrators)
        ):
            raise ValueError("family/class calibrator state has an invalid contract")
        if not np.array_equal(
            counts[:, NEGATIVE_CLASS_INDEX].sum(axis=0, dtype=np.int64),
            scaler.local_row_counts,
        ):
            raise ValueError("pooled negative scaler counts do not match family libraries")
        for family_index, pair in enumerate(calibrators):
            for class_index, calibrator in enumerate(pair):
                expected = counts[family_index, class_index]
                if int(expected.sum(dtype=np.int64)) == 0:
                    if calibrator is not None:
                        raise ValueError("an absent class must not have a calibrator")
                    continue
                if calibrator is None:
                    raise ValueError("a present class must have a calibrator")
                _require_identity_calibrator_state(
                    calibrator,
                    expected_scale_counts=expected,
                    dimension=dimension,
                    ks=fitted_ks,
                    shrinkage_lambda=selected_lambda,
                )
        self._family_order = families
        self._family_index = MappingProxyType(
            {family: index for index, family in enumerate(families)}
        )
        self._ks = fitted_ks
        self._required_family_count = _strict_majority(len(families))
        self._scaler = scaler
        self._calibrators = calibrators
        self._class_scale_counts = _freeze_array(counts, dtype=np.int64)
        self._dimension = int(dimension)
        self._shrinkage_lambda = selected_lambda
        self._tail_calibrator = FamilyClassTailCalibratorBundle(self)

    @property
    def family_order(self) -> tuple[str, ...]:
        return self._family_order

    @property
    def ks(self) -> tuple[int, ...]:
        return self._ks

    @property
    def required_family_count(self) -> int:
        return self._required_family_count

    @property
    def scaler(self) -> PerScaleNegativeScaler:
        return self._scaler

    @property
    def tail_calibrator(self) -> FamilyClassTailCalibratorBundle:
        return self._tail_calibrator

    def _tail_fit_audit(self) -> dict[str, object]:
        family_rows: dict[str, dict[str, int]] = {}
        calibration: dict[str, dict[str, object | None]] = {}
        for family_index, family in enumerate(self._family_order):
            family_rows[family] = {
                CLASS_NAMES[class_index]: int(
                    self._class_scale_counts[family_index, class_index].sum(
                        dtype=np.int64
                    )
                )
                for class_index in range(len(CLASS_NAMES))
            }
            calibration[family] = {
                CLASS_NAMES[class_index]: (
                    None if calibrator is None else calibrator.fit_audit
                )
                for class_index, calibrator in enumerate(
                    self._calibrators[family_index]
                )
            }
        return {
            "family_order": list(self._family_order),
            "fit_family_count": len(self._family_order),
            "required_family_count": self._required_family_count,
            "feature_dimension": self._dimension,
            "ks": list(self._ks),
            "shrinkage_lambda": self._shrinkage_lambda,
            "score_semantics": "family_balanced_positive_negative_tail_conformity",
            "probability_claim": False,
            "family_class_rows": family_rows,
            "family_class_calibration": calibration,
        }

    @property
    def fit_audit(self) -> dict[str, object]:
        result = self._tail_fit_audit()
        result["per_scale_negative_scaler"] = self._scaler.fit_audit
        return result

    def calibrator_for(
        self, family: str, *, positive: bool
    ) -> ScaleConditionedNegativeTailCalibrator | None:
        if family not in self._family_index:
            raise ValueError(f"unknown fit family: {family!r}")
        if not isinstance(positive, (bool, np.bool_)):
            raise ValueError("positive must be boolean")
        return self._calibrators[self._family_index[family]][int(positive)]

    def query(
        self,
        features: object,
        scale_ids: object,
        *,
        ks: Sequence[int] | None = None,
        device: str | torch.device = "cpu",
        query_chunk_size: int = 1024,
        library_chunk_size: int = 8192,
    ) -> ClassConditionalQueryResult:
        scales = _validated_scale_ids(scale_ids, name="scale_ids")
        transformed = self._scaler.transform(features, scales)
        requested = self._ks if ks is None else _ks(ks)
        if any(k not in self._ks for k in requested):
            raise ValueError("query ks must be a subset of fitted ks")
        positive_sums = {
            k: np.zeros(len(scales), dtype=np.float64) for k in requested
        }
        negative_sums = {
            k: np.zeros(len(scales), dtype=np.float64) for k in requested
        }
        family_counts = {
            k: np.zeros(len(scales), dtype=np.int64) for k in requested
        }
        retrieval_family_counts = {
            k: np.zeros(len(scales), dtype=np.int64) for k in requested
        }
        negative_distance_sums = {
            k: np.zeros(len(scales), dtype=np.float64) for k in requested
        }
        per_family_positive_retrieval = {
            k: np.zeros((len(scales), len(self._family_order)), dtype=np.bool_)
            for k in requested
        }
        per_family_positive_calibration = {
            k: np.zeros((len(scales), len(self._family_order)), dtype=np.bool_)
            for k in requested
        }
        per_family_negative_retrieval = {
            k: np.zeros((len(scales), len(self._family_order)), dtype=np.bool_)
            for k in requested
        }
        per_family_negative_calibration = {
            k: np.zeros((len(scales), len(self._family_order)), dtype=np.bool_)
            for k in requested
        }
        scaler_supported = np.asarray(
            self._scaler.local_row_counts[scales] > 0, dtype=np.bool_
        )
        for family_index, (
            negative_calibrator,
            positive_calibrator,
        ) in enumerate(self._calibrators):
            negative = (
                None
                if negative_calibrator is None
                else negative_calibrator.query(
                    transformed,
                    scales,
                    ks=requested,
                    device=device,
                    query_chunk_size=query_chunk_size,
                    library_chunk_size=library_chunk_size,
                )
            )
            positive = (
                None
                if positive_calibrator is None
                else positive_calibrator.query(
                    transformed,
                    scales,
                    ks=requested,
                    device=device,
                    query_chunk_size=query_chunk_size,
                    library_chunk_size=library_chunk_size,
                )
            )
            for k in requested:
                if positive is not None:
                    per_family_positive_retrieval[k][:, family_index] = (
                        positive.retrieval_supported[k] & scaler_supported
                    )
                    per_family_positive_calibration[k][:, family_index] = (
                        positive.calibration_supported[k] & scaler_supported
                    )
                if negative is not None:
                    per_family_negative_retrieval[k][:, family_index] = (
                        negative.retrieval_supported[k] & scaler_supported
                    )
                    per_family_negative_calibration[k][:, family_index] = (
                        negative.calibration_supported[k] & scaler_supported
                    )
                if negative is None or positive is None:
                    continue
                retrieval_joint = (
                    per_family_positive_retrieval[k][:, family_index]
                    & per_family_negative_retrieval[k][:, family_index]
                )
                retrieval_family_counts[k][retrieval_joint] += 1
                negative_distance_sums[k][retrieval_joint] += np.asarray(
                    negative.raw_distances[k][retrieval_joint], dtype=np.float64
                )
                joint = (
                    retrieval_joint
                    & per_family_positive_calibration[k][:, family_index]
                    & per_family_negative_calibration[k][:, family_index]
                )
                positive_sums[k][joint] += positive.tail_probabilities[k][joint]
                negative_sums[k][joint] += negative.tail_probabilities[k][joint]
                family_counts[k][joint] += 1
            del negative, positive

        combined = {
            k: _finalize_family_sums(
                positive_sums[k],
                negative_sums[k],
                family_counts[k],
                required_family_count=self._required_family_count,
                fit_family_count=len(self._family_order),
            )
            for k in requested
        }
        mean_negative_distances: dict[int, np.ndarray] = {}
        retrieval_supported: dict[int, np.ndarray] = {}
        joint_family_count: dict[int, np.ndarray] = {}
        for k in requested:
            retrieval_supported[k] = np.ascontiguousarray(
                retrieval_family_counts[k] >= self._required_family_count,
                dtype=np.bool_,
            )
            mean = np.full(len(scales), np.nan, dtype=np.float32)
            retrieval = retrieval_supported[k]
            mean[retrieval] = np.asarray(
                negative_distance_sums[k][retrieval]
                / retrieval_family_counts[k][retrieval],
                dtype=np.float32,
            )
            mean_negative_distances[k] = mean
            if np.any(family_counts[k] > np.iinfo(np.int16).max):
                raise RuntimeError("joint family count exceeds int16 range")
            joint_family_count[k] = np.asarray(
                family_counts[k], dtype=np.int16
            )
        return ClassConditionalQueryResult(
            scores={k: combined[k].scores for k in requested},
            mean_negative_distances=mean_negative_distances,
            retrieval_supported=retrieval_supported,
            joint_supported={k: combined[k].supported for k in requested},
            joint_family_count=joint_family_count,
            positive_conformity={
                k: combined[k].positive_conformity for k in requested
            },
            negative_conformity={
                k: combined[k].negative_conformity for k in requested
            },
            per_family_positive_retrieval_supported=per_family_positive_retrieval,
            per_family_positive_calibration_supported=per_family_positive_calibration,
            per_family_negative_retrieval_supported=per_family_negative_retrieval,
            per_family_negative_calibration_supported=per_family_negative_calibration,
            required_family_count=self._required_family_count,
            fit_family_count=len(self._family_order),
        )

    def _export_calibrator_arrays(self) -> dict[str, np.ndarray]:
        """Export the family/class tail bundle without the separate scaler."""

        maximum_name_length = max(len(name) for name in self._family_order)
        family_array = np.asarray(
            self._family_order, dtype=f"<U{maximum_name_length}"
        )
        class_present = self._class_scale_counts.sum(axis=2) > 0
        arrays: dict[str, np.ndarray] = {
            "serialization_version_int16": np.asarray(
                _SERIALIZATION_VERSION, dtype=np.int16
            ),
            "family_order_unicode": family_array,
            "family_order_copy_unicode": family_array.copy(),
            "required_family_count_int64": np.asarray(
                self._required_family_count, dtype=np.int64
            ),
            "ks_int64": np.asarray(self._ks, dtype=np.int64),
            "shrinkage_lambda_float64": np.asarray(
                self._shrinkage_lambda, dtype=np.float64
            ),
            "class_present_bool": np.asarray(class_present, dtype=np.bool_),
            "class_scale_counts_int64": self._class_scale_counts,
        }
        for family_index, pair in enumerate(self._calibrators):
            for class_index, calibrator in enumerate(pair):
                if calibrator is None:
                    continue
                prefix = _calibrator_prefix(family_index, class_index)
                arrays.update(
                    {
                        f"{prefix}{name}": value
                        for name, value in calibrator.export_arrays().items()
                    }
                )
        return {
            name: np.array(value, copy=True, order="C")
            for name, value in arrays.items()
        }

    def export_arrays(self) -> dict[str, np.ndarray]:
        """Export one combined pickle-free mapping for convenience."""

        arrays = self._export_calibrator_arrays()
        arrays.update(
            {
                f"{_SCALER_PREFIX}{name}": value
                for name, value in self._scaler.export_arrays().items()
            }
        )
        return {
            name: np.array(value, copy=True, order="C")
            for name, value in arrays.items()
        }

    @classmethod
    def _from_artifact_mappings(
        cls,
        scaler_arrays: Mapping[str, object],
        calibrator_arrays: Mapping[str, object],
    ) -> "ClassConditionalTemplateScoreModel":
        if not isinstance(calibrator_arrays, Mapping):
            raise ValueError("calibrator artifact must be a mapping of pure arrays")
        if not set(_CALIBRATOR_BASE_ARRAY_NAMES).issubset(calibrator_arrays):
            raise ValueError("serialized class-conditional state is incomplete")
        version = np.asarray(calibrator_arrays["serialization_version_int16"])
        family_values = np.asarray(calibrator_arrays["family_order_unicode"])
        family_copy = np.asarray(calibrator_arrays["family_order_copy_unicode"])
        required = np.asarray(calibrator_arrays["required_family_count_int64"])
        ks_array = np.asarray(calibrator_arrays["ks_int64"])
        lambda_array = np.asarray(calibrator_arrays["shrinkage_lambda_float64"])
        present = np.asarray(calibrator_arrays["class_present_bool"])
        counts = np.asarray(calibrator_arrays["class_scale_counts_int64"])
        if (
            version.shape != ()
            or version.dtype != np.dtype(np.int16)
            or int(version) != _SERIALIZATION_VERSION
            or family_values.ndim != 1
            or family_values.dtype.kind != "U"
            or family_copy.dtype != family_values.dtype
            or family_copy.shape != family_values.shape
            or not np.array_equal(family_values, family_copy)
            or required.shape != ()
            or required.dtype != np.dtype(np.int64)
            or ks_array.ndim != 1
            or ks_array.dtype != np.dtype(np.int64)
            or lambda_array.shape != ()
            or lambda_array.dtype != np.dtype(np.float64)
        ):
            raise ValueError("serialized class-conditional metadata drifted")
        families = _validated_family_order(family_values.tolist())
        fitted_ks = _ks(ks_array.tolist())
        selected_lambda = _validated_lambda(float(lambda_array))
        expected_required = _strict_majority(len(families))
        if int(required) != expected_required:
            raise ValueError("serialized strict-majority requirement drifted")
        if (
            present.shape != (len(families), len(CLASS_NAMES))
            or present.dtype != np.dtype(np.bool_)
            or counts.shape != (len(families), len(CLASS_NAMES), SCALE_COUNT)
            or counts.dtype != np.dtype(np.int64)
            or np.any(counts < 0)
            or not np.array_equal(present, counts.sum(axis=2) > 0)
        ):
            raise ValueError("serialized family/class counts drifted")

        expected_keys = set(_CALIBRATOR_BASE_ARRAY_NAMES)
        tail_names = _tail_array_names(fitted_ks)
        for family_index in range(len(families)):
            for class_index in range(len(CLASS_NAMES)):
                if present[family_index, class_index]:
                    prefix = _calibrator_prefix(family_index, class_index)
                    expected_keys.update(f"{prefix}{name}" for name in tail_names)
        if set(calibrator_arrays) != expected_keys:
            raise ValueError("serialized class-conditional member set drifted")

        scaler = PerScaleNegativeScaler.from_arrays(scaler_arrays)
        calibrators: list[
            tuple[
                ScaleConditionedNegativeTailCalibrator | None,
                ScaleConditionedNegativeTailCalibrator | None,
            ]
        ] = []
        for family_index in range(len(families)):
            current: list[ScaleConditionedNegativeTailCalibrator | None] = []
            for class_index in range(len(CLASS_NAMES)):
                if not present[family_index, class_index]:
                    current.append(None)
                    continue
                prefix = _calibrator_prefix(family_index, class_index)
                current.append(
                    ScaleConditionedNegativeTailCalibrator.from_arrays(
                        {
                            name: calibrator_arrays[f"{prefix}{name}"]
                            for name in tail_names
                        }
                    )
                )
            calibrators.append((current[0], current[1]))
        model = cls.__new__(cls)
        model._install_state(
            family_order=families,
            ks=fitted_ks,
            scaler=scaler,
            calibrators=tuple(calibrators),
            class_scale_counts=counts,
            dimension=scaler.dimension,
            shrinkage_lambda=selected_lambda,
        )
        return model

    @classmethod
    def from_artifacts(
        cls,
        scaler_arrays: Mapping[str, object],
        calibrator_arrays: Mapping[str, object],
    ) -> "ClassConditionalTemplateScoreModel":
        """Reconstruct separate scaler and family/class tail artifacts."""

        return cls._from_artifact_mappings(scaler_arrays, calibrator_arrays)

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, object]
    ) -> "ClassConditionalTemplateScoreModel":
        if not isinstance(arrays, Mapping):
            raise ValueError("serialized model must be a mapping of pure arrays")
        scaler_keys = {f"{_SCALER_PREFIX}{name}" for name in SCALER_ARRAY_NAMES}
        if not scaler_keys.issubset(arrays):
            raise ValueError("serialized combined model is missing scaler arrays")
        scaler_arrays = {
            name: arrays[f"{_SCALER_PREFIX}{name}"] for name in SCALER_ARRAY_NAMES
        }
        calibrator_arrays = {
            name: value
            for name, value in arrays.items()
            if name not in scaler_keys
        }
        model = cls.from_artifacts(scaler_arrays, calibrator_arrays)
        if set(arrays) != scaler_keys | set(calibrator_arrays):
            raise ValueError("serialized combined model member set drifted")
        return model


__all__ = [
    "CLASS_NAMES",
    "NEGATIVE_CLASS_INDEX",
    "POSITIVE_CLASS_INDEX",
    "ClassConditionalQueryResult",
    "ClassConditionalTemplateScoreModel",
    "CombinedClassConditionalScore",
    "FamilyFitBatch",
    "FamilyClassTailCalibratorBundle",
    "combine_joint_family_conformity",
    "strict_threshold_predictions",
]
