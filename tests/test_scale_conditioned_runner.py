from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathline_template_matching.portable_flow import (
    canonical_array_sha256,
    sha256_file,
)
from pathline_template_matching.metrics import average_precision, auroc
from pathline_template_matching.nested_scale_validation import (
    CandidateSpec,
    DECISION_RANK_THRESHOLD,
)
from pathline_template_matching.scale_one_class import ScaleConditionedNegativeKNN
from scripts.run_verify_scale_conditioned_retrieval_1_1 import (
    CacheProjection,
    CacheRow,
    PREDICTION_FILE,
    PREDICTION_MANIFEST_FILE,
    _load_outer_reference_after_prediction,
    _partial_supported_query,
    _ranking_metrics_one_sort,
    _threshold_confusion_series,
    _verify_outer_prediction_artifacts,
    _write_outer_prediction,
    load_cache_projection,
    load_plan,
)


def _expect_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, RuntimeError, FileNotFoundError):
        return
    raise AssertionError("expected a fail-closed exception")


def _plan():
    return load_plan(
        Path(__file__).resolve().parents[1]
        / "config"
        / "Verify_ScaleConditionedRetrieval_1.1.yaml"
    )


def _write_valid_outer_artifacts(root, plan, projection):
    candidate = CandidateSpec(
        representation="fmt161",
        k=1,
        sigma=0.0,
        decision_rule=DECISION_RANK_THRESHOLD,
        decision_value=0.5,
    )
    arrays = {
        "dataset_code": np.asarray([0, 0], dtype=np.int8),
        "source_ordinal": np.asarray([0, 0], dtype=np.int8),
        "block_index": np.asarray([0, 0], dtype=np.int8),
        "center_index": np.asarray([0, 1], dtype=np.int64),
        "assigned_row_index": np.asarray([0, 1], dtype=np.int64),
        "scale_id": np.asarray([0, 1], dtype=np.int32),
        "raw_negative_distance": np.asarray([0.1, 0.2], dtype=np.float32),
        "spatial_score": np.asarray([0.25, 0.75], dtype=np.float64),
        "spatial_denominator": np.asarray([1.0, 1.0], dtype=np.float64),
        "retrieval_supported": np.asarray([True, True], dtype=np.bool_),
        "spatial_imputed": np.asarray([False, False], dtype=np.bool_),
        "spatial_unimputable": np.asarray([False, False], dtype=np.bool_),
        "prediction": np.asarray([False, True], dtype=np.bool_),
    }
    groups = [
        {
            "dataset": "f22raptor",
            "source_ordinal": 0,
            "block": "legacy_2_1",
            "row_start": 0,
            "row_stop_exclusive": 2,
            "row_count": 2,
            "supported_count": 2,
            "imputed_count": 0,
            "unimputable_count": 0,
        }
    ]
    selected_sha = "a" * 64
    _, manifest_sha = _write_outer_prediction(
        plan,
        root,
        "f22_raptor",
        [projection],
        candidate,
        selected_sha,
        arrays,
        groups,
        git_commit="b" * 40,
        device="cpu",
    )
    return candidate, selected_sha, manifest_sha, arrays


def test_partial_query_support_is_fit_only_and_independent_for_each_k():
    model = ScaleConditionedNegativeKNN(
        np.asarray([[0.0], [10.0], [11.0], [12.0], [13.0], [14.0]], dtype=np.float32),
        np.asarray([0, 1, 1, 1, 1, 1], dtype=np.int64),
    )
    scores, support = _partial_supported_query(
        model,
        np.asarray([[0.5], [12.5], [100.0]], dtype=np.float32),
        np.asarray([0, 1, 2], dtype=np.int64),
        (1, 5),
        device="cpu",
        query_chunk_size=2,
        library_chunk_size=2,
    )
    np.testing.assert_array_equal(support[1], [True, True, False])
    np.testing.assert_array_equal(support[5], [False, True, False])
    assert np.isfinite(scores[1][:2]).all() and np.isnan(scores[1][2])
    assert np.isnan(scores[5][0]) and np.isfinite(scores[5][1])
    assert np.isnan(scores[5][2])


def test_outer_prediction_projection_never_opens_label_or_metadata_members():
    plan = _plan()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "outer.npz"
        with path.open("wb") as destination:
            np.savez_compressed(
                destination,
                fmt_features=np.zeros((2, 161), dtype=np.float32),
                valid_scale_id=np.asarray([0, 1000], dtype=np.int32),
                valid_center_seed_index=np.asarray([0, 1], dtype=np.int64),
                valid_scale_block_index=np.asarray([0, 1], dtype=np.int8),
                valid_assigned_row_index=np.asarray([0, 64001], dtype=np.int64),
                # Accessing either object member with allow_pickle=False would
                # fail.  The prediction projection must leave both untouched.
                valid_labels=np.asarray([object(), object()], dtype=object),
                metadata_json=np.asarray({"reference_leak": True}, dtype=object),
            )
        row = CacheRow(
            dataset="f22raptor",
            family="f22_raptor",
            source_ordinal=0,
            source_index=0,
            path=path,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        projection = load_cache_projection(plan, row, include_labels=False)
        assert projection.labels is None
        assert projection.metadata == {}
        assert projection.fmt_features.shape == (2, 161)


def test_outer_reference_member_is_gated_by_closed_prediction_artifacts():
    plan = _plan()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        labels = np.asarray([False, True], dtype=np.bool_)
        metadata = {
            "schema": plan.cache_schema,
            "dataset": "f22raptor",
            "physical_family": "f22_raptor",
            "split": "train",
            "config_sha256": plan.parent_config_sha256,
            "descriptor_id": plan.descriptor_id,
            "array_sha256": {"valid_labels": canonical_array_sha256(labels)},
        }
        cache_path = root / "cache.npz"
        with cache_path.open("wb") as destination:
            np.savez_compressed(
                destination,
                valid_labels=labels,
                metadata_json=np.asarray(__import__("json").dumps(metadata)),
            )
        row = CacheRow(
            dataset="f22raptor",
            family="f22_raptor",
            source_ordinal=0,
            source_index=0,
            path=cache_path,
            size_bytes=cache_path.stat().st_size,
            sha256=sha256_file(cache_path),
        )
        projection = CacheProjection(
            row=row,
            fmt_features=np.zeros((2, 161), dtype=np.float32),
            scale_ids=np.asarray([0, 1], dtype=np.int32),
            center_indices=np.asarray([0, 1], dtype=np.int64),
            block_indices=np.asarray([0, 0], dtype=np.int8),
            assigned_row_indices=np.asarray([0, 1], dtype=np.int64),
            labels=None,
            metadata={},
        )
        manifest_path = root / PREDICTION_MANIFEST_FILE
        _expect_error(
            _verify_outer_prediction_artifacts,
            plan,
            root,
            "f22_raptor",
            CandidateSpec(
                representation="fmt161",
                k=1,
                sigma=0.0,
                decision_rule=DECISION_RANK_THRESHOLD,
                decision_value=0.5,
            ),
            "a" * 64,
            "c" * 64,
        )
        candidate, selected_sha, manifest_sha, _ = _write_valid_outer_artifacts(
            root, plan, projection
        )
        verified = _verify_outer_prediction_artifacts(
            plan,
            root,
            "f22_raptor",
            candidate,
            selected_sha,
            manifest_sha,
        )
        observed = _load_outer_reference_after_prediction(
            plan, projection, verified
        )
        np.testing.assert_array_equal(observed, labels)


def test_outer_prediction_authentication_rejects_rewritten_npz():
    plan = _plan()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cache_path = root / "unused-cache.npz"
        cache_path.write_bytes(b"reference-must-not-open")
        row = CacheRow(
            dataset="f22raptor",
            family="f22_raptor",
            source_ordinal=0,
            source_index=0,
            path=cache_path,
            size_bytes=cache_path.stat().st_size,
            sha256=sha256_file(cache_path),
        )
        projection = CacheProjection(
            row=row,
            fmt_features=np.zeros((2, 161), dtype=np.float32),
            scale_ids=np.asarray([0, 1], dtype=np.int32),
            center_indices=np.asarray([0, 1], dtype=np.int64),
            block_indices=np.asarray([0, 0], dtype=np.int8),
            assigned_row_indices=np.asarray([0, 1], dtype=np.int64),
            labels=None,
            metadata={},
        )
        candidate, selected_sha, manifest_sha, arrays = _write_valid_outer_artifacts(
            root, plan, projection
        )
        rewritten = dict(arrays)
        rewritten["spatial_score"] = np.asarray([0.9, 0.1], dtype=np.float64)
        with (root / PREDICTION_FILE).open("wb") as destination:
            np.savez_compressed(destination, **rewritten)
        _expect_error(
            _verify_outer_prediction_artifacts,
            plan,
            root,
            "f22_raptor",
            candidate,
            selected_sha,
            manifest_sha,
        )


def test_sorted_threshold_confusions_match_direct_predictions_with_ties():
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=bool)
    scores = np.asarray([0.5, 0.5, 0.7, 0.9, 0.9, 0.2])
    eligible = np.asarray([True, False, True, True, True, True])
    thresholds = (0.5, 0.7, 0.9, 0.99)
    observed = _threshold_confusion_series(labels, scores, eligible, thresholds)
    for threshold, row in zip(thresholds, observed):
        prediction = eligible & (scores >= threshold)
        assert row["true_positive"] == int(np.sum(labels & prediction))
        assert row["false_positive"] == int(np.sum(~labels & prediction))
        assert row["true_negative"] == int(np.sum(~labels & ~prediction))
        assert row["false_negative"] == int(np.sum(labels & ~prediction))


def test_one_sort_ranking_metrics_match_frozen_metric_definitions():
    generator = np.random.default_rng(718)
    labels = np.resize(np.asarray([0, 1], dtype=bool), 101)
    scores = np.round(generator.normal(size=101), decimals=1)
    observed_ap, observed_auroc = _ranking_metrics_one_sort(labels, scores)
    assert observed_ap == average_precision(labels, scores)
    assert observed_auroc == auroc(labels, scores)
