from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from pathline_template_matching.development_data import CacheSlice, load_development_config
from pathline_template_matching.development_library import build_balanced_library
from pathline_template_matching.development_experiment import (
    METHOD_FMT,
    METHOD_PCA,
    METHOD_PRIOR,
    METHOD_RAW,
    METHODS,
    _complete_fold,
    _validate_completed_fold,
)
from pathline_template_matching.development_report import (
    FLOAT_METRICS,
    _bootstrap_rows,
    _family_timeslice_macro_rows,
    finalize_development_run,
    _macro_rows,
    _validate_timeslice_coverage,
)
from pathline_template_matching.matcher import ExhaustiveOneNearestNeighbor
from pathline_template_matching.metrics import average_precision, auroc, binary_metrics
from pathline_template_matching.pca import DeterministicPCA


ROOT = Path(__file__).resolve().parents[1]


def test_binary_metrics_are_tie_aware_and_match_hand_computation():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    predictions = np.asarray([0, 1, 0, 1])
    result = binary_metrics(labels, predictions, scores)
    assert np.isclose(result.average_precision, 5.0 / 6.0)
    assert np.isclose(result.auroc, 0.75)
    assert result.true_positive == result.false_positive == 1
    assert result.true_negative == result.false_negative == 1
    assert np.isclose(result.precision, 0.5)
    assert np.isclose(result.recall, 0.5)
    assert np.isclose(result.f1, 0.5)
    assert np.isclose(result.balanced_accuracy, 0.5)
    tied = np.ones(4)
    assert np.isclose(average_precision(labels, tied), 0.5)
    assert np.isclose(auroc(labels, tied), 0.5)
    try:
        binary_metrics(np.zeros(4), np.zeros(4), np.zeros(4))
    except ValueError as error:
        assert "both classes" in str(error)
    else:
        raise AssertionError("single-class metric population was accepted")


def test_exhaustive_matcher_matches_direct_all_pair_distances_and_chunks():
    library = np.asarray(
        [[0.0, 0.0], [1.0, 0.25], [2.0, 2.0], [3.0, 2.5]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=bool)
    queries = np.asarray([[0.25, 0.1], [2.8, 2.4], [1.5, 1.25]], dtype=np.float32)
    matcher = ExhaustiveOneNearestNeighbor(library, labels, device="cpu")
    transformed_query = matcher.transform(queries)
    transformed_library = (library - matcher.feature_mean) / matcher.feature_scale
    squared = np.sum(
        (transformed_query[:, None, :] - transformed_library[None, :, :]) ** 2,
        axis=2,
    )
    expected_class = np.column_stack(
        [
            np.sqrt(np.min(squared[:, labels == bool(class_id)], axis=1))
            for class_id in (0, 1)
        ]
    )
    expected_scores = expected_class[:, 0] - expected_class[:, 1]
    for query_chunk, library_chunk in ((1, 1), (2, 3), (8, 8)):
        result = matcher.query(
            queries,
            query_chunk_size=query_chunk,
            library_chunk_size=library_chunk,
        )
        assert np.allclose(result.nearest_negative_distances, expected_class[:, 0])
        assert np.allclose(result.nearest_positive_distances, expected_class[:, 1])
        assert np.allclose(result.scores, expected_scores)
        assert np.array_equal(result.labels, expected_scores > 0)


def test_high_dimensional_matcher_has_exact_self_distance_and_duplicate_tie():
    rng = np.random.default_rng(25068)
    library = rng.normal(size=(40, 672)).astype(np.float32)
    labels = np.repeat(np.asarray([False, True]), 20)
    library[20] = library[0]
    matcher = ExhaustiveOneNearestNeighbor(library, labels, device="cpu")
    for query_chunk, library_chunk in ((1, 1), (7, 13), (64, 64)):
        result = matcher.query(
            library[[0, 1, 20, 21]],
            query_chunk_size=query_chunk,
            library_chunk_size=library_chunk,
        )
        assert result.nearest_negative_distances[0] == 0.0
        assert result.nearest_positive_distances[0] == 0.0
        assert not bool(result.labels[0])
        assert result.nearest_distances[0] == 0.0
        assert result.nearest_distances[1] == 0.0
        assert result.nearest_distances[2] == 0.0
        assert result.nearest_distances[3] == 0.0


def test_fold_completion_hashes_reject_corrupted_artifact():
    with tempfile.TemporaryDirectory() as directory:
        fold_dir = Path(directory) / "family"
        fold_dir.mkdir()
        artifact = fold_dir / "per_timeslice.csv"
        artifact.write_text("value\n1\n", encoding="utf-8")
        manifest = {
            "held_out_family": "family",
            "input_manifest_sha256": "a" * 64,
            "elapsed_seconds": 1.0,
        }
        _complete_fold(fold_dir, manifest)
        _validate_completed_fold(fold_dir, input_manifest_sha256="a" * 64)
        artifact.write_text("value\n2\n", encoding="utf-8")
        try:
            _validate_completed_fold(fold_dir, input_manifest_sha256="a" * 64)
        except RuntimeError as error:
            assert "digest changed" in str(error)
        else:
            raise AssertionError("corrupted fold artifact was accepted")


def test_completed_report_is_immutable():
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        (run_dir / "result_manifest.json").write_text("{}", encoding="utf-8")
        try:
            finalize_development_run(
                ROOT / "config/mainExp_TemplateMatching_1.1_development.yaml",
                run_dir,
            )
        except FileExistsError as error:
            assert "immutable" in str(error)
        else:
            raise AssertionError("completed report was allowed to overwrite in place")


def test_bootstrap_point_estimate_equals_reported_family_timeslice_macro_delta():
    rows: list[dict[str, object]] = []
    method_values = {
        METHOD_PRIOR: (0.05, 0.05),
        METHOD_RAW: (0.20, 0.10),
        METHOD_PCA: (0.30, 0.25),
        METHOD_FMT: (0.85, 0.45),
    }
    for regime in ("seen_scale", "unseen_scale"):
        for family_index in range(7):
            for method in METHODS:
                for ordinal, value in enumerate(method_values[method]):
                    rows.append(
                        {
                            "held_out_family": f"family_{family_index}",
                            "dataset": f"dataset_{family_index}",
                            "regime": regime,
                            "method": method,
                            "source_ordinal": ordinal,
                            "source_start_index": ordinal * 8,
                            "sample_count": 10 + ordinal,
                            "positive_count": 2 + ordinal,
                            "negative_count": 8,
                            **{metric: value for metric in FLOAT_METRICS},
                        }
                    )
    family_rows = _family_timeslice_macro_rows(rows)
    per_flow = []
    for regime in ("seen_scale", "unseen_scale"):
        for method in METHODS:
            for dataset_index in range(10):
                per_flow.append(
                    {
                        "held_out_family": f"family_{dataset_index % 7}",
                        "dataset": f"flow_{dataset_index}",
                        "regime": regime,
                        "method": method,
                        **{metric: 0.5 for metric in FLOAT_METRICS},
                    }
                )
    main_rows = _macro_rows(per_flow, family_rows)
    bootstrap = _bootstrap_rows(rows, seed=25068, replicates=25)
    for regime in ("seen_scale", "unseen_scale"):
        for metric in ("average_precision", "f1"):
            for comparator in (METHOD_RAW, METHOD_PCA):
                fmt_value = next(
                    row[metric]
                    for row in main_rows
                    if row["regime"] == regime
                    and row["aggregation"] == "physical_family_macro"
                    and row["method"] == METHOD_FMT
                )
                comparator_value = next(
                    row[metric]
                    for row in main_rows
                    if row["regime"] == regime
                    and row["aggregation"] == "physical_family_macro"
                    and row["method"] == comparator
                )
                bootstrap_row = next(
                    row
                    for row in bootstrap
                    if row["regime"] == regime
                    and row["metric"] == metric
                    and row["comparator"] == comparator
                )
                assert np.isclose(
                    bootstrap_row["point_estimate"], fmt_value - comparator_value
                )


def test_timeslice_coverage_rejects_missing_cartesian_key():
    config = {
        "physical_families": {"family": ["dataset"]},
        "split": {
            "library_and_seen_scale_query": {"source_ordinals": [0]},
            "unseen_scale_query": {"source_ordinals": [0]},
        },
    }
    rows = []
    for regime in ("seen_scale", "unseen_scale"):
        for method in METHODS:
            rows.append(
                {
                    "held_out_family": "family",
                    "dataset": "dataset",
                    "regime": regime,
                    "source_ordinal": 0,
                    "method": method,
                    "positive_count": 1,
                    "negative_count": 1,
                    **{metric: 0.5 for metric in FLOAT_METRICS},
                }
            )
    _validate_timeslice_coverage(rows, config)
    try:
        _validate_timeslice_coverage(rows[:-1], config)
    except RuntimeError as error:
        assert "coverage mismatch" in str(error)
    else:
        raise AssertionError("missing timeslice/method key was accepted")


def test_deterministic_full_svd_pca_is_library_only_and_sign_frozen():
    rng = np.random.default_rng(7068)
    library = rng.normal(size=(30, 6)).astype(np.float32)
    query = rng.normal(size=(5, 6)).astype(np.float32)
    first = DeterministicPCA.fit(library, components=3)
    second = DeterministicPCA.fit(library, components=3)
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.components, second.components)
    assert np.allclose(first.components @ first.components.T, np.eye(3), atol=1e-5)
    pivots = np.argmax(np.abs(first.components), axis=1)
    assert np.all(first.components[np.arange(3), pivots] >= 0)
    before = first.transform(query)
    extreme_query = query.copy()
    extreme_query[0] = 1e6
    first.transform(extreme_query)
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(before, second.transform(query))


def _cache_slice(dataset: str, ordinal: int) -> CacheSlice:
    scale_id = np.repeat(np.arange(2), 8).astype(np.int16)
    reference = np.tile(np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool), 2)
    rng = np.random.default_rng(ordinal + 10)
    count = len(reference)
    metadata = {
        "source_start_index": ordinal * 8,
        "source_time": float(ordinal),
        "assigned_count_by_scale": [8, 8],
        "valid_count_by_scale": [8, 8],
    }
    return CacheSlice(
        path=Path(f"{dataset}_{ordinal}.npz"),
        file_sha256="a" * 64,
        dataset=dataset,
        physical_family="library_family",
        legacy_phase="development",
        ordinal=ordinal,
        raw_features=rng.normal(size=(count, 672)).astype(np.float32),
        fmt_features=rng.normal(size=(count, 161)).astype(np.float32),
        reference=reference,
        seeds=rng.normal(size=(count, 3)).astype(np.float32),
        scale_id=scale_id,
        physical_dt=np.ones(count, dtype=np.float32),
        integration_steps=np.full(count, 32, dtype=np.int16),
        metadata=metadata,
        canonical_scale_names=("scale_a", "scale_b"),
    )


def test_balanced_library_is_stratum_balanced_reproducible_and_leak_free():
    records = [_cache_slice("flow_a", 0), _cache_slice("flow_a", 1)]
    first = build_balanced_library(
        records,
        held_out_family="held_family",
        maximum_per_class_per_stratum=3,
        random_seed=15068,
    )
    second = build_balanced_library(
        records,
        held_out_family="held_family",
        maximum_per_class_per_stratum=3,
        random_seed=15068,
    )
    # 2 slices × 2 scales × 2 classes × 3 rows.
    assert len(first.labels) == 24
    assert int(first.labels.sum()) == 12
    assert np.array_equal(first.raw_features, second.raw_features)
    assert first.rows == second.rows
    assert all(row["physical_family"] != "held_family" for row in first.rows)
    assert len(first.audit_rows) == 8


def test_frozen_development_config_rejects_sealed_confirmation_access():
    path = ROOT / "config" / "mainExp_TemplateMatching_1.1_development.yaml"
    config = load_development_config(path)
    assert config["evidence_scope"]["sealed_confirmation_access"] == "forbidden"
    assert config["split"]["descriptor_selection_only"]["main_metric_access"] == "forbidden"
    assert len(config["physical_families"]) == 7
