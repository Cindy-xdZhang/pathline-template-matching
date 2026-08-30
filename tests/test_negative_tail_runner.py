from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
import operator
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.negative_tail_calibration import (
    CALIBRATION_BLOCK_FALLBACK,
    ScaleConditionedNegativeTailCalibrator,
)
from pathline_template_matching.portable_flow import canonical_array_sha256, sha256_file
from scripts import run_verify_negative_tail_calibration_1_1 as runner


CONFIG = ROOT / "config" / "Verify_NegativeTailCalibration_1.1.yaml"


def _assert_raises(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    expected = (
        ", ".join(value.__name__ for value in error_type)
        if isinstance(error_type, tuple)
        else error_type.__name__
    )
    raise AssertionError(f"expected {expected}")


def _fmt(values: list[float] | np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float32)
    output = np.zeros((len(numeric), 161), dtype=np.float32)
    output[:, 0] = numeric
    output[:, 1] = np.square(numeric)
    return output


def _projection(
    *,
    dataset: str,
    family: str,
    values: list[float],
    scales: list[int],
    centers: list[int],
    labels: list[bool] | None,
    path: Path | None = None,
    source_ordinal: int = 0,
    source_index: int = 0,
) -> runner.CacheProjection:
    scale_values = np.asarray(scales, dtype=np.int32)
    center_values = np.asarray(centers, dtype=np.int64)
    blocks = (scale_values >= 1000).astype(np.int8)
    assigned = blocks.astype(np.int64) * 64000 + center_values
    row = runner.CacheRow(
        dataset=dataset,
        family=family,
        source_ordinal=source_ordinal,
        source_index=source_index,
        path=Path("unused.npz") if path is None else path,
        size_bytes=0,
        sha256="0" * 64,
    )
    return runner.CacheProjection(
        row=row,
        fmt_features=_fmt(values),
        scale_ids=scale_values,
        center_indices=center_values,
        block_indices=blocks,
        assigned_row_indices=assigned,
        labels=None if labels is None else np.asarray(labels, dtype=bool),
        metadata={},
    )


def _write_parent_cache(
    path: Path,
    plan: runner.Plan,
    *,
    dataset: str = "cylinder3d",
    family: str = "half_cylinder",
    source_ordinal: int = 0,
    source_index: int = 0,
) -> runner.CacheRow:
    features = _fmt([0.25, 4.0, 2.25, 5.0])
    scales = np.asarray([0, 0, 1000, 1000], dtype=np.int32)
    centers = np.asarray([0, 1, 2, 3], dtype=np.int64)
    blocks = np.asarray([0, 0, 1, 1], dtype=np.int8)
    assigned = blocks.astype(np.int64) * 64000 + centers
    labels = np.asarray([False, True, False, True], dtype=bool)
    arrays = {
        "fmt_features": features,
        "valid_scale_id": scales,
        "valid_center_seed_index": centers,
        "valid_scale_block_index": blocks,
        "valid_assigned_row_index": assigned,
        "valid_labels": labels,
    }
    metadata = {
        "schema": plan.cache_schema,
        "experiment": "mainExp_TemplateMatching_3.1",
        "split": "train",
        "dataset": dataset,
        "physical_family": family,
        "source_ordinal": source_ordinal,
        "source_index": source_index,
        "config_sha256": plan.parent_config_sha256,
        "cache_builder_git_commit": plan.cache_commit,
        "descriptor_id": plan.descriptor_id,
        "valid_count": len(labels),
        "array_sha256": {
            name: canonical_array_sha256(values) for name, values in arrays.items()
        },
    }
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return runner.CacheRow(
        dataset=dataset,
        family=family,
        source_ordinal=source_ordinal,
        source_index=source_index,
        path=path,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def _small_model() -> ScaleConditionedNegativeTailCalibrator:
    return ScaleConditionedNegativeTailCalibrator(
        _fmt([0.0, 1.0, 2.0, 3.0]),
        np.asarray([0, 0, 1000, 1000], dtype=np.int64),
        ks=(1,),
        device="cpu",
        query_chunk_size=1,
        library_chunk_size=1,
    )


def _write_synthetic_inner_evidence(
    output: Path,
    plan: runner.Plan,
    selected: runner.TailCandidateSpec,
    *,
    outer_family: str,
):
    candidates = runner.candidate_specs(plan)
    inner_families = [family for family in plan.family_order if family != outer_family]
    groups = [
        (family, dataset, source_ordinal, block)
        for family in inner_families
        for dataset in plan.families[family]
        for source_ordinal in range(4)
        for block in runner.BLOCK_NAMES
    ]

    def metrics(candidate):
        winner = candidate.candidate_id == selected.candidate_id
        high = 1.0 if winner else 0.1
        return {
            "accuracy": high,
            "average_precision": high,
            "f1": high,
            "balanced_accuracy": high,
            "auroc": high,
            "precision": high,
            "recall": high,
            "retrieval_support_fraction": high,
            "calibration_support_fraction": high,
            "spatial_imputed_fraction": high,
            "spatial_unimputable_fraction": high,
        }

    summaries = []
    for candidate in candidates:
        summaries.append(
            {
                **runner._candidate_payload(candidate),
                **metrics(candidate),
                "inner_family_count": 4,
                "group_count": len(groups),
            }
        )

    def metric_rows():
        for candidate in candidates:
            values = metrics(candidate)
            for family, dataset, source_ordinal, block in groups:
                yield {
                    "outer_family": outer_family,
                    "inner_family": family,
                    "dataset": dataset,
                    "source_ordinal": source_ordinal,
                    "block": block,
                    **runner._candidate_payload(candidate),
                    **values,
                }

    metrics_path = output / "inner_group_metrics.csv"
    summary_path = output / "inner_candidate_summary.csv"
    audits_path = output / "inner_fit_audits.json"
    metrics_sha = runner._atomic_csv(metrics_path, runner.METRIC_FIELDS, metric_rows())
    summary_sha = runner._atomic_csv(summary_path, runner.SUMMARY_FIELDS, summaries)
    fits = [
        {"inner_family": family, "representation": representation}
        for family in inner_families
        for representation in plan.representations
    ]
    audits_sha = runner._atomic_json(
        audits_path,
        runner._manifest_with_self_hash(
            {
                "schema": "pathline_template_matching.negative_tail_inner_fit_audits.v1",
                "experiment": runner.EXPERIMENT,
                "outer_family": outer_family,
                "fit_count": len(fits),
                "fits": fits,
            }
        ),
    )
    selected_summary = next(
        row for row in summaries if row["candidate_id"] == selected.candidate_id
    )
    return (
        metrics_path,
        metrics_sha,
        summary_path,
        summary_sha,
        audits_path,
        audits_sha,
        selected_summary,
    )


def _write_synthetic_input_manifest(
    temporary_path: Path,
    plan: runner.Plan,
    *,
    outer_family: str,
) -> runner.Plan:
    cache_root = temporary_path / "primitive_cache" / "train"
    cache_root.mkdir(parents=True)
    rows = []
    source_index = 0
    for family in plan.family_order:
        for dataset in plan.families[family]:
            for source_ordinal in range(4):
                cache_path = cache_root / f"{dataset}_{source_ordinal}.npz"
                if family == outer_family:
                    row = _write_parent_cache(
                        cache_path,
                        plan,
                        dataset=dataset,
                        family=family,
                        source_ordinal=source_ordinal,
                        source_index=source_index,
                    )
                    size_bytes = row.size_bytes
                    file_sha = row.sha256
                else:
                    size_bytes = 0
                    file_sha = "0" * 64
                rows.append(
                    {
                        "dataset": dataset,
                        "source_ordinal": source_ordinal,
                        "source_index": source_index,
                        "cache_path": str(cache_path),
                        "cache_size_bytes": size_bytes,
                        "cache_file_sha256": file_sha,
                    }
                )
                source_index += 1
    assert len(rows) == 32
    rows_sha = "a" * 64
    manifest = {
        "schema": plan.manifest_schema,
        "row_count": 32,
        "rows_content_sha256": rows_sha,
        "git_commit": plan.cache_commit,
        "main_config_sha256": plan.parent_config_sha256,
        "test_dataset_access": False,
        "rows": rows,
    }
    manifest_path = temporary_path / "train_cache_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return replace(
        plan,
        manifest_path=manifest_path,
        manifest_size=manifest_path.stat().st_size,
        manifest_sha256=sha256_file(manifest_path),
        manifest_rows_sha256=rows_sha,
    )


def test_negative_tail_plan_and_candidate_grid_are_exact_and_rank_free():
    plan = runner.load_plan(CONFIG)
    candidates = runner.candidate_specs(plan)
    assert plan.sha256 == runner.EXPECTED_CONFIG_SHA256
    assert len(candidates) == 3060
    assert len({candidate.candidate_id for candidate in candidates}) == 3060
    assert {candidate.decision_rule for candidate in candidates} == {
        "fixed_top_fraction",
        "calibrated_tail_anomaly_threshold",
    }
    assert all("rank_threshold" not in candidate.candidate_id for candidate in candidates)
    thresholds = [
        candidate.decision_value
        for candidate in candidates
        if candidate.representation == "fmt161"
        and candidate.k == 1
        and candidate.sigma == 0.0
        and candidate.decision_rule == "calibrated_tail_anomaly_threshold"
    ]
    assert thresholds == list(runner.TAIL_THRESHOLDS)
    fixed = candidates[0]
    prediction = runner.candidate_predictions(
        fixed,
        np.asarray([0.9, 0.8, 0.0], dtype=np.float64),
        np.asarray([2, 1, 0], dtype=np.int64),
        np.asarray([True, True, False]),
    )
    assert prediction.tolist() == [True, False, False]
    assert runner.BLOCK_NAMES == ("legacy_2_1", "expanded_3_1")


def test_cpu_environment_audit_does_not_query_cpu_as_cuda_device():
    original_is_available = runner.torch.cuda.is_available
    original_device_count = runner.torch.cuda.device_count
    original_get_device_name = runner.torch.cuda.get_device_name
    original_get_device_capability = runner.torch.cuda.get_device_capability
    try:
        runner.torch.cuda.is_available = lambda: True
        runner.torch.cuda.device_count = lambda: 2

        def reject_cpu_device(device):
            raise AssertionError(f"unexpected CUDA device query: {device}")

        runner.torch.cuda.get_device_name = reject_cpu_device
        runner.torch.cuda.get_device_capability = reject_cpu_device
        audit = runner._environment_audit("cpu")
    finally:
        runner.torch.cuda.is_available = original_is_available
        runner.torch.cuda.device_count = original_device_count
        runner.torch.cuda.get_device_name = original_get_device_name
        runner.torch.cuda.get_device_capability = original_get_device_capability

    assert audit["requested_device"] == "cpu"
    assert audit["cuda_available"] is True
    assert audit["cuda_device_count"] == 2
    assert audit["cuda_device_query_skipped"] == "requested_device_is_not_cuda"
    assert "cuda_device_name" not in audit
    assert "cuda_device_capability" not in audit


def test_negative_tail_query_path_never_calls_query_rank_and_is_membership_invariant():
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "spatial_support_scores" not in called_names
    assert "supported_rank_scores" not in called_names
    source = inspect.getsource(runner._query_cache_batch)
    assert "anomaly_scores" in source
    assert "calibration_supported" in source

    plan = runner.load_plan(CONFIG)
    model = _small_model()
    one = _projection(
        dataset="channel",
        family="channel",
        values=[0.5],
        scales=[0],
        centers=[10],
        labels=None,
    )
    companion = _projection(
        dataset="channel",
        family="channel",
        values=[50.0],
        scales=[0],
        centers=[11],
        labels=None,
    )
    alone = runner._query_cache_batch(model, [one], "fmt161", plan, device="cpu", ks=(1,))[1][0]
    together = runner._query_cache_batch(model, [one, companion], "fmt161", plan, device="cpu", ks=(1,))[1][0]
    assert np.array_equal(alone["tail_anomaly"], together["tail_anomaly"])
    assert np.array_equal(alone["calibration_supported"], together["calibration_supported"])


def test_spatial_calibrated_tail_scores_match_independent_mask_normalized_gaussian():
    anomaly = np.asarray([0.2, 0.0, 0.8], dtype=np.float64)
    support = np.asarray([True, False, True])
    centers = np.asarray([0, 1, 2], dtype=np.int64)
    result = runner.spatial_calibrated_tail_scores(
        anomaly,
        support,
        centers,
        sigma=1.0,
        grid_shape=(1, 1, 3),
        truncate=3.0,
    )
    axis = np.arange(-3, 4, dtype=np.float64)
    kernel = np.exp(-0.5 * axis**2)
    kernel /= kernel.sum()
    expected_denominator = []
    expected_score = []
    for target in range(3):
        weights = np.asarray(
            [kernel[3] * kernel[3] * kernel[3 + target - source] for source in (0, 2)],
            dtype=np.float64,
        )
        expected_denominator.append(weights.sum())
        expected_score.append(float(np.dot(weights, [0.2, 0.8]) / weights.sum()))
    assert np.allclose(result.denominator, expected_denominator, rtol=0.0, atol=1e-15)
    assert np.allclose(result.scores, expected_score, rtol=0.0, atol=1e-15)
    assert np.array_equal(result.calibration_supported, support)
    assert np.array_equal(result.imputed, [False, True, False])
    assert not result.unimputable.any()

    zero = runner.spatial_calibrated_tail_scores(anomaly, support, centers, sigma=0.0, grid_shape=(1, 1, 3))
    assert np.array_equal(zero.scores, [0.2, 0.0, 0.8])
    assert np.array_equal(zero.denominator, support.astype(np.float64))
    assert np.array_equal(zero.unimputable, [False, True, False])


def test_runner_final_fit_accepts_n_equals_k_and_uses_calibration_fallback():
    plan = runner.load_plan(CONFIG)
    fit = _projection(
        dataset="channel",
        family="channel",
        values=[0.0, 1.0, 2.0],
        scales=[0, 1, 1],
        centers=[0, 1, 2],
        labels=[False, False, False],
    )
    model = runner._fit_tail_model([fit], "fmt161", plan, device="cpu", ks=(1,))
    query = _projection(
        dataset="channel",
        family="channel",
        values=[0.25],
        scales=[0],
        centers=[3],
        labels=None,
    )
    values = runner._query_cache_batch(model, [query], "fmt161", plan, device="cpu", ks=(1,))[1][0]
    assert values["retrieval_supported"].tolist() == [True]
    assert values["calibration_supported"].tolist() == [True]
    assert values["calibration_mode"].tolist() == [CALIBRATION_BLOCK_FALLBACK]


def test_batched_tail_threshold_metrics_match_direct_predictions_with_ties():
    plan = runner.load_plan(CONFIG)
    cache = _projection(
        dataset="channel",
        family="channel",
        values=[0.0, 1.0, 2.0, 3.0],
        scales=[0, 0, 0, 0],
        centers=[0, 1, 2, 3],
        labels=[False, True, False, True],
    )
    labels = np.asarray(cache.labels, dtype=bool)
    scores = np.asarray([0.50, 0.50, 0.75, 1.00], dtype=np.float64)
    eligible = np.ones(4, dtype=bool)
    retrieval = np.ones(4, dtype=bool)
    calibration = np.asarray([True, True, False, True])
    imputed = np.asarray([False, False, True, False])
    unimputable = np.zeros(4, dtype=bool)
    modes = np.asarray([3, 3, 0, 3], dtype=np.int8)
    candidates = tuple(
        runner.TailCandidateSpec(
            "fmt161", 1, 0.0, "calibrated_tail_anomaly_threshold", threshold
        )
        for threshold in plan.thresholds
    )
    ranking = runner._ranking_metrics_one_sort(labels, scores)
    rows = runner._threshold_metric_rows(
        outer_family="half_cylinder",
        inner_family="channel",
        cache=cache,
        block_name="legacy_2_1",
        candidates=candidates,
        labels=labels,
        scores=scores,
        eligible=eligible,
        retrieval_supported=retrieval,
        calibration_supported=calibration,
        imputed=imputed,
        unimputable=unimputable,
        calibration_modes=modes,
        ranking_metrics=ranking,
    )
    assert len(rows) == 50
    for row, candidate in zip(rows, candidates):
        direct = runner.candidate_predictions(
            candidate, scores, cache.center_indices, eligible
        )
        expected = runner._classification_counts(labels, direct)
        for field in (
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
            "f1",
        ):
            assert row[field] == expected[field]
        assert row["decision_rule"] == "calibrated_tail_anomaly_threshold"
        assert row["block"] == "legacy_2_1"


def test_final_calibration_authentication_rejects_structure_valid_content_tamper():
    plan = runner.load_plan(CONFIG)
    selected = runner.TailCandidateSpec("fmt161", 1, 0.0, "calibrated_tail_anomaly_threshold", 0.5)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        calibration_path, manifest_path, _, manifest_sha = runner.write_final_calibration_artifact(
            output,
            _small_model(),
            plan=plan,
            selected=selected,
            outer_family="half_cylinder",
            fit_families=("delta_wing", "f22_raptor", "channel", "boeing_747"),
            git_commit="0" * 40,
        )
        verified = runner.authenticate_and_rebuild_final_calibration(
            calibration_path,
            manifest_path,
            plan=plan,
            selected=selected,
            outer_family="half_cylinder",
            fit_families=("delta_wing", "f22_raptor", "channel", "boeing_747"),
            git_commit="0" * 40,
            expected_manifest_file_sha256=manifest_sha,
        )
        assert verified.model.ks == (1,)
        _assert_raises(
            FileExistsError,
            runner.write_final_calibration_artifact,
            output,
            _small_model(),
            plan=plan,
            selected=selected,
            outer_family="half_cylinder",
            fit_families=("delta_wing", "f22_raptor", "channel", "boeing_747"),
            git_commit="0" * 40,
        )
        _assert_raises(
            ValueError,
            runner.authenticate_and_rebuild_final_calibration,
            calibration_path,
            manifest_path,
            plan=plan,
            selected=selected,
            outer_family="delta_wing",
            fit_families=("half_cylinder", "f22_raptor", "channel", "boeing_747"),
            git_commit="0" * 40,
            expected_manifest_file_sha256=manifest_sha,
        )
        with np.load(calibration_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        arrays["negative_features"][0, 0] += np.float32(0.125)
        np.savez_compressed(calibration_path, **arrays)
        _assert_raises(
            (RuntimeError, ValueError),
            runner.authenticate_and_rebuild_final_calibration,
            calibration_path,
            manifest_path,
            plan=plan,
            selected=selected,
            outer_family="half_cylinder",
            fit_families=("delta_wing", "f22_raptor", "channel", "boeing_747"),
            git_commit="0" * 40,
            expected_manifest_file_sha256=manifest_sha,
        )


def test_outer_projection_and_reference_gate_do_not_open_labels_early():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "poisoned_outer.npz"
        features = _fmt([0.0])
        scales = np.asarray([0], dtype=np.int32)
        centers = np.asarray([0], dtype=np.int64)
        blocks = np.asarray([0], dtype=np.int8)
        assigned = np.asarray([0], dtype=np.int64)
        np.savez_compressed(
            path,
            fmt_features=features,
            valid_scale_id=scales,
            valid_center_seed_index=centers,
            valid_scale_block_index=blocks,
            valid_assigned_row_index=assigned,
            valid_labels=np.asarray([object()], dtype=object),
            metadata_json=np.asarray(object(), dtype=object),
        )
        row = runner.CacheRow(
            dataset="cylinder3d",
            family="half_cylinder",
            source_ordinal=0,
            source_index=0,
            path=path,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        projection = runner.load_cache_projection(plan, row, include_labels=False)
        assert projection.labels is None and projection.metadata == {}
        _assert_raises(
            ValueError,
            runner._validate_label_free_outer_scope,
            plan,
            "half_cylinder",
            [projection],
            [row],
        )


def test_synthetic_negative_tail_artifact_prediction_and_label_pipeline_end_to_end():
    base_plan = runner.load_plan(CONFIG)
    selected = runner.TailCandidateSpec("fmt161", 1, 0.0, "calibrated_tail_anomaly_threshold", 0.5)
    fit = _projection(
        dataset="channel",
        family="channel",
        values=[0.0, 1.0, 2.0, 3.0],
        scales=[0, 0, 1000, 1000],
        centers=[10, 11, 12, 13],
        labels=[False, False, False, False],
    )
    model = runner._fit_tail_model([fit], "fmt161", base_plan, device="cpu", ks=(1,))
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        output = temporary_path / "output"
        output.mkdir()
        plan = _write_synthetic_input_manifest(
            temporary_path, base_plan, outer_family="half_cylinder"
        )
        (
            inner_metrics_path,
            inner_metrics_sha,
            inner_summary_path,
            inner_summary_sha,
            inner_audits_path,
            inner_audits_sha,
            selected_summary,
        ) = _write_synthetic_inner_evidence(
            output, plan, selected, outer_family="half_cylinder"
        )
        all_rows, _ = runner.load_cache_rows(plan)
        outer_rows = [row for row in all_rows if row.family == "half_cylinder"]
        outer_projections = [
            runner.load_cache_projection(plan, row, include_labels=False)
            for row in outer_rows
        ]
        assert len(outer_rows) == len(outer_projections) == 12
        calibration_path, calibration_manifest_path, _, calibration_manifest_sha = runner.write_final_calibration_artifact(
            output,
            model,
            plan=plan,
            selected=selected,
            outer_family="half_cylinder",
            fit_families=("delta_wing", "f22_raptor", "channel", "boeing_747"),
            git_commit="0" * 40,
        )
        calibration = runner.authenticate_and_rebuild_final_calibration(
            calibration_path,
            calibration_manifest_path,
            plan=plan,
            selected=selected,
            outer_family="half_cylinder",
            fit_families=("delta_wing", "f22_raptor", "channel", "boeing_747"),
            git_commit="0" * 40,
            expected_manifest_file_sha256=calibration_manifest_sha,
        )
        selected_path, selected_sha, selected_payload = runner.write_selected_candidate(
            output,
            plan=plan,
            selected=selected,
            selected_summary=selected_summary,
            calibration=calibration,
            inner_group_metrics_path=inner_metrics_path,
            inner_group_metrics_sha256=inner_metrics_sha,
            inner_candidate_summary_path=inner_summary_path,
            inner_candidate_summary_sha256=inner_summary_sha,
            inner_fit_audits_path=inner_audits_path,
            inner_fit_audits_sha256=inner_audits_sha,
            outer_family="half_cylinder",
            git_commit="0" * 40,
        )
        assert selected_path.exists()
        assert selected_payload["final_calibration_manifest"]["file_sha256"] == calibration.manifest_file_sha256
        selected_artifact = runner.authenticate_selected_candidate(
            selected_path,
            plan=plan,
            selected=selected,
            calibration=calibration,
            inner_group_metrics_path=inner_metrics_path,
            inner_group_metrics_sha256=inner_metrics_sha,
            inner_candidate_summary_path=inner_summary_path,
            inner_candidate_summary_sha256=inner_summary_sha,
            inner_fit_audits_path=inner_audits_path,
            inner_fit_audits_sha256=inner_audits_sha,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            expected_file_sha256=selected_sha,
        )
        _assert_raises(
            TypeError,
            operator.setitem,
            calibration.manifest,
            "outer_family",
            "delta_wing",
        )
        _assert_raises(
            TypeError,
            operator.setitem,
            selected_artifact.manifest,
            "outer_family",
            "delta_wing",
        )
        _assert_raises(
            ValueError,
            runner.authenticate_selected_candidate,
            selected_path,
            plan=plan,
            selected=selected,
            calibration=calibration,
            inner_group_metrics_path=inner_metrics_path,
            inner_group_metrics_sha256=inner_metrics_sha,
            inner_candidate_summary_path=inner_summary_path,
            inner_candidate_summary_sha256=inner_summary_sha,
            inner_fit_audits_path=inner_audits_path,
            inner_fit_audits_sha256=inner_audits_sha,
            outer_family="delta_wing",
            git_commit="0" * 40,
            expected_file_sha256=selected_sha,
        )

        bad_selected_output = temporary_path / "bad_selected"
        bad_selected_output.mkdir()
        bad_selected_summary = dict(selected_summary)
        bad_selected_summary["f1"] = 0.99
        bad_selected_path, bad_selected_sha, _ = runner.write_selected_candidate(
            bad_selected_output,
            plan=plan,
            selected=selected,
            selected_summary=bad_selected_summary,
            calibration=calibration,
            inner_group_metrics_path=inner_metrics_path,
            inner_group_metrics_sha256=inner_metrics_sha,
            inner_candidate_summary_path=inner_summary_path,
            inner_candidate_summary_sha256=inner_summary_sha,
            inner_fit_audits_path=inner_audits_path,
            inner_fit_audits_sha256=inner_audits_sha,
            outer_family="half_cylinder",
            git_commit="0" * 40,
        )
        _assert_raises(
            ValueError,
            runner.authenticate_selected_candidate,
            bad_selected_path,
            plan=plan,
            selected=selected,
            calibration=calibration,
            inner_group_metrics_path=inner_metrics_path,
            inner_group_metrics_sha256=inner_metrics_sha,
            inner_candidate_summary_path=inner_summary_path,
            inner_candidate_summary_sha256=inner_summary_sha,
            inner_fit_audits_path=inner_audits_path,
            inner_fit_audits_sha256=inner_audits_sha,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            expected_file_sha256=bad_selected_sha,
        )

        arrays, group_audits = runner.build_outer_prediction_arrays(
            outer_projections, calibration.model, selected, plan, device="cpu"
        )
        assert set(arrays) == set(runner.PREDICTION_ARRAY_DTYPES)
        prediction_path, prediction_manifest_path, _, prediction_manifest_sha = runner.write_outer_prediction(
            output,
            arrays,
            group_audits,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            calibration=calibration,
            outer_family="half_cylinder",
            git_commit="0" * 40,
        )
        _assert_raises(
            ValueError,
            runner.authenticate_outer_prediction,
            prediction_path,
            prediction_manifest_path,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            calibration=calibration,
            outer_projections=outer_projections,
            expected_outer_rows=outer_rows,
            outer_family="delta_wing",
            git_commit="0" * 40,
            device="cpu",
            expected_manifest_file_sha256=prediction_manifest_sha,
        )
        _assert_raises(
            ValueError,
            runner.authenticate_outer_prediction,
            prediction_path,
            prediction_manifest_path,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            calibration=calibration,
            outer_projections=outer_projections,
            expected_outer_rows=outer_rows,
            outer_family="half_cylinder",
            git_commit="1" * 40,
            device="cpu",
            expected_manifest_file_sha256=prediction_manifest_sha,
        )
        _assert_raises(
            ValueError,
            runner.authenticate_outer_prediction,
            prediction_path,
            prediction_manifest_path,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            calibration=calibration,
            outer_projections=outer_projections[:-1],
            expected_outer_rows=outer_rows[:-1],
            outer_family="half_cylinder",
            git_commit="0" * 40,
            device="cpu",
            expected_manifest_file_sha256=prediction_manifest_sha,
        )
        verified_prediction = runner.authenticate_outer_prediction(
            prediction_path,
            prediction_manifest_path,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            calibration=calibration,
            outer_projections=outer_projections,
            expected_outer_rows=outer_rows,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            device="cpu",
            expected_manifest_file_sha256=prediction_manifest_sha,
        )
        _assert_raises(
            ValueError,
            verified_prediction.arrays["tail_anomaly"].__setitem__,
            0,
            0.0,
        )
        _assert_raises(
            TypeError,
            operator.setitem,
            verified_prediction.manifest,
            "outer_family",
            "delta_wing",
        )
        metrics, reference_audit = runner.evaluate_outer_prediction(
            plan,
            selected,
            output,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            device="cpu",
            expected_calibration_manifest_sha256=calibration_manifest_sha,
            expected_selected_candidate_sha256=selected_sha,
            expected_prediction_manifest_sha256=prediction_manifest_sha,
            inner_group_metrics_path=inner_metrics_path,
            inner_group_metrics_sha256=inner_metrics_sha,
            inner_candidate_summary_path=inner_summary_path,
            inner_candidate_summary_sha256=inner_summary_sha,
            inner_fit_audits_path=inner_audits_path,
            inner_fit_audits_sha256=inner_audits_sha,
        )
        assert len(metrics) == 24
        assert len(reference_audit) == 12
        assert all(row["positive_count"] == row["negative_count"] == 1 for row in metrics)
        assert all(row["calibration_supported_count"] == 2 for row in metrics)
        assert "verified" not in inspect.signature(
            runner.load_outer_references_after_prediction
        ).parameters

        bad_output = temporary_path / "bad_prediction"
        bad_output.mkdir()
        bad_arrays = {name: np.array(values, copy=True) for name, values in arrays.items()}
        bad_arrays["prediction"][0] = ~bad_arrays["prediction"][0]
        bad_path, bad_manifest_path, _, bad_manifest_sha = runner.write_outer_prediction(
            bad_output,
            bad_arrays,
            group_audits,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            calibration=calibration,
            outer_family="half_cylinder",
            git_commit="0" * 40,
        )
        _assert_raises(
            (RuntimeError, ValueError),
            runner.authenticate_outer_prediction,
            bad_path,
            bad_manifest_path,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            calibration=calibration,
            outer_projections=outer_projections,
            expected_outer_rows=outer_rows,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            device="cpu",
            expected_manifest_file_sha256=bad_manifest_sha,
        )

        bad_query_output = temporary_path / "bad_query_prediction"
        bad_query_output.mkdir()
        bad_query_arrays = {name: np.array(values, copy=True) for name, values in arrays.items()}
        bad_query_arrays["raw_negative_distance"][0] += np.float32(0.125)
        bad_query_path, bad_query_manifest_path, _, bad_query_manifest_sha = runner.write_outer_prediction(
            bad_query_output,
            bad_query_arrays,
            group_audits,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            calibration=calibration,
            outer_family="half_cylinder",
            git_commit="0" * 40,
        )
        _assert_raises(
            ValueError,
            runner.authenticate_outer_prediction,
            bad_query_path,
            bad_query_manifest_path,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            calibration=calibration,
            outer_projections=outer_projections,
            expected_outer_rows=outer_rows,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            device="cpu",
            expected_manifest_file_sha256=bad_query_manifest_sha,
        )

        selected_bytes = selected_path.read_bytes()
        selected_path.write_bytes(selected_bytes + b"\n")
        _assert_raises(
            ValueError,
            runner.load_outer_references_after_prediction,
            plan,
            selected,
            output,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            device="cpu",
            expected_calibration_manifest_sha256=calibration_manifest_sha,
            expected_selected_candidate_sha256=selected_sha,
            expected_prediction_manifest_sha256=prediction_manifest_sha,
            inner_group_metrics_path=inner_metrics_path,
            inner_group_metrics_sha256=inner_metrics_sha,
            inner_candidate_summary_path=inner_summary_path,
            inner_candidate_summary_sha256=inner_summary_sha,
            inner_fit_audits_path=inner_audits_path,
            inner_fit_audits_sha256=inner_audits_sha,
        )
        selected_path.write_bytes(selected_bytes)

        incomplete_manifest = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
        incomplete_manifest["rows"] = incomplete_manifest["rows"][:-1]
        incomplete_path = temporary_path / "incomplete_manifest.json"
        incomplete_path.write_text(
            json.dumps(incomplete_manifest, sort_keys=True), encoding="utf-8"
        )
        incomplete_plan = replace(
            plan,
            manifest_path=incomplete_path,
            manifest_size=incomplete_path.stat().st_size,
            manifest_sha256=sha256_file(incomplete_path),
        )
        _assert_raises(
            ValueError,
            runner.load_outer_references_after_prediction,
            incomplete_plan,
            selected,
            output,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            device="cpu",
            expected_calibration_manifest_sha256=calibration_manifest_sha,
            expected_selected_candidate_sha256=selected_sha,
            expected_prediction_manifest_sha256=prediction_manifest_sha,
            inner_group_metrics_path=inner_metrics_path,
            inner_group_metrics_sha256=inner_metrics_sha,
            inner_candidate_summary_path=inner_summary_path,
            inner_candidate_summary_sha256=inner_summary_sha,
            inner_fit_audits_path=inner_audits_path,
            inner_fit_audits_sha256=inner_audits_sha,
        )

        with np.load(prediction_path, allow_pickle=False) as archive:
            tampered = {name: np.array(archive[name], copy=True) for name in archive.files}
        tampered["tail_anomaly"][0] = 0.123
        np.savez_compressed(prediction_path, **tampered)
        _assert_raises(
            (RuntimeError, ValueError),
            runner.authenticate_outer_prediction,
            prediction_path,
            prediction_manifest_path,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            calibration=calibration,
            outer_projections=outer_projections,
            expected_outer_rows=outer_rows,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            device="cpu",
            expected_manifest_file_sha256=prediction_manifest_sha,
        )
