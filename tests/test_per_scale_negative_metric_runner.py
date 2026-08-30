from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT, ROOT / "tests"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.per_scale_negative_metric import (  # noqa: E402
    SCALER_LOCAL_BLOCK_SHRINK,
    PerScaleNegativeTailModel,
)
from scripts import run_verify_per_scale_negative_metric_1_1 as runner  # noqa: E402
from test_negative_tail_runner import (  # noqa: E402
    _projection,
    _write_synthetic_input_manifest,
)


CONFIG = ROOT / "config" / "Verify_PerScaleNegativeMetric_1.1.yaml"


def _expect_error(error_types, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_types:
        return
    raise AssertionError("expected an exception")


def _model(*, second_scale: int = 1) -> PerScaleNegativeTailModel:
    features = np.zeros((6, 161), dtype=np.float32)
    features[:, 0] = np.asarray([0.1, 1.2, 3.7, 10.0, 12.5, 18.2], dtype=np.float32)
    features[:, 1] = np.asarray([2.0, 3.0, 7.0, -4.0, -1.0, 5.0], dtype=np.float32)
    return PerScaleNegativeTailModel(
        features,
        np.asarray([0, 0, 0, second_scale, second_scale, second_scale]),
        ks=(1,),
        device="cpu",
        query_chunk_size=2,
        library_chunk_size=2,
    )


def _selected() -> runner.TailCandidateSpec:
    return runner.TailCandidateSpec(
        "fmt161", 1, 0.0, "calibrated_tail_anomaly_threshold", 0.5
    )


def test_spatial_replay_portability_gate_is_narrow_and_field_specific():
    assert dict(runner.SPATIAL_REPLAY_ULP_BOUNDS) == {
        "spatial_score": 8,
        "spatial_denominator": 8,
    }
    base = np.asarray([0.0, 0.125, 0.5, 1.0], dtype=np.float64)
    within = base.copy()
    for _ in range(8):
        within[1:] = np.nextafter(within[1:], np.inf)
    assert runner._require_portable_spatial_replay(
        "spatial_score", base, within
    ) == 8
    assert runner._require_portable_spatial_replay(
        "spatial_denominator", base, within
    ) == 8

    beyond = within.copy()
    beyond[1:] = np.nextafter(beyond[1:], np.inf)
    _expect_error(
        ValueError,
        runner._require_portable_spatial_replay,
        "spatial_score",
        base,
        beyond,
    )
    zero_drift = base.copy()
    zero_drift[0] = np.nextafter(np.float64(0.0), np.float64(np.inf))
    _expect_error(
        ValueError,
        runner._require_portable_spatial_replay,
        "spatial_score",
        base,
        zero_drift,
    )
    _expect_error(
        ValueError,
        runner._require_portable_spatial_replay,
        "raw_negative_distance",
        base.astype(np.float32),
        base.astype(np.float32),
    )


def _write_inner_evidence(
    output: Path,
    plan: runner.Plan,
    selected: runner.TailCandidateSpec,
    *,
    outer_family: str,
):
    candidates = runner.candidate_specs(plan)
    groups = [
        (family, dataset, source_ordinal, block)
        for family in plan.family_order
        if family != outer_family
        for dataset in plan.families[family]
        for source_ordinal in range(4)
        for block in runner.BLOCK_NAMES
    ]

    def score(candidate):
        return 0.9 if candidate.candidate_id == selected.candidate_id else 0.1

    summaries = [
        {
            **runner._candidate_payload(candidate),
            **{
                field: score(candidate)
                for field in (
                    "accuracy",
                    "average_precision",
                    "f1",
                    "balanced_accuracy",
                    "auroc",
                    "precision",
                    "recall",
                    "retrieval_support_fraction",
                    "calibration_support_fraction",
                    "spatial_imputed_fraction",
                    "spatial_unimputable_fraction",
                )
            },
            "inner_family_count": 4,
            "group_count": len(groups),
        }
        for candidate in candidates
    ]

    def metric_rows():
        for candidate in candidates:
            value = score(candidate)
            for family, dataset, source_ordinal, block in groups:
                row = {field: 0 for field in runner.METRIC_FIELDS}
                row.update(
                    {
                        "outer_family": outer_family,
                        "inner_family": family,
                        "dataset": dataset,
                        "source_ordinal": source_ordinal,
                        "block": block,
                        **runner._candidate_payload(candidate),
                    }
                )
                for field in (
                    "accuracy",
                    "average_precision",
                    "f1",
                    "balanced_accuracy",
                    "auroc",
                    "precision",
                    "recall",
                    "retrieval_support_fraction",
                    "calibration_support_fraction",
                    "spatial_imputed_fraction",
                    "spatial_unimputable_fraction",
                    "retrieval_supported_subset_f1",
                    "calibration_supported_subset_f1",
                    "imputed_subset_f1",
                    "unimputable_subset_f1",
                ):
                    row[field] = value
                yield row

    metrics_path = output / "inner_group_metrics.csv"
    summary_path = output / "inner_candidate_summary.csv"
    audits_path = output / "inner_fit_audits.json"
    metrics_sha = runner._atomic_csv(
        metrics_path, runner.METRIC_FIELDS, metric_rows()
    )
    summary_sha = runner._atomic_csv(
        summary_path, runner.SUMMARY_FIELDS, summaries
    )
    fits = [
        {"inner_family": family, "representation": representation}
        for family in plan.family_order
        if family != outer_family
        for representation in plan.representations
    ]
    audits_sha = runner._atomic_json(
        audits_path,
        runner._manifest_with_self_hash(
            {
                "schema": "pathline_template_matching.per_scale_negative_metric_inner_fit_audits.v1",
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


def test_per_scale_plan_candidate_grid_and_output_contract_are_exact():
    plan = runner.load_plan(CONFIG)
    assert plan.sha256 == runner.EXPECTED_CONFIG_SHA256
    assert len(runner.candidate_specs(plan)) == 3060
    assert len(plan.required_fold_files) == 15
    assert "final_per_scale_scaler.npz" in plan.required_fold_files
    assert "final_per_scale_scaler_manifest.json" in plan.required_fold_files
    assert plan.raw["library"]["local_variance_ddof"] == 0
    assert plan.raw["library"]["shrinkage_domain"] == "variance_before_square_root"


def test_per_scale_runner_query_path_is_rank_free_and_reports_scaler_modes():
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "spatial_support_scores" not in called_names
    assert "supported_rank_scores" not in called_names
    source = inspect.getsource(runner._fit_tail_model)
    assert "PerScaleNegativeTailModel" in source

    plan = runner.load_plan(CONFIG)
    fit = _projection(
        dataset="channel",
        family="channel",
        values=[0.0, 1.0, 4.0, 10.0, 14.0],
        scales=[0, 0, 0, 1, 1],
        centers=[0, 1, 2, 3, 4],
        labels=[False, False, False, False, False],
    )
    model = runner._fit_tail_model(
        [fit], "fmt161", plan, device="cpu", ks=(1,)
    )
    assert isinstance(model, PerScaleNegativeTailModel)
    query = _projection(
        dataset="channel",
        family="channel",
        values=[0.5, 20.0],
        scales=[0, 77],
        centers=[5, 6],
        labels=None,
    )
    values = runner._query_cache_batch(
        model, [query], "fmt161", plan, device="cpu", ks=(1,)
    )[1][0]
    assert values["scaler_mode"].tolist() == [SCALER_LOCAL_BLOCK_SHRINK, 0]
    assert values["retrieval_supported"].tolist() == [True, False]
    assert np.isnan(values["raw_distance"][1])


def test_final_scaler_then_tail_artifacts_round_trip_and_bind_each_other():
    plan = runner.load_plan(CONFIG)
    selected = _selected()
    fit_families = ("delta_wing", "f22_raptor", "channel", "boeing_747")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        model = _model()
        scaler_path, scaler_manifest_path, _, scaler_manifest_sha = (
            runner.write_final_scaler_artifact(
                output,
                model,
                plan=plan,
                selected=selected,
                outer_family="half_cylinder",
                fit_families=fit_families,
                git_commit="0" * 40,
            )
        )
        scaler = runner.authenticate_and_rebuild_final_scaler(
            scaler_path,
            scaler_manifest_path,
            plan=plan,
            selected=selected,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit="0" * 40,
            expected_manifest_file_sha256=scaler_manifest_sha,
        )
        calibration_path, calibration_manifest_path, _, calibration_manifest_sha = (
            runner.write_final_calibration_artifact(
                output,
                model,
                plan=plan,
                selected=selected,
                scaler=scaler,
                outer_family="half_cylinder",
                fit_families=fit_families,
                git_commit="0" * 40,
            )
        )
        calibration = runner.authenticate_and_rebuild_final_calibration(
            calibration_path,
            calibration_manifest_path,
            plan=plan,
            selected=selected,
            scaler=scaler,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit="0" * 40,
            expected_manifest_file_sha256=calibration_manifest_sha,
        )
        assert calibration.model.ks == (1,)
        assert (
            calibration.manifest["final_per_scale_scaler_manifest"][
                "file_sha256"
            ]
            == scaler.manifest_file_sha256
        )

        # A tail model fitted with another scaler cannot be bound to this
        # already-authenticated scaler even when width and row count match.
        _expect_error(
            ValueError,
            runner.write_final_calibration_artifact,
            output / "unused",
            _model(second_scale=2),
            plan=plan,
            selected=selected,
            scaler=scaler,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit="0" * 40,
        )


def test_final_scaler_authentication_rejects_formula_valid_shape_tamper():
    plan = runner.load_plan(CONFIG)
    selected = _selected()
    fit_families = ("delta_wing", "f22_raptor", "channel", "boeing_747")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        scaler_path, manifest_path, _, manifest_sha = (
            runner.write_final_scaler_artifact(
                output,
                _model(),
                plan=plan,
                selected=selected,
                outer_family="half_cylinder",
                fit_families=fit_families,
                git_commit="0" * 40,
            )
        )
        with np.load(scaler_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        arrays["shrunk_variance_float64"][0, 0] += 0.125
        np.savez_compressed(scaler_path, **arrays)
        _expect_error(
            (RuntimeError, ValueError),
            runner.authenticate_and_rebuild_final_scaler,
            scaler_path,
            manifest_path,
            plan=plan,
            selected=selected,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit="0" * 40,
            expected_manifest_file_sha256=manifest_sha,
        )


def test_synthetic_double_artifact_prediction_and_label_gate_end_to_end():
    base_plan = runner.load_plan(CONFIG)
    selected = _selected()
    fit_families = ("delta_wing", "f22_raptor", "channel", "boeing_747")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "output"
        output.mkdir()
        plan = _write_synthetic_input_manifest(
            root, base_plan, outer_family="half_cylinder"
        )
        evidence = _write_inner_evidence(
            output, plan, selected, outer_family="half_cylinder"
        )
        (
            inner_metrics_path,
            inner_metrics_sha,
            inner_summary_path,
            inner_summary_sha,
            inner_audits_path,
            inner_audits_sha,
            selected_summary,
        ) = evidence
        model = _model(second_scale=1000)
        scaler_path, scaler_manifest_path, _, scaler_manifest_sha = (
            runner.write_final_scaler_artifact(
                output,
                model,
                plan=plan,
                selected=selected,
                outer_family="half_cylinder",
                fit_families=fit_families,
                git_commit="0" * 40,
            )
        )
        scaler = runner.authenticate_and_rebuild_final_scaler(
            scaler_path,
            scaler_manifest_path,
            plan=plan,
            selected=selected,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit="0" * 40,
            expected_manifest_file_sha256=scaler_manifest_sha,
        )
        calibration_path, calibration_manifest_path, _, calibration_manifest_sha = (
            runner.write_final_calibration_artifact(
                output,
                model,
                plan=plan,
                selected=selected,
                scaler=scaler,
                outer_family="half_cylinder",
                fit_families=fit_families,
                git_commit="0" * 40,
            )
        )
        calibration = runner.authenticate_and_rebuild_final_calibration(
            calibration_path,
            calibration_manifest_path,
            plan=plan,
            selected=selected,
            scaler=scaler,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit="0" * 40,
            expected_manifest_file_sha256=calibration_manifest_sha,
        )
        selected_path, selected_sha, selected_payload = (
            runner.write_selected_candidate(
                output,
                plan=plan,
                selected=selected,
                selected_summary=selected_summary,
                scaler=scaler,
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
        )
        assert selected_payload["final_per_scale_scaler_manifest"][
            "file_sha256"
        ] == scaler.manifest_file_sha256
        selected_artifact = runner.authenticate_selected_candidate(
            selected_path,
            plan=plan,
            selected=selected,
            scaler=scaler,
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
        all_rows, _ = runner.load_cache_rows(plan)
        outer_rows = [row for row in all_rows if row.family == "half_cylinder"]
        outer_projections = [
            runner.load_cache_projection(plan, row, include_labels=False)
            for row in outer_rows
        ]
        arrays, group_audits = runner.build_outer_prediction_arrays(
            outer_projections,
            calibration.model,
            selected,
            plan,
            device="cpu",
        )
        assert "scaler_mode" in arrays
        prediction_path, prediction_manifest_path, _, prediction_manifest_sha = (
            runner.write_outer_prediction(
                output,
                arrays,
                group_audits,
                plan=plan,
                selected=selected,
                selected_artifact=selected_artifact,
                scaler=scaler,
                calibration=calibration,
                outer_family="half_cylinder",
                git_commit="0" * 40,
            )
        )
        verified = runner.authenticate_outer_prediction(
            prediction_path,
            prediction_manifest_path,
            plan=plan,
            selected=selected,
            selected_artifact=selected_artifact,
            scaler=scaler,
            calibration=calibration,
            outer_projections=outer_projections,
            expected_outer_rows=outer_rows,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            device="cpu",
            expected_manifest_file_sha256=prediction_manifest_sha,
        )
        assert verified.manifest["final_per_scale_scaler_file_sha256"] == scaler.scaler_file_sha256
        metrics, reference_audit = runner.evaluate_outer_prediction(
            plan,
            selected,
            output,
            outer_family="half_cylinder",
            git_commit="0" * 40,
            device="cpu",
            expected_scaler_manifest_sha256=scaler_manifest_sha,
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
        assert all(
            sum(row[f"scaler_mode_{mode}_count"] for mode in range(4))
            == row["sample_count"]
            for row in metrics
        )

        # Damage the scaler only after a successful prediction.  A fresh label
        # gate must stop at scaler authentication and never request labels.
        original_scaler = scaler_path.read_bytes()
        scaler_path.write_bytes(original_scaler + b"damage")
        original_loader = runner.load_cache_projection

        def guarded_loader(plan_value, row, *, include_labels):
            if include_labels:
                raise AssertionError("outer label opened after scaler damage")
            return original_loader(plan_value, row, include_labels=False)

        runner.load_cache_projection = guarded_loader
        try:
            _expect_error(
                ValueError,
                runner.load_outer_references_after_prediction,
                plan,
                selected,
                output,
                outer_family="half_cylinder",
                git_commit="0" * 40,
                device="cpu",
                expected_scaler_manifest_sha256=scaler_manifest_sha,
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
        finally:
            runner.load_cache_projection = original_loader
