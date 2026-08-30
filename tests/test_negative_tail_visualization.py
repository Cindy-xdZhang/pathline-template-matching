from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from pathline_template_matching.negative_tail_visualization import (
    BLOCKS,
    DATASETS,
    EXPERIMENT,
    PANEL_TITLES,
    PREDICTION_ARRAY_NAMES,
    PREDICTION_DTYPES,
    TAIL_RESULT_ARTIFACT_NAMES,
    AuthenticatedFold,
    PredictionGroup,
    _audit_pdf_text,
    _authenticate_exact_fold_file_set,
    _stable_file,
    _validate_fold_chain,
    compare_metrics_to_parent,
    exact_bind_prediction_group,
    load_prediction_groups,
    load_visualization_plan,
    run_negative_tail_visualization,
)
from pathline_template_matching.portable_flow import (
    canonical_array_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/Other_NegativeTailVisualization_1.1.yaml"
CONFIG_SHA256 = "5a82a9d1af406043066316262e5dcefb1a0d559f6d66e82da16440a2066df131"


def test_negative_tail_visualization_plan_freezes_two_outer_candidates_and_eight_figures():
    plan = load_visualization_plan(CONFIG)
    assert plan.config_sha256 == CONFIG_SHA256
    assert plan.config["experiment"] == EXPERIMENT
    assert tuple(value.dataset for value in plan.datasets) == DATASETS
    assert tuple(value.block_id for value in plan.blocks) == BLOCKS
    assert tuple(plan.config["figure_contract"]["panel_titles"]) == PANEL_TITLES
    assert tuple(plan.config["prediction_contract"]["ordered_array_names"]) == PREDICTION_ARRAY_NAMES
    assert plan.folds[0].candidate["candidate_id"].startswith(
        "representation=chirality_all35|k=15"
    )
    assert plan.folds[1].candidate["candidate_id"].startswith(
        "representation=real_neighbor36|k=1"
    )
    assert tuple(fold.expected_inner_group_count for fold in plan.folds) == (40, 56)
    assert plan.assigned_count == 64_000
    assert plan.metric_tolerance == 1e-12
    assert plan.png_dpi == 360


def test_visualization_plan_rejects_any_config_byte_drift_even_outside_parsed_fields():
    with tempfile.TemporaryDirectory() as directory:
        changed = Path(directory) / CONFIG.name
        text = CONFIG.read_text(encoding="utf-8")
        changed.write_text(text.replace("purpose: audited_", "purpose: changed_audited_", 1), encoding="utf-8")
        try:
            load_visualization_plan(changed)
        except ValueError as error:
            assert "config SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("modified frozen visualization config was accepted")


def _parent_identity(count: int = 3) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    center = np.arange(count, dtype=np.int64)
    return (
        {
            "dataset": "cylinder3d",
            "source_ordinal": 2,
            "source_index": 68,
            "scale_block_id": "legacy_2_1",
        },
        {
            "valid_center_seed_index": center,
            "valid_assigned_row_index": center.copy(),
            "valid_scale_id": np.arange(count, dtype=np.int32),
            "valid_scale_block_index": np.zeros(count, dtype=np.int8),
        },
    )


def _prediction_group(count: int = 3) -> PredictionGroup:
    center = np.arange(count, dtype=np.int64)
    return PredictionGroup(
        dataset="cylinder3d",
        source_ordinal=2,
        source_index=68,
        block="legacy_2_1",
        outer_family="half_cylinder",
        candidate={
            "candidate_id": "representation=chirality_all35|k=15|sigma=1.0|fixed_top_fraction=0.05",
            "representation": "chirality_all35",
            "k": 15,
            "sigma": 1.0,
            "decision_rule": "fixed_top_fraction",
            "decision_value": 0.05,
        },
        center_seed_index=center,
        assigned_row_index=center.copy(),
        scale_id=np.arange(count, dtype=np.int32),
        scale_block_index=np.zeros(count, dtype=np.int8),
        spatial_score=np.linspace(0.1, 0.9, count, dtype=np.float64),
        prediction=np.asarray([False, True, False][:count], dtype=np.bool_),
    )


def test_exact_binding_accepts_only_same_identity_population_and_order():
    metadata, arrays = _parent_identity()
    group = _prediction_group()
    prediction, score = exact_bind_prediction_group(metadata, arrays, group)
    assert np.array_equal(prediction, group.prediction)
    assert np.array_equal(score, group.spatial_score)

    reordered = replace(
        group,
        center_seed_index=group.center_seed_index[::-1],
        assigned_row_index=group.assigned_row_index[::-1],
        scale_id=group.scale_id[::-1],
        scale_block_index=group.scale_block_index[::-1],
        spatial_score=group.spatial_score[::-1],
        prediction=group.prediction[::-1],
    )
    try:
        exact_bind_prediction_group(metadata, arrays, reordered)
    except ValueError as error:
        assert "row order differs" in str(error)
    else:
        raise AssertionError("reordered NegativeTail rows were accepted")

    missing = replace(
        group,
        center_seed_index=group.center_seed_index[:-1],
        assigned_row_index=group.assigned_row_index[:-1],
        scale_id=group.scale_id[:-1],
        scale_block_index=group.scale_block_index[:-1],
        spatial_score=group.spatial_score[:-1],
        prediction=group.prediction[:-1],
    )
    try:
        exact_bind_prediction_group(metadata, arrays, missing)
    except ValueError as error:
        assert "missing=1" in str(error)
    else:
        raise AssertionError("missing NegativeTail row was accepted")

    extra = replace(
        group,
        center_seed_index=np.append(group.center_seed_index, 9),
        assigned_row_index=np.append(group.assigned_row_index, 9),
        scale_id=np.append(group.scale_id, np.int32(9)),
        scale_block_index=np.append(group.scale_block_index, np.int8(0)),
        spatial_score=np.append(group.spatial_score, 0.5),
        prediction=np.append(group.prediction, False),
    )
    try:
        exact_bind_prediction_group(metadata, arrays, extra)
    except ValueError as error:
        assert "extra=1" in str(error)
    else:
        raise AssertionError("extra NegativeTail row was accepted")


def test_exact_binding_rejects_duplicate_and_tampered_composite_identity():
    metadata, arrays = _parent_identity()
    group = _prediction_group()
    duplicate = replace(
        group,
        center_seed_index=np.asarray([0, 0, 2], dtype=np.int64),
        assigned_row_index=np.asarray([0, 0, 2], dtype=np.int64),
        scale_id=np.asarray([0, 0, 2], dtype=np.int32),
    )
    try:
        exact_bind_prediction_group(metadata, arrays, duplicate)
    except ValueError as error:
        assert "duplicate center_seed_index" in str(error)
    else:
        raise AssertionError("duplicate NegativeTail identity was accepted")

    changed_scale = group.scale_id.copy()
    changed_scale[1] = 7
    tampered = replace(group, scale_id=changed_scale)
    try:
        exact_bind_prediction_group(metadata, arrays, tampered)
    except ValueError as error:
        assert "exact join failed" in str(error)
    else:
        raise AssertionError("tampered scale identity was accepted")

    wrong_fold = replace(group, outer_family="boeing_747", dataset="boeing747")
    try:
        exact_bind_prediction_group(metadata, arrays, wrong_fold)
    except ValueError as error:
        assert "dataset differs" in str(error)
    else:
        raise AssertionError("cross-fold dataset identity was accepted")

    same_center_different_scale = replace(
        group,
        center_seed_index=np.asarray([0, 0, 2], dtype=np.int64),
        assigned_row_index=np.asarray([0, 0, 2], dtype=np.int64),
        scale_id=np.asarray([0, 1, 2], dtype=np.int32),
    )
    assert len(
        set(
            zip(
                same_center_different_scale.center_seed_index.tolist(),
                same_center_different_scale.assigned_row_index.tolist(),
                same_center_different_scale.scale_id.tolist(),
                same_center_different_scale.scale_block_index.tolist(),
            )
        )
    ) == 3
    try:
        exact_bind_prediction_group(metadata, arrays, same_center_different_scale)
    except ValueError as error:
        assert "duplicate center_seed_index" in str(error)
    else:
        raise AssertionError("same center with different scale bypassed the one-to-one join")


def _valid_fold_chain(fold):
    final_file_sha = "1" * 64
    final_manifest_file_sha = "2" * 64
    final_manifest_content_sha = "3" * 64
    result = {
        "schema": "pathline_template_matching.negative_tail_result.v1",
        "experiment": "Verify_NegativeTailCalibration_1.1",
        "status": "completed",
        "git_commit": "e9d4d3f11428bd2e13fc0fabf657be7c7e57db7c",
        "config_sha256": "4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b",
        "outer_family": fold.outer_family,
        "content_sha256": fold.result_manifest_content_sha256,
        "selected_candidate": dict(fold.candidate),
        "artifacts": {
            "outer_prediction_manifest.json": {"sha256": fold.prediction_manifest_sha256},
            "outer_predictions.npz": {"sha256": fold.predictions_sha256},
            "selected_candidate.json": {"sha256": fold.selected_candidate_sha256},
            "outer_group_metrics.csv": {"sha256": fold.outer_group_metrics_sha256},
            "inner_candidate_summary.csv": {"sha256": "5" * 64, "size_bytes": 101},
            "inner_fit_audits.json": {"sha256": "6" * 64, "size_bytes": 102},
            "inner_group_metrics.csv": {"sha256": "7" * 64, "size_bytes": 103},
        },
        "prediction_manifest_file_sha256": fold.prediction_manifest_sha256,
        "prediction_file_sha256": fold.predictions_sha256,
        "selected_candidate_file_sha256": fold.selected_candidate_sha256,
        "selected_candidate_content_sha256": fold.selected_candidate_content_sha256,
        "outer_group_metrics_file_sha256": fold.outer_group_metrics_sha256,
        "final_calibration_file_sha256": final_file_sha,
        "final_calibration_manifest_file_sha256": final_manifest_file_sha,
    }
    arrays = {
        name: {
            "dtype": PREDICTION_DTYPES[name],
            "shape": [fold.predictions_row_count],
            "sha256": "4" * 64,
        }
        for name in PREDICTION_ARRAY_NAMES
    }
    prediction_manifest = {
        "schema": "pathline_template_matching.negative_tail_outer_prediction_manifest.v1",
        "prediction_schema": "pathline_template_matching.negative_tail_outer_prediction.v1",
        "experiment": "Verify_NegativeTailCalibration_1.1",
        "git_commit": "e9d4d3f11428bd2e13fc0fabf657be7c7e57db7c",
        "config_sha256": "4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b",
        "outer_family": fold.outer_family,
        "content_sha256": fold.prediction_manifest_content_sha256,
        "row_count": fold.predictions_row_count,
        "array_count": 18,
        "arrays": arrays,
        "prediction_file": {
            "path": "outer_predictions.npz",
            "sha256": fold.predictions_sha256,
            "size_bytes": fold.predictions_size_bytes,
        },
        "selected_candidate": dict(fold.candidate),
        "selected_candidate_artifact": {
            "path": "selected_candidate.json",
            "file_sha256": fold.selected_candidate_sha256,
            "content_sha256": fold.selected_candidate_content_sha256,
        },
        "valid_labels_opened": False,
        "metadata_json_opened": False,
        "final_calibration_file_sha256": final_file_sha,
        "final_calibration_manifest": {
            "path": "final_tail_calibration_manifest.json",
            "file_sha256": final_manifest_file_sha,
            "content_sha256": final_manifest_content_sha,
        },
    }
    selected = {
        "schema": "pathline_template_matching.negative_tail_selected_candidate.v1",
        "experiment": "Verify_NegativeTailCalibration_1.1",
        "git_commit": "e9d4d3f11428bd2e13fc0fabf657be7c7e57db7c",
        "config_sha256": "4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b",
        "outer_family": fold.outer_family,
        "content_sha256": fold.selected_candidate_content_sha256,
        "candidate": dict(fold.candidate),
        "outer_feature_member_opened": False,
        "candidate_count": 3060,
        "inner_evidence": {
            "inner_candidate_summary": {
                "path": "inner_candidate_summary.csv",
                "sha256": "5" * 64,
                "size_bytes": 101,
            },
            "inner_fit_audits": {
                "path": "inner_fit_audits.json",
                "sha256": "6" * 64,
                "size_bytes": 102,
            },
            "inner_group_metrics": {
                "path": "inner_group_metrics.csv",
                "sha256": "7" * 64,
                "size_bytes": 103,
            },
        },
        "inner_selection_summary": {
            **dict(fold.candidate),
            "inner_family_count": 4,
            "group_count": 40 if fold.outer_family == "half_cylinder" else 56,
        },
        "final_calibration_file": {
            "path": "final_tail_calibration.npz",
            "sha256": final_file_sha,
        },
        "final_calibration_manifest": {
            "path": "final_tail_calibration_manifest.json",
            "file_sha256": final_manifest_file_sha,
            "content_sha256": final_manifest_content_sha,
        },
    }
    complete = {
        "schema": "pathline_template_matching.negative_tail_run_complete.v1",
        "experiment": "Verify_NegativeTailCalibration_1.1",
        "git_commit": "e9d4d3f11428bd2e13fc0fabf657be7c7e57db7c",
        "config_sha256": "4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b",
        "outer_family": fold.outer_family,
        "content_sha256": fold.run_complete_content_sha256,
        "result_manifest_file": "result_manifest.json",
        "result_manifest_file_sha256": fold.result_manifest_sha256,
        "result_manifest_content_sha256": fold.result_manifest_content_sha256,
    }
    return result, prediction_manifest, selected, complete


def test_fold_chain_rejects_wrong_outer_fold_and_candidate():
    fold = load_visualization_plan(CONFIG).folds[0]
    values = _valid_fold_chain(fold)
    _validate_fold_chain(fold, *values)

    wrong_fold = copy.deepcopy(values)
    wrong_fold[1]["outer_family"] = "boeing_747"
    try:
        _validate_fold_chain(fold, *wrong_fold)
    except ValueError as error:
        assert "wrong outer fold" in str(error)
    else:
        raise AssertionError("wrong prediction outer fold was accepted")

    wrong_candidate = copy.deepcopy(values)
    wrong_candidate[2]["candidate"]["k"] = 1
    try:
        _validate_fold_chain(fold, *wrong_candidate)
    except ValueError as error:
        assert "wrong frozen outer-fold candidate" in str(error)
    else:
        raise AssertionError("wrong selected candidate was accepted")

    boeing_fold = load_visualization_plan(CONFIG).folds[1]
    boeing_values = _valid_fold_chain(boeing_fold)
    _validate_fold_chain(boeing_fold, *boeing_values)
    assert boeing_values[2]["inner_selection_summary"]["group_count"] == 56
    wrong_boeing_population = copy.deepcopy(boeing_values)
    wrong_boeing_population[2]["inner_selection_summary"]["group_count"] = 40
    try:
        _validate_fold_chain(boeing_fold, *wrong_boeing_population)
    except ValueError as error:
        assert "inner selection group population changed" in str(error)
    else:
        raise AssertionError("half-cylinder group count was accepted for Boeing fold")


def test_metric_comparison_and_file_hash_are_fail_closed():
    metrics = {
        "sample_count": 3,
        "positive_count": 1,
        "negative_count": 2,
        "true_positive": 1,
        "false_positive": 0,
        "true_negative": 2,
        "false_negative": 0,
        "accuracy": 1.0,
        "average_precision": 1.0,
        "f1": 1.0,
        "balanced_accuracy": 1.0,
        "auroc": 1.0,
        "precision": 1.0,
        "recall": 1.0,
    }
    parent = {name: str(value) for name, value in metrics.items()}
    compare_metrics_to_parent(metrics, parent, 1e-12)
    changed = dict(parent)
    changed["f1"] = str(1.0 - 2e-12)
    try:
        compare_metrics_to_parent(metrics, changed, 1e-12)
    except ValueError as error:
        assert "f1 mismatch" in str(error)
    else:
        raise AssertionError("metric mismatch outside tolerance was accepted")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "input.bin"
        path.write_bytes(b"authenticated")
        _stable_file(path, sha256_file(path), role="test")
        try:
            _stable_file(path, "0" * 64, role="test")
        except ValueError as error:
            assert "SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("wrong input file hash was accepted")


def _synthetic_prediction_arrays(datasets: tuple[str, ...]) -> dict[str, np.ndarray]:
    rows = [
        (dataset, source, block, local_center)
        for dataset in datasets
        for source in range(4)
        for block in range(2)
        for local_center in range(2)
    ]
    count = len(rows)
    dataset = np.asarray([value[0] for value in rows], dtype="<U64")
    source = np.asarray([value[1] for value in rows], dtype="<i2")
    block = np.asarray([value[2] for value in rows], dtype="|i1")
    center = np.asarray([value[3] for value in rows], dtype="<i8")
    scale = block.astype("<i4") * 1000 + center.astype("<i4")
    assigned = center + block.astype("<i8") * 64_000
    zeros_f4 = np.zeros(count, dtype="<f4")
    zeros_f8 = np.zeros(count, dtype="<f8")
    true = np.ones(count, dtype="|b1")
    false = np.zeros(count, dtype="|b1")
    return {
        "dataset": dataset,
        "source_ordinal": source,
        "source_index": source.astype("<i8") * 10,
        "scale_id": scale,
        "center_seed_index": center,
        "scale_block_index": block,
        "assigned_row_index": assigned,
        "raw_negative_distance": zeros_f4,
        "tail_probability": zeros_f8,
        "tail_anomaly": zeros_f8,
        "spatial_score": zeros_f8,
        "spatial_denominator": zeros_f8,
        "retrieval_supported": true,
        "calibration_supported": true,
        "spatial_imputed": false,
        "spatial_unimputable": false,
        "calibration_mode": np.ones(count, dtype="|i1"),
        "prediction": false,
    }


def test_complete_18_array_schema_and_hash_are_verified_before_projection():
    plan = load_visualization_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        authenticated = []
        for original in plan.folds:
            fold_root = root / original.outer_family
            fold_root.mkdir()
            datasets = tuple(
                value.dataset
                for value in plan.datasets
                if value.outer_family == original.outer_family
            )
            arrays = _synthetic_prediction_arrays(datasets)
            np.savez_compressed(
                fold_root / "outer_predictions.npz",
                **{name: arrays[name] for name in PREDICTION_ARRAY_NAMES},
            )
            prediction_path = fold_root / "outer_predictions.npz"
            spec = replace(
                original,
                run_root=fold_root,
                predictions_sha256=sha256_file(prediction_path),
                predictions_size_bytes=prediction_path.stat().st_size,
                predictions_row_count=len(arrays["dataset"]),
            )
            manifest = {
                "arrays": {
                    name: {
                        "dtype": arrays[name].dtype.str,
                        "shape": list(arrays[name].shape),
                        "sha256": canonical_array_sha256(arrays[name]),
                    }
                    for name in PREDICTION_ARRAY_NAMES
                }
            }
            authenticated.append(
                AuthenticatedFold(spec, {}, manifest, {}, {})
            )
        groups = load_prediction_groups(plan, authenticated)
        assert set(groups) == {
            (dataset, block) for dataset in DATASETS for block in BLOCKS
        }

        changed = list(authenticated)
        bad_manifest = copy.deepcopy(changed[0].prediction_manifest)
        bad_manifest["arrays"]["scale_id"]["sha256"] = "0" * 64
        changed[0] = replace(changed[0], prediction_manifest=bad_manifest)
        try:
            load_prediction_groups(plan, changed)
        except ValueError as error:
            assert "array hash mismatch" in str(error)
        else:
            raise AssertionError("tampered 18-array manifest hash was accepted")


def test_run_writes_input_manifest_before_npz_projection():
    plan = load_visualization_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run"

        def gate(_plan, _folds):
            assert (run_dir / "input_manifest.json").is_file()
            raise RuntimeError("projection gate reached")

        with patch(
            "pathline_template_matching.negative_tail_visualization.authenticate_inputs",
            return_value=([], (), []),
        ), patch(
            "pathline_template_matching.negative_tail_visualization.load_prediction_groups",
            side_effect=gate,
        ):
            try:
                run_negative_tail_visualization(
                    plan,
                    run_dir=run_dir,
                    git_commit="a" * 40,
                )
            except RuntimeError as error:
                assert str(error) == "projection gate reached"
            else:
                raise AssertionError("test projection gate was not reached")


def test_exact_tail_fold_file_set_is_result_plus_eleven_artifacts_plus_completion():
    original = load_visualization_plan(CONFIG).folds[0]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifacts = {}
        for name in TAIL_RESULT_ARTIFACT_NAMES:
            path = root / name
            path.write_bytes(f"authenticated:{name}\n".encode("utf-8"))
            artifacts[name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        result_path = root / "result_manifest.json"
        result_path.write_bytes(b"authenticated result manifest\n")
        complete_path = root / "RUN_COMPLETE.json"
        complete_path.write_bytes(b"authenticated completion marker\n")
        spec = replace(
            original,
            run_root=root,
            result_manifest_sha256=sha256_file(result_path),
            prediction_manifest_sha256=artifacts["outer_prediction_manifest.json"]["sha256"],
            predictions_sha256=artifacts["outer_predictions.npz"]["sha256"],
            predictions_size_bytes=artifacts["outer_predictions.npz"]["size_bytes"],
            selected_candidate_sha256=artifacts["selected_candidate.json"]["sha256"],
            outer_group_metrics_sha256=artifacts["outer_group_metrics.csv"]["sha256"],
            run_complete_sha256=sha256_file(complete_path),
        )
        result = {"artifacts": artifacts}
        result_evidence = _stable_file(
            result_path,
            spec.result_manifest_sha256,
            role="test_result_manifest",
        )
        evidence = _authenticate_exact_fold_file_set(spec, result, result_evidence)
        assert len(evidence) == 13
        assert len({row["path"] for row in evidence}) == 13
        assert {Path(row["path"]).name for row in evidence} == {
            "result_manifest.json",
            *TAIL_RESULT_ARTIFACT_NAMES,
            "RUN_COMPLETE.json",
        }

        extra = copy.deepcopy(result)
        extra["artifacts"]["unexpected.txt"] = {"sha256": "0" * 64, "size_bytes": 0}
        try:
            _authenticate_exact_fold_file_set(spec, extra, result_evidence)
        except ValueError as error:
            assert "exact 11-artifact set" in str(error)
        else:
            raise AssertionError("extra tail result artifact was accepted")

        missing = copy.deepcopy(result)
        missing["artifacts"].pop("inner_fit_audits.json")
        try:
            _authenticate_exact_fold_file_set(spec, missing, result_evidence)
        except ValueError as error:
            assert "exact 11-artifact set" in str(error)
        else:
            raise AssertionError("missing tail result artifact was accepted")


def test_prediction_projection_rejects_same_center_with_different_scale():
    plan = load_visualization_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        authenticated = []
        for original in plan.folds:
            fold_root = root / original.outer_family
            fold_root.mkdir()
            datasets = tuple(
                value.dataset
                for value in plan.datasets
                if value.outer_family == original.outer_family
            )
            arrays = _synthetic_prediction_arrays(datasets)
            if original.outer_family == "half_cylinder":
                mask = (
                    (arrays["dataset"] == "cylinder3d")
                    & (arrays["source_ordinal"] == 2)
                    & (arrays["scale_block_index"] == 0)
                )
                rows = np.flatnonzero(mask)
                assert len(rows) == 2 and arrays["scale_id"][rows[0]] != arrays["scale_id"][rows[1]]
                arrays["center_seed_index"][rows[1]] = arrays["center_seed_index"][rows[0]]
                arrays["assigned_row_index"][rows[1]] = arrays["assigned_row_index"][rows[0]]
            np.savez_compressed(
                fold_root / "outer_predictions.npz",
                **{name: arrays[name] for name in PREDICTION_ARRAY_NAMES},
            )
            prediction_path = fold_root / "outer_predictions.npz"
            spec = replace(
                original,
                run_root=fold_root,
                predictions_sha256=sha256_file(prediction_path),
                predictions_size_bytes=prediction_path.stat().st_size,
                predictions_row_count=len(arrays["dataset"]),
            )
            manifest = {
                "arrays": {
                    name: {
                        "dtype": arrays[name].dtype.str,
                        "shape": list(arrays[name].shape),
                        "sha256": canonical_array_sha256(arrays[name]),
                    }
                    for name in PREDICTION_ARRAY_NAMES
                }
            }
            authenticated.append(AuthenticatedFold(spec, {}, manifest, {}, {}))
        try:
            load_prediction_groups(plan, authenticated)
        except ValueError as error:
            assert "duplicate center_seed_index" in str(error)
        else:
            raise AssertionError("same center with different scale bypassed prediction projection")


def test_prediction_projection_requires_same_source_index_across_blocks():
    plan = load_visualization_plan(CONFIG)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        authenticated = []
        for original in plan.folds:
            fold_root = root / original.outer_family
            fold_root.mkdir()
            datasets = tuple(
                value.dataset
                for value in plan.datasets
                if value.outer_family == original.outer_family
            )
            arrays = _synthetic_prediction_arrays(datasets)
            if original.outer_family == "half_cylinder":
                changed = (
                    (arrays["dataset"] == "cylinder3d")
                    & (arrays["source_ordinal"] == 2)
                    & (arrays["scale_block_index"] == 1)
                )
                arrays["source_index"][changed] = 999
            np.savez_compressed(
                fold_root / "outer_predictions.npz",
                **{name: arrays[name] for name in PREDICTION_ARRAY_NAMES},
            )
            prediction_path = fold_root / "outer_predictions.npz"
            spec = replace(
                original,
                run_root=fold_root,
                predictions_sha256=sha256_file(prediction_path),
                predictions_size_bytes=prediction_path.stat().st_size,
                predictions_row_count=len(arrays["dataset"]),
            )
            manifest = {
                "arrays": {
                    name: {
                        "dtype": arrays[name].dtype.str,
                        "shape": list(arrays[name].shape),
                        "sha256": canonical_array_sha256(arrays[name]),
                    }
                    for name in PREDICTION_ARRAY_NAMES
                }
            }
            authenticated.append(AuthenticatedFold(spec, {}, manifest, {}, {}))
        try:
            load_prediction_groups(plan, authenticated)
        except ValueError as error:
            assert "different source_index" in str(error)
        else:
            raise AssertionError("different legacy/expanded physical sources were accepted")


def test_pdf_text_gate_is_dependency_free_and_fails_below_five_points():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        passed_pdf = root / "passed.pdf"
        passed_pdf.write_bytes(
            b"%PDF-1.4\n<< >>\nstream\n/F1 7 Tf\nendstream\n%%EOF\n"
        )
        report = _audit_pdf_text(passed_pdf, root / "passed.audit.json")
        assert report["status"] == "PASS"
        assert report["minimum_found_pt"] == 7.0
        assert report["below_minimum_count"] == 0

        failed_pdf = root / "failed.pdf"
        failed_pdf.write_bytes(
            b"%PDF-1.4\n<< >>\nstream\n/F1 4.99 Tf\nendstream\n%%EOF\n"
        )
        try:
            _audit_pdf_text(failed_pdf, root / "failed.audit.json")
        except ValueError as error:
            assert "below 5 pt" in str(error)
        else:
            raise AssertionError("sub-5-point PDF text was accepted")


def _run_zero_argument_tests() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"negative_tail_visualization_tests={len(tests)} PASS")
    return len(tests)


if __name__ == "__main__":
    _run_zero_argument_tests()
