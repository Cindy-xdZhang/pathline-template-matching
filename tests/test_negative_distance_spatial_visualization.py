from __future__ import annotations

import csv
import copy
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np

from pathline_template_matching.negative_distance_spatial_visualization import (
    BLOCKS,
    DATASETS,
    EXPERIMENT,
    PANEL_TITLES,
    PREDICTION_COLUMN,
    SCORE_COLUMN,
    PARENT_CANDIDATE_CONFIG_SHA256,
    PARENT_CANDIDATE_EXPERIMENT,
    PARENT_CANDIDATE_GIT_COMMIT,
    PARENT_PER_QUERY_PATH,
    PARENT_PER_QUERY_SHA256,
    PARENT_SCENE_CONFIG_SHA256,
    PARENT_SCENE_EXPERIMENT,
    PARENT_SCENE_GIT_COMMIT,
    PREDICTION_MANIFEST_SHA256,
    _validate_parent_provenance,
    exact_join_candidate_group,
    load_visualization_plan,
    read_candidate_groups,
)


ROOT = Path(__file__).resolve().parents[1]


def test_negative_spatial_visualization_plan_freezes_candidate_and_eight_figures():
    plan = load_visualization_plan(
        ROOT / "config/Other_NegativeDistanceSpatialVisualization_1.1.yaml"
    )
    assert plan.config["experiment"] == EXPERIMENT
    assert tuple(value.dataset for value in plan.datasets) == DATASETS
    assert tuple(value.block_id for value in plan.blocks) == BLOCKS
    assert tuple(plan.config["figure_contract"]["panel_titles"]) == PANEL_TITLES
    assert plan.assigned_count == 64_000
    assert plan.png_dpi == 360
    assert plan.metric_tolerance == 1e-12


def _write_predictions(path: Path, *, duplicate: bool) -> None:
    fields = (
        "input_id",
        "dataset",
        "source_ordinal",
        "block",
        "center_index",
        SCORE_COLUMN,
        PREDICTION_COLUMN,
    )
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "input_id": "other_input_is_filtered",
                "dataset": "ignored",
                "source_ordinal": 0,
                "block": "ignored",
                "center_index": 0,
                SCORE_COLUMN: 0.1,
                PREDICTION_COLUMN: 0,
            }
        )
        for dataset in DATASETS:
            for block in BLOCKS:
                for center in (7, 3):
                    writer.writerow(
                        {
                            "input_id": "main31_train_family_holdouts_source2",
                            "dataset": dataset,
                            "source_ordinal": 2,
                            "block": block,
                            "center_index": center,
                            SCORE_COLUMN: 0.25 + center / 100.0,
                            PREDICTION_COLUMN: int(center == 7),
                        }
                    )
        if duplicate:
            writer.writerow(
                {
                    "input_id": "main31_train_family_holdouts_source2",
                    "dataset": DATASETS[0],
                    "source_ordinal": 2,
                    "block": BLOCKS[0],
                    "center_index": 7,
                    SCORE_COLUMN: 0.9,
                    PREDICTION_COLUMN: 1,
                }
            )


def test_candidate_projection_uses_fixed_columns_and_rejects_duplicate_join_identity():
    plan = load_visualization_plan(
        ROOT / "config/Other_NegativeDistanceSpatialVisualization_1.1.yaml"
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions = root / "predictions.csv"
        _write_predictions(predictions, duplicate=False)
        test_plan = replace(plan, predictions_csv=predictions)
        groups = read_candidate_groups(test_plan)
        assert set(groups) == {
            (dataset, block) for dataset in DATASETS for block in BLOCKS
        }
        assert all(set(group) == {3, 7} for group in groups.values())
        assert all(group[7][0] and not group[3][0] for group in groups.values())

        duplicate = root / "duplicate.csv"
        _write_predictions(duplicate, duplicate=True)
        duplicate_plan = replace(plan, predictions_csv=duplicate)
        try:
            read_candidate_groups(duplicate_plan)
        except ValueError as error:
            assert "duplicate candidate row identity" in str(error)
        else:
            raise AssertionError("duplicate candidate join identity was accepted")


def test_candidate_projection_rejects_extra_identity_and_reordered_parent_join():
    plan = load_visualization_plan(
        ROOT / "config/Other_NegativeDistanceSpatialVisualization_1.1.yaml"
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions = root / "predictions.csv"
        _write_predictions(predictions, duplicate=False)
        rows = predictions.read_text(encoding="utf-8").splitlines()
        fields = rows[0].split(",")
        source_index = fields.index("source_ordinal")
        values = rows[1].split(",")
        values[0] = "main31_train_family_holdouts_source2"
        values[source_index] = "3"
        rows[1] = ",".join(values)
        predictions.write_text("\n".join(rows) + "\n", encoding="utf-8")
        try:
            read_candidate_groups(replace(plan, predictions_csv=predictions))
        except ValueError as error:
            assert "unexpected source ordinal" in str(error)
        else:
            raise AssertionError("extra candidate source identity was accepted")

    candidates = {7: (True, 0.9), 3: (False, 0.1)}
    prediction, score = exact_join_candidate_group(
        np.asarray([7, 3], dtype=np.int64),
        candidates,
        identity=(DATASETS[0], BLOCKS[0]),
    )
    assert prediction.tolist() == [True, False]
    assert score.tolist() == [0.9, 0.1]
    try:
        exact_join_candidate_group(
            np.asarray([3, 7], dtype=np.int64),
            candidates,
            identity=(DATASETS[0], BLOCKS[0]),
        )
    except ValueError as error:
        assert "row order differs" in str(error)
    else:
        raise AssertionError("reordered candidate identities were accepted")


def test_parent_provenance_closes_scene_candidate_and_prediction_source_chain():
    plan = load_visualization_plan(
        ROOT / "config/Other_NegativeDistanceSpatialVisualization_1.1.yaml"
    )
    parent_result = {
        "experiment": PARENT_SCENE_EXPERIMENT,
        "status": "family_held_out_exposed_development_completed",
        "git_commit": PARENT_SCENE_GIT_COMMIT,
        "config_sha256": PARENT_SCENE_CONFIG_SHA256,
        "visualization_manifest_file_sha256": plan.parent_visualization_sha256,
        "artifacts": [
            {
                "relative_path": "visualization_manifest.json",
                "sha256": plan.parent_visualization_sha256,
            },
            {
                "relative_path": "per_query_matches.csv",
                "sha256": PARENT_PER_QUERY_SHA256,
            },
        ],
    }
    parent_visualization = {
        "experiment": PARENT_SCENE_EXPERIMENT,
        "git_commit": PARENT_SCENE_GIT_COMMIT,
        "config_sha256": PARENT_SCENE_CONFIG_SHA256,
    }
    candidate_result = {
        "experiment": PARENT_CANDIDATE_EXPERIMENT,
        "status": "complete",
        "git_commit": PARENT_CANDIDATE_GIT_COMMIT,
        "config_sha256": PARENT_CANDIDATE_CONFIG_SHA256,
        "prediction_manifest_file_sha256": PREDICTION_MANIFEST_SHA256,
        "artifacts": [
            {"path": "prediction_manifest.json", "sha256": PREDICTION_MANIFEST_SHA256},
            {"path": "predictions.csv", "sha256": plan.predictions_sha256},
            {"path": "per_group_metrics.csv", "sha256": plan.per_group_metrics_sha256},
        ],
    }
    prediction_manifest = {
        "experiment": PARENT_CANDIDATE_EXPERIMENT,
        "config_sha256": PARENT_CANDIDATE_CONFIG_SHA256,
        "phase": "prediction_complete_before_explicit_reference_projection",
        "predictions_file": "predictions.csv",
        "predictions_file_sha256": plan.predictions_sha256,
        "reference_column_projection_to_prediction_logic": "excluded",
        "input_files": [
            {
                "input_id": "main31_train_family_holdouts_source2",
                "path": PARENT_PER_QUERY_PATH,
                "file_sha256": PARENT_PER_QUERY_SHA256,
                "row_count": 406177,
                "score_column": "nearest_negative_distance",
                "observed_datasets": [
                    "boeing747",
                    "cylinder3d",
                    "halfcylinderRe640",
                    "halfcylinderRe6400",
                ],
            }
        ],
    }
    _validate_parent_provenance(
        parent_result,
        parent_visualization,
        candidate_result,
        prediction_manifest,
        plan,
    )

    changed = copy.deepcopy(prediction_manifest)
    changed["input_files"][0]["file_sha256"] = "0" * 64
    try:
        _validate_parent_provenance(
            parent_result,
            parent_visualization,
            candidate_result,
            changed,
            plan,
        )
    except ValueError as error:
        assert "source SHA-256 changed" in str(error)
    else:
        raise AssertionError("changed parent prediction-source SHA-256 was accepted")
