from __future__ import annotations

from pathlib import Path

import numpy as np

import pathline_template_matching.family_heldout_visualization as heldout
from pathline_template_matching.family_heldout_visualization import (
    CONFIG_SHA256,
    EXPERIMENT,
    REQUESTED_DATASETS,
    load_other_visualization_plan,
)
from pathline_template_matching.phase21_pipeline import ScaleAssignmentBlock
from pathline_template_matching.portable_flow import (
    canonical_array_sha256,
    canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def test_family_heldout_visualization_plan_freezes_complete_families_and_eight_figures():
    plan = load_other_visualization_plan(
        ROOT / "config/Other_MainExp31FamilyHeldOutVisualization_1.1.yaml"
    )
    assert plan.config["experiment"] == EXPERIMENT
    assert plan.config_sha256 == CONFIG_SHA256
    assert plan.config["visualization"]["expected_figure_count"] == 8
    assert plan.parent_plan.experiment == "mainExp_TemplateMatching_3.1"
    assert tuple(
        dataset for fold in plan.folds for dataset in fold.query_datasets
    ) == REQUESTED_DATASETS
    assert [fold.held_out_family for fold in plan.folds] == [
        "half_cylinder",
        "boeing_747",
    ]
    for fold in plan.folds:
        assert not set(fold.query_datasets).intersection(fold.library_datasets)
        assert all(
            plan.parent_plan.family_by_dataset[dataset] == fold.held_out_family
            for dataset in fold.query_datasets
        )
        assert all(
            plan.parent_plan.family_by_dataset[dataset] != fold.held_out_family
            for dataset in fold.library_datasets
        )


def test_family_heldout_block_cache_rehashes_every_filtered_array_and_combined_identity():
    count = 4
    cache = {
        "raw_features": np.arange(count * 672, dtype=np.float32).reshape(count, 672),
        "fmt_features": np.arange(count * 161, dtype=np.float32).reshape(count, 161),
        "valid_labels": np.asarray([False, True, False, True]),
        "valid_seed_index": np.asarray([0, 1, 4, 5], dtype=np.int64),
        "valid_scale_id": np.asarray([0, 1, 1000, 1001], dtype=np.int32),
        "center_sample_time": np.zeros((count, 32), dtype=np.float32),
        "valid_assigned_row_index": np.asarray([0, 1, 4, 5], dtype=np.int64),
        "valid_center_seed_index": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "valid_scale_block_index": np.asarray([0, 0, 1, 1], dtype=np.int8),
        "seeds_xyz": np.arange(18, dtype=np.float64).reshape(6, 3),
        "ivd_volume": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
    }
    hashes = {name: canonical_array_sha256(values) for name, values in cache.items()}
    cache["metadata"] = {
        "array_sha256": hashes,
        "combined_array_sha256": canonical_json_sha256(hashes),
    }
    selected, mask = heldout._block_scene_cache(
        cache,
        0,
        ScaleAssignmentBlock("legacy_2_1", 0, 1000, 15068),
    )
    np.testing.assert_array_equal(mask, np.asarray([True, True, False, False]))
    stored = selected["metadata"]["array_sha256"]
    for name in (
        "raw_features",
        "fmt_features",
        "valid_labels",
        "valid_seed_index",
        "valid_scale_id",
        "center_sample_time",
        "valid_assigned_row_index",
        "valid_center_seed_index",
        "valid_scale_block_index",
    ):
        assert stored[name] == canonical_array_sha256(selected[name])
    assert selected["metadata"]["combined_array_sha256"] == canonical_json_sha256(
        stored
    )
