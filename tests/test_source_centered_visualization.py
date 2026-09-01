from __future__ import annotations

import inspect
import json
import csv
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys

for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.source_centered_visualization import (  # noqa: E402
    CENTER_COUNT,
    SCENE_ARRAY_NAMES,
    bind_valid_projection,
    combine_parent_block_scenes,
    render_source_centered_triptych,
    scene_arrays,
)
from scripts import (  # noqa: E402
    aggregate_verify_source_centered_paired_scale_template_1_1 as aggregate,
)
from scripts import audit_source_centered_paired_scale_template_visualizations as qa  # noqa: E402
from scripts import render_source_centered_paired_scale_template_visualizations as report  # noqa: E402
from scripts import run_verify_source_centered_paired_scale_template_1_1 as runner  # noqa: E402


def _raises(function, *args, contains: str, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except (ValueError, FileExistsError) as error:
        assert contains in str(error), str(error)
    else:
        raise AssertionError(f"expected failure containing {contains!r}")


def _write_self_hashed(path: Path, payload: dict) -> str:
    value = dict(payload)
    value["content_sha256"] = canonical_json_sha256(value)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return sha256_file(path)


def _parent_scene(
    block: str,
    centers: np.ndarray,
    seeds: np.ndarray,
    reference: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    index = 0 if block == "legacy_2_1" else 1
    pathlines = np.zeros((120, 2, 4), dtype=np.float64)
    pathlines[:, 0, :3] = np.arange(120, dtype=np.float64)[:, None] * 0.001
    pathlines[:, 1, :3] = pathlines[:, 0, :3] + 0.01 + index
    pathlines[:, 1, 3] = 1.0
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    arrays = {
        "bounds": np.asarray([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float64),
        "seeds": np.asarray(seeds, dtype=np.float64),
        "reference": np.asarray(reference, dtype=np.bool_),
        "valid_scale_id": np.arange(len(centers), dtype=np.int32) + index * 1000,
        "valid_assigned_row_index": np.asarray(centers, dtype=np.int64) + index * CENTER_COUNT,
        "valid_center_seed_index": np.asarray(centers, dtype=np.int64),
        "valid_scale_block_index": np.full(len(centers), index, dtype=np.int8),
        "display_pathlines": pathlines,
        "ivd_mesh_vertices": vertices,
        "ivd_mesh_faces": np.asarray([[0, 1, 2]], dtype=np.int32),
        "ivd_mesh_normals": np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        "ivd_mesh_values": np.ones(3, dtype=np.float32),
        "ivd_mesh_level": np.asarray(0.5, dtype=np.float64),
    }
    metadata = {
        "dataset": "cylinder3d",
        "source_ordinal": 2,
        "source_index": 7,
        "scale_block_id": block,
    }
    return metadata, arrays


def _unique_prediction() -> dict[str, np.ndarray]:
    legacy = np.zeros(CENTER_COUNT, dtype=np.bool_)
    expanded = np.zeros(CENTER_COUNT, dtype=np.bool_)
    legacy[[0, 1]] = True
    expanded[[1, 2]] = True
    prediction = np.zeros(CENTER_COUNT, dtype=np.bool_)
    prediction[[1, 2]] = True
    return {
        "unique_dataset": np.full(CENTER_COUNT, "cylinder3d", dtype="<U64"),
        "unique_source_ordinal": np.full(CENTER_COUNT, 2, dtype=np.int16),
        "unique_source_index": np.full(CENTER_COUNT, 7, dtype=np.int64),
        "unique_center_seed_index": np.arange(CENTER_COUNT, dtype=np.int64),
        "paired_score": np.linspace(0.0, 1.0, CENTER_COUNT, dtype=np.float64),
        "legacy_valid": legacy,
        "expanded_valid": expanded,
        "paired_prediction": prediction,
    }


def _valid_prediction() -> dict[str, np.ndarray]:
    # Deliberately not in parent scene order. The binder must restore the
    # frozen legacy-parent then expanded-parent order by exact identity.
    block = np.asarray([1, 0, 1, 0], dtype=np.int8)
    center = np.asarray([2, 1, 1, 0], dtype=np.int64)
    scale = np.asarray([1001, 1, 1000, 0], dtype=np.int32)
    assigned = block.astype(np.int64) * CENTER_COUNT + center
    return {
        "valid_dataset": np.full(4, "cylinder3d", dtype="<U64"),
        "valid_source_ordinal": np.full(4, 2, dtype=np.int16),
        "valid_source_index": np.full(4, 7, dtype=np.int64),
        "valid_scale_id": scale,
        "valid_center_seed_index": center,
        "valid_scale_block_index": block,
        "valid_assigned_row_index": assigned,
        "valid_paired_score": np.linspace(
            0.0, 1.0, CENTER_COUNT, dtype=np.float64
        )[center],
        "valid_paired_prediction": np.asarray([True, True, True, False], dtype=np.bool_),
    }


def _synthetic_scene_inputs():
    legacy = _parent_scene(
        "legacy_2_1",
        np.asarray([0, 1]),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        np.asarray([False, True]),
    )
    expanded = _parent_scene(
        "expanded_3_1",
        np.asarray([1, 2]),
        np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        np.asarray([True, False]),
    )
    return legacy, expanded


def test_frozen_config_defines_four_single_paired_center_figures() -> None:
    value = report._authenticate_config()
    assert sha256_file(report.CONFIG_PATH) == report.CONFIG_SHA256
    assert value["figure_contract"]["expected_figure_count"] == 4
    assert value["method_parent"]["plotted_prediction_array"] == "paired_prediction"
    assert value["method_parent"]["separate_block_predictions_are_not_plotted"] is True
    assert value["figure_contract"]["background_pathlines"]["legacy_parent_scene_prefix_count"] == 120
    assert value["figure_contract"]["background_pathlines"]["expanded_parent_scene_prefix_count"] == 120
    assert tuple(value["scene_schema"]["ordered_arrays"]) == SCENE_ARRAY_NAMES


def test_combined_scene_uses_union_once_and_fixed_pathline_prefixes() -> None:
    legacy, expanded = _synthetic_scene_inputs()
    scene = combine_parent_block_scenes(
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded[1],
        unique_prediction=_unique_prediction(),
        title="Cylinder3D Re160",
    )
    assert scene.center_seed_index.tolist() == [0, 1, 2]
    assert scene.reference.tolist() == [False, True, False]
    assert scene.prediction.tolist() == [False, True, True]
    assert scene.display_pathlines.shape == (240, 2, 4)
    assert scene.display_pathline_block_index.tolist() == [0] * 120 + [1] * 120
    assert scene.legacy_valid.tolist() == [True, True, False]
    assert scene.expanded_valid.tolist() == [False, True, True]


def test_combined_scene_fails_on_reorder_mask_and_overlap_drift() -> None:
    legacy, expanded = _synthetic_scene_inputs()
    reordered = _unique_prediction()
    reordered["unique_center_seed_index"] = reordered["unique_center_seed_index"].copy()
    reordered["unique_center_seed_index"][[0, 1]] = reordered["unique_center_seed_index"][[1, 0]]
    _raises(
        combine_parent_block_scenes,
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded[1],
        unique_prediction=reordered,
        title="x",
        contains="ascending 0..63999",
    )
    mask_drift = _unique_prediction()
    mask_drift["legacy_valid"] = mask_drift["legacy_valid"].copy()
    mask_drift["legacy_valid"][3] = True
    _raises(
        combine_parent_block_scenes,
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded[1],
        unique_prediction=mask_drift,
        title="x",
        contains="legacy_valid mask",
    )
    expanded_drift = {key: np.array(value, copy=True) for key, value in expanded[1].items()}
    expanded_drift["seeds"][0, 0] += 0.01
    _raises(
        combine_parent_block_scenes,
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded_drift,
        unique_prediction=_unique_prediction(),
        title="x",
        contains="different coordinates",
    )


def test_valid_projection_exact_join_restores_frozen_parent_order() -> None:
    legacy, expanded = _synthetic_scene_inputs()
    projection = bind_valid_projection(
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded[1],
        unique_prediction=_unique_prediction(),
        valid_prediction=_valid_prediction(),
    )
    assert projection.scale_block_index.tolist() == [0, 0, 1, 1]
    assert projection.center_seed_index.tolist() == [0, 1, 1, 2]
    assert projection.assigned_row_index.tolist() == [0, 1, 64001, 64002]
    assert projection.reference.tolist() == [False, True, True, False]
    assert projection.prediction.tolist() == [False, True, True, True]
    assert np.array_equal(
        projection.score,
        np.linspace(0.0, 1.0, CENTER_COUNT, dtype=np.float64)[[0, 1, 1, 2]],
    )


def test_valid_projection_rejects_rowwise_unique_prediction_or_score_drift() -> None:
    legacy, expanded = _synthetic_scene_inputs()
    prediction_drift = _valid_prediction()
    prediction_drift["valid_paired_prediction"] = prediction_drift[
        "valid_paired_prediction"
    ].copy()
    prediction_drift["valid_paired_prediction"][0] = False
    _raises(
        bind_valid_projection,
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded[1],
        unique_prediction=_unique_prediction(),
        valid_prediction=prediction_drift,
        contains="valid_paired_prediction is not the exact unique-center projection",
    )
    score_drift = _valid_prediction()
    score_drift["valid_paired_score"] = score_drift["valid_paired_score"].copy()
    score_drift["valid_paired_score"][0] = np.nextafter(
        score_drift["valid_paired_score"][0], np.inf
    )
    _raises(
        bind_valid_projection,
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded[1],
        unique_prediction=_unique_prediction(),
        valid_prediction=score_drift,
        contains="valid_paired_score is not the exact unique-center projection",
    )


def test_valid_projection_rejects_duplicate_missing_and_extra_identity() -> None:
    legacy, expanded = _synthetic_scene_inputs()
    duplicate = _valid_prediction()
    for field in (
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
    ):
        duplicate[field] = duplicate[field].copy()
        duplicate[field][1] = duplicate[field][0]
    _raises(
        bind_valid_projection,
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded[1],
        unique_prediction=_unique_prediction(),
        valid_prediction=duplicate,
        contains="duplicate valid prediction identity",
    )
    extra = _valid_prediction()
    for field, value in list(extra.items()):
        if field == "valid_dataset":
            tail = np.asarray(["cylinder3d"], dtype="<U64")
        elif field == "valid_paired_score":
            tail = np.asarray([0.0], dtype=np.float64)
        elif field == "valid_paired_prediction":
            tail = np.asarray([False], dtype=np.bool_)
        else:
            tail = np.asarray([value[-1]], dtype=value.dtype)
        extra[field] = np.concatenate((value, tail))
    extra["valid_center_seed_index"][-1] = 9
    extra["valid_assigned_row_index"][-1] = CENTER_COUNT + 9
    extra["valid_scale_id"][-1] = 1002
    extra["valid_paired_score"][-1] = np.linspace(
        0.0, 1.0, CENTER_COUNT, dtype=np.float64
    )[9]
    extra["valid_paired_prediction"][-1] = False
    _raises(
        bind_valid_projection,
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded[1],
        unique_prediction=_unique_prediction(),
        valid_prediction=extra,
        contains="extra valid rows",
    )


def test_scene_payload_is_ordered_pickle_free_and_records_both_blocks() -> None:
    legacy, expanded = _synthetic_scene_inputs()
    scene = combine_parent_block_scenes(
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded[1],
        unique_prediction=_unique_prediction(),
        title="Cylinder3D Re160",
    )
    arrays = scene_arrays(scene, '{"dataset":"cylinder3d"}')
    assert tuple(arrays) == SCENE_ARRAY_NAMES
    assert arrays["metadata_json"].dtype.kind == "U"
    assert arrays["display_pathline_block_index"].tolist() == [0] * 120 + [1] * 120
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scene.npz"
        np.savez(path, **arrays)
        with np.load(path, allow_pickle=False) as archive:
            assert tuple(archive.files) == SCENE_ARRAY_NAMES
            assert str(archive["metadata_json"].reshape(())) == '{"dataset":"cylinder3d"}'
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        metadata = {
            "dataset": "cylinder3d",
            "display_title": "Cylinder3D Re160",
            "source_index": scene.source_index,
        }
        payload = scene_arrays(scene, json.dumps(metadata, sort_keys=True))
        scene_path = root / "combined.scene.npz"
        manifest_path = root / "combined.scene.json"
        report._atomic_npz(scene_path, payload)
        manifest = report._scene_manifest(
            scene_path=scene_path,
            arrays=payload,
            metadata=metadata,
            dataset="cylinder3d",
            reporting_identity={"reporting_git_commit": "a" * 40},
        )
        report._atomic_json(manifest_path, manifest)
        loaded = report._load_combined_scene(scene_path, manifest_path)
        assert np.array_equal(loaded.center_seed_index, scene.center_seed_index)
        assert np.array_equal(loaded.display_pathline_block_index, [0] * 120 + [1] * 120)


def test_synthetic_triptych_writes_all_exports_one_classifier_and_fixed_pathlines(
    tmp_path: Path,
) -> None:
    legacy, expanded = _synthetic_scene_inputs()
    scene = combine_parent_block_scenes(
        legacy_metadata=legacy[0],
        legacy_arrays=legacy[1],
        expanded_metadata=expanded[0],
        expanded_arrays=expanded[1],
        unique_prediction=_unique_prediction(),
        title="Cylinder3D Re160",
    )
    png = tmp_path / "figure.png"
    pdf = tmp_path / "figure.pdf"
    svg = tmp_path / "figure.svg"
    alignment = tmp_path / "figure.alignment.json"
    metadata = render_source_centered_triptych(
        scene,
        png_path=png,
        pdf_path=pdf,
        svg_path=svg,
        alignment_path=alignment,
    )
    assert qa._png_header(png)["width_pixels"] == 7560
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert "<svg" in svg.read_text(encoding="utf-8")
    audit = json.loads(alignment.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert metadata["counts"]["combined_valid_center_count"] == len(scene.seeds)
    assert metadata["prediction_semantics"].startswith("one authenticated")
    assert metadata["pathline_audit"]["block_counts"] == {
        "legacy_2_1": 120,
        "expanded_3_1": 120,
    }
    assert metadata["camera"]["identical_across_panels"] is True
    assert metadata["ivd_audit"]["identical_across_panels"] is True
    assert len(metadata["ivd_audit"]["panels"]) == 3
    assert all(
        row == metadata["ivd_audit"]["panels"][0]
        for row in metadata["ivd_audit"]["panels"][1:]
    )
    svg_audit = qa._audit_svg_editable_text(svg)
    assert svg_audit["result"] == "PASS"
    assert svg_audit["text_element_count"] > 0
    collision_tool = (
        Path.home()
        / ".codex"
        / "skills"
        / "nature-figure"
        / "scripts"
        / "audit_figure_collisions.py"
    )
    if collision_tool.is_file():
        completed = subprocess.run(
            [sys.executable, str(collision_tool), str(pdf), "--json"],
            check=False,
            capture_output=True,
            text=True,
        )
        collision = json.loads(completed.stdout)
        assert completed.returncode == 0, completed.stderr
        assert collision["auditable"] is True
        assert collision["summary"]["fail"] == 0


def _candidate() -> dict:
    return {
        "candidate_id": (
            "representation=fmt161_plus_source_centered_seed4|k=1|sigma=0|"
            "weight=0.5|fixed_top_fraction=0.05"
        ),
        "representation": "fmt161_plus_source_centered_seed4",
        "k": 1,
        "sigma": 0.0,
        "weight": 0.5,
        "decision_rule": "fixed_top_fraction",
        "decision_value": 0.05,
    }


def _synthetic_release_evidence(commit: str) -> dict:
    row_identities = []
    for index in range(32):
        row_identities.append(
            {
                "dataset": f"dataset_{index // 4}",
                "dataset_index": index // 4,
                "physical_family": f"family_{index // 4}",
                "source_ordinal": index % 4,
                "source_index": index,
                "completion_file_sha256": "1" * 64,
                "sidecar_file_sha256": "2" * 64,
                "sidecar_combined_array_sha256": "3" * 64,
                "valid_projection_sha256": "4" * 64,
                "assigned_row_count": 128_000,
                "valid_projection_row_count": 100,
            }
        )
    return {
        "experiment": runner.EXPERIMENT,
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "git_commit": commit,
        "source_centered_input_manifest": {
            "path": "/ibex/synthetic/input_manifest.json",
            "size_bytes": 100,
            "file_sha256": "5" * 64,
            "content_sha256": "6" * 64,
        },
        "source_centered_sidecars": {
            "root": "/ibex/synthetic/sidecars",
            "population_manifest_path": "/ibex/synthetic/sidecars/SIDECAR_POPULATION.json",
            "population_manifest_size_bytes": 100,
            "population_manifest_file_sha256": "7" * 64,
            "population_manifest_content_sha256": "8" * 64,
            "sidecar_count": 32,
            "rows_content_sha256": "9" * 64,
            "assigned_row_count_total": 4_096_000,
            "valid_projection_row_count_total": 3_200,
            "row_identities": row_identities,
        },
    }


def _write_synthetic_fold(root: Path, family: str, commit: str) -> tuple[str, str]:
    root.mkdir()
    candidate = _candidate()
    for name in runner.REQUIRED_FOLD_FILES:
        if name in {"result_manifest.json", "RUN_COMPLETE.json"}:
            continue
        path = root / name
        if name == "outer_predictions.npz":
            path.write_bytes(b"opaque-prediction")
        elif name == "selected_candidate.json":
            _write_self_hashed(
                path,
                {
                    "schema": runner.SELECTED_SCHEMA,
                    "experiment": runner.EXPERIMENT,
                    "outer_family": family,
                    "git_commit": commit,
                    "config_sha256": runner.EXPECTED_CONFIG_SHA256,
                    "candidate_count": runner.FROZEN_CANDIDATE_COUNT,
                    "candidate": candidate,
                },
            )
        elif name == "outer_prediction_manifest.json":
            prediction_path = root / "outer_predictions.npz"
            if not prediction_path.exists():
                prediction_path.write_bytes(b"opaque-prediction")
            _write_self_hashed(
                path,
                {
                    "schema": runner.PREDICTION_MANIFEST_SCHEMA,
                    "prediction_schema": runner.PREDICTION_SCHEMA,
                    "experiment": runner.EXPERIMENT,
                    "outer_family": family,
                    "git_commit": commit,
                    "config_sha256": runner.EXPECTED_CONFIG_SHA256,
                    "valid_labels_opened": False,
                    "reference_labels_all_opened": False,
                    "selected_candidate": candidate,
                    "prediction_file": {
                        "path": "outer_predictions.npz",
                        "size_bytes": prediction_path.stat().st_size,
                        "sha256": sha256_file(prediction_path),
                    },
                    "arrays": {name: {} for name in runner.PREDICTION_DTYPES},
                },
            )
        elif not path.exists():
            path.write_bytes(name.encode("utf-8"))
    artifacts = {
        name: {
            "size_bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        for name in runner.REQUIRED_FOLD_FILES
        if name not in {"result_manifest.json", "RUN_COMPLETE.json"}
    }
    result_path = root / "result_manifest.json"
    result_sha = _write_self_hashed(
        result_path,
        {
            "schema": runner.RESULT_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "outer_family": family,
            "git_commit": commit,
            "config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "selected_candidate": candidate,
            "artifacts": artifacts,
        },
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    completion_path = root / "RUN_COMPLETE.json"
    completion_sha = _write_self_hashed(
        completion_path,
        {
            "schema": runner.COMPLETE_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "outer_family": family,
            "git_commit": commit,
            "config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "result_manifest_file_sha256": result_sha,
            "result_manifest_content_sha256": result["content_sha256"],
        },
    )
    return completion_sha, result_sha


def _write_synthetic_release(
    root: Path,
    *,
    fold_root: Path,
    family: str,
    commit: str,
    completion_sha: str,
    result_sha: str,
) -> None:
    root.mkdir()
    evidence = _synthetic_release_evidence(commit)
    table = root / "outer_family_summary.csv"
    table.write_text("outer_family\n" + family + "\n", encoding="utf-8")
    report_path = root / "single_fold_authentication_report.json"
    report_sha = _write_self_hashed(
        report_path,
        {
            "schema": aggregate.SINGLE_FOLD_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": "single_fold_authentication",
            "config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "aggregator_git_commit": commit,
            "fold_git_commit": commit,
            "source_centered_evidence": evidence,
            "outer_families": [family],
        },
    )
    manifest_path = root / "aggregate_manifest.json"
    manifest_sha = _write_self_hashed(
        manifest_path,
        {
            "schema": aggregate.AGGREGATE_MANIFEST_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": "single_fold_authentication",
            "config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "aggregator_git_commit": commit,
            "fold_git_commit": commit,
            "source_centered_evidence": evidence,
            "outer_family_summary_file": table.name,
            "outer_family_summary_file_sha256": sha256_file(table),
            "report_file": report_path.name,
            "report_file_sha256": report_sha,
            "source_folds": [
                {
                    "outer_family": family,
                    "run_directory": str(fold_root),
                    "completion_file_sha256": completion_sha,
                    "result_manifest_file_sha256": result_sha,
                }
            ],
        },
    )
    _write_self_hashed(
        root / "AGGREGATE_COMPLETE.json",
        {
            "schema": aggregate.AGGREGATE_COMPLETE_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "completed",
            "mode": "single_fold_authentication",
            "config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "aggregator_git_commit": commit,
            "fold_git_commit": commit,
            "source_centered_evidence": evidence,
            "aggregate_manifest_file": manifest_path.name,
            "aggregate_manifest_file_sha256": manifest_sha,
            "report_file": report_path.name,
            "report_file_sha256": report_sha,
        },
    )


def test_release_and_fold_chain_are_opaque_authenticated_before_npz_open(tmp_path: Path) -> None:
    commit = report.TRUSTED_NUMERICAL_AGGREGATOR_GIT_COMMIT
    fold_root = tmp_path / "slurm_1_0_commit_outer_half_cylinder"
    completion_sha, result_sha = _write_synthetic_fold(
        fold_root, "half_cylinder", commit
    )
    release_root = tmp_path / "single_release"
    _write_synthetic_release(
        release_root,
        fold_root=fold_root,
        family="half_cylinder",
        commit=commit,
        completion_sha=completion_sha,
        result_sha=result_sha,
    )
    release = report.authenticate_release_root(release_root)
    fold = report.authenticate_fold_root(
        fold_root,
        expected_family="half_cylinder",
        release_record=release.source_folds["half_cylinder"],
        expected_fold_commit=commit,
    )
    assert release.mode == "single_fold_authentication"
    assert fold.candidate == _candidate()
    # Authentication only uses whole-file identity for the opaque prediction;
    # the synthetic file is deliberately not a valid NPZ archive.
    assert (fold_root / "outer_predictions.npz").read_bytes() == b"opaque-prediction"
    (fold_root / "outer_predictions.npz").write_bytes(b"tampered")
    _raises(
        report.authenticate_fold_root,
        fold_root,
        expected_family="half_cylinder",
        release_record=release.source_folds["half_cylinder"],
        expected_fold_commit=commit,
        contains="fold artifact identity changed: outer_predictions.npz",
    )


def test_release_rejects_untrusted_aggregator_and_cross_artifact_evidence(
    tmp_path: Path,
) -> None:
    fold_root = tmp_path / "fold"
    fold_root.mkdir()
    untrusted = tmp_path / "untrusted"
    _write_synthetic_release(
        untrusted,
        fold_root=fold_root,
        family="half_cylinder",
        commit="b" * 40,
        completion_sha="1" * 64,
        result_sha="2" * 64,
    )
    _raises(
        report.authenticate_release_root,
        untrusted,
        contains="aggregator commit is not the trusted numerical revision",
    )

    trusted = tmp_path / "trusted"
    commit = report.TRUSTED_NUMERICAL_AGGREGATOR_GIT_COMMIT
    _write_synthetic_release(
        trusted,
        fold_root=fold_root,
        family="half_cylinder",
        commit=commit,
        completion_sha="1" * 64,
        result_sha="2" * 64,
    )
    report_path = trusted / "single_fold_authentication_report.json"
    report_value = json.loads(report_path.read_text(encoding="utf-8"))
    report_value.pop("content_sha256")
    report_value["source_centered_evidence"] = dict(
        report_value["source_centered_evidence"]
    )
    report_value["source_centered_evidence"]["git_commit"] = "c" * 40
    report_sha = _write_self_hashed(report_path, report_value)
    manifest_path = trusted / "aggregate_manifest.json"
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_value.pop("content_sha256")
    manifest_value["report_file_sha256"] = report_sha
    manifest_sha = _write_self_hashed(manifest_path, manifest_value)
    complete_path = trusted / "AGGREGATE_COMPLETE.json"
    complete_value = json.loads(complete_path.read_text(encoding="utf-8"))
    complete_value.pop("content_sha256")
    complete_value["report_file_sha256"] = report_sha
    complete_value["aggregate_manifest_file_sha256"] = manifest_sha
    _write_self_hashed(complete_path, complete_value)
    _raises(
        report.authenticate_release_root,
        trusted,
        contains="evidence differs across completion/manifest/report",
    )


def test_report_source_orders_input_manifest_before_any_npz_member_access() -> None:
    source = inspect.getsource(report.render_bundle)
    manifest_write = source.index('_atomic_json(output_root / "input_manifest.json"')
    prediction_open = source.index("groups = load_prediction_groups")
    scene_open = source.index("_load_parent_scene(paths)")
    assert manifest_write < prediction_open < scene_open
    assert "scale_blocks_are_context_not_separate_classifiers" in source


def test_producer_metric_rows_bind_complete_group_and_candidate_identity(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    groups: dict[str, report.PredictionGroup] = {}
    folds: dict[str, report.AuthenticatedFold] = {}
    datasets_by_family = {
        family: [
            dataset
            for dataset in report.DATASETS
            if report.DATASET_TO_FAMILY[dataset] == family
        ]
        for family in report.REQUIRED_FAMILIES
    }
    for family_index, (family, datasets) in enumerate(datasets_by_family.items()):
        root = tmp_path / family
        root.mkdir()
        rows = []
        for dataset_index, dataset in enumerate(datasets):
            source_index = 100 * family_index + dataset_index
            groups[dataset] = report.PredictionGroup(
                dataset=dataset,
                outer_family=family,
                candidate=candidate,
                unique={"unique_source_index": np.asarray([source_index], dtype=np.int64)},
                valid={},
            )
            for population in (
                "all_parent_valid_rows",
                "combined_valid_unique_centers",
            ):
                row = {field: 0 for field in runner.OUTER_METRIC_FIELDS}
                row.update(
                    {
                        "outer_family": family,
                        "dataset": dataset,
                        "source_ordinal": 2,
                        "source_index": source_index,
                        "arm": "source_centered_paired_centers",
                        "population": population,
                        "template_success_eligible": population
                        == "all_parent_valid_rows",
                        **candidate,
                    }
                )
                rows.append(row)
        with (root / "outer_group_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as target:
            writer = csv.DictWriter(target, fieldnames=runner.OUTER_METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(
                {
                    name: runner.early._csv_value(row[name])
                    for name in runner.OUTER_METRIC_FIELDS
                }
                for row in rows
            )
        folds[family] = report.AuthenticatedFold(
            root=root,
            outer_family=family,
            git_commit=report.TRUSTED_NUMERICAL_AGGREGATOR_GIT_COMMIT,
            result={},
            prediction_manifest={},
            candidate=candidate,
            evidence=(),
        )
    selected = report.read_producer_metric_rows(folds, groups)
    assert len(selected) == 8

    path = folds["half_cylinder"].root / "outer_group_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    rows[0]["candidate_id"] = "tampered"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=runner.OUTER_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _raises(
        report.read_producer_metric_rows,
        folds,
        groups,
        contains="producer metric identity changed",
    )


def _visual_review_payload(entries: list[dict], *, warning_dataset: str | None = None) -> dict:
    rows = []
    for entry in entries:
        has_warning = entry["dataset"] == warning_dataset
        rows.append(
            {
                "dataset": entry["dataset"],
                "png_sha256": entry["png_sha256"],
                "result": "PASS",
                "checks": {name: True for name in qa.VISUAL_CHECKS},
                "collision_warning_review": (
                    "accepted_after_final_size_review"
                    if has_warning
                    else "not_applicable_no_warnings"
                ),
                "notes": "Reviewed the warning boxes at final size." if has_warning else "",
            }
        )
    return {
        "schema": qa.VISUAL_REVIEW_SCHEMA,
        "experiment": report.EXPERIMENT,
        "review_scope": "every PNG at final 21x5-inch physical size",
        "reviewer": "unit-test reviewer",
        "reviewed_at_utc": "2026-09-01T00:00:00Z",
        "result": "PASS",
        "entries": rows,
    }


def test_visual_review_binds_every_png_and_collision_warning_disposition(
    tmp_path: Path,
) -> None:
    entries = [
        {"dataset": dataset, "png_sha256": str(index) * 64}
        for index, dataset in enumerate(report.DATASETS, 1)
    ]
    warnings = {dataset: int(dataset == "boeing747") for dataset in report.DATASETS}
    payload = _visual_review_payload(entries, warning_dataset="boeing747")
    path = tmp_path / "visual-review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    value, by_dataset = qa._load_visual_review(
        path, entries=entries, collision_warn_counts=warnings
    )
    assert value["result"] == "PASS"
    assert tuple(by_dataset) == report.DATASETS
    payload["entries"][0]["checks"][qa.VISUAL_CHECKS[0]] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    _raises(
        qa._load_visual_review,
        path,
        entries=entries,
        collision_warn_counts=warnings,
        contains="visual-review checks are incomplete: cylinder3d",
    )


def test_delivery_auditor_is_post_render_only_and_keeps_all_qa_gates() -> None:
    source = inspect.getsource(qa.audit_delivery)
    assert "np.load" not in source
    assert source.index("_authenticate_machine_bundle") < source.index("source preflight")
    for name in qa.TOOL_NAMES:
        assert name in inspect.getsource(qa)
    assert "panel_alignment_strict_pass_count" in source
    assert "pdf_text_pass_count" in source
    assert "svg_editable_text_pass_count" in source
    assert "collision_hard_fail_count" in source
    assert "visual_review_pass_count" in source
    assert "_authenticate_local_qa_checkout" in source
    assert "qa_auditor" in source


def test_reporting_dependency_and_source_warning_dispositions_are_exact() -> None:
    assert qa.AUDITOR_RELATIVE_PATH in report.REPORTING_DEPENDENCY_RELATIVE_PATHS
    assert (
        "ibex/other_source_centered_paired_scale_template_visualization_1.1.sh"
        in report.REPORTING_DEPENDENCY_RELATIVE_PATHS
    )
    audit = {
        "findings": [
            {"check_id": check_id, "level": "WARN"}
            for check_id in qa.EXPECTED_SOURCE_WARNINGS
        ]
    }
    dispositions = qa._source_preflight_warning_dispositions(audit)
    assert tuple(row["check_id"] for row in dispositions) == qa.EXPECTED_SOURCE_WARNINGS
    assert all(
        row["disposition"] == "ACCEPTED_FROZEN_REPORT_REQUIREMENT"
        and row["config_sha256"] == report.CONFIG_SHA256
        and row["rationale"]
        for row in dispositions
    )
    audit["findings"].append({"check_id": "NEW-WARNING", "level": "WARN"})
    _raises(
        qa._source_preflight_warning_dispositions,
        audit,
        contains="warning set changed",
    )


def _run_standalone() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    assert len(tests) == 15
    for function in tests:
        if inspect.signature(function).parameters:
            with tempfile.TemporaryDirectory() as directory:
                function(Path(directory))
        else:
            function()


if __name__ == "__main__":
    _run_standalone()
